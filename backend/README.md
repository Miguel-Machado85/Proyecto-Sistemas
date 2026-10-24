# Backend — Agente ciudadano SIGEM (Alcaldía de Marinilla)

API (FastAPI) de un **sistema multi-agente** (LangGraph + Gemini) que responde preguntas sobre las normas del
repositorio SIGEM (≈72 PDFs de decretos, resoluciones, leyes y acuerdos indexados en Chroma), adaptando la
respuesta al perfil del usuario (P0–P3). Acepta texto y voz.

> Especificaciones: la migración al sistema multi-agente está en [`../docs/MIGRACION_AGENTICA.md`](../docs/MIGRACION_AGENTICA.md)
> y la entrega de documentos, Google Drive y Chroma como servicio en [`../docs/ARCHIVOS_AGENTE.md`](../docs/ARCHIVOS_AGENTE.md);
> ambas terminan con una sección "Estado de implementación" que lista las desviaciones.

Además de responder, el agente **entrega el documento original** (botón de descarga) cuando el usuario lo pide, y el índice
vectorial corre como **servicio aparte** (contenedor Chroma), no embebido en el proceso de la API.

## Arquitectura

```
START
  │
  ▼
analista ───────────┬─ saludo / fuera_de_dominio / ambigua / no_comprensible ─► respuesta_directa ─► END
 (flash-lite,       │
  salida            └─ consulta_sigem / seguimiento
  estructurada)                       │
                                      ▼
                          investigador  (subagente ReAct, flash)  ◄───────────┐
                            tools: buscar_documento_por_referencia           │ reintento con
                                   listar_documentos                         │ consulta reformulada
                                   buscar_fragmentos                         │ (máx. MAX_INTENTOS_BUSQUEDA)
                                   leer_documento                            │
                                   preparar_descarga  ← solo si el analista  │
                                                        marcó solicita_archivo
                                      │                                      │
                                      ▼                                      │
                          evaluador (flash-lite) ── evidencia insuficiente ──┘
                                      │ suficiente / hay candidatos / sin resultados
                                      ▼
                          redactor (flash): estilo por perfil, citas en línea ─► END
```

| Agente / nodo | Modelo (variable) | Rol |
|---|---|---|
| `analista` | flash-lite (`ANALISTA_MODEL`), temp. 0 | Entiende la pregunta con salida estructurada (`AnalisisConsulta`): intención, consulta autónoma, filtros (tipo, número, año, código, tema), si se refiere al documento activo y si el usuario **pide el archivo** (`solicita_archivo`) o además su contenido (`incluye_contenido`). Corrige ortografía y convierte números dichos en palabras. **Fail-open**: si falla, sigue como `consulta_sigem`. |
| `respuesta_directa` | — (sin LLM) | Saludo, fuera de dominio, aclaración o alerta de audio no comprensible. |
| `investigador` | flash (`INVESTIGADOR_MODEL`), temp. 0.1 | Subagente ReAct (`create_agent`) que decide qué tools usar y reúne la evidencia. Historial nuevo en cada intento, límite `INVESTIGADOR_MAX_PASOS`. |
| `evaluador` | flash-lite (`EVALUADOR_MODEL`), temp. 0 | Corrective RAG: decide si la evidencia basta o reformula la consulta. Atajos sin LLM; se puede apagar (`EVALUADOR_HABILITADO=false`). Fail-open. |
| `redactor` | flash (`REDACTOR_MODEL`), temp. 0.3 | Redacta solo con la evidencia, cita `[Decreto 001 de 2019, p. 3]`, adapta el estilo al perfil y actualiza el `documento_activo`. Cinco modos: respuesta, aclaración, sin resultados, **entrega** (el archivo ya está listo) y **descarga no disponible**. Nunca escribe URLs: la interfaz muestra el botón. |

**Regla de diseño:** lo que *siempre* debe ocurrir es un nodo con arista fija; lo que *depende de la pregunta* es una
tool que el investigador decide usar. No se interpreta con regex lo que escribe el usuario: eso lo hace el analista
(o el investigador al llenar los argumentos de las tools) y el código **valida contra el catálogo**.

### Estado y memoria
`EstadoAgente` (`agente_sigem/estado.py`) separa lo que **persiste** entre turnos (`messages` con solo la pregunta
y la respuesta final, y `documento_activo`) de lo que se **reinicia** en cada turno (`analisis`, `evidencia`,
`fuentes`, `candidatos`, `archivos`, `archivos_no_disponibles`, `intentos_busqueda`, `consulta_actual`, `pasos`,
`es_alerta`; sin reducer, la API los pasa vacíos). El checkpointer es **PostgreSQL** (`CHECKPOINTER=postgres`) o memoria (`CHECKPOINTER=memory`, para
tests y desarrollo). El grafo se construye en el `lifespan` de FastAPI, no al importar.

### Tools del investigador (`agente_sigem/tools.py`)
Todas devuelven un JSON `{"fuentes", "fragmentos", "candidatos"?, "aclaracion"?, "sugerencias"?, "documentos"?, "error"?}`
y nunca lanzan excepciones.

| Tool | Cuándo |
|---|---|
| `buscar_documento_por_referencia(tipo?, numero?, anio?, pregunta?, codigo?)` | El usuario nombra un documento por tipo y número ("Decreto 83 de 2018") o por código de archivo ("documento 18399"). Devuelve sus fragmentos, `candidatos` si hay varios, o `sugerencias` si no existe. |
| `listar_documentos(tipo?, anio?, tema?, limite=15)` | Preguntas de inventario ("¿qué decretos hay de 2019?"). No trae contenido. |
| `buscar_fragmentos(consulta, archivo?)` | Preguntas sobre un tema. Búsqueda híbrida (texto + semántica) → reranker → diversificación → top 8; con `archivo` busca dentro de ese PDF. |
| `leer_documento(archivo, paginas?)` | Resumir o ampliar un documento ya identificado (hasta 12 fragmentos por página). |
| `preparar_descarga(archivo)` | **Solo** cuando el analista marcó `solicita_archivo`: valida el PDF contra el catálogo y la tabla `archivos_documento` y devuelve `archivos` con la `url_descarga` (la construye el código, nunca el modelo) o un `error` "no disponible para descarga". Está en `HERRAMIENTAS_DESCARGA`, aparte de `HERRAMIENTAS_INVESTIGADOR`. |

### Recuperación y reranker
`retrieval.recuperar_sigem(consulta, archivo=None)`: búsqueda literal en Chroma + búsqueda semántica (embeddings
`EMBEDDING_MODEL`, **no cambiar**: el índice ya está construido con él) → unión sin duplicados → **reranker**
(`sentence-transformers` `CrossEncoder`, `RERANKER_MODEL`) → diversificación (máx. 2 fragmentos por documento) → top 8.
Si el reranker está apagado o falla, `combinar_y_diversificar` actúa como respaldo (con un aviso en el log).
El reranker corre en CPU (torch CPU); la primera carga tarda ~30 s, por eso `RERANKER_PRECARGAR=true` lo carga al arrancar.

### Entrega de documentos (qué pide el usuario)
El analista distingue **contenido** de **archivo**. Dos capas de control:

1. **El analista decide (LLM):** "¿qué dice el Decreto 83?" → solo contenido. "Descárgame / mándame el PDF / ¿dónde lo veo completo?" → solo el
   archivo (`solicita_archivo=true`, `incluye_contenido=false`). "Dame / necesito / quiero / pásame el decreto 83" (no dice qué quiere de él) → **archivo y
   contenido** (se explica con citas y además se entrega el botón). Sin referencia ni documento activo ("pásame el archivo") se pide aclaración.
2. **Compuerta determinística (sin LLM):** `preparar_descarga` solo existe para el investigador cuando `solicita_archivo` es true; la API devuelve `archivos: []`
   si el analista no lo marcó; y la URL la construye el backend (`/documentos/{archivo}/descargar`), así que el modelo no puede inventar enlaces.

Si hay varios documentos posibles se pide aclaración (no se entrega ninguno); si el documento existe pero su original no está en el repositorio, el agente lo explica.

## API (contrato estable; los cambios son aditivos)
| Endpoint | Descripción |
|---|---|
| `POST /chat` `{mensaje, thread_id, perfil}` | → `{respuesta, thread_id, pasos[], fuentes[], candidatos[], archivos[], tiempo_ms}`; `archivos[] = {titulo, archivo, url_descarga}` (relativa; vacío salvo que el usuario pida el archivo). |
| `POST /chat-voz` (form: `audio`, `thread_id`, `perfil`) | → `{transcripcion, respuesta, thread_id, pregunta_audio_url, respuesta_audio_url, fuentes, candidatos, archivos, alerta}`. Si el audio no es comprensible: `alerta=true` y sin audio de respuesta. |
| `GET /documentos/{archivo}/descargar` | Entrega el original del documento indexado (`Content-Disposition: attachment`, nombre original). `404` amable si no está en el catálogo o no tiene original; `503` si el repositorio (Drive) no está disponible. `DRIVE_MODO_ENTREGA=redirect` responde `302` al enlace de Drive. |
| `GET /historial/{thread_id}` | Turnos del hilo con URLs firmadas de audio. |
| `GET /documentos/sugerencias?q=` | Autocompletado del catálogo. |
| `POST /admin/lotes` | Recibe uno o varios PDF/PNG y prepara un lote sin generar embeddings. Requiere `X-Admin-Token`. |
| `GET /admin/lotes/{lote_id}` | Consulta el estado y los errores por archivo de un lote administrativo. Requiere `X-Admin-Token`. |
| `POST /admin/lotes/{lote_id}/indexar` | Confirma la metadata editable e indexa los documentos listos en Chroma. Requiere `X-Admin-Token`. |
| `POST /admin/lotes/{lote_id}/cancelar` | Cancela un lote pendiente y elimina sus originales sin indexar. Requiere `X-Admin-Token`. |
| `POST /admin/lotes/{lote_id}/revertir-indexacion` | Elimina de Chroma todo lo añadido por un lote indexado, manda sus originales a la papelera del repositorio, borra sus filas de `archivos_documento` y devuelve el lote a pendiente. Requiere `X-Admin-Token`. |
| `GET /health` | Estado de la API; con `CHROMA_MODO=http` también el de Chroma (`{"status":"ok","chroma":"ok"}`, o `503` si el servidor no responde). |
| `GET /health/llm` | Estado del modelo de Gemini. |

## Variables de entorno
Copia `.env.example` a `.env` (nunca se versiona). Las principales:

| Variable | Por defecto | Descripción |
|---|---|---|
| `GOOGLE_API_KEY` | — | **Obligatoria.** Embeddings, chat, STT y TTS de Gemini. |
| `ANALISTA_MODEL`, `EVALUADOR_MODEL` | `gemini-3.5-flash-lite` | Modelos de clasificación (baratos). |
| `INVESTIGADOR_MODEL`, `REDACTOR_MODEL` | `gemini-3.5-flash` | Modelos de investigación y redacción. |
| `LLM_MAX_RETRIES` | `3` | Reintentos con backoff ante errores transitorios de Gemini. |
| `MAX_INTENTOS_BUSQUEDA` | `2` | Máximo de pasadas del investigador por turno. |
| `INVESTIGADOR_MAX_PASOS` | `10` | `recursion_limit` del subagente. |
| `MAX_HISTORIAL_ANALISTA` | `6` | Mensajes de historial que ve el analista. |
| `EVALUADOR_HABILITADO` | `true` | `false` ahorra cuota: siempre pasa al redactor. |
| `CHECKPOINTER` | `postgres` | `postgres` o `memory`. |
| `RERANKER_HABILITADO` / `RERANKER_MODEL` / `RERANKER_PRECARGAR` | `true` / `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` / `false` | Reranker. |
| `DATABASE_URL` | `postgresql://sigem:sigem@db:5432/sigem` | Historial de turnos y checkpointer. |
| `CHROMA_MODO` | `local` | `local` (carpeta `CHROMA_PERSIST_DIR`: tests, desarrollo y origen de la migración) o `http` (servidor Chroma; **Compose lo fuerza a `http`**). |
| `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_SSL` | `localhost`, `8000`, `false` | Servidor Chroma (modo `http`). En Compose: `chroma`, `8000`. |
| `CHROMA_PERSIST_DIR`, `COLLECTION_NAME`, `EMBEDDING_MODEL` | `./chroma_db`, `langchain`, `models/gemini-embedding-2` | Índice vectorial existente (no cambiar modelo ni colección). |
| `ARCHIVOS_BACKEND` | `local` | Dónde viven los originales: `drive` (Google Drive) o `local` (`UPLOAD_ROOT/repositorio`, sin credenciales). |
| `GOOGLE_DRIVE_FOLDER_ID` | — | ID o URL completa de la carpeta de Drive con los PDF. |
| `GOOGLE_OAUTH_CLIENT_FILE`, `GOOGLE_OAUTH_TOKEN_FILE` | `./secrets/oauth-client.json`, `./secrets/oauth-token.json` | Credenciales OAuth de Drive (fuera de Git; en Compose, `/app/secrets` de solo lectura). |
| `DRIVE_MODO_ENTREGA` | `stream` | `stream` (el backend reenvía el archivo) o `redirect` (302 al enlace de Drive). |
| `ADMIN_UPLOAD_TOKEN` | — | Secreto obligatorio para los endpoints y la pantalla administrativa. |
| `UPLOAD_ROOT` | `./documentos` | Carpeta persistente de lotes pendientes e indexados; no se versiona. |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `1000`, `200` | Particionado compatible con los documentos históricos. |
| `EMBEDDING_BATCH_SIZE`, `EMBEDDING_BATCH_PAUSE_SECONDS` | `25`, `20` | Lotes y pausa de Gemini durante la indexación administrativa. |
| `AWS_*`, `S3_BUCKET_NAME` | — | Audios en S3. |
| `LANGFUSE_*` | — | Trazas (opcional). |

> **Cuota de Gemini.** Cada consulta con búsqueda gasta de 3 a 6 llamadas (analista, investigador, evaluador,
> redactor). El plan gratuito limita `gemini-3.5-flash` a ~20 solicitudes **por día** por proyecto; para uso real
> hace falta facturación o mover investigador/redactor a flash-lite.

## Cómo correr

```bash
cd backend
uv sync                                   # instala dependencias (torch CPU vía índice pytorch-cpu)
cp .env.example .env                      # y completa GOOGLE_API_KEY, etc.
uv run python -m agente_sigem.api         # API en :8001 (necesita PostgreSQL en DATABASE_URL)
```
`CHECKPOINTER=memory` mueve solo la memoria conversacional a memoria en proceso; el historial de turnos
(`/chat`, `/historial`) sigue necesitando PostgreSQL. Para levantarlo rápido: `docker compose -f Compose.yml up db`
(expone `localhost:55432`; usa `DATABASE_URL=postgresql://sigem:sigem@localhost:55432/sigem`).

Con Docker (Postgres + Chroma + API + frontend), desde la raíz del repositorio:

```bash
docker compose -f Compose.yml up --build
```
El servicio `backend` usa `CHECKPOINTER=postgres`, `CHROMA_MODO=http` contra el servicio `chroma`, el volumen `hf-cache` para el
modelo del reranker (`HF_HOME=/app/.cache/huggingface`) y monta `backend/secrets` (solo lectura) con las credenciales de Drive.
Ya no monta `backend/chroma_db`: el índice vive en el volumen `chroma-data` del servicio `chroma`.

## Chroma como servicio

Chroma corre en su propio contenedor (`chromadb/chroma:1.5.9`, **la misma versión que el paquete `chromadb` del backend**: cliente y servidor de
versiones distintas suelen fallar). Escala aparte de la API (más recursos, varias réplicas del backend comparten el mismo índice) y la
ingesta y las consultas ya no compiten por un archivo SQLite dentro del proceso. El puerto 8000 solo se publica en `127.0.0.1` (migración y depuración);
el backend lo usa por la red interna de Compose. Si algún día se mueve a otra máquina basta con cambiar `CHROMA_HOST`.

`obtener_catalogo_documentos()` guarda el `collection.count()` con el que construyó el catálogo y lo reconstruye si cambia, así que un documento indexado por
otro proceso (otra réplica) aparece sin invalidación explícita; `invalidar_cache_documentos()` sigue existiendo para el proceso que indexa.

**Primera vez (el volumen `chroma-data` está vacío):** copia el índice existente **sin regenerar embeddings** con el script de migración
(solo lee `backend/chroma_db`, trabaja sobre una copia, es idempotente y verifica conteo, ids y top-8):

```bash
docker compose -f Compose.yml up -d chroma
cd backend
CHROMA_MODO=http uv run python scripts/migrar_chroma.py            # migra y verifica
CHROMA_MODO=http uv run python scripts/migrar_chroma.py --solo-verificar
```
El HNSW es aproximado: el top-8 de origen y destino puede diferir en vecinos casi empatados, por eso la verificación compara cada lado con el top-8 exacto y
exige que el destino no recupere peor que el origen. **No** cambies `EMBEDDING_MODEL` ni `COLLECTION_NAME`.

## Originales y Google Drive

Los PDF originales se guardan en un repositorio de archivos (`agente_sigem/repositorio_archivos.py`): `RepositorioDrive` (Drive API v3, OAuth de usuario) o
`RepositorioLocal` (desarrollo y tests). La tabla `archivos_documento` de Postgres enlaza cada `nombre_archivo` de Chroma con su `id_externo`
(no se modifica la colección). `retirar()` mueve a la papelera; **nunca borra de forma permanente**.

**Configuración del acceso a la carpeta (una vez):**
1. La carpeta de Drive se comparte **por correo** con la cuenta de Google que usará el backend, con rol **Editor** (Lector basta solo para descargar; Editor es
   necesario para que la ingesta suba los documentos nuevos).
2. El acceso general del enlace queda en **Lector** (o restringido). **No** lo dejes como "cualquiera con el enlace puede editar": cualquiera podría subir o borrar decretos.
3. Si la carpeta está en una unidad personal, los archivos que suba el backend pertenecen a esa cuenta y ocupan su espacio; lo ideal a largo plazo es que la
   carpeta pertenezca a una cuenta del proyecto.

**Credenciales OAuth (paso a paso, una sola vez):**
1. En [Google Cloud Console](https://console.cloud.google.com) crea o elige un proyecto (puede ser distinto del de la `GOOGLE_API_KEY`) y habilita la **Google Drive API**.
2. **Google Auth Platform** → *Branding*: nombre de la app y correo de asistencia. → *Público* (*Audience*): tipo **Externo**.
   - En modo **Pruebas** agrega como *usuario de prueba* la cuenta que autorizará; **el token caduca a los 7 días** y hay que repetir el paso 5.
   - En **Producción** (publicar la app) el token no caduca, pero Google exige además URLs de página principal y de política de privacidad.
3. *Clientes* → **Crear cliente** → tipo **Aplicación de escritorio** → descarga el JSON y guárdalo como `backend/secrets/oauth-client.json` (esa carpeta está en `.gitignore` y `.dockerignore`).
4. En `backend/.env`: `ARCHIVOS_BACKEND=drive` y `GOOGLE_DRIVE_FOLDER_ID=<ID o URL de la carpeta>`.
5. Autoriza (abre el navegador; inicia sesión con la cuenta que tiene la carpeta y acepta la advertencia de app no verificada: *Avanzado* → *Ir a …*):
   ```bash
   cd backend && uv run python scripts/autorizar_drive.py        # crea secrets/oauth-token.json
   ```
6. Enlaza los documentos ya indexados con sus PDF de Drive (simulación por defecto; con `--aplicar` escribe la tabla; con `--subir-faltantes DIR` sube los que falten):
   ```bash
   DATABASE_URL=postgresql://sigem:sigem@localhost:55432/sigem CHROMA_MODO=http \
     uv run python scripts/sincronizar_drive.py                  # reporte: enlazados / en Chroma sin Drive / en Drive sin indexar
   DATABASE_URL=... CHROMA_MODO=http uv run python scripts/sincronizar_drive.py --aplicar [--subir-faltantes ./carpeta_con_pdfs]
   ```
Si el token falta, está dañado o caducó, la API **arranca igual** y el chat funciona; solo `GET /documentos/{archivo}/descargar` responde `503`
(y la ingesta avisa al admin que no pudo guardar el original). Se arregla repitiendo `autorizar_drive.py`.

## Carga administrativa

Configura `ADMIN_UPLOAD_TOKEN` en `backend/.env`, inicia los contenedores y abre
`http://localhost:8080/admin.html`. El token se mantiene solo en memoria del navegador y se envía en el encabezado
`X-Admin-Token`; el backend lo valida en cada operación.

La pantalla acepta hasta 20 PDF o PNG por lote. La primera acción guarda los archivos en
`documentos/pendientes/<lote_id>/`, extrae texto (con OCR en español cuando hace falta) y propone la metadata.
La segunda acción confirma la indexación y, por cada documento: **(1)** sube el original al repositorio de archivos (si falla, el documento no se indexa y el
administrador ve el motivo), **(2)** aplica chunks de 1000 caracteres con overlap de 200, genera embeddings con `models/gemini-embedding-2` e inserta los chunks y su
metadata en la colección `langchain`, **(3)** registra la fila en `archivos_documento` y mueve el original local a `documentos/indexados/<lote_id>/`. Si falla el paso 2 o el 3,
se limpia Chroma y el original sube a la papelera. Los documentos omitidos (mismo contenido o título) no se suben. Un manifiesto conserva el resultado individual de cada
archivo, con su `id_externo` y, si el título final no se lee como tipo y número ("Decreto 045 de 2026"), una `advertencias` (el documento se encuentra por contenido, no por referencia).
Revertir el lote retira los originales (papelera) y borra sus filas; si falla a medias se puede repetir.

Un documento nuevo funciona con **todas** las tools del investigador sin reiniciar nada: sale del mismo Chroma y la misma tabla (limitación conocida: "documento 18399"
solo existe para los originales cuyo nombre empieza por el código de SIGEM).

## Solución de problemas
- **El reranker no se descarga en Docker** (`CERTIFICATE_VERIFY_FAILED ... self-signed certificate`): ocurre en redes con un
  proxy o antivirus que intercepta TLS. La API **sigue funcionando** con el respaldo léxico (un aviso en el log) y la
  precarga corre en segundo plano, sin bloquear el arranque. Opciones: precargar el volumen `hf-cache` con la caché del
  host (`~/.cache/huggingface`), usar `HF_HUB_OFFLINE=1` una vez poblado, o `RERANKER_HABILITADO=false`.
- **`429 RESOURCE_EXHAUSTED`**: cuota de Gemini agotada (ver el aviso de cuota arriba). `/chat` responde 200 con un mensaje de "mucha demanda" y `/chat-voz` con `alerta=true`; el turno no se guarda, así que el usuario puede reintentar. El detalle queda en el log del backend.

- **Chroma `503` en `/health` o error al arrancar:** el servicio `chroma` no está sano. `docker compose -f Compose.yml ps` y revisa que el volumen `chroma-data` tenga el índice (`migrar_chroma.py --solo-verificar`).
- **`GET /documentos/.../descargar` responde `503`:** falta, está dañado o caducó `secrets/oauth-token.json` (con la app en modo *Pruebas* caduca a los 7 días): repite `scripts/autorizar_drive.py`. Si responde `404`, ese documento no tiene su original enlazado (ver `sincronizar_drive.py`).
- **La ingesta dice que no pudo guardar el original (`permiso de Editor`):** la cuenta del token no es Editor de la carpeta; compártela por correo con rol Editor.

## Pruebas
Los tests **no llaman a Gemini, al Chroma servidor ni a Drive** (LLM falsos, Chroma en memoria con embeddings falsos, cliente de Drive falso y `RepositorioLocal`).

```bash
cd backend
uv run pytest -q
uv run python -c "import agente_sigem.api"     # debe importar sin Postgres
```

## Evaluación
Dataset: `evaluacion/dataset.jsonl` (54 casos con documentos reales del catálogo, 14 categorías, perfiles P0–P3,
turnos múltiples; la categoría `solicitud_archivo` mezcla pedidos de archivo, de contenido y mixtos).

```bash
cd backend
# Solo recuperación (sin LLM, solo embeddings): hit@8 y MRR por categoría
uv run python evaluacion/evaluar_recuperacion.py
uv run python evaluacion/evaluar_recuperacion.py --sin-reranker
# contra el Chroma servidor (CHROMA_MODO=http CHROMA_HOST=localhost CHROMA_PORT=8000) en vez de la carpeta local

# Agente completo (grafo con CHECKPOINTER=memory; consume cuota de Gemini)
uv run python evaluacion/evaluar_agente.py --estimar                         # cuántas llamadas implica
uv run python evaluacion/evaluar_agente.py --pausa 6 --categorias saludo voz_ruido
uv run python evaluacion/evaluar_agente.py --pausa 8                         # todo el dataset
```
`evaluar_agente.py` mide, sobre el último turno de cada caso: acierto de intención, presencia de los archivos
esperados en `fuentes` (o en `candidatos`), candidatos cuando se esperan, alerta de audio, que la respuesta
**no niegue** un documento que existe (`info`), que diga "no existe" sin fuentes de relleno cuando no existe (`neg.`) y, en los casos que lo
definen, que el analista marque bien `solicita_archivo` / `incluye_contenido` y que el archivo esperado llegue en `archivos` (`sol.arch` y `entrega`).
Marca como *degradados* los casos en que el analista o el evaluador cayeron en fail-open y reintenta si se agota la
cuota por minuto (usar `--pausa 20` con flash-lite; `--ids inv-04 inex-01` para casos sueltos). Escribe
`evaluacion/resultados_<fecha>.json` y detiene la corrida si se agota la cuota diaria.
La línea base previa a la migración está en `evaluacion/linea_base_recuperacion.json`.

## Estructura
```
agente_sigem/
  api.py            FastAPI: endpoints, lifespan, ejecutar_turno
  agent.py          construir_grafo(checkpointer)
  estado.py         EstadoAgente, AnalisisConsulta, FiltrosSigem, EvaluacionEvidencia
  prompts.py        prompts de los agentes, estilos por perfil, respuestas directas
  nodos/            analista, respuesta_directa, investigador, evaluador, redactor
  tools.py          las 4 tools de búsqueda del investigador + preparar_descarga (aparte)
  retrieval.py      búsqueda híbrida + diversificación; cliente Chroma (local/http) y catálogo que se reconstruye solo
  repositorio_archivos.py   repositorio de originales: Drive (OAuth) o local; errores en español
  ingest.py         carga administrativa: lotes, embeddings, subida del original, reversa
  reranker.py       CrossEncoder con respaldo
  catalogo.py       catálogo de documentos (referencias, códigos, similares)
  llm.py            fábrica de modelos Gemini (caché por modelo y temperatura)
  checkpointer.py   PostgresSaver o MemorySaver
  config.py, db.py, storage.py, audio.py, tracing.py
scripts/            migrar_chroma.py, autorizar_drive.py, sincronizar_drive.py
secrets/            credenciales OAuth de Drive (no se versiona)
evaluacion/         dataset y scripts de evaluación
tests/              pruebas (fakes en fakes_llm.py y grafo_falso.py)
```
