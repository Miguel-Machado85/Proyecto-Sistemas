# Backend — Agente ciudadano SIGEM (Alcaldía de Marinilla)

API (FastAPI) de un **sistema multi-agente** (LangGraph + Gemini) que responde preguntas sobre las normas del
repositorio SIGEM (≈72 PDFs de decretos, resoluciones, leyes y acuerdos indexados en Chroma), adaptando la
respuesta al perfil del usuario (P0–P3). Acepta texto y voz.

> La especificación de la migración está en [`../docs/MIGRACION_AGENTICA.md`](../docs/MIGRACION_AGENTICA.md)
> (con una sección final "Estado de implementación" que lista las desviaciones).

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
                                      │                                      │
                                      ▼                                      │
                          evaluador (flash-lite) ── evidencia insuficiente ──┘
                                      │ suficiente / hay candidatos / sin resultados
                                      ▼
                          redactor (flash): estilo por perfil, citas en línea ─► END
```

| Agente / nodo | Modelo (variable) | Rol |
|---|---|---|
| `analista` | flash-lite (`ANALISTA_MODEL`), temp. 0 | Entiende la pregunta con salida estructurada (`AnalisisConsulta`): intención, consulta autónoma, filtros (tipo, número, año, código, tema) y si se refiere al documento activo. Corrige ortografía y convierte números dichos en palabras. **Fail-open**: si falla, sigue como `consulta_sigem`. |
| `respuesta_directa` | — (sin LLM) | Saludo, fuera de dominio, aclaración o alerta de audio no comprensible. |
| `investigador` | flash (`INVESTIGADOR_MODEL`), temp. 0.1 | Subagente ReAct (`create_agent`) que decide qué tools usar y reúne la evidencia. Historial nuevo en cada intento, límite `INVESTIGADOR_MAX_PASOS`. |
| `evaluador` | flash-lite (`EVALUADOR_MODEL`), temp. 0 | Corrective RAG: decide si la evidencia basta o reformula la consulta. Atajos sin LLM; se puede apagar (`EVALUADOR_HABILITADO=false`). Fail-open. |
| `redactor` | flash (`REDACTOR_MODEL`), temp. 0.3 | Redacta solo con la evidencia, cita `[Decreto 001 de 2019, p. 3]`, adapta el estilo al perfil y actualiza el `documento_activo`. Tres modos: respuesta, aclaración y sin resultados. |

**Regla de diseño:** lo que *siempre* debe ocurrir es un nodo con arista fija; lo que *depende de la pregunta* es una
tool que el investigador decide usar. No se interpreta con regex lo que escribe el usuario: eso lo hace el analista
(o el investigador al llenar los argumentos de las tools) y el código **valida contra el catálogo**.

### Estado y memoria
`EstadoAgente` (`agente_sigem/estado.py`) separa lo que **persiste** entre turnos (`messages` con solo la pregunta
y la respuesta final, y `documento_activo`) de lo que se **reinicia** en cada turno (`analisis`, `evidencia`,
`fuentes`, `candidatos`, `intentos_busqueda`, `consulta_actual`, `pasos`, `es_alerta`; sin reducer, la API los pasa
vacíos). El checkpointer es **PostgreSQL** (`CHECKPOINTER=postgres`) o memoria (`CHECKPOINTER=memory`, para
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

### Recuperación y reranker
`retrieval.recuperar_sigem(consulta, archivo=None)`: búsqueda literal en Chroma + búsqueda semántica (embeddings
`EMBEDDING_MODEL`, **no cambiar**: el índice ya está construido con él) → unión sin duplicados → **reranker**
(`sentence-transformers` `CrossEncoder`, `RERANKER_MODEL`) → diversificación (máx. 2 fragmentos por documento) → top 8.
Si el reranker está apagado o falla, `combinar_y_diversificar` actúa como respaldo (con un aviso en el log).
El reranker corre en CPU (torch CPU); la primera carga tarda ~30 s, por eso `RERANKER_PRECARGAR=true` lo carga al arrancar.

## API (contrato estable; el frontend no cambia)
| Endpoint | Descripción |
|---|---|
| `POST /chat` `{mensaje, thread_id, perfil}` | → `{respuesta, thread_id, pasos[], fuentes[], candidatos[], tiempo_ms}` |
| `POST /chat-voz` (form: `audio`, `thread_id`, `perfil`) | → `{transcripcion, respuesta, thread_id, pregunta_audio_url, respuesta_audio_url, fuentes, candidatos, alerta}`. Si el audio no es comprensible: `alerta=true` y sin audio de respuesta. |
| `GET /historial/{thread_id}` | Turnos del hilo con URLs firmadas de audio. |
| `GET /documentos/sugerencias?q=` | Autocompletado del catálogo. |
| `GET /health`, `GET /health/llm` | Estado de la API y del modelo de Gemini. |

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
| `CHROMA_PERSIST_DIR`, `COLLECTION_NAME`, `EMBEDDING_MODEL` | `./chroma_db`, `sigem_documentos`, `models/gemini-embedding-2` | Índice vectorial (no reindexar). |
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

Con Docker (Postgres + API + frontend), desde la raíz del repositorio:

```bash
docker compose -f Compose.yml up --build
```
El servicio `backend` usa `CHECKPOINTER=postgres`, el volumen `hf-cache` para el modelo del reranker
(`HF_HOME=/app/.cache/huggingface`) y monta `backend/chroma_db`.

## Solución de problemas
- **El reranker no se descarga en Docker** (`CERTIFICATE_VERIFY_FAILED ... self-signed certificate`): ocurre en redes con un
  proxy o antivirus que intercepta TLS. La API **sigue funcionando** con el respaldo léxico (un aviso en el log) y la
  precarga corre en segundo plano, sin bloquear el arranque. Opciones: precargar el volumen `hf-cache` con la caché del
  host (`~/.cache/huggingface`), usar `HF_HUB_OFFLINE=1` una vez poblado, o `RERANKER_HABILITADO=false`.
- **`429 RESOURCE_EXHAUSTED`**: cuota de Gemini agotada (ver el aviso de cuota arriba). `/chat` responde 200 con un mensaje de "mucha demanda" y `/chat-voz` con `alerta=true`; el turno no se guarda, así que el usuario puede reintentar. El detalle queda en el log del backend.

## Pruebas
Los tests **no llaman a Gemini ni al Chroma real** (LLM falsos, Chroma en memoria con embeddings falsos).

```bash
cd backend
uv run pytest -q
uv run python -c "import agente_sigem.api"     # debe importar sin Postgres
```

## Evaluación
Dataset: `evaluacion/dataset.jsonl` (43 casos con documentos reales del catálogo, 13 categorías, perfiles P0–P3,
turnos múltiples).

```bash
cd backend
# Solo recuperación (sin LLM, solo embeddings): hit@8 y MRR por categoría
uv run python evaluacion/evaluar_recuperacion.py
uv run python evaluacion/evaluar_recuperacion.py --sin-reranker

# Agente completo (grafo con CHECKPOINTER=memory; consume cuota de Gemini)
uv run python evaluacion/evaluar_agente.py --estimar                         # cuántas llamadas implica
uv run python evaluacion/evaluar_agente.py --pausa 6 --categorias saludo voz_ruido
uv run python evaluacion/evaluar_agente.py --pausa 8                         # todo el dataset
```
`evaluar_agente.py` mide, sobre el último turno de cada caso: acierto de intención, presencia de los archivos
esperados en `fuentes` (o en `candidatos`), candidatos cuando se esperan, alerta de audio, que la respuesta
**no niegue** un documento que existe (`info`) y que diga "no existe" sin fuentes de relleno cuando no existe (`neg.`).
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
  tools.py          las 4 tools del investigador
  retrieval.py      búsqueda híbrida + diversificación
  reranker.py       CrossEncoder con respaldo
  catalogo.py       catálogo de documentos (referencias, códigos, similares)
  llm.py            fábrica de modelos Gemini (caché por modelo y temperatura)
  checkpointer.py   PostgresSaver o MemorySaver
  config.py, db.py, storage.py, audio.py, tracing.py
evaluacion/         dataset y scripts de evaluación
tests/              pruebas (fakes en fakes_llm.py y grafo_falso.py)
```
