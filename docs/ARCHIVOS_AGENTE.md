# Plan: entrega de documentos, Google Drive y Chroma como servicio

Rama de trabajo: `feature-archivosAgente` (hoy idéntica a `main`, commit `f74111f`).
Este documento tiene dos usos: es el plan para ustedes y, copiado a `docs/ARCHIVOS_AGENTE.md`, es la
especificación que Claude Code lee en cada fase (igual que `MIGRACION_AGENTICA.md` en la migración).

### Decisiones confirmadas (23 de octubre)

- **Drive:** la carpeta `lossetentayuno` (ID `1fE_77pk8np6OEt3dIHN7fO4gmbuC6nZe`) está en la **unidad
  personal** del compañero, compartida por enlace como **lector**. Por eso:
  - se descarta la cuenta de servicio (no puede subir archivos a una unidad personal);
  - el backend se autentica con **OAuth como un usuario del equipo** al que el dueño agregue como
    **Editor por correo**. El enlace público se queda como lector: no se cambia a "cualquiera puede editar";
  - mientras llega ese permiso, todo lo de **leer y entregar** los 71 documentos funciona con acceso de
    lector. Solo la **subida** de documentos nuevos (fase 3) necesita ser Editor.
- **Credenciales:** hoy no hay ninguna; el `.env` solo apunta a `chroma_db`. Se crean en la fase 2.
- **Chroma:** el requisito es que sea **escalable para RAG**. Se cumple con Chroma como **contenedor
  servidor** en Compose (opción 1). Chroma Cloud y EC2 quedan fuera de alcance; el código solo soporta
  `CHROMA_MODO=local|http`.

---

## 1. Qué queda funcionando al final

| Fase | Resultado |
|---|---|
| 0 | Rama lista, `CLAUDE.md` actualizado con las nuevas reglas, versiones verificadas |
| 1 | Chroma corre como **servicio aparte** (contenedor), con el índice migrado sin re-generar embeddings |
| 2 | Módulo `repositorio_archivos` (Drive o local), tabla en Postgres, endpoint de descarga, 71 documentos actuales enlazados |
| 3 | La ingesta del admin sube el original a Drive al indexar y lo retira al revertir |
| 4 | El agente detecta cuándo el usuario pide el archivo y lo entrega como botón de descarga |
| 5 | `index.html` muestra el botón de descarga (texto y voz) |
| 6 | Evaluación, documentación (`home.md`, READMEs) y PR a `main` |

Las fases 1 y 2 son independientes entre sí; la 4 necesita la 2. **Solo la fase 3 necesita permiso de
Editor en Drive**, así que va de última entre las de código: **0 → 1 → 2 → 4 → 5 → 3 → 6**. Si el
tiempo aprieta, la prioridad es entregar documentos: **2 → 4 → 5**.

---

## 2. Parte A: detectar que el usuario realmente pide el documento

### El problema
Hoy "¿qué dice el Decreto 83 de 2018?" y "pásame el Decreto 83 de 2018" llegan al mismo flujo y
ambos terminan en un resumen. Necesitamos distinguir **pedir el contenido** de **pedir el archivo**,
y también el caso mixto ("pásame el decreto y dime de qué trata").

### Diseño: dos capas de control

**Capa 1, el analista decide (LLM).** Se agrega un campo a `AnalisisConsulta` en `estado.py`:

```python
solicita_archivo: bool = False  # True solo si el usuario pide el ARCHIVO (descargar, enviar, PDF, formato para llenar)
```

No es una intención nueva: la petición de archivo sigue siendo `consulta_sigem` o `seguimiento`,
porque igual hay que **resolver qué documento es** con las tools que ya existen. Así no se rompe
`ruta_despues_analista` ni ninguno de los casos actuales.

`PROMPT_ANALISTA` recibe reglas y ejemplos few-shot **en pares contrastados** con documentos reales del
catálogo:

| Mensaje | `solicita_archivo` |
|---|---|
| "¿Qué dice el Decreto 001 de 2019?" | false |
| "Descárgame el Decreto 001 de 2019" / "mándame el PDF" | true |
| "Necesito el formato para … " (cuando es un formulario) | true |
| (con documento activo) "pásamelo" / "¿me lo puedes enviar?" | true, `usa_documento_activo=true` |
| "¿Dónde puedo ver el decreto completo?" | true |
| "Pásame el decreto de taxis y dime cuánto cuesta la carrera mínima" | true (y se busca evidencia igual) |

**Capa 2, compuerta determinística (sin LLM).** El LLM del investigador *físicamente no puede*
entregar un archivo si el analista no lo marcó:

- La tool nueva `preparar_descarga` **solo se agrega a la lista de herramientas del subagente cuando
  `analisis.solicita_archivo` es true**. En los demás turnos ni siquiera existe para el modelo.
- `api.py` descarta cualquier `archivos` del estado si `solicita_archivo` es false (defensa en profundidad).
- La URL de descarga **la construye el backend**, nunca el LLM: no hay forma de que invente un enlace.

Esto es la "revisión" que piden: el clasificador decide, y el código garantiza que la decisión se respete.

### Cambios en el grafo (sin nodos nuevos)

```
analista ──(solicita_archivo=true)──▶ investigador ──▶ evaluador ──▶ redactor
                                        │ resuelve el documento con las tools actuales
                                        └ llama preparar_descarga(archivo)
```

| Pieza | Cambio |
|---|---|
| `estado.py` | `solicita_archivo` en `AnalisisConsulta`; campo por turno `archivos: list[dict]` en `EstadoAgente` (sin reducer, como los demás) |
| `tools.py` | `preparar_descarga(archivo: str)`: valida el archivo contra el catálogo y contra la tabla `archivos_documento`; devuelve `{"archivos": [{"titulo","archivo","url_descarga"}]}` o `{"error": "...no disponible para descarga"}`. No lanza excepciones, como las otras 4 |
| `nodos/investigador.py` | Lista de tools dinámica según `solicita_archivo`; el encargo le dice que primero identifique el documento (referencia, búsqueda o documento activo) y luego llame `preparar_descarga`; `acumular_resultados` extrae `archivos` sin duplicados (máx. 3) |
| `nodos/evaluador.py` | Atajo sin LLM: si hay `archivos` y no hay `evidencia` → redactor. Si el pedido es mixto, evalúa la evidencia como siempre |
| `nodos/redactor.py` + `prompts.py` | Modo nuevo `entrega` (archivos sin evidencia): mensaje breve en el estilo del perfil ("Aquí tienes el Decreto 001 de 2019…"). Regla para todos los modos: **nunca escribir URLs**, la interfaz muestra el botón. Si se pidió archivo pero hubo varios candidatos → modo `aclaracion` que ya existe. `documento_activo` se actualiza con el archivo entregado |
| `api.py` | `ChatResponse` y `ChatVozResponse` ganan `archivos: list[ArchivoDescargable] = []`. Es un cambio **aditivo**: el frontend actual sigue funcionando sin tocarlo |

### Persistencia en el historial (opcional)
Sin persistir, al recargar la página se ve el texto "Aquí tienes…" pero no el botón. Si hay tiempo:
columna JSON nullable `archivos` en `turnos`, creada con `ALTER TABLE … ADD COLUMN IF NOT EXISTS` en
`init_db` (`create_all` no altera tablas existentes), y `/historial` la devuelve.

---

## 3. Parte B: conexión con Google Drive

### Qué hace la conexión
1. **Enlazar** los ~71 PDFs ya indexados con su copia en la carpeta de Drive (la que está en el `.env`).
2. **Servir** el archivo cuando el agente lo entrega.
3. **Subir** cada documento nuevo que el admin indexe, y retirarlo si se revierte la indexación.

### Herramientas
- `google-api-python-client` (Drive API v3) y `google-auth` / `google-auth-oauthlib`.
- Un proyecto de Google Cloud con la **Drive API habilitada**. Pueden usar el mismo proyecto donde
  crearon la `GOOGLE_API_KEY` de Gemini. Ojo: la API key de Gemini **no sirve** para Drive; Drive
  necesita una identidad (cuenta de servicio u OAuth).

### Decisión 1: cómo se autentica el backend
**Decidido: OAuth de usuario** (ver "Decisiones confirmadas"). La tabla queda como registro de por qué:

| Opción | Cuándo sirve | Ventaja | Trampa |
|---|---|---|---|
| **Cuenta de servicio + Unidad compartida (Shared Drive)** | Si tienen Google Workspace (p. ej. cuenta institucional de la universidad) | Sin tokens que caduquen; un JSON de credenciales y listo | Una cuenta de servicio **no tiene cuota de almacenamiento propia**: no puede subir archivos a una carpeta de "Mi unidad" de un Gmail personal, solo a una Unidad compartida. Hay que pasar `supportsAllDrives=True` en cada llamada |
| **OAuth de usuario (token de refresco)** | Si la carpeta está en un Gmail personal | Funciona con cualquier cuenta | Hay que generar el token una vez con un script local. Si la pantalla de consentimiento se deja en modo "Testing", **el token caduca a los 7 días**: hay que pasar la app a "In production" (aparece la advertencia de app no verificada, se acepta, y ya) |

**Cómo queda con OAuth:**
- El token se genera **una sola vez** en el computador de quien tenga acceso a la carpeta, con
  `scripts/autorizar_drive.py`, y se guarda en `backend/secrets/oauth-token.json`.
- Ese usuario necesita **acceso por correo** a la carpeta, no basta con el enlace público:
  - como **Lector**, el backend ya puede listar y descargar los 71 PDFs (fases 2, 4 y 5);
  - como **Editor**, además puede subir los documentos nuevos (fase 3). Los archivos que suba ocupan
    la cuota de ese usuario, no la del dueño de la carpeta.
- Lo que hay que pedirle al compañero: **"Compartir → agregar mi correo como Editor"**. No cambiar el
  acceso general del enlace a Editor: cualquiera con el enlace podría borrar los decretos.
- Lo más limpio a largo plazo es que la carpeta pertenezca a una cuenta del proyecto (por ejemplo, un
  Gmail creado para SIGEM) y no a una persona; para el MVP basta con el permiso de Editor.

Alcance de OAuth: `https://www.googleapis.com/auth/drive`. El alcance reducido `drive.file` **no
sirve** aquí, porque solo da acceso a archivos que creó la propia app, y los 71 los subió su compañero.

### Decisión 2: cómo llega el archivo al usuario
**Recomendado: endpoint propio que hace streaming.** `GET /documentos/{archivo}/descargar` busca el
archivo en la tabla, lo descarga de Drive y lo envía con `Content-Disposition: attachment`.

- La carpeta de Drive puede seguir **privada**.
- El frontend no sabe que existe Drive: si mañana pasan los PDFs a S3 (ya usan S3 para audio), solo
  cambia la implementación del repositorio.
- Es el mismo patrón que ya usan con los audios: se guarda una **clave**, no una URL.

Alternativa más simple: carpeta con "cualquiera con el enlace puede ver" y el endpoint responde con
un redirect 302 al `webViewLink`. Los decretos son públicos, así que es aceptable, pero ata el
frontend a Drive. Se puede dejar como opción `DRIVE_MODO_ENTREGA=stream|redirect`.

### Diseño del módulo `agente_sigem/repositorio_archivos.py`

```python
class RepositorioArchivos(Protocol):
    def subir(self, ruta: Path, nombre: str, mime_type: str) -> str: ...      # devuelve id_externo
    def descargar(self, id_externo: str) -> Iterator[bytes]: ...              # para StreamingResponse
    def retirar(self, id_externo: str) -> None: ...                           # papelera, NUNCA borrado permanente
    def listar(self) -> list[ArchivoRemoto]: ...                              # para la sincronización inicial

class RepositorioDrive: ...   # Drive API v3, supportsAllDrives=True
class RepositorioLocal: ...   # carpeta en UPLOAD_ROOT/repositorio; para tests y desarrollo sin credenciales

def obtener_repositorio() -> RepositorioArchivos   # según ARCHIVOS_BACKEND=drive|local, singleton
```

**Tabla `archivos_documento` en Postgres** (en `db.py`, con SQLAlchemy como `turnos`):

| Columna | Tipo | Nota |
|---|---|---|
| `nombre_archivo` | text PK | La misma clave `nombre_archivo` de los metadatos de Chroma |
| `titulo` | text | |
| `proveedor` | text | `drive` o `local` |
| `id_externo` | text | ID del archivo en Drive |
| `mime_type`, `tamano_bytes`, `hash_sha256` | | |
| `creado_en` | timestamp | |

¿Por qué Postgres y no un metadato más en Chroma? Porque así **no se modifica la colección**, el enlace
sobrevive a la migración de Chroma de la parte C, y se puede consultar sin abrir el índice vectorial.

### Variables de entorno nuevas

```bash
ARCHIVOS_BACKEND=drive                 # drive | local
GOOGLE_DRIVE_FOLDER_ID=1fE_77pk8np6OEt3dIHN7fO4gmbuC6nZe   # acepta el ID o la URL completa (config extrae el ID)
GOOGLE_OAUTH_CLIENT_FILE=/app/secrets/oauth-client.json
GOOGLE_OAUTH_TOKEN_FILE=/app/secrets/oauth-token.json
DRIVE_MODO_ENTREGA=stream              # stream | redirect
```

Los JSON de credenciales viven en `backend/secrets/` (en `.gitignore` y `.dockerignore`), montados
**de solo lectura** en Compose. Nunca dentro de la imagen.

### Sincronización inicial de los 71 documentos
Script `backend/scripts/sincronizar_drive.py`:
1. Lista la carpeta de Drive y el catálogo de Chroma (`nombre_archivo`).
2. Empareja por nombre de archivo exacto y, si no, por nombre normalizado; reporta los dudosos sin
   enlazarlos.
3. Con `--aplicar` escribe la tabla; sin él, solo muestra el reporte (modo simulación por defecto).
4. Reporta tres listas: enlazados, **en Chroma pero no en Drive** (no se podrán descargar) y en Drive
   pero no indexados. Con `--subir-faltantes DIR` sube los que falten desde una carpeta local.

### Ingesta (coordinar con su compañero)
En `ingest._indexar_documento`, orden propuesto:
1. Subir el original al repositorio → `id_externo`.
2. Embeddings + `upsert` en Chroma (como hoy).
3. Registrar la fila en `archivos_documento`.
4. Si falla el paso 2, se retira el archivo de Drive (a la papelera). Si falla el 1, no se indexa: un
   documento que el agente encuentra pero no puede entregar es peor que un error visible al admin.

`revertir_indexacion` retira el archivo de Drive y borra la fila. El manifiesto del lote guarda
`id_externo` por archivo para que el revertir sepa qué retirar.

### Que el documento nuevo funcione con todas las tools del investigador
La mayor parte **ya existe** desde el Sprint 6 (`ingest.py`, en `main`). Las 4 tools no leen otra
fuente: todas salen de la misma colección de Chroma, así que un documento indexado por el admin
queda disponible para ellas sin código nuevo.

| Tool | De dónde sale el dato | Por qué ya funciona con un documento nuevo |
|---|---|---|
| `buscar_fragmentos` | Búsqueda textual y semántica en la colección | El upsert guarda texto y embeddings en la misma colección `langchain` |
| `leer_documento` | `where={"nombre_archivo": ...}` + `page` | Cada chunk lleva `nombre_archivo`, `page` y `page_label` |
| `buscar_documento_por_referencia` y `listar_documentos` | Catálogo en memoria (`catalogo.py`) armado con los metadatos de Chroma | Los chunks llevan `titulo`, `descripcion` y `fecha_expedicion`, y al terminar el lote se llama `invalidar_cache_documentos()` |
| `preparar_descarga` (nueva) | Catálogo + tabla `archivos_documento` | La fase 3 registra la fila al indexar |

Lo que **faltaba** y se agrega:

1. **Caché del catálogo con Chroma como servidor (fase 1).** `invalidar_cache_documentos()` solo
   limpia la caché del proceso que indexó. Con un solo proceso no pasa nada, pero con varios workers
   o réplicas del backend (que es justo lo que habilita el contenedor de Chroma), los demás procesos
   seguirían con el catálogo viejo: la búsqueda semántica encontraría el documento, pero
   "Decreto 045 de 2026" o "¿qué decretos hay de 2026?" no. La solución: guardar junto al catálogo el
   `collection.count()` con el que se construyó y reconstruirlo si cambió. Es una llamada barata por
   consulta y no requiere coordinación entre procesos.
2. **El título debe tener el formato canónico (fase 3).** `buscar_por_atributos` depende de que el
   título se pueda leer como "Decreto 045 de 2026". `_metadata_sugerida` ya lo propone así, pero el
   admin lo puede editar libremente. Si el título final no tiene tipo y número reconocibles, el
   resultado del lote debe advertirlo, sin bloquear la indexación (un documento sin número sigue
   encontrándose por texto y semántica).
3. **Limitación conocida:** "documento 18399" (`buscar_por_codigo`) solo funciona con los 71 originales,
   cuyo nombre empieza por el código de SIGEM. Los documentos subidos por el admin no lo tienen. Se
   documenta y no se cambia.
4. **Prueba de extremo a extremo (fase 3):** indexar un documento de prueba y comprobar que cada tool,
   incluida `preparar_descarga`, lo encuentra sin reiniciar la API.

---

## 4. Parte C: Chroma fuera del proceso del backend

### Situación actual
Chroma corre **embebido**: el backend abre `backend/chroma_db` con `langchain-chroma` y esa carpeta se
monta como volumen. No hay servidor. Es exactamente lo que les piden cambiar.

### Opciones

| Opción | Qué es | Herramientas | Pros | Contras |
|---|---|---|---|---|
| **1. Contenedor Chroma en Compose** | Servicio `chroma` con la imagen oficial `chromadb/chroma`; el backend se conecta por red con `chromadb.HttpClient` | Docker, `chromadb.HttpClient`, volumen nombrado | Dockerizado, gratis, cero cambios de datos, mismo `docker compose up` | Sigue en la misma máquina (aunque ya es un servicio independiente) |
| **2. Chroma Cloud** | Chroma gestionado por el fabricante; el backend usa `chromadb.CloudClient(tenant, database, api_key)` | Cuenta en trychroma.com, API key | Nada local; nada que operar | Cuenta externa, latencia por internet, depende de créditos gratuitos |
| **3. Contenedor Chroma en una VM** | La misma imagen de la opción 1, pero en un EC2 (ya tienen AWS) | Docker en EC2, security group | Remoto y bajo su control | Hay que asegurarlo: no expongan el puerto 8000 a internet. Restrinjan el security group a la IP del backend o pongan un proxy con autenticación delante |

**Decidido: opción 1.** El requisito es que sea escalable para RAG, y un servidor Chroma en su propio
contenedor lo cumple. Escala independiente del backend: se le pueden dar más recursos, varias réplicas
del backend comparten el mismo índice, y la ingesta y las consultas ya no compiten por un archivo
SQLite dentro del proceso de la API. Como `HttpClient` habla por red, mover el contenedor a otra máquina
más adelante es cambiar `CHROMA_HOST`.

```bash
CHROMA_MODO=http          # local | http
CHROMA_HOST=chroma        # nombre del servicio en Compose
CHROMA_PORT=8000
CHROMA_SSL=false
CHROMA_PERSIST_DIR=./chroma_db   # solo modo local (tests, desarrollo y origen de la migración)
```

### Cambios de código
- `retrieval.py`: una fábrica `crear_cliente_chroma()` según `CHROMA_MODO`, y
  `Chroma(client=cliente, collection_name=..., embedding_function=...)`. El resto de `retrieval.py`,
  `tools.py` e `ingest.py` (que usa `vector_store._collection.upsert`) funcionan igual contra un
  servidor.
- `/health` agrega `client.heartbeat()` para reportar si Chroma responde.
- `Compose.yml`:
  - servicio `chroma` con la imagen **fijada a la misma versión que el paquete `chromadb` instalado**
    (hoy `langchain-chroma` exige `chromadb>=1.3.5,<2`; cliente y servidor de versiones distintas
    suelen fallar);
  - volumen nombrado `chroma-data` en `/data`;
  - **sin puerto publicado** (solo el backend lo ve por la red interna de Compose); para la migración y
    depuración se puede publicar como `127.0.0.1:8000:8000`;
  - el backend con `depends_on` hacia `chroma`;
  - se elimina el montaje `./backend/chroma_db:/app/chroma_db`.

### Migración sin volver a generar embeddings
Script `backend/scripts/migrar_chroma.py` (destino: el servidor de `CHROMA_HOST`/`CHROMA_PORT`):
1. Abre la carpeta local con `chromadb.PersistentClient` (solo lectura, no se modifica nada).
2. Lee la colección `langchain` en páginas de 500 con `include=["embeddings","documents","metadatas"]`
   y los mismos `ids`.
3. Crea la colección en el destino **con la misma configuración de distancia** que la origen
   (copiar la métrica, por ejemplo `hnsw:space`; si no coincide, la búsqueda semántica cambia en silencio).
4. Inserta con `upsert` (idempotente: se puede re-ejecutar si se corta).
5. Verifica: mismo número de registros, mismos `ids`, y para 10 fragmentos al azar usa **su propio
   embedding guardado** como consulta en origen y destino y compara el top 8. Así la verificación no
   gasta cuota de Gemini.

Después, `evaluacion/evaluar_recuperacion.py` contra el servidor debe dar las mismas métricas que la
línea base (eso sí gasta cuota de embeddings, solo para las consultas).

> La regla de siempre sigue en pie: no cambiar `EMBEDDING_MODEL` ni `COLLECTION_NAME`. La migración
> copia vectores, no los recalcula.

---

## 5. Coordinación con su compañero (login de admin)

Archivos que probablemente tocan los dos: `config.py`, `.env.example`, `db.py`, `admin.py`,
`Compose.yml`, `api.py` (lifespan).

- **Él es dueño de** `admin.py` (autenticación) y `admin.html`. **Usted es dueño de** `ingest.py`,
  `retrieval.py`, `repositorio_archivos.py`, `tools.py`, los nodos e `index.html`.
- La subida a Drive vive en `ingest.py`, que no depende de cómo se autentique el admin. Pídanle que
  `admin.py` siga llamando a las mismas funciones de `ingest.py`.
- En `config.py` y `.env.example`, cada uno agrega su bloque con su propio encabezado de comentario
  para que los merges sean triviales.
- Hagan `git merge main` en su rama al empezar cada fase.

---

## 6. Antes de empezar (checklist)

- [x] Carpeta en Gmail personal → OAuth de usuario.
- [x] El `.env` no tiene credenciales de Drive → se crean en la fase 2.
- [x] Chroma: contenedor servidor en Compose (requisito: escalable para RAG).
- [ ] Pedirle al compañero que agregue **por correo** como Editor al usuario que generará el token (no cambiar el enlace público). **Solo bloquea la fase 3.**
- [ ] En Google Cloud: habilitar la Drive API, configurar la pantalla de consentimiento (externa, **In production**) y crear un OAuth client tipo "Desktop app". Descargar su JSON a `backend/secrets/oauth-client.json`. **Necesario para probar con Drive real en la fase 2.**
- [ ] Copia de seguridad de `backend/chroma_db` antes de la fase 1 (aunque el script no lo modifica).

---

## 7. Prompts para Claude Code

**Cómo usarlos:** igual que en la migración. Copien este documento a `docs/ARCHIVOS_AGENTE.md` sin
hacer commit (la fase 0 lo commitea), abran Claude Code en la raíz del repo y peguen **un prompt por
fase**. Entre fases pueden hacer `/clear`. Al terminar cada fase, corran ustedes la verificación.

### Fase 0: Preparación

```
Vamos a implementar la entrega de documentos del agente SIGEM, la conexión con Google Drive y Chroma como servicio aparte. La especificación está en docs/ARCHIVOS_AGENTE.md: léela entera, junto con CLAUDE.md, antes de hacer nada. En esta fase NO modifiques código de agente_sigem/.

1. Verifica que el árbol esté limpio salvo docs/ARCHIVOS_AGENTE.md (si hay otros cambios, detente y avísame). Cámbiate a la rama feature-archivosAgente, haz git pull y git merge main para traer lo último.
2. Actualiza CLAUDE.md:
   - Agrega docs/ARCHIVOS_AGENTE.md como especificación de esta etapa (manda sobre suposiciones; MIGRACION_AGENTICA.md sigue vigente para lo que no cambia).
   - Cambia las convenciones que ya no aplican: ahora SÍ se puede modificar frontend/index.html (solo para mostrar archivos; admin.html es de otro compañero, no lo toques); backend/chroma_db solo se LEE como origen de la migración, nunca se escribe; sigue prohibido reindexar y cambiar EMBEDDING_MODEL o COLLECTION_NAME.
   - Agrega la lista de fases 0 a 6 de la sección 1 de la especificación con casillas, y la regla de coordinación de la sección 5 (archivos que son del compañero).
3. cd backend && uv sync && uv run pytest -q. Reporta el resultado.
4. Verifica en el entorno instalado, sin suponer, y anótalo en CLAUDE.md en "Notas técnicas verificadas": (a) versión exacta de chromadb instalada y si langchain_chroma.Chroma acepta el parámetro client=; (b) que chromadb.HttpClient existe y su firma; (c) la configuración de distancia de la colección langchain en backend/chroma_db si existe localmente (no la modifiques); (d) la última versión de google-api-python-client, google-auth y google-auth-oauthlib compatibles con nuestra versión de Python.
5. Haz commit de esta fase (incluye docs/ARCHIVOS_AGENTE.md).

Al terminar dame: resultado de los tests, los hallazgos del punto 4 y cualquier contradicción entre la especificación y el código real.
```

### Fase 1: Chroma como servicio

```
Lee CLAUDE.md y la parte C (sección 4) de docs/ARCHIVOS_AGENTE.md. Fase 1: Chroma deja de ser embebido y pasa a ser un servicio. No toques nodos, tools ni prompts.

Antes de editar, preséntame un plan breve: cambios en retrieval.py y config.py, el servicio nuevo de Compose (imagen y versión exacta según lo verificado en la fase 0) y cómo vas a probar la migración. Espera mi confirmación.

Después:
1. config.py: CHROMA_MODO (local|http, por defecto local para no romper tests ni desarrollo), CHROMA_HOST, CHROMA_PORT, CHROMA_SSL. Actualiza .env.example en un bloque propio.
2. retrieval.py: crear_cliente_chroma() según CHROMA_MODO y obtener_vector_store() usando Chroma(client=...). Mantén el lru_cache. ingest.py no debería necesitar cambios: verifícalo.
3. /health: incluye el estado de Chroma (heartbeat) sin cambiar el código HTTP actual cuando todo está bien.
3b. Catálogo con varios procesos (subsección "Que el documento nuevo funcione con todas las tools" de la parte B): obtener_catalogo_documentos guarda el collection.count() con el que se construyó y se reconstruye si cambió. invalidar_cache_documentos() sigue existiendo. Test: un proceso indexa (simulado con otro cliente sobre la misma colección) y el catálogo de este proceso ve el documento nuevo sin invalidación explícita.
4. Compose.yml: servicio chroma (imagen fijada a la versión del cliente), volumen nombrado chroma-data en /data, sin puerto publicado salvo 127.0.0.1:8000 para migración y depuración, backend con depends_on y CHROMA_MODO=http, CHROMA_HOST=chroma. Quita el montaje de ./backend/chroma_db en el backend. Si el contenedor de Chroma no tiene curl para el healthcheck, busca una alternativa y explícamela.
5. backend/scripts/migrar_chroma.py exactamente como dice la especificación: lectura paginada con ids y embeddings, misma configuración de distancia, upsert idempotente, y verificación de conteo, ids y top-8 usando embeddings guardados como consulta (sin llamar a Gemini). Debe abrir chroma_db solo para leer.
6. Tests: crear_cliente_chroma con cada modo (monkeypatch, sin servidor real) y la lógica de paginación y verificación del script con dos clientes Chroma en memoria (EphemeralClient). uv run pytest -q.
7. Si tienes Docker y backend/chroma_db: docker compose up -d chroma, corre la migración y muéstrame la verificación. Si no, deja todo listo y dime los comandos exactos que debo correr yo.
8. Marca la fase en CLAUDE.md y haz commit.
```

### Fase 2: Repositorio de archivos y Google Drive

```
Lee CLAUDE.md y la parte B (sección 3) de docs/ARCHIVOS_AGENTE.md. Fase 2: módulo repositorio_archivos, tabla archivos_documento, endpoint de descarga y sincronización de los documentos ya indexados. Todavía NO toques el agente ni ingest.py.

Contexto: la carpeta de Drive es de la unidad personal de un compañero; la autenticación es SOLO OAuth de usuario (sin cuenta de servicio). Puede que el usuario del token todavía tenga solo permiso de lector: listar y descargar deben funcionar así; subir y retirar deben fallar con un mensaje claro ("el usuario de Drive no tiene permiso de Editor sobre la carpeta").

Antes de editar, preséntame un plan con: la interfaz RepositorioArchivos, cómo se carga y se refresca el token OAuth (y qué pasa si caduca o falta el archivo: la API debe arrancar igual y solo la descarga responde 503), qué llamadas de la Drive API vas a usar y con qué parámetros, y cómo vas a probarlo sin credenciales reales. Espera mi confirmación.

Después:
1. Dependencias: google-api-python-client, google-auth, google-auth-oauthlib.
2. config.py: las variables de la sección 3 en un bloque propio. GOOGLE_DRIVE_FOLDER_ID acepta el ID o la URL completa de la carpeta y extrae el ID. Actualiza .env.example. Agrega backend/secrets/ a .gitignore y .dockerignore.
3. agente_sigem/repositorio_archivos.py: RepositorioDrive (supportsAllDrives=True en todas las llamadas; retirar() mueve a la papelera, NUNCA borra de forma permanente) y RepositorioLocal; obtener_repositorio() según ARCHIVOS_BACKEND. Los errores de Drive se convierten en una excepción propia con un mensaje claro (403 por permisos de lector, token inválido o caducado, archivo no encontrado).
4. db.py: modelo ArchivoDocumento (tabla archivos_documento) creado por init_db, y funciones registrar_archivo, obtener_archivo y eliminar_registro_archivo.
5. backend/scripts/autorizar_drive.py: genera el token OAuth una sola vez con InstalledAppFlow y alcance drive. En el docstring, explica que la app debe estar "In production" en la pantalla de consentimiento o el token caduca en 7 días.
6. backend/scripts/sincronizar_drive.py: lo que dice la especificación, en modo simulación por defecto y con --aplicar y --subir-faltantes DIR.
7. api.py: GET /documentos/{archivo}/descargar. Busca en la tabla (si no está, 404 con un mensaje amable), hace streaming con Content-Disposition attachment y el nombre original, o redirect si DRIVE_MODO_ENTREGA=redirect. Valida que el archivo exista en el catálogo; nunca construyas rutas de disco con el parámetro.
8. Compose.yml: monta backend/secrets en /app/secrets de solo lectura.
9. Tests sin red: RepositorioLocal completo; RepositorioDrive con el cliente de la API falso (monkeypatch de build); el endpoint con TestClient (200, 404 y redirect); la sincronización con un listado falso de Drive y un catálogo falso (enlazados, faltantes y huérfanos). uv run pytest -q.
10. Si hay credenciales en backend/secrets y la carpeta en el .env: corre sincronizar_drive.py en modo simulación y muéstrame el reporte (la carpeta debería tener los 71 PDFs del catálogo). NO uses --aplicar sin preguntarme. Si no hay credenciales, dime los pasos exactos para crearlas y generar el token.
11. Marca la fase en CLAUDE.md y haz commit.
```

### Fase 3: Ingesta conectada a Drive

```
Lee CLAUDE.md y la subsección "Ingesta" de la parte B de docs/ARCHIVOS_AGENTE.md. Requisito previo: el usuario del token OAuth ya debe ser Editor de la carpeta (los tests no lo necesitan; la prueba real sí). Fase 3: al indexar un documento nuevo, su original se sube al repositorio de archivos. Solo puedes tocar ingest.py, sus tests y lo mínimo necesario en db.py. admin.py y admin.html son de mi compañero: si necesitas cambiarlos, detente y dime qué cambio harías.

1. En _indexar_documento sigue el orden de la especificación: subir → embeddings y upsert en Chroma → registrar la fila. Si falla el upsert, retira el archivo del repositorio. Si falla la subida, el documento no se indexa y el resultado del lote lo dice con un mensaje entendible para el admin.
2. Guarda id_externo por archivo en el manifiesto del lote.
3. revertir_indexacion retira el archivo del repositorio (papelera) y borra la fila de archivos_documento, además de lo que ya hace en Chroma.
4. Los documentos omitidos (mismo hash o mismo título) no se suben.
4b. Si el título final no se puede leer como tipo + número (catalogo.extraer_referencia devuelve None), el resultado de ese archivo en el lote incluye una advertencia: "No se reconoce tipo y número en el título; el documento se encontrará por contenido, pero no por referencia como 'Decreto 045 de 2026'". No bloquea la indexación.
5. Tests en test_ingest.py con RepositorioLocal o uno falso: indexación correcta crea la fila; fallo de Chroma retira el archivo; fallo de subida no indexa; revertir limpia los tres lugares; advertencia de título no canónico.
5b. Test de extremo a extremo (tests/test_documento_nuevo.py) con Chroma en memoria, embeddings falsos y RepositorioLocal: indexa un PDF de prueba con título "Decreto 045 de 2026" y verifica, SIN reiniciar nada, que lo encuentran buscar_documento_por_referencia (tipo, número y año), listar_documentos (filtro por año 2026), buscar_fragmentos (por una frase de su contenido), leer_documento (sus páginas en orden) y preparar_descarga (devuelve url_descarga). Luego revierte el lote y verifica que ninguna de las cinco lo encuentra. uv run pytest -q.
6. Marca la fase en CLAUDE.md y haz commit.
```

### Fase 4: El agente entrega documentos

```
Lee CLAUDE.md y la parte A (sección 2) de docs/ARCHIVOS_AGENTE.md con mucho cuidado. Fase 4: el agente detecta cuándo el usuario pide el archivo y lo entrega. El contrato de la API solo puede cambiar de forma aditiva (campo archivos con valor por defecto []).

Antes de editar, preséntame un plan con: los cambios en estado.py, la firma y el docstring de preparar_descarga, cómo vas a armar la lista de tools dinámica en el investigador, los cambios en el evaluador y el redactor, y la lista de ejemplos few-shot que vas a agregar al analista (con documentos reales de backend/catalogo_chroma/catalogo_chroma.csv). Espera mi confirmación.

Después:
1. estado.py: solicita_archivo en AnalisisConsulta (con description clara en Field) y archivos en EstadoAgente, sin reducer.
2. prompts.py: reglas y ejemplos en pares contrastados para el analista (contenido vs archivo, documento activo "pásamelo", pedido mixto); instrucciones del investigador para cuando se pide un archivo; modo "entrega" del redactor; y la regla de que el redactor NUNCA escribe URLs.
3. tools.py: preparar_descarga según la especificación (valida contra el catálogo y contra archivos_documento, JSON, sin excepciones). NO la agregues a HERRAMIENTAS_INVESTIGADOR: va en una lista aparte.
4. nodos/investigador.py: tools dinámicas según solicita_archivo; acumular_resultados extrae archivos sin duplicados (máximo 3).
5. nodos/evaluador.py: atajo archivos sin evidencia → redactor. nodos/redactor.py: modo entrega y actualización de documento_activo con el archivo entregado.
6. api.py: modelo ArchivoDescargable {titulo, archivo, url_descarga}; campo archivos en ChatResponse y ChatVozResponse; "archivos": [] en los campos que se reinician por turno en ejecutar_turno; descarta archivos si analisis.solicita_archivo es false. url_descarga es relativa (/documentos/{archivo}/descargar) para que el frontend la complete con su API_URL.
7. Tests con LLM falsos (test_nodos.py, test_grafo.py, test_tools.py, test_api.py):
   - "pásame el Decreto 001 de 2019" → archivos con un elemento, sin evidencia, modo entrega.
   - "¿qué dice el Decreto 001 de 2019?" → archivos vacío y preparar_descarga no está entre las tools del subagente.
   - dos turnos: pregunta sobre un decreto y luego "pásamelo" → entrega el documento activo.
   - referencia ambigua con solicita_archivo → aclaración con candidatos, sin archivos.
   - archivo indexado pero sin fila en archivos_documento → el redactor explica que no está disponible para descarga.
   - compuerta: estado con archivos pero solicita_archivo false → la API devuelve archivos [].
   - dos turnos seguidos: archivos no se acumula entre turnos.
8. Agrega a backend/evaluacion/dataset.jsonl unos 10 casos de la categoría "solicitud_archivo" (positivos, negativos que se parecen y uno mixto) con un campo esperado solicita_archivo, y haz que evaluar_agente.py reporte la exactitud de ese campo.
9. uv run pytest -q, marca la fase en CLAUDE.md y haz commit.
```

### Fase 5: Botón de descarga en el chat

```
Lee CLAUDE.md. Fase 5: mostrar en frontend/index.html los archivos que devuelve la API. Solo puedes tocar index.html (admin.html es de mi compañero). Respeta el estilo gov.co que ya tiene la página.

1. Después de appendSources, si data.archivos trae elementos, agrega a la burbuja del agente un botón o enlace por archivo con el título del documento, un ícono de descarga y el atributo download. Completa url_descarga con la API_URL que ya usa la página. Funciona igual en /chat y /chat-voz.
2. Accesibilidad para el perfil P1 (sin experiencia digital): texto del botón explícito ("Descargar Decreto 001 de 2019 (PDF)"), aria-label, tamaño táctil de al menos 44 px y foco visible.
3. Si la descarga devuelve 404, muestra un mensaje amable en el chat en vez de una página de error.
4. Si en la fase 2 se persistieron los archivos en el historial, píntalos también al recargar el hilo; si no, no hagas nada con el historial.
5. Explícame cómo probarlo a mano (qué preguntar y qué debería verse) y haz commit. 
```

### Fase 6: Evaluación, documentación y PR

```
Lee CLAUDE.md y docs/ARCHIVOS_AGENTE.md. Fase 6: dejar todo evaluado, documentado y listo para el PR a main.

1. Si hay GOOGLE_API_KEY y Chroma con datos: corre evaluar_recuperacion.py contra el Chroma en servidor y compáralo con evaluacion/linea_base_recuperacion.json (deben coincidir). Corre evaluar_agente.py solo con los casos de solicitud_archivo y con los de categorías que hayan podido cambiar (usa --pausa para cuidar la cuota). Muéstrame una tabla y señala regresiones, pero NO las corrijas sin preguntarme.
2. Documentación: actualiza backend/README.md, el README raíz y .env.example. Redacta en un archivo docs/home_actualizacion.md las secciones nuevas o cambiadas para la wiki (Docker con el servicio chroma, la sección de Chroma ya no embebida, Drive y el endpoint de descarga, el campo solicita_archivo y la tool preparar_descarga, y quitar de "Known pending items" el punto de Chroma embebido). Incluye cómo generar las credenciales de Drive paso a paso.
3. Agrega a docs/ARCHIVOS_AGENTE.md una sección final "Estado de implementación" con las desviaciones respecto a la especificación y su razón.
4. Revisión final del diff contra main: secretos, archivos de credenciales, prints de depuración, imports muertos, TODOs. Confirma que backend/secrets y chroma_db no están en Git.
5. git merge main por si mi compañero ya mergeó el login; resuelve conflictos SOLO en mis archivos y avísame de los suyos. uv run pytest -q, marca la fase y haz commit. Dime qué falta para abrir el PR.
```

---

## 8. Riesgos y cómo se cubren

| Riesgo | Mitigación |
|---|---|
| El analista marca `solicita_archivo` de más (falsos positivos) | Ejemplos negativos parecidos en el few-shot + métrica propia en la evaluación. Un falso positivo solo agrega un botón; no quita la respuesta |
| Cuota de Gemini al evaluar | Las verificaciones de la migración no usan Gemini; la evaluación del agente se limita a las categorías afectadas |
| Token OAuth caduca a los 7 días | Pantalla de consentimiento en "In production"; si caduca, se regenera con `autorizar_drive.py` y la API sigue respondiendo el chat (solo la descarga da 503) |
| El permiso de Editor no llega a tiempo | Las fases 2, 4 y 5 funcionan con permiso de lector; la 3 se deja para el final |
| La carpeta depende de la cuenta personal de un compañero | Para el MVP basta; a futuro, mover la carpeta a una cuenta del proyecto |
| Versión del servidor Chroma distinta del cliente | Imagen fijada a la versión verificada en la fase 0 |
| PDF en Chroma pero no en Drive | El reporte de sincronización los lista; `preparar_descarga` responde "no disponible" y el redactor lo dice con amabilidad |
| Conflictos con el login del compañero | Dueños por archivo (sección 5) y `git merge main` al inicio de cada fase |


---

## 9. Estado de implementación (octubre de 2026)

Las seis fases se implementaron en la rama `feature-archivosAgente` (un commit por fase, orden real 0 → 1 → 2 → 4 → 5 → 3 → 6). El PR va hacia `feature-Sprint7`, no hacia `main`.
Esta sección lista lo que **se desvía** de la especificación y por qué.

### Desviaciones

| # | Especificación | Implementación | Razón |
|---|---|---|---|
| 1 | `AnalisisConsulta` solo gana `solicita_archivo` | Además `incluye_contenido` | Para distinguir "solo el archivo" de "archivo y explicación". La tool que identifica el documento siempre devuelve fragmentos, así que sin esa señal "pásame el decreto" terminaba en un resumen. Regla de producto: *dame / necesito / quiero / pásame* piden **archivo y contenido**; solo *descargar / PDF / archivo / enlace* piden solo el archivo. |
| 2 | Campo de estado `archivos` | Además `archivos_no_disponibles` | Para explicar un documento que existe pero no tiene original (modo `descarga_no_disponible`), sin usar el texto del error de la tool. |
| 3 | Atajo del evaluador: archivos sin evidencia → redactor | Los `candidatos` ganan a los `archivos` (en el evaluador y en `elegir_modo`) | Si hay varios documentos posibles no se entrega ninguno que el modelo haya elegido por su cuenta: primero se pide aclaración. |
| 4 | Modo `entrega` del redactor | Además `descarga_no_disponible`; en modo `entrega` no hay salida estructurada ni fuentes | El modelo veía la regla de `informacion_encontrada` y la copiaba al texto de la respuesta; ahora esa regla vive solo en el modo `respuesta`. |
| 5 | `/health` agrega `heartbeat()` | `503` con `"chroma":"no disponible"` si el servidor no responde (solo en modo `http`) | Decisión del equipo: que el healthcheck del contenedor refleje que el RAG está caído. |
| 6 | Imagen `chromadb/chroma` "con la misma versión" | `chromadb/chroma:1.5.9` y `chromadb==1.5.9` fijado en `pyproject.toml` | Verificado en la fase 0. La imagen no trae `curl` ni Python: el healthcheck usa `/dev/tcp` de bash. |
| 7 | Verificación de la migración: top-8 igual en origen y destino | Se compara cada lado con el top-8 **exacto** (fuerza bruta) y el destino no puede recuperar peor que el origen | HNSW es aproximado y se construye distinto en cada instalación: en 1 de 10 muestras el top-8 difería y el destino era el exacto. El script trabaja sobre una copia de `chroma_db` (el original no se toca). |
| 8 | `RepositorioArchivos`: `subir`, `descargar`, `retirar`, `listar` | Además `enlace_ver` y `proveedor`; `descargar` pide el primer trozo al llamarse | `enlace_ver` hace posible `DRIVE_MODO_ENTREGA=redirect`; así un 403/404 sale como `503`/`404` antes de empezar a responder. |
| 9 | Servicio de Drive reutilizable | Un `build` por operación | El cliente HTTP de googleapiclient no es seguro entre hilos y la descarga corre en el pool de hilos. |
| 10 | OAuth "In production" | Probado con la app en modo *Pruebas* (el token caduca a los 7 días) | Publicar exige URL de página principal y de política de privacidad. El código y la documentación sirven igual para ambos modos; la renovación es `autorizar_drive.py`. |
| 11 | Endpoint de descarga | `503` también si el proveedor registrado en la fila no coincide con el repositorio activo; sin `Content-Length` | Evita buscar un archivo de Drive en el disco local (o al revés) y que un tamaño guardado erróneo corte la descarga. |
| 12 | Ingesta: subir → Chroma → fila | Si falla el registro de la fila también se deshace Chroma y se retira el original; el resultado lleva `id_externo`, `proveedor_original` y, en su caso, `advertencias`; revertir es reintentable (`original_retirado` en el manifiesto) | Un documento que el agente encuentra pero no puede entregar es peor que un error visible; un 403 a medias no debe dejar el lote bloqueado. |
| 13 | Persistencia opcional de `archivos` en `turnos` | **No se implementó** | Era opcional. Al recargar un hilo se ve el texto "Aquí tienes…" pero no el botón. |
| 14 | `<a href download>` en `index.html` | El clic hace `fetch` y descarga el blob | Un enlace directo mostraría una página de error JSON ante un 404; además el atributo `download` se ignora entre orígenes distintos (frontend y API en puertos distintos). |
| 15 | El catálogo se reconstruye si cambió `collection.count()` | Con candado; si `count()` falla se conserva el catálogo anterior | No tumbar las tools porque Chroma tarde un instante. `invalidar_cache_documentos()` sigue existiendo. |
| 16 | "≈72 PDF" | El catálogo tiene 71; la carpeta de Drive tiene 69 | `2401_decretono187.pdf` y `2403_decretono184.pdf` no están en Drive: responden "no disponible para descarga" hasta que se suban con `sincronizar_drive.py --subir-faltantes`. |

### Acceso a la carpeta de Drive
La configuración definitiva es: la carpeta compartida **por correo** con la cuenta del backend como **Editor**, y el acceso general del enlace en **Lector** o restringido. Con una carpeta
en una unidad personal, el token de un Editor que no es dueño **no puede mandar a la papelera los archivos del dueño** (`canTrash: false`); `retirar` solo actúa sobre lo que sube la
propia cuenta del backend, que es justo lo que revierte la ingesta. Los archivos que suba el backend ocupan el espacio de esa cuenta.

### Resultados de la evaluación (fase 6)
- **Tests:** 317, sin llamadas a Gemini, a Chroma servidor ni a Drive.
- **Recuperación contra el servidor Chroma** (33 casos originales, con reranker): hit@8 **0,79** / MRR **0,67**, idéntico caso por caso a la medición de la fase 1. Frente a la línea base previa a la migración
  (0,88 / 0,77) las diferencias son `ref-04`, `amb-02`, `glob-04` y `glob-02` (casos de referencia que ahora resuelven el analista y las tools, y el reranker), y `tema-04` mejora. Los 10 casos nuevos
  de `solicitud_archivo` dan 0,90 (el que falla, `arch-09` "pásamelo", necesita al analista y al documento activo). Archivos: `evaluacion/fase6b_servidor_con_reranker.json`.
- **Agente completo** (`evaluar_agente.py --pausa 20`, 54 casos, 14 categorías, los cuatro modelos en `gemini-3.5-flash-lite`; `evaluacion/resultados_fase6_completo.json`):
  intención 1,00, archivos 1,00, candidatos 1,00, alerta 1,00, negativa 1,00, `solicita_archivo` 1,00, `incluye_contenido` 1,00, entrega 1,00, `info` 0,98 (ver abajo); 0 errores y 0 degradados;
  mediana 15 s por caso (incluye las pausas de 20 s entre turnos de los casos de varios turnos).

### Hallazgo abierto de la evaluación
La instrucción que se añadió al modo `respuesta` del redactor ("si la entrada indica ARCHIVO(S) PARA DESCARGAR, cierra diciendo que el documento queda disponible en el botón…") la aplica
el modelo flash-lite en **casi todas** las respuestas con contenido: 24 de las 35 respuestas que mencionan el botón lo hacen sin que haya archivo que entregar (p. ej. `ref-01`, `tema-02`, `seg-02`).
El usuario leería "queda disponible en el botón de descarga" sin que aparezca ningún botón. La métrica `info` (0,98) marcó `glob-01` por un falso positivo de la propia métrica ("hum**ano se encuentra**"
contiene "no se encuentra"), pero el defecto de fondo es real. Corrección propuesta (pendiente de decisión): dejar de pedirlo en el prompt fijo y añadir esa línea desde el código solo cuando el estado
trae `archivos`, con un test, y añadir a `evaluar_agente.py` una métrica que detecte la mención al botón sin archivo entregado.
