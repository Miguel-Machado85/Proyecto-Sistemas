# SIGEM — Agente de respuesta ciudadana (Alcaldía de Marinilla)

Asistente conversacional (texto y voz) que responde preguntas sobre normas del repositorio SIGEM —decretos,
resoluciones, leyes y acuerdos municipales— citando la fuente y adaptando el lenguaje al perfil del usuario
(P0 neutro, P1 sin experiencia digital, P2 ciudadano general, P3 funcionario/técnico). Cuando el usuario lo pide, también **entrega el
documento original** como botón de descarga (en texto y en voz).

## Componentes

| Carpeta | Qué es |
|---|---|
| `backend/` | API FastAPI con un **sistema multi-agente** (LangGraph + Gemini), recuperación sobre Chroma y reranker. Ver [`backend/README.md`](backend/README.md). |
| `frontend/` | Cliente web en HTML plano (servido con nginx): `index.html` (chat) y `admin.html` (carga administrativa). |
| `Compose.yml` | Orquesta `backend`, `chroma` (servidor del índice vectorial), `db` (PostgreSQL) y `frontend`. |
| `docs/MIGRACION_AGENTICA.md` | Especificación de la migración al sistema multi-agente y su estado de implementación. |
| `docs/ARCHIVOS_AGENTE.md` | Especificación de la entrega de documentos, la conexión con Google Drive y Chroma como servicio, con su estado de implementación. |

## Arquitectura

```
Usuario ──► frontend ──► backend (FastAPI)
                            │
                            ▼
   analista ─► respuesta_directa (saludo, fuera de dominio, aclaración, audio no comprensible)
      │
      └─► investigador ──► evaluador ──► redactor ──► respuesta con citas y fuentes
           (tools sobre      (¿basta la       (estilo según
            catálogo y        evidencia?       el perfil)
            Chroma)           si no, reintenta)
```

- **Analista** (Gemini flash-lite): entiende la pregunta, corrige ortografía, convierte "decreto doce de dos mil
  diecinueve" en `decreto 12 / 2019` y decide la ruta. No usa listas de palabras ni regex sobre lo que escribe el usuario.
- **Investigador** (Gemini flash, agente ReAct): elige entre buscar un documento por referencia o código, listar
  documentos, buscar fragmentos o leer un documento, y reúne la evidencia.
- **Evaluador** (flash-lite): si la evidencia no basta, reformula la consulta y reintenta una vez.
- **Redactor** (flash): responde solo con la evidencia, con citas `[Decreto 001 de 2019, p. 3]`.
- **Memoria:** el hilo y el documento activo viven en el estado del grafo, persistido en PostgreSQL.
- **Reranker:** `sentence-transformers` (CrossEncoder multilingüe, CPU) reordena los fragmentos recuperados.
- **Entrega de documentos:** el analista marca si el usuario pide el **archivo** (`solicita_archivo`) y solo entonces el investigador recibe la tool
  `preparar_descarga`; la URL de descarga la construye el backend, nunca el modelo. "Dame / necesito el decreto 83" entrega el archivo **y** lo explica;
  "descárgame / el PDF" entrega solo el archivo.
- **Chroma como servicio:** el índice vive en su propio contenedor (`chromadb/chroma:1.5.9`), no embebido en la API.
- **Originales en Google Drive:** los PDF se sirven por `GET /documentos/{archivo}/descargar` (el backend hace de intermediario; la carpeta puede seguir privada).
  La ingesta administrativa sube cada documento nuevo a Drive y lo retira (papelera) al revertir.

## Puesta en marcha

Requisitos: Docker, una `GOOGLE_API_KEY` de Gemini, el índice `backend/chroma_db/` (solo como origen de la migración inicial) y, para
entregar documentos, las credenciales de Google Drive (ver [`backend/README.md`](backend/README.md#originales-y-google-drive)).

```bash
cp backend/.env.example backend/.env      # completa GOOGLE_API_KEY (y AWS_* / LANGFUSE_* si los usas)
docker compose -f Compose.yml up -d --build
```

**Solo la primera vez** (el volumen `chroma-data` nace vacío): copia el índice existente al servidor Chroma, sin regenerar embeddings, y enlaza los PDF de Drive:

```bash
cd backend
CHROMA_MODO=http uv run python scripts/migrar_chroma.py                 # chroma_db -> servidor, con verificación
uv run python scripts/autorizar_drive.py                                # token OAuth de Drive (una vez; ver backend/README.md)
DATABASE_URL=postgresql://sigem:sigem@localhost:55432/sigem CHROMA_MODO=http uv run python scripts/sincronizar_drive.py --aplicar
```
Sin credenciales de Drive (`ARCHIVOS_BACKEND=local`, el valor por defecto) todo funciona salvo la descarga de originales.

**Acceso a la carpeta de Drive:** compártela **por correo** con la cuenta que autorizó el token, con rol **Editor** (para que la ingesta pueda subir),
y deja el acceso general del enlace en **Lector** o restringido; nunca "cualquiera con el enlace puede editar".

| Servicio | URL |
|---|---|
| Frontend | http://localhost:8080 |
| API y documentación Swagger | http://localhost:8001 · http://localhost:8001/docs |
| PostgreSQL | `localhost:55432` (usuario/clave/base: `sigem`) |
| Chroma (solo para migración y depuración) | `127.0.0.1:8000` |

La primera vez, el backend descarga el modelo del reranker en el volumen `hf-cache` (unos 30 s a un par de minutos);
después se reutiliza.

## Desarrollo y pruebas

```bash
cd backend
uv sync
uv run pytest -q                              # pruebas con LLM falsos: no gastan cuota
uv run python evaluacion/evaluar_recuperacion.py   # recuperación (solo embeddings)
uv run python evaluacion/evaluar_agente.py --estimar   # evaluación completa del agente (consume cuota)
```

Pruebas del frontend (formato de las respuestas; Node 20+, sin dependencias), desde la raíz del repositorio:

```bash
node --test "frontend/tests/*.test.js"
```

## Notas operativas
- **Cuota de Gemini:** una consulta con búsqueda gasta de 3 a 6 llamadas. El plan gratuito de `gemini-3.5-flash`
  permite ~20 solicitudes por día; para uso real hace falta facturación (o usar flash-lite en investigador y redactor).
- **No reindexar** ni cambiar `EMBEDDING_MODEL` ni `COLLECTION_NAME`: las consultas deben usar el mismo modelo con que se generó el índice.
  La migración al servidor Chroma copia los vectores, no los recalcula, y `backend/chroma_db/` solo se lee.
- **Secretos:** `backend/.env` y `backend/secrets/` (credenciales OAuth de Drive) están fuera de Git y de la imagen Docker.
- **Token de Drive:** si la app OAuth está en modo *Pruebas*, el token caduca a los 7 días (repite `autorizar_drive.py`); mientras tanto el chat funciona y solo la descarga responde `503`.
- Un turno con búsqueda puede tardar entre 10 y 60 s (varias llamadas encadenadas y el reranker en CPU).
