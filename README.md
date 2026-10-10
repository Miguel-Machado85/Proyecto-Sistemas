# SIGEM — Agente de respuesta ciudadana (Alcaldía de Marinilla)

Asistente conversacional (texto y voz) que responde preguntas sobre normas del repositorio SIGEM —decretos,
resoluciones, leyes y acuerdos municipales— citando la fuente y adaptando el lenguaje al perfil del usuario
(P0 neutro, P1 sin experiencia digital, P2 ciudadano general, P3 funcionario/técnico).

## Componentes

| Carpeta | Qué es |
|---|---|
| `backend/` | API FastAPI con un **sistema multi-agente** (LangGraph + Gemini), recuperación sobre Chroma y reranker. Ver [`backend/README.md`](backend/README.md). |
| `frontend/` | Cliente web en HTML plano (servido con nginx). |
| `Compose.yml` | Orquesta `backend`, `db` (PostgreSQL) y `frontend`. |
| `docs/MIGRACION_AGENTICA.md` | Especificación de la migración al sistema multi-agente y su estado de implementación. |

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

## Puesta en marcha

Requisitos: Docker, una `GOOGLE_API_KEY` de Gemini y el índice `backend/chroma_db/` ya construido.

```bash
cp backend/.env.example backend/.env      # completa GOOGLE_API_KEY (y AWS_* / LANGFUSE_* si los usas)
docker compose -f Compose.yml up --build
```

| Servicio | URL |
|---|---|
| Frontend | http://localhost:8080 |
| API y documentación Swagger | http://localhost:8001 · http://localhost:8001/docs |
| PostgreSQL | `localhost:55432` (usuario/clave/base: `sigem`) |

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

## Notas operativas
- **Cuota de Gemini:** una consulta con búsqueda gasta de 3 a 6 llamadas. El plan gratuito de `gemini-3.5-flash`
  permite ~20 solicitudes por día; para uso real hace falta facturación (o usar flash-lite en investigador y redactor).
- **No reindexar** `backend/chroma_db/` ni cambiar `EMBEDDING_MODEL`: las consultas deben usar el mismo modelo con
  que se generó el índice.
- Un turno con búsqueda puede tardar entre 10 y 60 s (varias llamadas encadenadas y el reranker en CPU).
