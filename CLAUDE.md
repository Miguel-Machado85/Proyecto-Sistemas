# CLAUDE.md — Proyecto SIGEM (Alcaldía de Marinilla)

## Resumen del proyecto
1. Agente ciudadano de la Alcaldía de Marinilla que responde sobre normas del repositorio SIGEM.
2. El repositorio son ≈72 PDFs (decretos, resoluciones, leyes, acuerdos) indexados en Chroma.
3. Backend: FastAPI + LangGraph en `backend/agente_sigem/`; frontend HTML plano en `frontend/`.
4. Recuperación híbrida (texto literal + semántica) sobre Chroma con embeddings de Gemini.
5. La respuesta se adapta al perfil del usuario (P0–P3) con prompts de estilo distintos.
6. Entrada por texto y por voz (STT/TTS con Gemini; audios en S3).
7. Historial de turnos en Postgres; trazas en Langfuse.
8. Hoy el LLM es Ollama (`llama3`) y las decisiones están en regex/listas fuera del grafo.
9. Se migra a un sistema multi-agente (analista → investigador → evaluador → redactor) con Gemini.
10. Rama de trabajo: `feature-agentico` (base `feature-Sprint6`); un commit por fase.

## Fuente de verdad
La especificación está en **`docs/MIGRACION_AGENTICA.md`** y **manda sobre cualquier otra suposición**
(incluido lo que diga un prompt de fase). Léela entera antes de empezar una fase.

## Convenciones (sección 14 de la especificación)
- Identificadores, comentarios, prompts y docstrings en **español**, como el código existente.
- **No tocar `chroma_db/`**, no reindexar, no cambiar `EMBEDDING_MODEL`.
- **No modificar `frontend/`** (el contrato de la API no cambia).
- Un commit por fase, en la rama `feature-agentico`, con un mensaje que describa la fase.
- Los tests no llaman a Gemini ni a Chroma reales (fakes / monkeypatch de `crear_llm`).

## Comandos de verificación (desde `backend/`)
```bash
cd backend && uv run pytest -q
cd backend && uv run python -c "import agente_sigem.api"
```
Línea base de recuperación (necesita `GOOGLE_API_KEY` y `backend/chroma_db`):
```bash
cd backend && uv run python evaluacion/evaluar_recuperacion.py
```

## Fases
- [x] **Fase 0** — Preparación: rama, CLAUDE.md, dependencias, verificación del entorno, dataset y línea base de recuperación
- [x] **Fase 1** — Configuración y LLM: `config.py`, `llm.py`, variables Gemini, `.env.example`, retirar Ollama
- [x] **Fase 2** — Recuperación: `retrieval.py` refactor, `reranker.py`, `catalogo.py` (`buscar_por_atributos`, `similares`)
- [x] **Fase 3** — Tools: `tools.py` (4 tools, sin conectar al grafo)
- [ ] **Fase 3b** — Estado y prompts: `estado.py`, `prompts.py` (pendiente; necesarios para la Fase 4)
- [ ] **Fase 4** — Nodos y grafo: `nodos/*`, `agent.py`, `checkpointer.py`
- [ ] **Fase 5** — API y persistencia: `api.py` simplificado + lifespan, `db.py`, eliminar `contexto.py`/`calidad.py`/`test_contexto.py`
- [ ] **Fase 6** — Evaluación final y despliegue: `evaluar_agente.py`, Dockerfile/Compose (torch CPU, `hf-cache`), README

> Las fases 1–6 son la agrupación de los entregables de la sección 11 de la especificación;
> la especificación no numera fases, así que esta división es la que usamos como guía.
> Ajustarla aquí si cambia.

## Notas técnicas verificadas
Verificado en el entorno instalado (`uv sync`, Python 3.13) el 2026-10-10:

**Versiones:** langchain 1.4.2 · langgraph 1.2.11 · langgraph-prebuilt 1.1.0 · langchain-core 1.6.5 ·
langchain-google-genai 4.3.7 · google-genai 2.21.0 · langchain-chroma 1.1.0 · pytest 9.1.1.
**No instalados todavía:** `langgraph-checkpoint-postgres`, `psycopg-pool`, `sentence-transformers`, `torch`, `rapidfuzz`
(`psycopg[binary]` 3.3.6 sí está).

- **`langchain.agents.create_agent` existe** → usar este para el investigador (no hace falta el respaldo).
  Firma: `create_agent(model: str | BaseChatModel, tools=None, *, system_prompt=None, middleware=(), response_format=None,
  state_schema=None, context_schema=None, checkpointer=None, store=None, interrupt_before=None, interrupt_after=None,
  debug=False, name=None, cache=None, transformers=None)`. El prompt va en `system_prompt` (no `prompt`).
  Devuelve un `CompiledStateGraph` con estado `messages`.
- **`langgraph.prebuilt.create_react_agent` también existe** (parámetro `prompt`, `version='v2'`, hooks `pre/post_model_hook`).
  Está disponible como respaldo, pero el camino principal es `create_agent`.
- **`langgraph.prebuilt.ToolNode` funciona, pero solo dentro de un grafo**: invocarlo suelto lanza
  `ValueError: Missing required config key ... for 'tools'` porque necesita el runtime/config. Dentro de un `StateGraph`
  devuelve el `ToolMessage` correcto. Para tests unitarios de las tools, llamar a la tool directamente (`tool.invoke({...})`)
  o probar el grafo completo.
- **`ChatGoogleGenerativeAI.with_structured_output(schema, method='json_schema', include_raw=False)` funciona** con la versión
  instalada: probado en vivo con `gemini-3.5-flash-lite` y un modelo Pydantic con `Literal`; devolvió la instancia Pydantic.
- **Modelos disponibles para la API key** (`genai.Client().models.list()`): `gemini-3.5-flash`, `gemini-3.5-flash-lite`,
  `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`, `gemini-3.1-flash-lite`, `gemini-2.5-flash`,
  `gemini-2.5-flash-lite`, `gemini-2.5-pro`, `gemini-3.1-pro-preview`. Los valores por defecto actuales del repo
  (`VERIFICACION_MODEL=gemini-3.5-flash-lite`) existen; para flash usar `gemini-3.5-flash` (verificar de nuevo en la Fase 1).
- **Pytest necesitaba configuración**: sin `pythonpath`, `import agente_sigem` fallaba en la recolección. Se agregó
  `[tool.pytest.ini_options] pythonpath = ["."]` y `testpaths = ["tests"]` en `backend/pyproject.toml`.
  Los scripts fuera de `tests/` deben ejecutarse desde `backend/` (las rutas de `config.py` son relativas: `./chroma_db`).
- **`AIMessage.content` de Gemini 3.x es una lista de bloques** (`[{'type':'text','text':...}]`), no un `str`. Usar siempre
  `mensaje.text` (propiedad de langchain-core 1.x, concatena los bloques de texto) para obtener el texto; nunca `.content`.
  `api.py` ya lo hace. Los `ToolMessage` de nuestras tools sí son `str` (JSON).
- **Modelos por defecto (Fase 1, verificados con `models.list()`):** flash-lite = `gemini-3.5-flash-lite` (analista, evaluador),
  flash = `gemini-3.5-flash` (investigador, redactor). `/health/llm` usa `client.models.get(INVESTIGADOR_MODEL)`.
  Prueba de humo con `gemini-3.5-flash` + `bind_tools` correcta (llamó a `buscar_en_sigem` y respondió con citas del decreto).
- **Consola Windows:** imprimir texto con tildes/`​` desde Python puede fallar por cp1252; usar `PYTHONIOENCODING=utf-8`.

### Fase 2 — recuperación y reranker
- **torch CPU confirmado:** `torch 2.14.1+cpu` (`torch.version.cuda is None`), `sentence-transformers 6.1.0`, `rapidfuzz 3.14.6`.
  Hay wheel para Python 3.13 en el índice `pytorch-cpu`; Dockerfile (3.14) aún por revisar en la Fase 6.
- **Reglas de recuperación:** `recuperar_sigem(consulta, archivo=None)` = texto + semántica → unión → reranker → `diversificar`
  (sin tope por documento si hay `archivo`) → top 8. Si el reranker no aplica (`reranker.reordenar_si_aplica` devuelve `None`),
  se usa `combinar_y_diversificar` como respaldo. `recuperar_sigem` **ya no resuelve referencias**.
- **Referencias:** viven en `agente_sigem/referencias.py` (`buscar_por_referencia`, `candidatos_referencia_ambigua`,
  `respuesta_referencia_ambigua`, `fragmentos_de_archivo`, `recuperar_con_referencia`). Siguen usando `resolver()` (regex);
  `buscar_en_sigem` llama a `recuperar_con_referencia` hasta que la Fase 3 las reemplace por las tools con `buscar_por_atributos`.
- `evaluar_recuperacion.py` tiene `--sin-reranker` y `--con-referencia` (esta última es comparable con la línea base).
- **Resultados (33 casos, hit@8 / MRR / ms por consulta):**
  | Configuración | hit@8 | MRR | ms/consulta |
  |---|---|---|---|
  | Línea base (código viejo) | 0.88 | 0.77 | — |
  | Nuevo, sin reranker, con referencia | 0.91 | 0.79 | 327 |
  | Nuevo, con reranker, con referencia | 0.88 | 0.79 | 1706 |
  | Nuevo, con reranker, `recuperar_sigem` puro | 0.82 | 0.68 | 2308 |
  El reranker mmarco **mejora** `tema_sin_palabras_clave` (MRR 0.90→1.00) e inventario, pero **empeora** voz con números en
  palabras (MRR 0.33→0.06), `mal_escrita` y `glob-02`. En conjunto no supera al respaldo; revisar tras el analista (Fase 4),
  que reescribe la consulta, antes de decidir si se deja activo o se prueba `BAAI/bge-reranker-v2-m3`.
  Archivos: `evaluacion/fase2_*.json`.

### Fase 3 — tools (`agente_sigem/tools.py`, aún sin conectar al grafo)
- `HERRAMIENTAS_INVESTIGADOR = [buscar_documento_por_referencia, listar_documentos, buscar_fragmentos, leer_documento]`.
  Todas devuelven JSON `str`, nunca lanzan: un fallo interno o un argumento inválido (p. ej. `tipo` fuera del `Literal`,
  gestionado con `handle_validation_error`) vuelve como `{"fuentes": [], "fragmentos": [], "error": "..."}`.
- Las tools llaman a `retrieval.obtener_vector_store()` / `retrieval.obtener_catalogo_documentos()` / `retrieval.recuperar_sigem`
  **a través del módulo** (`retrieval.x`), para poder sustituirlos con monkeypatch en los tests (ver `tests/test_tools.py`:
  Chroma en memoria + `DeterministicFakeEmbedding`; `RERANKER_HABILITADO=False` para que nunca cargue un modelo).
- `CatalogoDocumentos` ganó `listar(tipo, anio, tema, limite)` y `DocumentoCatalogado.fecha_expedicion` (fecha completa).
  `listar_documentos` también devuelve `total` y acepta `tipo` en plural/mayúsculas ("Decretos"); un tipo desconocido
  devuelve `aclaracion` con los válidos.
- Candidatos: máximo 5 (la spec) — `agente_sigem/referencias.py` sigue limitado a 3 hasta que la Fase 4/5 lo elimine.
- Con Chroma real: referencia exacta/ambigua/inexistente, listados y lectura funcionan (listar tema "presupuesto" → 29 coincidencias).
  `buscar_fragmentos` con reranker ≈ 1.8–3.5 s por llamada; **la primera llamada tarda ~30 s** (import de torch + pesos), por eso
  hay que llamar a `reranker.precargar()` en el lifespan de FastAPI (`RERANKER_PRECARGAR=true`).
- `test_salida_es_compatible_con_fuente_y_candidato_de_la_api` importa `agente_sigem.api`, que hoy construye el agente al
  importarse (necesita `GOOGLE_API_KEY`); si falla el import, el test se salta. Deja de ser necesario cuando la Fase 5
  construya el grafo en el lifespan.

### Estado de tests al iniciar (Fase 0)
`13 passed, 1 failed` (**corregido en la Fase 2**: 26 passed). Fallaba `tests/test_retrieval_diversidad.py::test_limita_fragmentos_por_documento_en_consulta_amplia`:
`clave_documento` (retrieval.py) construye la clave con `metadata["source"]`, pero el fixture del test solo trae
`nombre_archivo`; los fragmentos de `principal.pdf` y `secundario.pdf` con la misma página y contenido colisionan y se
descarta `secundario.pdf`. Es un fallo preexistente (introducido en `9dd88aa`), **no** de esta fase. Arreglo previsto en
la Fase 2: que `clave_documento` use `archivo_documento(...)` (`nombre_archivo` con respaldo a `source`).

### Línea base de recuperación (código actual, sin reranker)
`backend/evaluacion/linea_base_recuperacion.json` — 33 casos con `archivos_esperados`, hit@8 global **0.88**, MRR **0.77**.
Fallos: `tema-04` (arma/porte de armas), `mal-01` ("decrto 83… horaro de atension"), `voz-num-01` y `voz-num-03`
(números en palabras). `voz_numeros_en_palabras` hit@8 = 0.33. Esos son exactamente los casos que debe mejorar el analista.

### Datos del catálogo a tener en cuenta
- `catalogo_chroma.csv` tiene 71 filas (la especificación dice ≈72).
- Dos documentos con título "Decreto 116 de 2018" (`22529_…1162018.pdf` y `22531_…1262018.pdf`): sirve para ambigüedad.
- "Resolución 4397" existe como 2017 (`23661`) y 2018 (`23655`).
- `23638_resolucion-no-00772019.pdf` tiene título "Decreto 077 de 2019" (nombre de archivo dice resolución).
- `2403_decretono184.pdf` tiene como título el texto de la descripción (sin tipo ni número parseables).
- Hay ruido en descripciones (`​`, `�`); no asumir texto limpio.
- `dataset.jsonl` agrega dos campos opcionales fuera de la sección 13: `consulta_recuperacion` (pregunta autónoma
  equivalente para evaluar sin LLM en casos de varios turnos) y `candidatos_esperados` (bool). `intencion_esperada` y
  `archivos_esperados` se refieren al **último** turno.
