# CLAUDE.md — Proyecto SIGEM (Alcaldía de Marinilla)

## Resumen del proyecto
1. Agente ciudadano de la Alcaldía de Marinilla que responde sobre normas del repositorio SIGEM.
2. El repositorio son ≈72 PDFs (decretos, resoluciones, leyes, acuerdos) indexados en Chroma.
3. Backend: FastAPI + LangGraph en `backend/agente_sigem/`; frontend HTML plano en `frontend/`.
4. Recuperación híbrida (texto literal + semántica) sobre Chroma con embeddings de Gemini.
5. La respuesta se adapta al perfil del usuario (P0–P3) con prompts de estilo distintos.
6. Entrada por texto y por voz (STT/TTS con Gemini; audios en S3).
7. Historial de turnos en Postgres; trazas en Langfuse.
8. LLM: Gemini (flash-lite para clasificar, flash para investigar y redactar); las decisiones las toman los agentes, no regex.
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
- [x] **Fase 4** — Estado, prompts y nodos: `estado.py`, `prompts.py`, `nodos/*` (módulos probados por separado)
- [x] **Fase 5** — Grafo conectado: `agent.py` (`construir_grafo`), `checkpointer.py`, `api.py` simplificado + lifespan, `db.py`; eliminados `contexto.py`, `calidad.py`, `referencias.py`, `test_contexto.py`
- [x] **Fase 6** — Despliegue y documentación: Dockerfile/Compose (torch CPU, `hf-cache`), `evaluar_agente.py` (escrito; **sin correr**), READMEs, sección 15 de la spec
- [x] **Fase 7** — Ajustes sin cuota (429 amable, `fuentes` citadas, audio vacío, `HF_HUB_OFFLINE`) y **evaluación de extremo a extremo ejecutada** (ver abajo). Pendientes: ver "Hallazgos de la evaluación"

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

### Fase 4 — estado, prompts y nodos (aún sin conectar a `agent.py`/`api.py`)
- **Módulos:** `estado.py`, `prompts.py`, `nodos/{analista,respuesta_directa,investigador,evaluador,redactor}.py` (+ `_comun.py`).
  `nodos/__init__.py` no reexporta funciones, para que `agente_sigem.nodos.analista` siga siendo el módulo (monkeypatch de `crear_llm`).
- **Rutas:** `nodos.analista.ruta_despues_analista` → `"investigador" | "respuesta_directa"`;
  `nodos.evaluador.ruta_despues_evaluador` → `"investigador" | "redactor"`.
- **Decisiones tomadas (confirmadas):**
  1. `intentos_busqueda` lo incrementa el **investigador** en cada ejecución; el evaluador solo reformula `consulta_actual`.
     `MAX_INTENTOS_BUSQUEDA=2` = máximo 2 pasadas del investigador. (Evita bucles con evidencia vacía.)
  2. Un listado (`listar_documentos`) no activa ningún `documento_activo`: el investigador lo vuelve evidencia con `"tipo": "listado"`
     y el redactor lo ignora para `documento_activo`.
- **Señal de reintento sin campo extra:** el evaluador deja el paso `"Reintentando: <motivo>"` (`estado.PREFIJO_REINTENTO`);
  `ruta_despues_evaluador` mira el **último** paso y el investigador lo lee para incluir el motivo en su encargo.
- **Subagente:** `create_agent(..., checkpointer=False)` para que cada intento empiece sin historial aunque el grafo padre tenga
  checkpointer; se consume con `stream(stream_mode="values")` para conservar lo recolectado si salta `GraphRecursionError`.
- **Fails:** analista y evaluador son fail-open; el investigador no lanza (usa lo recolectado); el redactor **sí propaga** sus errores
  (decisión pendiente para la capa API de la Fase 5: mensaje de error o 500).
- **Campos extra de salida:** nodos devuelven solo las claves que cambian; `respuesta_directa` añade un paso (`💬`) que la spec no pide.
- **Prompt del analista validado con Gemini real** (flash-lite, 7 casos distintos a los few-shot): taxi/licor, voz con números en
  palabras, ruido, seguimiento, búsqueda global con doc activo, fuera de dominio y ambigua → todos correctos.
- Constantes de respuesta directa y `ESTILO_POR_PERFIL` ahora viven en `prompts.py`; `agent.py` y `api.py` las importan.
- `tests/test_nodos.py`: 53 casos (fakes `FakeEstructurado`/`ModeloFalso`, tools dobles y un grafo mínimo con `MemorySaver`
  que verifica persistencia de `documento_activo` y reinicio de `pasos`/`fuentes`/`candidatos`).

### Fase 5 — grafo conectado y API simplificada
- **Grafo:** `agent.construir_grafo(checkpointer)`: START → analista → (respuesta_directa | investigador) → evaluador → (investigador | redactor) → END.
  No se compila al importar; `api.lifespan` hace `init_db` → `crear_checkpointer()` → `app.state.agente` → precarga del reranker si `RERANKER_PRECARGAR`.
- **`checkpointer.crear_checkpointer() -> (saver, cerrar)`:** `PostgresSaver` + `ConnectionPool(autocommit, prepare_threshold=0, dict_row)` o `MemorySaver`.
- **`api.ejecutar_turno(agente, thread_id, mensaje, perfil, origen, tags)`** (corre en `run_in_threadpool`): pasa los campos por turno vacíos.
  Respuesta desde el estado. `/chat-voz`: si `es_alerta` → turno con rol `alerta`, `alerta=True`, sin TTS.
- **Eliminado:** `buscar_en_sigem`, `INSTRUCCIONES_BASE`, `contexto.py`, `calidad.py`, `referencias.py`, `tests/test_contexto.py`, y de `api.py` todas las
  funciones/constantes de la sección 10; de `db.py` el contexto SIGEM y el modelo `ContextoSigem` (la tabla `contextos_sigem` no se toca).
  `evaluar_recuperacion.py` ya no tiene `--con-referencia`.
- **Regex que quedan (solo datos propios / léxico):** `catalogo.py` (`PATRON_REFERENCIA`, `PATRON_CODIGO_ARCHIVO`, `PATRON_ANIO`, `re.match` del código de archivo,
  `re.findall` en `sugerencias`), usados para parsear títulos/nombres de archivo y por `resolver()`/`extraer_referencia()` del autocompletado;
  `retrieval.py` (`normalizar`, `terminos_textuales` y las stopwords `PALABRAS_NO_SIGNIFICATIVAS`/`PALABRAS_BUSQUEDA_GENERICAS`, que tokenizan la consulta ya reescrita).
- **Tests:** `tests/fakes_llm.py` y `tests/grafo_falso.py` (dobles compartidos), `test_grafo.py` (18), `test_api.py` (13). Total: 117.
  `httpx` es dependencia de desarrollo (TestClient).
- **Humo con la API real (CHECKPOINTER=memory, `db` en memoria):** hola, taxi, "decreto 83", "qué dice el decrto 1 dl 2019" + "explícame más", inventario 2019 y
  fuera de dominio respondieron bien y con citas. **Latencia alta: 10–60 s por turno** (taxi: 61 s con 3 llamadas del investigador + evaluador + redactor).
- **Hallazgos abiertos para la Fase 6 / siguiente iteración:**
  1. ~~**Código de archivo**~~ **RESUELTO:** `buscar_documento_por_referencia` acepta `codigo` (tipo y numero pasan a opcionales; se exige tipo+numero o codigo),
     `FiltrosSigem.codigo` y `CatalogoDocumentos.buscar_por_codigo` (compara los dígitos con el prefijo del archivo). "Necesito el documento 18399" →
     `consulta_sigem` con `codigo=18399` (validado con Gemini real). Un código inexistente devuelve `aclaracion`; con tipo+numero como respaldo.
  2. **Cuota de Gemini (diagnosticada):** el plan gratuito limita `gemini-3.5-flash` a **20 solicitudes por día y por proyecto**
     (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`; el 429 trae `retryDelay`). Cada turno con búsqueda gasta 3–6 llamadas flash (investigador + redactor),
     así que el plan gratuito alcanza para ~4–6 consultas al día. `gemini-3.5-flash-lite` tiene su propia cuota y seguía disponible. Para uso real hace falta
     facturación o mover investigador/redactor a flash-lite (medir calidad antes). Al agotarse: al agotarse (`429 RESOURCE_EXHAUSTED`) `/chat` responde 500 con el texto crudo del proveedor en `detail`; conviene un mensaje amable.
  3. **`fuentes` acumula todas las búsquedas** del turno (taxi devolvió 12 fuentes, solo una citada); se podría filtrar a las realmente citadas por el redactor.
  4. Una transcripción vacía queda en `messages` como `HumanMessage("")` + la alerta; inocuo pero ensucia el historial del hilo.
  5. `VERIFICACION_MODEL` quedó huérfano en `config.py` (lo usaba `calidad.py`).

### Fase 6 — despliegue, evaluación y documentación
- **Imagen Docker** (`docker compose build backend`, Python 3.14 + `torch 2.14.1+cpu`, sin CUDA): ≈ 2,0 GB descomprimida (997 MB comprimida). El primer build daba
  7 GB porque `chown -R /app` duplicaba el venv en otra capa; ahora solo se hace `chown` de `/app/.cache`. `uv.lock` está en `.gitignore`: el Dockerfile lo copia
  si existe y no usa `--locked`. Hay `backend/.dockerignore`.
- **Compose validado en vivo:** `db` + `backend` con `CHECKPOINTER=postgres`: el `PostgresSaver` creó `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`,
  `checkpoint_migrations` y escribió 8 checkpoints para 2 turnos; `turnos` se guardó igual que antes. (Filas de prueba borradas.)
- **Red con proxy TLS:** dentro del contenedor `huggingface.co` falla con `CERTIFICATE_VERIFY_FAILED` (certificado autofirmado en la cadena). La API
  arrancó igual y usa el respaldo sin reranker; por eso la precarga ahora es en segundo plano (`app.state.precarga_reranker`).
- **Recuperación (33 casos, `recuperar_sigem` puro, sin referencias en el camino):** con reranker hit@8 0.82 / MRR 0.68; sin reranker 0.85 / 0.70
  (línea base de la Fase 0: 0.88 / 0.77, que sí incluía el regex de referencias). Las "regresiones" en `referencia_exacta` (1.00→0.80),
  `busqueda_global_con_doc_activo` y `mal_escrita` son casos de referencia (ref-04, glob-04, mal-01) que ahora resuelven el analista + la tool,
  no la recuperación; solo `evaluar_agente.py` puede medirlo. Archivos: `evaluacion/fase6_*.json`.
- **Limpieza final:** eliminados `VERIFICACION_MODEL`, `combinar_sin_duplicados` y `RESULTADOS_SEMANTICOS`; sin restos de Ollama, TODOs ni prints de depuración
  (los `print` que quedan son de los scripts de evaluación); sin secretos versionados (`.env` ignorado; `.env.example` sin valores).
- **`evaluar_agente.py`:** ~55 turnos / ~240 llamadas a Gemini; `--estimar`, `--categorias`, `--max-casos`, `--pausa`; aborta si detecta cuota diaria agotada.
  `tests/test_evaluacion.py` cubre su lógica sin Gemini. Total tests: 132.

### Fase 7 (parcial) — robustez sin gastar cuota
- **429 de Gemini:** `llm.es_error_de_cuota(error)` (mira código 429, `RESOURCE_EXHAUSTED`, `TooManyRequests` y la cadena de causas). El investigador la **propaga** (antes la
  tragaba y producía un "sin resultados" falso). `/chat` responde **200** con `prompts.MENSAJE_SERVICIO_SATURADO` y `/chat-voz` con `alerta=true` y sin TTS, sin guardar el turno:
  el frontend ignora `detail` (muestra siempre "Hubo un problema al contactar al agente") y no se puede modificar.
- **`fuentes` citadas:** `nodos.redactor.fuentes_citadas(texto, fuentes)` filtra por las citas `[Título, p. N]` (varias páginas y `;` soportados); sin citas utilizables se conservan
  todas. `documento_activo` prioriza el primer documento citado. El redactor ahora devuelve `fuentes` en su actualización de estado.
- **Audio vacío:** transcripción de menos de `LONGITUD_MINIMA_VOZ` (3) caracteres → alerta en la API sin invocar el grafo (no entra al historial del hilo).
- **Hugging Face:** `HF_HUB_OFFLINE` pasa por Compose/.env. Tras agregar la excepción en el antivirus, `huggingface.co` es alcanzable desde el contenedor (HTTP 200).
- **Camino de referencias sin LLM** (tool con los filtros que daría un analista perfecto): 15/15 casos con referencia (línea base 12/15; recuperación pura 10/15). Es un techo,
  no una medición del analista real; los casos que la línea base fallaba (mal-01, voz-num-01, voz-num-03) quedan cubiertos si el analista extrae bien los filtros.
- Tests: 144.

### Evaluación de extremo a extremo (2026-10-11)
- **Límites de Gemini medidos:** `gemini-3.5-flash` 20 solicitudes/día (plan gratuito) y `gemini-3.5-flash-lite` **15 por minuto** (`GenerateRequestsPerMinutePerProjectPerModel-FreeTier`).
  Una corrida sin pausas produjo 15 errores y varios *fail-open* silenciosos del analista/evaluador que invalidaban la medición.
- **`evaluar_agente.py` endurecido:** marca resultados `degradado` (razonamiento "Análisis no disponible…" o paso "Evaluación no disponible…"), los excluye de los porcentajes,
  reintenta el caso esperando 65 s si la cuota por minuto se agota, y aborta ante cuota diaria. `evaluador` deja ahora el paso `PASO_EVALUACION_NO_DISPONIBLE` al hacer fail-open.
  Usar `--pausa 20` con flash-lite.
- **Resultado (43 casos; búsqueda con flash-lite en los 4 agentes):** intención 1.00 y archivos 1.00 en todas las categorías, 0 errores, 0 degradados; mediana 13 s, p90 48 s.
  Tabla completa en la sección 15 de `docs/MIGRACION_AGENTICA.md`. Archivos: `evaluacion/resultados_fase7_*.json`.
- **Hallazgos de la evaluación y correcciones (aprobadas y aplicadas):**
  1. `inv-04` falso negativo → **A** `listar(tema=...)` por palabras significativas (`catalogo.PALABRAS_VACIAS_TEMA`, incluye "municipal", "marinilla"…; exige la mitad,
     tolera variantes; un tema solo de palabras vacías no filtra) y **B** `investigador.fragmentos_primero` (los fragmentos antes que los listados en `evidencia`).
  2. `inex-01/02` con fuentes de relleno → **C** `RespuestaRedactor.informacion_encontrada`: si es `false`, `fuentes=[]`, `documento_activo` intacto y paso
     "… · sin información pertinente en los fragmentos". Salida estructurada solo en modo respuesta; si falla, texto libre; la cuota se propaga.
  3. Métrica generosa → **D** `evaluar_agente.py` mide `informacion_ok` y `negativa_ok` (columnas `info` y `neg.`); `--ids` para correr casos sueltos.
  4. Reranker: se mantiene activo (decisión del equipo); esta evaluación no lo discrimina.
- **Cuota:** `gemini-3.5-flash-lite` 500/día y 15/min; `gemini-3.5-flash` 20/día. Tras las correcciones se repitieron 5 casos con Gemini (todos bien) y se agotó flash-lite.
  Sin repetir con Gemini: seg-02, voz-num-03, glob-02. Tests: 160.

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
