# Especificación: migración a sistema multi-agente (SIGEM)

> Documento de referencia para Claude Code. Va en la raíz del repo como `docs/MIGRACION_AGENTICA.md`.
> Cada prompt de fase remite a secciones de este archivo. Si algo de un prompt contradice este archivo, **este archivo manda**.

---

## 1. Contexto

Agente ciudadano de la Alcaldía de Marinilla que responde sobre normas del repositorio SIGEM (≈72 PDFs indexados en Chroma), adaptando la respuesta al perfil del usuario (P0–P3). Stack: FastAPI + LangGraph + Chroma + Postgres + Langfuse, frontend HTML plano.

Rama base: `feature-Sprint6`. Código en `backend/agente_sigem/`.

### Problema actual
La mayoría de las decisiones están codificadas a mano **fuera** del grafo, con listas de palabras y regex que no cubren el lenguaje real de los usuarios:

| Decisión | Dónde está hoy | Problema |
|---|---|---|
| Saludo, fuera de dominio, consulta genérica | `api.py → respuesta_directa_si_aplica` + `PALABRAS_SALUDO`, `PALABRAS_DOMINIO_SIGEM`, `PALABRAS_CONSULTA_GENERICA` | Por ejemplo, "¿cuánto cuesta el taxi?" se responde como *fuera de contexto*, aunque el Decreto 001 de 2019 trata de tarifas de taxi |
| Pregunta de seguimiento | `contexto.py → es_pregunta_de_seguimiento` (regex/marcadores) | Frágil |
| Reescritura con el documento activo | `contexto.py → enriquecer_consulta_con_contexto` | **Reemplaza el mensaje del usuario** y lo guarda así en la memoria |
| Documento activo | Tabla `contextos_sigem` en Postgres (`db.py`) | Vive fuera del estado del grafo |
| Referencia "decreto 12 de 2019" | `catalogo.extraer_referencia` aplicado a la pregunta | No entiende "decreto doce de dos mil diecinueve" (voz) ni errores de ortografía |
| Calidad del audio | `calidad.py` (llamada aparte a Gemini) | Llamada extra; no distingue ruido de una transcripción mal hecha |
| Estrategia de búsqueda | `retrieval.recuperar_sigem`, cascada fija dentro de la única tool `buscar_en_sigem` | El LLM no decide nada |
| Memoria | `MemorySaver` | Se pierde al reiniciar el contenedor |
| LLM | Ollama local (`llama3`) | Sin tool calling nativo; se migra a Gemini |

## 2. Objetivo

Convertir el agente en un **sistema multi-agente de flujo personalizado (LangGraph custom workflow)**:
- Un **agente analista** que entiende la pregunta.
- Un **agente investigador** (ReAct con tools) que consigue la evidencia.
- Un **evaluador** de evidencia (Corrective RAG).
- Un **redactor** que adapta la respuesta al perfil.

Además: re-ranking con `sentence-transformers`, migración de todos los LLM a Gemini y checkpointer en Postgres.

**Criterio de éxito:** responde igual o mejor que hoy en los casos ya cubiertos (referencia exacta, referencia ambigua, seguimiento, "explícame más", saludo, fuera de dominio, búsqueda global, audio poco claro) y además cubre preguntas mal escritas, números dichos en palabras, preguntas sin palabras clave del dominio y preguntas de inventario ("¿qué decretos hay de 2019?").

## 3. Arquitectura objetivo

```
START
  │
  ▼
analista ──────────────┬─ saludo / fuera_de_dominio / ambigua ──► respuesta_directa ──► END
  (flash-lite,          ├─ no_comprensible (audio) ─────────────► respuesta_directa (alerta) ──► END
   salida estructurada) │
                        └─ consulta_sigem / seguimiento
                                    │
                                    ▼
                        investigador  (subagente ReAct, flash)
                          tools: buscar_documento_por_referencia
                                 listar_documentos
                                 buscar_fragmentos
                                 leer_documento
                                    │
                                    ▼
                        evaluador ── insuficiente y intentos < MAX ──► investigador
                          (flash-lite)       (con consulta_reformulada)
                                    │ suficiente / hay candidatos / sin resultados
                                    ▼
                        redactor  (flash, estilo según perfil, citas)
                                    │
                                    ▼
                                   END
```

### Regla para decidir nodo o tool
- Lo que **siempre** debe pasar es un **nodo** con arista fija (analizar, evaluar, redactar).
- Lo que **depende de la pregunta** es una **tool** que el investigador decide usar.

### Regla para los regex
- **Sí** se permiten para parsear datos que controlamos: títulos del catálogo, nombres de archivo, el autocompletado `/documentos/sugerencias`.
- **No** se permiten para interpretar lo que escribe o dice el usuario. Eso lo hace el analista con salida estructurada, o el investigador al llenar los argumentos de las tools.
- Las tools **validan** contra el catálogo lo que el LLM extrajo: el LLM interpreta y el código verifica.

## 4. Estado del grafo (`agente_sigem/estado.py`)

```python
class EstadoAgente(TypedDict):
    # ── Persisten entre turnos (checkpointer) ──
    messages: Annotated[list[BaseMessage], add_messages]   # SOLO Human + AI final por turno
    documento_activo: dict | None    # {"titulo","archivo","descripcion","fecha_expedicion"}

    # ── Entrada del turno ──
    perfil: str                      # "P0".."P3"
    origen: str                      # "texto" | "voz"

    # ── Se REINICIAN en cada invoke (sin reducer; api.py los pasa vacíos) ──
    analisis: dict | None            # AnalisisConsulta.model_dump()
    evidencia: list[dict]            # fragmentos {"fuente": {...}, "contenido": str}
    fuentes: list[dict]              # mismo formato de FuenteSigem
    candidatos: list[dict]           # mismo formato de CandidatoDocumento
    intentos_busqueda: int
    consulta_actual: str             # la que usa el investigador (puede reformularla el evaluador)
    pasos: list[dict]                # {"agente","accion","icono"} para ChatResponse.pasos
    es_alerta: bool                  # audio no comprensible
```

**Detalle crítico:** los campos por turno **no llevan reducer**, para que pasarlos vacíos en el `invoke` los reinicie. Si se les pone `operator.add`, se acumularían entre turnos por culpa del checkpointer. Cada nodo que agrega un paso devuelve `{"pasos": estado["pasos"] + [nuevo]}`.

**`messages` se mantiene limpio:** el mensaje original del usuario (nunca reescrito) y la respuesta final del redactor o de respuesta_directa. Las llamadas a tools del investigador viven en **su propio historial interno** (el subagente) y no entran a `messages`.

## 5. Nodos

Todos los nodos reciben `(estado, config: RunnableConfig)` y **pasan `config` a cualquier `invoke` interno** para que Langfuse trace los sub-pasos. Todos los prompts de sistema van en `agente_sigem/prompts.py`.

### 5.1 `analista` (`nodos/analista.py`)
- Modelo: `ANALISTA_MODEL` (flash-lite), `temperature=0`, `with_structured_output(AnalisisConsulta)`.
- Entrada: los últimos `MAX_HISTORIAL_ANALISTA` (por defecto 6) mensajes de `messages`, el mensaje actual, `documento_activo` y `origen`.
- Salida:

```python
class FiltrosSigem(BaseModel):
    tipo: Literal["decreto","resolucion","acuerdo","ley","formato"] | None = None
    numero: int | None = None
    anio: int | None = None
    tema: str | None = None

class AnalisisConsulta(BaseModel):
    intencion: Literal["saludo","fuera_de_dominio","ambigua","consulta_sigem","seguimiento","no_comprensible"]
    consulta_autonoma: str      # pregunta reescrita para entenderse sin el historial, ortografía corregida
    filtros: FiltrosSigem | None = None
    usa_documento_activo: bool = False  # True si la pregunta se refiere al documento activo
    pregunta_aclaracion: str | None = None  # solo si intencion == "ambigua"
    razonamiento: str           # 1 frase, para depurar en Langfuse
```

- El prompt debe dejar claro que:
  - `fuera_de_dominio` es **solo** para temas ajenos a la gestión municipal (recetas, fútbol, programación). Ante la duda, debe elegir `consulta_sigem`, porque la búsqueda decidirá si hay información.
  - `seguimiento`: la pregunta se refiere al documento activo ("¿y la vigencia?", "explícame más"). Si el usuario menciona **otro** documento o pide una búsqueda general ("¿qué normas hay sobre…?"), es `consulta_sigem` con `usa_documento_activo=False`.
  - `ambigua`: no hay forma razonable de buscar ("necesito un documento"). Debe incluir `pregunta_aclaracion`.
  - `no_comprensible`: solo cuando `origen == "voz"` y la transcripción es ruido, silencio o palabras sueltas sin sentido. Si es una pregunta real mal transcrita, debe reconstruirla en `consulta_autonoma`.
  - Debe convertir números en palabras a dígitos ("doce de dos mil diecinueve" → `numero=12, anio=2019`) y corregir errores de ortografía ("decrto", "resolusion").
- Incluir **5–8 ejemplos few-shot** en el prompt que cubran: taxi sin palabras clave, seguimiento, búsqueda global con documento activo, otro documento con documento activo, voz con números en palabras, ruido de voz, saludo y fuera de dominio.
- **Fail-open:** si el LLM falla o la salida no se puede parsear, se usa `AnalisisConsulta(intencion="consulta_sigem", consulta_autonoma=mensaje, ...)` y se registra en el log. Nunca se bloquea al usuario por un error nuestro, igual que hace hoy `calidad.py`.
- Atajo determinístico permitido: si `origen == "voz"` y la transcripción tiene menos de 3 caracteres, el resultado es `no_comprensible` sin llamar al LLM.
- Paso: `{"agente":"analista","accion":"Analizando la pregunta: <intencion>","icono":"🧭"}`.

### 5.2 `respuesta_directa` (`nodos/respuesta_directa.py`)
Sin LLM. Según la intención:
- `saludo` → `RESPUESTA_SALUDO`
- `fuera_de_dominio` → `RESPUESTA_FUERA_DE_CONTEXTO`
- `ambigua` → `analisis.pregunta_aclaracion` o `RESPUESTA_CONSULTA_AMBIGUA`
- `no_comprensible` → `MENSAJE_AUDIO_POCO_CLARO` y `es_alerta=True`

Las constantes se mueven de `api.py` a `prompts.py` **con el mismo texto**. Agrega un `AIMessage` a `messages`.

### 5.3 `investigador` (`nodos/investigador.py`)
- Es un **subagente** creado con `langchain.agents.create_agent` (LangChain 1.x). Si la versión instalada no lo trae, se usa `langgraph.prebuilt.create_react_agent`. Hay que verificar cuál existe antes de escribir el código.
- Modelo: `INVESTIGADOR_MODEL` (flash), `temperature=0.1`. Tools: las 4 de la sección 6.
- Se invoca con un historial **nuevo** en cada intento:
  `[HumanMessage(<consulta_actual> + filtros + documento_activo si usa_documento_activo + motivo del evaluador si es un reintento)]`.
- Límite de pasos: `recursion_limit = INVESTIGADOR_MAX_PASOS` (por defecto 10). Si se alcanza, se usa lo que se haya recolectado.
- Prompt: su trabajo es **solo reunir evidencia**, no redactar para el ciudadano. Debe preferir `buscar_documento_por_referencia` cuando hay tipo y número, `listar_documentos` para preguntas de inventario y `buscar_fragmentos` (con `archivo` si hay documento activo) para el contenido. Debe terminar con una nota breve de qué encontró.
- Después de invocar el subagente, se recorren sus `ToolMessage`, se parsea el JSON y se **acumulan sin duplicados** `evidencia`, `fuentes` y `candidatos` en el estado. La clave de deduplicación es `(archivo, pagina, contenido[:160])`.
- Paso por cada tool call: `{"agente":"investigador","accion":"<nombre_tool>: <args resumidos>","icono":"🔎"}`.

### 5.4 `evaluador` (`nodos/evaluador.py`)
- Si hay `candidatos` y la evidencia está vacía, va directo al redactor (modo aclaración) sin llamar al LLM.
- Si la evidencia está vacía e `intentos_busqueda >= MAX_INTENTOS_BUSQUEDA` (por defecto 2), va al redactor (modo sin resultados).
- En otro caso, llama al LLM (`EVALUADOR_MODEL`, flash-lite) con salida estructurada:
  ```python
  class EvaluacionEvidencia(BaseModel):
      suficiente: bool
      motivo: str
      consulta_reformulada: str | None = None
  ```
- Si `suficiente=False` e `intentos_busqueda < MAX_INTENTOS_BUSQUEDA`, actualiza `consulta_actual`, incrementa `intentos_busqueda` y vuelve al investigador. Si no, sigue al redactor.
- Se puede apagar con `EVALUADOR_HABILITADO=false` (para ahorrar cuota). En ese caso pasa siempre al redactor.
- Fail-open: si hay error, sigue al redactor.
- Paso: `{"agente":"evaluador","accion":"Evidencia suficiente|Reintentando: <motivo>","icono":"✅"}`.

### 5.5 `redactor` (`nodos/redactor.py`)
- Modelo: `REDACTOR_MODEL` (flash), `temperature=0.3`.
- Prompt = reglas base + `ESTILO_POR_PERFIL[perfil]`. Los textos actuales de P0–P3 se mueven **sin cambios** a `prompts.py`.
- Entrada: el mensaje original del usuario, `consulta_autonoma`, la evidencia (fragmentos numerados con título y página) y los candidatos.
- Tres modos:
  - **respuesta**: responde **solo** con la evidencia y cita en línea con el formato `[Decreto 001 de 2019, p. 3]`.
  - **aclaración**: si hay candidatos, le pide al usuario elegir y menciona títulos y años. No responde con documentos que no estén en la lista.
  - **sin resultados**: dice que la información no está disponible en SIGEM y sugiere acudir a la Alcaldía de Marinilla.
- Reglas que se conservan del prompt actual: no inventar, no completar con conocimiento general y no decir "no hay información" si algún fragmento es pertinente. Si el usuario pidió ampliar, debe sintetizar los detalles de los fragmentos.
- Agrega el `AIMessage` final a `messages`.
- **Actualiza `documento_activo`** con el documento de la primera fuente citada, solo si hubo fuentes. Si no hubo fuentes, conserva el anterior.
- Paso: `{"agente":"redactor","accion":"Redactando respuesta para perfil <P>","icono":"✍️"}`.

## 6. Tools (`agente_sigem/tools.py`)

Todas devuelven **JSON string** con el formato que ya consume `api.py`: `{"fuentes": [...], "fragmentos": [...], "candidatos": [...]?, "aclaracion": str?}`. Se reutilizan `fuente_de_documento` y `respuesta_de_recuperacion` de `retrieval.py`. Los docstrings deben estar en español y ser muy explícitos, porque son lo que el LLM lee para decidir.

| Tool | Firma | Usa | Comportamiento |
|---|---|---|---|
| `buscar_documento_por_referencia` | `(tipo: Literal[...], numero: int, anio: int \| None = None, pregunta: str \| None = None)` | nuevo `CatalogoDocumentos.buscar_por_atributos` | Si hay una coincidencia, trae los fragmentos de ese archivo (`where={"nombre_archivo": ...}`) y los ordena con el reranker contra `pregunta` (o, si no hay pregunta, por página), hasta 8. Si hay varias, devuelve `candidatos` (máximo 5) y `aclaracion`. Si no hay ninguna, devuelve `fuentes: []`, `fragmentos: []` y una sección `sugerencias` con los 3 títulos más parecidos (`rapidfuzz`) para que el agente se corrija. |
| `listar_documentos` | `(tipo: str \| None = None, anio: int \| None = None, tema: str \| None = None, limite: int = 15)` | catálogo en memoria | Para preguntas de inventario. Filtra por tipo y año, y por tema con coincidencia aproximada sobre título y descripción. Devuelve `documentos: [{titulo, archivo, fecha_expedicion, descripcion}]` y **también** los pone en `fuentes`, para que la API los muestre. No trae contenido. |
| `buscar_fragmentos` | `(consulta: str, archivo: str \| None = None)` | `recuperar_sigem` refactorizado | Búsqueda híbrida (texto + semántica) → reranker → diversificación (máximo 2 por documento si no hay `archivo`) → top 8. Si se pasa `archivo`, filtra Chroma con `where={"nombre_archivo": archivo}` en las dos estrategias y no diversifica. |
| `leer_documento` | `(archivo: str, paginas: list[int] \| None = None)` | Chroma `get` | Trae los fragmentos del archivo, ordenados por página (y filtrados por `paginas` si se pasan), hasta 12. Sirve para "resúmelo" o "explícame más". |

Todas capturan excepciones y devuelven `{"fuentes": [], "fragmentos": [], "error": "..."}`, nunca lanzan.

## 7. Recuperación y re-ranking

### `retrieval.py` (refactor, sin cambiar los embeddings)
- **Prohibido** cambiar `EMBEDDING_MODEL` o reindexar: Chroma ya está indexado con ese modelo y la consulta debe usar el mismo.
- `obtener_vector_store()` pasa a ser un singleton con `lru_cache`, porque hoy se crea uno por consulta.
- `recuperar_sigem(consulta, archivo=None)` → `buscar_por_texto(vs, consulta, archivo)` + búsqueda semántica (`similarity_search(consulta, k=24, filter={"nombre_archivo": archivo} si aplica)`) → unión sin duplicados → `reranker.reordenar(consulta, docs)` → `diversificar(docs, max_por_doc)` → top 8.
- Se elimina `buscar_por_referencia` y la rama de referencia dentro de `recuperar_sigem`, porque eso pasa a la tool. `candidatos_referencia_ambigua` y `respuesta_referencia_ambigua` también se eliminan o se mueven a `tools.py`.
- `combinar_y_diversificar` se conserva y **sigue siendo el respaldo** cuando el reranker está apagado o falla. Su test actual debe seguir pasando.
- `terminos_textuales` y las stopwords se conservan: son parte de la estrategia de búsqueda léxica, no interpretan la intención.

### `reranker.py` (nuevo)
- Usa `sentence_transformers.CrossEncoder`. Modelo por env `RERANKER_MODEL`, por defecto `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingüe, liviano, va bien en CPU). Alternativa de más calidad pero más pesada: `BAAI/bge-reranker-v2-m3`.
- Singleton con carga perezosa. `RERANKER_HABILITADO` (por defecto `true`). `max_length=512`.
- `reordenar(consulta, documentos, top_n=None) -> list[Document]`. Si está deshabilitado, no se puede importar o falla, **devuelve la lista sin cambios** y deja un aviso en el log una sola vez.
- Precarga opcional al iniciar FastAPI (`RERANKER_PRECARGAR=true`) para que la primera consulta no sea lenta.
- **Docker:** torch solo CPU mediante índice de uv:
  ```toml
  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true

  [tool.uv.sources]
  torch = { index = "pytorch-cpu" }
  ```
  Si no hay wheel de torch para Python 3.14, cambiar el Dockerfile a `python:3.13-slim` (coincide con `.python-version`). El caché de Hugging Face debe estar en una carpeta escribible por el usuario `app`: `ENV HF_HOME=/app/.cache/huggingface` y un volumen `hf-cache` en `Compose.yml`.

### `catalogo.py`
- Se conserva el parseo de títulos y nombres de archivo con regex (son datos).
- Se agrega `buscar_por_atributos(tipo, numero, anio) -> ResolucionCatalogo` sin regex sobre texto de usuario.
- Se agrega `similares(texto, limite)` con `rapidfuzz` sobre título y descripción.
- `resolver(consulta)` y `sugerencias()` se conservan para el endpoint de autocompletado y sus tests, pero **el agente ya no los usa**.

## 8. Modelos (Gemini)

- Nuevo `agente_sigem/llm.py` con una fábrica `crear_llm(nombre_modelo, temperature)` que devuelve `ChatGoogleGenerativeAI(model=..., google_api_key=config.GOOGLE_API_KEY, temperature=..., max_retries=config.LLM_MAX_RETRIES)`. Se cachea por `(modelo, temperatura)`.
- Variables nuevas en `config.py` (todas por env):
  `ANALISTA_MODEL`, `EVALUADOR_MODEL` (por defecto el mismo flash-lite que `VERIFICACION_MODEL`), `INVESTIGADOR_MODEL`, `REDACTOR_MODEL` (por defecto un flash), `LLM_MAX_RETRIES=3`, `MAX_INTENTOS_BUSQUEDA=2`, `INVESTIGADOR_MAX_PASOS=10`, `MAX_HISTORIAL_ANALISTA=6`, `EVALUADOR_HABILITADO=true`, `RERANKER_*`, `CHECKPOINTER=postgres|memory`.
- **Verificar los nombres de modelo** disponibles para la API key (`genai.Client().models.list()`) antes de fijar los valores por defecto. Deben quedar en `.env.example`.
- Se eliminan `langchain-ollama`, `OLLAMA_*` y `CHAT_MODEL`. `/health/llm` pasa a verificar Gemini (por ejemplo `models.get(INVESTIGADOR_MODEL)`) con el mismo formato de respuesta `{"status","model"}`.
- Embeddings, STT y TTS **no cambian**.
- Plan gratuito: hay entre 3 y 6 llamadas por turno. `max_retries` con backoff es obligatorio. Los nodos de clasificación usan flash-lite para no gastar la cuota del flash.

## 9. Persistencia

- `PostgresSaver` (`langgraph-checkpoint-postgres`) con `psycopg_pool.ConnectionPool(config.DATABASE_URL, kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})` y `checkpointer.setup()` al iniciar.
- **El grafo no se compila al importar el módulo.** Hoy `agent.py` lo hace en la última línea. Debe construirse en el `lifespan` de FastAPI y guardarse en `app.state.agente`, o con una función `obtener_agente()` que lo cree al primer uso. Así los tests importan sin necesitar Postgres.
- Con `CHECKPOINTER=memory` se usa `MemorySaver` (para tests y desarrollo local).
- Las tablas `turnos` (historial visible al usuario) **se conservan sin cambios**. La tabla `contextos_sigem` deja de usarse: el `documento_activo` vive en el estado. Hay que quitar `guardar_contexto_sigem` y `obtener_contexto_sigem` de `db.py`, pero **no** borrar la tabla de la base (no se hace ninguna migración destructiva).

## 10. Contrato de la API (no debe cambiar)

El frontend (`frontend/index.html`) usa estos campos y **no se modifica**:
- `POST /chat` `{mensaje, thread_id, perfil}` → `ChatResponse{respuesta, thread_id, pasos[{agente,accion,icono}], fuentes[FuenteSigem], candidatos[CandidatoDocumento], tiempo_ms}`
- `POST /chat-voz` (form: audio, thread_id, perfil) → `ChatVozResponse{transcripcion, respuesta, thread_id, pregunta_audio_url, respuesta_audio_url, fuentes, candidatos, alerta}`
- `GET /historial/{thread_id}`, `GET /documentos/sugerencias?q=`, `GET /health`, `GET /health/llm`

### `api.py` después de la migración
- Se eliminan `respuesta_directa_si_aplica`, `respuesta_chat_directa`, `normalizar_texto`, los tres conjuntos `PALABRAS_*`, `mensaje_para_agente`, `fuentes_del_ultimo_turno`, `candidatos_del_ultimo_turno` y `guardar_fuentes_como_contexto`.
- Una función común `ejecutar_turno(thread_id, mensaje, perfil, origen, tags) -> dict` invoca el grafo con:
  `{"messages":[HumanMessage(mensaje)], "perfil":..., "origen":..., "analisis":None, "evidencia":[], "fuentes":[], "candidatos":[], "intentos_busqueda":0, "consulta_actual":"", "pasos":[], "es_alerta":False}`
  y la `config` de Langfuse que ya existe (session_id y tags).
- La respuesta se construye **desde el estado**: `resultado["messages"][-1].content`, `resultado["fuentes"]`, `resultado["candidatos"]`, `resultado["pasos"]` y `resultado["es_alerta"]`.
- `/chat-voz`: transcribe → `ejecutar_turno(origen="voz")` → si `es_alerta`, se guarda el turno como `alerta` y se devuelve `alerta=True` **sin TTS** (igual que hoy). Si no, se hace TTS, se sube a S3 y se guarda en la base como hoy.
- Se elimina `calidad.py`, porque su función ahora la cumple el analista.
- Se conservan `db.guardar_turno` y los mensajes en Postgres exactamente igual.

## 11. Mapa de archivos

| Archivo | Acción |
|---|---|
| `agente_sigem/config.py` | Modificar: variables Gemini/reranker/grafo; quitar Ollama |
| `agente_sigem/llm.py` | **Nuevo**: fábrica de LLMs de Gemini |
| `agente_sigem/reranker.py` | **Nuevo** |
| `agente_sigem/retrieval.py` | Refactor (sección 7) |
| `agente_sigem/catalogo.py` | Agregar `buscar_por_atributos` y `similares` |
| `agente_sigem/tools.py` | **Nuevo**: las 4 tools |
| `agente_sigem/estado.py` | **Nuevo**: `EstadoAgente`, `AnalisisConsulta`, `FiltrosSigem`, `EvaluacionEvidencia` |
| `agente_sigem/prompts.py` | **Nuevo**: prompts de los 4 agentes, `ESTILO_POR_PERFIL` y constantes de respuesta directa |
| `agente_sigem/nodos/__init__.py`, `analista.py`, `respuesta_directa.py`, `investigador.py`, `evaluador.py`, `redactor.py` | **Nuevos** |
| `agente_sigem/agent.py` | Reescribir: solo arma el grafo (`construir_grafo(checkpointer)`) y las rutas |
| `agente_sigem/checkpointer.py` | **Nuevo**: Postgres o memoria |
| `agente_sigem/api.py` | Simplificar (sección 10) + lifespan |
| `agente_sigem/db.py` | Quitar funciones de `contextos_sigem` |
| `agente_sigem/contexto.py`, `agente_sigem/calidad.py` | **Eliminar** |
| `tests/test_contexto.py` | **Eliminar** (sus casos pasan al dataset de evaluación y a los tests del analista) |
| `tests/test_catalogo.py`, `test_sugerencias_catalogo.py`, `test_retrieval_diversidad.py` | Deben seguir pasando |
| `tests/test_tools.py`, `test_reranker.py`, `test_grafo.py`, `test_api.py` | **Nuevos** |
| `evaluacion/dataset.jsonl`, `evaluacion/evaluar_recuperacion.py`, `evaluacion/evaluar_agente.py` | **Nuevos** |
| `pyproject.toml` | + `sentence-transformers`, `torch` (CPU), `rapidfuzz`, `langgraph-checkpoint-postgres`, `psycopg-pool`, `pytest` (dev); − `langchain-ollama` |
| `Dockerfile`, `../Compose.yml` | torch CPU, `HF_HOME`, volumen `hf-cache` |
| `.env.example` | **Nuevo** (sin secretos) |
| `README.md` | Actualizar arquitectura |

## 12. Pruebas

- **Los tests no llaman a Gemini ni a Chroma reales.** Se usa `langchain_core.language_models.fake_chat_models.GenericFakeChatModel` o `FakeMessagesListChatModel`, o se hace monkeypatch de `crear_llm`, para devolver salidas estructuradas y tool calls predefinidas. El vector store se reemplaza por un fake o por un Chroma en memoria con embeddings falsos (`langchain_core.embeddings.FakeEmbeddings`).
- `test_grafo.py` cubre las rutas: saludo → respuesta_directa; no_comprensible → `es_alerta`; consulta → investigador → evaluador (reintento una vez) → redactor; candidatos → redactor en modo aclaración; analista que lanza excepción → fail-open; `documento_activo` que persiste entre dos turnos con `MemorySaver`; `pasos`, `fuentes` y `candidatos` que **no** se acumulan entre turnos.
- `test_api.py` usa `TestClient` con el grafo falso y verifica que la forma de `ChatResponse` y `ChatVozResponse` no cambió.
- Comandos de verificación en cada fase: `uv run pytest -q` y `uv run python -c "import agente_sigem.api"`.

## 13. Evaluación (antes y después)

- `evaluacion/dataset.jsonl`, unos 40 casos construidos con `catalogo_chroma/catalogo_chroma.csv` (títulos y descripciones reales). Formato:
  `{"id","turnos":[{"origen":"texto|voz","mensaje":"..."}],"perfil":"P0..P3","intencion_esperada":"...","archivos_esperados":["..."],"categoria":"..."}`.
  Categorías: referencia_exacta, referencia_ambigua, referencia_inexistente, tema_sin_palabras_clave, inventario, seguimiento (varios turnos), ampliacion, busqueda_global_con_doc_activo, mal_escrita, voz_numeros_en_palabras, voz_ruido, saludo, fuera_de_dominio.
- `evaluar_recuperacion.py`: **sin LLM**, solo con embeddings. Calcula hit@8 y MRR de `archivos_esperados` usando `recuperar_sigem`, con reranker y sin él (`--sin-reranker`). Se puede correr sobre la versión vieja y la nueva.
- `evaluar_agente.py`: corre el grafo completo con `CHECKPOINTER=memory`, con un `thread_id` por caso y una pausa configurable entre llamadas (por la cuota). Mide el acierto de intención, si `archivos_esperados` aparece en `fuentes` y si hay candidatos cuando se esperan. Escribe `evaluacion/resultados_<fecha>.json` y un resumen por categoría.

## 14. Convenciones

- Identificadores, comentarios, prompts y docstrings en **español**, como el código existente.
- No tocar `chroma_db/`, no reindexar, no cambiar `EMBEDDING_MODEL`.
- No modificar `frontend/`.
- Un commit por fase, en la rama `feature-agentico`, con un mensaje que describa la fase.

---

## 15. Estado de implementación

> Escrito al cerrar la Fase 6 (rama `feature-agentico`). Lista lo que se implementó, las desviaciones respecto
> a esta especificación y su razón. Donde esta sección y las anteriores difieren, **rige lo implementado**
> (el código y los tests son la fuente de verdad); las secciones 1–14 se conservan como diseño original.

### Resumen
| Sección | Estado |
|---|---|
| 3 Arquitectura (analista → investigador → evaluador → redactor) | Implementada tal cual (`agent.construir_grafo`). |
| 4 Estado | Implementado; campos por turno sin reducer. |
| 5 Nodos | Implementados con las desviaciones D1–D6 de abajo. |
| 6 Tools | Implementadas con la desviación D7. |
| 7 Recuperación y reranker | Implementada. El reranker queda **activo por defecto** pero su efecto neto es discutible (ver D12). |
| 8 Modelos (Gemini) | Implementada; Ollama retirado. |
| 9 Persistencia | Implementada (`PostgresSaver` / `MemorySaver`; grafo construido en el `lifespan`). |
| 10 Contrato de la API | Sin cambios de contrato; código basado en regex eliminado. |
| 12 Pruebas | 131 tests, sin llamadas reales a Gemini ni a Chroma. |
| 13 Evaluación | `evaluar_recuperacion.py` ejecutado; `evaluar_agente.py` escrito y probado en su lógica, **pendiente de correr** por la cuota (ver D14). |

### Desviaciones y su razón
| # | Desviación | Razón |
|---|---|---|
| D1 | `intentos_busqueda` lo incrementa el **investigador** en cada ejecución; el evaluador solo reformula `consulta_actual` (la spec decía que lo hacía el evaluador). | Con la lectura literal, una búsqueda con evidencia vacía dejaba `intentos=0` y el evaluador volvía a reintentar sin tope claro. Así `MAX_INTENTOS_BUSQUEDA` significa "máximo de pasadas del investigador". |
| D2 | La señal de reintento se codifica en el último paso del evaluador (`"Reintentando: <motivo>"`, `estado.PREFIJO_REINTENTO`); no hay un campo nuevo en `EstadoAgente`. | Mantener el estado "exactamente como en la sección 4". `ruta_despues_evaluador` y el investigador leen ese paso. |
| D3 | Un listado (`listar_documentos`) se convierte en evidencia marcada `"tipo": "listado"` y **no** activa `documento_activo`. | Sin fragmentos, el redactor habría entrado en modo "sin resultados"; y activar el primer documento de un listado sería arbitrario. |
| D4 | `respuesta_directa` agrega un paso (`💬`); el analista corrige `seguimiento`/`usa_documento_activo` a `consulta_sigem`/`false` si no hay documento activo. | Coherencia del estado y trazabilidad en `pasos`. |
| D5 | El subagente se crea con `create_agent(..., checkpointer=False)` y se consume con `stream(stream_mode="values")`. | Que cada intento empiece sin historial aunque el padre tenga checkpointer, y conservar lo recolectado si salta `GraphRecursionError`. |
| D6 | El redactor **propaga** sus errores (solo analista y evaluador son fail-open). La API los devuelve como 500 con el texto del error. | La spec no pide fail-open para el redactor. Pendiente: mensaje amable para errores de cuota. |
| D7 | `buscar_documento_por_referencia` acepta además `codigo` (código del archivo) y `tipo`/`numero` pasan a opcionales; `FiltrosSigem` tiene `codigo`; `CatalogoDocumentos` suma `buscar_por_codigo` y `listar`. `listar_documentos` devuelve también `total` y acepta `tipo` en plural. | Cubrir "documento 18399", que antes resolvía un regex y el analista clasificaba como `ambigua`; pedido por los usuarios de la alcaldía. |
| D8 | `ejecutar_turno` recibe el `agente` como primer parámetro y corre en `run_in_threadpool`; el arranque usa `lifespan` en lugar de `@app.on_event`. | Testabilidad y no bloquear el event loop con `invoke` síncrono; `on_event` está deprecado. |
| D9 | `referencias.py` existió en las fases 2–4 como transición y se eliminó en la 5; `catalogo.resolver()`/`extraer_referencia()` y sus regex se conservan solo para el autocompletado y sus tests. | Cumple la regla de regex de la sección 3: el agente ya no interpreta texto del usuario con regex. |
| D10 | `dataset.jsonl` añade campos opcionales (`consulta_recuperacion`, `candidatos_esperados`, `origen_caso`); 43 casos en lugar de "unos 40". | Evaluar recuperación sin LLM en casos de varios turnos y medir candidatos. |
| D11 | `VERIFICACION_MODEL` y `combinar_sin_duplicados`/`RESULTADOS_SEMANTICOS` se eliminaron; `clave_documento` ahora usa `nombre_archivo` (con respaldo a `source`). | Código muerto tras retirar `calidad.py`; y corrección del test de diversidad que fallaba desde `9dd88aa`. |
| D12 | El reranker mmarco queda habilitado por defecto, aunque en la evaluación de recuperación no mejora el conjunto (hit@8 0.82 con reranker vs 0.85 sin él, sobre `recuperar_sigem` puro). Ayuda en preguntas sin palabras clave; empeora números dichos en palabras y consultas genéricas. | Es lo que pide la spec. Debe decidirse con `evaluar_agente.py` (consultas ya reescritas por el analista); `RERANKER_HABILITADO=false` es un interruptor sin riesgo. |
| D13 | `Dockerfile`: `uv.lock` es opcional (`COPY pyproject.toml uv.lock*`) y se quitó `--locked`; hay `.dockerignore` y `healthcheck`; el `chown` se limita a `/app/.cache`. Python 3.14 se mantiene (hay wheel de torch CPU para 3.14; imagen ≈ 2,0 GB descomprimida / ≈ 1,0 GB comprimida, `torch 2.14.1+cpu`, sin CUDA). | `uv.lock` está en el `.gitignore` del backend, así que `--locked` fallaba en un clon limpio; `chown -R /app` duplicaba el venv (~2 GB) en una capa. |
| D15 | La precarga del reranker corre en segundo plano (`asyncio.create_task`) en lugar de bloquear el `lifespan`; `Compose.yml` fija `CHECKPOINTER=postgres`, `HF_HOME`, `RERANKER_PRECARGAR=true`, el volumen `hf-cache` y un `healthcheck`. | En una red con proxy TLS la descarga del modelo tardó ~4 min en fallar y el arranque quedó bloqueado; la API debe estar disponible aunque el reranker no. |
| D14 | `evaluar_agente.py` no se ejecutó en la Fase 6. | El plan gratuito de Gemini limita `gemini-3.5-flash` a ~20 solicitudes/día; el dataset implica ~240 llamadas. Ver "Pendientes". |

### Pendientes conocidos
1. **Correr `evaluar_agente.py`** con cuota suficiente (facturación, o `INVESTIGADOR_MODEL`/`REDACTOR_MODEL` en flash-lite) y decidir sobre el reranker y los few-shot del analista con esos números.
2. **Errores de cuota (429)**: `/chat` devuelve 500 con el texto crudo del proveedor; conviene un mensaje amable (y quizá 503).
3. **`fuentes` acumula todas las búsquedas del turno** (p. ej. 12 fuentes con una sola citada); filtrar a las realmente citadas por el redactor.
4. Una transcripción vacía queda en `messages` como `HumanMessage("")` junto con la alerta.
5. Latencia de 10–60 s por turno con búsqueda (varias llamadas encadenadas + reranker en CPU).
