import asyncio
import logging
import time
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from google import genai
from google.genai import errors as genai_errors
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import db, repositorio_archivos, reranker, retrieval, storage
from .admin import router as admin_router
from .agent import construir_grafo
from .audio import limpiar_formato_para_voz, synthesize, transcribe
from .checkpointer import crear_checkpointer
from .config import config
from .llm import es_error_de_cuota
from .nodos.analista import LONGITUD_MINIMA_VOZ
from .prompts import MENSAJE_AUDIO_POCO_CLARO, MENSAJE_SERVICIO_SATURADO
from .retrieval import obtener_catalogo_documentos
from .tracing import obtener_langfuse_handler

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# MODELOS PYDANTIC (el contrato con el frontend no cambia)
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    mensaje: str = Field(min_length=1)
    thread_id: str = Field(default="default_user", min_length=1)
    perfil: str = Field(default="P0")

class PasoAgente(BaseModel):
    agente: str
    accion: str
    icono: str


class FuenteSigem(BaseModel):
    titulo: str
    archivo: str
    pagina: str | None = None
    fecha_expedicion: str | None = None
    descripcion: str | None = None


class CandidatoDocumento(BaseModel):
    titulo: str
    archivo: str
    fecha_expedicion: str | None = None
    descripcion: str | None = None


class ArchivoDescargable(BaseModel):
    titulo: str
    archivo: str
    url_descarga: str  # relativa (/documentos/{archivo}/descargar): el frontend la completa con su API_URL


class ChatResponse(BaseModel):
    respuesta: str
    thread_id: str
    pasos: list[PasoAgente]
    fuentes: list[FuenteSigem] = []
    candidatos: list[CandidatoDocumento] = []
    archivos: list[ArchivoDescargable] = []
    tiempo_ms: float

class ChatVozResponse(BaseModel):
    transcripcion: str
    respuesta: str
    thread_id: str
    pregunta_audio_url: str | None
    respuesta_audio_url: str | None
    fuentes: list[FuenteSigem] = []
    candidatos: list[CandidatoDocumento] = []
    archivos: list[ArchivoDescargable] = []
    alerta: bool = False


class TurnoOut(BaseModel):
    rol: str
    texto: str
    audio_url: str | None
    creado_en: str


# ══════════════════════════════════════════════════════════════
# EJECUCIÓN DE UN TURNO
# ══════════════════════════════════════════════════════════════

def ejecutar_turno(agente, thread_id: str, mensaje: str, perfil: str, origen: str, tags: list[str]) -> dict:
    """Invoca el grafo para un turno y devuelve el estado final.

    Los campos por turno se pasan reiniciados: el checkpointer conserva `messages` y
    `documento_activo` del hilo, pero el resto empieza vacío en cada turno.
    """
    langfuse_handler = obtener_langfuse_handler()
    config_graph = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler] if langfuse_handler else [],
        "metadata": {
            # Agrupa los turnos del hilo en una sesión de Langfuse y etiqueta el trace con el
            # perfil (P0-P3) para comparar el desempeño por perfil en el dashboard.
            "langfuse_session_id": thread_id,
            "langfuse_tags": tags,
        },
    }
    return agente.invoke(
        {
            "messages": [HumanMessage(content=mensaje)],
            "perfil": perfil,
            "origen": origen,
            "analisis": None,
            "evidencia": [],
            "fuentes": [],
            "candidatos": [],
            "intentos_busqueda": 0,
            "consulta_actual": "",
            "pasos": [],
            "es_alerta": False,
            "archivos": [],
            "archivos_no_disponibles": [],
        },
        config=config_graph,
    )


def _texto_respuesta(resultado: dict) -> str:
    return resultado["messages"][-1].text


def _fuentes(resultado: dict) -> list[FuenteSigem]:
    return [FuenteSigem(**fuente) for fuente in resultado.get("fuentes", [])]


def _candidatos(resultado: dict) -> list[CandidatoDocumento]:
    return [CandidatoDocumento(**candidato) for candidato in resultado.get("candidatos", [])]


def _archivos(resultado: dict) -> list[ArchivoDescargable]:
    """Compuerta determinística: solo hay archivos si el analista marcó que el usuario los pidió."""
    if not (resultado.get("analisis") or {}).get("solicita_archivo"):
        return []
    return [
        ArchivoDescargable(titulo=a.get("titulo") or a["archivo"], archivo=a["archivo"], url_descarga=a["url_descarga"])
        for a in resultado.get("archivos", [])
    ]


def _pasos(resultado: dict) -> list[PasoAgente]:
    return [PasoAgente(**paso) for paso in resultado.get("pasos", [])]


# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Arranque: tablas de historial, checkpointer, grafo y (opcional) precarga del reranker en segundo plano.

    El grafo no se construye al importar este módulo: así los tests y las herramientas
    pueden importar la API sin Postgres.
    """
    await run_in_threadpool(db.init_db)
    checkpointer, cerrar_checkpointer = await run_in_threadpool(crear_checkpointer)
    app.state.agente = construir_grafo(checkpointer)
    if config.RERANKER_PRECARGAR:
        # En segundo plano: si la descarga del modelo es lenta o falla (sin red, proxy con TLS propio),
        # la API no se queda esperando; el reranker cae al respaldo léxico (ver reranker.py).
        app.state.precarga_reranker = asyncio.create_task(run_in_threadpool(reranker.precargar))
    try:
        yield
    finally:
        cerrar_checkpointer()


app = FastAPI(
    title="Agente de respuesta ciudadana - SIGEM",
    description="API del sistema multi-agente (LangGraph + Gemini) que responde preguntas sobre procesos, formatos y normatividad del municipio de Marinilla, con base en el repositorio SIGEM.",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(admin_router)


@app.get("/health", tags=["sistema"])
async def health_check():
    """Indica si el proceso del API está disponible y, en modo http, si Chroma responde."""
    if config.CHROMA_MODO != "http":
        return {"status": "ok"}
    try:
        await run_in_threadpool(lambda: retrieval.crear_cliente_chroma().heartbeat())
    except Exception as error:
        logger.warning("Chroma no responde: %s", error)
        return JSONResponse(status_code=503, content={"status": "error", "chroma": "no disponible"})
    return {"status": "ok", "chroma": "ok"}


@app.get("/health/llm", tags=["sistema"])
def health_check_llm():
    """Comprueba que la API key de Gemini sea válida y que el modelo del agente exista."""
    if not config.GOOGLE_API_KEY:
        raise HTTPException(status_code=503, detail="Falta GOOGLE_API_KEY para conectar con Gemini.")

    try:
        cliente = genai.Client(api_key=config.GOOGLE_API_KEY)
        cliente.models.get(model=config.INVESTIGADOR_MODEL)
    except genai_errors.APIError as error:
        if error.code == 404:
            detail = f"Gemini está disponible, pero no existe el modelo {config.INVESTIGADOR_MODEL}."
        else:
            detail = f"Gemini respondió HTTP {error.code}. Revisa GOOGLE_API_KEY y la cuota."
        raise HTTPException(status_code=503, detail=detail) from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="No fue posible conectar con Gemini. Revisa la conexión a internet y GOOGLE_API_KEY.",
        ) from error

    return {"status": "ok", "model": config.INVESTIGADOR_MODEL}

app.add_middleware(
    CORSMiddleware,
    # SPRINT 7: las cookies administrativas requieren un origen explícito; el frontend público
    # sigue pudiendo consultar la API, pero las cookies administrativas
    # solo se aceptan desde orígenes explícitos, nunca junto a CORS con "*".
    allow_origins=config.ADMIN_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/documentos/sugerencias", response_model=list[CandidatoDocumento], tags=["sigem"])
def sugerencias_documentos(q: str = ""):
    """Sugiere documentos del catálogo sin ejecutar una búsqueda semántica."""
    return [
        CandidatoDocumento(
            titulo=documento.titulo,
            archivo=documento.archivo,
            fecha_expedicion=str(documento.anio) if documento.anio else None,
            descripcion=documento.descripcion,
        )
        for documento in obtener_catalogo_documentos().sugerencias(q)
    ]

def _detalle_503_descarga(error: Exception) -> str:
    logger.warning("Descarga no disponible: %s", error)
    return "La descarga de documentos no está disponible en este momento. Intenta de nuevo más tarde."


@app.get("/documentos/{archivo}/descargar", tags=["sigem"])
def descargar_documento(archivo: str):
    """Entrega el original de un documento indexado (streaming o redirect, según DRIVE_MODO_ENTREGA)."""
    # El parámetro solo se compara con los nombres del catálogo: nunca forma parte de una ruta de disco.
    if not any(documento.archivo == archivo for documento in obtener_catalogo_documentos().documentos):
        raise HTTPException(status_code=404, detail="No encontré ese documento en el repositorio de la Alcaldía.")
    registro = db.obtener_archivo(archivo)
    if registro is None:
        raise HTTPException(
            status_code=404,
            detail="Ese documento aún no está disponible para descarga. Puedes consultar su contenido en el chat.",
        )
    try:
        repositorio = repositorio_archivos.obtener_repositorio()
        if registro.proveedor != repositorio.proveedor:
            raise repositorio_archivos.ErrorRepositorio(
                f"El documento está en '{registro.proveedor}' pero el repositorio activo es '{repositorio.proveedor}'."
            )
        if config.DRIVE_MODO_ENTREGA == "redirect":
            enlace = repositorio.enlace_ver(registro.id_externo)
            if enlace:
                return RedirectResponse(enlace, status_code=302)
        trozos = repositorio.descargar(registro.id_externo)
    except repositorio_archivos.ErrorRepositorio as error:
        raise HTTPException(status_code=503, detail=_detalle_503_descarga(error)) from error
    cabeceras = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(archivo)}"}
    return StreamingResponse(trozos, media_type=registro.mime_type or "application/octet-stream", headers=cabeceras)


@app.post("/chat", response_model=ChatResponse)
async def endpoint_chat(request: ChatRequest, http_request: Request):
    """
    Chatea con el agente pasándole un thread_id para mantener la memoria conversacional.
    """
    inicio = time.time()
    try:
        resultado = await run_in_threadpool(
            ejecutar_turno,
            http_request.app.state.agente,
            request.thread_id,
            request.mensaje,
            request.perfil,
            "texto",
            [request.perfil],
        )
        respuesta = _texto_respuesta(resultado)

        # Persistimos ambos lados del intercambio (sin audio, este endpoint es texto).
        db.guardar_turno(request.thread_id, "usuario", request.mensaje)
        db.guardar_turno(request.thread_id, "agente", respuesta)

        return ChatResponse(
            respuesta=respuesta,
            thread_id=request.thread_id,
            pasos=_pasos(resultado),
            fuentes=_fuentes(resultado),
            candidatos=_candidatos(resultado),
            archivos=_archivos(resultado),
            tiempo_ms=round((time.time() - inicio) * 1000, 1),
        )
    except Exception as e:
        if es_error_de_cuota(e):
            # El frontend solo muestra un texto genérico ante un error HTTP; con 200 el usuario ve este mensaje.
            # No se guarda el turno: el usuario puede reintentar el mismo mensaje.
            logger.warning("Cuota de Gemini agotada en /chat: %s", str(e)[:200])
            return ChatResponse(
                respuesta=MENSAJE_SERVICIO_SATURADO,
                thread_id=request.thread_id,
                pasos=[],
                tiempo_ms=round((time.time() - inicio) * 1000, 1),
            )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-voz", response_model=ChatVozResponse, tags=["voz"])
async def endpoint_chat_voz(
    http_request: Request,
    audio: UploadFile = File(...),
    thread_id: str = Form(default="default_user"),
    perfil: str = Form(default="P0"),
):
    """Recibe audio, lo transcribe, lo pasa al agente SIGEM y responde en texto + audio.

    Ambos audios (la pregunta grabada y la respuesta sintetizada) se suben a S3
    y se guardan junto con el texto en Postgres, para que se puedan volver a
    escuchar más tarde vía /historial/{thread_id}.
    """
    extension = (audio.filename or "").split(".")[-1].lower()
    if extension not in config.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(415, "Formato de audio no permitido.")

    data = await audio.read(config.MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(400, "El audio está vacío.")
    if len(data) > config.MAX_AUDIO_BYTES:
        raise HTTPException(413, f"El audio supera el límite de {config.MAX_AUDIO_BYTES // (1024 * 1024)} MB.")

    transcripcion = ""
    try:
        transcripcion = transcribe(data, audio.content_type or "audio/webm")

        # Silencio o envío accidental: no se gasta una llamada al agente ni se ensucia el historial del hilo.
        if len(transcripcion.strip()) < LONGITUD_MINIMA_VOZ:
            return _respuesta_alerta(data, audio.content_type, extension, thread_id, transcripcion, MENSAJE_AUDIO_POCO_CLARO)

        resultado = await run_in_threadpool(
            ejecutar_turno,
            http_request.app.state.agente,
            thread_id,
            transcripcion,
            perfil,
            "voz",
            [perfil, "voz"],
        )
        respuesta_texto = _texto_respuesta(resultado)

        # US-09: si el analista determinó que el audio no es comprensible (ruido, palabras sueltas),
        # avisamos al usuario sin sintetizar audio de respuesta.
        if resultado.get("es_alerta"):
            return _respuesta_alerta(data, audio.content_type, extension, thread_id, transcripcion, respuesta_texto)

        # Solo la voz va sin formato; el texto devuelto y guardado conserva negrilla y listas para pintarlas.
        audio_respuesta = synthesize(limpiar_formato_para_voz(respuesta_texto))

        # Subimos ambos audios a S3 (guardamos la key, no la URL: el bucket es privado).
        pregunta_key = storage.subir_audio(data, audio.content_type or "audio/webm", extension, thread_id)
        respuesta_key = storage.subir_audio(audio_respuesta, "audio/wav", "wav", thread_id)

        db.guardar_turno(thread_id, "usuario", transcripcion, audio_key=pregunta_key)
        db.guardar_turno(thread_id, "agente", respuesta_texto, audio_key=respuesta_key)

    except HTTPException:
        raise
    except Exception as e:
        if es_error_de_cuota(e):
            logger.warning("Cuota de Gemini agotada en /chat-voz: %s", str(e)[:200])
            return ChatVozResponse(
                transcripcion=transcripcion,
                respuesta=MENSAJE_SERVICIO_SATURADO,
                thread_id=thread_id,
                pregunta_audio_url=None,
                respuesta_audio_url=None,
                alerta=True,
            )
        raise HTTPException(502, f"No fue posible procesar el turno de voz: {e}")

    return ChatVozResponse(
        transcripcion=transcripcion,
        respuesta=respuesta_texto,
        thread_id=thread_id,
        pregunta_audio_url=storage.url_firmada(pregunta_key),
        respuesta_audio_url=storage.url_firmada(respuesta_key),
        fuentes=_fuentes(resultado),
        candidatos=_candidatos(resultado),
        archivos=_archivos(resultado),
    )


def _respuesta_alerta(
    data: bytes, content_type: str | None, extension: str, thread_id: str, transcripcion: str, texto: str
) -> ChatVozResponse:
    """Audio no comprensible: se conserva la pregunta, se guarda el turno con rol `alerta` y no hay TTS."""
    pregunta_key = storage.subir_audio(data, content_type or "audio/webm", extension, thread_id)
    db.guardar_turno(thread_id, "usuario", transcripcion or "[audio no comprensible]", audio_key=pregunta_key)
    db.guardar_turno(thread_id, "alerta", texto)
    return ChatVozResponse(
        transcripcion=transcripcion or "",
        respuesta=texto,
        thread_id=thread_id,
        pregunta_audio_url=storage.url_firmada(pregunta_key),
        respuesta_audio_url=None,
        alerta=True,
    )


@app.get("/historial/{thread_id}", response_model=list[TurnoOut], tags=["voz"])
async def endpoint_historial(thread_id: str):
    """Devuelve el historial completo de un hilo, con URLs firmadas para los audios."""
    turnos = db.obtener_historial(thread_id)
    return [
        TurnoOut(
            rol=turno.rol,
            texto=turno.texto,
            audio_url=storage.url_firmada(turno.audio_key),
            creado_en=turno.creado_en.isoformat(),
        )
        for turno in turnos
    ]

if __name__ == "__main__":
    import uvicorn
    print("Iniciando API del agente SIGEM en http://localhost:8001")
    print("Swagger Docs: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)
