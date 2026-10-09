import time
import re
import unicodedata
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, ToolMessage

from . import db, storage
from .agent import agente_sigem
from .retrieval import obtener_catalogo_documentos
from .audio import synthesize, transcribe
from .config import config
from .contexto import enriquecer_consulta_con_contexto, es_pregunta_de_seguimiento

# ══════════════════════════════════════════════════════════════
# MODELOS PYDANTIC
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


class ChatResponse(BaseModel):
    respuesta: str
    thread_id: str
    pasos: list[PasoAgente]
    fuentes: list[FuenteSigem] = []
    candidatos: list[CandidatoDocumento] = []
    tiempo_ms: float

class ChatVozResponse(BaseModel):
    transcripcion: str
    respuesta: str
    thread_id: str
    pregunta_audio_url: str | None
    respuesta_audio_url: str | None
    fuentes: list[FuenteSigem] = []
    candidatos: list[CandidatoDocumento] = []

class TurnoOut(BaseModel):
    rol: str
    texto: str
    audio_url: str | None
    creado_en: str


RESPUESTA_SALUDO = (
    "¡Hola! Soy el asistente virtual de la Alcaldía de Marinilla. "
    "Puedo ayudarte con información disponible en los documentos de SIGEM. "
    "Cuéntame qué trámite, norma, formato o proceso quieres consultar."
)

RESPUESTA_FUERA_DE_CONTEXTO = (
    "Soy especialista en el conocimiento de los documentos de SIGEM de la Alcaldía de Marinilla. "
    "No puedo responder sobre temas que estén por fuera de ese repositorio. "
    "Si tienes una pregunta sobre trámites, normas, formatos o procesos municipales, con gusto te ayudo."
)

RESPUESTA_CONSULTA_AMBIGUA = (
    "Claro, puedo ayudarte a buscar en los documentos de SIGEM. "
    "Para orientarte mejor, dime algún dato adicional: el tema, número, año, dependencia "
    "o trámite relacionado con el documento que necesitas."
)

PALABRAS_SALUDO = {
    "hola", "buenas", "buenos", "dias", "tardes", "noches", "saludos",
    "hey", "ola", "que tal", "como estas", "como vas",
}

PALABRAS_CONSULTA_GENERICA = {
    "busco", "buscar", "necesito", "quiero", "consultar", "consulta", "informacion",
    "dame", "dar", "saber", "ver", "encontrar", "un", "una", "el", "la", "sobre",
    "tramite", "tramites", "proceso", "procesos", "procedimiento", "procedimientos",
    "formato", "formatos", "documento", "documentos", "norma", "normas",
    "normativa", "ley", "leyes", "decreto", "decretos", "resolucion", "resoluciones",
    "acuerdo", "acuerdos", "clausula", "clausulas",
}

PALABRAS_DOMINIO_SIGEM = {
    "sigem", "marinilla", "alcaldia", "municipio", "municipal", "ciudadano",
    "tramite", "tramites", "proceso", "procesos", "procedimiento", "procedimientos",
    "formato", "formatos", "documento", "documentos", "norma", "normas",
    "normativa", "ley", "leyes", "decreto", "decretos", "resolucion", "resoluciones",
    "acuerdo", "acuerdos", "clausula", "clausulas", "requisito", "requisitos",
    "permiso", "permisos", "solicitud", "solicitudes", "certificado", "certificados",
    "impuesto", "impuestos", "catastro", "predial", "movilidad", "vial",
    "contratacion", "licencia", "licencias", "peticion", "pqrs", "queja",
    "reclamo", "reclamos", "alcalde", "secretaria", "dependencia",
}


def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texto).strip()


def respuesta_directa_si_aplica(mensaje: str) -> str | None:
    texto = normalizar_texto(mensaje)
    palabras = set(re.findall(r"[a-z0-9]+", texto))

    if not texto:
        return RESPUESTA_SALUDO

    es_saludo_corto = len(palabras) <= 4 and any(saludo in texto for saludo in PALABRAS_SALUDO)
    if es_saludo_corto:
        return RESPUESTA_SALUDO

    tiene_contexto_sigem = any(palabra in palabras for palabra in PALABRAS_DOMINIO_SIGEM)
    if not tiene_contexto_sigem:
        return RESPUESTA_FUERA_DE_CONTEXTO

    consulta_muy_generica = len(palabras) <= 5 and palabras.issubset(PALABRAS_CONSULTA_GENERICA)
    if consulta_muy_generica:
        return RESPUESTA_CONSULTA_AMBIGUA

    return None


def respuesta_chat_directa(request: ChatRequest, inicio: float, respuesta: str) -> ChatResponse:
    db.guardar_turno(request.thread_id, "usuario", request.mensaje)
    db.guardar_turno(request.thread_id, "agente", respuesta)
    return ChatResponse(
        respuesta=respuesta,
        thread_id=request.thread_id,
        pasos=[],
        fuentes=[],
        tiempo_ms=round((time.time() - inicio) * 1000, 1),
    )


def fuentes_del_ultimo_turno(mensajes) -> list[FuenteSigem]:
    """Extrae las fuentes de las herramientas ejecutadas después de la última pregunta."""
    ultimo_usuario = max(
        (indice for indice, mensaje in enumerate(mensajes) if isinstance(mensaje, HumanMessage)),
        default=-1,
    )
    fuentes = []
    vistas = set()
    for mensaje in mensajes[ultimo_usuario + 1:]:
        if not isinstance(mensaje, ToolMessage):
            continue
        try:
            contenido = json.loads(str(mensaje.content))
        except (TypeError, json.JSONDecodeError):
            continue
        for fuente in contenido.get("fuentes", []):
            fuente_sigem = FuenteSigem(**fuente)
            clave = (fuente_sigem.archivo, fuente_sigem.pagina)
            if clave not in vistas:
                vistas.add(clave)
                fuentes.append(fuente_sigem)
    return fuentes


def candidatos_del_ultimo_turno(mensajes) -> list[CandidatoDocumento]:
    candidatos = []
    vistos = set()
    for mensaje in mensajes:
        if not isinstance(mensaje, ToolMessage):
            continue
        try:
            contenido = json.loads(str(mensaje.content))
        except (TypeError, json.JSONDecodeError):
            continue
        for candidato in contenido.get("candidatos", []):
            candidato_documento = CandidatoDocumento(**candidato)
            if candidato_documento.archivo not in vistos:
                vistos.add(candidato_documento.archivo)
                candidatos.append(candidato_documento)
    return candidatos


def guardar_fuentes_como_contexto(thread_id: str, fuentes: list[FuenteSigem]) -> None:
    if fuentes:
        db.guardar_contexto_sigem(thread_id, [fuente.model_dump() for fuente in fuentes])


def mensaje_para_agente(thread_id: str, mensaje: str) -> tuple[str, bool]:
    fuentes_activas = db.obtener_contexto_sigem(thread_id)
    if es_pregunta_de_seguimiento(mensaje, fuentes_activas):
        return enriquecer_consulta_con_contexto(mensaje, fuentes_activas), True
    return mensaje, False

# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Agente de respuesta ciudadana - SIGEM",
    description="API del agente RAG (LangGraph + Ollama) que responde preguntas sobre procesos, formatos y normatividad del municipio de Marinilla, con base en el repositorio SIGEM.",
    version="0.1.0",
)


@app.on_event("startup")
def crear_tablas():
    """Crea la tabla de historial en Postgres si todavía no existe."""
    db.init_db()


@app.get("/health", tags=["sistema"])
async def health_check():
    """Indica si el proceso del API está disponible."""
    return {"status": "ok"}


@app.get("/health/llm", tags=["sistema"])
def health_check_llm():
    """Comprueba el acceso al servidor Ollama y al modelo configurado."""
    headers = {}
    if config.OLLAMA_PROXY_TOKEN:
        headers["X-Ollama-Proxy-Token"] = config.OLLAMA_PROXY_TOKEN

    try:
        request = Request(f"{config.OLLAMA_BASE_URL}/api/tags", headers=headers)
        with urlopen(request, timeout=config.OLLAMA_HEALTH_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except HTTPError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama respondió HTTP {error.code}. Revisa la URL y el token del proxy.",
        ) from error
    except URLError as error:
        raise HTTPException(
            status_code=503,
            detail="No fue posible conectar con Ollama. Revisa el túnel de Colab y OLLAMA_BASE_URL.",
        ) from error

    installed_models = {model.get("name") for model in payload.get("models", [])}
    if config.CHAT_MODEL not in installed_models:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama está disponible, pero no tiene instalado el modelo {config.CHAT_MODEL}.",
        )

    return {"status": "ok", "model": config.CHAT_MODEL}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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

@app.post("/chat", response_model=ChatResponse)
async def endpoint_chat(request: ChatRequest):
    """
    Chatea con el agente pasándole un thread_id para mantener la memoria conversacional.
    """
    inicio = time.time()
    try:
        mensaje_agente, es_seguimiento = mensaje_para_agente(request.thread_id, request.mensaje)
        respuesta_directa = None if es_seguimiento else respuesta_directa_si_aplica(request.mensaje)
        if respuesta_directa:
            return respuesta_chat_directa(request, inicio, respuesta_directa)

        config_graph = {"configurable": {"thread_id": request.thread_id}}
        
        resultado = agente_sigem.invoke(
            {"messages": [HumanMessage(content=mensaje_agente)],
             "perfil":request.perfil},
            config=config_graph
        )
        
        respuesta = resultado["messages"][-1].content
        fuentes = fuentes_del_ultimo_turno(resultado["messages"])
        candidatos = candidatos_del_ultimo_turno(resultado["messages"])
        guardar_fuentes_como_contexto(request.thread_id, fuentes)
        
        pasos = []
        for msg in resultado["messages"]:
            if type(msg).__name__ == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    pasos.append(PasoAgente(
                        agente="herramienta",
                        accion=f"Buscando en SIGEM: {tc.get('args', {}).get('consulta', '')}",
                        icono="🔎"
                    ))
        
        tiempo_ms = (time.time() - inicio) * 1000

        # Persistimos ambos lados del intercambio (sin audio, este endpoint es texto).
        db.guardar_turno(request.thread_id, "usuario", request.mensaje)
        db.guardar_turno(request.thread_id, "agente", respuesta)

        return ChatResponse(
            respuesta=respuesta,
            thread_id=request.thread_id,
            pasos=pasos,
            fuentes=fuentes,
            candidatos=candidatos,
            tiempo_ms=round(tiempo_ms, 1)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-voz", response_model=ChatVozResponse, tags=["voz"])
async def endpoint_chat_voz(audio: UploadFile = File(...), thread_id: str = Form(default="default_user"), perfil: str = Form(default="P0")):
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

    try:
        transcripcion = transcribe(data, audio.content_type or "audio/webm")
        if not transcripcion:
            raise HTTPException(400, "No se detectó texto en el audio.")

        mensaje_agente, _ = mensaje_para_agente(thread_id, transcripcion)
        config_graph = {"configurable": {"thread_id": thread_id}}
        resultado = agente_sigem.invoke(
            {"messages": [HumanMessage(content=mensaje_agente)],
             "perfil": perfil},
            config=config_graph,    
        )
        respuesta_texto = resultado["messages"][-1].content
        fuentes = fuentes_del_ultimo_turno(resultado["messages"])
        candidatos = candidatos_del_ultimo_turno(resultado["messages"])
        guardar_fuentes_como_contexto(thread_id, fuentes)

        audio_respuesta = synthesize(respuesta_texto)

        # Subimos ambos audios a S3 (guardamos la key, no la URL: el bucket es privado).
        pregunta_key = storage.subir_audio(data, audio.content_type or "audio/webm", extension, thread_id)
        respuesta_key = storage.subir_audio(audio_respuesta, "audio/wav", "wav", thread_id)

        db.guardar_turno(thread_id, "usuario", transcripcion, audio_key=pregunta_key)
        db.guardar_turno(thread_id, "agente", respuesta_texto, audio_key=respuesta_key)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"No fue posible procesar el turno de voz: {e}")

    return ChatVozResponse(
        transcripcion=transcripcion,
        respuesta=respuesta_texto,
        thread_id=thread_id,
        pregunta_audio_url=storage.url_firmada(pregunta_key),
        respuesta_audio_url=storage.url_firmada(respuesta_key),
        fuentes=fuentes,
        candidatos=candidatos,
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
