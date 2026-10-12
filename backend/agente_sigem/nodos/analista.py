"""Nodo analista: entiende la pregunta y decide la ruta (salida estructurada)."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import config as configuracion
from ..estado import AnalisisConsulta
from ..llm import crear_llm
from ..prompts import PROMPT_ANALISTA
from ._comun import paso, texto_de, ultimo_mensaje_usuario

logger = logging.getLogger(__name__)

LONGITUD_MINIMA_VOZ = 3
INTENCIONES_CON_BUSQUEDA = ("consulta_sigem", "seguimiento")


def _transcripcion(mensajes) -> str:
    lineas = []
    for mensaje in mensajes:
        rol = "Usuario" if mensaje.type == "human" else "Asistente"
        lineas.append(f"{rol}: {texto_de(mensaje)}")
    return "\n".join(lineas) if lineas else "(sin historial)"


def _descripcion_documento(documento: dict | None) -> str:
    if not documento:
        return "ninguno"
    descripcion = documento.get("descripcion")
    return f"{documento.get('titulo', '')}" + (f" — {descripcion}" if descripcion else "")


def _entrada_para_llm(estado: dict, mensaje: str) -> str:
    # El mensaje actual ya es el último de `messages`; el historial va sin él.
    mensajes = list(estado.get("messages", []))
    historial = mensajes[:-1][-configuracion.MAX_HISTORIAL_ANALISTA:] if mensajes else []
    return (
        f"HISTORIAL RECIENTE:\n{_transcripcion(historial)}\n\n"
        f"DOCUMENTO ACTIVO: {_descripcion_documento(estado.get('documento_activo'))}\n"
        f"ORIGEN: {estado.get('origen', 'texto')}\n\n"
        f"MENSAJE ACTUAL: {mensaje}"
    )


def _fail_open(mensaje: str, motivo: str) -> AnalisisConsulta:
    return AnalisisConsulta(
        intencion="consulta_sigem",
        consulta_autonoma=mensaje,
        razonamiento=f"Análisis no disponible ({motivo}); se busca con el mensaje original.",
    )


def _coherente(analisis: AnalisisConsulta, tiene_documento_activo: bool) -> AnalisisConsulta:
    """Sin documento activo no puede haber seguimiento ni uso del documento activo."""
    if tiene_documento_activo:
        return analisis
    if analisis.intencion == "seguimiento":
        analisis = analisis.model_copy(update={"intencion": "consulta_sigem"})
    if analisis.usa_documento_activo:
        analisis = analisis.model_copy(update={"usa_documento_activo": False})
    return analisis


def analizar(estado: dict, config: RunnableConfig | None = None) -> AnalisisConsulta:
    mensaje = ultimo_mensaje_usuario(estado)

    # Atajo determinístico: una transcripción casi vacía no merece una llamada al LLM.
    if estado.get("origen") == "voz" and len(mensaje.strip()) < LONGITUD_MINIMA_VOZ:
        return AnalisisConsulta(
            intencion="no_comprensible",
            consulta_autonoma=mensaje,
            razonamiento="Transcripción de voz con menos de 3 caracteres.",
        )

    try:
        llm = crear_llm(configuracion.ANALISTA_MODEL, 0).with_structured_output(AnalisisConsulta)
        salida = llm.invoke(
            [SystemMessage(content=PROMPT_ANALISTA), HumanMessage(content=_entrada_para_llm(estado, mensaje))],
            config,
        )
        if isinstance(salida, dict):
            salida = AnalisisConsulta(**salida)
        if not isinstance(salida, AnalisisConsulta):
            raise ValueError(f"Salida inesperada del analista: {type(salida).__name__}")
        if not salida.consulta_autonoma.strip():
            salida = salida.model_copy(update={"consulta_autonoma": mensaje})
        return _coherente(salida, bool(estado.get("documento_activo")))
    except Exception as error:
        # Fail-open: un error nuestro nunca bloquea al usuario.
        logger.warning("El analista falló; se continúa como consulta_sigem: %s", error)
        return _fail_open(mensaje, type(error).__name__)


def analista(estado: dict, config: RunnableConfig | None = None) -> dict:
    analisis = analizar(estado, config)
    return {
        "analisis": analisis.model_dump(),
        "consulta_actual": analisis.consulta_autonoma,
        "pasos": estado.get("pasos", []) + [
            paso("analista", f"Analizando la pregunta: {analisis.intencion}", "🧭")
        ],
    }


def ruta_despues_analista(estado: dict) -> str:
    """'investigador' si hay que buscar en SIGEM; 'respuesta_directa' en los demás casos."""
    analisis = estado.get("analisis") or {}
    if analisis.get("intencion", "consulta_sigem") in INTENCIONES_CON_BUSQUEDA:
        return "investigador"
    return "respuesta_directa"
