"""Nodo evaluador (Corrective RAG): decide si la evidencia basta o hay que reintentar."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import config as configuracion
from ..estado import PREFIJO_REINTENTO, EvaluacionEvidencia
from ..llm import crear_llm
from ..prompts import PROMPT_EVALUADOR
from ._comun import paso

logger = logging.getLogger(__name__)

MAX_FRAGMENTOS_EVALUACION = 8
PASO_EVALUACION_NO_DISPONIBLE = "Evaluación no disponible: se continúa con la evidencia recuperada"
MAX_CARACTERES_FRAGMENTO = 600


def _resumen_evidencia(evidencia: list[dict]) -> str:
    bloques = []
    for numero, fragmento in enumerate(evidencia[:MAX_FRAGMENTOS_EVALUACION], start=1):
        fuente = fragmento.get("fuente") or {}
        pagina = f", p. {fuente['pagina']}" if fuente.get("pagina") else ""
        contenido = str(fragmento.get("contenido", ""))[:MAX_CARACTERES_FRAGMENTO]
        bloques.append(f"[{numero}] {fuente.get('titulo', 'Documento')}{pagina}\n{contenido}")
    return "\n\n".join(bloques) if bloques else "(sin fragmentos)"


def _evaluar(estado: dict, config: RunnableConfig | None) -> EvaluacionEvidencia:
    analisis = estado.get("analisis") or {}
    entrada = (
        f"PREGUNTA DEL USUARIO: {analisis.get('consulta_autonoma') or estado.get('consulta_actual', '')}\n"
        f"CONSULTA USADA EN LA BÚSQUEDA: {estado.get('consulta_actual', '')}\n"
        f"INTENTO NÚMERO: {estado.get('intentos_busqueda', 0)} de {configuracion.MAX_INTENTOS_BUSQUEDA}\n\n"
        f"EVIDENCIA RECUPERADA:\n{_resumen_evidencia(estado.get('evidencia', []))}"
    )
    llm = crear_llm(configuracion.EVALUADOR_MODEL, 0).with_structured_output(EvaluacionEvidencia)
    salida = llm.invoke([SystemMessage(content=PROMPT_EVALUADOR), HumanMessage(content=entrada)], config)
    if isinstance(salida, dict):
        salida = EvaluacionEvidencia(**salida)
    if not isinstance(salida, EvaluacionEvidencia):
        raise ValueError(f"Salida inesperada del evaluador: {type(salida).__name__}")
    return salida


def evaluador(estado: dict, config: RunnableConfig | None = None) -> dict:
    evidencia = estado.get("evidencia", [])
    candidatos = estado.get("candidatos", [])
    intentos = estado.get("intentos_busqueda", 0)
    pasos = estado.get("pasos", [])

    def resultado(accion: str, **cambios) -> dict:
        return {"pasos": pasos + [paso("evaluador", accion, "✅")], **cambios}

    # Atajos sin LLM.
    if candidatos and not evidencia:
        return resultado("Hay varios documentos posibles: se pedirá aclaración")
    if (estado.get("archivos") or estado.get("archivos_no_disponibles")) and not evidencia:
        return resultado("Archivo listo para entregar" if estado.get("archivos") else "El original no está disponible para descarga")
    if not evidencia and intentos >= configuracion.MAX_INTENTOS_BUSQUEDA:
        return resultado("Sin resultados tras agotar los intentos de búsqueda")
    if not configuracion.EVALUADOR_HABILITADO:
        return resultado("Evaluación desactivada")

    try:
        evaluacion = _evaluar(estado, config)
    except Exception as error:
        # Fail-open: ante un error se redacta con lo que hay.
        logger.warning("El evaluador falló; se continúa al redactor: %s", error)
        return resultado(PASO_EVALUACION_NO_DISPONIBLE)

    if evaluacion.suficiente or intentos >= configuracion.MAX_INTENTOS_BUSQUEDA:
        return resultado("Evidencia suficiente")

    # Reintento: el investigador incrementa `intentos_busqueda` al ejecutarse.
    reformulada = (evaluacion.consulta_reformulada or "").strip()
    return resultado(
        f"{PREFIJO_REINTENTO}{evaluacion.motivo}",
        consulta_actual=reformulada or estado.get("consulta_actual", ""),
    )


def ruta_despues_evaluador(estado: dict) -> str:
    """'investigador' si el evaluador pidió un reintento; 'redactor' en los demás casos."""
    pasos = estado.get("pasos", [])
    if pasos and pasos[-1].get("agente") == "evaluador" and pasos[-1].get("accion", "").startswith(PREFIJO_REINTENTO):
        return "investigador"
    return "redactor"
