"""Nodo respuesta_directa: saludo, fuera de dominio, ambigua y audio no comprensible (sin LLM)."""

from langchain_core.runnables import RunnableConfig

from ..prompts import (
    MENSAJE_AUDIO_POCO_CLARO,
    RESPUESTA_CONSULTA_AMBIGUA,
    RESPUESTA_FUERA_DE_CONTEXTO,
    RESPUESTA_SALUDO,
)
from ._comun import paso, respuesta_ai


def respuesta_directa(estado: dict, config: RunnableConfig | None = None) -> dict:
    analisis = estado.get("analisis") or {}
    intencion = analisis.get("intencion")

    es_alerta = False
    if intencion == "saludo":
        texto = RESPUESTA_SALUDO
    elif intencion == "fuera_de_dominio":
        texto = RESPUESTA_FUERA_DE_CONTEXTO
    elif intencion == "no_comprensible":
        texto = MENSAJE_AUDIO_POCO_CLARO
        es_alerta = True
    else:  # "ambigua" (o cualquier valor inesperado)
        texto = analisis.get("pregunta_aclaracion") or RESPUESTA_CONSULTA_AMBIGUA

    return {
        "messages": [respuesta_ai(texto)],
        "es_alerta": es_alerta,
        "pasos": estado.get("pasos", []) + [paso("respuesta_directa", f"Respuesta directa: {intencion}", "💬")],
    }
