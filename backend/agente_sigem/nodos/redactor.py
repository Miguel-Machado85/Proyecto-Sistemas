"""Nodo redactor: arma la respuesta final al ciudadano según el perfil y la evidencia."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import config as configuracion
from ..llm import crear_llm
from ..prompts import construir_prompt_redactor
from ._comun import paso, respuesta_ai, texto_de, ultimo_mensaje_usuario

MAX_FRAGMENTOS_REDACCION = 12


def elegir_modo(estado: dict) -> str:
    """respuesta (hay evidencia) | aclaracion (solo candidatos) | sin_resultados."""
    if estado.get("evidencia"):
        return "respuesta"
    if estado.get("candidatos"):
        return "aclaracion"
    return "sin_resultados"


def _evidencia_numerada(evidencia: list[dict]) -> str:
    bloques = []
    for numero, fragmento in enumerate(evidencia[:MAX_FRAGMENTOS_REDACCION], start=1):
        fuente = fragmento.get("fuente") or {}
        pagina = f", p. {fuente['pagina']}" if fuente.get("pagina") else ""
        bloques.append(f"[{numero}] {fuente.get('titulo', 'Documento SIGEM')}{pagina}\n{fragmento.get('contenido', '')}")
    return "\n\n".join(bloques)


def _candidatos_en_texto(candidatos: list[dict]) -> str:
    return "\n".join(
        f"- {c.get('titulo', 'Documento')} ({c.get('fecha_expedicion') or 'sin fecha'})"
        + (f": {c['descripcion']}" if c.get("descripcion") else "")
        for c in candidatos
    )


def construir_entrada(estado: dict, modo: str) -> str:
    analisis = estado.get("analisis") or {}
    partes = [
        f"PREGUNTA ORIGINAL DEL USUARIO: {ultimo_mensaje_usuario(estado)}",
        f"PREGUNTA ENTENDIDA: {analisis.get('consulta_autonoma') or estado.get('consulta_actual', '')}",
    ]
    documento = estado.get("documento_activo")
    if analisis.get("usa_documento_activo") and documento:
        partes.append(f"DOCUMENTO ACTIVO DE LA CONVERSACIÓN: {documento.get('titulo', '')}")

    if modo == "respuesta":
        partes.append("EVIDENCIA:\n" + _evidencia_numerada(estado.get("evidencia", [])))
    elif modo == "aclaracion":
        partes.append("CANDIDATOS:\n" + _candidatos_en_texto(estado.get("candidatos", [])))
    else:
        partes.append("EVIDENCIA: ninguna. No se encontró información pertinente.")
    return "\n\n".join(partes)


def _documento_activo_desde(estado: dict) -> dict | None:
    """Documento de la primera fuente que tiene fragmentos; None si solo hubo listados o no hubo fuentes."""
    for fragmento in estado.get("evidencia", []):
        if fragmento.get("tipo") == "listado":
            continue
        fuente = fragmento.get("fuente") or {}
        if fuente.get("archivo"):
            return {
                "titulo": fuente.get("titulo"),
                "archivo": fuente.get("archivo"),
                "descripcion": fuente.get("descripcion"),
                "fecha_expedicion": fuente.get("fecha_expedicion"),
            }
    return None


def redactor(estado: dict, config: RunnableConfig | None = None) -> dict:
    perfil = estado.get("perfil", "P0")
    modo = elegir_modo(estado)

    llm = crear_llm(configuracion.REDACTOR_MODEL, 0.3)
    respuesta = llm.invoke(
        [
            SystemMessage(content=construir_prompt_redactor(perfil, modo)),
            HumanMessage(content=construir_entrada(estado, modo)),
        ],
        config,
    )

    # Un listado (solo fuentes, sin fragmentos) no convierte ningún documento en "activo".
    documento_activo = _documento_activo_desde(estado) or estado.get("documento_activo")
    return {
        "messages": [respuesta_ai(texto_de(respuesta))],
        "documento_activo": documento_activo,
        "pasos": estado.get("pasos", []) + [paso("redactor", f"Redactando respuesta para perfil {perfil}", "✍️")],
    }
