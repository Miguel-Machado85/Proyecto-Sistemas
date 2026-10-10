"""Utilidades compartidas por los nodos del grafo."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def texto_de(mensaje: BaseMessage) -> str:
    """Texto plano de un mensaje (Gemini devuelve el contenido como lista de bloques)."""
    return mensaje.text if hasattr(mensaje, "text") else str(mensaje.content)


def ultimo_mensaje_usuario(estado: dict) -> str:
    """El mensaje original del usuario en este turno (nunca reescrito)."""
    for mensaje in reversed(estado.get("messages", [])):
        if isinstance(mensaje, HumanMessage):
            return texto_de(mensaje)
    return ""


def paso(agente: str, accion: str, icono: str) -> dict:
    return {"agente": agente, "accion": accion, "icono": icono}


def respuesta_ai(texto: str) -> AIMessage:
    return AIMessage(content=texto)
