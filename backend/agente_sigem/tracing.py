"""US-10: observabilidad del agente con Langfuse.

Se engancha vía el sistema de callbacks de LangChain/LangGraph, así que no
hay que instrumentar agent.py a mano: cada nodo del grafo, cada llamada a
Ollama y cada llamada al retriever quedan trazadas automáticamente en cuanto
le pasamos este handler a agente_sigem.invoke(..., config={"callbacks": [...]}).

Igual que con el cliente de Gemini en audio.py, el CallbackHandler se crea
UNA SOLA VEZ (singleton) y se reutiliza, en vez de crearlo por request.
"""
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from .config import config

_handler: CallbackHandler | None = None
_langfuse_inicializado = False


def obtener_langfuse_handler() -> CallbackHandler | None:
    """Devuelve el CallbackHandler de Langfuse, o None si no hay credenciales configuradas.

    Diseñado para degradar con gracia: si LANGFUSE_PUBLIC_KEY/SECRET_KEY no
    están en el .env, el agente sigue funcionando exactamente igual, solo
    que sin mandar traces (útil mientras cada quien del equipo decide si
    configura su propia cuenta de Langfuse o no).
    """
    global _handler, _langfuse_inicializado

    if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
        return None

    if not _langfuse_inicializado:
        get_client()  # inicializa el cliente singleton de Langfuse con las env vars
        _handler = CallbackHandler()
        _langfuse_inicializado = True

    return _handler