from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import config
from .llm import crear_llm
from .prompts import ESTILO_POR_PERFIL
from .referencias import (
    candidatos_referencia_ambigua,
    recuperar_con_referencia,
    respuesta_referencia_ambigua,
)
from .retrieval import respuesta_de_recuperacion

# ══════════════════════════════════════════════════════════════
# RETRIEVER (Herramienta)
# ══════════════════════════════════════════════════════════════

@tool
def buscar_en_sigem(consulta: str) -> str:
    """
    Busca información en el repositorio SIGEM: procesos, formatos y normatividad
    de la Alcaldía de Marinilla. Usa esta herramienta SIEMPRE que te pregunten
    sobre un trámite, un documento oficial, un formato o una norma municipal.
    """
    try:
        candidatos = candidatos_referencia_ambigua(consulta)
        if candidatos:
            return respuesta_referencia_ambigua(candidatos)

        resultados = recuperar_con_referencia(consulta)

        if not resultados:
            return "No se encontró información relevante en el repositorio SIGEM."

        return respuesta_de_recuperacion(resultados)
    except Exception as e:
        return f"Error al buscar en SIGEM: {str(e)}"

# ══════════════════════════════════════════════════════════════
# AGENTE CON MEMORIA (LangGraph + Gemini)
# ══════════════════════════════════════════════════════════════

class EstadoAgente(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    perfil: str
INSTRUCCIONES_BASE = """Eres el agente de respuesta ciudadana de la Alcaldía de Marinilla, Antioquia.
Tu función es ayudar a ciudadanos y funcionarios a entender procesos, formatos y normatividad
municipal, usando exclusivamente la información del repositorio SIGEM.

REGLAS IMPORTANTES:
1. Tienes acceso a una herramienta de búsqueda del repositorio SIGEM.
2. Si te preguntan sobre un trámite, un formato, un documento oficial o una norma, SIEMPRE usa la herramienta 'buscar_en_sigem' antes de responder.
3. Recuerda el contexto de la conversación (memoria). Si el usuario hace referencia a algo dicho antes, usa el historial.
4. Si te hacen una pregunta sobre normativa, trámites o formatos, RESPONDE ÚNICA Y EXCLUSIVAMENTE con base en la información que devuelva la herramienta 'buscar_en_sigem'. No completes con conocimiento general ni inventes información.
5. Si la herramienta no devuelve información útil, indícale al usuario que ese contenido no está disponible en SIGEM por el momento, y sugiere que consulte directamente en la Alcaldía de Marinilla.
6. La herramienta devuelve JSON con los campos `fragmentos` y `fuentes`. Usa exclusivamente el contenido de `fragmentos` para responder y no afirmes que no hay información si alguno es pertinente.
7. Si el mensaje indica que es continuación de un documento SIGEM activo, llama obligatoriamente a `buscar_en_sigem` incluyendo el título de ese documento, aunque la pregunta sea corta.
8. Si el usuario pide ampliar, explicar o hablar más sobre el documento activo y la herramienta devuelve fragmentos, sintetiza detalles de esos fragmentos; nunca respondas que no hay contenido relevante.
9. Si la herramienta devuelve `aclaracion` y `candidatos`, pide al usuario que seleccione el año o tema del documento. No respondas usando documentos distintos a esos candidatos.
"""

def construir_system_prompt(perfil: str) -> str:
    estilo = ESTILO_POR_PERFIL.get(perfil, ESTILO_POR_PERFIL["P0"])
    return INSTRUCCIONES_BASE + estilo


def crear_agente():
    """
    Crea el agente RAG usando el modelo de Gemini configurado (INVESTIGADOR_MODEL).
    """
    modelo = crear_llm(config.INVESTIGADOR_MODEL, 0.3)
    herramientas = [buscar_en_sigem]
    modelo_con_tools = modelo.bind_tools(herramientas)

    checkpointer = MemorySaver()

    def nodo_asistente(estado: EstadoAgente):
        mensajes_historial = estado["messages"]
        perfil = estado.get("perfil", "P0")
        system_prompt = construir_system_prompt(perfil)
        respuesta = modelo_con_tools.invoke([SystemMessage(content=system_prompt)] + mensajes_historial)
        return {"messages": [respuesta]}


    def enrutador_herramientas(estado: EstadoAgente):
        ultimo_mensaje = estado["messages"][-1]
        if hasattr(ultimo_mensaje, "tool_calls") and ultimo_mensaje.tool_calls:
            return "tools"
        return END

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("asistente", nodo_asistente)
    grafo.add_node("tools", ToolNode(herramientas))

    grafo.add_edge(START, "asistente")
    grafo.add_conditional_edges("asistente", enrutador_herramientas)
    grafo.add_edge("tools", "asistente")

    agente_compilado = grafo.compile(checkpointer=checkpointer)

    return agente_compilado, checkpointer

# Instancia única
agente_sigem, memoria_checkpointer = crear_agente()
