from typing import Annotated
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import config

# ══════════════════════════════════════════════════════════════
# RETRIEVER (Herramienta)
# ══════════════════════════════════════════════════════════════

def obtener_retriever():
    """Conecta a la base vectorial de Chroma (la administra el compañero de datos) y retorna el retriever."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )

    
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    return vector_store.as_retriever(search_kwargs={"k": 3})

@tool
def buscar_en_sigem(consulta: str) -> str:
    """
    Busca información en el repositorio SIGEM: procesos, formatos y normatividad
    de la Alcaldía de Marinilla. Usa esta herramienta SIEMPRE que te pregunten
    sobre un trámite, un documento oficial, un formato o una norma municipal.
    """
    try:
        retriever = obtener_retriever()
        resultados = retriever.invoke(consulta)

        if not resultados:
            return "No se encontró información relevante en el repositorio SIGEM."

        contexto = []
        for doc in resultados:
            origen = doc.metadata.get("source", "Documento SIGEM sin identificar")
            contexto.append(f"--- Información de {origen} ---\n{doc.page_content}")

        return "\n\n".join(contexto)
    except Exception as e:
        return f"Error al buscar en SIGEM: {str(e)}"

# ══════════════════════════════════════════════════════════════
# AGENTE CON MEMORIA (LangGraph + Ollama)
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
"""

ESTILO_POR_PERFIL = {
    "P0": """
ESTILO DE RESPUESTA (Perfil 0 - Neutro):
Usa un lenguaje intermedio: ni muy técnico ni muy básico. Responde en texto, de forma clara y respetuosa, explicando brevemente cualquier término que no sea de uso común.""",

    "P1": """
ESTILO DE RESPUESTA (Perfil 1 - Sin experiencia digital):
Usa frases muy cortas y lenguaje muy sencillo. No uses tecnicismos ni jerga jurídica. Evita párrafos largos: prefiere listas de pasos simples. Explica todo como si fuera la primera vez que la persona hace un trámite.""",

    "P2": """
ESTILO DE RESPUESTA (Perfil 2 - Ciudadano general):
Usa lenguaje sencillo y cercano. Si necesitas usar un término legal o técnico, explícalo brevemente antes de usarlo. Responde en texto, de forma clara y organizada.""",

    "P3": """
ESTILO DE RESPUESTA (Perfil 3 - Funcionario/técnico):
Puedes usar lenguaje técnico y citar artículos o normas directamente cuando el repositorio SIGEM los mencione. No es necesario explicar términos jurídicos básicos.""",
}

def construir_system_prompt(perfil: str) -> str:
    estilo = ESTILO_POR_PERFIL.get(perfil, ESTILO_POR_PERFIL["P0"])
    return INSTRUCCIONES_BASE + estilo


def crear_agente():
    """
    Crea el agente RAG usando el modelo de Ollama configurado.
    """
    modelo = ChatOllama(
        model=config.CHAT_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.3,
    )
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