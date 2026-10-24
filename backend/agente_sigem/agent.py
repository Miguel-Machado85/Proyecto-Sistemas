"""Grafo multi-agente de SIGEM.

START → analista → (respuesta_directa | investigador) → evaluador → (investigador | redactor) → END

El grafo NO se compila al importar este módulo: la API lo construye en su lifespan con el
checkpointer que corresponda (ver api.py y checkpointer.py).
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .estado import EstadoAgente
from .nodos.analista import analista, ruta_despues_analista
from .nodos.evaluador import evaluador, ruta_despues_evaluador
from .nodos.investigador import investigador
from .nodos.redactor import redactor
from .nodos.respuesta_directa import respuesta_directa


def construir_grafo(checkpointer: BaseCheckpointSaver | None = None):
    """Arma y compila el grafo. Con `checkpointer=None` no hay memoria entre turnos."""
    grafo = StateGraph(EstadoAgente)

    grafo.add_node("analista", analista)
    grafo.add_node("respuesta_directa", respuesta_directa)
    grafo.add_node("investigador", investigador)
    grafo.add_node("evaluador", evaluador)
    grafo.add_node("redactor", redactor)

    grafo.add_edge(START, "analista")
    grafo.add_conditional_edges(
        "analista",
        ruta_despues_analista,
        {"respuesta_directa": "respuesta_directa", "investigador": "investigador"},
    )
    grafo.add_edge("respuesta_directa", END)

    grafo.add_edge("investigador", "evaluador")
    grafo.add_conditional_edges(
        "evaluador",
        ruta_despues_evaluador,
        {"investigador": "investigador", "redactor": "redactor"},
    )
    grafo.add_edge("redactor", END)

    return grafo.compile(checkpointer=checkpointer)
