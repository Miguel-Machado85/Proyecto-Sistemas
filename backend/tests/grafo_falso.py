"""Instala LLM falsos y tools dobles en los nodos para probar el grafo completo."""

import importlib

from langchain_core.messages import AIMessage

from agente_sigem import tools
from agente_sigem.estado import AnalisisConsulta, EvaluacionEvidencia
from fakes_llm import FakeEstructurado, ModeloFalso, instalar, llamada

M_ANALISTA = importlib.import_module("agente_sigem.nodos.analista")
M_INVESTIGADOR = importlib.import_module("agente_sigem.nodos.investigador")
M_EVALUADOR = importlib.import_module("agente_sigem.nodos.evaluador")
M_REDACTOR = importlib.import_module("agente_sigem.nodos.redactor")

__all__ = [
    "AIMessage", "AnalisisConsulta", "EvaluacionEvidencia", "FakeEstructurado", "ModeloFalso",
    "M_ANALISTA", "M_EVALUADOR", "M_INVESTIGADOR", "M_REDACTOR", "instalar_grafo", "llamada",
]


def instalar_grafo(monkeypatch, analisis, evaluaciones=None, investigador_msgs=None, redactor_msgs=None, salidas_tools=None):
    """Sustituye los LLM de los 4 nodos con LLM y las tools del investigador.

    Devuelve un dict con los fakes para poder inspeccionar qué recibió cada uno.
    """
    fakes = {
        "analista": FakeEstructurado(analisis),
        "evaluador": FakeEstructurado(evaluaciones or [EvaluacionEvidencia(suficiente=True, motivo="ok")]),
        "investigador": ModeloFalso(respuestas=list(investigador_msgs or [])),
        "redactor": ModeloFalso(respuestas=list(redactor_msgs or [])),
    }
    instalar(monkeypatch, M_ANALISTA, fakes["analista"])
    instalar(monkeypatch, M_EVALUADOR, fakes["evaluador"])
    instalar(monkeypatch, M_INVESTIGADOR, fakes["investigador"])
    instalar(monkeypatch, M_REDACTOR, fakes["redactor"])

    from fakes_llm import herramientas_dobles

    monkeypatch.setattr(tools, "HERRAMIENTAS_INVESTIGADOR", herramientas_dobles(salidas_tools or {"buscar_fragmentos": {}}))
    return fakes
