"""Flujos del grafo multi-agente (sección 12): grafo real, MemorySaver y LLM falsos."""

import pytest
from fakes_llm import CANDIDATOS, FRAGMENTO_TAXI, FUENTE_TAXI, llamada
from grafo_falso import (
    AIMessage,
    AnalisisConsulta,
    EvaluacionEvidencia,
    FakeEstructurado,
    M_ANALISTA,
    M_EVALUADOR,
    instalar,
    instalar_grafo,
)
from langgraph.checkpoint.memory import MemorySaver

from agente_sigem import api, prompts
from agente_sigem.agent import construir_grafo
from agente_sigem.config import config

SALIDAS_TAXI = {"buscar_fragmentos": {"fuentes": [FUENTE_TAXI], "fragmentos": [FRAGMENTO_TAXI]}}


@pytest.fixture(autouse=True)
def sin_langfuse(monkeypatch):
    monkeypatch.setattr(api, "obtener_langfuse_handler", lambda: None)


@pytest.fixture
def agente():
    return construir_grafo(MemorySaver())


def turno(agente, mensaje, thread_id="hilo", perfil="P0", origen="texto"):
    """Usa la misma función que la API para invocar el grafo (estado por turno reiniciado)."""
    return api.ejecutar_turno(agente, thread_id, mensaje, perfil, origen, [perfil])


def agentes(resultado) -> list[str]:
    return [paso["agente"] for paso in resultado["pasos"]]


def consulta(texto="tarifa del taxi", **extra):
    return AnalisisConsulta(intencion="consulta_sigem", consulta_autonoma=texto, razonamiento="x", **extra)


# ── rutas directas ──

@pytest.mark.parametrize(
    "intencion, texto",
    [
        ("saludo", prompts.RESPUESTA_SALUDO),
        ("fuera_de_dominio", prompts.RESPUESTA_FUERA_DE_CONTEXTO),
    ],
)
def test_saludo_y_fuera_de_dominio_van_a_respuesta_directa(monkeypatch, agente, intencion, texto):
    fakes = instalar_grafo(monkeypatch, [AnalisisConsulta(intencion=intencion, consulta_autonoma="x", razonamiento="x")])

    resultado = turno(agente, "hola")

    assert resultado["messages"][-1].content == texto
    assert agentes(resultado) == ["analista", "respuesta_directa"]
    assert resultado["es_alerta"] is False
    assert fakes["investigador"].entradas == [] and fakes["redactor"].entradas == []


def test_ambigua_responde_con_la_pregunta_de_aclaracion(monkeypatch, agente):
    instalar_grafo(monkeypatch, [AnalisisConsulta(
        intencion="ambigua", consulta_autonoma="x", pregunta_aclaracion="¿De qué año?", razonamiento="x")])

    resultado = turno(agente, "necesito un documento")

    assert resultado["messages"][-1].content == "¿De qué año?"


def test_no_comprensible_marca_es_alerta(monkeypatch, agente):
    instalar_grafo(monkeypatch, [AnalisisConsulta(
        intencion="no_comprensible", consulta_autonoma="eh", razonamiento="ruido")])

    resultado = turno(agente, "eh mm ajá", origen="voz")

    assert resultado["es_alerta"] is True
    assert resultado["messages"][-1].content == prompts.MENSAJE_AUDIO_POCO_CLARO


def test_voz_demasiado_corta_no_llama_al_analista_llm(monkeypatch, agente):
    fakes = instalar_grafo(monkeypatch, [consulta()])

    resultado = turno(agente, "a", origen="voz")

    assert fakes["analista"].llamadas == 0
    assert resultado["es_alerta"] is True


# ── búsqueda ──

def test_consulta_pasa_por_investigador_evaluador_y_redactor(monkeypatch, agente):
    fakes = instalar_grafo(
        monkeypatch,
        analisis=[consulta()],
        investigador_msgs=[llamada("buscar_fragmentos", consulta="tarifa del taxi"), AIMessage(content="Listo.")],
        redactor_msgs=[AIMessage(content="La tarifa mínima es $5.500 [Decreto 001 de 2019, p. 1].")],
        salidas_tools=SALIDAS_TAXI,
    )

    resultado = turno(agente, "¿cuánto cuesta el taxi?")

    assert agentes(resultado) == ["analista", "investigador", "evaluador", "redactor"]
    assert resultado["messages"][-1].content.startswith("La tarifa mínima")
    assert resultado["fuentes"] == [FUENTE_TAXI]
    assert resultado["documento_activo"]["archivo"] == FUENTE_TAXI["archivo"]
    assert "MODO: RESPUESTA" in fakes["redactor"].entradas[0][0].content


def test_evaluador_pide_un_reintento_y_el_investigador_corre_dos_veces(monkeypatch, agente):
    fakes = instalar_grafo(
        monkeypatch,
        analisis=[consulta()],
        evaluaciones=[
            EvaluacionEvidencia(suficiente=False, motivo="otro tema", consulta_reformulada="tarifas de taxi 2019"),
            EvaluacionEvidencia(suficiente=True, motivo="ok"),
        ],
        investigador_msgs=[
            llamada("buscar_fragmentos", consulta="tarifa del taxi"), AIMessage(content="Nada claro."),
            llamada("buscar_fragmentos", consulta="tarifas de taxi 2019"), AIMessage(content="Listo."),
        ],
        redactor_msgs=[AIMessage(content="Respuesta.")],
        salidas_tools=SALIDAS_TAXI,
    )

    resultado = turno(agente, "¿cuánto cuesta el taxi?")

    assert resultado["intentos_busqueda"] == 2
    assert resultado["consulta_actual"] == "tarifas de taxi 2019"
    assert agentes(resultado) == ["analista", "investigador", "evaluador", "investigador", "evaluador", "redactor"]
    assert len(resultado["evidencia"]) == 1  # el mismo fragmento no se duplica entre intentos
    assert "REINTENTO" in fakes["investigador"].entradas[2][1].content  # segundo intento: encargo con motivo


def test_los_reintentos_se_detienen_en_el_maximo(monkeypatch, agente):
    monkeypatch.setattr(config, "MAX_INTENTOS_BUSQUEDA", 2)
    instalar_grafo(
        monkeypatch,
        analisis=[consulta()],
        evaluaciones=[EvaluacionEvidencia(suficiente=False, motivo="nunca basta", consulta_reformulada="otra")],
        investigador_msgs=[AIMessage(content="Nada.")] * 5,
        redactor_msgs=[AIMessage(content="No hay información.")],
        salidas_tools=SALIDAS_TAXI,
    )

    resultado = turno(agente, "¿cuánto cuesta el taxi?")

    assert resultado["intentos_busqueda"] == 2
    # El investigador no hizo llamadas a tools, así que no deja pasos; el evaluador sí: reintento y cierre.
    assert agentes(resultado) == ["analista", "evaluador", "evaluador", "redactor"]
    assert resultado["pasos"][1]["accion"].startswith("Reintentando")
    assert resultado["pasos"][2]["accion"].startswith("Sin resultados")


def test_sin_resultados_redacta_en_modo_sin_resultados(monkeypatch, agente):
    fakes = instalar_grafo(
        monkeypatch,
        analisis=[consulta("tema inexistente")],
        evaluaciones=[EvaluacionEvidencia(suficiente=False, motivo="vacío", consulta_reformulada="otra")],
        investigador_msgs=[AIMessage(content="Nada."), AIMessage(content="Nada.")],
        redactor_msgs=[AIMessage(content="Esa información no está disponible en SIGEM.")],
        salidas_tools={"buscar_fragmentos": {"fuentes": [], "fragmentos": []}},
    )

    resultado = turno(agente, "algo que no existe")

    assert resultado["evidencia"] == [] and resultado["fuentes"] == []
    assert resultado["documento_activo"] is None
    assert "MODO: SIN RESULTADOS" in fakes["redactor"].entradas[0][0].content
    assert resultado["intentos_busqueda"] == config.MAX_INTENTOS_BUSQUEDA


def test_candidatos_llevan_al_redactor_en_modo_aclaracion(monkeypatch, agente):
    fakes = instalar_grafo(
        monkeypatch,
        analisis=[consulta("Resolución 4397")],
        investigador_msgs=[
            llamada("buscar_documento_por_referencia", tipo="resolucion", numero=4397), AIMessage(content="Dos."),
        ],
        redactor_msgs=[AIMessage(content="¿Cuál de las dos resoluciones?")],
        salidas_tools={"buscar_documento_por_referencia": {"fuentes": [], "fragmentos": [], "candidatos": CANDIDATOS}},
    )

    resultado = turno(agente, "Resolución 4397")

    assert resultado["candidatos"] == CANDIDATOS
    assert agentes(resultado) == ["analista", "investigador", "evaluador", "redactor"]
    assert fakes["evaluador"].llamadas == 0  # atajo sin LLM
    assert "MODO: ACLARACIÓN" in fakes["redactor"].entradas[0][0].content


def test_evaluador_deshabilitado_va_directo_al_redactor(monkeypatch, agente):
    monkeypatch.setattr(config, "EVALUADOR_HABILITADO", False)
    fakes = instalar_grafo(
        monkeypatch,
        analisis=[consulta()],
        investigador_msgs=[llamada("buscar_fragmentos", consulta="x"), AIMessage(content="ok")],
        redactor_msgs=[AIMessage(content="Respuesta.")],
        salidas_tools=SALIDAS_TAXI,
    )

    resultado = turno(agente, "¿cuánto cuesta el taxi?")

    assert fakes["evaluador"].llamadas == 0
    assert resultado["messages"][-1].content == "Respuesta."


# ── fail-open ──

def test_analista_con_excepcion_sigue_como_consulta_sigem(monkeypatch, agente):
    instalar_grafo(
        monkeypatch,
        analisis=[consulta()],
        investigador_msgs=[llamada("buscar_fragmentos", consulta="x"), AIMessage(content="ok")],
        redactor_msgs=[AIMessage(content="Respuesta.")],
        salidas_tools=SALIDAS_TAXI,
    )
    instalar(monkeypatch, M_ANALISTA, FakeEstructurado(error=RuntimeError("Gemini caído")))

    resultado = turno(agente, "¿cuánto cuesta el taxi?")

    assert resultado["analisis"]["intencion"] == "consulta_sigem"
    assert resultado["consulta_actual"] == "¿cuánto cuesta el taxi?"
    assert resultado["messages"][-1].content == "Respuesta."


def test_evaluador_con_excepcion_sigue_al_redactor(monkeypatch, agente):
    instalar_grafo(
        monkeypatch,
        analisis=[consulta()],
        investigador_msgs=[llamada("buscar_fragmentos", consulta="x"), AIMessage(content="ok")],
        redactor_msgs=[AIMessage(content="Respuesta.")],
        salidas_tools=SALIDAS_TAXI,
    )
    instalar(monkeypatch, M_EVALUADOR, FakeEstructurado(error=RuntimeError("cuota")))

    resultado = turno(agente, "¿cuánto cuesta el taxi?")

    assert resultado["messages"][-1].content == "Respuesta."


# ── persistencia entre turnos (checkpointer) ──

def test_documento_activo_persiste_y_el_estado_por_turno_no_se_acumula(monkeypatch, agente):
    fakes = instalar_grafo(
        monkeypatch,
        analisis=[
            consulta("tarifa del taxi"),
            AnalisisConsulta(
                intencion="seguimiento", consulta_autonoma="¿y los recargos del Decreto 001 de 2019?",
                usa_documento_activo=True, razonamiento="x"),
        ],
        investigador_msgs=[
            llamada("buscar_fragmentos", consulta="tarifa del taxi"), AIMessage(content="ok"),
            llamada("buscar_fragmentos", consulta="recargos", archivo=FUENTE_TAXI["archivo"]), AIMessage(content="ok"),
        ],
        redactor_msgs=[AIMessage(content="Primera respuesta."), AIMessage(content="Segunda respuesta.")],
        salidas_tools={
            "buscar_fragmentos": {"fuentes": [FUENTE_TAXI], "fragmentos": [FRAGMENTO_TAXI]},
            "buscar_documento_por_referencia": {"fuentes": [], "fragmentos": [], "candidatos": CANDIDATOS},
        },
    )

    turno1 = turno(agente, "¿cuánto cuesta el taxi?", thread_id="mismo-hilo")
    turno2 = turno(agente, "¿y los recargos?", thread_id="mismo-hilo")

    # El documento activo del turno 1 llegó al analista y al investigador del turno 2.
    assert turno1["documento_activo"]["archivo"] == FUENTE_TAXI["archivo"]
    assert "DOCUMENTO ACTIVO: Decreto 001 de 2019" in fakes["analista"].entradas[1][1].content
    assert FUENTE_TAXI["archivo"] in fakes["investigador"].entradas[2][1].content
    assert turno2["documento_activo"]["archivo"] == FUENTE_TAXI["archivo"]

    # Pasos, fuentes, evidencia y candidatos son del turno actual, no del acumulado.
    assert agentes(turno2) == ["analista", "investigador", "evaluador", "redactor"]
    assert turno2["fuentes"] == [FUENTE_TAXI]
    assert len(turno2["evidencia"]) == 1
    assert turno2["candidatos"] == []
    assert turno2["intentos_busqueda"] == 1

    # messages solo acumula las preguntas y las respuestas finales.
    assert [type(m).__name__ for m in turno2["messages"]] == ["HumanMessage", "AIMessage"] * 2
    assert [m.content for m in turno2["messages"] if m.type == "ai"] == ["Primera respuesta.", "Segunda respuesta."]


def test_candidatos_de_un_turno_no_pasan_al_siguiente(monkeypatch, agente):
    instalar_grafo(
        monkeypatch,
        analisis=[consulta("Resolución 4397"), AnalisisConsulta(
            intencion="saludo", consulta_autonoma="gracias", razonamiento="x")],
        investigador_msgs=[
            llamada("buscar_documento_por_referencia", tipo="resolucion", numero=4397), AIMessage(content="Dos."),
        ],
        redactor_msgs=[AIMessage(content="¿Cuál?")],
        salidas_tools={"buscar_documento_por_referencia": {"fuentes": [], "fragmentos": [], "candidatos": CANDIDATOS}},
    )

    turno1 = turno(agente, "Resolución 4397", thread_id="h")
    turno2 = turno(agente, "gracias", thread_id="h")

    assert turno1["candidatos"] == CANDIDATOS
    assert turno2["candidatos"] == [] and turno2["fuentes"] == []
    assert agentes(turno2) == ["analista", "respuesta_directa"]


def test_hilos_distintos_no_comparten_documento_activo(monkeypatch, agente):
    instalar_grafo(
        monkeypatch,
        analisis=[consulta(), AnalisisConsulta(intencion="saludo", consulta_autonoma="hola", razonamiento="x")],
        investigador_msgs=[llamada("buscar_fragmentos", consulta="x"), AIMessage(content="ok")],
        redactor_msgs=[AIMessage(content="Respuesta.")],
        salidas_tools=SALIDAS_TAXI,
    )

    turno(agente, "¿cuánto cuesta el taxi?", thread_id="hilo-a")
    otro = turno(agente, "hola", thread_id="hilo-b")

    assert otro.get("documento_activo") is None
    assert len(otro["messages"]) == 2


def test_un_listado_no_define_documento_activo(monkeypatch, agente):
    documento = {k: FUENTE_TAXI[k] for k in ("titulo", "archivo", "fecha_expedicion", "descripcion")}
    instalar_grafo(
        monkeypatch,
        analisis=[consulta("¿qué decretos hay de 2019?")],
        investigador_msgs=[llamada("listar_documentos", tipo="decreto", anio=2019), AIMessage(content="Hay uno.")],
        redactor_msgs=[AIMessage(content="Hay un decreto de 2019: Decreto 001.")],
        salidas_tools={"listar_documentos": {
            "fuentes": [{**documento, "pagina": None}], "fragmentos": [], "documentos": [documento], "total": 1}},
    )

    resultado = turno(agente, "¿qué decretos hay de 2019?")

    assert resultado["fuentes"][0]["archivo"] == FUENTE_TAXI["archivo"]
    assert resultado["documento_activo"] is None


def test_construir_grafo_no_requiere_checkpointer():
    assert construir_grafo() is not None
