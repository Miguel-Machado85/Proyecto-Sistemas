"""Lógica pura de evaluacion/evaluar_agente.py (sin Gemini): métricas, resumen y estimación."""

import importlib.util
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

RUTA = Path(__file__).resolve().parent.parent / "evaluacion" / "evaluar_agente.py"
_spec = importlib.util.spec_from_file_location("evaluar_agente", RUTA)
evaluar_agente = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluar_agente)


def estado_final(intencion, fuentes=(), candidatos=(), es_alerta=False):
    return {
        "messages": [HumanMessage(content="p"), AIMessage(content="respuesta")],
        "analisis": {"intencion": intencion},
        "fuentes": [{"archivo": a} for a in fuentes],
        "candidatos": [{"archivo": a} for a in candidatos],
        "pasos": [],
        "es_alerta": es_alerta,
    }


def caso(**cambios):
    base = {
        "id": "c1", "categoria": "referencia_exacta", "perfil": "P0", "intencion_esperada": "consulta_sigem",
        "archivos_esperados": ["a.pdf"], "turnos": [{"origen": "texto", "mensaje": "¿qué dice el decreto?"}],
    }
    base.update(cambios)
    return base


def test_evaluar_caso_mide_intencion_y_archivos():
    llamadas = []

    def ejecutar(agente, thread_id, mensaje, perfil, origen, etiquetas):
        llamadas.append((thread_id, mensaje, origen))
        return estado_final("consulta_sigem", fuentes=["a.pdf", "b.pdf"])

    resultado = evaluar_agente.evaluar_caso(None, ejecutar, caso(), pausa=0, sello="x")

    assert resultado["metricas"] == {"intencion_ok": True, "archivos_ok": True}
    assert llamadas == [("eval-c1-x", "¿qué dice el decreto?", "texto")]


def test_evaluar_caso_multiturno_usa_un_thread_y_evalua_el_ultimo_turno():
    llamadas = []

    def ejecutar(agente, thread_id, mensaje, perfil, origen, etiquetas):
        llamadas.append((thread_id, mensaje))
        return estado_final("seguimiento" if len(llamadas) == 2 else "consulta_sigem", fuentes=["a.pdf"])

    turnos = [{"origen": "texto", "mensaje": "uno"}, {"origen": "texto", "mensaje": "dos"}]
    resultado = evaluar_agente.evaluar_caso(None, ejecutar, caso(intencion_esperada="seguimiento", turnos=turnos), 0, "x")

    assert [m for _, m in llamadas] == ["uno", "dos"]
    assert len({t for t, _ in llamadas}) == 1
    assert resultado["metricas"]["intencion_ok"] is True and resultado["mensaje"] == "dos"


def test_candidatos_esperados_se_buscan_en_candidatos_y_no_en_fuentes():
    def ejecutar(*args):
        return estado_final("consulta_sigem", candidatos=["a.pdf", "b.pdf"])

    resultado = evaluar_agente.evaluar_caso(
        None, ejecutar, caso(candidatos_esperados=True, archivos_esperados=["a.pdf", "b.pdf"]), 0, "x"
    )

    assert resultado["metricas"] == {"intencion_ok": True, "archivos_ok": True, "candidatos_ok": True}


def test_alerta_esperada_en_ruido_de_voz():
    def ejecutar(*args):
        return estado_final("no_comprensible", es_alerta=True)

    resultado = evaluar_agente.evaluar_caso(
        None, ejecutar, caso(intencion_esperada="no_comprensible", archivos_esperados=[], categoria="voz_ruido"), 0, "x"
    )

    assert resultado["metricas"] == {"intencion_ok": True, "alerta_ok": True}


def test_resumen_por_categoria_ignora_los_errores_en_los_porcentajes():
    resultados = [
        {"id": "1", "categoria": "saludo", "metricas": {"intencion_ok": True}, "tiempo_s": 2.0},
        {"id": "2", "categoria": "saludo", "metricas": {"intencion_ok": False}, "tiempo_s": 4.0},
        {"id": "3", "categoria": "saludo", "metricas": {}, "tiempo_s": 0.0, "error": "429"},
    ]

    resumen = evaluar_agente.resumir(resultados)

    assert resumen["saludo"] == {
        "casos": 3, "errores": 1, "degradados": 0, "intencion": 0.5, "archivos": None, "candidatos": None, "alerta": None,
        "tiempo_s_promedio": 3.0,
    }
    assert resumen["GLOBAL"]["casos"] == 3


def test_detecta_cuota_diaria_y_estima_llamadas():
    assert evaluar_agente.es_cuota_diaria(RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProject"))
    assert not evaluar_agente.es_cuota_diaria(RuntimeError("429 RESOURCE_EXHAUSTED PerMinute"))

    estimacion = evaluar_agente.estimar([caso(), caso(intencion_esperada="saludo", archivos_esperados=[])])
    assert estimacion == {"casos": 2, "turnos": 2, "llamadas_aprox": 6}


def test_detecta_resultados_degradados_por_fail_open():
    sano = {"analisis": {"razonamiento": "Pregunta sobre tarifas"}, "pasos": [{"agente": "evaluador", "accion": "Evidencia suficiente"}]}
    degradado = {
        "analisis": {"razonamiento": "Análisis no disponible (RuntimeError); se busca con el mensaje original."},
        "pasos": [{"agente": "evaluador", "accion": "Evaluación no disponible: se continúa con la evidencia recuperada"}],
    }

    assert evaluar_agente.motivos_de_degradacion(sano) == []
    assert evaluar_agente.motivos_de_degradacion(degradado) == ["analista_fail_open", "evaluador_fail_open"]


def test_reintenta_el_caso_si_la_cuota_por_minuto_se_agota(monkeypatch):
    monkeypatch.setattr(evaluar_agente.time, "sleep", lambda s: None)
    llamadas = []

    def ejecutar(*args):
        llamadas.append(1)
        if len(llamadas) == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
        return estado_final("consulta_sigem", fuentes=["a.pdf"])

    resultado = evaluar_agente.evaluar_con_reintentos(None, ejecutar, caso(), pausa=0, sello="x")

    assert len(llamadas) == 2 and resultado["intentos"] == 2 and resultado["metricas"]["archivos_ok"] is True


def test_no_reintenta_la_cuota_diaria():
    import pytest

    def ejecutar(*args):
        raise RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(RuntimeError):
        evaluar_agente.evaluar_con_reintentos(None, ejecutar, caso(), pausa=0, sello="x")


def test_el_resumen_excluye_los_degradados_de_los_porcentajes():
    resultados = [
        {"id": "1", "categoria": "x", "metricas": {"intencion_ok": True}, "tiempo_s": 1.0, "degradado": []},
        {"id": "2", "categoria": "x", "metricas": {"intencion_ok": False}, "tiempo_s": 1.0, "degradado": ["analista_fail_open"]},
    ]

    resumen = evaluar_agente.resumir(resultados)["x"]

    assert resumen["degradados"] == 1 and resumen["intencion"] == 1.0
