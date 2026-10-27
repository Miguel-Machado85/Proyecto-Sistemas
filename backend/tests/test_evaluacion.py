"""Lógica pura de evaluacion/evaluar_agente.py (sin Gemini): métricas, resumen y estimación."""

import importlib.util
from pathlib import Path

import pytest
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

    assert resultado["metricas"] == {"intencion_ok": True, "archivos_ok": True, "boton_ok": True, "informacion_ok": True}
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

    assert resultado["metricas"] == {"intencion_ok": True, "archivos_ok": True, "boton_ok": True, "candidatos_ok": True}


def test_alerta_esperada_en_ruido_de_voz():
    def ejecutar(*args):
        return estado_final("no_comprensible", es_alerta=True)

    resultado = evaluar_agente.evaluar_caso(
        None, ejecutar, caso(intencion_esperada="no_comprensible", archivos_esperados=[], categoria="voz_ruido"), 0, "x"
    )

    assert resultado["metricas"] == {"intencion_ok": True, "boton_ok": True, "alerta_ok": True}


def test_resumen_por_categoria_ignora_los_errores_en_los_porcentajes():
    resultados = [
        {"id": "1", "categoria": "saludo", "metricas": {"intencion_ok": True}, "tiempo_s": 2.0},
        {"id": "2", "categoria": "saludo", "metricas": {"intencion_ok": False}, "tiempo_s": 4.0},
        {"id": "3", "categoria": "saludo", "metricas": {}, "tiempo_s": 0.0, "error": "429"},
    ]

    resumen = evaluar_agente.resumir(resultados)

    assert resumen["saludo"] == {
        "casos": 3, "errores": 1, "degradados": 0, "intencion": 0.5, "archivos": None, "candidatos": None, "alerta": None,
        "informacion": None, "negativa": None, "solicita_archivo": None, "entrega": None, "boton": None,
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


def estado_con_respuesta(texto, fuentes=(), pasos=()):
    estado = estado_final("consulta_sigem", fuentes=fuentes)
    estado["messages"] = [HumanMessage(content="p"), AIMessage(content=texto)]
    estado["pasos"] = list(pasos)
    return estado


def test_informacion_ok_detecta_un_falso_negativo_aunque_haya_fuentes():
    # inv-04: el archivo esperado está en `fuentes`, pero la respuesta dice que no hay información.
    def ejecutar(*args):
        return estado_con_respuesta("No se encuentra información sobre horarios.", fuentes=["a.pdf"])

    resultado = evaluar_agente.evaluar_caso(None, ejecutar, caso(), 0, "x")

    assert resultado["metricas"]["archivos_ok"] is True
    assert resultado["metricas"]["informacion_ok"] is False


def test_informacion_ok_usa_la_senal_del_redactor():
    pasos = [{"agente": "redactor", "accion": "Redactando respuesta para perfil P0 · sin información pertinente en los fragmentos"}]

    def ejecutar(*args):
        return estado_con_respuesta("Texto neutro.", fuentes=["a.pdf"], pasos=pasos)

    resultado = evaluar_agente.evaluar_caso(None, ejecutar, caso(), 0, "x")

    assert resultado["metricas"]["informacion_ok"] is False


def test_negativa_ok_en_documentos_inexistentes_exige_decirlo_y_no_mostrar_fuentes():
    inexistente = caso(categoria="referencia_inexistente", archivos_esperados=[])

    bien = evaluar_agente.evaluar_caso(
        None, lambda *a: estado_con_respuesta("El Decreto 999 no se encuentra en SIGEM."), inexistente, 0, "x"
    )
    con_relleno = evaluar_agente.evaluar_caso(
        None, lambda *a: estado_con_respuesta("No se encuentra.", fuentes=["x.pdf"]), inexistente, 0, "x"
    )

    assert bien["metricas"]["negativa_ok"] is True
    assert con_relleno["metricas"]["negativa_ok"] is False


# ── solicitud de archivo ──

def estado_con_archivos(solicita, incluye=False, archivos=(), fuentes=()):
    estado = estado_final("consulta_sigem", fuentes=fuentes)
    estado["analisis"] = {"intencion": "consulta_sigem", "solicita_archivo": solicita, "incluye_contenido": incluye}
    estado["archivos"] = [{"archivo": a} for a in archivos]
    return estado


def evaluar_con_estado(estado, **cambios):
    return evaluar_agente.evaluar_caso(None, lambda *a: estado, caso(**cambios), pausa=0, sello="x")


def test_solicitud_de_archivo_mide_la_bandera_la_entrega_y_el_contenido():
    resultado = evaluar_con_estado(
        estado_con_archivos(True, False, archivos=["a.pdf"]), solicita_archivo_esperado=True, incluye_contenido_esperado=False
    )

    assert resultado["metricas"]["solicita_archivo_ok"] and resultado["metricas"]["incluye_contenido_ok"]
    assert resultado["metricas"]["entrega_ok"] and resultado["metricas"]["archivos_ok"]  # el archivo entregado cuenta como encontrado
    assert resultado["archivos_entregados"] == ["a.pdf"]


def test_falso_positivo_de_solicita_archivo_se_detecta():
    resultado = evaluar_con_estado(estado_con_archivos(True, archivos=["a.pdf"], fuentes=["a.pdf"]), solicita_archivo_esperado=False)

    assert resultado["metricas"]["solicita_archivo_ok"] is False
    assert resultado["metricas"]["entrega_ok"] is False  # no se esperaba ningún botón


def test_pedido_de_contenido_correcto_no_entrega_archivos():
    resultado = evaluar_con_estado(estado_con_archivos(False, fuentes=["a.pdf"]), solicita_archivo_esperado=False)

    assert resultado["metricas"]["solicita_archivo_ok"] and resultado["metricas"]["entrega_ok"]
    assert "incluye_contenido_ok" not in resultado["metricas"]


def test_archivos_del_estado_sin_la_bandera_no_cuentan_como_entregados():
    # Misma compuerta que la API: sin solicita_archivo el usuario no recibe nada.
    resultado = evaluar_con_estado(estado_con_archivos(False, archivos=["a.pdf"]), solicita_archivo_esperado=True)

    assert resultado["metricas"]["entrega_ok"] is False and resultado["archivos_entregados"] == []


def test_los_casos_sin_campo_de_archivo_no_cambian_sus_metricas():
    resultado = evaluar_con_estado(estado_final("consulta_sigem", fuentes=["a.pdf"]))

    assert set(resultado["metricas"]) == {"intencion_ok", "archivos_ok", "informacion_ok", "boton_ok"}


def test_el_resumen_reporta_la_exactitud_de_solicita_archivo():
    resultados = [
        {"categoria": "solicitud_archivo", "tiempo_s": 1.0, "metricas": {"solicita_archivo_ok": ok, "entrega_ok": True}}
        for ok in (True, True, True, False)
    ]

    resumen = evaluar_agente.resumir(resultados)["solicitud_archivo"]

    assert resumen["solicita_archivo"] == 0.75 and resumen["entrega"] == 1.0


def test_el_dataset_define_casos_de_solicitud_de_archivo_con_pares_contrastados():
    casos = [c for c in evaluar_agente.cargar_casos(RUTA.parent / "dataset.jsonl") if c["categoria"] == "solicitud_archivo"]

    assert len(casos) >= 10
    assert any(c["solicita_archivo_esperado"] for c in casos) and any(not c["solicita_archivo_esperado"] for c in casos)
    assert any(c.get("incluye_contenido_esperado") for c in casos)  # al menos un pedido mixto
    assert all("solicita_archivo_esperado" in c for c in casos)


# ── límites de palabra y botón inexistente (hallazgo de la evaluación de la fase 6) ──

def respuesta_de(texto, **extra):
    estado = estado_final("consulta_sigem", fuentes=["a.pdf"])
    estado["messages"] = [HumanMessage(content="p"), AIMessage(content=texto)]
    estado.update(extra)
    return estado


def test_las_frases_negativas_exigen_limite_de_palabra():
    sin_informacion = evaluar_agente.respuesta_sin_informacion
    assert not sin_informacion(respuesta_de("Sobre el talento humano se encuentra la Resolución 3615."))
    assert not sin_informacion(respuesta_de("El decreto reúne a los signos se encuentra vigente."))
    assert sin_informacion(respuesta_de("No se encuentra información sobre eso."))
    assert sin_informacion(respuesta_de("Lo siento, no hay información sobre ese tema."))
    assert sin_informacion(respuesta_de("Ese documento no existe en SIGEM."))


def test_el_caso_de_talento_humano_ya_no_se_marca_como_sin_informacion():
    resultado = evaluar_con_estado(respuesta_de("Sobre talento humano se encuentra la Resolución No. 3615 [Resolución No. 3615]."))

    assert resultado["metricas"]["informacion_ok"] is True


def test_boton_ok_detecta_la_promesa_de_un_boton_sin_archivo():
    resultado = evaluar_con_estado(respuesta_de("El horario es de 8 a 12. El documento queda disponible en el botón de descarga debajo del mensaje."))

    assert resultado["metricas"]["boton_ok"] is False


def test_boton_ok_acepta_el_boton_cuando_si_se_entrega_el_archivo():
    estado = respuesta_de("Aquí tienes el documento, en el botón de descarga.")
    estado["analisis"] = {"intencion": "consulta_sigem", "solicita_archivo": True, "incluye_contenido": False}
    estado["archivos"] = [{"archivo": "a.pdf"}]

    assert evaluar_con_estado(estado)["metricas"]["boton_ok"] is True


def test_boton_ok_no_se_deja_engañar_por_archivos_sin_la_compuerta():
    # El estado trae archivos pero el analista no marcó solicita_archivo: la API no los muestra, el botón no existe.
    estado = respuesta_de("Queda disponible en el botón de descarga.")
    estado["archivos"] = [{"archivo": "a.pdf"}]

    assert evaluar_con_estado(estado)["metricas"]["boton_ok"] is False


@pytest.mark.parametrize("texto", ["Los botones de descarga están abajo.", "Pulsa el Botón.", "BOTÓN de descarga"])
def test_menciona_boton_cubre_variantes(texto):
    assert evaluar_agente.menciona_boton_inexistente(texto, set()) is True


@pytest.mark.parametrize("texto", ["El pie de página del documento", "La palabra sobrebotón no cuenta", "Respuesta sin nada"])
def test_menciona_boton_no_confunde_otras_palabras(texto):
    assert evaluar_agente.menciona_boton_inexistente(texto, set()) is False


def test_el_resumen_reporta_la_exactitud_de_boton_ok():
    resultados = [
        {"categoria": "x", "tiempo_s": 1.0, "metricas": {"boton_ok": ok}} for ok in (True, True, True, False)
    ]

    assert evaluar_agente.resumir(resultados)["x"]["boton"] == 0.75
