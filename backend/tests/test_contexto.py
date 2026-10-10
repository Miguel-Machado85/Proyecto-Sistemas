from agente_sigem.contexto import enriquecer_consulta_con_contexto, es_pregunta_de_seguimiento


FUENTES = [
    {
        "titulo": "Resolución No. 3615",
        "archivo": "18399_resolucion-3615.pdf",
        "pagina": "1",
        "descripcion": "Adopta el plan estratégico de talento humano.",
    }
]


def test_detecta_pregunta_breve_sobre_documento_activo():
    assert es_pregunta_de_seguimiento("¿Para qué periodo tiene vigencia?", FUENTES)
    assert es_pregunta_de_seguimiento("Háblame más", FUENTES)


def test_no_confunde_saludo_con_continuacion():
    assert not es_pregunta_de_seguimiento("Gracias", FUENTES)


def test_no_restringe_una_busqueda_global_al_documento_activo():
    assert not es_pregunta_de_seguimiento("¿Qué normas SIGEM hay sobre talento humano?", FUENTES)
    assert not es_pregunta_de_seguimiento("¿Qué documentos existen sobre presupuesto municipal?", FUENTES)
    assert not es_pregunta_de_seguimiento("Busca decretos sobre transporte", FUENTES)


def test_no_confunde_otro_acto_con_una_continuacion():
    assert not es_pregunta_de_seguimiento("¿Qué modifica el Decreto 83 de 2018?", FUENTES)
    assert not es_pregunta_de_seguimiento("Necesito el documento 18399", FUENTES)


def test_enriquece_pregunta_con_documento_activo():
    consulta = enriquecer_consulta_con_contexto("¿Para qué periodo tiene vigencia?", FUENTES)
    assert "Resolución No. 3615" in consulta
    assert "¿Para qué periodo tiene vigencia?" in consulta


def test_enriquece_solicitud_de_ampliacion():
    consulta = enriquecer_consulta_con_contexto("Háblame más", FUENTES)
    assert "solicita ampliar la explicación" in consulta
