from agente_sigem.catalogo import CatalogoDocumentos, extraer_referencia


def metadata(archivo: str, titulo: str, fecha: str) -> dict:
    return {
        "nombre_archivo": archivo,
        "titulo": titulo,
        "fecha_expedicion": fecha,
    }


def test_reconoce_variantes_de_numero_de_acto():
    for consulta in (
        "Decreto 83 de 2018",
        "Decreto No. 83 de 2018",
        "Decreto N.º 083 de 2018",
        "Decreto número 83/2018",
    ):
        referencia = extraer_referencia(consulta)
        assert referencia is not None
        assert (referencia.tipo, referencia.numero, referencia.anio) == ("decreto", 83, 2018)


def test_resuelve_documento_por_numero_y_anio():
    catalogo = CatalogoDocumentos.desde_metadatas(
        [metadata("10322_decreto-no-0832018.pdf", "Decreto 083 de 2018", "2018-08-29T08:00:00")]
    )

    resultado = catalogo.resolver("Que modifica el Decreto No. 83 de 2018?")

    assert resultado.estado == "exacta"
    assert resultado.documentos[0].archivo == "10322_decreto-no-0832018.pdf"


def test_pide_resolucion_ambigua_si_falta_el_anio():
    catalogo = CatalogoDocumentos.desde_metadatas(
        [
            metadata("a.pdf", "Decreto 083 de 2018", "2018-08-29T08:00:00"),
            metadata("b.pdf", "Decreto 083 de 2019", "2019-08-29T08:00:00"),
        ]
    )

    resultado = catalogo.resolver("Que dice el Decreto 83?")

    assert resultado.estado == "ambigua"
    assert len(resultado.documentos) == 2


def test_resuelve_codigo_de_archivo():
    catalogo = CatalogoDocumentos.desde_metadatas(
        [metadata("18399_resolucion-3615.pdf", "Resolución No. 3615", "2018-10-29T08:00:00")]
    )

    resultado = catalogo.resolver("Necesito el documento 18399")

    assert resultado.estado == "exacta"
    assert resultado.documentos[0].titulo == "Resolución No. 3615"
