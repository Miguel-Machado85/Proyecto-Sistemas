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


CATALOGO_ATRIBUTOS = [
    metadata("22598_decreto-no-0012019.pdf", "Decreto 001 de 2019", "2019-01-03T08:00:00")
    | {"descripcion": "Por medio del cual se fijan las tarifas para el servicio público de transporte de taxis y buses"},
    metadata("a.pdf", "Decreto 083 de 2018", "2018-08-29T08:00:00"),
    metadata("b.pdf", "Decreto 083 de 2019", "2019-08-29T08:00:00"),
    metadata("23655_resolucion-4397.pdf", "Resolución 4397 de 2018", "2018-12-13T08:00:00"),
]


def test_buscar_por_atributos_exacta_con_anio():
    catalogo = CatalogoDocumentos.desde_metadatas(CATALOGO_ATRIBUTOS)

    resultado = catalogo.buscar_por_atributos("decreto", 83, 2018)

    assert resultado.estado == "exacta"
    assert resultado.documentos[0].archivo == "a.pdf"


def test_buscar_por_atributos_ambigua_sin_anio():
    catalogo = CatalogoDocumentos.desde_metadatas(CATALOGO_ATRIBUTOS)

    resultado = catalogo.buscar_por_atributos("decreto", 83)

    assert resultado.estado == "ambigua"
    assert {d.archivo for d in resultado.documentos} == {"a.pdf", "b.pdf"}


def test_buscar_por_atributos_normaliza_el_tipo_y_reporta_sin_coincidencia():
    catalogo = CatalogoDocumentos.desde_metadatas(CATALOGO_ATRIBUTOS)

    assert catalogo.buscar_por_atributos("Resolución", 4397, 2018).estado == "exacta"
    assert catalogo.buscar_por_atributos("decreto", 999, 2018).estado == "sin_coincidencia"
    assert catalogo.buscar_por_atributos("decreto", 83, 2020).estado == "sin_coincidencia"


def test_similares_tolera_errores_de_ortografia_en_el_titulo():
    catalogo = CatalogoDocumentos.desde_metadatas(CATALOGO_ATRIBUTOS)

    similares = catalogo.similares("decrto 083 de 2018")

    assert similares[0].archivo == "a.pdf"


def test_similares_encuentra_por_descripcion_y_respeta_limite():
    catalogo = CatalogoDocumentos.desde_metadatas(CATALOGO_ATRIBUTOS)

    assert catalogo.similares("tarifas taxis")[0].archivo == "22598_decreto-no-0012019.pdf"
    assert len(catalogo.similares("decreto", limite=2)) <= 2
    assert catalogo.similares("ab") == []


def test_buscar_por_codigo_de_archivo():
    catalogo = CatalogoDocumentos.desde_metadatas(
        [
            metadata("18399_resolucion-3615.pdf", "Resolución No. 3615", "2018-10-29T08:00:00"),
            metadata("22598_decreto-no-0012019.pdf", "Decreto 001 de 2019", "2019-01-03T08:00:00"),
        ]
    )

    exacta = catalogo.buscar_por_codigo("18399")
    assert exacta.estado == "exacta" and exacta.documentos[0].titulo == "Resolución No. 3615"
    assert catalogo.buscar_por_codigo(" archivo 22598 ").documentos[0].archivo == "22598_decreto-no-0012019.pdf"
    assert catalogo.buscar_por_codigo(18399).estado == "exacta"
    assert catalogo.buscar_por_codigo("00000").estado == "sin_coincidencia"
    assert catalogo.buscar_por_codigo("sin digitos").estado == "sin_coincidencia"


CATALOGO_TEMAS = CatalogoDocumentos.desde_metadatas(
    [
        {"nombre_archivo": "a.pdf", "titulo": "Decreto 083 de 2018", "fecha_expedicion": "2018-08-29",
         "descripcion": "Por el cual se modifica el horario de atención al publico en la inspección municipal de policía"},
        {"nombre_archivo": "b.pdf", "titulo": "Decreto 144 de 2018", "fecha_expedicion": "2018-12-10",
         "descripcion": "Por el cual se suspende la atención al público el día catorce de diciembre"},
        {"nombre_archivo": "c.pdf", "titulo": "Decreto 001 de 2019", "fecha_expedicion": "2019-01-03",
         "descripcion": "Por medio del cual se fijan las tarifas para el servicio público de transporte de taxis"},
        {"nombre_archivo": "d.pdf", "titulo": "Decreto 002 de 2019", "fecha_expedicion": "2019-01-11",
         "descripcion": "Por medio del cual se realizan unos traslados al presupuesto de la actual vigencia"},
    ]
)


def archivos_de(resultado):
    return {d.archivo for d in resultado[0]}


def test_listar_por_tema_ignora_palabras_vacias_como_de_o_al():
    # Regresión inv-04: "de"/"al" hacían coincidir decretos de tarifas con una pregunta sobre horarios.
    resultado = CATALOGO_TEMAS.listar(tema="horarios de atención al público")

    assert archivos_de(resultado) == {"a.pdf", "b.pdf"}
    assert resultado[1] == 2


def test_listar_por_tema_tolera_variantes_y_exige_la_mitad_de_las_palabras():
    assert archivos_de(CATALOGO_TEMAS.listar(tema="horario atencion")) == {"a.pdf", "b.pdf"}
    assert archivos_de(CATALOGO_TEMAS.listar(tema="presupuestos")) == {"d.pdf"}
    assert archivos_de(CATALOGO_TEMAS.listar(tema="tarifas de taxi")) == {"c.pdf"}
    assert archivos_de(CATALOGO_TEMAS.listar(tema="cultura deporte")) == set()


def test_listar_con_tema_solo_de_palabras_vacias_no_filtra():
    assert archivos_de(CATALOGO_TEMAS.listar(tema="decretos", limite=10)) == {"a.pdf", "b.pdf", "c.pdf", "d.pdf"}
    assert archivos_de(CATALOGO_TEMAS.listar(tipo="decreto", anio=2019, tema="normas sobre")) == {"c.pdf", "d.pdf"}
