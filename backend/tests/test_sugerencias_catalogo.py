from agente_sigem.catalogo import CatalogoDocumentos


def test_sugiere_documento_por_referencia_normalizada():
    catalogo = CatalogoDocumentos.desde_metadatas(
        [
            {
                "nombre_archivo": "18399_resolucion-3615.pdf",
                "titulo": "Resolución No. 3615 de 2018",
                "fecha_expedicion": "2018-10-29T08:00:00",
                "descripcion": "Plan Estratégico de Talento Humano",
            }
        ]
    )

    sugerencias = catalogo.sugerencias("resolucion 3615")

    assert len(sugerencias) == 1
    assert sugerencias[0].archivo == "18399_resolucion-3615.pdf"


def test_no_sugiere_con_terminos_demasiado_cortos():
    catalogo = CatalogoDocumentos.desde_metadatas([])

    assert catalogo.sugerencias("re") == []
