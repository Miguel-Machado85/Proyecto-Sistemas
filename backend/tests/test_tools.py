import json
import uuid

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from agente_sigem import reranker, retrieval, tools
from agente_sigem.catalogo import CatalogoDocumentos
from agente_sigem.config import config


def fragmento(archivo, titulo, fecha, pagina, texto, descripcion="Por medio del cual se regula un asunto"):
    return Document(
        page_content=texto,
        metadata={
            "nombre_archivo": archivo,
            "source": archivo,
            "titulo": titulo,
            "fecha_expedicion": fecha,
            "descripcion": descripcion,
            "page": pagina,
        },
    )


def construir_documentos() -> list[Document]:
    documentos = [
        # Dos "Decreto 83": uno de 2018 y otro de 2019 (referencia ambigua sin año).
        fragmento("d83_2018.pdf", "Decreto 083 de 2018", "2018-08-29T08:00:00", 0,
                  "Se modifica el horario de atención al público en la inspección de policía",
                  "Por el cual se modifica el horario de atención al publico en la inspeccion municipal"),
        fragmento("d83_2018.pdf", "Decreto 083 de 2018", "2018-08-29T08:00:00", 1, "El horario será de 8 a 12 y de 2 a 6"),
        fragmento("d83_2018.pdf", "Decreto 083 de 2018", "2018-08-29T08:00:00", 2, "Rige desde su publicación"),
        fragmento("d83_2019.pdf", "Decreto 083 de 2019", "2019-08-29T08:00:00", 0, "Se adiciona el presupuesto de la vigencia",
                  "Por medio del cual se adiciona el presupuesto para la presente vigencia fiscal"),
        fragmento("r4496_2018.pdf", "Resolución 4496 de 2018", "2018-12-28T08:00:00", 0, "Políticas contables del municipio",
                  "Por medio del cual se adopta el documento Políticas Contables"),
        fragmento("l387_1997.pdf", "Ley 387 de 1997", "1997-07-18T08:00:00", 0, "Medidas contra el desplazamiento forzado",
                  "Por la cual se adoptan medidas para la prevención del desplazamiento forzado"),
        fragmento("18399_resolucion-3615.pdf", "Resolución No. 3615", "2018-10-29T08:00:00", 0,
                  "Plan estratégico de talento humano 2018-2020",
                  "Por medio de la cual se adopta el plan estratégico de talento humano"),
        fragmento("d1_2019.pdf", "Decreto 001 de 2019", "2019-01-03T08:00:00", 0, "Tarifas de taxis y buses",
                  "Por medio del cual se fijan las tarifas para el servicio público de transporte de taxis"),
    ]
    # Siete "Decreto 10" de años distintos: la ambigüedad debe truncarse a 5 candidatos.
    for anio in range(2010, 2017):
        documentos.append(
            fragmento(f"d10_{anio}.pdf", f"Decreto 010 de {anio}", f"{anio}-02-01T08:00:00", 0, f"Texto del decreto 10 de {anio}")
        )
    return documentos


@pytest.fixture
def almacen(monkeypatch):
    """Chroma en memoria con embeddings falsos; reemplaza al vector store y al catálogo reales."""
    store = Chroma(
        collection_name=f"test_{uuid.uuid4().hex}",
        embedding_function=DeterministicFakeEmbedding(size=16),
    )
    store.add_documents(construir_documentos())
    catalogo = CatalogoDocumentos.desde_metadatas(store.get(include=["metadatas"])["metadatas"])

    monkeypatch.setattr(retrieval, "obtener_vector_store", lambda: store)
    monkeypatch.setattr(retrieval, "obtener_catalogo_documentos", lambda: catalogo)
    monkeypatch.setattr(config, "RERANKER_HABILITADO", False)  # nunca se carga un modelo real
    return store


def ejecutar(herramienta, **argumentos) -> dict:
    salida = herramienta.invoke(argumentos)
    assert isinstance(salida, str)
    return json.loads(salida)


def archivos(resultado: dict) -> set[str]:
    return {fuente["archivo"] for fuente in resultado["fuentes"]}


# ── buscar_documento_por_referencia ──

def test_referencia_exacta_devuelve_los_fragmentos_del_documento(almacen):
    resultado = ejecutar(tools.buscar_documento_por_referencia, tipo="decreto", numero=83, anio=2018)

    assert archivos(resultado) == {"d83_2018.pdf"}
    assert len(resultado["fragmentos"]) == 3
    # Sin pregunta, el orden es por página.
    assert [f["fuente"]["pagina"] for f in resultado["fragmentos"]] == ["1", "2", "3"]
    assert "candidatos" not in resultado


def test_referencia_exacta_con_pregunta_ordena_con_el_reranker(almacen, monkeypatch):
    llamadas = []

    def reranker_falso(consulta, documentos, top_n=None):
        llamadas.append(consulta)
        return list(reversed(documentos))

    monkeypatch.setattr(reranker, "reordenar", reranker_falso)

    resultado = ejecutar(
        tools.buscar_documento_por_referencia, tipo="decreto", numero=83, anio=2018, pregunta="¿desde cuándo rige?"
    )

    assert llamadas == ["¿desde cuándo rige?"]
    assert len(resultado["fragmentos"]) == 3


def test_referencia_ambigua_devuelve_candidatos_y_aclaracion(almacen):
    resultado = ejecutar(tools.buscar_documento_por_referencia, tipo="decreto", numero=83)

    assert {c["archivo"] for c in resultado["candidatos"]} == {"d83_2018.pdf", "d83_2019.pdf"}
    assert resultado["aclaracion"]
    assert resultado["fuentes"] == [] and resultado["fragmentos"] == []


def test_referencia_ambigua_limita_a_cinco_candidatos(almacen):
    resultado = ejecutar(tools.buscar_documento_por_referencia, tipo="decreto", numero=10)

    assert len(resultado["candidatos"]) == 5


def test_referencia_inexistente_devuelve_sugerencias(almacen):
    resultado = ejecutar(tools.buscar_documento_por_referencia, tipo="decreto", numero=84, anio=2018)

    assert resultado["fuentes"] == [] and resultado["fragmentos"] == []
    assert 1 <= len(resultado["sugerencias"]) <= 3
    assert resultado["sugerencias"][0]["archivo"] == "d83_2018.pdf"


def test_tipo_fuera_del_literal_no_lanza_excepcion(almacen):
    salida = tools.buscar_documento_por_referencia.invoke({"tipo": "circular", "numero": 1})

    resultado = json.loads(salida)
    assert "error" in resultado and resultado["fuentes"] == []


def test_referencia_por_codigo_de_archivo(almacen):
    resultado = ejecutar(tools.buscar_documento_por_referencia, codigo="18399", pregunta="¿qué periodo cubre?")

    assert archivos(resultado) == {"18399_resolucion-3615.pdf"}
    assert resultado["fragmentos"]


def test_codigo_tolera_espacios_y_texto_alrededor(almacen):
    resultado = ejecutar(tools.buscar_documento_por_referencia, codigo=" documento 18399 ")

    assert archivos(resultado) == {"18399_resolucion-3615.pdf"}


def test_codigo_inexistente_pide_verificar_sin_error(almacen):
    resultado = ejecutar(tools.buscar_documento_por_referencia, codigo="99999")

    assert resultado["fuentes"] == [] and resultado["fragmentos"] == []
    assert "error" not in resultado
    assert "99999" in resultado["aclaracion"]


def test_codigo_inexistente_usa_tipo_y_numero_como_respaldo(almacen):
    resultado = ejecutar(tools.buscar_documento_por_referencia, codigo="99999", tipo="decreto", numero=83, anio=2018)

    assert archivos(resultado) == {"d83_2018.pdf"}


def test_referencia_sin_tipo_numero_ni_codigo_devuelve_error_explicativo(almacen):
    resultado = ejecutar(tools.buscar_documento_por_referencia, anio=2018)

    assert resultado["fuentes"] == [] and "codigo" in resultado["error"]


# ── listar_documentos ──

def test_listar_por_tipo_y_anio(almacen):
    resultado = ejecutar(tools.listar_documentos, tipo="decreto", anio=2019)

    assert {d["archivo"] for d in resultado["documentos"]} == {"d83_2019.pdf", "d1_2019.pdf"}
    assert resultado["total"] == 2
    # También van en `fuentes` para que la API los muestre; no traen contenido.
    assert archivos(resultado) == {d["archivo"] for d in resultado["documentos"]}
    assert resultado["fragmentos"] == []


def test_listar_acepta_plural_y_respeta_el_limite(almacen):
    resultado = ejecutar(tools.listar_documentos, tipo="Decretos", limite=3)

    assert len(resultado["documentos"]) == 3
    assert resultado["total"] > 3
    assert all(d["titulo"].startswith("Decreto") for d in resultado["documentos"])


def test_listar_por_tema_usa_titulo_y_descripcion(almacen):
    resultado = ejecutar(tools.listar_documentos, tema="presupuesto")

    assert [d["archivo"] for d in resultado["documentos"]] == ["d83_2019.pdf"]


def test_listar_con_tipo_desconocido_explica_los_validos(almacen):
    resultado = ejecutar(tools.listar_documentos, tipo="circular")

    assert resultado["documentos"] == [] and "decreto" in resultado["aclaracion"]


# ── buscar_fragmentos ──

def test_buscar_fragmentos_con_archivo_solo_devuelve_ese_archivo(almacen):
    resultado = ejecutar(tools.buscar_fragmentos, consulta="horario atención", archivo="d83_2018.pdf")

    assert resultado["fragmentos"]
    assert archivos(resultado) == {"d83_2018.pdf"}


def test_buscar_fragmentos_sin_archivo_busca_en_todo_el_repositorio(almacen):
    resultado = ejecutar(tools.buscar_fragmentos, consulta="tarifas taxis")

    assert "d1_2019.pdf" in archivos(resultado)
    assert len(archivos(resultado)) > 1


# ── leer_documento ──

def test_leer_documento_en_orden_de_pagina(almacen):
    resultado = ejecutar(tools.leer_documento, archivo="d83_2018.pdf")

    assert [f["fuente"]["pagina"] for f in resultado["fragmentos"]] == ["1", "2", "3"]


def test_leer_documento_filtra_por_paginas(almacen):
    resultado = ejecutar(tools.leer_documento, archivo="d83_2018.pdf", paginas=[2])

    assert [f["fuente"]["pagina"] for f in resultado["fragmentos"]] == ["2"]


def test_leer_documento_inexistente_devuelve_error_con_sugerencias(almacen):
    resultado = ejecutar(tools.leer_documento, archivo="d83_2018.pdf", paginas=[99])

    assert resultado["fragmentos"] == []
    assert "error" in resultado


# ── errores internos y contrato con la API ──

def test_error_interno_devuelve_json_con_error(monkeypatch):
    def explota(*args, **kwargs):
        raise RuntimeError("chroma caído")

    monkeypatch.setattr(retrieval, "obtener_vector_store", explota)
    monkeypatch.setattr(retrieval, "obtener_catalogo_documentos", explota)

    llamadas = [
        (tools.buscar_documento_por_referencia, {"tipo": "decreto", "numero": 83}),
        (tools.listar_documentos, {"tipo": "decreto"}),
        (tools.buscar_fragmentos, {"consulta": "taxi"}),
        (tools.leer_documento, {"archivo": "x.pdf"}),
    ]
    for herramienta, argumentos in llamadas:
        resultado = ejecutar(herramienta, **argumentos)
        assert resultado["fuentes"] == [] and resultado["fragmentos"] == []
        assert "chroma caído" in resultado["error"]


def test_salida_es_compatible_con_fuente_y_candidato_de_la_api(almacen):
    try:
        from agente_sigem.api import CandidatoDocumento, FuenteSigem
    except Exception as error:  # la API construye el agente actual al importarse (necesita GOOGLE_API_KEY)
        pytest.skip(f"No se pudo importar agente_sigem.api: {error}")

    salidas = [
        ejecutar(tools.buscar_documento_por_referencia, tipo="decreto", numero=83, anio=2018),
        ejecutar(tools.buscar_documento_por_referencia, tipo="decreto", numero=83),
        ejecutar(tools.listar_documentos, tipo="decreto", anio=2019),
        ejecutar(tools.buscar_fragmentos, consulta="tarifas taxis"),
        ejecutar(tools.leer_documento, archivo="r4496_2018.pdf"),
    ]
    fuentes = [FuenteSigem(**f) for s in salidas for f in s["fuentes"]]
    candidatos = [CandidatoDocumento(**c) for s in salidas for c in s.get("candidatos", [])]
    fragmentos = [FuenteSigem(**f["fuente"]) for s in salidas for f in s["fragmentos"]]

    assert fuentes and candidatos and fragmentos
    assert all(f.titulo and f.archivo for f in fuentes + fragmentos)


def test_herramientas_del_investigador_exportadas():
    nombres = [herramienta.name for herramienta in tools.HERRAMIENTAS_INVESTIGADOR]

    assert nombres == ["buscar_documento_por_referencia", "listar_documentos", "buscar_fragmentos", "leer_documento"]
