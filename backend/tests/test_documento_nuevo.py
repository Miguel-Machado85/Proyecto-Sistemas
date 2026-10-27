"""De extremo a extremo: un documento nuevo indexado por el admin funciona con TODAS las tools, sin reiniciar nada.

Chroma en memoria, embeddings falsos, RepositorioLocal y la tabla archivos_documento en SQLite en memoria.
No se llama a Gemini, a Chroma servidor ni a Drive. Las tools son las reales.
"""

import json
import uuid

import fitz
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from agente_sigem import db, ingest, retrieval, tools
from agente_sigem.config import config
from agente_sigem.repositorio_archivos import RepositorioLocal

ARCHIVO = "decreto-045-2026.pdf"
TITULO = "Decreto 045 de 2026"
FRASE = "licencia provisional de construcción para ferias artesanales"
PAGINAS = [
    "DECRETO No. 045 DE 2026\nPor medio del cual se regulan las ferias artesanales del municipio.\n"
    "Dado en Marinilla a 12 de febrero de 2026.",
    f"ARTÍCULO 2. Se otorgará una {FRASE} a los expositores inscritos.",
]


def pdf_de_prueba() -> bytes:
    documento = fitz.open()
    for texto in PAGINAS:
        pagina = documento.new_page()
        pagina.insert_textbox(fitz.Rect(50, 50, 550, 750), texto, fontsize=11)
    contenido = documento.tobytes()
    documento.close()
    return contenido


def llamar(herramienta, **argumentos) -> dict:
    return json.loads(herramienta.invoke(argumentos))


@pytest.fixture
def sistema(monkeypatch, tmp_path):
    """Un 'proceso' completo: Chroma con un documento histórico, catálogo ya construido, repositorio y base."""
    store = Chroma(collection_name=f"nuevo_{uuid.uuid4().hex}", embedding_function=DeterministicFakeEmbedding(size=16))
    store.add_documents([Document(
        page_content="Tarifas de taxis y buses para el año 2019",
        metadata={
            "nombre_archivo": "d1_2019.pdf", "source": "d1_2019.pdf", "titulo": "Decreto 001 de 2019",
            "fecha_expedicion": "2019-01-03T08:00:00", "descripcion": "Tarifas de taxis", "page": 0,
        },
    )])
    for modulo in (retrieval, ingest):
        monkeypatch.setattr(modulo, "obtener_vector_store", lambda: store)
    retrieval.invalidar_cache_documentos()

    motor = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    db.Base.metadata.create_all(motor)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=motor, autoflush=False, autocommit=False))
    repositorio = RepositorioLocal(tmp_path / "repositorio")
    monkeypatch.setattr(ingest, "obtener_repositorio", lambda: repositorio)

    monkeypatch.setattr(config, "UPLOAD_ROOT", tmp_path / "documentos")
    monkeypatch.setattr(config, "EMBEDDING_BATCH_PAUSE_SECONDS", 0)
    monkeypatch.setattr(config, "RERANKER_HABILITADO", False)

    retrieval.obtener_catalogo_documentos()  # el catálogo "de este proceso" ya existe antes de indexar
    yield store, repositorio
    retrieval.invalidar_cache_documentos()


def indexar_documento_nuevo() -> str:
    lote = ingest.crear_lote([(ARCHIVO, pdf_de_prueba())])
    assert lote["documentos"][0]["estado"] == "listo", lote["documentos"][0]
    assert lote["documentos"][0]["metadata_sugerida"]["titulo"] == TITULO
    resultado = ingest.indexar_lote(lote["lote_id"])["resultados"][0]
    assert resultado["estado"] == "indexado", resultado
    assert "advertencias" not in resultado  # el título es canónico
    return lote["lote_id"]


def encuentran_el_documento() -> dict[str, bool]:
    """Qué tools lo encuentran ahora mismo (sin reiniciar ni invalidar nada a mano)."""
    referencia = llamar(tools.buscar_documento_por_referencia, tipo="decreto", numero=45, anio=2026, pregunta="ferias")
    listado = llamar(tools.listar_documentos, anio=2026)
    fragmentos = llamar(tools.buscar_fragmentos, consulta=FRASE)
    lectura = llamar(tools.leer_documento, archivo=ARCHIVO)
    descarga = llamar(tools.preparar_descarga, archivo=ARCHIVO)
    return {
        "buscar_documento_por_referencia": any(f["archivo"] == ARCHIVO for f in referencia["fuentes"]),
        "listar_documentos": any(d["archivo"] == ARCHIVO for d in listado.get("documentos", [])),
        "buscar_fragmentos": any(f["fuente"]["archivo"] == ARCHIVO for f in fragmentos["fragmentos"]),
        "leer_documento": any(f["fuente"]["archivo"] == ARCHIVO for f in lectura["fragmentos"]),
        "preparar_descarga": any(a["archivo"] == ARCHIVO for a in descarga.get("archivos", [])),
    }


def test_un_documento_nuevo_funciona_con_las_cinco_tools_y_desaparece_al_revertir(sistema):
    store, repositorio = sistema
    assert not any(encuentran_el_documento().values())  # antes de indexar, ninguna lo encuentra

    lote_id = indexar_documento_nuevo()

    # Las cinco tools, sin reiniciar nada.
    assert encuentran_el_documento() == dict.fromkeys(
        ["buscar_documento_por_referencia", "listar_documentos", "buscar_fragmentos", "leer_documento", "preparar_descarga"], True
    )
    referencia = llamar(tools.buscar_documento_por_referencia, tipo="decreto", numero=45, anio=2026)
    assert {f["titulo"] for f in referencia["fuentes"]} == {TITULO}
    listado = llamar(tools.listar_documentos, anio=2026)
    assert [d["titulo"] for d in listado["documentos"]] == [TITULO] and listado["total"] == 1
    fragmentos = llamar(tools.buscar_fragmentos, consulta=FRASE)
    assert FRASE in fragmentos["fragmentos"][0]["contenido"] and fragmentos["fragmentos"][0]["fuente"]["pagina"] == "2"
    lectura = llamar(tools.leer_documento, archivo=ARCHIVO)
    assert [f["fuente"]["pagina"] for f in lectura["fragmentos"]] == ["1", "2"]  # sus páginas, en orden
    descarga = llamar(tools.preparar_descarga, archivo=ARCHIVO)
    assert descarga["archivos"][0]["url_descarga"] == f"/documentos/{ARCHIVO}/descargar"
    assert descarga["archivos"][0]["titulo"] == TITULO
    # El original quedó en el repositorio y el documento histórico sigue intacto.
    assert [r.nombre for r in repositorio.listar()] == [ARCHIVO]
    assert llamar(tools.listar_documentos, anio=2019)["total"] == 1

    ingest.revertir_indexacion(lote_id)

    assert not any(encuentran_el_documento().values())  # ninguna lo encuentra después de revertir
    assert repositorio.listar() == [] and db.obtener_archivo(ARCHIVO) is None
    assert llamar(tools.listar_documentos, anio=2019)["total"] == 1  # lo histórico no se tocó
    assert llamar(tools.buscar_documento_por_referencia, tipo="decreto", numero=1, anio=2019)["fuentes"]


def test_otro_proceso_ve_el_documento_sin_invalidar_su_cache(sistema, monkeypatch):
    # Simula un segundo worker: este proceso NO recibe la invalidación explícita del que indexa.
    monkeypatch.setattr(ingest, "invalidar_cache_documentos", lambda: None)

    lote_id = indexar_documento_nuevo()

    assert all(encuentran_el_documento().values())
    ingest.revertir_indexacion(lote_id)
    assert not any(encuentran_el_documento().values())


def test_un_documento_con_titulo_libre_se_encuentra_por_contenido_y_avisa_al_admin(sistema):
    lote = ingest.crear_lote([(ARCHIVO, pdf_de_prueba())])

    resultado = ingest.indexar_lote(lote["lote_id"], {ARCHIVO: {"titulo": "Normas de las ferias"}})["resultados"][0]

    assert resultado["estado"] == "indexado" and resultado["advertencias"] == [ingest.ADVERTENCIA_TITULO_NO_CANONICO]
    estado = encuentran_el_documento()
    assert estado["buscar_fragmentos"] and estado["leer_documento"] and estado["preparar_descarga"]
    assert not estado["buscar_documento_por_referencia"]  # sin tipo y número no hay referencia posible
