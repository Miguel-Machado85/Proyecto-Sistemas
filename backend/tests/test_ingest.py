from types import SimpleNamespace

from langchain_core.documents import Document

from agente_sigem import ingest
from agente_sigem.config import config


def metadata_base():
    return {
        "titulo": "Decreto 001 de 2026",
        "descripcion": "Documento de prueba",
        "categoria": None,
        "fecha_expedicion": "2026-01-02T08:00:00",
        "total_pages": 1,
        "requirio_ocr": False,
        "author": None,
        "creator": "Prueba",
        "producer": None,
        "creationdate": None,
        "moddate": None,
        "keywords": None,
        "title": None,
        "subject": None,
    }


def test_crear_lote_prepara_archivo_sin_indexarlo(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "UPLOAD_ROOT", tmp_path / "documentos")
    monkeypatch.setattr(ingest, "extraer_documento", lambda ruta: (["Texto legal"], metadata_base()))
    monkeypatch.setattr(ingest, "_documento_existente", lambda archivo, titulo: None)

    lote = ingest.crear_lote([("decreto.pdf", b"pdf de prueba")])

    assert lote["estado"] == "pendiente"
    assert lote["documentos"][0]["estado"] == "listo"
    assert lote["documentos"][0]["paginas_detectadas"] == 1
    assert (tmp_path / "documentos" / "pendientes" / lote["lote_id"] / "decreto.pdf").is_file()


def test_crear_lote_mueve_a_rechazados_los_archivos_no_procesables(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "UPLOAD_ROOT", tmp_path / "documentos")
    monkeypatch.setattr(ingest, "extraer_documento", lambda ruta: (_ for _ in ()).throw(ingest.ErrorIngesta("PDF inválido")))
    monkeypatch.setattr(ingest, "_documento_existente", lambda archivo, titulo: None)

    lote = ingest.crear_lote([("dañado.pdf", b"sin contenido PDF")])

    assert lote["documentos"][0]["estado"] == "rechazado"
    assert (tmp_path / "documentos" / "rechazados" / lote["lote_id"] / "dañado.pdf").is_file()


def test_indexar_lote_agrega_metadata_y_limpia_catalogo(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "UPLOAD_ROOT", tmp_path / "documentos")
    monkeypatch.setattr(config, "EMBEDDING_BATCH_PAUSE_SECONDS", 0)
    monkeypatch.setattr(ingest, "extraer_documento", lambda ruta: (["Texto legal"], metadata_base()))
    monkeypatch.setattr(ingest, "_documento_existente", lambda archivo, titulo: None)
    lote = ingest.crear_lote([("decreto.pdf", b"pdf de prueba")])

    class StoreFalso:
        def __init__(self):
            self.documentos = []
            self._embedding_function = SimpleNamespace(embed_documents=lambda textos: [[0.1] for _ in textos])
            self._collection = SimpleNamespace(upsert=self.upsert)

        def upsert(self, ids, embeddings, documents, metadatas):
            self.documentos.extend(Document(page_content=texto, metadata=metadata) for texto, metadata in zip(documents, metadatas))

        def get(self, **kwargs):
            return {"ids": []}

        def delete(self, **kwargs):
            raise AssertionError("No se debe borrar en una indexación exitosa")

    store = StoreFalso()
    monkeypatch.setattr(ingest, "obtener_vector_store", lambda: store)
    invalidaciones = []
    monkeypatch.setattr(ingest, "invalidar_cache_documentos", lambda: invalidaciones.append(True))

    resultado = ingest.indexar_lote(lote["lote_id"], {"decreto.pdf": {"titulo": "Decreto definitivo"}})

    assert resultado["resultados"][0]["estado"] == "indexado"
    assert invalidaciones == [True]
    assert store.documentos
    metadata = store.documentos[0].metadata
    assert metadata["titulo"] == "Decreto definitivo"
    assert metadata["nombre_archivo"] == "decreto.pdf"
    assert metadata["documento_id"]
    assert metadata["chunk_index"] == 0
    assert "categoria" not in metadata  # Chroma no recibe valores nulos.
    assert (tmp_path / "documentos" / "indexados" / lote["lote_id"] / "decreto.pdf").is_file()


def test_crear_lote_rechaza_un_titulo_ya_indexado(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "UPLOAD_ROOT", tmp_path / "documentos")
    monkeypatch.setattr(ingest, "extraer_documento", lambda ruta: (["Texto legal"], metadata_base()))
    monkeypatch.setattr(ingest, "_documento_existente", lambda archivo, titulo: "Decreto 001 de 2026")

    lote = ingest.crear_lote([("otro.pdf", b"pdf de prueba")])

    assert lote["documentos"][0]["estado"] == "rechazado"
    assert "Ya existe en Chroma" in lote["documentos"][0]["error"]


def test_cancelar_lote_elimina_los_archivos_pendientes(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "UPLOAD_ROOT", tmp_path / "documentos")
    monkeypatch.setattr(ingest, "extraer_documento", lambda ruta: (["Texto legal"], metadata_base()))
    monkeypatch.setattr(ingest, "_documento_existente", lambda archivo, titulo: None)
    lote = ingest.crear_lote([("decreto.pdf", b"pdf de prueba")])

    resultado = ingest.cancelar_lote(lote["lote_id"])

    assert resultado["estado"] == "cancelado"
    assert not (tmp_path / "documentos" / "pendientes" / lote["lote_id"]).exists()


def test_compara_el_nombre_original_con_los_archivos_de_chroma(monkeypatch):
    catalogo = SimpleNamespace(documentos=[SimpleNamespace(
        archivo="22598_decreto-no-0012019.pdf", titulo="Decreto 001 de 2019"
    )])
    monkeypatch.setattr(ingest, "obtener_catalogo_documentos", lambda: catalogo)

    existente = ingest._documento_existente("22598_decreto-no-0012019.pdf", "Título extraído distinto")

    assert existente == "Decreto 001 de 2019"


def test_no_guarda_chunks_si_falla_un_lote_de_embeddings(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "UPLOAD_ROOT", tmp_path / "documentos")
    monkeypatch.setattr(config, "EMBEDDING_BATCH_PAUSE_SECONDS", 0)
    monkeypatch.setattr(ingest, "extraer_documento", lambda ruta: (["Texto legal"], metadata_base()))
    monkeypatch.setattr(ingest, "_documento_existente", lambda archivo, titulo: None)
    lote = ingest.crear_lote([("decreto.pdf", b"pdf de prueba")])

    class StoreFalso:
        def __init__(self):
            self.upserts = []
            self._embedding_function = SimpleNamespace(embed_documents=self.fallar)
            self._collection = SimpleNamespace(upsert=lambda **datos: self.upserts.append(datos))

        def fallar(self, textos):
            raise RuntimeError("Gemini no respondió")

        def get(self, **kwargs):
            return {"ids": []}

        def delete(self, **kwargs):
            pass

    store = StoreFalso()
    monkeypatch.setattr(ingest, "obtener_vector_store", lambda: store)

    resultado = ingest.indexar_lote(lote["lote_id"])

    assert resultado["resultados"][0]["estado"] == "error"
    assert store.upserts == []


def test_revertir_indexacion_borra_solo_los_documentos_del_lote(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "UPLOAD_ROOT", tmp_path / "documentos")
    monkeypatch.setattr(config, "EMBEDDING_BATCH_PAUSE_SECONDS", 0)
    monkeypatch.setattr(ingest, "extraer_documento", lambda ruta: (["Texto legal"], metadata_base()))
    monkeypatch.setattr(ingest, "_documento_existente", lambda archivo, titulo: None)
    lote = ingest.crear_lote([("decreto.pdf", b"pdf de prueba")])

    class StoreFalso:
        def __init__(self):
            self.borrados = []
            self._embedding_function = SimpleNamespace(embed_documents=lambda textos: [[0.1] for _ in textos])
            self._collection = SimpleNamespace(upsert=lambda **datos: None)

        def get(self, **kwargs):
            return {"ids": []}

        def delete(self, **kwargs):
            self.borrados.append(kwargs)

    store = StoreFalso()
    monkeypatch.setattr(ingest, "obtener_vector_store", lambda: store)
    monkeypatch.setattr(ingest, "invalidar_cache_documentos", lambda: None)
    ingest.indexar_lote(lote["lote_id"])

    resultado = ingest.revertir_indexacion(lote["lote_id"])

    assert resultado["estado"] == "pendiente"
    assert len(store.borrados) == 1
    assert (tmp_path / "documentos" / "pendientes" / lote["lote_id"] / "decreto.pdf").is_file()
