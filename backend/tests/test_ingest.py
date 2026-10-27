from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from agente_sigem import ingest
from agente_sigem.config import config
from agente_sigem.repositorio_archivos import ErrorRepositorio, RepositorioLocal


@pytest.fixture(autouse=True)
def original_y_base(monkeypatch, tmp_path):
    """Repositorio local y tabla archivos_documento en memoria: la ingesta nunca toca Drive ni Postgres."""
    repositorio = RepositorioLocal(tmp_path / "repositorio")
    filas: dict[str, dict] = {}
    monkeypatch.setattr(ingest, "obtener_repositorio", lambda: repositorio)
    monkeypatch.setattr(ingest.db, "registrar_archivo", lambda nombre_archivo, **datos: filas.__setitem__(nombre_archivo, datos))
    monkeypatch.setattr(ingest.db, "eliminar_registro_archivo", lambda nombre: filas.pop(nombre, None) is not None)
    return SimpleNamespace(repositorio=repositorio, filas=filas)


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


def test_guardar_manifiesto_reemplaza_el_archivo_de_forma_atomica(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "UPLOAD_ROOT", tmp_path / "documentos")
    lote_id = "12345678-1234-1234-1234-123456789abc"
    ruta = ingest._ruta_lote(lote_id)
    ruta.mkdir(parents=True)
    (ruta / "manifest.json").write_text('{"estado": "anterior"}', encoding="utf-8")

    ingest._guardar_manifiesto(lote_id, {"estado": "indexando", "progreso": 60})

    assert ingest._leer_manifiesto(lote_id) == {"estado": "indexando", "progreso": 60}
    assert not list(ruta.glob(".manifest.json.*.tmp"))


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

# ══════════════════════════════════════════════════════════════
# Original en el repositorio de archivos (Fase 3)
# ══════════════════════════════════════════════════════════════

class StoreConEventos:
    """Vector store falso que anota el orden de los pasos y puede fallar en embeddings o upsert."""

    def __init__(self, eventos, falla_embeddings=False, falla_upsert=False):
        self.eventos, self.borrados, self.upserts = eventos, [], []
        self._falla_upsert = falla_upsert

        def embeber(textos):
            eventos.append("embeddings")
            if falla_embeddings:
                raise RuntimeError("Gemini no respondió")
            return [[0.1] for _ in textos]

        self._embedding_function = SimpleNamespace(embed_documents=embeber)
        self._collection = SimpleNamespace(upsert=self._upsert)

    def _upsert(self, **datos):
        self.eventos.append("upsert")
        if self._falla_upsert:
            raise RuntimeError("Chroma no responde")
        self.upserts.append(datos)

    def get(self, **kwargs):
        return {"ids": []}

    def delete(self, **kwargs):
        self.eventos.append("delete_chroma")
        self.borrados.append(kwargs)


def preparar(monkeypatch, tmp_path, titulo="Decreto 001 de 2026", **opciones_store):
    monkeypatch.setattr(config, "UPLOAD_ROOT", tmp_path / "documentos")
    monkeypatch.setattr(config, "EMBEDDING_BATCH_PAUSE_SECONDS", 0)
    monkeypatch.setattr(ingest, "extraer_documento", lambda ruta: (["Texto legal"], {**metadata_base(), "titulo": titulo}))
    monkeypatch.setattr(ingest, "_documento_existente", lambda archivo, titulo: None)
    monkeypatch.setattr(ingest, "invalidar_cache_documentos", lambda: None)
    eventos: list[str] = []
    store = StoreConEventos(eventos, **opciones_store)
    monkeypatch.setattr(ingest, "obtener_vector_store", lambda: store)
    lote = ingest.crear_lote([("decreto.pdf", b"%PDF contenido de prueba")])
    return lote, store, eventos


def anotar_eventos_del_repositorio(repositorio, eventos):
    subir, retirar = repositorio.subir, repositorio.retirar

    def subir_anotado(*args, **kwargs):
        eventos.append("subir")
        return subir(*args, **kwargs)

    def retirar_anotado(*args, **kwargs):
        eventos.append("retirar")
        return retirar(*args, **kwargs)

    repositorio.subir, repositorio.retirar = subir_anotado, retirar_anotado


def test_indexar_sube_el_original_antes_de_los_embeddings_y_registra_la_fila(monkeypatch, tmp_path, original_y_base):
    lote, store, eventos = preparar(monkeypatch, tmp_path)
    anotar_eventos_del_repositorio(original_y_base.repositorio, eventos)

    resultado = ingest.indexar_lote(lote["lote_id"])["resultados"][0]

    assert eventos == ["subir", "embeddings", "upsert"]
    assert resultado["estado"] == "indexado"
    [remoto] = original_y_base.repositorio.listar()
    assert (remoto.nombre, remoto.mime_type) == ("decreto.pdf", "application/pdf")
    assert resultado["id_externo"] == remoto.id_externo and resultado["proveedor_original"] == "local"
    fila = original_y_base.filas["decreto.pdf"]
    assert fila["id_externo"] == remoto.id_externo and fila["proveedor"] == "local"
    assert fila["titulo"] == "Decreto 001 de 2026" and fila["mime_type"] == "application/pdf"
    assert fila["tamano_bytes"] == len(b"%PDF contenido de prueba") and len(fila["hash_sha256"]) == 64


def test_el_manifiesto_del_lote_guarda_el_id_externo(monkeypatch, tmp_path, original_y_base):
    lote, _, _ = preparar(monkeypatch, tmp_path)
    ingest.indexar_lote(lote["lote_id"])

    guardado = ingest.obtener_lote(lote["lote_id"])["resultados"][0]

    assert guardado["id_externo"] == original_y_base.repositorio.listar()[0].id_externo


@pytest.mark.parametrize("opcion", ["falla_embeddings", "falla_upsert"])
def test_si_falla_chroma_se_retira_el_original_y_no_queda_fila(monkeypatch, tmp_path, original_y_base, opcion):
    lote, store, eventos = preparar(monkeypatch, tmp_path, **{opcion: True})
    anotar_eventos_del_repositorio(original_y_base.repositorio, eventos)

    resultado = ingest.indexar_lote(lote["lote_id"])["resultados"][0]

    assert resultado["estado"] == "error" and "embeddings" in resultado["error"]
    assert store.upserts == [] and store.borrados  # Chroma limpio
    assert original_y_base.repositorio.listar() == []  # el original fue a la papelera
    assert any(original_y_base.repositorio.raiz.joinpath("_papelera").glob("*.meta.json"))
    assert original_y_base.filas == {}
    assert eventos.index("subir") < eventos.index("retirar")


def test_si_falla_la_subida_el_documento_no_se_indexa(monkeypatch, tmp_path, original_y_base):
    lote, store, eventos = preparar(monkeypatch, tmp_path)

    def sin_permiso(*args, **kwargs):
        raise ErrorRepositorio("No se pudo subir el archivo a Drive: el usuario de Drive no tiene permiso de Editor sobre la carpeta.")

    original_y_base.repositorio.subir = sin_permiso

    manifiesto = ingest.indexar_lote(lote["lote_id"])
    resultado = manifiesto["resultados"][0]

    assert resultado["estado"] == "error"
    assert "no se indexó" in resultado["error"] and "permiso de Editor" in resultado["error"]
    assert eventos == []  # ni embeddings ni upsert: no se gastó cuota
    assert store.upserts == [] and original_y_base.filas == {}
    assert (tmp_path / "documentos" / "pendientes" / lote["lote_id"] / "decreto.pdf").is_file()  # sigue sin indexar
    assert manifiesto["estado"] == "indexado"  # el lote termina; el error queda en el resultado del archivo


def test_un_error_inesperado_al_subir_tambien_se_explica_al_admin(monkeypatch, tmp_path, original_y_base):
    lote, _, eventos = preparar(monkeypatch, tmp_path)
    original_y_base.repositorio.subir = lambda *a, **k: (_ for _ in ()).throw(OSError("disco lleno"))

    resultado = ingest.indexar_lote(lote["lote_id"])["resultados"][0]

    assert resultado["estado"] == "error" and "no se indexó" in resultado["error"] and eventos == []


def test_si_falla_el_registro_en_la_base_se_deshace_todo(monkeypatch, tmp_path, original_y_base):
    lote, store, eventos = preparar(monkeypatch, tmp_path)

    def base_caida(nombre_archivo, **datos):
        raise RuntimeError("postgres caído")

    monkeypatch.setattr(ingest.db, "registrar_archivo", base_caida)

    resultado = ingest.indexar_lote(lote["lote_id"])["resultados"][0]

    assert resultado["estado"] == "error" and "registrar el original" in resultado["error"]
    assert store.borrados  # los fragmentos que ya estaban en Chroma se quitan
    assert original_y_base.repositorio.listar() == [] and original_y_base.filas == {}


def test_si_ademas_no_se_puede_retirar_el_original_el_admin_lo_sabe(monkeypatch, tmp_path, original_y_base):
    lote, _, _ = preparar(monkeypatch, tmp_path, falla_upsert=True)
    original_y_base.repositorio.retirar = lambda *a, **k: (_ for _ in ()).throw(ErrorRepositorio("sin permiso"))

    resultado = ingest.indexar_lote(lote["lote_id"])["resultados"][0]

    assert resultado["estado"] == "error" and "no se pudo retirar el original" in resultado["error"]


def test_los_omitidos_por_contenido_no_se_suben(monkeypatch, tmp_path, original_y_base):
    lote, store, eventos = preparar(monkeypatch, tmp_path)
    monkeypatch.setattr(ingest, "_ya_indexado", lambda hash_archivo: True)  # mismo contenido

    resultado = ingest.indexar_lote(lote["lote_id"])["resultados"][0]

    assert resultado["estado"] == "omitido"
    assert original_y_base.repositorio.listar() == [] and original_y_base.filas == {} and eventos == []


def test_un_titulo_ya_existente_no_se_sube(monkeypatch, tmp_path, original_y_base):
    lote, store, eventos = preparar(monkeypatch, tmp_path)
    monkeypatch.setattr(ingest, "_documento_existente", lambda archivo, titulo: "Decreto 001 de 2026")

    resultado = ingest.indexar_lote(lote["lote_id"])["resultados"][0]

    assert resultado["estado"] == "omitido" and original_y_base.repositorio.listar() == [] and eventos == []


def test_titulo_sin_tipo_y_numero_indexa_con_una_advertencia(monkeypatch, tmp_path, original_y_base):
    lote, _, _ = preparar(monkeypatch, tmp_path, titulo="Informe de gestión del año")

    resultado = ingest.indexar_lote(lote["lote_id"])["resultados"][0]

    assert resultado["estado"] == "indexado"  # no bloquea
    assert resultado["advertencias"] == [ingest.ADVERTENCIA_TITULO_NO_CANONICO]
    assert "Decreto 045 de 2026" in resultado["advertencias"][0]
    assert "decreto.pdf" in original_y_base.filas


def test_titulo_canonico_no_lleva_advertencia(monkeypatch, tmp_path):
    lote, _, _ = preparar(monkeypatch, tmp_path, titulo="Decreto 045 de 2026")

    assert "advertencias" not in ingest.indexar_lote(lote["lote_id"])["resultados"][0]


def test_la_advertencia_usa_el_titulo_final_que_edito_el_admin(monkeypatch, tmp_path):
    lote, _, _ = preparar(monkeypatch, tmp_path, titulo="Informe de gestión del año")

    resultado = ingest.indexar_lote(lote["lote_id"], {"decreto.pdf": {"titulo": "Decreto 045 de 2026"}})["resultados"][0]

    assert "advertencias" not in resultado


def test_revertir_limpia_chroma_el_repositorio_y_la_tabla(monkeypatch, tmp_path, original_y_base):
    lote, store, _ = preparar(monkeypatch, tmp_path)
    ingest.indexar_lote(lote["lote_id"])
    assert original_y_base.filas and original_y_base.repositorio.listar()

    manifiesto = ingest.revertir_indexacion(lote["lote_id"])

    assert manifiesto["estado"] == "pendiente"
    assert len(store.borrados) == 1  # Chroma
    assert original_y_base.repositorio.listar() == []  # repositorio (a la papelera, no borrado)
    assert any(original_y_base.repositorio.raiz.joinpath("_papelera").glob("*.meta.json"))
    assert original_y_base.filas == {}  # tabla
    assert (tmp_path / "documentos" / "pendientes" / lote["lote_id"] / "decreto.pdf").is_file()


def test_revertir_es_reintentable_si_no_se_pudo_retirar_el_original(monkeypatch, tmp_path, original_y_base):
    lote, store, _ = preparar(monkeypatch, tmp_path)
    ingest.indexar_lote(lote["lote_id"])
    retirar_real = original_y_base.repositorio.retirar
    original_y_base.repositorio.retirar = lambda *a, **k: (_ for _ in ()).throw(
        ErrorRepositorio("el usuario de Drive no tiene permiso de Editor sobre la carpeta")
    )

    with pytest.raises(ingest.ErrorIngesta) as error:
        ingest.revertir_indexacion(lote["lote_id"])

    assert "decreto.pdf" in str(error.value) and "permiso de Editor" in str(error.value)
    assert ingest.obtener_lote(lote["lote_id"])["estado"] == "indexado"  # sigue revertible
    assert "decreto.pdf" in original_y_base.filas  # la fila no se borra hasta retirar el original

    original_y_base.repositorio.retirar = retirar_real
    assert ingest.revertir_indexacion(lote["lote_id"])["estado"] == "pendiente"
    assert original_y_base.repositorio.listar() == [] and original_y_base.filas == {}


def test_revertir_no_retira_dos_veces_un_original_ya_retirado(monkeypatch, tmp_path, original_y_base):
    lote, store, _ = preparar(monkeypatch, tmp_path)
    ingest.indexar_lote(lote["lote_id"])
    llamadas = []
    retirar_real = original_y_base.repositorio.retirar
    original_y_base.repositorio.retirar = lambda id_externo: (llamadas.append(id_externo), retirar_real(id_externo))
    manifiesto = ingest.obtener_lote(lote["lote_id"])
    manifiesto["resultados"][0]["original_retirado"] = True  # lo dejó marcado un intento anterior
    ingest._guardar_manifiesto(lote["lote_id"], manifiesto)

    assert ingest.revertir_indexacion(lote["lote_id"])["estado"] == "pendiente"
    assert llamadas == []


def test_revertir_con_otro_proveedor_activo_no_toca_el_original(monkeypatch, tmp_path, original_y_base):
    lote, _, _ = preparar(monkeypatch, tmp_path)
    ingest.indexar_lote(lote["lote_id"])
    manifiesto = ingest.obtener_lote(lote["lote_id"])
    manifiesto["resultados"][0]["proveedor_original"] = "drive"
    ingest._guardar_manifiesto(lote["lote_id"], manifiesto)

    with pytest.raises(ingest.ErrorIngesta, match="'drive'"):
        ingest.revertir_indexacion(lote["lote_id"])

    assert len(original_y_base.repositorio.listar()) == 1  # sin tocar


def test_revertir_un_lote_indexado_antes_de_la_fase_3_no_falla(monkeypatch, tmp_path, original_y_base):
    lote, store, _ = preparar(monkeypatch, tmp_path)
    ingest.indexar_lote(lote["lote_id"])
    manifiesto = ingest.obtener_lote(lote["lote_id"])
    for clave in ("id_externo", "proveedor_original"):
        manifiesto["resultados"][0].pop(clave)  # lote antiguo: sin original registrado
    ingest._guardar_manifiesto(lote["lote_id"], manifiesto)

    assert ingest.revertir_indexacion(lote["lote_id"])["estado"] == "pendiente"
    assert len(store.borrados) == 1
