"""Chroma como servicio: cliente según modo, catálogo que se reconstruye solo y script de migración."""

import importlib.util
import sys
from pathlib import Path

import chromadb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from agente_sigem import api, retrieval
from agente_sigem.config import config

RAIZ = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("migrar_chroma", RAIZ / "scripts" / "migrar_chroma.py")
migrar_chroma = importlib.util.module_from_spec(_spec)
sys.modules["migrar_chroma"] = migrar_chroma
_spec.loader.exec_module(migrar_chroma)


# ── crear_cliente_chroma ──

def test_cliente_http_usa_host_puerto_y_ssl(monkeypatch):
    llamadas = []
    monkeypatch.setattr(config, "CHROMA_MODO", "http")
    monkeypatch.setattr(config, "CHROMA_HOST", "chroma")
    monkeypatch.setattr(config, "CHROMA_PORT", 8123)
    monkeypatch.setattr(config, "CHROMA_SSL", True)
    monkeypatch.setattr(retrieval.chromadb, "HttpClient", lambda **kw: llamadas.append(kw) or "http")
    assert retrieval.crear_cliente_chroma() == "http"
    assert llamadas == [{"host": "chroma", "port": 8123, "ssl": True}]


def test_cliente_local_usa_carpeta(monkeypatch):
    llamadas = []
    monkeypatch.setattr(config, "CHROMA_MODO", "local")
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", "./una_carpeta")
    monkeypatch.setattr(retrieval.chromadb, "PersistentClient", lambda **kw: llamadas.append(kw) or "local")
    assert retrieval.crear_cliente_chroma() == "local"
    assert llamadas == [{"path": "./una_carpeta"}]


def test_modo_invalido_falla_con_mensaje_claro(monkeypatch):
    monkeypatch.setattr(config, "CHROMA_MODO", "nube")
    with pytest.raises(ValueError, match="CHROMA_MODO"):
        retrieval.crear_cliente_chroma()


# ── /health ──

def _cliente_api(monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINTER", "memory")
    monkeypatch.setattr(config, "RERANKER_PRECARGAR", False)
    monkeypatch.setattr(api.db, "init_db", lambda: None)
    return TestClient(api.app)


def test_health_local_no_consulta_chroma(monkeypatch):
    monkeypatch.setattr(config, "CHROMA_MODO", "local")
    with _cliente_api(monkeypatch) as cliente:
        respuesta = cliente.get("/health")
    assert respuesta.status_code == 200 and respuesta.json() == {"status": "ok"}


def test_health_http_reporta_heartbeat(monkeypatch):
    class Cliente:
        def heartbeat(self):
            return 1

    monkeypatch.setattr(config, "CHROMA_MODO", "http")
    monkeypatch.setattr(retrieval, "crear_cliente_chroma", lambda: Cliente())
    with _cliente_api(monkeypatch) as cliente:
        respuesta = cliente.get("/health")
    assert respuesta.status_code == 200 and respuesta.json() == {"status": "ok", "chroma": "ok"}


def test_health_http_con_chroma_caido_responde_503(monkeypatch):
    def caido():
        raise ConnectionError("sin conexión")

    monkeypatch.setattr(config, "CHROMA_MODO", "http")
    monkeypatch.setattr(retrieval, "crear_cliente_chroma", caido)
    with _cliente_api(monkeypatch) as cliente:
        respuesta = cliente.get("/health")
    assert respuesta.status_code == 503 and respuesta.json()["chroma"] == "no disponible"


# ── catálogo que se reconstruye solo ──
# EphemeralClient comparte el sistema en memoria dentro del proceso: dos clientes ven la misma colección.

class VectorStoreFalso:
    """Lo mínimo de langchain_chroma.Chroma que usa el catálogo, sobre una colección real en memoria."""

    def __init__(self, coleccion):
        self._collection = coleccion

    def get(self, include=None):
        return self._collection.get(include=include)


def _agregar(coleccion, nombre, titulo):
    coleccion.upsert(
        ids=[nombre],
        embeddings=[[1.0, 0.0, 0.0]],
        documents=["texto"],
        metadatas=[{"nombre_archivo": nombre, "titulo": titulo, "descripcion": "d", "page": 0}],
    )


def _coleccion_limpia(nombre):
    cliente = chromadb.EphemeralClient()
    if nombre in [c.name for c in cliente.list_collections()]:
        cliente.delete_collection(nombre)
    return cliente, cliente.create_collection(nombre)


@pytest.fixture
def catalogo_en_memoria(monkeypatch):
    _cliente, coleccion = _coleccion_limpia("sigem_test")
    retrieval.invalidar_cache_documentos()
    monkeypatch.setattr(retrieval, "obtener_vector_store", lambda: VectorStoreFalso(coleccion))
    yield coleccion
    retrieval.invalidar_cache_documentos()
    chromadb.EphemeralClient().delete_collection("sigem_test")


def test_catalogo_ve_documentos_agregados_por_otro_cliente(catalogo_en_memoria):
    _agregar(catalogo_en_memoria, "1_decreto-001-2019.pdf", "Decreto 001 de 2019")
    assert len(retrieval.obtener_catalogo_documentos().documentos) == 1

    # Otro proceso (otro cliente sobre la misma colección) indexa un documento nuevo.
    otro = chromadb.EphemeralClient().get_collection("sigem_test")
    _agregar(otro, "nuevo-045-2026.pdf", "Decreto 045 de 2026")

    titulos = {documento.titulo for documento in retrieval.obtener_catalogo_documentos().documentos}
    assert "Decreto 045 de 2026" in titulos


def test_catalogo_no_se_reconstruye_si_el_conteo_no_cambia(catalogo_en_memoria, monkeypatch):
    _agregar(catalogo_en_memoria, "1_decreto-001-2019.pdf", "Decreto 001 de 2019")
    construcciones = []
    original = retrieval.CatalogoDocumentos.desde_metadatas

    def contado(metadatas):
        construcciones.append(1)
        return original(metadatas)

    monkeypatch.setattr(retrieval.CatalogoDocumentos, "desde_metadatas", staticmethod(contado))
    primero = retrieval.obtener_catalogo_documentos()
    segundo = retrieval.obtener_catalogo_documentos()
    assert primero is segundo and len(construcciones) == 1


def test_invalidar_fuerza_la_reconstruccion(catalogo_en_memoria):
    _agregar(catalogo_en_memoria, "1_decreto-001-2019.pdf", "Decreto 001 de 2019")
    primero = retrieval.obtener_catalogo_documentos()
    retrieval.invalidar_cache_documentos()
    assert retrieval.obtener_catalogo_documentos() is not primero


def test_si_count_falla_se_conserva_el_catalogo_anterior(catalogo_en_memoria, monkeypatch):
    _agregar(catalogo_en_memoria, "1_decreto-001-2019.pdf", "Decreto 001 de 2019")
    anterior = retrieval.obtener_catalogo_documentos()

    class Roto:
        def count(self):
            raise ConnectionError("Chroma no responde")

    monkeypatch.setattr(retrieval, "obtener_vector_store", lambda: VectorStoreFalso(Roto()))
    assert retrieval.obtener_catalogo_documentos() is anterior


# ── script de migración ──

def _poblar(coleccion, cantidad):
    rng = np.random.default_rng(3)
    coleccion.upsert(
        ids=[f"id-{i:04d}" for i in range(cantidad)],
        embeddings=rng.normal(size=(cantidad, 8)).tolist(),
        documents=[f"fragmento {i}" for i in range(cantidad)],
        metadatas=[{"nombre_archivo": f"doc{i % 5}.pdf", "page": i} for i in range(cantidad)],
    )


@pytest.fixture
def origen_y_cliente():
    cliente, origen = _coleccion_limpia("origen_mig")
    yield origen, cliente
    for nombre in ("origen_mig", "destino_mig"):
        if nombre in [c.name for c in cliente.list_collections()]:
            cliente.delete_collection(nombre)


def test_leer_paginas_recorre_todo_sin_repetir(origen_y_cliente):
    origen, _ = origen_y_cliente
    _poblar(origen, 23)
    paginas = list(migrar_chroma.leer_paginas(origen, tamano=10))
    assert [len(p["ids"]) for p in paginas] == [10, 10, 3]
    assert len({i for p in paginas for i in p["ids"]}) == 23


def test_migrar_y_verificar_es_idempotente(origen_y_cliente):
    origen, cliente = origen_y_cliente
    _poblar(origen, 23)
    destino = migrar_chroma.crear_destino(cliente, origen, nombre="destino_mig")
    assert migrar_chroma.migrar(origen, destino, tamano=10) == 23
    assert migrar_chroma.migrar(origen, destino, tamano=10) == 23  # re-ejecución tras un corte
    assert destino.count() == 23
    assert migrar_chroma.verificar(origen, destino, muestras=5) == []


def test_verificar_detecta_registros_faltantes(origen_y_cliente):
    origen, cliente = origen_y_cliente
    _poblar(origen, 12)
    destino = migrar_chroma.crear_destino(cliente, origen, nombre="destino_mig")
    migrar_chroma.migrar(origen, destino, tamano=5)
    destino.delete(ids=["id-0003"])
    problemas = migrar_chroma.verificar(origen, destino, muestras=3)
    assert any("Conteo distinto" in p for p in problemas)
    assert any("Ids distintos" in p for p in problemas)


def test_crear_destino_aborta_si_la_distancia_no_coincide(origen_y_cliente):
    _, cliente = origen_y_cliente
    cosine = cliente.create_collection("dist_origen", configuration={"hnsw": {"space": "cosine"}})

    class ClienteQueIgnoraLaMetrica:
        def get_or_create_collection(self, name, metadata=None, configuration=None):
            return cliente.get_or_create_collection("dist_destino")  # l2 por defecto

    try:
        with pytest.raises(RuntimeError, match="distancia"):
            migrar_chroma.crear_destino(ClienteQueIgnoraLaMetrica(), cosine)
    finally:
        for nombre in ("dist_origen", "dist_destino"):
            if nombre in [c.name for c in cliente.list_collections()]:
                cliente.delete_collection(nombre)


def test_verificar_detecta_un_destino_que_recupera_peor(origen_y_cliente):
    origen, cliente = origen_y_cliente
    _poblar(origen, 20)
    destino = migrar_chroma.crear_destino(cliente, origen, nombre="destino_mig")
    migrar_chroma.migrar(origen, destino)

    class DestinoMalo:
        """Misma colección pero con una consulta que devuelve vecinos equivocados."""
        name = "destino_mig"
        configuration = destino.configuration

        def count(self):
            return destino.count()

        def get(self, **kw):
            return destino.get(**kw)

        def query(self, **kw):
            lejanos = destino.get(ids=[f"id-{i:04d}" for i in range(12, 20)], include=[])["ids"]
            return {"ids": [lejanos]}

    problemas = migrar_chroma.verificar(origen, DestinoMalo(), muestras=3)
    assert any("recupera peor" in p for p in problemas)
