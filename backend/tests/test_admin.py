from fastapi import FastAPI
from fastapi.testclient import TestClient

from agente_sigem import admin
from agente_sigem.config import config


def crear_cliente(monkeypatch):
    app = FastAPI()
    app.include_router(admin.router)
    monkeypatch.setattr(config, "ADMIN_UPLOAD_TOKEN", "secreto-prueba")
    return TestClient(app)


def test_cargar_lote_exige_token(monkeypatch):
    cliente = crear_cliente(monkeypatch)
    respuesta = cliente.post("/admin/lotes", files=[("files", ("norma.pdf", b"contenido", "application/pdf"))])
    assert respuesta.status_code == 401


def test_cargar_lote_acepta_varios_archivos_con_token(monkeypatch):
    cliente = crear_cliente(monkeypatch)
    recibidos = []
    monkeypatch.setattr(admin, "crear_lote", lambda archivos: recibidos.extend(archivos) or {"lote_id": "lote"})

    respuesta = cliente.post(
        "/admin/lotes",
        headers={"X-Admin-Token": "secreto-prueba"},
        files=[
            ("files", ("norma.pdf", b"pdf", "application/pdf")),
            ("files", ("acta.png", b"png", "image/png")),
        ],
    )

    assert respuesta.status_code == 200
    assert [nombre for nombre, _ in recibidos] == ["norma.pdf", "acta.png"]


def test_cancelar_lote_exige_token(monkeypatch):
    cliente = crear_cliente(monkeypatch)
    respuesta = cliente.post("/admin/lotes/123e4567-e89b-12d3-a456-426614174000/cancelar")
    assert respuesta.status_code == 401
