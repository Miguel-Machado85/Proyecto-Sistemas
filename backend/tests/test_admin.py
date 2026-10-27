import bcrypt
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agente_sigem import admin
from agente_sigem.config import config


def crear_cliente(monkeypatch):
    app = FastAPI()
    app.include_router(admin.router)
    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin-prueba")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", bcrypt.hashpw(b"secreto-prueba", bcrypt.gensalt()).decode())
    monkeypatch.setattr(config, "ADMIN_JWT_SECRET", "secreto-jwt-de-prueba-suficientemente-largo")
    monkeypatch.setattr(config, "ADMIN_SESSION_HOURS", 1)
    monkeypatch.setattr(config, "ADMIN_COOKIE_SECURE", False)
    return TestClient(app)


def iniciar_sesion(cliente):
    return cliente.post("/admin/login", json={"usuario": "admin-prueba", "contrasena": "secreto-prueba"})


def test_cargar_lote_exige_sesion(monkeypatch):
    cliente = crear_cliente(monkeypatch)
    respuesta = cliente.post("/admin/lotes", files=[("files", ("norma.pdf", b"contenido", "application/pdf"))])
    assert respuesta.status_code == 401


def test_login_guarda_cookie_jwt_y_permite_cargar(monkeypatch):
    cliente = crear_cliente(monkeypatch)
    recibidos = []
    monkeypatch.setattr(admin, "crear_lote", lambda archivos: recibidos.extend(archivos) or {"lote_id": "lote"})

    respuesta_login = iniciar_sesion(cliente)
    assert respuesta_login.status_code == 204
    assert "sigem_admin_session" in respuesta_login.headers["set-cookie"]
    assert cliente.get("/admin/sesion").json() == {"autenticado": True, "usuario": "admin-prueba"}

    respuesta = cliente.post(
        "/admin/lotes",
        files=[
            ("files", ("norma.pdf", b"pdf", "application/pdf")),
            ("files", ("acta.png", b"png", "image/png")),
        ],
    )

    assert respuesta.status_code == 200
    assert [nombre for nombre, _ in recibidos] == ["norma.pdf", "acta.png"]


def test_login_invalido_no_entrega_cookie(monkeypatch):
    cliente = crear_cliente(monkeypatch)
    respuesta = cliente.post("/admin/login", json={"usuario": "admin-prueba", "contrasena": "incorrecta"})
    assert respuesta.status_code == 401
    assert "sigem_admin_session" not in respuesta.headers.get("set-cookie", "")


def test_logout_borra_cookie_y_revoca_acceso_del_navegador(monkeypatch):
    cliente = crear_cliente(monkeypatch)
    assert iniciar_sesion(cliente).status_code == 204
    assert cliente.post("/admin/logout").status_code == 204
    assert cliente.get("/admin/sesion").status_code == 401
