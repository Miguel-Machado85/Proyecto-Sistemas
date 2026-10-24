"""Endpoint /documentos/{archivo}/descargar y script de sincronización con Drive. Sin red."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agente_sigem import api, repositorio_archivos as repo
from agente_sigem.catalogo import CatalogoDocumentos
from agente_sigem.config import config
from agente_sigem.repositorio_archivos import ArchivoRemoto, ErrorRepositorio, RepositorioLocal

RAIZ = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("sincronizar_drive", RAIZ / "scripts" / "sincronizar_drive.py")
sincronizar = importlib.util.module_from_spec(_spec)
sys.modules["sincronizar_drive"] = sincronizar
_spec.loader.exec_module(sincronizar)

ARCHIVO = "22598_decreto-no-0012019.pdf"
CONTENIDO = b"%PDF-1.4 decreto 001 " * 200


# ── endpoint ──

@pytest.fixture
def entorno(monkeypatch, tmp_path):
    catalogo = CatalogoDocumentos.desde_metadatas([
        {"nombre_archivo": ARCHIVO, "titulo": "Decreto 001 de 2019", "fecha_expedicion": "2019-01-03T08:00:00"},
        {"nombre_archivo": "99_sin_original.pdf", "titulo": "Decreto 002 de 2019", "fecha_expedicion": "2019-02-01T08:00:00"},
    ])
    monkeypatch.setattr(api, "obtener_catalogo_documentos", lambda: catalogo)

    local = RepositorioLocal(tmp_path / "repositorio")
    origen = tmp_path / "o.pdf"
    origen.write_bytes(CONTENIDO)
    id_externo = local.subir(origen, ARCHIVO, "application/pdf")
    filas = {ARCHIVO: SimpleNamespace(proveedor="local", id_externo=id_externo, mime_type="application/pdf", tamano_bytes=len(CONTENIDO))}
    monkeypatch.setattr(api.db, "obtener_archivo", lambda nombre: filas.get(nombre))
    monkeypatch.setattr(repo, "obtener_repositorio", lambda: local)

    monkeypatch.setattr(config, "CHECKPOINTER", "memory")
    monkeypatch.setattr(config, "RERANKER_PRECARGAR", False)
    monkeypatch.setattr(config, "DRIVE_MODO_ENTREGA", "stream")
    monkeypatch.setattr(api.db, "init_db", lambda: None)
    with TestClient(api.app) as cliente:
        yield SimpleNamespace(cliente=cliente, filas=filas, local=local, id_externo=id_externo)


def test_descarga_en_streaming_con_nombre_original(entorno):
    respuesta = entorno.cliente.get(f"/documentos/{ARCHIVO}/descargar")
    assert respuesta.status_code == 200
    assert respuesta.content == CONTENIDO
    assert respuesta.headers["content-type"] == "application/pdf"
    assert respuesta.headers["content-disposition"] == f"attachment; filename*=UTF-8''{ARCHIVO}"


def test_documento_fuera_del_catalogo_da_404_amable(entorno):
    respuesta = entorno.cliente.get("/documentos/inventado.pdf/descargar")
    assert respuesta.status_code == 404 and "No encontré" in respuesta.json()["detail"]


def test_documento_indexado_sin_fila_da_404_amable(entorno):
    respuesta = entorno.cliente.get("/documentos/99_sin_original.pdf/descargar")
    assert respuesta.status_code == 404 and "aún no está disponible" in respuesta.json()["detail"]


@pytest.mark.parametrize("peligroso", ["..%2F..%2Fetc%2Fpasswd", "%2e%2e%2fsecreto", "..%5C..%5Cx"])
def test_el_parametro_nunca_se_usa_como_ruta(entorno, peligroso):
    assert entorno.cliente.get(f"/documentos/{peligroso}/descargar").status_code == 404


def test_modo_redirect_responde_302_al_enlace(entorno, monkeypatch):
    monkeypatch.setattr(config, "DRIVE_MODO_ENTREGA", "redirect")
    monkeypatch.setattr(RepositorioLocal, "enlace_ver", lambda self, i: "https://drive.google.com/file/d/x/view")
    respuesta = entorno.cliente.get(f"/documentos/{ARCHIVO}/descargar", follow_redirects=False)
    assert respuesta.status_code == 302 and respuesta.headers["location"].startswith("https://drive.google.com/")


def test_modo_redirect_sin_enlace_cae_a_streaming(entorno, monkeypatch):
    monkeypatch.setattr(config, "DRIVE_MODO_ENTREGA", "redirect")
    respuesta = entorno.cliente.get(f"/documentos/{ARCHIVO}/descargar", follow_redirects=False)
    assert respuesta.status_code == 200 and respuesta.content == CONTENIDO


def test_error_del_repositorio_responde_503_sin_filtrar_detalles(entorno, monkeypatch):
    def caido():
        raise ErrorRepositorio("No hay token de Drive en C:/secreto/oauth-token.json")

    monkeypatch.setattr(repo, "obtener_repositorio", caido)
    respuesta = entorno.cliente.get(f"/documentos/{ARCHIVO}/descargar")
    assert respuesta.status_code == 503
    assert "secreto" not in respuesta.json()["detail"] and "no está disponible" in respuesta.json()["detail"]


def test_proveedor_distinto_al_activo_responde_503(entorno):
    entorno.filas[ARCHIVO].proveedor = "drive"
    assert entorno.cliente.get(f"/documentos/{ARCHIVO}/descargar").status_code == 503


def test_la_api_arranca_y_el_chat_no_depende_del_token_de_drive(monkeypatch, tmp_path):
    # Drive configurado pero sin token: el arranque y /health funcionan; solo la descarga da 503.
    monkeypatch.setattr(config, "ARCHIVOS_BACKEND", "drive")
    monkeypatch.setattr(config, "GOOGLE_DRIVE_FOLDER_ID", "CARPETA")
    monkeypatch.setattr(config, "GOOGLE_OAUTH_TOKEN_FILE", tmp_path / "no_existe.json")
    repo.obtener_repositorio.cache_clear()
    repo.reiniciar_credenciales()
    catalogo = CatalogoDocumentos.desde_metadatas([{"nombre_archivo": ARCHIVO, "titulo": "Decreto 001 de 2019"}])
    monkeypatch.setattr(api, "obtener_catalogo_documentos", lambda: catalogo)
    monkeypatch.setattr(api.db, "obtener_archivo", lambda n: SimpleNamespace(proveedor="drive", id_externo="x", mime_type=None, tamano_bytes=None))
    monkeypatch.setattr(config, "CHECKPOINTER", "memory")
    monkeypatch.setattr(config, "RERANKER_PRECARGAR", False)
    monkeypatch.setattr(config, "CHROMA_MODO", "local")
    monkeypatch.setattr(api.db, "init_db", lambda: None)
    try:
        with TestClient(api.app) as cliente:
            assert cliente.get("/health").status_code == 200
            assert cliente.get(f"/documentos/{ARCHIVO}/descargar").status_code == 503
    finally:
        repo.obtener_repositorio.cache_clear()


# ── sincronización ──

def remoto(nombre, id_externo=None):
    return ArchivoRemoto(nombre=nombre, id_externo=id_externo or f"id-{nombre}", mime_type="application/pdf", tamano_bytes=10)


def test_clave_normalizada_ignora_tildes_mayusculas_signos_y_extension():
    assert sincronizar.clave_normalizada("Resolución_4496 (2).PDF") == sincronizar.clave_normalizada("resolucion-4496-2.pdf")


def test_planificar_separa_enlazados_dudosos_faltantes_y_huerfanos():
    catalogo = {
        "a_decreto-001.pdf": "Decreto 001",       # exacto
        "b_decreto-002.pdf": "Decreto 002",       # normalizado
        "c_decreto-003.pdf": "Decreto 003",       # dos candidatos en Drive -> dudoso
        "d_decreto-004.pdf": "Decreto 004",       # no está en Drive
        "e_decreto-005.pdf": "Decreto 005",       # comparte archivo de Drive con f -> dudosos
        "f_decreto-005.pdf": "Decreto 005 bis",
    }
    remotos = [
        remoto("a_decreto-001.pdf"),
        remoto("B_Decreto_002.PDF"),
        remoto("c-decreto-003.pdf", "id-c1"), remoto("C_decreto_003.pdf", "id-c2"),
        remoto("ef_decreto005.pdf", "id-ef"),  # no coincide con e ni f -> huérfano
        remoto("sobrante.pdf"),
    ]
    plan = sincronizar.planificar(catalogo, remotos)
    assert {(a, r.nombre, c) for a, r, c in plan.enlazados} == {
        ("a_decreto-001.pdf", "a_decreto-001.pdf", "nombre exacto"),
        ("b_decreto-002.pdf", "B_Decreto_002.PDF", "nombre normalizado"),
    }
    assert [a for a, _ in plan.dudosos] == ["c_decreto-003.pdf"]
    assert sorted(plan.sin_drive) == ["d_decreto-004.pdf", "e_decreto-005.pdf", "f_decreto-005.pdf"]
    assert {r.nombre for r in plan.huerfanos} == {"ef_decreto005.pdf", "sobrante.pdf"}


def test_un_archivo_de_drive_pedido_por_dos_documentos_no_se_enlaza():
    catalogo = {"x-decreto-1.pdf": "A", "x_decreto_1.pdf": "B"}
    plan = sincronizar.planificar(catalogo, [remoto("x decreto 1.pdf", "unico")])
    assert plan.enlazados == [] and len(plan.dudosos) == 2


def test_aplicar_registra_enlaces_y_sube_los_faltantes(tmp_path):
    catalogo = {"a.pdf": "Decreto A", "falta.pdf": "Decreto F", "sin_local.pdf": "Decreto S"}
    plan = sincronizar.planificar(catalogo, [remoto("a.pdf")])
    carpeta = tmp_path / "locales"
    carpeta.mkdir()
    (carpeta / "falta.pdf").write_bytes(b"%PDF contenido")
    destino = RepositorioLocal(tmp_path / "repositorio")
    registros = {}
    resultado = sincronizar.aplicar(plan, catalogo, "local", lambda **kw: registros.setdefault(kw["nombre_archivo"], kw), destino, carpeta)
    assert resultado["enlazados"] == 1 and resultado["subidos"] == ["falta.pdf"]
    assert resultado["no_encontrados_localmente"] == ["sin_local.pdf"]
    assert registros["a.pdf"]["id_externo"] == "id-a.pdf" and registros["a.pdf"]["titulo"] == "Decreto A"
    assert registros["falta.pdf"]["mime_type"] == "application/pdf"
    assert [r.nombre for r in destino.listar()] == ["falta.pdf"]


def test_aplicar_sin_carpeta_no_sube_nada_y_un_error_de_subida_no_corta_el_resto(tmp_path):
    catalogo = {"a.pdf": None, "b.pdf": None}
    plan = sincronizar.planificar(catalogo, [])
    registros = []

    class SoloLectura:
        def subir(self, *a):
            raise ErrorRepositorio("el usuario de Drive no tiene permiso de Editor sobre la carpeta")

    assert sincronizar.aplicar(plan, catalogo, "drive", lambda **kw: registros.append(kw), SoloLectura(), None)["subidos"] == []
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "b.pdf").write_bytes(b"x")
    resultado = sincronizar.aplicar(plan, catalogo, "drive", lambda **kw: registros.append(kw), SoloLectura(), tmp_path)
    assert len(resultado["errores"]) == 2 and registros == []
