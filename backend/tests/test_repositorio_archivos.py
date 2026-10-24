"""Repositorio de archivos (local y Drive con cliente falso), tabla archivos_documento y config de Drive. Sin red."""

import json
from types import SimpleNamespace

import httplib2
import pytest
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from agente_sigem import db, repositorio_archivos as repo
from agente_sigem.config import extraer_id_carpeta_drive
from agente_sigem.repositorio_archivos import ErrorRepositorio, RepositorioDrive, RepositorioLocal


# ── config ──

@pytest.mark.parametrize("valor,esperado", [
    ("1fE_77pk8np6OEt3dIHN7fO4gmbuC6nZe", "1fE_77pk8np6OEt3dIHN7fO4gmbuC6nZe"),
    ("https://drive.google.com/drive/folders/1fE_77pk8np6OEt3dIHN7fO4gmbuC6nZe?usp=sharing", "1fE_77pk8np6OEt3dIHN7fO4gmbuC6nZe"),
    ("https://drive.google.com/drive/u/0/folders/AbC-123_x", "AbC-123_x"),
    ("https://drive.google.com/open?id=AbC-123_x", "AbC-123_x"),
    ("  AbC-123_x  ", "AbC-123_x"),
    ("", None),
    (None, None),
])
def test_extrae_el_id_de_la_carpeta(valor, esperado):
    assert extraer_id_carpeta_drive(valor) == esperado


# ── RepositorioLocal ──

@pytest.fixture
def local(tmp_path):
    return RepositorioLocal(tmp_path / "repositorio")


@pytest.fixture
def pdf(tmp_path):
    ruta = tmp_path / "origen.pdf"
    ruta.write_bytes(b"%PDF-1.4 contenido de prueba " * 100)
    return ruta


def test_local_sube_descarga_y_lista(local, pdf):
    id_externo = local.subir(pdf, "14354_decreto-no-0862018.pdf", "application/pdf")
    assert b"".join(local.descargar(id_externo)) == pdf.read_bytes()
    [archivo] = local.listar()
    assert (archivo.nombre, archivo.id_externo, archivo.mime_type) == ("14354_decreto-no-0862018.pdf", id_externo, "application/pdf")
    assert archivo.tamano_bytes == pdf.stat().st_size


def test_local_retirar_mueve_a_la_papelera_y_no_borra(local, pdf):
    id_externo = local.subir(pdf, "a.pdf", "application/pdf")
    local.retirar(id_externo)
    assert local.listar() == []
    assert (local.raiz / "_papelera" / id_externo).is_file()
    with pytest.raises(ErrorRepositorio):
        local.descargar(id_externo)


def test_local_nombre_del_usuario_nunca_forma_parte_de_la_ruta(local, pdf):
    id_externo = local.subir(pdf, "../../escape.pdf", "application/pdf")
    assert not (local.raiz.parent.parent / "escape.pdf").exists()
    assert {p.name for p in local.raiz.iterdir()} == {id_externo, f"{id_externo}.meta.json"}


@pytest.mark.parametrize("malo", ["../secreto", "a/b", "..\\x", "", "ZZ" * 16, "abc"])
def test_local_rechaza_ids_invalidos(local, malo):
    with pytest.raises(ErrorRepositorio):
        local.descargar(malo)
    with pytest.raises(ErrorRepositorio):
        local.retirar(malo)


def test_local_listar_sin_carpeta_devuelve_vacio(tmp_path):
    assert RepositorioLocal(tmp_path / "no_existe").listar() == []


# ── RepositorioDrive con cliente falso ──

def _http_error(estado, razon=""):
    contenido = json.dumps({"error": {"errors": [{"reason": razon}]}}).encode()
    return HttpError(httplib2.Response({"status": estado}), contenido)


class Peticion:
    def __init__(self, resultado=None, error=None):
        self.resultado, self.error = resultado, error

    def execute(self):
        if self.error:
            raise self.error
        return self.resultado


class ArchivosFalsos:
    def __init__(self):
        self.llamadas: list[tuple[str, dict]] = []
        self.paginas: list[dict] = []
        self.error: Exception | None = None
        self.trozos: list[bytes] = []

    def _registrar(self, nombre, kwargs):
        self.llamadas.append((nombre, kwargs))
        return Peticion(error=self.error)

    def list(self, **kw):
        self.llamadas.append(("list", kw))
        if self.error:
            return Peticion(error=self.error)
        return Peticion(self.paginas.pop(0))

    def get_media(self, **kw):
        self.llamadas.append(("get_media", kw))
        return SimpleNamespace(trozos=list(self.trozos), error=self.error)

    def create(self, **kw):
        self.llamadas.append(("create", kw))
        return Peticion({"id": "nuevo-id"}, self.error)

    def update(self, **kw):
        self.llamadas.append(("update", kw))
        return Peticion({"id": kw["fileId"]}, self.error)

    def get(self, **kw):
        self.llamadas.append(("get", kw))
        return Peticion({"webViewLink": "https://drive.google.com/file/d/x/view"}, self.error)

    def delete(self, **kw):  # no debe llamarse nunca
        raise AssertionError("Se intentó un borrado permanente en Drive.")


class DescargaFalsa:
    def __init__(self, buffer, solicitud, chunksize):
        self.buffer, self.solicitud = buffer, solicitud

    def next_chunk(self):
        if self.solicitud.error:
            raise self.solicitud.error
        self.buffer.write(self.solicitud.trozos.pop(0))
        return None, not self.solicitud.trozos


@pytest.fixture
def drive(monkeypatch):
    archivos = ArchivosFalsos()
    monkeypatch.setattr(repo, "build", lambda *a, **k: SimpleNamespace(files=lambda: archivos))
    monkeypatch.setattr(repo, "_cargar_credenciales", lambda: object())
    monkeypatch.setattr(repo, "MediaIoBaseDownload", DescargaFalsa)
    monkeypatch.setattr(repo, "MediaFileUpload", lambda ruta, mimetype, resumable: ("media", ruta, mimetype, resumable))
    return RepositorioDrive("CARPETA"), archivos


def _kw(archivos, nombre):
    return [kw for n, kw in archivos.llamadas if n == nombre]


def test_drive_listar_pagina_y_usa_supports_all_drives(drive):
    repositorio, archivos = drive
    archivos.paginas = [
        {"files": [{"id": "1", "name": "a.pdf", "mimeType": "application/pdf", "size": "10"}], "nextPageToken": "T2"},
        {"files": [{"id": "2", "name": "b.pdf"}]},
    ]
    resultado = repositorio.listar()
    assert [(r.nombre, r.id_externo, r.tamano_bytes) for r in resultado] == [("a.pdf", "1", 10), ("b.pdf", "2", None)]
    primera, segunda = _kw(archivos, "list")
    assert "'CARPETA' in parents" in primera["q"] and "trashed = false" in primera["q"]
    assert primera["supportsAllDrives"] and primera["includeItemsFromAllDrives"]
    assert primera["pageToken"] is None and segunda["pageToken"] == "T2"


def test_drive_descarga_en_trozos(drive):
    repositorio, archivos = drive
    archivos.trozos = [b"uno", b"dos", b"tres"]
    assert b"".join(repositorio.descargar("ID")) == b"unodostres"
    assert _kw(archivos, "get_media") == [{"fileId": "ID", "supportsAllDrives": True}]


def test_drive_descarga_falla_antes_de_responder_si_no_hay_acceso(drive):
    repositorio, archivos = drive
    archivos.error = _http_error(404)
    with pytest.raises(ErrorRepositorio, match="no existe en Drive"):
        repositorio.descargar("ID")  # el error sale al llamar, no al iterar


def test_drive_subir_envia_carpeta_padre_y_devuelve_el_id(drive, pdf):
    repositorio, archivos = drive
    assert repositorio.subir(pdf, "x.pdf", "application/pdf") == "nuevo-id"
    [kw] = _kw(archivos, "create")
    assert kw["body"] == {"name": "x.pdf", "parents": ["CARPETA"]}
    assert kw["supportsAllDrives"] and kw["media_body"][3] is True  # resumable


def test_drive_retirar_manda_a_la_papelera_y_nunca_borra(drive):
    repositorio, archivos = drive
    repositorio.retirar("ID")
    [kw] = _kw(archivos, "update")
    assert kw["body"] == {"trashed": True} and kw["supportsAllDrives"] and kw["fileId"] == "ID"


@pytest.mark.parametrize("operacion", ["subir", "retirar"])
def test_drive_403_al_escribir_dice_que_falta_permiso_de_editor(drive, pdf, operacion):
    repositorio, archivos = drive
    archivos.error = _http_error(403, "insufficientFilePermissions")
    with pytest.raises(ErrorRepositorio, match="no tiene permiso de Editor sobre la carpeta"):
        repositorio.subir(pdf, "x.pdf", "application/pdf") if operacion == "subir" else repositorio.retirar("ID")


def test_drive_403_al_leer_habla_de_permiso_de_lectura(drive):
    repositorio, archivos = drive
    archivos.error = _http_error(403, "forbidden")
    with pytest.raises(ErrorRepositorio, match="permiso de lectura"):
        repositorio.listar()


def test_drive_401_y_limite_de_cuota_tienen_mensajes_propios(drive):
    repositorio, archivos = drive
    archivos.error = _http_error(401)
    with pytest.raises(ErrorRepositorio, match="token de Drive"):
        repositorio.listar()
    archivos.error = _http_error(403, "userRateLimitExceeded")
    with pytest.raises(ErrorRepositorio, match="limitando"):
        repositorio.listar()


def test_drive_enlace_ver(drive):
    repositorio, archivos = drive
    assert repositorio.enlace_ver("ID").startswith("https://drive.google.com/")
    assert _kw(archivos, "get")[0]["fields"] == "webViewLink"


# ── credenciales OAuth ──

@pytest.fixture
def token(tmp_path, monkeypatch):
    ruta = tmp_path / "oauth-token.json"
    monkeypatch.setattr(repo.config, "GOOGLE_OAUTH_TOKEN_FILE", ruta)
    repo.reiniciar_credenciales()
    yield ruta
    repo.reiniciar_credenciales()


def test_sin_archivo_de_token_el_error_es_claro(token):
    with pytest.raises(ErrorRepositorio, match="autorizar_drive.py"):
        repo._cargar_credenciales()


def test_token_dañado_es_un_error_claro(token):
    token.write_text("{ no es json", encoding="utf-8")
    with pytest.raises(ErrorRepositorio, match="dañado"):
        repo._cargar_credenciales()


def test_token_caducado_que_no_se_puede_renovar_es_un_error_claro(token, monkeypatch):
    token.write_text("{}", encoding="utf-8")

    class Caducado:
        valid, refresh_token = False, "r"

        def refresh(self, _):
            raise RefreshError("invalid_grant")

    monkeypatch.setattr(repo.Credentials, "from_authorized_user_file", lambda *a, **k: Caducado())
    with pytest.raises(ErrorRepositorio, match="caducó"):
        repo._cargar_credenciales()


def test_token_vencido_se_renueva_y_se_guarda(token, monkeypatch):
    token.write_text("{}", encoding="utf-8")

    class Vencido:
        valid, refresh_token = False, "r"

        def refresh(self, _):
            self.valid = True

        def to_json(self):
            return '{"token": "nuevo"}'

    monkeypatch.setattr(repo.Credentials, "from_authorized_user_file", lambda *a, **k: Vencido())
    assert repo._cargar_credenciales().valid
    assert json.loads(token.read_text(encoding="utf-8")) == {"token": "nuevo"}


def test_obtener_repositorio_segun_la_configuracion(monkeypatch, tmp_path):
    monkeypatch.setattr(repo.config, "ARCHIVOS_BACKEND", "local")
    monkeypatch.setattr(repo.config, "UPLOAD_ROOT", tmp_path)
    repo.obtener_repositorio.cache_clear()
    assert repo.obtener_repositorio().proveedor == "local"
    repo.obtener_repositorio.cache_clear()
    monkeypatch.setattr(repo.config, "ARCHIVOS_BACKEND", "drive")
    monkeypatch.setattr(repo.config, "GOOGLE_DRIVE_FOLDER_ID", None)
    with pytest.raises(ErrorRepositorio, match="GOOGLE_DRIVE_FOLDER_ID"):
        repo.obtener_repositorio()
    monkeypatch.setattr(repo.config, "GOOGLE_DRIVE_FOLDER_ID", "CARPETA")
    assert repo.obtener_repositorio().proveedor == "drive"  # crearlo no lee credenciales
    repo.obtener_repositorio.cache_clear()
    monkeypatch.setattr(repo.config, "ARCHIVOS_BACKEND", "nube")
    with pytest.raises(ErrorRepositorio, match="ARCHIVOS_BACKEND"):
        repo.obtener_repositorio()
    repo.obtener_repositorio.cache_clear()


# ── tabla archivos_documento ──

@pytest.fixture
def base(monkeypatch):
    motor = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    db.Base.metadata.create_all(motor)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=motor, autoflush=False, autocommit=False))
    return motor


def test_registrar_obtener_y_eliminar_archivo(base):
    assert db.obtener_archivo("a.pdf") is None
    db.registrar_archivo("a.pdf", "drive", "id-1", titulo="Decreto 1", mime_type="application/pdf", tamano_bytes=99)
    fila = db.obtener_archivo("a.pdf")
    assert (fila.proveedor, fila.id_externo, fila.titulo, fila.tamano_bytes) == ("drive", "id-1", "Decreto 1", 99)
    assert fila.creado_en is not None
    assert db.eliminar_registro_archivo("a.pdf") is True
    assert db.obtener_archivo("a.pdf") is None
    assert db.eliminar_registro_archivo("a.pdf") is False


def test_registrar_archivo_actualiza_si_ya_existe(base):
    db.registrar_archivo("a.pdf", "local", "viejo")
    db.registrar_archivo("a.pdf", "drive", "nuevo", titulo="T")
    fila = db.obtener_archivo("a.pdf")
    assert (fila.proveedor, fila.id_externo, fila.titulo) == ("drive", "nuevo", "T")
