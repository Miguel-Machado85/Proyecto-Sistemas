"""Repositorio de los archivos originales (PDF/PNG) que el agente entrega al ciudadano.

Dos implementaciones con la misma interfaz:
- `RepositorioDrive`: carpeta de Google Drive, con OAuth de usuario (token de `scripts/autorizar_drive.py`).
- `RepositorioLocal`: carpeta en disco (`UPLOAD_ROOT/repositorio`), para desarrollo y tests sin credenciales.

Principios:
- Nada se conecta al importar ni al arrancar la API: las credenciales se cargan en la primera operación. Si
  faltan o caducaron, solo falla la operación (`ErrorRepositorio`) y la API sigue respondiendo el chat.
- `retirar()` mueve a la papelera; NUNCA borra de forma permanente.
- El resto del sistema guarda un `id_externo` (clave), no una URL.
"""

import json
import logging
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Iterator, Protocol

from google.auth.exceptions import GoogleAuthError, RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from .config import config

logger = logging.getLogger(__name__)

ALCANCE_DRIVE = ["https://www.googleapis.com/auth/drive"]
TAMANO_TROZO = 1024 * 1024
MIME_CARPETA = "application/vnd.google-apps.folder"

MENSAJE_SIN_EDITOR = "el usuario de Drive no tiene permiso de Editor sobre la carpeta"
MENSAJE_TOKEN = (
    "el token de Drive no es válido o caducó; vuelve a generarlo con scripts/autorizar_drive.py"
)


class ErrorRepositorio(Exception):
    """Fallo del repositorio de archivos, con un mensaje claro en español."""


@dataclass(frozen=True)
class ArchivoRemoto:
    nombre: str
    id_externo: str
    mime_type: str | None = None
    tamano_bytes: int | None = None


class RepositorioArchivos(Protocol):
    proveedor: str

    def subir(self, ruta: Path, nombre: str, mime_type: str) -> str:
        """Sube el archivo y devuelve su `id_externo`."""

    def descargar(self, id_externo: str) -> Iterator[bytes]:
        """Trozos del archivo. Los fallos de acceso se lanzan al llamar, antes del primer trozo."""

    def retirar(self, id_externo: str) -> None:
        """Mueve el archivo a la papelera (nunca lo borra de forma permanente)."""

    def listar(self) -> list[ArchivoRemoto]:
        """Archivos de la carpeta del repositorio."""

    def enlace_ver(self, id_externo: str) -> str | None:
        """Enlace para ver el archivo en el proveedor (None si no existe uno)."""


# ── Local ──

_PATRON_ID_LOCAL = re.compile(r"^[0-9a-f]{32}$")


class RepositorioLocal:
    """Carpeta en disco. El nombre del archivo nunca forma parte de la ruta: se guarda con un id aleatorio."""

    proveedor = "local"

    def __init__(self, raiz: Path | None = None):
        self.raiz = Path(raiz) if raiz else config.UPLOAD_ROOT / "repositorio"

    def _rutas(self, id_externo: str) -> tuple[Path, Path]:
        if not _PATRON_ID_LOCAL.match(id_externo or ""):
            raise ErrorRepositorio("El identificador del archivo no es válido.")
        return self.raiz / id_externo, self.raiz / f"{id_externo}.meta.json"

    def subir(self, ruta: Path, nombre: str, mime_type: str) -> str:
        id_externo = uuid.uuid4().hex
        contenido, meta = self._rutas(id_externo)
        self.raiz.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ruta, contenido)
        meta.write_text(
            json.dumps({"nombre": nombre, "mime_type": mime_type, "tamano_bytes": contenido.stat().st_size}),
            encoding="utf-8",
        )
        return id_externo

    def descargar(self, id_externo: str) -> Iterator[bytes]:
        contenido, _ = self._rutas(id_externo)
        if not contenido.is_file():
            raise ErrorRepositorio("El archivo no existe en el repositorio local.")

        def trozos() -> Iterator[bytes]:
            with open(contenido, "rb") as archivo:
                while trozo := archivo.read(TAMANO_TROZO):
                    yield trozo

        return trozos()

    def retirar(self, id_externo: str) -> None:
        contenido, meta = self._rutas(id_externo)
        if not contenido.is_file():
            raise ErrorRepositorio("El archivo no existe en el repositorio local.")
        papelera = self.raiz / "_papelera"
        papelera.mkdir(parents=True, exist_ok=True)
        shutil.move(str(contenido), papelera / contenido.name)
        if meta.exists():
            shutil.move(str(meta), papelera / meta.name)

    def listar(self) -> list[ArchivoRemoto]:
        if not self.raiz.is_dir():
            return []
        archivos = []
        for meta in sorted(self.raiz.glob("*.meta.json")):
            datos = json.loads(meta.read_text(encoding="utf-8"))
            archivos.append(ArchivoRemoto(
                nombre=datos["nombre"], id_externo=meta.name.removesuffix(".meta.json"),
                mime_type=datos.get("mime_type"), tamano_bytes=datos.get("tamano_bytes"),
            ))
        return archivos

    def enlace_ver(self, id_externo: str) -> str | None:
        return None


# ── Drive ──

def _razon(error: HttpError) -> str:
    try:
        return json.loads(error.content.decode("utf-8"))["error"]["errors"][0]["reason"]
    except Exception:
        return ""


def _traducir_error(error: Exception, operacion: str, escritura: bool = False) -> ErrorRepositorio:
    """Convierte un fallo de Drive o de autenticación en un mensaje claro."""
    if isinstance(error, ErrorRepositorio):
        return error
    if isinstance(error, (RefreshError, GoogleAuthError)):
        return ErrorRepositorio(f"No se pudo {operacion}: {MENSAJE_TOKEN}.")
    if isinstance(error, HttpError):
        estado = error.resp.status
        razon = _razon(error)
        if estado == 401:
            return ErrorRepositorio(f"No se pudo {operacion}: {MENSAJE_TOKEN}.")
        if estado == 429 or "ratelimit" in razon.lower():
            return ErrorRepositorio(f"No se pudo {operacion}: Drive está limitando las solicitudes; intenta de nuevo en unos minutos.")
        if estado == 403 and escritura:
            return ErrorRepositorio(f"No se pudo {operacion}: {MENSAJE_SIN_EDITOR}.")
        if estado == 403:
            return ErrorRepositorio(f"No se pudo {operacion}: el usuario de Drive no tiene permiso de lectura sobre el archivo o la carpeta.")
        if estado == 404:
            return ErrorRepositorio(f"No se pudo {operacion}: el archivo o la carpeta no existe en Drive (o el usuario no tiene acceso).")
        return ErrorRepositorio(f"No se pudo {operacion}: Drive respondió HTTP {estado}.")
    return ErrorRepositorio(f"No se pudo {operacion}: {type(error).__name__}.")


_candado_credenciales = threading.Lock()
_credenciales: Credentials | None = None


def _cargar_credenciales() -> Credentials:
    """Carga (una vez) y refresca el token OAuth. Lanza ErrorRepositorio si falta, está dañado o caducó."""
    global _credenciales
    with _candado_credenciales:
        ruta = config.GOOGLE_OAUTH_TOKEN_FILE
        if _credenciales is None:
            if not ruta.is_file():
                raise ErrorRepositorio(
                    f"No hay token de Drive en {ruta}; genéralo con scripts/autorizar_drive.py."
                )
            try:
                _credenciales = Credentials.from_authorized_user_file(str(ruta), ALCANCE_DRIVE)
            except (ValueError, KeyError, OSError) as error:
                raise ErrorRepositorio(f"El archivo de token de Drive está dañado: {MENSAJE_TOKEN}.") from error
        if not _credenciales.valid:
            if not _credenciales.refresh_token:
                _credenciales = None
                raise ErrorRepositorio(f"{MENSAJE_TOKEN[0].upper()}{MENSAJE_TOKEN[1:]}.")
            try:
                _credenciales.refresh(Request())
            except Exception as error:
                _credenciales = None
                raise _traducir_error(error, "renovar el acceso a Drive") from error
            try:
                ruta.write_text(_credenciales.to_json(), encoding="utf-8")
            except OSError:
                logger.warning("No se pudo guardar el token renovado en %s (¿montaje de solo lectura?).", ruta)
        return _credenciales


def reiniciar_credenciales() -> None:
    global _credenciales
    with _candado_credenciales:
        _credenciales = None


class RepositorioDrive:
    """Carpeta de Google Drive. Todas las llamadas llevan supportsAllDrives=True."""

    proveedor = "drive"

    def __init__(self, carpeta_id: str):
        self.carpeta_id = carpeta_id

    def _servicio(self):
        # Un servicio por operación: el cliente HTTP de googleapiclient no es seguro entre hilos.
        return build("drive", "v3", credentials=_cargar_credenciales(), cache_discovery=False)

    def listar(self) -> list[ArchivoRemoto]:
        try:
            servicio = self._servicio()
            archivos, pagina = [], None
            while True:
                respuesta = servicio.files().list(
                    q=f"'{self.carpeta_id}' in parents and trashed = false and mimeType != '{MIME_CARPETA}'",
                    fields="nextPageToken, files(id, name, mimeType, size)",
                    pageSize=1000,
                    pageToken=pagina,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                ).execute()
                for dato in respuesta.get("files", []):
                    archivos.append(ArchivoRemoto(
                        nombre=dato["name"], id_externo=dato["id"], mime_type=dato.get("mimeType"),
                        tamano_bytes=int(dato["size"]) if dato.get("size") else None,
                    ))
                pagina = respuesta.get("nextPageToken")
                if not pagina:
                    return archivos
        except Exception as error:
            raise _traducir_error(error, "listar la carpeta de Drive") from error

    def descargar(self, id_externo: str) -> Iterator[bytes]:
        # Se pide el primer trozo aquí para que un 403/404 salga antes de empezar a responder.
        try:
            solicitud = self._servicio().files().get_media(fileId=id_externo, supportsAllDrives=True)
            buffer = BytesIO()
            descarga = MediaIoBaseDownload(buffer, solicitud, chunksize=TAMANO_TROZO)
            _, terminado = descarga.next_chunk()
        except Exception as error:
            raise _traducir_error(error, "descargar el archivo de Drive") from error

        def trozos() -> Iterator[bytes]:
            nonlocal terminado
            while True:
                dato = buffer.getvalue()
                buffer.seek(0)
                buffer.truncate()
                if dato:
                    yield dato
                if terminado:
                    return
                try:
                    _, terminado = descarga.next_chunk()
                except Exception as error:
                    raise _traducir_error(error, "descargar el archivo de Drive") from error

        return trozos()

    def subir(self, ruta: Path, nombre: str, mime_type: str) -> str:
        try:
            medio = MediaFileUpload(str(ruta), mimetype=mime_type, resumable=True)
            creado = self._servicio().files().create(
                body={"name": nombre, "parents": [self.carpeta_id]},
                media_body=medio,
                fields="id",
                supportsAllDrives=True,
            ).execute()
            return creado["id"]
        except Exception as error:
            raise _traducir_error(error, "subir el archivo a Drive", escritura=True) from error

    def retirar(self, id_externo: str) -> None:
        # Papelera de Drive (recuperable). Nunca files().delete(): borraría de forma permanente.
        try:
            self._servicio().files().update(
                fileId=id_externo, body={"trashed": True}, fields="id", supportsAllDrives=True,
            ).execute()
        except Exception as error:
            raise _traducir_error(error, "retirar el archivo de Drive", escritura=True) from error

    def enlace_ver(self, id_externo: str) -> str | None:
        try:
            datos = self._servicio().files().get(
                fileId=id_externo, fields="webViewLink", supportsAllDrives=True,
            ).execute()
            return datos.get("webViewLink")
        except Exception as error:
            raise _traducir_error(error, "obtener el enlace del archivo en Drive") from error


@lru_cache(maxsize=1)
def obtener_repositorio() -> RepositorioArchivos:
    """Repositorio según ARCHIVOS_BACKEND. Crearlo no abre conexiones ni lee credenciales."""
    if config.ARCHIVOS_BACKEND == "local":
        return RepositorioLocal()
    if config.ARCHIVOS_BACKEND == "drive":
        if not config.GOOGLE_DRIVE_FOLDER_ID:
            raise ErrorRepositorio("Falta GOOGLE_DRIVE_FOLDER_ID para usar Google Drive.")
        return RepositorioDrive(config.GOOGLE_DRIVE_FOLDER_ID)
    raise ErrorRepositorio(f"ARCHIVOS_BACKEND inválido: {config.ARCHIVOS_BACKEND!r} (usa 'local' o 'drive').")
