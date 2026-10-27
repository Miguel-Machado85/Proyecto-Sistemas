"""SPRINT 7: autenticación del administrador mediante JWT firmado en una cookie HttpOnly."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from typing import Annotated, Any

import bcrypt
from fastapi import Cookie, HTTPException, Response

from .config import config

NOMBRE_COOKIE_ADMIN = "sigem_admin_session"
ALGORITMO_JWT = "HS256"


def _base64url_codificar(valor: bytes) -> str:
    return base64.urlsafe_b64encode(valor).rstrip(b"=").decode("ascii")


def _base64url_decodificar(valor: str) -> bytes:
    relleno = "=" * (-len(valor) % 4)
    return base64.urlsafe_b64decode(valor + relleno)


def _configuracion_lista() -> bool:
    return bool(config.ADMIN_USERNAME and config.ADMIN_PASSWORD_HASH and config.ADMIN_JWT_SECRET)


def _error_configuracion() -> HTTPException:
    return HTTPException(503, "La autenticación administrativa no está configurada en el servidor.")


def verificar_credenciales(usuario: str, contrasena: str) -> bool:
    """Compara usuario y hash bcrypt sin revelar cuál de los dos falló."""
    if not _configuracion_lista():
        raise _error_configuracion()
    try:
        usuario_valido = hmac.compare_digest(usuario, config.ADMIN_USERNAME or "")
        contrasena_valida = bcrypt.checkpw(contrasena.encode("utf-8"), (config.ADMIN_PASSWORD_HASH or "").encode("utf-8"))
    except (ValueError, TypeError):
        return False
    return usuario_valido and contrasena_valida


def crear_jwt_administrador() -> str:
    """Crea un JWT HS256 de duración limitada para el único administrador."""
    if not _configuracion_lista():
        raise _error_configuracion()
    ahora = int(time.time())
    cabecera = _base64url_codificar(json.dumps({"alg": ALGORITMO_JWT, "typ": "JWT"}, separators=(",", ":")).encode())
    carga = _base64url_codificar(json.dumps({
        "sub": config.ADMIN_USERNAME,
        "rol": "admin",
        "iat": ahora,
        "exp": ahora + config.ADMIN_SESSION_HOURS * 3600,
        "jti": secrets.token_urlsafe(16),
    }, separators=(",", ":")).encode())
    firmado = f"{cabecera}.{carga}".encode("ascii")
    firma = hmac.new((config.ADMIN_JWT_SECRET or "").encode("utf-8"), firmado, hashlib.sha256).digest()
    return f"{cabecera}.{carga}.{_base64url_codificar(firma)}"


def validar_jwt_administrador(token: str | None) -> dict[str, Any]:
    """Verifica firma, algoritmo, rol, usuario y vencimiento del JWT."""
    if not _configuracion_lista():
        raise _error_configuracion()
    if not token:
        raise HTTPException(401, "No autorizado.")
    try:
        cabecera_codificada, carga_codificada, firma_codificada = token.split(".")
        cabecera = json.loads(_base64url_decodificar(cabecera_codificada))
        carga = json.loads(_base64url_decodificar(carga_codificada))
        firma = _base64url_decodificar(firma_codificada)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error):
        raise HTTPException(401, "No autorizado.") from None
    firmado = f"{cabecera_codificada}.{carga_codificada}".encode("ascii")
    esperada = hmac.new((config.ADMIN_JWT_SECRET or "").encode("utf-8"), firmado, hashlib.sha256).digest()
    if cabecera.get("alg") != ALGORITMO_JWT or not hmac.compare_digest(firma, esperada):
        raise HTTPException(401, "No autorizado.")
    if carga.get("rol") != "admin" or not hmac.compare_digest(str(carga.get("sub", "")), config.ADMIN_USERNAME or ""):
        raise HTTPException(401, "No autorizado.")
    try:
        vencimiento = int(carga["exp"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(401, "No autorizado.") from None
    if vencimiento <= int(time.time()):
        raise HTTPException(401, "La sesión expiró. Inicia sesión de nuevo.")
    return carga


def requerir_administrador(
    token: Annotated[str | None, Cookie(alias=NOMBRE_COOKIE_ADMIN)] = None,
) -> dict[str, Any]:
    return validar_jwt_administrador(token)


def configurar_cookie_sesion(respuesta: Response, token: str) -> None:
    respuesta.set_cookie(
        key=NOMBRE_COOKIE_ADMIN,
        value=token,
        max_age=config.ADMIN_SESSION_HOURS * 3600,
        httponly=True,
        secure=config.ADMIN_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def borrar_cookie_sesion(respuesta: Response) -> None:
    respuesta.delete_cookie(key=NOMBRE_COOKIE_ADMIN, path="/")
