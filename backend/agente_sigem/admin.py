"""Endpoints protegidos para la carga y la indexación administrativa."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import config
from .autenticacion import (
    borrar_cookie_sesion,
    configurar_cookie_sesion,
    crear_jwt_administrador,
    requerir_administrador,
    validar_jwt_administrador,
    verificar_credenciales,
)
from .ingest import (
    ErrorIngesta, cancelar_lote, crear_lote, eliminar_documento_lote, indexar_lote,
    iniciar_indexacion, obtener_lote,
    revertir_indexacion,
)


router = APIRouter(prefix="/admin", tags=["administración"])


class MetadataDocumento(BaseModel):
    archivo: str
    titulo: str | None = None
    descripcion: str | None = None
    categoria: str | None = None
    fecha_expedicion: str | None = None


class IndexarLoteRequest(BaseModel):
    documentos: list[MetadataDocumento] = Field(default_factory=list)


class LoginRequest(BaseModel):
    usuario: str = Field(min_length=1, max_length=100)
    contrasena: str = Field(min_length=1, max_length=256)


# SPRINT 7: acceso administrativo con usuario, contraseña y sesión JWT.
@router.post("/login", status_code=204)
def iniciar_sesion(peticion: LoginRequest) -> Response:
    """Valida las credenciales y guarda el JWT firmado en una cookie HttpOnly."""
    if not verificar_credenciales(peticion.usuario, peticion.contrasena):
        raise HTTPException(401, "Usuario o contraseña incorrectos.")
    respuesta = Response(status_code=204)
    configurar_cookie_sesion(respuesta, crear_jwt_administrador())
    return respuesta


# SPRINT 7: el frontend consulta este endpoint antes de mostrar admin.html.
@router.get("/sesion")
def consultar_sesion(administrador: Annotated[dict, Depends(requerir_administrador)]) -> dict:
    return {"autenticado": True, "usuario": administrador["sub"]}


# SPRINT 7: cierra la sesión del navegador borrando la cookie HttpOnly.
@router.post("/logout", status_code=204)
def cerrar_sesion() -> Response:
    respuesta = Response(status_code=204)
    borrar_cookie_sesion(respuesta)
    return respuesta


@router.post("/lotes")
async def cargar_lote(
    files: Annotated[list[UploadFile], File(...)],
    _: Annotated[dict, Depends(requerir_administrador)] = None,
):
    archivos = []
    for archivo in files:
        contenido = await archivo.read(config.MAX_DOCUMENT_BYTES + 1)
        archivos.append((archivo.filename or "", contenido))
    try:
        return await run_in_threadpool(crear_lote, archivos)
    except ErrorIngesta as error:
        raise HTTPException(400, str(error)) from error


@router.get("/lotes/{lote_id}")
async def consultar_lote(
    lote_id: str, _: Annotated[dict, Depends(requerir_administrador)] = None
):
    try:
        return await run_in_threadpool(obtener_lote, lote_id)
    except ErrorIngesta as error:
        raise HTTPException(404, str(error)) from error


@router.post("/lotes/{lote_id}/indexar", status_code=202)
async def confirmar_indexacion(
    lote_id: str,
    peticion: IndexarLoteRequest,
    tareas: BackgroundTasks,
    _: Annotated[dict, Depends(requerir_administrador)] = None,
):
    cambios = {
        documento.archivo: documento.model_dump(exclude={"archivo"}, exclude_none=True)
        for documento in peticion.documentos
    }
    try:
        lote = await run_in_threadpool(iniciar_indexacion, lote_id)
        tareas.add_task(indexar_lote, lote_id, cambios)
        return lote
    except ErrorIngesta as error:
        raise HTTPException(400, str(error)) from error


@router.post("/lotes/{lote_id}/cancelar")
async def cancelar_operacion(
    lote_id: str, _: Annotated[dict, Depends(requerir_administrador)] = None
):
    try:
        return await run_in_threadpool(cancelar_lote, lote_id)
    except ErrorIngesta as error:
        raise HTTPException(400, str(error)) from error


@router.delete("/lotes/{lote_id}/documentos/{archivo}")
async def quitar_documento(
    lote_id: str, archivo: str, _: Annotated[dict, Depends(requerir_administrador)] = None
):
    try:
        return await run_in_threadpool(eliminar_documento_lote, lote_id, archivo)
    except ErrorIngesta as error:
        raise HTTPException(400, str(error)) from error


@router.post("/lotes/{lote_id}/revertir-indexacion")
async def revertir_lote(
    lote_id: str, _: Annotated[dict, Depends(requerir_administrador)] = None
):
    try:
        return await run_in_threadpool(revertir_indexacion, lote_id)
    except ErrorIngesta as error:
        raise HTTPException(400, str(error)) from error
