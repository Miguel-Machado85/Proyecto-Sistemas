"""Endpoints protegidos para la carga y la indexación administrativa."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import config
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


def verificar_administrador(x_admin_token: Annotated[str | None, Header()] = None) -> None:
    """La protección se aplica al API, nunca solo a la pantalla administrativa."""
    if not config.ADMIN_UPLOAD_TOKEN:
        raise HTTPException(503, "La carga administrativa no está configurada en el servidor.")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, config.ADMIN_UPLOAD_TOKEN):
        raise HTTPException(401, "No autorizado.")


@router.post("/lotes")
async def cargar_lote(
    files: Annotated[list[UploadFile], File(...)],
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
):
    verificar_administrador(x_admin_token)
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
    lote_id: str, x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None
):
    verificar_administrador(x_admin_token)
    try:
        return await run_in_threadpool(obtener_lote, lote_id)
    except ErrorIngesta as error:
        raise HTTPException(404, str(error)) from error


@router.post("/lotes/{lote_id}/indexar", status_code=202)
async def confirmar_indexacion(
    lote_id: str,
    peticion: IndexarLoteRequest,
    tareas: BackgroundTasks,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
):
    verificar_administrador(x_admin_token)
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
    lote_id: str, x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None
):
    verificar_administrador(x_admin_token)
    try:
        return await run_in_threadpool(cancelar_lote, lote_id)
    except ErrorIngesta as error:
        raise HTTPException(400, str(error)) from error


@router.delete("/lotes/{lote_id}/documentos/{archivo}")
async def quitar_documento(
    lote_id: str, archivo: str, x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None
):
    verificar_administrador(x_admin_token)
    try:
        return await run_in_threadpool(eliminar_documento_lote, lote_id, archivo)
    except ErrorIngesta as error:
        raise HTTPException(400, str(error)) from error


@router.post("/lotes/{lote_id}/revertir-indexacion")
async def revertir_lote(
    lote_id: str, x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None
):
    verificar_administrador(x_admin_token)
    try:
        return await run_in_threadpool(revertir_indexacion, lote_id)
    except ErrorIngesta as error:
        raise HTTPException(400, str(error)) from error
