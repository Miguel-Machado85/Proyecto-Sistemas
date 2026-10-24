"""Ingesta administrativa de documentos SIGEM en la colección Chroma existente.

Los archivos se reciben primero en un lote persistente. Solo una confirmación posterior
genera embeddings; así un administrador puede revisar la metadata sugerida y conocer los
errores antes de alterar el índice que consulta el agente.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import fitz
import pytesseract
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from . import db
from .catalogo import extraer_referencia, normalizar
from .config import config
from .repositorio_archivos import ErrorRepositorio, obtener_repositorio
from .retrieval import invalidar_cache_documentos, obtener_catalogo_documentos, obtener_vector_store

logger = logging.getLogger(__name__)

SEPARADORES_CHUNKS = ["\n\n", "\n", ".", " ", ""]
ADVERTENCIA_TITULO_NO_CANONICO = (
    "No se reconoce tipo y número en el título; el documento se encontrará por contenido, "
    "pero no por referencia como 'Decreto 045 de 2026'."
)
_PATRON_REFERENCIA = re.compile(
    r"\b(?P<tipo>decreto|resoluci[oó]n|acuerdo|ley|formato)\s*(?:n[.º°o]*\s*)?(?P<numero>\d{1,5})(?:\s*(?:de|/|-)\s*(?P<anio>(?:19|20)\d{2}))?",
    re.IGNORECASE,
)
_PATRON_FECHA_NUMERICA = re.compile(r"\b(?P<dia>0?[1-9]|[12]\d|3[01])[/-](?P<mes>0?[1-9]|1[0-2])[/-](?P<anio>(?:19|20)\d{2})\b")
_MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
_PATRON_FECHA_TEXTO = re.compile(
    r"\b(?P<dia>0?[1-9]|[12]\d|3[01])\s+de\s+(?P<mes>[a-záéíóú]+)\s+de\s+(?P<anio>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


class ErrorIngesta(ValueError):
    """Error seguro de mostrar al administrador."""


def _ahora() -> str:
    return datetime.now(UTC).isoformat()


def _ruta_lote(lote_id: str, estado: str = "pendientes") -> Path:
    if not re.fullmatch(r"[0-9a-f-]{36}", lote_id):
        raise ErrorIngesta("Identificador de lote inválido.")
    return config.UPLOAD_ROOT / estado / lote_id


def _ruta_manifiesto(lote_id: str) -> Path:
    return _ruta_lote(lote_id) / "manifest.json"


def _leer_manifiesto(lote_id: str) -> dict[str, Any]:
    ruta = _ruta_manifiesto(lote_id)
    if not ruta.is_file():
        raise ErrorIngesta("El lote no existe o ya no está disponible.")
    return json.loads(ruta.read_text(encoding="utf-8"))


def _guardar_manifiesto(lote_id: str, manifiesto: dict[str, Any]) -> None:
    ruta = _ruta_manifiesto(lote_id)
    temporal = ruta.with_name(f".{ruta.name}.{uuid.uuid4().hex}.tmp")
    contenido = json.dumps(manifiesto, ensure_ascii=False, indent=2)

    # El lector puede consultar el lote mientras la indexación actualiza su progreso.
    # Reemplazar el archivo completo evita que llegue a observar un JSON a medio escribir.
    try:
        with temporal.open("w", encoding="utf-8") as archivo:
            archivo.write(contenido)
            archivo.flush()
            os.fsync(archivo.fileno())
        os.replace(temporal, ruta)
    finally:
        temporal.unlink(missing_ok=True)


def _extension(nombre: str) -> str:
    return Path(nombre).suffix.lower().lstrip(".")


def _nombre_seguro(nombre: str) -> str:
    nombre_seguro = Path(nombre).name
    if not nombre_seguro or nombre_seguro in {".", ".."}:
        raise ErrorIngesta("El archivo no tiene un nombre válido.")
    if _extension(nombre_seguro) not in config.ALLOWED_DOCUMENT_EXTENSIONS:
        permitidos = ", ".join(sorted(config.ALLOWED_DOCUMENT_EXTENSIONS)).upper()
        raise ErrorIngesta(f"Formato no permitido. Solo se aceptan {permitidos}.")
    return nombre_seguro


def _sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _fecha_detectada(texto: str) -> str | None:
    coincidencia = _PATRON_FECHA_NUMERICA.search(texto)
    if coincidencia:
        try:
            return datetime(
                int(coincidencia["anio"]), int(coincidencia["mes"]), int(coincidencia["dia"]), 8
            ).isoformat()
        except ValueError:
            return None
    coincidencia = _PATRON_FECHA_TEXTO.search(texto.lower())
    if coincidencia:
        mes = _MESES.get(coincidencia["mes"].replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u"))
        if mes:
            try:
                return datetime(int(coincidencia["anio"]), mes, int(coincidencia["dia"]), 8).isoformat()
            except ValueError:
                return None
    return None


def _titulo_desde_texto(texto: str, archivo: str) -> str:
    for linea in texto.splitlines():
        linea = re.sub(r"\s+", " ", linea).strip(" -:;\t")
        if len(linea) >= 8:
            return linea[:250]
    return Path(archivo).stem.replace("_", " ")


def _metadata_sugerida(archivo: str, texto: str, tecnica: dict[str, Any]) -> dict[str, Any]:
    referencia = _PATRON_REFERENCIA.search(texto) or _PATRON_REFERENCIA.search(archivo.replace("_", " "))
    titulo = _titulo_desde_texto(texto, archivo)
    if referencia:
        tipo = {
            "decreto": "Decreto", "resolucion": "Resolución", "resolución": "Resolución",
            "acuerdo": "Acuerdo", "ley": "Ley", "formato": "Formato",
        }[referencia["tipo"].lower()]
        numero = int(referencia["numero"])
        anio = referencia["anio"]
        titulo = f"{tipo} {numero:03d}" + (f" de {anio}" if anio else "")
    descripcion = re.sub(r"\s+", " ", texto).strip()[:700]
    return {
        "titulo": titulo,
        "descripcion": descripcion or None,
        "categoria": None,
        "fecha_expedicion": _fecha_detectada(texto),
        **tecnica,
    }


def _ocr_imagen(imagen: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(imagen, lang="spa+eng").strip()
    except pytesseract.TesseractNotFoundError as error:
        raise ErrorIngesta("El servidor no tiene instalado el motor OCR requerido.") from error


def _extraer_pdf(ruta: Path) -> tuple[list[str], dict[str, Any]]:
    try:
        pdf = fitz.open(ruta)
    except Exception as error:
        raise ErrorIngesta("El PDF está dañado o no se puede abrir.") from error
    try:
        paginas: list[str] = []
        uso_ocr = False
        for pagina in pdf:
            texto = pagina.get_text("text").strip()
            if not texto:
                uso_ocr = True
                pixmap = pagina.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                texto = _ocr_imagen(Image.open(io.BytesIO(pixmap.tobytes("png"))))
            paginas.append(texto)
        if not any(paginas):
            raise ErrorIngesta("No se pudo extraer texto del PDF, ni siquiera mediante OCR.")
        meta = pdf.metadata or {}
        tecnica = {
            "total_pages": len(pdf),
            "requirio_ocr": uso_ocr,
            "author": meta.get("author") or None,
            "creator": meta.get("creator") or None,
            "producer": meta.get("producer") or None,
            "creationdate": meta.get("creationDate") or None,
            "moddate": meta.get("modDate") or None,
            "keywords": meta.get("keywords") or None,
            "title": meta.get("title") or None,
            "subject": meta.get("subject") or None,
        }
        return paginas, tecnica
    finally:
        pdf.close()


def _extraer_png(ruta: Path) -> tuple[list[str], dict[str, Any]]:
    try:
        with Image.open(ruta) as imagen:
            texto = _ocr_imagen(imagen.convert("RGB"))
            tecnica = {
                "total_pages": 1,
                "requirio_ocr": True,
                "author": None,
                "creator": None,
                "producer": None,
                "creationdate": None,
                "moddate": None,
                "keywords": None,
                "title": None,
                "subject": None,
                "image_width": imagen.width,
                "image_height": imagen.height,
            }
    except ErrorIngesta:
        raise
    except Exception as error:
        raise ErrorIngesta("La imagen PNG está dañada o no se puede abrir.") from error
    if not texto:
        raise ErrorIngesta("No se pudo reconocer texto en la imagen PNG.")
    return [texto], tecnica


def extraer_documento(ruta: Path) -> tuple[list[str], dict[str, Any]]:
    """Extrae páginas y una metadata inicial que el administrador puede corregir."""
    if _extension(ruta.name) == "pdf":
        paginas, tecnica = _extraer_pdf(ruta)
    else:
        paginas, tecnica = _extraer_png(ruta)
    return paginas, _metadata_sugerida(ruta.name, "\n".join(paginas[:2]), tecnica)


def _clave_identidad(texto: str) -> str:
    """Compara títulos y nombres sin depender de tildes, extensión o puntuación."""
    return re.sub(r"[^a-z0-9]", "", normalizar(Path(texto).stem))


def _documento_existente(archivo: str, titulo: str) -> str | None:
    """Busca por nombre original y título, los dos identificadores de documentos históricos."""
    claves_candidatas = {_clave_identidad(archivo), _clave_identidad(titulo)} - {""}
    if not claves_candidatas:
        return None
    return next(
        (
            documento.titulo
            for documento in obtener_catalogo_documentos().documentos
            if claves_candidatas
            & {_clave_identidad(documento.archivo), _clave_identidad(documento.titulo)}
        ),
        None,
    )


def _rechazar_archivo(lote_id: str, ruta: Path | None) -> None:
    if ruta and ruta.is_file():
        rechazados = _ruta_lote(lote_id, "rechazados")
        rechazados.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ruta), rechazados / ruta.name)


def crear_lote(archivos: list[tuple[str, bytes]]) -> dict[str, Any]:
    """Guarda y valida archivos, sin crear embeddings todavía."""
    if not archivos:
        raise ErrorIngesta("Selecciona al menos un documento.")
    if len(archivos) > config.MAX_DOCUMENTS_PER_BATCH:
        raise ErrorIngesta(f"Solo se permiten {config.MAX_DOCUMENTS_PER_BATCH} documentos por lote.")

    lote_id = str(uuid.uuid4())
    carpeta = _ruta_lote(lote_id)
    carpeta.mkdir(parents=True, exist_ok=False)
    documentos = []
    nombres = set()
    identificadores_lote = set()
    for nombre_original, contenido in archivos:
        entrada: dict[str, Any] = {"archivo": Path(nombre_original).name, "estado": "rechazado", "error": None}
        ruta: Path | None = None
        try:
            nombre = _nombre_seguro(nombre_original)
            if nombre.lower() in nombres:
                raise ErrorIngesta("Hay nombres de archivo repetidos en el mismo lote.")
            nombres.add(nombre.lower())
            if not contenido:
                raise ErrorIngesta("El archivo está vacío.")
            if len(contenido) > config.MAX_DOCUMENT_BYTES:
                raise ErrorIngesta(f"El archivo supera el límite de {config.MAX_DOCUMENT_BYTES // (1024 * 1024)} MB.")
            ruta = carpeta / nombre
            ruta.write_bytes(contenido)
            paginas, metadata = extraer_documento(ruta)
            titulo = str(metadata["titulo"])
            existente = _documento_existente(nombre, titulo)
            if existente:
                raise ErrorIngesta(f"Ya existe en Chroma un documento con el título '{existente}'.")
            identificadores = {_clave_identidad(nombre), _clave_identidad(titulo)} - {""}
            if identificadores_lote & identificadores:
                raise ErrorIngesta("Hay otro archivo del lote con el mismo nombre o título detectado.")
            identificadores_lote.update(identificadores)
            entrada.update({
                "archivo": nombre,
                "estado": "listo",
                "tamano_bytes": len(contenido),
                "paginas_detectadas": len(paginas),
                "metadata_sugerida": metadata,
                "hash_archivo": _sha256(ruta),
            })
        except ErrorIngesta as error:
            entrada["error"] = str(error)
        except Exception:
            entrada["error"] = "No fue posible preparar el archivo para indexación."
        if entrada["estado"] == "rechazado":
            _rechazar_archivo(lote_id, ruta)
        documentos.append(entrada)

    manifiesto = {"lote_id": lote_id, "creado_en": _ahora(), "estado": "pendiente", "documentos": documentos}
    _guardar_manifiesto(lote_id, manifiesto)
    return manifiesto


def obtener_lote(lote_id: str) -> dict[str, Any]:
    return _leer_manifiesto(lote_id)


def _metadata_final(sugerida: dict[str, Any], cambios: dict[str, Any] | None) -> dict[str, Any]:
    permitidos = {"titulo", "descripcion", "categoria", "fecha_expedicion"}
    cambios = cambios or {}
    resultado = dict(sugerida)
    for campo in permitidos:
        if campo in cambios:
            valor = cambios[campo]
            resultado[campo] = valor.strip() if isinstance(valor, str) else valor
    if not resultado.get("titulo"):
        raise ErrorIngesta("Cada documento debe tener un título.")
    return resultado


def _ya_indexado(hash_archivo: str) -> bool:
    try:
        existente = obtener_vector_store().get(where={"hash_archivo": hash_archivo}, include=[])
        return bool(existente.get("ids"))
    except Exception:
        # Los documentos antiguos no tienen hash; una falla de esta optimización no bloquea la carga.
        return False


def _crear_chunks(ruta: Path, metadata: dict[str, Any], documento_id: str, hash_archivo: str) -> list[Document]:
    paginas, _ = extraer_documento(ruta)
    metadata_base = {
        **metadata,
        "nombre_archivo": ruta.name,
        "source": ruta.name,
        "documento_id": documento_id,
        "hash_archivo": hash_archivo,
        "mime_type": "application/pdf" if _extension(ruta.name) == "pdf" else "image/png",
        "fecha_carga": _ahora(),
    }
    # Chroma acepta valores escalares, no estructuras ni valores nulos. Los campos ausentes
    # se preservan como ausencia, igual que en los documentos históricos.
    metadata_base = {campo: valor for campo, valor in metadata_base.items() if valor is not None}
    documentos = [
        Document(page_content=texto, metadata={**metadata_base, "page": indice, "page_label": str(indice + 1)})
        for indice, texto in enumerate(paginas)
        if texto.strip()
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=SEPARADORES_CHUNKS,
    )
    chunks = splitter.split_documents(documentos)
    for indice, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = indice
    if not chunks:
        raise ErrorIngesta("El documento no produjo fragmentos indexables.")
    return chunks


def _mime_original(ruta: Path) -> str:
    return "application/pdf" if _extension(ruta.name) == "pdf" else "image/png"


def _subir_original(ruta: Path) -> tuple[Any, str]:
    """Paso 1: sube el original al repositorio. Si falla, el documento no se indexa."""
    try:
        repositorio = obtener_repositorio()
        return repositorio, repositorio.subir(ruta, ruta.name, _mime_original(ruta))
    except Exception as error:
        detalle = f" Detalle: {error}" if isinstance(error, ErrorRepositorio) else ""
        logger.warning("No se pudo subir el original de %s: %s", ruta.name, error)
        raise ErrorIngesta(
            "No se pudo guardar el original del documento en el repositorio de archivos, "
            f"así que no se indexó.{detalle}"
        ) from error


def _retirar_original(repositorio: Any, id_externo: str) -> str:
    """Retira (papelera) un original subido; devuelve un aviso si no fue posible, o '' si todo salió bien."""
    try:
        repositorio.retirar(id_externo)
        return ""
    except Exception as error:
        logger.warning("No se pudo retirar el original %s del repositorio: %s", id_externo, error)
        return f" Además, no se pudo retirar el original ya subido ({id_externo}); revísalo en el repositorio de archivos."


def _deshacer_indexacion(vector_store: Any, documento_id: str, repositorio: Any, id_externo: str) -> str:
    """Deja todo como estaba si un paso posterior a la subida falla: Chroma limpio y original en la papelera."""
    try:
        vector_store.delete(where={"documento_id": documento_id})
    except Exception:
        logger.warning("No se pudo limpiar Chroma tras un fallo de indexación (%s).", documento_id, exc_info=True)
    return _retirar_original(repositorio, id_externo)


def _advertencias_de_titulo(titulo: str) -> list[str]:
    """Sin tipo y número reconocibles 'Decreto 045 de 2026' no se encuentra por referencia; no bloquea."""
    return [] if extraer_referencia(titulo) else [ADVERTENCIA_TITULO_NO_CANONICO]


def _indexar_documento(
    ruta: Path,
    entrada: dict[str, Any],
    cambios: dict[str, Any] | None,
    identificadores_lote: set[str],
    reportar_progreso: Callable[[int, int, str], None],
) -> dict[str, Any]:
    hash_archivo = str(entrada["hash_archivo"])
    if _ya_indexado(hash_archivo):
        return {"archivo": ruta.name, "estado": "omitido", "error": "Ya existe un documento con el mismo contenido."}
    documento_id = str(uuid.uuid4())
    metadata = _metadata_final(entrada["metadata_sugerida"], cambios)
    existente = _documento_existente(ruta.name, str(metadata["titulo"]))
    if existente:
        return {"archivo": ruta.name, "estado": "omitido", "error": f"Ya existe en Chroma el título '{existente}'."}
    identificadores = {_clave_identidad(ruta.name), _clave_identidad(str(metadata["titulo"]))} - {""}
    if identificadores_lote & identificadores:
        return {"archivo": ruta.name, "estado": "omitido", "error": "Otro archivo del lote usa el mismo nombre o título."}
    chunks = _crear_chunks(ruta, metadata, documento_id, hash_archivo)
    vector_store = obtener_vector_store()
    embedding_function = getattr(vector_store, "_embedding_function", None)
    collection = getattr(vector_store, "_collection", None)
    if embedding_function is None or collection is None:
        raise ErrorIngesta("No fue posible preparar la conexión de embeddings para Chroma.")
    advertencias = _advertencias_de_titulo(str(metadata["titulo"]))
    reportar_progreso(len(chunks), 0, "subiendo_original")
    repositorio, id_externo = _subir_original(ruta)  # 1. original al repositorio
    embeddings = []
    reportar_progreso(len(chunks), 0, "generando_embeddings")
    try:
        for inicio in range(0, len(chunks), config.EMBEDDING_BATCH_SIZE):
            lote = chunks[inicio : inicio + config.EMBEDDING_BATCH_SIZE]
            embeddings.extend(embedding_function.embed_documents([chunk.page_content for chunk in lote]))
            reportar_progreso(len(chunks), len(embeddings), "generando_embeddings")
            if inicio + config.EMBEDDING_BATCH_SIZE < len(chunks) and config.EMBEDDING_BATCH_PAUSE_SECONDS:
                time.sleep(config.EMBEDDING_BATCH_PAUSE_SECONDS)
        # Chroma recibe el documento completo en una sola operación: ningún chunk aparece
        # en la colección consultable hasta que todos sus embeddings estén listos.
        reportar_progreso(len(chunks), len(chunks), "guardando_en_chroma")
        collection.upsert(
            ids=[f"{documento_id}:{indice}" for indice in range(len(chunks))],
            embeddings=embeddings,
            documents=[chunk.page_content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )
    except Exception as error:  # 2. falló Chroma: no queda nada ni en el índice ni en el repositorio
        aviso = _deshacer_indexacion(vector_store, documento_id, repositorio, id_externo)
        raise ErrorIngesta("No fue posible generar los embeddings del documento." + aviso) from error
    try:  # 3. fila que enlaza el documento con su original
        db.registrar_archivo(
            nombre_archivo=ruta.name,
            proveedor=repositorio.proveedor,
            id_externo=id_externo,
            titulo=str(metadata["titulo"]),
            mime_type=_mime_original(ruta),
            tamano_bytes=ruta.stat().st_size,
            hash_sha256=hash_archivo,
        )
    except Exception as error:
        # Un documento que el agente encuentra pero no puede entregar es peor que un error visible.
        aviso = _deshacer_indexacion(vector_store, documento_id, repositorio, id_externo)
        raise ErrorIngesta("No fue posible registrar el original del documento en la base de datos." + aviso) from error
    identificadores_lote.update(identificadores)
    resultado = {
        "archivo": ruta.name,
        "estado": "indexado",
        "documento_id": documento_id,
        "fragmentos_indexados": len(chunks),
        "metadata": metadata,
        "proveedor_original": repositorio.proveedor,
        "id_externo": id_externo,
    }
    if advertencias:
        resultado["advertencias"] = advertencias
    return resultado


def iniciar_indexacion(lote_id: str) -> dict[str, Any]:
    """Marca el lote como en proceso antes de iniciar el trabajo en segundo plano."""
    manifiesto = _leer_manifiesto(lote_id)
    if manifiesto.get("estado") == "indexado":
        raise ErrorIngesta("El lote ya fue indexado.")
    if manifiesto.get("estado") == "indexando":
        raise ErrorIngesta("El lote ya se está indexando.")
    manifiesto["estado"] = "indexando"
    manifiesto["progreso"] = {
        "fase": "en_espera",
        "archivo_actual": None,
        "documentos_totales": len(manifiesto["documentos"]),
        "documentos_procesados": 0,
        "fragmentos_documento_total": 0,
        "fragmentos_documento_procesados": 0,
    }
    _guardar_manifiesto(lote_id, manifiesto)
    return manifiesto


def indexar_lote(lote_id: str, cambios_por_archivo: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Indexa los documentos listos y conserva los resultados individuales del lote."""
    manifiesto = _leer_manifiesto(lote_id)
    if manifiesto.get("estado") == "indexado":
        return manifiesto
    if manifiesto.get("estado") != "indexando":
        manifiesto = iniciar_indexacion(lote_id)
    cambios_por_archivo = cambios_por_archivo or {}
    resultados = []
    hubo_indexados = False
    identificadores_lote: set[str] = set()
    destino = _ruta_lote(lote_id, "indexados")
    destino.mkdir(parents=True, exist_ok=True)

    for entrada in manifiesto["documentos"]:
        archivo = entrada["archivo"]
        progreso = manifiesto["progreso"]
        progreso.update({"archivo_actual": archivo, "fase": "validando"})
        _guardar_manifiesto(lote_id, manifiesto)
        if entrada.get("estado") != "listo":
            resultados.append({"archivo": archivo, "estado": entrada.get("estado"), "error": entrada.get("error")})
            progreso["documentos_procesados"] += 1
            _guardar_manifiesto(lote_id, manifiesto)
            continue
        try:
            def reportar(total: int, procesados: int, fase: str) -> None:
                progreso.update({
                    "fase": fase,
                    "fragmentos_documento_total": total,
                    "fragmentos_documento_procesados": procesados,
                })
                _guardar_manifiesto(lote_id, manifiesto)

            resultado = _indexar_documento(
                _ruta_lote(lote_id) / archivo,
                entrada,
                cambios_por_archivo.get(archivo),
                identificadores_lote,
                reportar,
            )
            if resultado["estado"] == "indexado":
                shutil.move(str(_ruta_lote(lote_id) / archivo), destino / archivo)
                hubo_indexados = True
            resultados.append(resultado)
        except ErrorIngesta as error:
            resultados.append({"archivo": archivo, "estado": "error", "error": str(error)})
        progreso["documentos_procesados"] += 1
        progreso.update({"fragmentos_documento_total": 0, "fragmentos_documento_procesados": 0})
        _guardar_manifiesto(lote_id, manifiesto)

    manifiesto["estado"] = "indexado"
    manifiesto["indexado_en"] = _ahora()
    manifiesto["resultados"] = resultados
    manifiesto["progreso"].update({"fase": "completado", "archivo_actual": None})
    _guardar_manifiesto(lote_id, manifiesto)
    if hubo_indexados:
        invalidar_cache_documentos()
    return manifiesto


def cancelar_lote(lote_id: str) -> dict[str, str]:
    """Elimina un lote aún no indexado, incluidos los archivos rechazados de esa carga."""
    manifiesto = _leer_manifiesto(lote_id)
    if manifiesto.get("estado") in {"indexado", "indexando"}:
        raise ErrorIngesta("No se puede cancelar un lote que ya fue indexado o está en proceso.")
    shutil.rmtree(_ruta_lote(lote_id))
    shutil.rmtree(_ruta_lote(lote_id, "rechazados"), ignore_errors=True)
    return {"lote_id": lote_id, "estado": "cancelado"}


def eliminar_documento_lote(lote_id: str, archivo: str) -> dict[str, Any]:
    """Quita un único archivo pendiente y su metadata de revisión."""
    manifiesto = _leer_manifiesto(lote_id)
    if manifiesto.get("estado") != "pendiente":
        raise ErrorIngesta("Solo se pueden quitar documentos de un lote pendiente.")
    archivo = _nombre_seguro(archivo)
    documento = next((item for item in manifiesto["documentos"] if item["archivo"] == archivo), None)
    if documento is None:
        raise ErrorIngesta("El documento no pertenece al lote.")
    for estado in ("pendientes", "rechazados"):
        ruta = _ruta_lote(lote_id, estado) / archivo
        if ruta.is_file():
            ruta.unlink()
    manifiesto["documentos"].remove(documento)
    _guardar_manifiesto(lote_id, manifiesto)
    return manifiesto


def _retirar_original_del_lote(lote_id: str, manifiesto: dict[str, Any], resultado: dict[str, Any]) -> None:
    """Manda el original a la papelera y borra su fila. Es reintentable: lo ya retirado queda marcado en el manifiesto."""
    id_externo = resultado.get("id_externo")
    if id_externo and not resultado.get("original_retirado"):
        repositorio = obtener_repositorio()
        proveedor = resultado.get("proveedor_original")
        if proveedor and proveedor != repositorio.proveedor:
            raise ErrorRepositorio(
                f"El original está en '{proveedor}' pero el repositorio activo es '{repositorio.proveedor}'."
            )
        repositorio.retirar(id_externo)
        resultado["original_retirado"] = True
        _guardar_manifiesto(lote_id, manifiesto)
    db.eliminar_registro_archivo(resultado["archivo"])


def revertir_indexacion(lote_id: str) -> dict[str, Any]:
    """Elimina de Chroma, de la tabla de originales y del repositorio (papelera) lo que este lote añadió."""
    manifiesto = _leer_manifiesto(lote_id)
    if manifiesto.get("estado") != "indexado":
        raise ErrorIngesta("Solo se puede revertir un lote cuya indexación haya terminado.")

    resultados_indexados = [resultado for resultado in manifiesto.get("resultados", []) if resultado.get("estado") == "indexado"]
    vector_store = obtener_vector_store()
    errores = []
    motivos = []
    for resultado in resultados_indexados:
        try:
            vector_store.delete(where={"documento_id": resultado["documento_id"]})
        except Exception:
            errores.append(resultado["archivo"])
            continue
        try:
            _retirar_original_del_lote(lote_id, manifiesto, resultado)
        except Exception as error:
            logger.warning("No se pudo retirar el original de %s al revertir: %s", resultado["archivo"], error)
            errores.append(resultado["archivo"])
            if isinstance(error, ErrorRepositorio):
                motivos.append(str(error))
    if errores:
        detalle = (" Motivo: " + "; ".join(sorted(set(motivos)))) if motivos else ""
        raise ErrorIngesta("No fue posible revertir completamente el lote. Intenta de nuevo para: " + ", ".join(errores) + "." + detalle)

    origen = _ruta_lote(lote_id, "indexados")
    destino = _ruta_lote(lote_id)
    for resultado in resultados_indexados:
        archivo = resultado["archivo"]
        if (origen / archivo).is_file():
            shutil.move(str(origen / archivo), destino / archivo)
    if origen.exists():
        shutil.rmtree(origen)

    manifiesto["estado"] = "pendiente"
    manifiesto.pop("indexado_en", None)
    manifiesto.pop("resultados", None)
    manifiesto["progreso"] = {
        "fase": "revertido",
        "archivo_actual": None,
        "documentos_totales": len(manifiesto["documentos"]),
        "documentos_procesados": 0,
        "fragmentos_documento_total": 0,
        "fragmentos_documento_procesados": 0,
    }
    _guardar_manifiesto(lote_id, manifiesto)
    if resultados_indexados:
        invalidar_cache_documentos()
    return manifiesto
