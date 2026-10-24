"""Re-ranking de fragmentos con un cross-encoder de sentence-transformers.

Nunca debe romper la búsqueda: si está apagado, no se puede importar o falla,
`reordenar` devuelve la lista recibida sin cambios (con un aviso en el log una sola vez).
"""

import logging
import threading

from langchain_core.documents import Document

from .config import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_modelo = None
_no_disponible = False
_avisos_emitidos: set[str] = set()


def _avisar_una_vez(clave: str, mensaje: str, *args) -> None:
    if clave not in _avisos_emitidos:
        _avisos_emitidos.add(clave)
        logger.warning(mensaje, *args)


def _cargar_modelo():
    """Carga perezosa y única del CrossEncoder; devuelve None si no es posible."""
    global _modelo, _no_disponible
    if _modelo is not None or _no_disponible:
        return _modelo
    with _lock:
        if _modelo is None and not _no_disponible:
            try:
                from sentence_transformers import CrossEncoder

                _modelo = CrossEncoder(config.RERANKER_MODEL, max_length=512)
                logger.info("Reranker cargado: %s", config.RERANKER_MODEL)
            except Exception as error:
                _no_disponible = True
                _avisar_una_vez("carga", "Reranker no disponible, se usa el orden original: %s", error)
    return _modelo


def precargar() -> bool:
    """Carga el modelo al iniciar la app para que la primera consulta no sea lenta."""
    if not config.RERANKER_HABILITADO:
        return False
    return _cargar_modelo() is not None


def reordenar_si_aplica(
    consulta: str, documentos: list[Document], top_n: int | None = None
) -> list[Document] | None:
    """Como `reordenar`, pero devuelve None si el reranker no se pudo aplicar (para usar un respaldo)."""
    if not config.RERANKER_HABILITADO:
        return None
    if len(documentos) < 2:
        return list(documentos)

    modelo = _cargar_modelo()
    if modelo is None:
        return None

    try:
        pares = [(consulta, documento.page_content) for documento in documentos]
        puntajes = modelo.predict(pares)
        ordenados = [
            documento
            for _, documento in sorted(zip(puntajes, documentos), key=lambda par: float(par[0]), reverse=True)
        ]
    except Exception as error:
        _avisar_una_vez("predict", "Falló el reranker, se usa el orden original: %s", error)
        return None

    return ordenados[:top_n] if top_n else ordenados


def reordenar(consulta: str, documentos: list[Document], top_n: int | None = None) -> list[Document]:
    """Ordena los documentos por relevancia frente a la consulta (mayor puntaje primero).

    Si el reranker está deshabilitado o falla, devuelve `documentos` sin cambios.
    """
    reordenados = reordenar_si_aplica(consulta, documentos, top_n)
    return documentos if reordenados is None else reordenados
