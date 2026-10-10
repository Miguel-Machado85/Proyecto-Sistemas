"""Resolución de referencias explícitas a documentos ("Decreto 83 de 2018", "documento 18399").

Esta lógica salió de retrieval.py. Es la versión transitoria con regex sobre el texto del usuario
(`CatalogoDocumentos.resolver`); en la tool `buscar_documento_por_referencia` de la fase 3 el LLM
extrae tipo/número/año y se usa `CatalogoDocumentos.buscar_por_atributos`.
"""

import json

from langchain_core.documents import Document

from . import reranker
from .retrieval import (
    RESULTADOS_FINALES,
    RESULTADOS_POR_ESTRATEGIA,
    documentos_desde_resultado,
    obtener_catalogo_documentos,
    obtener_vector_store,
    recuperar_sigem,
    unir_sin_duplicados,
)


def fragmentos_de_archivo(archivo: str, consulta: str | None = None, limite: int = RESULTADOS_FINALES) -> list[Document]:
    """Fragmentos de un PDF: ordenados por relevancia frente a `consulta` o, sin ella, en el orden de Chroma."""
    resultado = obtener_vector_store().get(
        where={"nombre_archivo": archivo},
        include=["documents", "metadatas"],
        limit=RESULTADOS_POR_ESTRATEGIA,
    )
    documentos = documentos_desde_resultado(resultado)
    if consulta:
        documentos = reranker.reordenar(consulta, documentos)
    return unir_sin_duplicados(documentos, limite=limite)


def buscar_por_referencia(consulta: str) -> list[Document]:
    """Fragmentos del documento citado en la consulta, solo si la referencia es exacta."""
    resolucion = obtener_catalogo_documentos().resolver(consulta)
    if resolucion.estado != "exacta":
        return []
    return fragmentos_de_archivo(resolucion.documentos[0].archivo, consulta)


def candidatos_referencia_ambigua(consulta: str) -> list[dict]:
    """Devuelve candidatos para pedir precisión cuando tipo y número no identifican un único PDF."""
    resolucion = obtener_catalogo_documentos().resolver(consulta)
    if resolucion.estado != "ambigua":
        return []
    return [
        {
            "titulo": documento.titulo,
            "archivo": documento.archivo,
            "fecha_expedicion": str(documento.anio) if documento.anio else None,
            "descripcion": documento.descripcion,
        }
        for documento in resolucion.documentos[:3]
    ]


def respuesta_referencia_ambigua(candidatos: list[dict]) -> str:
    opciones = "; ".join(
        f"{candidato['titulo']} ({candidato['fecha_expedicion'] or 'sin año'})" for candidato in candidatos
    )
    return json.dumps(
        {
            "fuentes": [],
            "fragmentos": [],
            "candidatos": candidatos,
            "aclaracion": f"Encontré varios documentos posibles: {opciones}. Pide al usuario que indique el año o tema.",
        },
        ensure_ascii=False,
    )


def recuperar_con_referencia(consulta: str, archivo: str | None = None) -> list[Document]:
    """Comportamiento anterior de `recuperar_sigem`: una referencia exacta gana sobre la búsqueda híbrida."""
    if archivo is None:
        encontrados = buscar_por_referencia(consulta)
        if encontrados:
            return encontrados
    return recuperar_sigem(consulta, archivo)
