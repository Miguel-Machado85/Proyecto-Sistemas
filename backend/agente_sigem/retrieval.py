import json
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from . import reranker
from .config import config
from .catalogo import CatalogoDocumentos


RESULTADOS_FINALES = 8
RESULTADOS_SEMANTICOS = RESULTADOS_FINALES  # nombre histórico, conservado por compatibilidad
RESULTADOS_POR_ESTRATEGIA = 24
MAX_FRAGMENTOS_POR_DOCUMENTO = 2
PALABRAS_NO_SIGNIFICATIVAS = {
    "acerca", "alcaldia", "cual", "como", "con", "de", "del", "el", "en",
    "es", "esta", "este", "la", "las", "lo", "los", "para", "por", "que",
    "hay", "segun", "sobre", "un", "una", "y",
}
PALABRAS_BUSQUEDA_GENERICAS = {
    "documento", "documentos", "informacion", "norma", "normas", "repositorio", "sigem",
}


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(caracter for caracter in texto if unicodedata.category(caracter) != "Mn")
    return re.sub(r"\s+", " ", texto).strip()


@lru_cache(maxsize=1)
def obtener_vector_store() -> Chroma:
    """Abre Chroma (una sola vez) con el modelo de embeddings que generó sus vectores existentes."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def documentos_desde_resultado(resultado: dict) -> list[Document]:
    documentos = []
    for contenido, metadata in zip(resultado.get("documents", []), resultado.get("metadatas", [])):
        if contenido:
            documentos.append(Document(page_content=contenido, metadata=metadata or {}))
    return documentos


@lru_cache(maxsize=1)
def obtener_catalogo_documentos() -> CatalogoDocumentos:
    """Carga una sola vez las metadata de Chroma como documentos únicos normalizados."""
    vector_store = obtener_vector_store()
    resultado = vector_store.get(include=["metadatas"])
    return CatalogoDocumentos.desde_metadatas(resultado["metadatas"])


def terminos_textuales(consulta: str) -> list[str]:
    palabras = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{3,}", consulta)
    return [
        palabra
        for palabra in palabras
        if normalizar(palabra) not in PALABRAS_NO_SIGNIFICATIVAS | PALABRAS_BUSQUEDA_GENERICAS
    ]


def puntaje_textual(documento: Document, consulta: str) -> int:
    metadata = documento.metadata
    titulo = normalizar(str(metadata.get("titulo", "")))
    descripcion = normalizar(str(metadata.get("descripcion", "")))
    contenido = normalizar(documento.page_content)
    puntaje = 0
    for termino in terminos_textuales(consulta):
        termino_normalizado = normalizar(termino)
        puntaje += 5 if termino_normalizado in titulo else 0
        puntaje += 2 if termino_normalizado in descripcion else 0
        puntaje += 1 if termino_normalizado in contenido else 0
    return puntaje


def buscar_por_texto(vector_store: Chroma, consulta: str, archivo: str | None = None) -> list[Document]:
    terminos = terminos_textuales(consulta)
    if not terminos:
        return []

    # Chroma filtra primero contenido literal con los dos términos más específicos de la pregunta.
    terminos_especificos = sorted(terminos, key=lambda termino: (termino.isdigit(), len(termino)), reverse=True)[:2]
    filtros = [{"$contains": termino} for termino in terminos_especificos]
    where_document = filtros[0] if len(filtros) == 1 else {"$and": filtros}
    # Con archivo, `where` (metadatos) se combina con `where_document` (contenido literal).
    filtro_metadatos = {"where": {"nombre_archivo": archivo}} if archivo else {}
    resultado = vector_store.get(
        where_document=where_document,
        include=["documents", "metadatas"],
        limit=RESULTADOS_POR_ESTRATEGIA,
        **filtro_metadatos,
    )
    documentos = documentos_desde_resultado(resultado)
    return sorted(documentos, key=lambda documento: puntaje_textual(documento, consulta), reverse=True)


def buscar_por_semantica(vector_store: Chroma, consulta: str, archivo: str | None = None) -> list[Document]:
    filtro = {"nombre_archivo": archivo} if archivo else None
    return vector_store.similarity_search(consulta, k=RESULTADOS_POR_ESTRATEGIA, filter=filtro)


def archivo_documento(documento: Document) -> str:
    metadata = documento.metadata
    return str(metadata.get("nombre_archivo") or metadata.get("source") or "sin_archivo")


def clave_documento(documento: Document) -> tuple[str, str, str]:
    """Identifica un fragmento: archivo, página y arranque del contenido."""
    metadata = documento.metadata
    return (
        archivo_documento(documento),
        str(metadata.get("page_label", metadata.get("page", ""))),
        documento.page_content[:160],
    )


def unir_sin_duplicados(*grupos: list[Document], limite: int | None = None) -> list[Document]:
    """Une grupos conservando el orden recibido y descartando fragmentos repetidos."""
    unidos = []
    vistos = set()
    for grupo in grupos:
        for documento in grupo:
            clave = clave_documento(documento)
            if clave in vistos:
                continue
            vistos.add(clave)
            unidos.append(documento)
            if limite is not None and len(unidos) >= limite:
                return unidos
    return unidos


def combinar_sin_duplicados(*grupos: list[Document]) -> list[Document]:
    """Conserva el orden recibido, con tope de resultados finales."""
    return unir_sin_duplicados(*grupos, limite=RESULTADOS_FINALES)


def diversificar(
    documentos: list[Document],
    max_por_documento: int | None = MAX_FRAGMENTOS_POR_DOCUMENTO,
    limite: int = RESULTADOS_FINALES,
) -> list[Document]:
    """Toma los primeros `limite` fragmentos evitando que un PDF ocupe toda la consulta.

    Con `max_por_documento=None` no hay tope por archivo (búsqueda dentro de un documento).
    """
    por_archivo = defaultdict(int)
    seleccionados = []
    for documento in documentos:
        archivo = archivo_documento(documento)
        if max_por_documento is not None and por_archivo[archivo] >= max_por_documento:
            continue
        por_archivo[archivo] += 1
        seleccionados.append(documento)
        if len(seleccionados) >= limite:
            break
    return seleccionados


def combinar_y_diversificar(
    consulta: str,
    resultados_textuales: list[Document],
    resultados_semanticos: list[Document],
    max_por_documento: int | None = MAX_FRAGMENTOS_POR_DOCUMENTO,
) -> list[Document]:
    """Respaldo sin reranker: prioriza coincidencias textuales y diversifica por documento."""
    candidatos = {}
    for posicion, documento in enumerate(resultados_textuales):
        clave = clave_documento(documento)
        candidatos[clave] = (documento, 200 + puntaje_textual(documento, consulta) - posicion)
    for posicion, documento in enumerate(resultados_semanticos):
        clave = clave_documento(documento)
        puntaje = 100 + puntaje_textual(documento, consulta) - posicion
        previo = candidatos.get(clave)
        if previo is None or puntaje > previo[1]:
            candidatos[clave] = (documento, puntaje)

    ordenados = sorted(candidatos.values(), key=lambda candidato: candidato[1], reverse=True)
    return diversificar([documento for documento, _ in ordenados], max_por_documento)


def fuente_de_documento(documento: Document) -> dict:
    metadata = documento.metadata
    pagina = metadata.get("page_label")
    if pagina is None and metadata.get("page") is not None:
        pagina = int(metadata["page"]) + 1
    # Solo se entregan metadatos útiles para auditoría, no propiedades técnicas del PDF.
    return {
        "titulo": metadata.get("titulo") or metadata.get("nombre_archivo") or "Documento SIGEM",
        "archivo": metadata.get("nombre_archivo") or metadata.get("source", ""),
        "pagina": str(pagina) if pagina is not None else None,
        "fecha_expedicion": metadata.get("fecha_expedicion") or None,
        "descripcion": metadata.get("descripcion") or None,
    }


def respuesta_de_recuperacion(documentos: list[Document]) -> str:
    fuentes = []
    fragmentos = []
    fuentes_vistas = set()
    for documento in documentos:
        fuente = fuente_de_documento(documento)
        clave_fuente = (fuente["archivo"], fuente["pagina"])
        if clave_fuente not in fuentes_vistas:
            fuentes_vistas.add(clave_fuente)
            fuentes.append(fuente)
        fragmentos.append({"fuente": fuente, "contenido": documento.page_content})
    # ToolNode conserva este JSON para que el LLM y la API compartan exactamente las mismas fuentes.
    return json.dumps({"fuentes": fuentes, "fragmentos": fragmentos}, ensure_ascii=False)


def recuperar_sigem(consulta: str, archivo: str | None = None) -> list[Document]:
    """Búsqueda híbrida (texto + semántica) → reranker → diversificación → top 8.

    Con `archivo` la búsqueda se limita a ese PDF y no se diversifica.
    Las referencias explícitas ("Decreto 83 de 2018") ya no se resuelven aquí: las resuelve la tool
    buscar_documento_por_referencia con los atributos que extrae el analista.
    """
    vector_store = obtener_vector_store()
    textuales = buscar_por_texto(vector_store, consulta, archivo)
    semanticos = buscar_por_semantica(vector_store, consulta, archivo)
    max_por_documento = None if archivo else MAX_FRAGMENTOS_POR_DOCUMENTO

    candidatos = unir_sin_duplicados(textuales, semanticos)
    reordenados = reranker.reordenar_si_aplica(consulta, candidatos)
    if reordenados is None:
        # Sin reranker (apagado o con error): estrategia léxica + posición semántica.
        return combinar_y_diversificar(consulta, textuales, semanticos, max_por_documento)
    return diversificar(reordenados, max_por_documento)
