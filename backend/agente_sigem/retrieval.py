import json
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import config
from .catalogo import CatalogoDocumentos


RESULTADOS_SEMANTICOS = 8
RESULTADOS_CANDIDATOS = 24
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


def obtener_vector_store() -> Chroma:
    """Abre Chroma con el modelo de embeddings que generó sus vectores existentes."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def obtener_retriever():
    vector_store = obtener_vector_store()
    # Se recuperan más candidatos que los ocho finales para poder reordenarlos y diversificarlos.
    return vector_store.as_retriever(search_kwargs={"k": RESULTADOS_CANDIDATOS})


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


def buscar_por_referencia(vector_store: Chroma, consulta: str) -> list[Document]:
    resolucion = obtener_catalogo_documentos().resolver(consulta)
    if resolucion.estado != "exacta":
        return []

    # El catálogo resuelve aliases y metadatos; Chroma recupera todos los fragmentos del PDF elegido.
    documentos = []
    for documento in resolucion.documentos:
        resultado = vector_store.get(
            where={"nombre_archivo": documento.archivo},
            include=["documents", "metadatas"],
            limit=RESULTADOS_POR_ESTRATEGIA,
        )
        documentos.extend(documentos_desde_resultado(resultado))
    return documentos


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


def buscar_por_texto(vector_store: Chroma, consulta: str) -> list[Document]:
    terminos = terminos_textuales(consulta)
    if not terminos:
        return []

    # Chroma filtra primero contenido literal con los dos términos más específicos de la pregunta.
    terminos_especificos = sorted(terminos, key=lambda termino: (termino.isdigit(), len(termino)), reverse=True)[:2]
    filtros = [{"$contains": termino} for termino in terminos_especificos]
    where_document = filtros[0] if len(filtros) == 1 else {"$and": filtros}
    resultado = vector_store.get(
        where_document=where_document,
        include=["documents", "metadatas"],
        limit=RESULTADOS_POR_ESTRATEGIA,
    )
    documentos = documentos_desde_resultado(resultado)
    return sorted(documentos, key=lambda documento: puntaje_textual(documento, consulta), reverse=True)


def clave_documento(documento: Document) -> tuple[str, str, str]:
    metadata = documento.metadata
    return (
        str(metadata.get("source", "")),
        str(metadata.get("page_label", metadata.get("page", ""))),
        documento.page_content[:160],
    )


def combinar_sin_duplicados(*grupos: list[Document]) -> list[Document]:
    """Conserva el orden recibido; se usa para un único documento de referencia exacta."""
    combinados = []
    vistos = set()
    for grupo in grupos:
        for documento in grupo:
            clave = clave_documento(documento)
            if clave not in vistos:
                vistos.add(clave)
                combinados.append(documento)
            if len(combinados) >= RESULTADOS_SEMANTICOS:
                return combinados
    return combinados


def archivo_documento(documento: Document) -> str:
    metadata = documento.metadata
    return str(metadata.get("nombre_archivo") or metadata.get("source") or "sin_archivo")


def combinar_y_diversificar(
    consulta: str, resultados_textuales: list[Document], resultados_semanticos: list[Document]
) -> list[Document]:
    """Prioriza coincidencias textuales y evita que un PDF ocupe toda una consulta amplia."""
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
    por_archivo = defaultdict(int)
    seleccionados = []
    for documento, _ in ordenados:
        archivo = archivo_documento(documento)
        if por_archivo[archivo] >= MAX_FRAGMENTOS_POR_DOCUMENTO:
            continue
        por_archivo[archivo] += 1
        seleccionados.append(documento)
        if len(seleccionados) >= RESULTADOS_SEMANTICOS:
            break
    return seleccionados


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


def recuperar_sigem(consulta: str) -> list[Document]:
    vector_store = obtener_vector_store()
    resultados_referencia = buscar_por_referencia(vector_store, consulta)
    if resultados_referencia:
        # Una referencia explícita identifica el documento pedido; no se mezclan vecinos semánticos ajenos.
        return combinar_sin_duplicados(resultados_referencia)

    resultados_textuales = buscar_por_texto(vector_store, consulta)
    # La similitud semántica es el respaldo cuando una referencia o texto literal no bastan.
    resultados_semanticos = obtener_retriever().invoke(consulta)
    return combinar_y_diversificar(consulta, resultados_textuales, resultados_semanticos)
