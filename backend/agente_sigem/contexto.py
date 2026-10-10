"""Resolución de preguntas de seguimiento sobre documentos SIGEM."""

import re
import unicodedata


PALABRAS_SOCIALES = {"hola", "gracias", "adios", "chao", "hasta", "buenas"}
MARCADORES_SEGUIMIENTO = {
    "anterior", "amplia", "ampliame", "cuentame", "cuentame", "eso", "esa", "ese", "explicame",
    "hablame", "mas", "mismo", "periodo", "profundiza", "tema", "vigencia",
}
INICIOS_PREGUNTA = {"como", "cuando", "cual", "cuales", "donde", "por", "porque", "que"}
MARCADORES_AMPLIACION = {"amplia", "ampliame", "cuentame", "explicame", "hablame", "mas", "profundiza"}
PATRONES_BUSQUEDA_GLOBAL = (
    re.compile(
        r"\b(?:que|cuales)\s+(?:normas|documentos|decretos|resoluciones|acuerdos|formatos|tramites)\b"
    ),
    re.compile(
        r"\b(?:normas|documentos|decretos|resoluciones|acuerdos|formatos|tramites)\s+(?:de\s+)?sigem\b"
    ),
    re.compile(r"\b(?:busca|buscar|muestra|muestrame|lista|listame|encuentra)\b"),
)
PATRON_REFERENCIA_DOCUMENTO = re.compile(
    r"\b(?:decreto|resolucion|acuerdo|ley|formato)\s*(?:n(?:o|ro)?\.?\s*)?\d{1,5}\b"
    r"|\b(?:documento|archivo|codigo)\s*(?:no\.?\s*)?\d{3,8}\b"
)


def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(caracter for caracter in texto if unicodedata.category(caracter) != "Mn")
    return re.sub(r"\s+", " ", texto).strip()


def fuentes_por_documento(fuentes: list[dict]) -> list[dict]:
    """Conserva una fuente por PDF para no repetir páginas en el contexto."""
    documentos = []
    archivos_vistos = set()
    for fuente in fuentes:
        archivo = fuente.get("archivo")
        if archivo and archivo not in archivos_vistos:
            archivos_vistos.add(archivo)
            documentos.append(fuente)
    return documentos


def es_busqueda_global_o_nueva_referencia(texto_normalizado: str) -> bool:
    """Evita que una nueva búsqueda herede por error el documento anterior."""
    return any(patron.search(texto_normalizado) for patron in PATRONES_BUSQUEDA_GLOBAL) or bool(
        PATRON_REFERENCIA_DOCUMENTO.search(texto_normalizado)
    )


def es_pregunta_de_seguimiento(mensaje: str, fuentes_activas: list[dict]) -> bool:
    """Identifica una referencia corta al documento recuperado en el turno previo."""
    if not fuentes_activas:
        return False

    texto_normalizado = normalizar_texto(mensaje)
    if es_busqueda_global_o_nueva_referencia(texto_normalizado):
        return False

    palabras = re.findall(r"[a-z0-9]+", texto_normalizado)
    if not palabras or set(palabras).issubset(PALABRAS_SOCIALES):
        return False
    if set(palabras) & MARCADORES_SEGUIMIENTO:
        return True
    return len(palabras) <= 8 and palabras[0] in INICIOS_PREGUNTA


def enriquecer_consulta_con_contexto(mensaje: str, fuentes_activas: list[dict]) -> str:
    """Ancla una continuación al documento que respaldó la respuesta anterior."""
    documentos = fuentes_por_documento(fuentes_activas)
    documento_principal = documentos[0]
    titulo = documento_principal.get("titulo") or documento_principal.get("archivo")
    descripcion = documento_principal.get("descripcion") or ""
    palabras = set(re.findall(r"[a-z0-9]+", normalizar_texto(mensaje)))
    if palabras & MARCADORES_AMPLIACION:
        objetivo = (
            "El usuario solicita ampliar la explicación. Después de recuperar los fragmentos, "
            "resume o desarrolla información concreta del documento. No afirmes que falta contenido "
            "si la herramienta devolvió fragmentos."
        )
    else:
        objetivo = "Responde la pregunta concreta usando exclusivamente los fragmentos recuperados."
    return (
        "La pregunta del usuario es una continuación de la conversación. "
        f"Documento SIGEM activo: {titulo}. "
        f"Descripción: {descripcion}. "
        "Debes usar buscar_en_sigem con una consulta que incluya el documento activo. "
        f"{objetivo} "
        f"Pregunta del usuario: {mensaje}"
    )
