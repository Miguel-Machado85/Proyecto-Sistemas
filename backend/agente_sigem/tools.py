"""Herramientas del agente investigador.

Todas devuelven un JSON (string) con el mismo formato que consume la API:
{"fuentes": [...], "fragmentos": [...], "candidatos"?, "aclaracion"?, "sugerencias"?, "documentos"?, "error"?}
y ninguna lanza excepciones: ante un fallo devuelven {"fuentes": [], "fragmentos": [], "error": "..."}.

Los docstrings son lo que lee el LLM para decidir qué herramienta usar: tratarlos como parte del prompt.
Aquí no se interpreta texto libre del usuario: el LLM llena los argumentos y el código los valida
contra el catálogo.
"""

import json
import logging
from typing import Literal

from langchain_core.documents import Document
from langchain_core.tools import tool

from urllib.parse import quote

from . import db, reranker, retrieval
from .catalogo import DocumentoCatalogado, normalizar

logger = logging.getLogger(__name__)

TipoDocumento = Literal["decreto", "resolucion", "acuerdo", "ley", "formato"]
TIPOS_VALIDOS = ("decreto", "resolucion", "acuerdo", "ley", "formato")
MAX_FRAGMENTOS = 8
MAX_FRAGMENTOS_LECTURA = 12
MAX_CANDIDATOS = 5
MAX_SUGERENCIAS = 3
MAX_LISTADO = 50
LIMITE_LECTURA_CHROMA = 200


# ══════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════

def _json(**campos) -> str:
    return json.dumps(campos, ensure_ascii=False)


def _error(mensaje: str) -> str:
    return _json(fuentes=[], fragmentos=[], error=mensaje)


def _error_de_argumentos(error: Exception) -> str:
    """Se usa como handle_validation_error: un argumento mal formado no debe lanzar excepciones."""
    return _error(f"Argumentos inválidos: {str(error)[:300]}")


def _datos_documento(documento: DocumentoCatalogado) -> dict:
    return {
        "titulo": documento.titulo,
        "archivo": documento.archivo,
        "fecha_expedicion": documento.fecha_expedicion or (str(documento.anio) if documento.anio else None),
        "descripcion": documento.descripcion,
    }


def _con_respuesta(documentos: list[Document], **extras) -> str:
    """Formato base de retrieval.respuesta_de_recuperacion más campos extra opcionales."""
    contenido = json.loads(retrieval.respuesta_de_recuperacion(documentos))
    contenido.update(extras)
    return json.dumps(contenido, ensure_ascii=False)


def _numero_de_pagina(documento: Document) -> int | None:
    pagina = retrieval.fuente_de_documento(documento)["pagina"]
    return int(pagina) if pagina is not None and str(pagina).isdigit() else None


def _fragmentos_de_archivo(archivo: str) -> list[Document]:
    resultado = retrieval.obtener_vector_store().get(
        where={"nombre_archivo": archivo},
        include=["documents", "metadatas"],
        limit=LIMITE_LECTURA_CHROMA,
    )
    return retrieval.documentos_desde_resultado(resultado)


def _ordenar_por_pagina(documentos: list[Document]) -> list[Document]:
    return sorted(documentos, key=lambda d: (_numero_de_pagina(d) is None, _numero_de_pagina(d) or 0))


def _tipo_canonico(tipo: str | None) -> str | None:
    """Normaliza el argumento `tipo` que llena el LLM ('Decretos' → 'decreto'); None si no es válido."""
    if not tipo:
        return None
    texto = normalizar(tipo).strip()
    for candidato in (texto, texto[:-1], texto[:-2]):
        if candidato in TIPOS_VALIDOS:
            return candidato
    return None


# ══════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════

@tool
def buscar_documento_por_referencia(
    tipo: TipoDocumento | None = None,
    numero: int | None = None,
    anio: int | None = None,
    pregunta: str | None = None,
    codigo: str | None = None,
) -> str:
    """Trae el contenido de UN documento concreto cuando el usuario lo identifica, ya sea por
    tipo y número ("Decreto 83 de 2018", "la Resolución 4496", "la Ley 387 de 1997") o por el
    código del archivo ("el documento 18399", "el archivo 22598").

    Identificación (usa una de las dos):
    - `tipo` (decreto, resolucion, acuerdo, ley o formato) + `numero`, y `anio` si el usuario lo dijo.
    - `codigo`: el número que antecede al nombre del archivo (p. ej. "18399" para
      "18399_resolucion-3615.pdf"). Un código identifica un único documento. Si pasas `codigo`
      junto con tipo y número, se usa primero el código.

    Pasa en `pregunta` lo que el usuario quiere saber del documento para que se devuelvan los
    fragmentos más pertinentes.

    Resultados posibles:
    - Un documento: devuelve sus fragmentos más relevantes con `fuentes` y `fragmentos`.
    - Varios documentos con ese número (por ejemplo de distintos años): devuelve `candidatos`
      y `aclaracion`; no inventes cuál es, pide al usuario que elija.
    - Ninguno: devuelve `sugerencias` con los títulos más parecidos (o una `aclaracion` si el
      código no existe); corrige y vuelve a intentar, o usa `buscar_fragmentos` si la referencia
      puede estar mal.

    NO la uses para preguntas sobre un tema sin documento nombrado (usa `buscar_fragmentos`),
    ni para listar documentos (usa `listar_documentos`).
    """
    try:
        catalogo = retrieval.obtener_catalogo_documentos()
        codigo_limpio = (str(codigo).strip() if codigo is not None else "") or None
        if codigo_limpio is None and (tipo is None or numero is None):
            return _json(
                fuentes=[],
                fragmentos=[],
                error="Indica `tipo` y `numero` del documento, o su `codigo` de archivo.",
            )

        resolucion = None
        if codigo_limpio is not None:
            resolucion = catalogo.buscar_por_codigo(codigo_limpio)
            etiqueta = f"el documento con código {codigo_limpio}"
        if (resolucion is None or resolucion.estado == "sin_coincidencia") and tipo and numero is not None:
            resolucion = catalogo.buscar_por_atributos(tipo, numero, anio)
            etiqueta = f"{tipo} {numero}" + (f" de {anio}" if anio else "")

        if resolucion.estado == "exacta":
            documentos = _fragmentos_de_archivo(resolucion.documentos[0].archivo)
            if pregunta:
                documentos = reranker.reordenar(pregunta, documentos)
            else:
                documentos = _ordenar_por_pagina(documentos)
            return _con_respuesta(retrieval.unir_sin_duplicados(documentos, limite=MAX_FRAGMENTOS))

        if resolucion.estado == "ambigua":
            candidatos = [_datos_documento(d) for d in resolucion.documentos[:MAX_CANDIDATOS]]
            opciones = "; ".join(f"{c['titulo']} ({c['fecha_expedicion'] or 'sin fecha'})" for c in candidatos)
            return _json(
                fuentes=[],
                fragmentos=[],
                candidatos=candidatos,
                aclaracion=(
                    f"Hay varios documentos posibles para {etiqueta}: {opciones}. "
                    "Pide al usuario que indique el año o el tema."
                ),
            )

        if codigo_limpio is not None and not (tipo and numero is not None):
            return _json(
                fuentes=[],
                fragmentos=[],
                sugerencias=[],
                aclaracion=(
                    f"No existe un documento con el código {codigo_limpio} en SIGEM. "
                    "Pide al usuario que verifique el código o que indique el tipo y número."
                ),
            )

        similares = catalogo.similares(etiqueta, limite=MAX_SUGERENCIAS)
        return _json(
            fuentes=[],
            fragmentos=[],
            sugerencias=[_datos_documento(d) for d in similares],
        )
    except Exception as error:
        logger.exception("buscar_documento_por_referencia falló")
        return _error(f"No fue posible buscar el documento: {error}")


@tool
def listar_documentos(
    tipo: str | None = None,
    anio: int | None = None,
    tema: str | None = None,
    limite: int = 15,
) -> str:
    """Lista qué documentos existen en el repositorio SIGEM; NO trae su contenido.

    Úsala para preguntas de inventario: "¿qué decretos hay de 2019?", "¿qué leyes tienen?",
    "¿qué normas hay sobre presupuesto?". Filtros opcionales y combinables:
    - `tipo`: decreto, resolucion, acuerdo, ley o formato.
    - `anio`: año de expedición (número, p. ej. 2019).
    - `tema`: palabra o frase del asunto (p. ej. "presupuesto", "horario de atención").
    - `limite`: máximo de documentos a devolver (por defecto 15).

    Devuelve `documentos` (título, archivo, fecha y descripción) y también `total` con cuántos
    coinciden en el repositorio. Si necesitas lo que DICE un documento, después usa
    `buscar_documento_por_referencia`, `buscar_fragmentos` o `leer_documento`.
    """
    try:
        tipo_normalizado = _tipo_canonico(tipo)
        if tipo and tipo_normalizado is None:
            return _json(
                fuentes=[],
                fragmentos=[],
                documentos=[],
                total=0,
                aclaracion=f"Tipo no reconocido: {tipo}. Usa uno de: {', '.join(TIPOS_VALIDOS)}.",
            )
        limite_efectivo = max(1, min(int(limite), MAX_LISTADO))
        catalogo = retrieval.obtener_catalogo_documentos()
        documentos, total = catalogo.listar(tipo_normalizado, anio, tema, limite_efectivo)
        datos = [_datos_documento(d) for d in documentos]
        # También van en `fuentes` para que la API los muestre, aunque no haya fragmentos.
        fuentes = [{**d, "pagina": None} for d in datos]
        return _json(fuentes=fuentes, fragmentos=[], documentos=datos, total=total)
    except Exception as error:
        logger.exception("listar_documentos falló")
        return _error(f"No fue posible listar los documentos: {error}")


@tool
def buscar_fragmentos(consulta: str, archivo: str | None = None) -> str:
    """Busca en el contenido de los documentos los fragmentos que responden una pregunta o tema.

    Úsala cuando el usuario pregunta por un tema, trámite, tarifa, horario, requisito o regla SIN
    nombrar un documento concreto ("¿cuánto cuesta el taxi?", "¿hasta qué hora se vende licor?").
    Escribe en `consulta` una frase completa y autónoma, con la ortografía corregida.

    Si ya sabes cuál es el documento (por el documento activo o porque lo encontraste antes),
    pasa su nombre en `archivo` (el campo `archivo` de las fuentes, p. ej.
    "22598_decreto-no-0012019.pdf") para buscar solo dentro de ese documento.

    NO la uses para listar qué documentos existen (usa `listar_documentos`) ni cuando el usuario
    nombra tipo y número (usa `buscar_documento_por_referencia`).
    """
    try:
        documentos = retrieval.recuperar_sigem(consulta, archivo or None)
        return _con_respuesta(documentos)
    except Exception as error:
        logger.exception("buscar_fragmentos falló")
        return _error(f"No fue posible buscar fragmentos: {error}")


@tool
def leer_documento(archivo: str, paginas: list[int] | None = None) -> str:
    """Lee fragmentos de un documento ya identificado, en orden de página (hasta 12 fragmentos).

    Úsala cuando el usuario pide resumir, explicar o ampliar un documento que ya conoces
    ("resúmelo", "explícame más", "¿qué más dice?"). `archivo` es el nombre del PDF tal como
    aparece en las fuentes (p. ej. "10322_decreto-no-0832018.pdf"). `paginas` es opcional: lista de
    números de página (1 es la primera) para leer solo esas.

    NO la uses para buscar un tema específico dentro del documento (usa `buscar_fragmentos` con
    `archivo`) ni si todavía no sabes qué documento es (usa `buscar_documento_por_referencia`).
    """
    try:
        documentos = _fragmentos_de_archivo(archivo)
        if paginas:
            solicitadas = set(paginas)
            documentos = [d for d in documentos if _numero_de_pagina(d) in solicitadas]
        if not documentos:
            similares = retrieval.obtener_catalogo_documentos().similares(archivo, limite=MAX_SUGERENCIAS)
            return _json(
                fuentes=[],
                fragmentos=[],
                sugerencias=[_datos_documento(d) for d in similares],
                error=f"No hay fragmentos para el archivo {archivo}" + (f" en las páginas {paginas}" if paginas else ""),
            )
        documentos = retrieval.unir_sin_duplicados(_ordenar_por_pagina(documentos), limite=MAX_FRAGMENTOS_LECTURA)
        return _con_respuesta(documentos)
    except Exception as error:
        logger.exception("leer_documento falló")
        return _error(f"No fue posible leer el documento: {error}")


@tool
def preparar_descarga(archivo: str) -> str:
    """Prepara la descarga del archivo ORIGINAL (PDF) de un documento ya identificado.

    Úsala solo cuando el usuario pidió el archivo o el documento. `archivo` es el nombre del PDF
    tal como aparece en las fuentes o candidatos (p. ej. "22598_decreto-no-0012019.pdf"). Si todavía
    no sabes cuál es el documento, identifícalo antes con `buscar_documento_por_referencia` o
    `buscar_fragmentos`; si hay varios candidatos, NO elijas por el usuario. Llámala una vez por documento.

    Devuelve `archivos` con el título del documento listo para descargar, o un `error` si ese documento
    no está disponible para descarga. NO escribas ni inventes enlaces: la interfaz muestra el botón.
    """
    try:
        catalogo = retrieval.obtener_catalogo_documentos()
        documento = next((d for d in catalogo.documentos if d.archivo == (archivo or "").strip()), None)
        if documento is None:
            similares = catalogo.similares(archivo or "", limite=MAX_SUGERENCIAS)
            return _json(
                fuentes=[],
                fragmentos=[],
                sugerencias=[_datos_documento(d) for d in similares],
                error=f"No existe el archivo {archivo} en el repositorio SIGEM.",
            )
        if db.obtener_archivo(documento.archivo) is None:
            return _json(
                fuentes=[],
                fragmentos=[],
                motivo="no_disponible",
                titulo=documento.titulo,
                archivo=documento.archivo,
                error=f"El documento {documento.titulo} está en SIGEM pero su original no está disponible para descarga.",
            )
        datos = _datos_documento(documento)
        return _json(
            fuentes=[],
            fragmentos=[],
            archivos=[{**datos, "url_descarga": f"/documentos/{quote(documento.archivo)}/descargar"}],
        )
    except Exception as error:
        logger.exception("preparar_descarga falló")
        return _error(f"No fue posible preparar la descarga: {error}")


HERRAMIENTAS_INVESTIGADOR = [
    buscar_documento_por_referencia,
    listar_documentos,
    buscar_fragmentos,
    leer_documento,
]

# Un argumento inválido (p. ej. un tipo fuera del Literal) también debe volver como JSON, no como excepción.
for _herramienta in HERRAMIENTAS_INVESTIGADOR:
    _herramienta.handle_validation_error = _error_de_argumentos

# Va aparte: el investigador solo la recibe cuando el analista marcó `solicita_archivo`.
HERRAMIENTAS_DESCARGA = [preparar_descarga]
preparar_descarga.handle_validation_error = _error_de_argumentos


def herramientas_investigador(solicita_archivo: bool = False) -> list:
    """Tools del subagente: las de búsqueda y, solo si el usuario pidió el archivo, `preparar_descarga`."""
    return list(HERRAMIENTAS_INVESTIGADOR) + (list(HERRAMIENTAS_DESCARGA) if solicita_archivo else [])
