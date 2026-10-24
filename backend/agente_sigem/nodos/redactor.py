"""Nodo redactor: arma la respuesta final al ciudadano según el perfil y la evidencia."""

import logging
import re
import unicodedata

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import config as configuracion
from ..estado import RespuestaRedactor
from ..llm import crear_llm, es_error_de_cuota
from ..prompts import construir_prompt_redactor
from ._comun import paso, respuesta_ai, texto_de, ultimo_mensaje_usuario

logger = logging.getLogger(__name__)

MAX_FRAGMENTOS_REDACCION = 12
SUFIJO_SIN_INFORMACION = " · sin información pertinente en los fragmentos"

# Formato de cita que el prompt le exige al redactor: [Decreto 001 de 2019, p. 3]. Se analiza el texto
# que genera nuestro propio modelo (formato controlado), no lo que escribe el usuario.
PATRON_CITA = re.compile(r"\[([^\[\]]{3,250})\]")
PATRON_TITULO_Y_PAGINAS = re.compile(r"^(.*?)(?:,\s*p[^\d,]*(\d.*))?$", re.DOTALL)


def elegir_modo(estado: dict) -> str:
    """respuesta | aclaracion (candidatos) | entrega (archivo sin evidencia) | descarga_no_disponible | sin_resultados."""
    if estado.get("evidencia"):
        return "respuesta"
    if estado.get("candidatos"):
        # Si hay varios documentos posibles no se entrega ninguno: primero se pide aclaración.
        return "aclaracion"
    if estado.get("archivos"):
        return "entrega"
    if estado.get("archivos_no_disponibles"):
        return "descarga_no_disponible"
    return "sin_resultados"


def _evidencia_numerada(evidencia: list[dict]) -> str:
    bloques = []
    for numero, fragmento in enumerate(evidencia[:MAX_FRAGMENTOS_REDACCION], start=1):
        fuente = fragmento.get("fuente") or {}
        pagina = f", p. {fuente['pagina']}" if fuente.get("pagina") else ""
        bloques.append(f"[{numero}] {fuente.get('titulo', 'Documento SIGEM')}{pagina}\n{fragmento.get('contenido', '')}")
    return "\n\n".join(bloques)


def _titulos(documentos: list[dict]) -> str:
    return "; ".join(d.get("titulo") or d.get("archivo") or "Documento" for d in documentos)


def _candidatos_en_texto(candidatos: list[dict]) -> str:
    return "\n".join(
        f"- {c.get('titulo', 'Documento')} ({c.get('fecha_expedicion') or 'sin fecha'})"
        + (f": {c['descripcion']}" if c.get("descripcion") else "")
        for c in candidatos
    )


def construir_entrada(estado: dict, modo: str) -> str:
    analisis = estado.get("analisis") or {}
    partes = [
        f"PREGUNTA ORIGINAL DEL USUARIO: {ultimo_mensaje_usuario(estado)}",
        f"PREGUNTA ENTENDIDA: {analisis.get('consulta_autonoma') or estado.get('consulta_actual', '')}",
    ]
    documento = estado.get("documento_activo")
    if analisis.get("usa_documento_activo") and documento:
        partes.append(f"DOCUMENTO ACTIVO DE LA CONVERSACIÓN: {documento.get('titulo', '')}")

    if estado.get("archivos"):
        partes.append("ARCHIVO(S) PARA DESCARGAR (la interfaz los muestra como botón): " + _titulos(estado["archivos"]))
    if estado.get("archivos_no_disponibles"):
        partes.append("ORIGINAL NO DISPONIBLE PARA DESCARGA: " + _titulos(estado["archivos_no_disponibles"]))

    if modo == "respuesta":
        partes.append("EVIDENCIA:\n" + _evidencia_numerada(estado.get("evidencia", [])))
    elif modo == "aclaracion":
        partes.append("CANDIDATOS:\n" + _candidatos_en_texto(estado.get("candidatos", [])))
    else:
        partes.append("EVIDENCIA: ninguna. No se encontró información pertinente.")
    return "\n\n".join(partes)


def _normalizar_titulo(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texto).strip(" .")


def _citas(texto: str) -> list[tuple[str, set[str]]]:
    """Lista de (título normalizado, páginas citadas) encontradas en la respuesta."""
    citas = []
    for contenido in PATRON_CITA.findall(texto):
        for parte in contenido.split(";"):
            coincidencia = PATRON_TITULO_Y_PAGINAS.match(parte.strip())
            if not coincidencia or not coincidencia.group(1).strip():
                continue
            paginas = set(re.findall(r"\d+", coincidencia.group(2) or ""))
            citas.append((_normalizar_titulo(coincidencia.group(1)), paginas))
    return citas


def _coincide_titulo(titulo_fuente: str, titulo_cita: str) -> bool:
    fuente = _normalizar_titulo(titulo_fuente)
    return bool(fuente) and (fuente == titulo_cita or fuente in titulo_cita or titulo_cita in fuente)


def fuentes_citadas(texto: str, fuentes: list[dict]) -> list[dict]:
    """Deja solo las fuentes que la respuesta cita; si no se puede determinar, conserva todas.

    Una cita sin página conserva todas las páginas de ese documento; una cita con página que no
    coincide con ninguna fuente conserva el documento (mejor mostrar de más que perder la fuente).
    """
    citas = _citas(texto)
    if not citas or not fuentes:
        return fuentes
    seleccionadas = []
    claves = set()
    for titulo_cita, paginas in citas:
        del_documento = [f for f in fuentes if _coincide_titulo(f.get("titulo", ""), titulo_cita)]
        exactas = [f for f in del_documento if not paginas or str(f.get("pagina")) in paginas]
        for fuente in exactas or del_documento:
            clave = (fuente.get("archivo"), fuente.get("pagina"))
            if clave not in claves:
                claves.add(clave)
                seleccionadas.append(fuente)
    return seleccionadas or fuentes


def _documento_activo_desde(estado: dict, citadas: list[dict] | None = None) -> dict | None:
    """Documento de la primera fuente citada que tiene fragmentos; None si solo hubo listados o no hubo fuentes."""
    archivos_citados = {f.get("archivo") for f in citadas or []}
    fragmentos = [f for f in estado.get("evidencia", []) if f.get("tipo") != "listado"]
    # Primero los fragmentos de documentos que la respuesta cita; si no, el más relevante.
    preferidos = [f for f in fragmentos if (f.get("fuente") or {}).get("archivo") in archivos_citados]
    for fragmento in preferidos + fragmentos:
        fuente = fragmento.get("fuente") or {}
        if fuente.get("archivo"):
            return {
                "titulo": fuente.get("titulo"),
                "archivo": fuente.get("archivo"),
                "descripcion": fuente.get("descripcion"),
                "fecha_expedicion": fuente.get("fecha_expedicion"),
            }
    return None


def _documento_de_archivo(archivo: dict) -> dict:
    return {
        "titulo": archivo.get("titulo"),
        "archivo": archivo.get("archivo"),
        "descripcion": archivo.get("descripcion"),
        "fecha_expedicion": archivo.get("fecha_expedicion"),
    }


def _redactar(llm, mensajes: list, config: RunnableConfig | None, estructurado: bool) -> tuple[str, bool]:
    """Devuelve (texto, informacion_encontrada). En modo respuesta pide la señal explícita al modelo."""
    if estructurado:
        try:
            salida = llm.with_structured_output(RespuestaRedactor).invoke(mensajes, config)
            if isinstance(salida, dict):
                salida = RespuestaRedactor(**salida)
            if isinstance(salida, RespuestaRedactor) and salida.respuesta.strip():
                return salida.respuesta, salida.informacion_encontrada
        except Exception as error:
            if es_error_de_cuota(error):
                raise
            logger.warning("Salida estructurada del redactor no disponible; se usa texto libre: %s", error)
    return texto_de(llm.invoke(mensajes, config)), True


def redactor(estado: dict, config: RunnableConfig | None = None) -> dict:
    perfil = estado.get("perfil", "P0")
    modo = elegir_modo(estado)

    llm = crear_llm(configuracion.REDACTOR_MODEL, 0.3)
    mensajes = [
        SystemMessage(content=construir_prompt_redactor(perfil, modo)),
        HumanMessage(content=construir_entrada(estado, modo)),
    ]
    texto, informacion_encontrada = _redactar(llm, mensajes, config, estructurado=(modo == "respuesta"))

    fuentes_estado = estado.get("fuentes", [])
    if modo == "respuesta" and not informacion_encontrada:
        # La respuesta dice que no hay información: no se muestran fuentes ni cambia el documento activo.
        fuentes = []
        documento_activo = estado.get("documento_activo")
    elif modo in ("entrega", "descarga_no_disponible"):
        # Sin fragmentos que citar: el documento activo pasa a ser el que se entregó (o el que se pidió).
        fuentes = []
        documento_activo = _documento_de_archivo((estado.get("archivos") or estado.get("archivos_no_disponibles"))[0])
    else:
        # Las fuentes que se muestran al usuario son las que la respuesta cita, no todo lo buscado.
        fuentes = fuentes_citadas(texto, fuentes_estado) if modo == "respuesta" else fuentes_estado
        # Un listado (solo fuentes, sin fragmentos) no convierte ningún documento en "activo".
        documento_activo = _documento_activo_desde(estado, fuentes) or estado.get("documento_activo")

    accion = f"Redactando respuesta para perfil {perfil}"
    if modo == "respuesta" and not informacion_encontrada:
        accion += SUFIJO_SIN_INFORMACION
    return {
        "messages": [respuesta_ai(texto)],
        "fuentes": fuentes,
        "documento_activo": documento_activo,
        "pasos": estado.get("pasos", []) + [paso("redactor", accion, "✍️")],
    }
