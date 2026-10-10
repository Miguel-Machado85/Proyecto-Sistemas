"""Nodo investigador: subagente ReAct que reúne evidencia con las tools de SIGEM."""

import json
import logging

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from .. import tools
from ..config import config as configuracion
from ..estado import PREFIJO_REINTENTO
from ..llm import crear_llm, es_error_de_cuota
from ..prompts import PROMPT_INVESTIGADOR
from ._comun import paso

logger = logging.getLogger(__name__)

LONGITUD_MAXIMA_ARGUMENTOS = 90


def _motivo_reintento(estado: dict) -> str | None:
    """Motivo con que el evaluador pidió reintentar (queda registrado en el último paso del evaluador)."""
    for registro in reversed(estado.get("pasos", [])):
        if registro.get("agente") == "evaluador":
            accion = registro.get("accion", "")
            return accion[len(PREFIJO_REINTENTO):] if accion.startswith(PREFIJO_REINTENTO) else None
    return None


def construir_encargo(estado: dict) -> str:
    """Mensaje inicial del subagente: consulta, filtros, documento activo y motivo del reintento."""
    analisis = estado.get("analisis") or {}
    partes = [f"Consulta: {estado.get('consulta_actual') or analisis.get('consulta_autonoma', '')}"]

    filtros = {k: v for k, v in (analisis.get("filtros") or {}).items() if v is not None}
    if filtros:
        partes.append("Filtros identificados: " + ", ".join(f"{k}={v}" for k, v in filtros.items()))

    documento = estado.get("documento_activo")
    if analisis.get("usa_documento_activo") and documento:
        partes.append(
            "La pregunta se refiere al documento activo: "
            f"{documento.get('titulo', '')} (archivo: {documento.get('archivo', '')}). "
            "Busca dentro de ese archivo."
        )

    motivo = _motivo_reintento(estado)
    if motivo:
        partes.append(
            f"Es un REINTENTO: la evidencia anterior no bastó ({motivo}). "
            "Prueba otra estrategia o una consulta distinta."
        )
    return "\n".join(partes)


def _resumir_argumentos(argumentos: dict) -> str:
    texto = ", ".join(f"{clave}={valor}" for clave, valor in argumentos.items() if valor is not None)
    if len(texto) > LONGITUD_MAXIMA_ARGUMENTOS:
        texto = texto[: LONGITUD_MAXIMA_ARGUMENTOS - 1] + "…"
    return texto


def _clave_evidencia(fragmento: dict) -> tuple:
    fuente = fragmento.get("fuente") or {}
    return (fuente.get("archivo"), fuente.get("pagina"), str(fragmento.get("contenido", ""))[:160])


def acumular_resultados(estado: dict, mensajes: list) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Recorre los ToolMessage del subagente y suma evidencia, fuentes y candidatos sin duplicados.

    Devuelve (evidencia, fuentes, candidatos, pasos_nuevos).
    """
    evidencia = list(estado.get("evidencia", []))
    fuentes = list(estado.get("fuentes", []))
    candidatos = list(estado.get("candidatos", []))
    pasos_nuevos = []

    claves_evidencia = {_clave_evidencia(f) for f in evidencia}
    claves_fuentes = {(f.get("archivo"), f.get("pagina")) for f in fuentes}
    claves_candidatos = {c.get("archivo") for c in candidatos}

    for mensaje in mensajes:
        if isinstance(mensaje, AIMessage):
            for llamada in mensaje.tool_calls or []:
                pasos_nuevos.append(
                    paso("investigador", f"{llamada['name']}: {_resumir_argumentos(llamada.get('args', {}))}", "🔎")
                )
            continue
        if not isinstance(mensaje, ToolMessage):
            continue
        try:
            contenido = json.loads(str(mensaje.content))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(contenido, dict):
            continue

        for fragmento in contenido.get("fragmentos", []):
            clave = _clave_evidencia(fragmento)
            if clave not in claves_evidencia:
                claves_evidencia.add(clave)
                evidencia.append(fragmento)
        # Los listados (listar_documentos) no traen fragmentos: se convierten en evidencia marcada
        # como "listado" para que el evaluador y el redactor puedan responder preguntas de inventario.
        documentos = contenido.get("documentos", [])
        for documento in documentos:
            fecha = f" ({documento['fecha_expedicion']})" if documento.get("fecha_expedicion") else ""
            descripcion = f": {documento['descripcion']}" if documento.get("descripcion") else ""
            fragmento = {
                "fuente": {**documento, "pagina": None},
                "contenido": f"{documento.get('titulo', 'Documento')}{fecha}{descripcion}",
                "tipo": "listado",
            }
            clave = _clave_evidencia(fragmento)
            if clave not in claves_evidencia:
                claves_evidencia.add(clave)
                evidencia.append(fragmento)
        total = contenido.get("total")
        if documentos and isinstance(total, int) and total > len(documentos):
            fragmento = {
                "fuente": {"titulo": "Listado SIGEM", "archivo": "", "pagina": None},
                "contenido": f"Se muestran {len(documentos)} de {total} documentos que coinciden con el filtro.",
                "tipo": "listado",
            }
            clave = _clave_evidencia(fragmento)
            if clave not in claves_evidencia:
                claves_evidencia.add(clave)
                evidencia.append(fragmento)

        for fuente in contenido.get("fuentes", []):
            clave = (fuente.get("archivo"), fuente.get("pagina"))
            if clave not in claves_fuentes:
                claves_fuentes.add(clave)
                fuentes.append(fuente)
        for candidato in contenido.get("candidatos", []):
            if candidato.get("archivo") not in claves_candidatos:
                claves_candidatos.add(candidato.get("archivo"))
                candidatos.append(candidato)

    return evidencia, fuentes, candidatos, pasos_nuevos


def _ejecutar_subagente(encargo: str, config: RunnableConfig | None) -> list:
    """Corre el subagente con historial nuevo y devuelve sus mensajes (incluso si se agota el límite)."""
    subagente = create_agent(
        model=crear_llm(configuracion.INVESTIGADOR_MODEL, 0.1),
        tools=tools.HERRAMIENTAS_INVESTIGADOR,
        system_prompt=PROMPT_INVESTIGADOR,
        # Sin checkpointer propio ni heredado: cada intento empieza con historial vacío.
        checkpointer=False,
    )
    configuracion_subagente = {**(config or {}), "recursion_limit": configuracion.INVESTIGADOR_MAX_PASOS}

    mensajes: list = []
    try:
        for valores in subagente.stream(
            {"messages": [HumanMessage(content=encargo)]}, configuracion_subagente, stream_mode="values"
        ):
            mensajes = valores.get("messages", mensajes)
    except GraphRecursionError:
        logger.warning("El investigador alcanzó el límite de pasos; se usa la evidencia recolectada.")
    except Exception as error:
        if es_error_de_cuota(error):
            # Sin cuota no hay búsqueda posible: responder "sin resultados" sería engañoso.
            raise
        logger.warning("El investigador falló; se usa la evidencia recolectada: %s", error)
    return mensajes


def investigador(estado: dict, config: RunnableConfig | None = None) -> dict:
    mensajes = _ejecutar_subagente(construir_encargo(estado), config)
    evidencia, fuentes, candidatos, pasos_nuevos = acumular_resultados(estado, mensajes)
    return {
        "evidencia": evidencia,
        "fuentes": fuentes,
        "candidatos": candidatos,
        # Cuenta ejecuciones del investigador: MAX_INTENTOS_BUSQUEDA es el máximo de pasadas.
        "intentos_busqueda": estado.get("intentos_busqueda", 0) + 1,
        "pasos": estado.get("pasos", []) + pasos_nuevos,
    }
