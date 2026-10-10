"""Estado compartido del grafo multi-agente y esquemas de salida estructurada."""

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict


class EstadoAgente(TypedDict):
    # ── Persisten entre turnos (checkpointer) ──
    messages: Annotated[list[BaseMessage], add_messages]  # SOLO Human + AI final por turno
    documento_activo: dict | None  # {"titulo","archivo","descripcion","fecha_expedicion"}

    # ── Entrada del turno ──
    perfil: str  # "P0".."P3"
    origen: str  # "texto" | "voz"

    # ── Se REINICIAN en cada invoke ──
    # Sin reducer a propósito: api.py los pasa vacíos y el valor nuevo reemplaza al anterior.
    # Con operator.add se acumularían entre turnos por culpa del checkpointer.
    analisis: dict | None  # AnalisisConsulta.model_dump()
    evidencia: list[dict]  # fragmentos {"fuente": {...}, "contenido": str}
    fuentes: list[dict]  # mismo formato de FuenteSigem
    candidatos: list[dict]  # mismo formato de CandidatoDocumento
    intentos_busqueda: int  # ejecuciones del investigador en este turno
    consulta_actual: str  # la que usa el investigador (puede reformularla el evaluador)
    pasos: list[dict]  # {"agente","accion","icono"} para ChatResponse.pasos
    es_alerta: bool  # audio no comprensible


class FiltrosSigem(BaseModel):
    tipo: Literal["decreto", "resolucion", "acuerdo", "ley", "formato"] | None = None
    numero: int | None = None
    anio: int | None = None
    tema: str | None = None


class AnalisisConsulta(BaseModel):
    intencion: Literal[
        "saludo", "fuera_de_dominio", "ambigua", "consulta_sigem", "seguimiento", "no_comprensible"
    ]
    consulta_autonoma: str  # pregunta reescrita para entenderse sin el historial, ortografía corregida
    filtros: FiltrosSigem | None = None
    usa_documento_activo: bool = False  # True si la pregunta se refiere al documento activo
    pregunta_aclaracion: str | None = None  # solo si intencion == "ambigua"
    razonamiento: str  # 1 frase, para depurar en Langfuse


class EvaluacionEvidencia(BaseModel):
    suficiente: bool
    motivo: str
    consulta_reformulada: str | None = None


# Prefijo del paso con que el evaluador marca que se reintenta la búsqueda; ruta_despues_evaluador
# y el investigador lo leen de `pasos` (el estado no tiene un campo aparte para esto).
PREFIJO_REINTENTO = "Reintentando: "
