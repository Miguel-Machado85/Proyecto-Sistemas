"""LLM falsos y tools dobles compartidos por test_nodos, test_grafo y test_api."""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field


class FakeEstructurado:
    """Sustituye a un chat model con `with_structured_output`: devuelve salidas predefinidas o lanza."""

    def __init__(self, salidas=None, error: Exception | None = None):
        self.salidas = list(salidas or [])
        self.error = error
        self.entradas = []
        self.configs = []

    def with_structured_output(self, esquema, **kwargs):
        self.esquema = esquema
        return self

    def invoke(self, entrada, config=None, **kwargs):
        self.entradas.append(entrada)
        self.configs.append(config)
        if self.error:
            raise self.error
        return self.salidas.pop(0) if len(self.salidas) > 1 else self.salidas[0]

    @property
    def llamadas(self) -> int:
        return len(self.entradas)


class ModeloFalso(BaseChatModel):
    """Chat model falso: devuelve `respuestas` en orden y acepta bind_tools (para create_agent)."""

    respuestas: list = Field(default_factory=list)
    entradas: list = Field(default_factory=list)
    etiquetas: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.entradas.append(list(messages))
        self.etiquetas.append(list(getattr(run_manager, "tags", []) or []))
        mensaje = self.respuestas.pop(0) if self.respuestas else AIMessage(content="Fin.")
        return ChatResult(generations=[ChatGeneration(message=mensaje)])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "modelo-falso"


def llamada(nombre: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": nombre, "args": args, "id": f"id-{nombre}-{len(args)}"}])


def instalar(monkeypatch, modulo, falso) -> list:
    """Reemplaza crear_llm del módulo; devuelve la lista de (modelo, temperatura) solicitados."""
    pedidos = []

    def crear(nombre_modelo, temperature=0.0):
        pedidos.append((nombre_modelo, temperature))
        return falso

    monkeypatch.setattr(modulo, "crear_llm", crear)
    return pedidos


FUENTE_TAXI = {
    "titulo": "Decreto 001 de 2019",
    "archivo": "22598_decreto-no-0012019.pdf",
    "pagina": "1",
    "fecha_expedicion": "2019-01-03T08:00:00",
    "descripcion": "Tarifas de taxis y buses",
}
FRAGMENTO_TAXI = {"fuente": FUENTE_TAXI, "contenido": "La tarifa mínima del taxi es de $5.500."}
CANDIDATOS = [
    {"titulo": "Resolución 4397 de 2018", "archivo": "23655.pdf", "fecha_expedicion": "2018-12-13", "descripcion": None},
    {"titulo": "Resolución 4397 de 2017", "archivo": "23661.pdf", "fecha_expedicion": "2018-12-19", "descripcion": None},
]


def herramientas_dobles(salidas: dict[str, dict]):
    """Tools con los nombres reales que devuelven JSON fijo (el investigador no toca Chroma)."""

    def crear(nombre, salida):
        @tool(nombre)
        def doble(consulta: str = "", archivo: str | None = None, tipo: str | None = None,
                  numero: int | None = None, anio: int | None = None, pregunta: str | None = None,
                  tema: str | None = None) -> str:
            """Doble de prueba."""
            return json.dumps(salida, ensure_ascii=False)

        return doble

    return [crear(nombre, salida) for nombre, salida in salidas.items()]
