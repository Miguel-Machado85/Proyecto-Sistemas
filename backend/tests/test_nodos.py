"""Pruebas unitarias de los nodos del grafo con LLM falsos (sin Gemini ni Chroma)."""

import importlib
import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from agente_sigem import prompts, tools
from agente_sigem.config import config
from agente_sigem.estado import (
    AnalisisConsulta,
    EstadoAgente,
    EvaluacionEvidencia,
    FiltrosSigem,
)

m_analista = importlib.import_module("agente_sigem.nodos.analista")
m_respuesta = importlib.import_module("agente_sigem.nodos.respuesta_directa")
m_investigador = importlib.import_module("agente_sigem.nodos.investigador")
m_evaluador = importlib.import_module("agente_sigem.nodos.evaluador")
m_redactor = importlib.import_module("agente_sigem.nodos.redactor")


# ══════════════════════════════════════════════════════════════
# FAKES
# ══════════════════════════════════════════════════════════════

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


def estado_base(**cambios) -> dict:
    estado = {
        "messages": [HumanMessage(content="¿cuánto cuesta el taxi?")],
        "documento_activo": None,
        "perfil": "P0",
        "origen": "texto",
        "analisis": None,
        "evidencia": [],
        "fuentes": [],
        "candidatos": [],
        "intentos_busqueda": 0,
        "consulta_actual": "",
        "pasos": [],
        "es_alerta": False,
    }
    estado.update(cambios)
    return estado


def analisis_dict(**cambios) -> dict:
    datos = {
        "intencion": "consulta_sigem",
        "consulta_autonoma": "¿Cuáles son las tarifas del taxi?",
        "filtros": None,
        "usa_documento_activo": False,
        "pregunta_aclaracion": None,
        "razonamiento": "x",
    }
    datos.update(cambios)
    return datos


# ══════════════════════════════════════════════════════════════
# ANALISTA
# ══════════════════════════════════════════════════════════════

def test_analista_clasifica_y_usa_flash_lite_con_temperatura_cero(monkeypatch):
    falso = FakeEstructurado([AnalisisConsulta(intencion="saludo", consulta_autonoma="Hola", razonamiento="saludo")])
    pedidos = instalar(monkeypatch, m_analista, falso)
    estado = estado_base(messages=[HumanMessage(content="Hola")])

    resultado = m_analista.analista(estado, {"tags": ["prueba"]})

    assert resultado["analisis"]["intencion"] == "saludo"
    assert resultado["consulta_actual"] == "Hola"
    assert resultado["pasos"] == [{"agente": "analista", "accion": "Analizando la pregunta: saludo", "icono": "🧭"}]
    assert pedidos == [(config.ANALISTA_MODEL, 0)]
    assert falso.configs[0] == {"tags": ["prueba"]}  # el config se propaga al LLM (Langfuse)
    assert m_analista.ruta_despues_analista(resultado) == "respuesta_directa"
    assert "messages" not in resultado  # el mensaje del usuario nunca se reescribe


def test_voz_con_numeros_en_palabras_llega_al_investigador(monkeypatch):
    falso = FakeEstructurado([
        AnalisisConsulta(
            intencion="consulta_sigem",
            consulta_autonoma="¿Qué dice el Decreto 12 de 2019?",
            filtros=FiltrosSigem(tipo="decreto", numero=12, anio=2019),
            razonamiento="Referencia dicha en palabras.",
        )
    ])
    instalar(monkeypatch, m_analista, falso)
    estado = estado_base(
        messages=[HumanMessage(content="qué dice el decreto doce de dos mil diecinueve")], origen="voz"
    )

    resultado = m_analista.analista(estado)

    filtros = resultado["analisis"]["filtros"]
    assert (filtros["tipo"], filtros["numero"], filtros["anio"]) == ("decreto", 12, 2019)
    assert resultado["consulta_actual"] == "¿Qué dice el Decreto 12 de 2019?"
    assert m_analista.ruta_despues_analista(resultado) == "investigador"
    # El LLM recibió el origen y el mensaje tal cual lo dijo el usuario.
    entrada = falso.entradas[0][1].content
    assert "ORIGEN: voz" in entrada and "decreto doce de dos mil diecinueve" in entrada


def test_voz_con_menos_de_tres_caracteres_no_llama_al_llm(monkeypatch):
    falso = FakeEstructurado(error=AssertionError("no debe llamarse al LLM"))
    instalar(monkeypatch, m_analista, falso)

    resultado = m_analista.analista(estado_base(messages=[HumanMessage(content="eh")], origen="voz"))

    assert falso.llamadas == 0
    assert resultado["analisis"]["intencion"] == "no_comprensible"
    assert m_analista.ruta_despues_analista(resultado) == "respuesta_directa"


def test_texto_corto_si_pasa_por_el_llm(monkeypatch):
    falso = FakeEstructurado([AnalisisConsulta(intencion="saludo", consulta_autonoma="hi", razonamiento="x")])
    instalar(monkeypatch, m_analista, falso)

    m_analista.analista(estado_base(messages=[HumanMessage(content="hi")], origen="texto"))

    assert falso.llamadas == 1


def test_analista_fail_open_si_el_llm_lanza_excepcion(monkeypatch):
    instalar(monkeypatch, m_analista, FakeEstructurado(error=RuntimeError("cuota agotada")))
    estado = estado_base(messages=[HumanMessage(content="¿cuánto cuesta el taxi?")])

    resultado = m_analista.analista(estado)

    assert resultado["analisis"]["intencion"] == "consulta_sigem"
    assert resultado["consulta_actual"] == "¿cuánto cuesta el taxi?"
    assert m_analista.ruta_despues_analista(resultado) == "investigador"


def test_analista_fail_open_si_la_salida_no_se_puede_parsear(monkeypatch):
    instalar(monkeypatch, m_analista, FakeEstructurado([None]))

    resultado = m_analista.analista(estado_base())

    assert resultado["analisis"]["intencion"] == "consulta_sigem"


def test_analista_limita_el_historial(monkeypatch):
    falso = FakeEstructurado([AnalisisConsulta(intencion="consulta_sigem", consulta_autonoma="x", razonamiento="x")])
    instalar(monkeypatch, m_analista, falso)
    monkeypatch.setattr(config, "MAX_HISTORIAL_ANALISTA", 4)
    mensajes = []
    for i in range(6):
        mensajes += [HumanMessage(content=f"pregunta-{i}"), AIMessage(content=f"respuesta-{i}")]
    mensajes.append(HumanMessage(content="mensaje-actual"))

    m_analista.analista(estado_base(messages=mensajes))

    entrada = falso.entradas[0][1].content
    assert "pregunta-5" in entrada and "respuesta-5" in entrada and "pregunta-4" in entrada
    assert "pregunta-3" not in entrada and "pregunta-0" not in entrada
    assert "MENSAJE ACTUAL: mensaje-actual" in entrada


def test_analista_incluye_documento_activo_en_la_entrada(monkeypatch):
    falso = FakeEstructurado([AnalisisConsulta(intencion="seguimiento", consulta_autonoma="x", usa_documento_activo=True, razonamiento="x")])
    instalar(monkeypatch, m_analista, falso)
    documento = {"titulo": "Resolución No. 3615", "archivo": "18399.pdf", "descripcion": "Plan de talento humano"}

    resultado = m_analista.analista(estado_base(documento_activo=documento))

    assert "DOCUMENTO ACTIVO: Resolución No. 3615" in falso.entradas[0][1].content
    assert resultado["analisis"]["intencion"] == "seguimiento"
    assert m_analista.ruta_despues_analista(resultado) == "investigador"


def test_seguimiento_sin_documento_activo_se_corrige_a_consulta(monkeypatch):
    falso = FakeEstructurado([AnalisisConsulta(intencion="seguimiento", consulta_autonoma="x", usa_documento_activo=True, razonamiento="x")])
    instalar(monkeypatch, m_analista, falso)

    resultado = m_analista.analista(estado_base(documento_activo=None))

    assert resultado["analisis"]["intencion"] == "consulta_sigem"
    assert resultado["analisis"]["usa_documento_activo"] is False


@pytest.mark.parametrize(
    "intencion, esperado",
    [
        ("saludo", "respuesta_directa"),
        ("fuera_de_dominio", "respuesta_directa"),
        ("ambigua", "respuesta_directa"),
        ("no_comprensible", "respuesta_directa"),
        ("consulta_sigem", "investigador"),
        ("seguimiento", "investigador"),
    ],
)
def test_ruta_despues_analista(intencion, esperado):
    assert m_analista.ruta_despues_analista({"analisis": {"intencion": intencion}}) == esperado


def test_ruta_despues_analista_sin_analisis_busca():
    assert m_analista.ruta_despues_analista({"analisis": None}) == "investigador"


# ══════════════════════════════════════════════════════════════
# RESPUESTA DIRECTA
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "intencion, texto, alerta",
    [
        ("saludo", prompts.RESPUESTA_SALUDO, False),
        ("fuera_de_dominio", prompts.RESPUESTA_FUERA_DE_CONTEXTO, False),
        ("no_comprensible", prompts.MENSAJE_AUDIO_POCO_CLARO, True),
    ],
)
def test_respuesta_directa_usa_los_textos_fijos(intencion, texto, alerta):
    estado = estado_base(analisis=analisis_dict(intencion=intencion))

    resultado = m_respuesta.respuesta_directa(estado)

    assert resultado["messages"][0].content == texto
    assert resultado["es_alerta"] is alerta
    assert resultado["pasos"][-1]["agente"] == "respuesta_directa"


def test_respuesta_directa_ambigua_usa_la_pregunta_de_aclaracion():
    estado = estado_base(analisis=analisis_dict(intencion="ambigua", pregunta_aclaracion="¿De qué año es?"))

    assert m_respuesta.respuesta_directa(estado)["messages"][0].content == "¿De qué año es?"


def test_respuesta_directa_ambigua_sin_pregunta_usa_el_texto_generico():
    estado = estado_base(analisis=analisis_dict(intencion="ambigua"))

    assert m_respuesta.respuesta_directa(estado)["messages"][0].content == prompts.RESPUESTA_CONSULTA_AMBIGUA


def test_las_constantes_de_prompts_son_las_que_usa_la_api():
    from agente_sigem import api

    assert api.RESPUESTA_SALUDO is prompts.RESPUESTA_SALUDO
    assert api.MENSAJE_AUDIO_POCO_CLARO is prompts.MENSAJE_AUDIO_POCO_CLARO


# ══════════════════════════════════════════════════════════════
# INVESTIGADOR
# ══════════════════════════════════════════════════════════════

def correr_investigador(monkeypatch, respuestas, salidas_tools, estado=None, config_ejecucion=None):
    modelo = ModeloFalso(respuestas=list(respuestas))
    pedidos = instalar(monkeypatch, m_investigador, modelo)
    monkeypatch.setattr(tools, "HERRAMIENTAS_INVESTIGADOR", herramientas_dobles(salidas_tools))
    resultado = m_investigador.investigador(estado or estado_base(
        analisis=analisis_dict(), consulta_actual="¿Cuáles son las tarifas del taxi?"), config_ejecucion)
    return resultado, modelo, pedidos


def test_investigador_acumula_evidencia_fuentes_y_pasos(monkeypatch):
    salidas = {"buscar_fragmentos": {"fuentes": [FUENTE_TAXI], "fragmentos": [FRAGMENTO_TAXI]}}
    respuestas = [llamada("buscar_fragmentos", consulta="tarifas del taxi"), AIMessage(content="Encontré el Decreto 001.")]

    resultado, modelo, pedidos = correr_investigador(monkeypatch, respuestas, salidas)

    assert resultado["evidencia"] == [FRAGMENTO_TAXI]
    assert resultado["fuentes"] == [FUENTE_TAXI]
    assert resultado["candidatos"] == []
    assert resultado["intentos_busqueda"] == 1
    assert resultado["pasos"] == [
        {"agente": "investigador", "accion": "buscar_fragmentos: consulta=tarifas del taxi", "icono": "🔎"}
    ]
    assert pedidos == [(config.INVESTIGADOR_MODEL, 0.1)]


def test_investigador_empieza_con_historial_nuevo_y_encargo_completo(monkeypatch):
    salidas = {"buscar_fragmentos": {"fuentes": [], "fragmentos": []}}
    estado = estado_base(
        messages=[HumanMessage(content="viejo"), AIMessage(content="viejo"), HumanMessage(content="¿y la vigencia?")],
        analisis=analisis_dict(
            consulta_autonoma="¿Cuál es la vigencia de la Resolución 3615?",
            filtros={"tipo": "resolucion", "numero": 3615, "anio": None, "tema": None},
            usa_documento_activo=True,
        ),
        consulta_actual="¿Cuál es la vigencia de la Resolución 3615?",
        documento_activo={"titulo": "Resolución No. 3615", "archivo": "18399.pdf"},
    )

    _, modelo, _ = correr_investigador(monkeypatch, [AIMessage(content="Nada.")], salidas, estado)

    primera = modelo.entradas[0]
    assert [type(m).__name__ for m in primera] == ["SystemMessage", "HumanMessage"]  # sin el historial del chat
    encargo = primera[1].content
    assert "¿Cuál es la vigencia de la Resolución 3615?" in encargo
    assert "tipo=resolucion" in encargo and "numero=3615" in encargo
    assert "18399.pdf" in encargo
    assert "REINTENTO" not in encargo


def test_investigador_no_menciona_el_documento_activo_si_no_se_usa(monkeypatch):
    estado = estado_base(
        analisis=analisis_dict(usa_documento_activo=False),
        consulta_actual="¿qué normas hay sobre presupuesto?",
        documento_activo={"titulo": "Resolución No. 3615", "archivo": "18399.pdf"},
    )

    _, modelo, _ = correr_investigador(monkeypatch, [AIMessage(content="Nada.")], {"buscar_fragmentos": {}}, estado)

    assert "18399.pdf" not in modelo.entradas[0][1].content


def test_investigador_incluye_el_motivo_del_reintento(monkeypatch):
    estado = estado_base(
        analisis=analisis_dict(),
        consulta_actual="tarifa taxi urbano",
        intentos_busqueda=1,
        pasos=[{"agente": "evaluador", "accion": "Reintentando: los fragmentos son de otro tema", "icono": "✅"}],
    )

    resultado, modelo, _ = correr_investigador(monkeypatch, [AIMessage(content="Nada.")], {"buscar_fragmentos": {}}, estado)

    assert "REINTENTO" in modelo.entradas[0][1].content
    assert "los fragmentos son de otro tema" in modelo.entradas[0][1].content
    assert resultado["intentos_busqueda"] == 2


def test_investigador_deduplica_entre_llamadas_y_con_el_estado_previo(monkeypatch):
    otro = {"fuente": {**FUENTE_TAXI, "pagina": "2"}, "contenido": "El recargo nocturno es de $500."}
    salidas = {
        "buscar_fragmentos": {"fuentes": [FUENTE_TAXI, otro["fuente"]], "fragmentos": [FRAGMENTO_TAXI, otro]},
        "leer_documento": {"fuentes": [FUENTE_TAXI], "fragmentos": [FRAGMENTO_TAXI]},
    }
    respuestas = [
        llamada("buscar_fragmentos", consulta="taxi"),
        llamada("leer_documento", archivo=FUENTE_TAXI["archivo"]),
        AIMessage(content="Listo."),
    ]
    estado = estado_base(
        analisis=analisis_dict(), consulta_actual="taxi", evidencia=[FRAGMENTO_TAXI], fuentes=[FUENTE_TAXI]
    )

    resultado, _, _ = correr_investigador(monkeypatch, respuestas, salidas, estado)

    assert resultado["evidencia"] == [FRAGMENTO_TAXI, otro]
    assert len(resultado["fuentes"]) == 2  # (archivo, página 1) y (archivo, página 2)
    assert [p["accion"].split(":")[0] for p in resultado["pasos"]] == ["buscar_fragmentos", "leer_documento"]


def test_investigador_extrae_candidatos(monkeypatch):
    salidas = {
        "buscar_documento_por_referencia": {
            "fuentes": [], "fragmentos": [], "candidatos": CANDIDATOS, "aclaracion": "Hay varios"
        }
    }
    respuestas = [
        llamada("buscar_documento_por_referencia", tipo="resolucion", numero=4397),
        AIMessage(content="Hay dos."),
    ]

    resultado, _, _ = correr_investigador(monkeypatch, respuestas, salidas)

    assert resultado["candidatos"] == CANDIDATOS
    assert resultado["evidencia"] == []
    assert resultado["pasos"][0]["accion"] == "buscar_documento_por_referencia: tipo=resolucion, numero=4397"


def test_investigador_convierte_listados_en_evidencia_marcada(monkeypatch):
    documento = {k: FUENTE_TAXI[k] for k in ("titulo", "archivo", "fecha_expedicion", "descripcion")}
    salidas = {
        "listar_documentos": {
            "fuentes": [{**documento, "pagina": None}], "fragmentos": [], "documentos": [documento], "total": 9
        }
    }
    respuestas = [llamada("listar_documentos", tipo="decreto", anio=2019), AIMessage(content="Hay 9.")]

    resultado, _, _ = correr_investigador(monkeypatch, respuestas, salidas)

    assert [f.get("tipo") for f in resultado["evidencia"]] == ["listado", "listado"]
    assert "Decreto 001 de 2019" in resultado["evidencia"][0]["contenido"]
    assert "1 de 9" in resultado["evidencia"][1]["contenido"]
    assert len(resultado["fuentes"]) == 1


def test_investigador_conserva_lo_recolectado_si_se_alcanza_el_limite_de_pasos(monkeypatch):
    monkeypatch.setattr(config, "INVESTIGADOR_MAX_PASOS", 4)
    salidas = {"buscar_fragmentos": {"fuentes": [FUENTE_TAXI], "fragmentos": [FRAGMENTO_TAXI]}}
    respuestas = [llamada("buscar_fragmentos", consulta=f"intento {i}") for i in range(12)]

    resultado, _, _ = correr_investigador(monkeypatch, respuestas, salidas)

    assert resultado["evidencia"] == [FRAGMENTO_TAXI]
    assert resultado["intentos_busqueda"] == 1
    assert 1 <= len(resultado["pasos"]) <= 4


def test_investigador_no_lanza_si_el_modelo_falla(monkeypatch):
    class ModeloRoto(ModeloFalso):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("503 de Gemini")

    instalar(monkeypatch, m_investigador, ModeloRoto())
    monkeypatch.setattr(tools, "HERRAMIENTAS_INVESTIGADOR", herramientas_dobles({"buscar_fragmentos": {}}))

    resultado = m_investigador.investigador(estado_base(analisis=analisis_dict(), consulta_actual="taxi"))

    assert resultado["evidencia"] == [] and resultado["intentos_busqueda"] == 1


def test_investigador_propaga_el_config(monkeypatch):
    salidas = {"buscar_fragmentos": {"fuentes": [], "fragmentos": []}}

    _, modelo, _ = correr_investigador(
        monkeypatch, [AIMessage(content="Nada.")], salidas, config_ejecucion={"tags": ["langfuse-prueba"]}
    )

    assert "langfuse-prueba" in modelo.etiquetas[0]


# ══════════════════════════════════════════════════════════════
# EVALUADOR
# ══════════════════════════════════════════════════════════════

def estado_evaluacion(**cambios):
    base = dict(
        analisis=analisis_dict(), consulta_actual="tarifa taxi", evidencia=[FRAGMENTO_TAXI],
        fuentes=[FUENTE_TAXI], intentos_busqueda=1,
    )
    base.update(cambios)
    return estado_base(**base)


def test_evaluador_suficiente_pasa_al_redactor(monkeypatch):
    falso = FakeEstructurado([EvaluacionEvidencia(suficiente=True, motivo="ok")])
    pedidos = instalar(monkeypatch, m_evaluador, falso)

    resultado = m_evaluador.evaluador(estado_evaluacion(), {"tags": ["t"]})

    assert resultado["pasos"][-1] == {"agente": "evaluador", "accion": "Evidencia suficiente", "icono": "✅"}
    assert m_evaluador.ruta_despues_evaluador({**estado_evaluacion(), **resultado}) == "redactor"
    assert pedidos == [(config.EVALUADOR_MODEL, 0)]
    assert falso.configs[0] == {"tags": ["t"]}


def test_evaluador_insuficiente_reformula_y_pide_reintento(monkeypatch):
    falso = FakeEstructurado([
        EvaluacionEvidencia(suficiente=False, motivo="son de otro tema", consulta_reformulada="tarifas servicio taxi 2019")
    ])
    instalar(monkeypatch, m_evaluador, falso)
    estado = estado_evaluacion()

    resultado = m_evaluador.evaluador(estado)
    nuevo = {**estado, **resultado}

    assert resultado["consulta_actual"] == "tarifas servicio taxi 2019"
    assert resultado["pasos"][-1]["accion"] == "Reintentando: son de otro tema"
    assert "intentos_busqueda" not in resultado  # lo incrementa el investigador al ejecutarse
    assert m_evaluador.ruta_despues_evaluador(nuevo) == "investigador"


def test_evaluador_sin_reformulacion_conserva_la_consulta(monkeypatch):
    instalar(monkeypatch, m_evaluador, FakeEstructurado([EvaluacionEvidencia(suficiente=False, motivo="poco")]))

    resultado = m_evaluador.evaluador(estado_evaluacion())

    assert resultado["consulta_actual"] == "tarifa taxi"


def test_evaluador_no_reintenta_si_se_agotaron_los_intentos(monkeypatch):
    falso = FakeEstructurado([EvaluacionEvidencia(suficiente=False, motivo="poco", consulta_reformulada="otra")])
    instalar(monkeypatch, m_evaluador, falso)
    estado = estado_evaluacion(intentos_busqueda=config.MAX_INTENTOS_BUSQUEDA)

    resultado = m_evaluador.evaluador(estado)

    assert "consulta_actual" not in resultado
    assert m_evaluador.ruta_despues_evaluador({**estado, **resultado}) == "redactor"


def test_evaluador_atajo_candidatos_sin_evidencia_no_llama_al_llm(monkeypatch):
    falso = FakeEstructurado(error=AssertionError("no debe llamarse al LLM"))
    instalar(monkeypatch, m_evaluador, falso)
    estado = estado_evaluacion(evidencia=[], candidatos=CANDIDATOS, intentos_busqueda=1)

    resultado = m_evaluador.evaluador(estado)

    assert falso.llamadas == 0
    assert m_evaluador.ruta_despues_evaluador({**estado, **resultado}) == "redactor"


def test_evaluador_atajo_sin_evidencia_e_intentos_agotados(monkeypatch):
    falso = FakeEstructurado(error=AssertionError("no debe llamarse al LLM"))
    instalar(monkeypatch, m_evaluador, falso)
    estado = estado_evaluacion(evidencia=[], intentos_busqueda=config.MAX_INTENTOS_BUSQUEDA)

    resultado = m_evaluador.evaluador(estado)

    assert falso.llamadas == 0
    assert m_evaluador.ruta_despues_evaluador({**estado, **resultado}) == "redactor"


def test_evaluador_sin_evidencia_y_con_intentos_pendientes_si_consulta_al_llm(monkeypatch):
    falso = FakeEstructurado([EvaluacionEvidencia(suficiente=False, motivo="vacío", consulta_reformulada="taxi tarifa")])
    instalar(monkeypatch, m_evaluador, falso)
    estado = estado_evaluacion(evidencia=[], intentos_busqueda=1)

    resultado = m_evaluador.evaluador(estado)

    assert falso.llamadas == 1
    assert m_evaluador.ruta_despues_evaluador({**estado, **resultado}) == "investigador"


def test_evaluador_deshabilitado_pasa_siempre_al_redactor(monkeypatch):
    falso = FakeEstructurado(error=AssertionError("no debe llamarse al LLM"))
    instalar(monkeypatch, m_evaluador, falso)
    monkeypatch.setattr(config, "EVALUADOR_HABILITADO", False)
    estado = estado_evaluacion(intentos_busqueda=1)

    resultado = m_evaluador.evaluador(estado)

    assert falso.llamadas == 0
    assert m_evaluador.ruta_despues_evaluador({**estado, **resultado}) == "redactor"


def test_evaluador_fail_open_si_el_llm_falla(monkeypatch):
    instalar(monkeypatch, m_evaluador, FakeEstructurado(error=RuntimeError("boom")))
    estado = estado_evaluacion()

    resultado = m_evaluador.evaluador(estado)

    assert m_evaluador.ruta_despues_evaluador({**estado, **resultado}) == "redactor"


def test_ruta_despues_evaluador_solo_depende_del_ultimo_paso():
    reintento = {"agente": "evaluador", "accion": "Reintentando: x", "icono": "✅"}
    investigador_paso = {"agente": "investigador", "accion": "buscar_fragmentos: consulta=x", "icono": "🔎"}

    assert m_evaluador.ruta_despues_evaluador({"pasos": [reintento]}) == "investigador"
    # Un reintento anterior (ya atendido por el investigador) no vuelve a disparar la ruta.
    assert m_evaluador.ruta_despues_evaluador({"pasos": [reintento, investigador_paso]}) == "redactor"
    assert m_evaluador.ruta_despues_evaluador({"pasos": []}) == "redactor"


# ══════════════════════════════════════════════════════════════
# REDACTOR
# ══════════════════════════════════════════════════════════════

def correr_redactor(monkeypatch, estado, respuesta="Respuesta final."):
    modelo = ModeloFalso(respuestas=[AIMessage(content=respuesta)])
    pedidos = instalar(monkeypatch, m_redactor, modelo)
    return m_redactor.redactor(estado), modelo, pedidos


def test_redactor_modo_respuesta_con_estilo_del_perfil_y_citas(monkeypatch):
    estado = estado_base(
        perfil="P1", analisis=analisis_dict(), evidencia=[FRAGMENTO_TAXI], fuentes=[FUENTE_TAXI],
        documento_activo=None,
    )

    resultado, modelo, pedidos = correr_redactor(monkeypatch, estado)

    sistema, entrada = modelo.entradas[0][0].content, modelo.entradas[0][1].content
    assert "MODO: RESPUESTA" in sistema
    assert prompts.ESTILO_POR_PERFIL["P1"] in sistema
    assert "[Decreto 001 de 2019, p. 3]" in sistema  # formato de cita
    assert "[1] Decreto 001 de 2019, p. 1" in entrada and "$5.500" in entrada
    assert "PREGUNTA ORIGINAL DEL USUARIO: ¿cuánto cuesta el taxi?" in entrada
    assert pedidos == [(config.REDACTOR_MODEL, 0.3)]
    assert resultado["messages"][0].content == "Respuesta final."
    assert resultado["pasos"][-1] == {"agente": "redactor", "accion": "Redactando respuesta para perfil P1", "icono": "✍️"}


def test_redactor_actualiza_documento_activo_con_la_primera_fuente(monkeypatch):
    estado = estado_base(analisis=analisis_dict(), evidencia=[FRAGMENTO_TAXI], fuentes=[FUENTE_TAXI],
                         documento_activo={"titulo": "Otro", "archivo": "otro.pdf"})

    resultado, _, _ = correr_redactor(monkeypatch, estado)

    assert resultado["documento_activo"] == {
        "titulo": "Decreto 001 de 2019",
        "archivo": "22598_decreto-no-0012019.pdf",
        "descripcion": "Tarifas de taxis y buses",
        "fecha_expedicion": "2019-01-03T08:00:00",
    }


def test_redactor_sin_resultados_conserva_el_documento_activo(monkeypatch):
    anterior = {"titulo": "Resolución No. 3615", "archivo": "18399.pdf", "descripcion": None, "fecha_expedicion": None}
    estado = estado_base(analisis=analisis_dict(), documento_activo=anterior)

    resultado, modelo, _ = correr_redactor(monkeypatch, estado, "No está disponible en SIGEM.")

    assert "MODO: SIN RESULTADOS" in modelo.entradas[0][0].content
    assert "Alcaldía de Marinilla" in modelo.entradas[0][0].content
    assert resultado["documento_activo"] == anterior


def test_redactor_modo_aclaracion_lista_los_candidatos(monkeypatch):
    estado = estado_base(analisis=analisis_dict(), candidatos=CANDIDATOS)

    resultado, modelo, _ = correr_redactor(monkeypatch, estado, "¿Cuál de estas?")

    assert "MODO: ACLARACIÓN" in modelo.entradas[0][0].content
    entrada = modelo.entradas[0][1].content
    assert "Resolución 4397 de 2018" in entrada and "Resolución 4397 de 2017" in entrada
    assert resultado["documento_activo"] is None


def test_redactor_un_listado_no_activa_ningun_documento(monkeypatch):
    listado = {"fuente": {**FUENTE_TAXI, "pagina": None}, "contenido": "Decreto 001 de 2019", "tipo": "listado"}
    estado = estado_base(analisis=analisis_dict(), evidencia=[listado], fuentes=[FUENTE_TAXI], documento_activo=None)

    resultado, modelo, _ = correr_redactor(monkeypatch, estado)

    assert "MODO: RESPUESTA" in modelo.entradas[0][0].content
    assert resultado["documento_activo"] is None


def test_redactor_perfil_desconocido_usa_p0_y_texto_plano_de_bloques(monkeypatch):
    estado = estado_base(perfil="P9", analisis=analisis_dict(), evidencia=[FRAGMENTO_TAXI])
    modelo = ModeloFalso(respuestas=[AIMessage(content=[{"type": "text", "text": "Hola "}, {"type": "text", "text": "mundo"}])])
    instalar(monkeypatch, m_redactor, modelo)

    resultado = m_redactor.redactor(estado)

    assert prompts.ESTILO_POR_PERFIL["P0"] in modelo.entradas[0][0].content
    assert resultado["messages"][0].content == "Hola mundo"


def test_redactor_propaga_el_config(monkeypatch):
    modelo = ModeloFalso(respuestas=[AIMessage(content="ok")])
    instalar(monkeypatch, m_redactor, modelo)

    m_redactor.redactor(estado_base(analisis=analisis_dict(), evidencia=[FRAGMENTO_TAXI]), {"tags": ["lf"]})

    assert "lf" in modelo.etiquetas[0]


# ══════════════════════════════════════════════════════════════
# INTEGRACIÓN DE NODOS EN UN GRAFO MÍNIMO (sección 12: persistencia entre turnos)
# ══════════════════════════════════════════════════════════════

def construir_grafo_minimo():
    grafo = StateGraph(EstadoAgente)
    grafo.add_node("analista", m_analista.analista)
    grafo.add_node("respuesta_directa", m_respuesta.respuesta_directa)
    grafo.add_node("investigador", m_investigador.investigador)
    grafo.add_node("evaluador", m_evaluador.evaluador)
    grafo.add_node("redactor", m_redactor.redactor)
    grafo.add_edge(START, "analista")
    grafo.add_conditional_edges(
        "analista", m_analista.ruta_despues_analista, {"investigador": "investigador", "respuesta_directa": "respuesta_directa"}
    )
    grafo.add_edge("investigador", "evaluador")
    grafo.add_conditional_edges(
        "evaluador", m_evaluador.ruta_despues_evaluador, {"investigador": "investigador", "redactor": "redactor"}
    )
    grafo.add_edge("redactor", END)
    grafo.add_edge("respuesta_directa", END)
    return grafo.compile(checkpointer=MemorySaver())


def entrada_turno(mensaje: str, origen: str = "texto") -> dict:
    """Lo que api.py pasa en cada invoke: los campos por turno van vacíos para reiniciarlos."""
    return {
        "messages": [HumanMessage(content=mensaje)], "perfil": "P0", "origen": origen, "analisis": None,
        "evidencia": [], "fuentes": [], "candidatos": [], "intentos_busqueda": 0, "consulta_actual": "",
        "pasos": [], "es_alerta": False,
    }


def instalar_grafo(monkeypatch, analisis, evaluaciones, investigador_msgs, redactor_msgs, salidas_tools):
    instalar(monkeypatch, m_analista, FakeEstructurado(analisis))
    instalar(monkeypatch, m_evaluador, FakeEstructurado(evaluaciones))
    instalar(monkeypatch, m_investigador, ModeloFalso(respuestas=list(investigador_msgs)))
    instalar(monkeypatch, m_redactor, ModeloFalso(respuestas=list(redactor_msgs)))
    monkeypatch.setattr(tools, "HERRAMIENTAS_INVESTIGADOR", herramientas_dobles(salidas_tools))


def test_grafo_minimo_consulta_con_reintento_y_estado_por_turno(monkeypatch):
    instalar_grafo(
        monkeypatch,
        analisis=[
            AnalisisConsulta(intencion="consulta_sigem", consulta_autonoma="tarifa del taxi", razonamiento="x"),
            AnalisisConsulta(intencion="saludo", consulta_autonoma="gracias", razonamiento="x"),
        ],
        evaluaciones=[
            EvaluacionEvidencia(suficiente=False, motivo="otro tema", consulta_reformulada="tarifas de taxi 2019"),
            EvaluacionEvidencia(suficiente=True, motivo="ok"),
        ],
        investigador_msgs=[
            llamada("buscar_fragmentos", consulta="tarifa del taxi"), AIMessage(content="Nada claro."),
            llamada("buscar_fragmentos", consulta="tarifas de taxi 2019"), AIMessage(content="Listo."),
        ],
        redactor_msgs=[AIMessage(content="La tarifa mínima es $5.500 [Decreto 001 de 2019, p. 1].")],
        salidas_tools={"buscar_fragmentos": {"fuentes": [FUENTE_TAXI], "fragmentos": [FRAGMENTO_TAXI]}},
    )
    grafo = construir_grafo_minimo()
    hilo = {"configurable": {"thread_id": "t1"}}

    turno1 = grafo.invoke(entrada_turno("¿cuánto cuesta el taxi?"), hilo)

    assert turno1["intentos_busqueda"] == 2  # el investigador corrió dos veces
    assert turno1["consulta_actual"] == "tarifas de taxi 2019"
    assert [p["agente"] for p in turno1["pasos"]] == [
        "analista", "investigador", "evaluador", "investigador", "evaluador", "redactor"
    ]
    assert turno1["fuentes"] == [FUENTE_TAXI] and len(turno1["evidencia"]) == 1  # sin duplicados entre intentos
    assert turno1["documento_activo"]["archivo"] == FUENTE_TAXI["archivo"]
    # messages queda limpio: solo la pregunta y la respuesta final.
    assert [type(m).__name__ for m in turno1["messages"]] == ["HumanMessage", "AIMessage"]

    turno2 = grafo.invoke(entrada_turno("gracias"), hilo)

    assert turno2["messages"][-1].content == prompts.RESPUESTA_SALUDO
    assert [p["agente"] for p in turno2["pasos"]] == ["analista", "respuesta_directa"]  # no se acumulan
    assert turno2["fuentes"] == [] and turno2["candidatos"] == [] and turno2["evidencia"] == []
    assert turno2["intentos_busqueda"] == 0
    assert turno2["documento_activo"]["archivo"] == FUENTE_TAXI["archivo"]  # persiste entre turnos
    assert len(turno2["messages"]) == 4


def test_grafo_minimo_candidatos_van_al_redactor_en_modo_aclaracion(monkeypatch):
    instalar_grafo(
        monkeypatch,
        analisis=[AnalisisConsulta(intencion="consulta_sigem", consulta_autonoma="Resolución 4397", razonamiento="x")],
        evaluaciones=[EvaluacionEvidencia(suficiente=True, motivo="no debe usarse")],
        investigador_msgs=[
            llamada("buscar_documento_por_referencia", tipo="resolucion", numero=4397), AIMessage(content="Dos."),
        ],
        redactor_msgs=[AIMessage(content="¿Cuál de las dos?")],
        salidas_tools={"buscar_documento_por_referencia": {"fuentes": [], "fragmentos": [], "candidatos": CANDIDATOS}},
    )
    grafo = construir_grafo_minimo()

    resultado = grafo.invoke(entrada_turno("Resolución 4397"), {"configurable": {"thread_id": "t2"}})

    assert resultado["candidatos"] == CANDIDATOS
    assert resultado["intentos_busqueda"] == 1
    assert resultado["documento_activo"] is None
    assert [p["agente"] for p in resultado["pasos"]] == ["analista", "investigador", "evaluador", "redactor"]


def test_grafo_minimo_voz_no_comprensible_marca_alerta(monkeypatch):
    instalar_grafo(
        monkeypatch,
        analisis=[AnalisisConsulta(intencion="no_comprensible", consulta_autonoma="eh mm", razonamiento="ruido")],
        evaluaciones=[EvaluacionEvidencia(suficiente=True, motivo="")],
        investigador_msgs=[], redactor_msgs=[], salidas_tools={"buscar_fragmentos": {}},
    )
    grafo = construir_grafo_minimo()

    resultado = grafo.invoke(entrada_turno("eh mm ajá", origen="voz"), {"configurable": {"thread_id": "t3"}})

    assert resultado["es_alerta"] is True
    assert resultado["messages"][-1].content == prompts.MENSAJE_AUDIO_POCO_CLARO


def test_grafo_minimo_analista_con_excepcion_busca_igual(monkeypatch):
    instalar_grafo(
        monkeypatch,
        analisis=[AnalisisConsulta(intencion="consulta_sigem", consulta_autonoma="x", razonamiento="x")],
        evaluaciones=[EvaluacionEvidencia(suficiente=True, motivo="ok")],
        investigador_msgs=[llamada("buscar_fragmentos", consulta="x"), AIMessage(content="ok")],
        redactor_msgs=[AIMessage(content="Respuesta.")],
        salidas_tools={"buscar_fragmentos": {"fuentes": [FUENTE_TAXI], "fragmentos": [FRAGMENTO_TAXI]}},
    )
    instalar(monkeypatch, m_analista, FakeEstructurado(error=RuntimeError("Gemini caído")))
    grafo = construir_grafo_minimo()

    resultado = grafo.invoke(entrada_turno("¿cuánto cuesta el taxi?"), {"configurable": {"thread_id": "t4"}})

    assert resultado["analisis"]["intencion"] == "consulta_sigem"
    assert resultado["messages"][-1].content == "Respuesta."
