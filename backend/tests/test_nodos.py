"""Pruebas unitarias de los nodos del grafo con LLM falsos (sin Gemini ni Chroma)."""

import importlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agente_sigem import prompts, tools
from agente_sigem.config import config
from agente_sigem.estado import AnalisisConsulta, EvaluacionEvidencia, FiltrosSigem

from fakes_llm import (  # noqa: E402
    CANDIDATOS,
    FRAGMENTO_TAXI,
    FUENTE_TAXI,
    FakeEstructurado,
    ModeloFalso,
    herramientas_dobles,
    instalar,
    llamada,
)

m_analista = importlib.import_module("agente_sigem.nodos.analista")
m_respuesta = importlib.import_module("agente_sigem.nodos.respuesta_directa")
m_investigador = importlib.import_module("agente_sigem.nodos.investigador")
m_evaluador = importlib.import_module("agente_sigem.nodos.evaluador")
m_redactor = importlib.import_module("agente_sigem.nodos.redactor")


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


def test_investigador_recibe_el_codigo_de_archivo_en_el_encargo(monkeypatch):
    estado = estado_base(
        analisis=analisis_dict(
            consulta_autonoma="¿Qué dice el documento 18399?",
            filtros={"tipo": None, "numero": None, "anio": None, "codigo": "18399", "tema": None},
        ),
        consulta_actual="¿Qué dice el documento 18399?",
    )

    _, modelo, _ = correr_investigador(monkeypatch, [AIMessage(content="Nada.")], {"buscar_fragmentos": {}}, estado)

    assert "codigo=18399" in modelo.entradas[0][1].content


def test_el_analista_distingue_codigo_de_archivo_en_su_prompt():
    assert "codigo:" in prompts.PROMPT_ANALISTA and "NUNCA \"ambigua\"" in prompts.PROMPT_ANALISTA
    assert FiltrosSigem(codigo="18399").codigo == "18399"


def test_investigador_no_traga_un_error_de_cuota(monkeypatch):
    class ModeloSinCuota(ModeloFalso):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("429 RESOURCE_EXHAUSTED. You exceeded your current quota")

    instalar(monkeypatch, m_investigador, ModeloSinCuota())
    monkeypatch.setattr(tools, "HERRAMIENTAS_INVESTIGADOR", herramientas_dobles({"buscar_fragmentos": {}}))

    with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
        m_investigador.investigador(estado_base(analisis=analisis_dict(), consulta_actual="taxi"))


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
# FUENTES CITADAS (redactor)
# ══════════════════════════════════════════════════════════════

FUENTES_VARIAS = [
    {"titulo": "Decreto 001 de 2019", "archivo": "a.pdf", "pagina": "1"},
    {"titulo": "Decreto 001 de 2019", "archivo": "a.pdf", "pagina": "2"},
    {"titulo": "Decreto 159 de 2018", "archivo": "b.pdf", "pagina": "1"},
    {"titulo": "Decreto 004 de 2019", "archivo": "c.pdf", "pagina": "3"},
]


def paginas(fuentes):
    return [(f["archivo"], f["pagina"]) for f in fuentes]


def test_fuentes_citadas_deja_solo_lo_que_la_respuesta_cita():
    texto = "La tarifa es $5.500 [Decreto 001 de 2019, p. 2]."

    assert paginas(m_redactor.fuentes_citadas(texto, FUENTES_VARIAS)) == [("a.pdf", "2")]


def test_fuentes_citadas_entiende_varias_paginas_y_varios_documentos():
    texto = "Uno [Decreto 001 de 2019, p. 1, 2] y dos [Decreto 004 de 2019, p. 3]; también [Decreto 159 de 2018]."

    assert paginas(m_redactor.fuentes_citadas(texto, FUENTES_VARIAS)) == [
        ("a.pdf", "1"), ("a.pdf", "2"), ("c.pdf", "3"), ("b.pdf", "1"),
    ]


def test_cita_sin_pagina_conserva_todas_las_paginas_del_documento():
    assert paginas(m_redactor.fuentes_citadas("Ver [Decreto 001 de 2019].", FUENTES_VARIAS)) == [
        ("a.pdf", "1"), ("a.pdf", "2"),
    ]


def test_pagina_inexistente_conserva_el_documento():
    assert paginas(m_redactor.fuentes_citadas("Ver [Decreto 004 de 2019, p. 9].", FUENTES_VARIAS)) == [("c.pdf", "3")]


def test_sin_citas_utilizables_se_conservan_todas_las_fuentes():
    assert m_redactor.fuentes_citadas("Respuesta sin citas.", FUENTES_VARIAS) == FUENTES_VARIAS
    assert m_redactor.fuentes_citadas("Cita rara [Decreto 999 de 2000, p. 1].", FUENTES_VARIAS) == FUENTES_VARIAS


def test_redactor_devuelve_las_fuentes_citadas_y_activa_ese_documento(monkeypatch):
    otro = {"fuente": {**FUENTE_TAXI, "titulo": "Decreto 159 de 2018", "archivo": "b.pdf", "pagina": "1"}, "contenido": "Cobro municipal"}
    estado = estado_base(
        analisis=analisis_dict(),
        evidencia=[otro, FRAGMENTO_TAXI],  # el primer fragmento NO es el citado
        fuentes=[otro["fuente"], FUENTE_TAXI],
    )

    resultado, _, _ = correr_redactor(monkeypatch, estado, "Tarifa $5.500 [Decreto 001 de 2019, p. 1].")

    assert resultado["fuentes"] == [FUENTE_TAXI]
    assert resultado["documento_activo"]["archivo"] == FUENTE_TAXI["archivo"]


def test_redactor_en_modo_aclaracion_no_toca_las_fuentes(monkeypatch):
    estado = estado_base(analisis=analisis_dict(), candidatos=CANDIDATOS, fuentes=[])

    resultado, _, _ = correr_redactor(monkeypatch, estado, "¿Cuál? [Resolución 4397 de 2018]")

    assert resultado["fuentes"] == []
