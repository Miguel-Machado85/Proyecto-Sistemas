"""Entrega de archivos (Fase 4): analista, investigador, evaluador, redactor, grafo y compuerta de la API.

Todo con LLM falsos: ni Gemini, ni Chroma, ni Drive.
"""

import importlib

import pytest
from fakes_llm import (
    ARCHIVO_TAXI,
    ARCHIVO_TAXI_SALIDA,
    CANDIDATOS,
    FRAGMENTO_TAXI,
    FUENTE_TAXI,
    NO_DISPONIBLE_SALIDA,
    FakeEstructurado,
    ModeloFalso,
    herramientas_dobles,
    instalar,
    llamada,
)
from fastapi.testclient import TestClient
from grafo_falso import AIMessage, AnalisisConsulta, EvaluacionEvidencia, M_EVALUADOR, instalar_grafo
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agente_sigem import api, prompts, tools
from agente_sigem.agent import construir_grafo
from agente_sigem.config import config

m_analista = importlib.import_module("agente_sigem.nodos.analista")
m_investigador = importlib.import_module("agente_sigem.nodos.investigador")
m_evaluador = importlib.import_module("agente_sigem.nodos.evaluador")
m_redactor = importlib.import_module("agente_sigem.nodos.redactor")

SALIDA_REFERENCIA = {"fuentes": [FUENTE_TAXI], "fragmentos": [FRAGMENTO_TAXI]}
LLAMADA_REFERENCIA = llamada("buscar_documento_por_referencia", tipo="decreto", numero=1, anio=2019)
LLAMADA_DESCARGA = llamada("preparar_descarga", archivo=ARCHIVO_TAXI["archivo"])


@pytest.fixture(autouse=True)
def sin_langfuse(monkeypatch):
    monkeypatch.setattr(api, "obtener_langfuse_handler", lambda: None)


@pytest.fixture
def agente():
    return construir_grafo(MemorySaver())


def turno(agente, mensaje, thread_id="hilo", perfil="P0"):
    return api.ejecutar_turno(agente, thread_id, mensaje, perfil, "texto", [perfil])


def pide_archivo(texto="Descargar el archivo del Decreto 001 de 2019", contenido=False, **extra):
    return AnalisisConsulta(
        intencion="consulta_sigem", consulta_autonoma=texto, solicita_archivo=True,
        incluye_contenido=contenido, razonamiento="x", **extra,
    )


def pide_contenido(texto="¿Qué dice el Decreto 001 de 2019?"):
    return AnalisisConsulta(intencion="consulta_sigem", consulta_autonoma=texto, razonamiento="x")


def sistema_del_redactor(fakes) -> str:
    return fakes["redactor"].entradas[0][0].content


# ══════════════════════════════════════════════════════════════
# estado y analista
# ══════════════════════════════════════════════════════════════

def test_el_analisis_no_pide_archivo_por_defecto():
    analisis = AnalisisConsulta(intencion="consulta_sigem", consulta_autonoma="x", razonamiento="x")

    assert analisis.solicita_archivo is False and analisis.incluye_contenido is False


def test_el_analista_ve_las_reglas_y_los_pares_contrastados():
    prompt = prompts.PROMPT_ANALISTA

    assert "solicita_archivo" in prompt and "incluye_contenido" in prompt
    for par in ("¿Qué dice el Decreto 001 de 2019?", "Descárgame el Decreto 001 de 2019", "Dame el Decreto 001 de 2019",
                "pásamelo", "Mándame el PDF de la Resolución 4496 de 2018", "¿Dónde puedo ver la Ley 387 de 1997 completa?"):
        assert par in prompt


@pytest.mark.parametrize("intencion", ["saludo", "fuera_de_dominio", "ambigua", "no_comprensible"])
def test_sin_busqueda_no_hay_archivo(monkeypatch, intencion):
    salida = AnalisisConsulta(
        intencion=intencion, consulta_autonoma="pásame el archivo", solicita_archivo=True, incluye_contenido=True, razonamiento="x"
    )
    instalar(monkeypatch, m_analista, FakeEstructurado([salida]))

    analisis = m_analista.analizar({"messages": [HumanMessage(content="pásame el archivo")], "origen": "texto"})

    assert analisis.solicita_archivo is False and analisis.incluye_contenido is False


def test_incluye_contenido_sin_pedir_archivo_se_descarta(monkeypatch):
    salida = AnalisisConsulta(intencion="consulta_sigem", consulta_autonoma="x", incluye_contenido=True, razonamiento="x")
    instalar(monkeypatch, m_analista, FakeEstructurado([salida]))

    assert m_analista.analizar({"messages": [HumanMessage(content="x")], "origen": "texto"}).incluye_contenido is False


# ══════════════════════════════════════════════════════════════
# investigador
# ══════════════════════════════════════════════════════════════

def estado_investigador(analisis, **cambios):
    estado = {
        "messages": [HumanMessage(content="pásame el decreto")], "documento_activo": None, "perfil": "P0", "origen": "texto",
        "analisis": analisis.model_dump(), "evidencia": [], "fuentes": [], "candidatos": [], "intentos_busqueda": 0,
        "consulta_actual": analisis.consulta_autonoma, "pasos": [], "es_alerta": False, "archivos": [], "archivos_no_disponibles": [],
    }
    estado.update(cambios)
    return estado


def correr_investigador(monkeypatch, analisis, respuestas, salidas, salidas_descarga=None, **cambios):
    modelo = ModeloFalso(respuestas=list(respuestas))
    instalar(monkeypatch, m_investigador, modelo)
    monkeypatch.setattr(tools, "HERRAMIENTAS_INVESTIGADOR", herramientas_dobles(salidas))
    monkeypatch.setattr(tools, "HERRAMIENTAS_DESCARGA", herramientas_dobles({"preparar_descarga": salidas_descarga or ARCHIVO_TAXI_SALIDA}))
    return m_investigador.investigador(estado_investigador(analisis, **cambios)), modelo


def test_sin_solicita_archivo_preparar_descarga_no_existe_para_el_modelo(monkeypatch):
    resultado, modelo = correr_investigador(
        monkeypatch, pide_contenido(), [LLAMADA_REFERENCIA, AIMessage(content="ok")], {"buscar_documento_por_referencia": SALIDA_REFERENCIA}
    )

    assert "preparar_descarga" not in modelo.herramientas_vinculadas
    assert "buscar_documento_por_referencia" in modelo.herramientas_vinculadas
    assert resultado["archivos"] == [] and resultado["evidencia"] == [FRAGMENTO_TAXI]
    assert "preparar_descarga" not in modelo.entradas[0][0].content  # ni siquiera aparece en el prompt


def test_con_solicita_archivo_el_modelo_recibe_la_tool_y_las_instrucciones(monkeypatch):
    resultado, modelo = correr_investigador(
        monkeypatch, pide_archivo(), [LLAMADA_REFERENCIA, LLAMADA_DESCARGA, AIMessage(content="ok")],
        {"buscar_documento_por_referencia": SALIDA_REFERENCIA},
    )

    assert "preparar_descarga" in modelo.herramientas_vinculadas
    sistema, encargo = modelo.entradas[0][0].content, modelo.entradas[0][1].content
    assert "EL USUARIO PIDE EL ARCHIVO" in sistema
    assert "ARCHIVO del documento" in encargo and "preparar_descarga" in encargo
    assert resultado["archivos"] == [ARCHIVO_TAXI]


def test_pedido_solo_de_archivo_descarta_los_fragmentos_usados_para_identificar(monkeypatch):
    resultado, _ = correr_investigador(
        monkeypatch, pide_archivo(contenido=False), [LLAMADA_REFERENCIA, LLAMADA_DESCARGA, AIMessage(content="ok")],
        {"buscar_documento_por_referencia": SALIDA_REFERENCIA},
    )

    assert resultado["archivos"] == [ARCHIVO_TAXI]
    assert resultado["evidencia"] == [] and resultado["fuentes"] == []


def test_pedido_mixto_conserva_la_evidencia_y_el_archivo(monkeypatch):
    resultado, modelo = correr_investigador(
        monkeypatch, pide_archivo(contenido=True), [LLAMADA_REFERENCIA, LLAMADA_DESCARGA, AIMessage(content="ok")],
        {"buscar_documento_por_referencia": SALIDA_REFERENCIA},
    )

    assert resultado["archivos"] == [ARCHIVO_TAXI]
    assert resultado["evidencia"] == [FRAGMENTO_TAXI] and resultado["fuentes"] == [FUENTE_TAXI]
    assert "reúne también la evidencia" in modelo.entradas[0][1].content


def test_archivos_sin_duplicados_y_maximo_tres(monkeypatch):
    salida = {"fuentes": [], "fragmentos": [], "archivos": [{**ARCHIVO_TAXI, "archivo": f"d{i}.pdf"} for i in range(5)]}
    respuestas = [LLAMADA_DESCARGA, llamada("preparar_descarga", archivo="d1.pdf"), AIMessage(content="ok")]

    resultado, _ = correr_investigador(monkeypatch, pide_archivo(), respuestas, {"buscar_fragmentos": {}}, salidas_descarga=salida)

    assert [a["archivo"] for a in resultado["archivos"]] == ["d0.pdf", "d1.pdf", "d2.pdf"]


def test_documento_sin_original_queda_como_no_disponible(monkeypatch):
    resultado, _ = correr_investigador(
        monkeypatch, pide_archivo(), [LLAMADA_REFERENCIA, LLAMADA_DESCARGA, AIMessage(content="ok")],
        {"buscar_documento_por_referencia": SALIDA_REFERENCIA}, salidas_descarga=NO_DISPONIBLE_SALIDA,
    )

    assert resultado["archivos"] == []
    assert resultado["archivos_no_disponibles"] == [{"titulo": "Decreto 001 de 2019", "archivo": ARCHIVO_TAXI["archivo"]}]
    assert resultado["evidencia"] == []  # pedido solo de archivo: no se resume


def test_si_nunca_se_llama_preparar_descarga_la_evidencia_se_conserva(monkeypatch):
    resultado, _ = correr_investigador(
        monkeypatch, pide_archivo(), [LLAMADA_REFERENCIA, AIMessage(content="ok")], {"buscar_documento_por_referencia": SALIDA_REFERENCIA}
    )

    assert resultado["archivos"] == [] and resultado["evidencia"] == [FRAGMENTO_TAXI]


# ══════════════════════════════════════════════════════════════
# evaluador
# ══════════════════════════════════════════════════════════════

def correr_evaluador(monkeypatch, **cambios):
    falso = FakeEstructurado([EvaluacionEvidencia(suficiente=True, motivo="ok")])
    instalar(monkeypatch, m_evaluador, falso)
    estado = estado_investigador(pide_archivo(), intentos_busqueda=1, **cambios)
    return m_evaluador.evaluador(estado), falso


def test_evaluador_atajo_archivos_sin_evidencia_no_llama_al_llm(monkeypatch):
    resultado, falso = correr_evaluador(monkeypatch, archivos=[ARCHIVO_TAXI])

    assert falso.llamadas == 0
    assert resultado["pasos"][-1]["accion"] == "Archivo listo para entregar"
    assert m_evaluador.ruta_despues_evaluador(resultado | {"pasos": resultado["pasos"]}) == "redactor"


def test_evaluador_atajo_no_disponible_sin_evidencia(monkeypatch):
    resultado, falso = correr_evaluador(monkeypatch, archivos_no_disponibles=[{"titulo": "T", "archivo": "a.pdf"}])

    assert falso.llamadas == 0 and "no está disponible" in resultado["pasos"][-1]["accion"]


def test_evaluador_pedido_mixto_evalua_la_evidencia_como_siempre(monkeypatch):
    resultado, falso = correr_evaluador(monkeypatch, archivos=[ARCHIVO_TAXI], evidencia=[FRAGMENTO_TAXI])

    assert falso.llamadas == 1 and resultado["pasos"][-1]["accion"] == "Evidencia suficiente"


def test_evaluador_los_candidatos_ganan_a_los_archivos(monkeypatch):
    resultado, falso = correr_evaluador(monkeypatch, archivos=[ARCHIVO_TAXI], candidatos=CANDIDATOS)

    assert falso.llamadas == 0 and "aclaración" in resultado["pasos"][-1]["accion"]


# ══════════════════════════════════════════════════════════════
# redactor
# ══════════════════════════════════════════════════════════════

def correr_redactor(monkeypatch, estado, respuesta="Aquí tienes el Decreto 001 de 2019."):
    modelo = ModeloFalso(respuestas=[AIMessage(content=respuesta)])
    instalar(monkeypatch, m_redactor, modelo)
    return m_redactor.redactor(estado), modelo


def test_redactor_modo_entrega_es_breve_sin_fuentes_y_activa_el_documento(monkeypatch):
    estado = estado_investigador(pide_archivo(), perfil="P1", archivos=[ARCHIVO_TAXI])

    resultado, modelo = correr_redactor(monkeypatch, estado)

    sistema, entrada = modelo.entradas[0][0].content, modelo.entradas[0][1].content
    assert "MODO: ENTREGA DEL ARCHIVO" in sistema and prompts.ESTILO_POR_PERFIL["P1"] in sistema
    assert "ARCHIVO(S) PARA DESCARGAR" in entrada and "Decreto 001 de 2019" in entrada
    assert "url_descarga" not in entrada and "/documentos/" not in entrada  # el modelo nunca ve la URL
    assert resultado["fuentes"] == []
    assert resultado["documento_activo"]["archivo"] == ARCHIVO_TAXI["archivo"]
    assert resultado["documento_activo"]["titulo"] == "Decreto 001 de 2019"


def test_redactor_modo_descarga_no_disponible_explica_y_no_cambia_a_otro_documento(monkeypatch):
    estado = estado_investigador(pide_archivo(), archivos_no_disponibles=[{"titulo": "Decreto 001 de 2019", "archivo": "22598.pdf"}])

    resultado, modelo = correr_redactor(monkeypatch, estado, "Ese documento aún no está disponible para descargar.")

    assert "MODO: DESCARGA NO DISPONIBLE" in modelo.entradas[0][0].content
    assert "ORIGINAL NO DISPONIBLE PARA DESCARGA: Decreto 001 de 2019" in modelo.entradas[0][1].content
    assert resultado["fuentes"] == []


def test_redactor_con_aclaracion_y_archivos_pide_aclarar(monkeypatch):
    estado = estado_investigador(pide_archivo(), candidatos=CANDIDATOS, archivos=[ARCHIVO_TAXI])

    _, modelo = correr_redactor(monkeypatch, estado, "¿Cuál de estas resoluciones?")

    assert "MODO: ACLARACIÓN" in modelo.entradas[0][0].content


def test_redactor_pedido_mixto_responde_con_citas_y_menciona_el_archivo(monkeypatch):
    estado = estado_investigador(
        pide_archivo(contenido=True), archivos=[ARCHIVO_TAXI], evidencia=[FRAGMENTO_TAXI], fuentes=[FUENTE_TAXI]
    )

    resultado, modelo = correr_redactor(monkeypatch, estado, "La tarifa mínima es $5.500 [Decreto 001 de 2019, p. 1].")

    assert "MODO: RESPUESTA" in modelo.entradas[0][0].content
    assert "ARCHIVO(S) PARA DESCARGAR" in modelo.entradas[0][1].content
    assert resultado["fuentes"] == [FUENTE_TAXI]
    assert resultado["documento_activo"]["archivo"] == FUENTE_TAXI["archivo"]


@pytest.mark.parametrize("modo", sorted(prompts.INSTRUCCIONES_MODO_REDACTOR))
def test_el_redactor_nunca_escribe_urls_en_ningun_modo(modo):
    sistema = prompts.construir_prompt_redactor("P0", modo)

    assert "NUNCA escribas URLs" in sistema


# ══════════════════════════════════════════════════════════════
# grafo completo
# ══════════════════════════════════════════════════════════════

def test_pasame_el_decreto_entrega_el_archivo_sin_evidencia_en_modo_entrega(monkeypatch, agente):
    fakes = instalar_grafo(
        monkeypatch, [pide_archivo()],
        investigador_msgs=[LLAMADA_REFERENCIA, LLAMADA_DESCARGA, AIMessage(content="ok")],
        redactor_msgs=[AIMessage(content="Aquí tienes el Decreto 001 de 2019.")],
        salidas_tools={"buscar_documento_por_referencia": SALIDA_REFERENCIA}, salidas_descarga={"preparar_descarga": ARCHIVO_TAXI_SALIDA},
    )

    resultado = turno(agente, "pásame el Decreto 001 de 2019")

    assert resultado["archivos"] == [ARCHIVO_TAXI]
    assert resultado["evidencia"] == [] and resultado["fuentes"] == []
    assert "MODO: ENTREGA DEL ARCHIVO" in sistema_del_redactor(fakes)
    assert fakes["evaluador"].llamadas == 0
    assert resultado["documento_activo"]["archivo"] == ARCHIVO_TAXI["archivo"]
    assert resultado["messages"][-1].text == "Aquí tienes el Decreto 001 de 2019."


def test_que_dice_el_decreto_no_entrega_archivos_ni_ofrece_la_tool(monkeypatch, agente):
    fakes = instalar_grafo(
        monkeypatch, [pide_contenido()],
        investigador_msgs=[LLAMADA_REFERENCIA, AIMessage(content="ok")],
        redactor_msgs=[AIMessage(content="Fija las tarifas [Decreto 001 de 2019, p. 1].")],
        salidas_tools={"buscar_documento_por_referencia": SALIDA_REFERENCIA},
    )

    resultado = turno(agente, "¿qué dice el Decreto 001 de 2019?")

    assert resultado["archivos"] == []
    assert "preparar_descarga" not in fakes["investigador"].herramientas_vinculadas
    assert "MODO: RESPUESTA" in sistema_del_redactor(fakes)


def test_pregunta_sobre_un_decreto_y_luego_descargamelo_entrega_el_documento_activo(monkeypatch, agente):
    seguimiento = AnalisisConsulta(
        intencion="seguimiento", consulta_autonoma="Descargar el archivo del Decreto 001 de 2019", usa_documento_activo=True,
        solicita_archivo=True, incluye_contenido=False, razonamiento="x",
    )
    fakes = instalar_grafo(
        monkeypatch, [pide_contenido(), seguimiento],
        investigador_msgs=[
            llamada("buscar_fragmentos", consulta="decreto 001"), AIMessage(content="ok"),
            LLAMADA_DESCARGA, AIMessage(content="ok"),
        ],
        redactor_msgs=[
            AIMessage(content="La tarifa mínima es $5.500 [Decreto 001 de 2019, p. 1]."),
            AIMessage(content="Aquí tienes el Decreto 001 de 2019."),
        ],
        salidas_tools={"buscar_fragmentos": SALIDA_REFERENCIA}, salidas_descarga={"preparar_descarga": ARCHIVO_TAXI_SALIDA},
    )

    primero = turno(agente, "¿qué dice el decreto de taxis?")
    segundo = turno(agente, "descárgamelo")

    assert primero["archivos"] == [] and primero["documento_activo"]["archivo"] == ARCHIVO_TAXI["archivo"]
    assert segundo["archivos"] == [ARCHIVO_TAXI]
    encargos = [m.content for entrada in fakes["investigador"].entradas for m in entrada if m.type == "human"]
    encargo = next(e for e in encargos if "Descargar el archivo" in e)
    assert FUENTE_TAXI["archivo"] in encargo and "documento activo" in encargo
    assert "preparar_descarga" in fakes["investigador"].herramientas_vinculadas  # lo recibió en el segundo turno
    assert "MODO: ENTREGA DEL ARCHIVO" in fakes["redactor"].entradas[-1][0].content


def test_referencia_ambigua_con_solicita_archivo_pide_aclaracion_sin_archivos(monkeypatch, agente):
    fakes = instalar_grafo(
        monkeypatch, [pide_archivo("Descargar la Resolución 4397")],
        investigador_msgs=[llamada("buscar_documento_por_referencia", tipo="resolucion", numero=4397), AIMessage(content="ok")],
        redactor_msgs=[AIMessage(content="¿Cuál de las dos resoluciones?")],
        salidas_tools={"buscar_documento_por_referencia": {"fuentes": [], "fragmentos": [], "candidatos": CANDIDATOS}},
    )

    resultado = turno(agente, "descárgame la resolución 4397")

    assert resultado["archivos"] == [] and resultado["candidatos"] == CANDIDATOS
    assert "MODO: ACLARACIÓN" in sistema_del_redactor(fakes)


def test_documento_indexado_sin_fila_se_explica_como_no_disponible(monkeypatch, agente):
    fakes = instalar_grafo(
        monkeypatch, [pide_archivo()],
        investigador_msgs=[LLAMADA_REFERENCIA, LLAMADA_DESCARGA, AIMessage(content="ok")],
        redactor_msgs=[AIMessage(content="Ese documento aún no está disponible para descargar.")],
        salidas_tools={"buscar_documento_por_referencia": SALIDA_REFERENCIA}, salidas_descarga={"preparar_descarga": NO_DISPONIBLE_SALIDA},
    )

    resultado = turno(agente, "pásame el PDF del Decreto 001 de 2019")

    assert resultado["archivos"] == []
    assert resultado["archivos_no_disponibles"][0]["archivo"] == ARCHIVO_TAXI["archivo"]
    assert "MODO: DESCARGA NO DISPONIBLE" in sistema_del_redactor(fakes)


def test_archivos_no_se_acumulan_entre_turnos(monkeypatch, agente):
    instalar_grafo(
        monkeypatch, [pide_archivo(), pide_contenido()],
        investigador_msgs=[
            LLAMADA_REFERENCIA, LLAMADA_DESCARGA, AIMessage(content="ok"),
            llamada("buscar_fragmentos", consulta="tarifa"), AIMessage(content="ok"),
        ],
        redactor_msgs=[AIMessage(content="Aquí tienes el documento."), AIMessage(content="La tarifa es $5.500 [Decreto 001 de 2019, p. 1].")],
        salidas_tools={"buscar_documento_por_referencia": SALIDA_REFERENCIA, "buscar_fragmentos": SALIDA_REFERENCIA},
        salidas_descarga={"preparar_descarga": ARCHIVO_TAXI_SALIDA},
    )

    primero = turno(agente, "pásame el PDF del Decreto 001 de 2019")
    segundo = turno(agente, "¿cuánto cuesta el taxi?")

    assert len(primero["archivos"]) == 1
    assert segundo["archivos"] == [] and segundo["archivos_no_disponibles"] == []


# ══════════════════════════════════════════════════════════════
# API: contrato y compuerta
# ══════════════════════════════════════════════════════════════

def test_la_compuerta_descarta_archivos_si_el_analista_no_los_pidio():
    estado = {"analisis": {"solicita_archivo": False}, "archivos": [ARCHIVO_TAXI]}

    assert api._archivos(estado) == []
    assert api._archivos({"archivos": [ARCHIVO_TAXI]}) == []  # sin análisis tampoco
    assert api._archivos({"analisis": {"solicita_archivo": True}, "archivos": [ARCHIVO_TAXI]})[0].archivo == ARCHIVO_TAXI["archivo"]


def test_archivo_descargable_solo_expone_titulo_archivo_y_url():
    [archivo] = api._archivos({"analisis": {"solicita_archivo": True}, "archivos": [ARCHIVO_TAXI]})

    assert set(archivo.model_dump()) == {"titulo", "archivo", "url_descarga"}
    assert archivo.url_descarga == "/documentos/22598_decreto-no-0012019.pdf/descargar"


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINTER", "memory")
    monkeypatch.setattr(config, "RERANKER_PRECARGAR", False)
    monkeypatch.setattr(api.db, "init_db", lambda: None)
    monkeypatch.setattr(api.db, "guardar_turno", lambda *a, **k: None)
    with TestClient(api.app) as cliente:
        yield cliente


def test_chat_devuelve_archivos_cuando_se_pidieron(monkeypatch, cliente):
    instalar_grafo(
        monkeypatch, [pide_archivo()],
        investigador_msgs=[LLAMADA_REFERENCIA, LLAMADA_DESCARGA, AIMessage(content="ok")],
        redactor_msgs=[AIMessage(content="Aquí tienes el Decreto 001 de 2019.")],
        salidas_tools={"buscar_documento_por_referencia": SALIDA_REFERENCIA}, salidas_descarga={"preparar_descarga": ARCHIVO_TAXI_SALIDA},
    )

    cuerpo = cliente.post("/chat", json={"mensaje": "descárgame el decreto 1 de 2019", "thread_id": "a1"}).json()

    assert cuerpo["archivos"] == [{
        "titulo": "Decreto 001 de 2019", "archivo": "22598_decreto-no-0012019.pdf",
        "url_descarga": "/documentos/22598_decreto-no-0012019.pdf/descargar",
    }]
    assert "http" not in cuerpo["respuesta"] and "/documentos/" not in cuerpo["respuesta"]
    assert cuerpo["fuentes"] == []


def test_chat_sin_pedido_de_archivo_devuelve_archivos_vacio(monkeypatch, cliente):
    instalar_grafo(
        monkeypatch, [pide_contenido()],
        investigador_msgs=[LLAMADA_REFERENCIA, AIMessage(content="ok")],
        redactor_msgs=[AIMessage(content="Fija las tarifas [Decreto 001 de 2019, p. 1].")],
        salidas_tools={"buscar_documento_por_referencia": SALIDA_REFERENCIA},
    )

    cuerpo = cliente.post("/chat", json={"mensaje": "¿qué dice el decreto 1 de 2019?", "thread_id": "a2"}).json()

    assert cuerpo["archivos"] == []


@pytest.mark.parametrize("modo", sorted(set(prompts.INSTRUCCIONES_MODO_REDACTOR) - {"respuesta"}))
def test_los_modos_sin_salida_estructurada_no_mencionan_informacion_encontrada(modo):
    # En evaluación real el modelo copiaba "informacion_encontrada: false" al texto de la respuesta.
    assert "informacion_encontrada" not in prompts.construir_prompt_redactor("P0", modo)


def test_el_modo_respuesta_si_pide_la_senal_informacion_encontrada():
    assert "informacion_encontrada" in prompts.construir_prompt_redactor("P0", "respuesta")
