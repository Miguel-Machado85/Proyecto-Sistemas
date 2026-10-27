"""Contrato HTTP de la API (sección 10): TestClient con CHECKPOINTER=memory y el grafo con LLM falsos."""

import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fakes_llm import CANDIDATOS, FRAGMENTO_TAXI, FUENTE_TAXI, llamada
from fastapi.testclient import TestClient
from grafo_falso import AIMessage, AnalisisConsulta, instalar_grafo

from agente_sigem import api, prompts
from agente_sigem.catalogo import CatalogoDocumentos
from agente_sigem.config import config

CLAVES_CHAT = {"respuesta", "thread_id", "pasos", "fuentes", "candidatos", "archivos", "tiempo_ms"}
CLAVES_VOZ = {
    "transcripcion", "respuesta", "thread_id", "pregunta_audio_url", "respuesta_audio_url",
    "fuentes", "candidatos", "archivos", "alerta",
}
CLAVES_PASO = {"agente", "accion", "icono"}
CLAVES_FUENTE = {"titulo", "archivo", "pagina", "fecha_expedicion", "descripcion"}
CLAVES_CANDIDATO = {"titulo", "archivo", "fecha_expedicion", "descripcion"}


class Entorno:
    """Todo lo que la API toca fuera del grafo, sustituido por dobles en memoria."""

    def __init__(self):
        self.turnos: list[tuple] = []
        self.audios_subidos: list[tuple] = []
        self.sintetizados: list[str] = []
        self.transcripcion = "¿cuánto cuesta el taxi?"


@pytest.fixture
def entorno(monkeypatch):
    entorno = Entorno()
    monkeypatch.setattr(config, "CHECKPOINTER", "memory")
    monkeypatch.setattr(config, "RERANKER_PRECARGAR", False)
    monkeypatch.setattr(api, "obtener_langfuse_handler", lambda: None)

    monkeypatch.setattr(api.db, "init_db", lambda: None)
    monkeypatch.setattr(
        api.db, "guardar_turno",
        lambda thread_id, rol, texto, audio_key=None: entorno.turnos.append((thread_id, rol, texto, audio_key)),
    )

    def subir_audio(data, content_type, extension, thread_id):
        entorno.audios_subidos.append((content_type, extension, thread_id))
        return f"audios/{thread_id}/{len(entorno.audios_subidos)}.{extension}"

    monkeypatch.setattr(api.storage, "subir_audio", subir_audio)
    monkeypatch.setattr(api.storage, "url_firmada", lambda key: f"https://s3.test/{key}" if key else None)
    monkeypatch.setattr(api, "transcribe", lambda data, content_type: entorno.transcripcion)

    def sintetizar(texto):
        entorno.sintetizados.append(texto)
        return b"RIFF-audio"

    monkeypatch.setattr(api, "synthesize", sintetizar)
    return entorno


@pytest.fixture
def cliente(entorno):
    with TestClient(api.app) as cliente:  # dispara el lifespan: checkpointer en memoria + grafo
        yield cliente


def consulta_taxi():
    return AnalisisConsulta(intencion="consulta_sigem", consulta_autonoma="tarifa del taxi", razonamiento="x")


def instalar_respuesta_con_fuentes(monkeypatch, candidatos=False):
    salidas = {"buscar_fragmentos": {"fuentes": [FUENTE_TAXI], "fragmentos": [FRAGMENTO_TAXI]}}
    if candidatos:
        salidas["buscar_documento_por_referencia"] = {"fuentes": [], "fragmentos": [], "candidatos": CANDIDATOS}
    instalar_grafo(
        monkeypatch,
        analisis=[consulta_taxi()],
        investigador_msgs=[
            llamada("buscar_fragmentos", consulta="tarifa del taxi"),
            llamada("buscar_documento_por_referencia", tipo="resolucion", numero=4397) if candidatos else AIMessage(content="ok"),
            AIMessage(content="Listo."),
        ],
        redactor_msgs=[AIMessage(content="La tarifa mínima es $5.500 [Decreto 001 de 2019, p. 1].")],
        salidas_tools=salidas,
    )


def audio_de_prueba(nombre="pregunta.webm"):
    return {"audio": (nombre, b"\x1aE\xdf\xa3audio", "audio/webm")}


# ── /chat ──

def test_chat_tiene_exactamente_la_forma_de_chat_response(monkeypatch, cliente, entorno):
    instalar_respuesta_con_fuentes(monkeypatch, candidatos=True)

    respuesta = cliente.post("/chat", json={"mensaje": "¿cuánto cuesta el taxi?", "thread_id": "t1", "perfil": "P2"})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert set(cuerpo) == CLAVES_CHAT
    assert cuerpo["respuesta"].startswith("La tarifa mínima")
    assert cuerpo["thread_id"] == "t1"
    assert isinstance(cuerpo["tiempo_ms"], float)
    assert [set(p) for p in cuerpo["pasos"]] == [CLAVES_PASO] * len(cuerpo["pasos"])
    assert {p["agente"] for p in cuerpo["pasos"]} >= {"analista", "investigador", "redactor"}
    assert cuerpo["fuentes"] and all(set(f) == CLAVES_FUENTE for f in cuerpo["fuentes"])
    assert cuerpo["fuentes"][0]["archivo"] == FUENTE_TAXI["archivo"]
    assert cuerpo["candidatos"] and all(set(c) == CLAVES_CANDIDATO for c in cuerpo["candidatos"])
    # Se persisten ambos lados del turno, sin audio.
    assert entorno.turnos == [
        ("t1", "usuario", "¿cuánto cuesta el taxi?", None),
        ("t1", "agente", cuerpo["respuesta"], None),
    ]


def test_chat_de_saludo_devuelve_el_texto_fijo_sin_fuentes(monkeypatch, cliente):
    instalar_grafo(monkeypatch, [AnalisisConsulta(intencion="saludo", consulta_autonoma="hola", razonamiento="x")])

    cuerpo = cliente.post("/chat", json={"mensaje": "hola"}).json()

    assert set(cuerpo) == CLAVES_CHAT
    assert cuerpo["respuesta"] == prompts.RESPUESTA_SALUDO
    assert cuerpo["thread_id"] == "default_user"  # valor por defecto del contrato
    assert cuerpo["fuentes"] == [] and cuerpo["candidatos"] == []


def test_chat_texto_no_comprensible_usa_el_mensaje_neutral(monkeypatch, cliente):
    instalar_grafo(monkeypatch, [AnalisisConsulta(
        intencion="no_comprensible", consulta_autonoma="asd;fj;rjfirjp", razonamiento="texto sin sentido")])

    cuerpo = cliente.post("/chat", json={"mensaje": "asd;fj;rjfirjp"}).json()

    assert cuerpo["respuesta"] == prompts.MENSAJE_TEXTO_POCO_CLARO
    assert cuerpo["respuesta"] != prompts.MENSAJE_AUDIO_POCO_CLARO
    assert cuerpo["fuentes"] == [] and cuerpo["candidatos"] == []


def test_chat_valida_el_mensaje_vacio(cliente):
    assert cliente.post("/chat", json={"mensaje": ""}).status_code == 422


def test_chat_con_error_interno_responde_500(monkeypatch, cliente):
    instalar_grafo(monkeypatch, [consulta_taxi()], investigador_msgs=[AIMessage(content="ok")], redactor_msgs=[])

    def redactor_roto(*args, **kwargs):
        raise RuntimeError("Gemini no disponible")

    monkeypatch.setattr(api, "ejecutar_turno", redactor_roto)

    respuesta = cliente.post("/chat", json={"mensaje": "hola"})

    assert respuesta.status_code == 500
    assert "Gemini no disponible" in respuesta.json()["detail"]


# ── /chat-voz ──

def test_chat_voz_tiene_exactamente_la_forma_de_chat_voz_response(monkeypatch, cliente, entorno):
    instalar_respuesta_con_fuentes(monkeypatch)

    respuesta = cliente.post("/chat-voz", files=audio_de_prueba(), data={"thread_id": "v1", "perfil": "P1"})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert set(cuerpo) == CLAVES_VOZ
    assert cuerpo["transcripcion"] == "¿cuánto cuesta el taxi?"
    assert cuerpo["alerta"] is False
    assert cuerpo["pregunta_audio_url"].startswith("https://s3.test/")
    assert cuerpo["respuesta_audio_url"].startswith("https://s3.test/")
    assert all(set(f) == CLAVES_FUENTE for f in cuerpo["fuentes"]) and cuerpo["fuentes"]
    assert entorno.sintetizados == [cuerpo["respuesta"]]
    assert [turno[1] for turno in entorno.turnos] == ["usuario", "agente"]
    assert all(turno[3] for turno in entorno.turnos)  # ambos con audio_key


def test_chat_voz_alerta_guarda_rol_alerta_y_no_sintetiza(monkeypatch, cliente, entorno):
    entorno.transcripcion = "eh mm ajá"
    instalar_grafo(
        monkeypatch,
        [AnalisisConsulta(intencion="no_comprensible", consulta_autonoma="eh mm", razonamiento="ruido")],
    )

    respuesta = cliente.post("/chat-voz", files=audio_de_prueba(), data={"thread_id": "v2"})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert set(cuerpo) == CLAVES_VOZ
    assert cuerpo["alerta"] is True
    assert cuerpo["respuesta"] == prompts.MENSAJE_AUDIO_POCO_CLARO
    assert cuerpo["respuesta_audio_url"] is None
    assert cuerpo["fuentes"] == [] and cuerpo["candidatos"] == []
    assert entorno.sintetizados == []  # sin TTS
    assert [turno[:2] for turno in entorno.turnos] == [("v2", "usuario"), ("v2", "alerta")]
    assert entorno.turnos[0][3] is not None  # el audio de la pregunta sí se conserva
    assert entorno.turnos[1][2] == prompts.MENSAJE_AUDIO_POCO_CLARO


def test_chat_voz_con_transcripcion_vacia_es_alerta(monkeypatch, cliente, entorno):
    entorno.transcripcion = ""
    instalar_grafo(monkeypatch, [consulta_taxi()])

    cuerpo = cliente.post("/chat-voz", files=audio_de_prueba()).json()

    assert cuerpo["alerta"] is True and cuerpo["transcripcion"] == ""
    assert entorno.turnos[0][2] == "[audio no comprensible]"


def test_transcripcion_vacia_no_invoca_el_grafo_ni_entra_al_historial(monkeypatch, cliente, entorno):
    entorno.transcripcion = "  "
    monkeypatch.setattr(api, "ejecutar_turno", lambda *a, **k: pytest.fail("no debe ejecutarse el grafo"))

    cuerpo = cliente.post("/chat-voz", files=audio_de_prueba(), data={"thread_id": "v3"}).json()

    assert cuerpo["alerta"] is True and cuerpo["respuesta"] == prompts.MENSAJE_AUDIO_POCO_CLARO
    assert [t[1] for t in entorno.turnos] == ["usuario", "alerta"]
    assert entorno.sintetizados == []


def test_chat_con_cuota_agotada_responde_un_mensaje_amable_sin_guardar_el_turno(monkeypatch, cliente, entorno):
    def sin_cuota(*args, **kwargs):
        raise RuntimeError("Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED): 429 You exceeded your quota")

    monkeypatch.setattr(api, "ejecutar_turno", sin_cuota)

    respuesta = cliente.post("/chat", json={"mensaje": "¿cuánto cuesta el taxi?", "thread_id": "q1"})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert set(cuerpo) == CLAVES_CHAT
    assert cuerpo["respuesta"] == prompts.MENSAJE_SERVICIO_SATURADO
    assert cuerpo["fuentes"] == [] and cuerpo["pasos"] == []
    assert "RESOURCE_EXHAUSTED" not in respuesta.text  # no se filtra el mensaje del proveedor
    assert entorno.turnos == []


def test_chat_voz_con_cuota_agotada_responde_con_alerta_y_sin_tts(monkeypatch, cliente, entorno):
    def sin_cuota(*args, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(api, "ejecutar_turno", sin_cuota)

    cuerpo = cliente.post("/chat-voz", files=audio_de_prueba(), data={"thread_id": "q2"}).json()

    assert set(cuerpo) == CLAVES_VOZ
    assert cuerpo["alerta"] is True and cuerpo["respuesta"] == prompts.MENSAJE_SERVICIO_SATURADO
    assert cuerpo["transcripcion"] == "¿cuánto cuesta el taxi?"
    assert entorno.sintetizados == [] and entorno.turnos == []


def test_es_error_de_cuota_reconoce_el_error_y_su_causa():
    from agente_sigem.llm import es_error_de_cuota

    assert es_error_de_cuota(RuntimeError("429 RESOURCE_EXHAUSTED. quota"))
    causa = RuntimeError("429 RESOURCE_EXHAUSTED")
    envuelto = ValueError("fallo en el modelo")
    envuelto.__cause__ = causa
    assert es_error_de_cuota(envuelto)
    assert not es_error_de_cuota(RuntimeError("503 Service Unavailable"))
    assert not es_error_de_cuota(None)


def test_chat_voz_rechaza_formatos_no_permitidos_y_audio_vacio(cliente):
    assert cliente.post("/chat-voz", files={"audio": ("nota.txt", b"x", "text/plain")}).status_code == 415
    assert cliente.post("/chat-voz", files={"audio": ("vacio.webm", b"", "audio/webm")}).status_code == 400


def test_chat_voz_el_fallo_del_turno_es_502(monkeypatch, cliente):
    monkeypatch.setattr(api, "ejecutar_turno", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    respuesta = cliente.post("/chat-voz", files=audio_de_prueba())

    assert respuesta.status_code == 502
    assert "No fue posible procesar el turno de voz" in respuesta.json()["detail"]


# ── resto del contrato ──

def test_historial_conserva_su_forma(monkeypatch, cliente):
    turnos = [
        SimpleNamespace(rol="usuario", texto="hola", audio_key="k1", creado_en=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        SimpleNamespace(rol="alerta", texto="No logré entender", audio_key=None, creado_en=datetime(2026, 1, 2, tzinfo=timezone.utc)),
    ]
    monkeypatch.setattr(api.db, "obtener_historial", lambda thread_id: turnos)

    cuerpo = cliente.get("/historial/h1").json()

    assert [set(t) for t in cuerpo] == [{"rol", "texto", "audio_url", "creado_en"}] * 2
    assert cuerpo[0]["audio_url"] == "https://s3.test/k1" and cuerpo[1]["audio_url"] is None


def test_sugerencias_de_documentos_conservan_su_forma(monkeypatch, cliente):
    catalogo = CatalogoDocumentos.desde_metadatas([{
        "nombre_archivo": "22598_decreto-no-0012019.pdf", "titulo": "Decreto 001 de 2019",
        "fecha_expedicion": "2019-01-03T08:00:00", "descripcion": "Tarifas de taxis",
    }])
    monkeypatch.setattr(api, "obtener_catalogo_documentos", lambda: catalogo)

    cuerpo = cliente.get("/documentos/sugerencias", params={"q": "decreto 1"}).json()

    assert cuerpo == [{
        "titulo": "Decreto 001 de 2019", "archivo": "22598_decreto-no-0012019.pdf",
        "fecha_expedicion": "2019", "descripcion": "Tarifas de taxis",
    }]


def test_la_precarga_del_reranker_no_bloquea_el_arranque(monkeypatch, entorno):
    liberar = threading.Event()
    llamadas = []

    def precargar_lento():
        llamadas.append(1)
        liberar.wait(10)
        return True

    monkeypatch.setattr(config, "RERANKER_PRECARGAR", True)
    monkeypatch.setattr(api.reranker, "precargar", precargar_lento)

    inicio = time.time()
    with TestClient(api.app) as cliente:
        assert cliente.get("/health").json() == {"status": "ok"}
        assert time.time() - inicio < 5  # no esperó a la precarga (que sigue bloqueada)
        liberar.set()


def test_health_responde_ok(cliente):
    assert cliente.get("/health").json() == {"status": "ok"}


def test_importar_la_api_no_conecta_a_postgres_ni_construye_el_grafo():
    """Importa en un proceso limpio con una base inalcanzable: no debe conectar ni compilar el grafo."""
    codigo = (
        "import agente_sigem.api as a;"
        "assert not hasattr(a.app.state, 'agente');"
        "print('import-ok')"
    )
    entorno = {**__import__("os").environ, "DATABASE_URL": "postgresql://x:x@127.0.0.1:1/x", "CHECKPOINTER": "postgres"}

    resultado = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, env=entorno, timeout=120)

    assert resultado.returncode == 0, resultado.stderr
    assert "import-ok" in resultado.stdout
