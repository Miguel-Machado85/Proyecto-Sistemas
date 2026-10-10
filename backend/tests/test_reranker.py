import pytest
from langchain_core.documents import Document

from agente_sigem import reranker
from agente_sigem.config import config
from agente_sigem.retrieval import diversificar


class CrossEncoderFalso:
    """Puntúa por la longitud del texto; cuenta las llamadas para verificar su uso."""

    def __init__(self):
        self.llamadas = 0

    def predict(self, pares):
        self.llamadas += 1
        return [len(contenido) for _, contenido in pares]


class CrossEncoderRoto:
    def predict(self, pares):
        raise RuntimeError("fallo simulado")


def doc(texto: str, archivo: str = "a.pdf", pagina: int = 1) -> Document:
    return Document(page_content=texto, metadata={"nombre_archivo": archivo, "page": pagina})


@pytest.fixture(autouse=True)
def estado_limpio(monkeypatch):
    """Cada test parte sin modelo cargado, con el reranker habilitado y sin avisos previos."""
    monkeypatch.setattr(reranker, "_modelo", None)
    monkeypatch.setattr(reranker, "_no_disponible", False)
    monkeypatch.setattr(reranker, "_avisos_emitidos", set())
    monkeypatch.setattr(config, "RERANKER_HABILITADO", True)


def test_ordena_por_puntaje_descendente(monkeypatch):
    falso = CrossEncoderFalso()
    monkeypatch.setattr(reranker, "_modelo", falso)
    corto, medio, largo = doc("a"), doc("aaa", pagina=2), doc("aaaaaa", pagina=3)

    resultado = reranker.reordenar("consulta", [corto, largo, medio])

    assert resultado == [largo, medio, corto]
    assert falso.llamadas == 1


def test_respeta_top_n(monkeypatch):
    monkeypatch.setattr(reranker, "_modelo", CrossEncoderFalso())
    documentos = [doc("a"), doc("aaa", pagina=2), doc("aaaaaa", pagina=3)]

    assert len(reranker.reordenar("consulta", documentos, top_n=2)) == 2


def test_no_se_llama_si_esta_deshabilitado(monkeypatch):
    falso = CrossEncoderFalso()
    monkeypatch.setattr(reranker, "_modelo", falso)
    monkeypatch.setattr(config, "RERANKER_HABILITADO", False)
    documentos = [doc("a"), doc("aaaaaa", pagina=2)]

    assert reranker.reordenar("consulta", documentos) is documentos
    assert reranker.reordenar_si_aplica("consulta", documentos) is None
    assert falso.llamadas == 0


def test_devuelve_lista_original_si_predict_falla_y_avisa_una_vez(monkeypatch, caplog):
    monkeypatch.setattr(reranker, "_modelo", CrossEncoderRoto())
    documentos = [doc("a"), doc("aaaaaa", pagina=2)]

    with caplog.at_level("WARNING", logger=reranker.logger.name):
        assert reranker.reordenar("consulta", documentos) == documentos
        assert reranker.reordenar("consulta", documentos) == documentos

    avisos = [r for r in caplog.records if "Falló el reranker" in r.getMessage()]
    assert len(avisos) == 1


def test_devuelve_lista_original_si_no_se_puede_cargar(monkeypatch):
    def cargar_falla():
        monkeypatch.setattr(reranker, "_no_disponible", True)
        return None

    monkeypatch.setattr(reranker, "_cargar_modelo", cargar_falla)
    documentos = [doc("a"), doc("aaaaaa", pagina=2)]

    assert reranker.reordenar("consulta", documentos) == documentos
    assert reranker.reordenar_si_aplica("consulta", documentos) is None


def test_precargar_respeta_el_interruptor(monkeypatch):
    monkeypatch.setattr(config, "RERANKER_HABILITADO", False)
    monkeypatch.setattr(reranker, "_cargar_modelo", lambda: pytest.fail("no debe cargar el modelo"))

    assert reranker.precargar() is False


def test_diversificar_sin_archivo_limita_por_documento_y_con_archivo_no():
    documentos = [doc("x", "principal.pdf", pagina) for pagina in range(5)] + [doc("y", "otro.pdf")]

    assert [d.metadata["nombre_archivo"] for d in diversificar(documentos)].count("principal.pdf") == 2
    assert len(diversificar(documentos, max_por_documento=None)) == 6
