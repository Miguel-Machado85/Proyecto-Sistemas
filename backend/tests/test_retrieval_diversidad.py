from langchain_core.documents import Document

from agente_sigem.retrieval import combinar_y_diversificar, terminos_textuales


def documento(archivo: str, pagina: int, contenido: str) -> Document:
    return Document(
        page_content=contenido,
        metadata={"nombre_archivo": archivo, "page": pagina, "titulo": archivo, "descripcion": contenido},
    )


def test_descarta_terminos_genericos_para_busqueda_textual():
    assert terminos_textuales("Que documentos SIGEM hay sobre talento humano") == ["talento", "humano"]


def test_limita_fragmentos_por_documento_en_consulta_amplia():
    texto = [documento("principal.pdf", pagina, "talento humano") for pagina in range(5)]
    semanticos = [documento("secundario.pdf", 1, "talento humano"), documento("tercero.pdf", 1, "talento humano")]

    resultados = combinar_y_diversificar("talento humano", texto, semanticos)

    archivos = [resultado.metadata["nombre_archivo"] for resultado in resultados]
    assert archivos.count("principal.pdf") == 2
    assert "secundario.pdf" in archivos
    assert "tercero.pdf" in archivos
