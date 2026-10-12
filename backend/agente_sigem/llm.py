"""Fábrica de modelos de chat de Gemini para los agentes del grafo."""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import config


@lru_cache(maxsize=None)
def crear_llm(nombre_modelo: str, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Devuelve un ChatGoogleGenerativeAI; se reutiliza por (modelo, temperatura).

    Los reintentos con backoff (LLM_MAX_RETRIES) protegen la cuota del plan gratuito.
    """
    return ChatGoogleGenerativeAI(
        model=nombre_modelo,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=temperature,
        max_retries=config.LLM_MAX_RETRIES,
    )


def es_error_de_cuota(error: BaseException | None) -> bool:
    """True si el error (o su causa) es un límite de uso de Gemini (HTTP 429 / RESOURCE_EXHAUSTED)."""
    visitados = 0
    while error is not None and visitados < 5:
        texto = f"{type(error).__name__} {error}"
        codigo = getattr(error, "code", None) or getattr(error, "status_code", None)
        if codigo == 429 or "RESOURCE_EXHAUSTED" in texto or "TooManyRequests" in texto:
            return True
        error = error.__cause__ or error.__context__
        visitados += 1
    return False
