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
