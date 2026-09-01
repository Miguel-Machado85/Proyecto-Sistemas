import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Ollama (chat, corriendo en tu Docker local) ─────────────────
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # TODO: pon aquí el nombre EXACTO del modelo que ya tienes descargado
    # (revísalo con `ollama list` en la terminal donde corre el contenedor).
    CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3:latest")

    # ── Gemini (solo para embeddings) ──
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

    # ── Chroma ────────
    # Chroma corre embebido (una carpeta local persistida), se usa
    # CHROMA_PERSIST_DIR. 
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sigem_documentos")

    DOCS_DIR = Path(os.getenv("DOCS_DIR", Path(__file__).resolve().parent.parent / "docs"))


config = Config()