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

    # ── Audio (Gemini STT/TTS) ──
    AUDIO_MODEL = os.getenv("AUDIO_MODEL", "gemini-2.5-flash")            # para transcribir
    TTS_MODEL = os.getenv("TTS_MODEL", "gemini-2.5-flash-preview-tts")    # para sintetizar
    TTS_VOICE = os.getenv("TTS_VOICE", "Kore")
    MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", 15 * 1024 * 1024))
    ALLOWED_AUDIO_EXTENSIONS = {"webm", "wav", "mp3", "m4a", "mp4", "ogg"}

    # ── Chroma ────────
    # Chroma corre embebido (una carpeta local persistida), se usa
    # CHROMA_PERSIST_DIR. 
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sigem_documentos")

    DOCS_DIR = Path(os.getenv("DOCS_DIR", Path(__file__).resolve().parent.parent / "docs"))

    # ── PostgreSQL (historial de conversaciones: texto + referencia a audios) ──
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sigem:sigem@db:5432/sigem")

    # ── AWS S3 (audios de preguntas y respuestas) ──
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
    # Las URLs firmadas expiran; así el bucket puede quedar privado sin problema.
    PRESIGNED_URL_EXPIRATION = int(os.getenv("PRESIGNED_URL_EXPIRATION", 3600))


config = Config()