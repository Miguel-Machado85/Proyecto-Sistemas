import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Gemini (embeddings y LLM de los agentes) ──
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

    # Modelos de chat. Los de clasificación usan flash-lite para cuidar la cuota del flash.
    # Nombres verificados contra models.list() de la API key del proyecto.
    ANALISTA_MODEL = os.getenv("ANALISTA_MODEL", "gemini-3.5-flash-lite")
    EVALUADOR_MODEL = os.getenv("EVALUADOR_MODEL", "gemini-3.5-flash-lite")
    INVESTIGADOR_MODEL = os.getenv("INVESTIGADOR_MODEL", "gemini-3.5-flash")
    REDACTOR_MODEL = os.getenv("REDACTOR_MODEL", "gemini-3.5-flash")
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # ── Grafo multi-agente ──
    MAX_INTENTOS_BUSQUEDA = int(os.getenv("MAX_INTENTOS_BUSQUEDA", "2"))
    INVESTIGADOR_MAX_PASOS = int(os.getenv("INVESTIGADOR_MAX_PASOS", "10"))
    MAX_HISTORIAL_ANALISTA = int(os.getenv("MAX_HISTORIAL_ANALISTA", "6"))
    EVALUADOR_HABILITADO = os.getenv("EVALUADOR_HABILITADO", "true").lower() == "true"
    CHECKPOINTER = os.getenv("CHECKPOINTER", "postgres").lower()  # postgres | memory

    # ── Reranker (sentence-transformers) ──
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    RERANKER_HABILITADO = os.getenv("RERANKER_HABILITADO", "true").lower() == "true"
    RERANKER_PRECARGAR = os.getenv("RERANKER_PRECARGAR", "false").lower() == "true"

    # ── Audio (Gemini STT/TTS) ──
    AUDIO_MODEL = os.getenv("AUDIO_MODEL", "gemini-3.5-transcribe")            # para transcribir
    TTS_MODEL = os.getenv("TTS_MODEL", "gemini-3.1-flash-tts-preview")    # para sintetizar
    TTS_VOICE = os.getenv("TTS_VOICE", "Kore")

    VERIFICACION_MODEL = os.getenv("VERIFICACION_MODEL", "gemini-3.5-flash-lite")

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

    # ── Langfuse (observabilidad/tracing del agente) ──
    # Si estas dos quedan vacías, el tracing simplemente se desactiva:
    # la app sigue funcionando igual, solo sin mandar traces.
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")


config = Config()
