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

    MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", 15 * 1024 * 1024))
    ALLOWED_AUDIO_EXTENSIONS = {"webm", "wav", "mp3", "m4a", "mp4", "ogg"}

    # ── Chroma ────────
    # local: carpeta persistida en CHROMA_PERSIST_DIR (tests, desarrollo y origen de la migración).
    # http: servidor Chroma aparte (contenedor `chroma` en Compose) en CHROMA_HOST:CHROMA_PORT.
    CHROMA_MODO = os.getenv("CHROMA_MODO", "local").strip().lower()
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_SSL = os.getenv("CHROMA_SSL", "false").strip().lower() in {"1", "true", "yes", "si", "sí"}
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # La colección existente se llama `langchain`; usar otro valor crea un índice vacío.
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "langchain")

    # ── Ingesta administrativa ──
    UPLOAD_ROOT = Path(os.getenv("UPLOAD_ROOT", Path(__file__).resolve().parent.parent / "documentos"))
    ALLOWED_DOCUMENT_EXTENSIONS = {"pdf", "png"}
    MAX_DOCUMENT_BYTES = int(os.getenv("MAX_DOCUMENT_BYTES", 25 * 1024 * 1024))
    MAX_DOCUMENTS_PER_BATCH = int(os.getenv("MAX_DOCUMENTS_PER_BATCH", "20"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "25"))
    EMBEDDING_BATCH_PAUSE_SECONDS = float(os.getenv("EMBEDDING_BATCH_PAUSE_SECONDS", "20"))
    ADMIN_UPLOAD_TOKEN = os.getenv("ADMIN_UPLOAD_TOKEN")

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
