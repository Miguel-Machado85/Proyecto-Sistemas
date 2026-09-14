"""Adaptadores de Google Gemini para audio (STT y TTS), vía la Interactions API."""
import base64
import io
import wave

from google import genai

from .config import config

# IMPORTANTE: el cliente se crea UNA SOLA VEZ y se reutiliza.
# Crearlo "al vuelo" en cada llamada hace que el httpx.Client interno se
# cierre antes de que la petición termine de enviarse:
#   RuntimeError: Cannot send a request, as the client has been closed.
# Ver: https://github.com/googleapis/python-genai/issues/1763
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GOOGLE_API_KEY:
            raise ValueError("Falta configurar GOOGLE_API_KEY en el archivo .env.")
        _client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _client


def transcribe(data: bytes, content_type: str) -> str:
    """Transcribe audio usando el modelo dedicado de STT (gemini-3.5-transcribe)."""
    # El navegador a veces manda "audio/webm;codecs=opus"; nos quedamos
    # solo con el mime_type base.
    mime_type = (content_type or "audio/webm").split(";")[0].strip()

    interaction = _get_client().interactions.create(
        model=config.AUDIO_MODEL,
        input=[
            {
                "type": "audio",
                "data": base64.b64encode(data).decode("utf-8"),
                "mime_type": mime_type,
            }
        ],
    )
    return (interaction.output_text or "").strip()


def synthesize(text: str) -> bytes:
    """Genera voz a partir de texto (gemini-3.1-flash-tts-preview) y devuelve un WAV."""
    interaction = _get_client().interactions.create(
        model=config.TTS_MODEL,
        input=text,
        response_format={"type": "audio"},
        generation_config={
            "speech_config": [{"voice": config.TTS_VOICE}],
        },
    )
    pcm = base64.b64decode(interaction.output_audio.data)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)
    return buffer.getvalue()