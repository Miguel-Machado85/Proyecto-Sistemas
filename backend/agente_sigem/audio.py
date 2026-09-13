"""Adaptadores de Google Gemini para audio (STT y TTS)."""
import io
import wave

from google import genai
from google.genai import types

from .config import config


def _client() -> genai.Client:
    if not config.GOOGLE_API_KEY:
        raise ValueError("Falta configurar GOOGLE_API_KEY en el archivo .env.")
    return genai.Client(api_key=config.GOOGLE_API_KEY)


def transcribe(data: bytes, content_type: str) -> str:
    """Envía el audio directamente a Gemini y devuelve la transcripción."""
    audio_part = types.Part.from_bytes(data=data, mime_type=content_type)
    respuesta = _client().models.generate_content(
        model=config.AUDIO_MODEL,
        contents=["Transcribe este audio literalmente, solo el texto, sin comentarios adicionales.", audio_part],
    )
    return (respuesta.text or "").strip()


def synthesize(text: str) -> bytes:
    """Genera voz a partir de texto y devuelve un WAV listo para reproducir."""
    respuesta = _client().models.generate_content(
        model=config.TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=config.TTS_VOICE)
                )
            ),
        ),
    )
    pcm = respuesta.candidates[0].content.parts[0].inline_data.data

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)
    return buffer.getvalue()