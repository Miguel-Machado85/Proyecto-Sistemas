"""US-09: verifica si un audio transcrito es comprensible antes de mandarlo al agente.

Corre como un paso independiente, después de transcribir y antes de invocar al
agente RAG: así evitamos gastar una llamada completa al LLM (y al TTS) cuando
el audio llegó en blanco, con ruido, o accidentalmente sin voz clara.
"""
from .audio import _get_client
from .config import config

_PROMPT_VERIFICACION = """Eres un verificador de calidad de audio transcrito para un asistente ciudadano.

Te voy a dar un texto que salió de transcribir un audio grabado por un usuario.
Decide si ese texto es un mensaje o pregunta comprensible en español, o si más
bien parece ruido de fondo, silencio, palabras sueltas sin sentido, o una
transcripción fallida/accidental.

Responde ÚNICAMENTE con una palabra: CLARO o POCO_CLARO. Nada más, sin explicación.

Texto transcrito: "{texto}"
"""


def es_audio_comprensible(texto: str) -> bool:
    """True si el texto transcrito parece un mensaje real; False si parece ruido/silencio."""
    texto = (texto or "").strip()
    if len(texto) < 3:
        return False

    try:
        interaction = _get_client().interactions.create(
            model=config.VERIFICACION_MODEL,
            input=_PROMPT_VERIFICACION.format(texto=texto),
        )
        veredicto = (interaction.output_text or "").strip().upper()
        return veredicto.startswith("CLARO")
    except Exception:
        # Si el verificador mismo falla (ej. cuota, red), no bloqueamos al
        # usuario por un problema nuestro: dejamos pasar el mensaje.
        return True