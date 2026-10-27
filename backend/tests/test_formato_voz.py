"""Formato de las respuestas: limpieza para el TTS y regla de formato en el prompt del redactor."""

import pytest

from agente_sigem import prompts
from agente_sigem.audio import limpiar_formato_para_voz


def test_quita_negrilla_y_conserva_el_texto():
    assert limpiar_formato_para_voz("Vence el **15 de marzo** y cuesta __$5.500__.") == "Vence el 15 de marzo y cuesta $5.500."


def test_convierte_vinetas_en_texto_plano_conservando_los_saltos():
    texto = "Requisitos:\n- Cédula\n* Recibo\n  • Foto\nFin."
    assert limpiar_formato_para_voz(texto) == "Requisitos:\nCédula\nRecibo\n  Foto\nFin."


def test_quita_los_numerales_de_encabezado():
    assert limpiar_formato_para_voz("# Título\n## Sub\n### Otro\nTexto") == "Título\nSub\nOtro\nTexto"


def test_no_toca_citas_ni_numeracion_ni_guiones_en_medio():
    texto = "1. Pida el turno [Decreto 001 de 2019, p. 3].\n2. Pague - luego regrese a las 5 * 2."
    assert limpiar_formato_para_voz(texto) == texto


def test_texto_sin_formato_queda_igual():
    texto = "Hola, ¿en qué te ayudo?\n\nGracias."
    assert limpiar_formato_para_voz(texto) == texto


def test_negrilla_sin_pareja_se_deja_como_esta_y_no_rompe_el_resto():
    assert limpiar_formato_para_voz("Un ** suelto y **otro** bien") == "Un ** suelto y otro bien"
    assert limpiar_formato_para_voz("Texto **sin cerrar") == "Texto **sin cerrar"


@pytest.mark.parametrize("modo", list(prompts.INSTRUCCIONES_MODO_REDACTOR))
@pytest.mark.parametrize("perfil", ["P0", "P1", "P2", "P3"])
def test_la_regla_de_formato_esta_en_el_prompt_del_redactor(modo, perfil):
    sistema = prompts.construir_prompt_redactor(perfil, modo)

    assert "**negrilla**" in sistema
    assert 'listas con "- "' in sistema
    assert "No uses títulos con #" in sistema
