"""Genera UNA vez el token OAuth con el que el backend accede a la carpeta de Google Drive.

El backend actúa como la cuenta de Google que autorice aquí. Esa cuenta debe tener acceso a la carpeta
por correo: como Lector alcanza para listar y descargar; para SUBIR documentos nuevos (ingesta) necesita
ser Editor. No hace falta cambiar el acceso general del enlace de la carpeta.

Antes de correrlo (una sola vez, en Google Cloud Console):
  1. Crear/usar un proyecto y habilitar la "Google Drive API".
  2. Pantalla de consentimiento de OAuth: tipo "Externo" y pasarla a "In production" (publicar la app).
     IMPORTANTE: si se deja en modo "Testing", el token de refresco CADUCA a los 7 días y hay que
     repetir este script. En producción aparece la advertencia "app no verificada": se acepta
     ("Avanzado" -> "Ir a ... (no seguro)") y listo.
  3. Credenciales -> "Crear credenciales" -> "ID de cliente de OAuth" -> tipo "Aplicación de escritorio".
     Descargar el JSON a backend/secrets/oauth-client.json (esa carpeta no se versiona).

Uso (desde backend/):
    uv run python scripts/autorizar_drive.py

Se abre el navegador, se inicia sesión con la cuenta elegida y se aceptan los permisos. El token queda en
backend/secrets/oauth-token.json (GOOGLE_OAUTH_TOKEN_FILE). Alcance: drive completo; el alcance reducido
`drive.file` no sirve porque solo ve archivos creados por la propia app y los originales los subió otra persona.
"""

import sys
from pathlib import Path

RAIZ_BACKEND = Path(__file__).resolve().parent.parent
if str(RAIZ_BACKEND) not in sys.path:
    sys.path.insert(0, str(RAIZ_BACKEND))


def main() -> int:
    from google_auth_oauthlib.flow import InstalledAppFlow

    from agente_sigem.config import config
    from agente_sigem.repositorio_archivos import ALCANCE_DRIVE

    cliente = config.GOOGLE_OAUTH_CLIENT_FILE
    token = config.GOOGLE_OAUTH_TOKEN_FILE
    if not cliente.is_file():
        print(f"No encuentro {cliente}. Descarga ahí el JSON del cliente OAuth tipo 'Aplicación de escritorio'.")
        return 1

    flujo = InstalledAppFlow.from_client_secrets_file(str(cliente), ALCANCE_DRIVE)
    # access_type=offline + prompt=consent garantizan que Google entregue un refresh_token.
    credenciales = flujo.run_local_server(port=0, access_type="offline", prompt="consent")
    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text(credenciales.to_json(), encoding="utf-8")
    print(f"Token guardado en {token}.")
    print("Comprueba el acceso con: uv run python scripts/sincronizar_drive.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
