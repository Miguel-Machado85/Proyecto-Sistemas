"""Guarda los audios de preguntas y respuestas en S3.

El bucket se asume PRIVADO: no guardamos URLs públicas, guardamos la
'key' del objeto en la base de datos y generamos una URL firmada
(con expiración) cada vez que el frontend necesita reproducir el audio.
Así el audio nunca queda expuesto de forma permanente.
"""
import uuid

import boto3
from botocore.config import Config as BotoConfig

from .config import config

_s3 = boto3.client(
    "s3",
    region_name=config.AWS_REGION,
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    config=BotoConfig(signature_version="s3v4"),
)


def subir_audio(data: bytes, content_type: str, extension: str, thread_id: str) -> str:
    """Sube el audio al bucket y devuelve la key del objeto (no la URL)."""
    key = f"audios/{thread_id}/{uuid.uuid4()}.{extension}"
    _s3.put_object(
        Bucket=config.S3_BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    return key


def url_firmada(key: str | None, expira_segundos: int | None = None) -> str | None:
    """Genera una URL temporal para reproducir un audio guardado en S3."""
    if not key:
        return None
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": config.S3_BUCKET_NAME, "Key": key},
        ExpiresIn=expira_segundos or config.PRESIGNED_URL_EXPIRATION,
    )