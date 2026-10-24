"""Historial de conversación (texto + referencia a audios), persistido en PostgreSQL.

El documento activo ya no se guarda aquí: vive en el estado del grafo (checkpointer).
La tabla `contextos_sigem` de bases existentes se deja intacta; create_all nunca borra tablas.
"""
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import config

engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Turno(Base):
    """Un mensaje individual (de usuario o del agente) dentro de un hilo."""

    __tablename__ = "turnos"

    id = Column(Integer, primary_key=True)
    thread_id = Column(String, index=True, nullable=False)
    rol = Column(String, nullable=False)  # 'usuario' | 'agente'
    texto = Column(Text, nullable=False)
    audio_key = Column(String, nullable=True)  # key en S3, no URL (el bucket es privado)
    creado_en = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ArchivoDocumento(Base):
    """Enlace entre un documento indexado y su original en el repositorio de archivos (Drive o local).

    La clave es la misma `nombre_archivo` de los metadatos de Chroma; así no se modifica la colección.
    """

    __tablename__ = "archivos_documento"

    nombre_archivo = Column(String, primary_key=True)
    titulo = Column(String, nullable=True)
    proveedor = Column(String, nullable=False)  # 'drive' | 'local'
    id_externo = Column(String, nullable=False)
    mime_type = Column(String, nullable=True)
    tamano_bytes = Column(BigInteger, nullable=True)
    hash_sha256 = Column(String, nullable=True)
    creado_en = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    """Crea la tabla si no existe. Se llama al arrancar la API."""
    Base.metadata.create_all(bind=engine)


def guardar_turno(thread_id: str, rol: str, texto: str, audio_key: str | None = None) -> Turno:
    with SessionLocal() as session:
        turno = Turno(thread_id=thread_id, rol=rol, texto=texto, audio_key=audio_key)
        session.add(turno)
        session.commit()
        session.refresh(turno)
        return turno


def obtener_historial(thread_id: str) -> list[Turno]:
    with SessionLocal() as session:
        return (
            session.query(Turno)
            .filter(Turno.thread_id == thread_id)
            .order_by(Turno.creado_en.asc())
            .all()
        )


def registrar_archivo(
    nombre_archivo: str,
    proveedor: str,
    id_externo: str,
    titulo: str | None = None,
    mime_type: str | None = None,
    tamano_bytes: int | None = None,
    hash_sha256: str | None = None,
) -> ArchivoDocumento:
    """Inserta o actualiza el enlace del documento con su original."""
    with SessionLocal() as session:
        fila = session.merge(ArchivoDocumento(
            nombre_archivo=nombre_archivo, titulo=titulo, proveedor=proveedor, id_externo=id_externo,
            mime_type=mime_type, tamano_bytes=tamano_bytes, hash_sha256=hash_sha256,
        ))
        session.commit()
        session.refresh(fila)
        session.expunge(fila)
        return fila


def obtener_archivo(nombre_archivo: str) -> ArchivoDocumento | None:
    with SessionLocal() as session:
        fila = session.get(ArchivoDocumento, nombre_archivo)
        if fila is not None:
            session.expunge(fila)
        return fila


def eliminar_registro_archivo(nombre_archivo: str) -> bool:
    """Borra solo la fila (el archivo en el repositorio se retira aparte). Devuelve si existía."""
    with SessionLocal() as session:
        fila = session.get(ArchivoDocumento, nombre_archivo)
        if fila is None:
            return False
        session.delete(fila)
        session.commit()
        return True
