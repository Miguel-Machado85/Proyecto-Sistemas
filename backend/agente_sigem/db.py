"""Historial de conversación (texto + referencia al audio en S3), en PostgreSQL."""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
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