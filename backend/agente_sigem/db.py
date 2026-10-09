"""Historial y contexto documental de conversación, persistidos en PostgreSQL."""
from datetime import datetime, timezone
import json

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


class ContextoSigem(Base):
    """La última fuente SIGEM que permite resolver mensajes de seguimiento."""

    __tablename__ = "contextos_sigem"

    thread_id = Column(String, primary_key=True)
    fuentes_json = Column(Text, nullable=False)
    actualizado_en = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


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


def guardar_contexto_sigem(thread_id: str, fuentes: list[dict]) -> None:
    """Reemplaza el contexto activo solo después de una respuesta respaldada por SIGEM."""
    with SessionLocal() as session:
        contexto = session.get(ContextoSigem, thread_id)
        fuentes_json = json.dumps(fuentes, ensure_ascii=False)
        if contexto is None:
            session.add(ContextoSigem(thread_id=thread_id, fuentes_json=fuentes_json))
        else:
            contexto.fuentes_json = fuentes_json
            contexto.actualizado_en = datetime.now(timezone.utc)
        session.commit()


def obtener_contexto_sigem(thread_id: str) -> list[dict]:
    with SessionLocal() as session:
        contexto = session.get(ContextoSigem, thread_id)
        if contexto is None:
            return []
        try:
            return json.loads(contexto.fuentes_json)
        except json.JSONDecodeError:
            return []
