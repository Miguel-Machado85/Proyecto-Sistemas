"""Checkpointer del grafo: Postgres en producción, memoria para pruebas y desarrollo local."""

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from .config import config

TIMEOUT_APERTURA_POOL_SEGUNDOS = 30


def crear_checkpointer() -> tuple[BaseCheckpointSaver, Callable[[], None]]:
    """Crea el checkpointer según CHECKPOINTER ("postgres" | "memory").

    Devuelve (checkpointer, cerrar). `cerrar` libera el pool de conexiones al apagar la API.
    No se llama al importar el módulo: se invoca en el lifespan de FastAPI.
    """
    modo = config.CHECKPOINTER
    if modo == "memory":
        return MemorySaver(), lambda: None
    if modo != "postgres":
        raise ValueError(f"CHECKPOINTER debe ser 'postgres' o 'memory', no '{modo}'.")

    # Importaciones perezosas: con CHECKPOINTER=memory no se necesita ni psycopg_pool ni Postgres.
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=config.DATABASE_URL,
        max_size=10,
        # autocommit y prepare_threshold=0 son requisitos de PostgresSaver; dict_row, de sus consultas.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    try:
        pool.open(wait=True, timeout=TIMEOUT_APERTURA_POOL_SEGUNDOS)
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()  # crea las tablas del checkpointer si no existen
    except Exception:
        pool.close()
        raise
    return checkpointer, pool.close
