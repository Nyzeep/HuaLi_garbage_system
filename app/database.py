from __future__ import annotations

from collections.abc import Generator
import sqlite3

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


SQLITE_BUSY_TIMEOUT_MS = 5000

settings = get_settings()


class Base(DeclarativeBase):
    pass


def _is_sqlite_url(database_url: str) -> bool:
    return database_url.startswith("sqlite")


def _build_connect_args(database_url: str) -> dict[str, object]:
    if not _is_sqlite_url(database_url):
        return {}
    return {
        "check_same_thread": False,
        "timeout": SQLITE_BUSY_TIMEOUT_MS / 1000,
    }


def _configure_sqlite_engine(database_engine: Engine) -> None:
    @event.listens_for(database_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        if not isinstance(dbapi_connection, sqlite3.Connection):
            return
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        cursor.close()


def create_database_engine(database_url: str) -> Engine:
    database_engine = create_engine(
        database_url,
        future=True,
        connect_args=_build_connect_args(database_url),
    )
    if _is_sqlite_url(database_url):
        _configure_sqlite_engine(database_engine)
    return database_engine


engine = create_database_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

