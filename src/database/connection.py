"""Lazy, health-checked SQLAlchemy connection setup."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker


DEFAULT_DATABASE_URL = "postgresql+psycopg://environment:environment@127.0.0.1:5432/environment"


def environment_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_database_url() -> str:
    explicit = os.getenv("DATABASE_URL", "").strip()
    if explicit:
        return explicit
    host = os.getenv("DATABASE_HOST", "").strip()
    if host:
        return URL.create(
            drivername="postgresql+psycopg",
            username=os.getenv("DATABASE_USER") or os.getenv("POSTGRES_USER") or "environment",
            password=os.getenv("DATABASE_PASSWORD") or os.getenv("POSTGRES_PASSWORD") or "environment",
            host=host,
            port=int(os.getenv("DATABASE_PORT", "5432")),
            database=os.getenv("DATABASE_NAME") or os.getenv("POSTGRES_DB") or "environment",
        ).render_as_string(hide_password=False)
    return DEFAULT_DATABASE_URL


@lru_cache(maxsize=4)
def get_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_database_url()
    options: dict = {
        "pool_pre_ping": True,
        "pool_recycle": int(os.getenv("DATABASE_POOL_RECYCLE_SECONDS", "1800")),
    }
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    elif url.startswith("postgresql"):
        options["connect_args"] = {
            "connect_timeout": int(os.getenv("DATABASE_CONNECT_TIMEOUT_SECONDS", "5"))
        }
        options["pool_size"] = int(os.getenv("DATABASE_POOL_SIZE", "5"))
        options["max_overflow"] = int(os.getenv("DATABASE_MAX_OVERFLOW", "5"))
    return create_engine(url, **options)


@lru_cache(maxsize=4)
def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(database_url),
        autoflush=False,
        expire_on_commit=False,
    )


@contextmanager
def session_scope(database_url: str | None = None) -> Iterator[Session]:
    session = get_session_factory(database_url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping_database(database_url: str | None = None) -> bool:
    with get_engine(database_url).connect() as connection:
        connection.execute(text("SELECT 1"))
    return True


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding one transactional session."""

    with session_scope() as session:
        yield session
