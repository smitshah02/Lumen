"""
Database Configuration
======================
Connection pooling and utilities for the Lumen Postgres instance.
"""

from __future__ import annotations

import os
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, Session

from src.config import DATA_PLANE

logger = logging.getLogger(__name__)

# Load from .env or environment
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:lumen@localhost:5434/lumen",
)

# Data-plane isolation. The synthetic demo plane gets its own database on the
# same server, so it can never resolve to the research (MIMIC) database.
# LUMEN_DEMO_DATABASE_URL overrides DATABASE_URL with only the database name
# changed by default.
RESEARCH_DB_NAME = make_url(DATABASE_URL).database
DEMO_DB_NAME = os.environ.get("LUMEN_DEMO_DB_NAME", "lumen_demo")
if DATA_PLANE == "demo":
    name = DEMO_DB_NAME
    DATABASE_URL = os.environ.get("LUMEN_DEMO_DATABASE_URL") or \
        make_url(DATABASE_URL).set(database=name).render_as_string(hide_password=False)
    actual = make_url(DATABASE_URL).database
    if actual != name or actual == RESEARCH_DB_NAME:
        raise RuntimeError(
            f"{DATA_PLANE} plane resolved to database {actual!r}, expected isolated {name!r}; refusing"
        )

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
    # Fail fast when Postgres is unreachable instead of waiting on TCP timeouts
    # (the API readiness probe depends on this).
    connect_args={"connect_timeout": int(os.environ.get("LUMEN_DB_CONNECT_TIMEOUT", "10"))},
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_session() -> Session:
    """Context manager for database sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def execute_sql(sql: str, params: dict = None):
    """Execute raw SQL against the database."""
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        conn.commit()
        return result


def check_connection() -> bool:
    """Verify the database is reachable."""
    try:
        execute_sql("SELECT 1")
        logger.info("Database connection OK")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
