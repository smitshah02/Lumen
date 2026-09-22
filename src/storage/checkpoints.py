"""Explicit, idempotent LangGraph checkpoint schema lifecycle."""

from __future__ import annotations

import logging

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sqlalchemy import text

from src.storage import engine

logger = logging.getLogger(__name__)

CHECKPOINT_TABLES = frozenset({
    "checkpoint_migrations", "checkpoints", "checkpoint_blobs", "checkpoint_writes",
})


def checkpoint_schema_status() -> dict:
    """Read-only checkpoint table status."""
    with engine.connect() as conn:
        present = {
            name for name in CHECKPOINT_TABLES
            if conn.execute(text("SELECT to_regclass(:name)"), {"name": name}).scalar()
        }
    missing = sorted(CHECKPOINT_TABLES - present)
    return {"ready": not missing, "missing": missing}


def initialize_checkpoint_schema() -> None:
    """Run LangGraph's own idempotent checkpoint migrations explicitly."""
    dsn = engine.url.set(drivername="postgresql").render_as_string(hide_password=False)
    pool = ConnectionPool(conninfo=dsn, min_size=1, max_size=2,
                          kwargs={"autocommit": True, "row_factory": dict_row})
    try:
        PostgresSaver(pool).setup()
    finally:
        pool.close()
    status = checkpoint_schema_status()
    if not status["ready"]:
        raise RuntimeError(f"checkpoint schema initialization incomplete: {status['missing']}")
    logger.info("LangGraph checkpoint schema ready")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    initialize_checkpoint_schema()
    print("LangGraph checkpoint schema ready")
