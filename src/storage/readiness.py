"""Data-source readiness for the data-foundation profiles (plan E7 / decision G2).

One rule, used everywhere a profile's data is selected or read: Lumen never
serves from a representation that is missing, half built, still loading or
failed.

    control, scoped   need nothing here; they read the ingested tables only
    structured        needs a completed structured load (src/storage/load_structured.py)
    v2                needs that, plus the completed note-index build named by
                      LUMEN_CHUNK_BUILD, plus a retriever that can query the
                      profile's chunk table

`problems()` lists what is wrong (empty when ready) and is what /ready reports.
`require()` raises DataSourceNotReady and is called at the point of use: the
structured lab query and the retriever. That exception is never a "no answer":
callers must let it through instead of falling back to another source.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import text

from src.config import CHUNK_BUILD, DATA_PROFILE, PROFILE_SETTINGS
from src.retrieval.index_provenance import RETRIEVER_CHUNK_TABLES

STRUCTURED_TABLES = ("labevents_full", "d_icd_diagnoses", "d_icd_procedures")
STRUCTURED = "structured_load"
CHUNKS = "chunk_build"
RETRIEVER = "retriever_chunk_table"


class DataSourceNotReady(RuntimeError):
    """The data a profile is configured to read is not usable."""
    def __init__(self, component: str, reason: str):
        super().__init__(f"{component}: {reason}")
        self.component, self.reason = component, reason


def needs(component: str, settings: Optional[dict] = None) -> bool:
    """Whether the active profile reads the data `component` stands for."""
    settings = PROFILE_SETTINGS if settings is None else settings
    if component == STRUCTURED:
        return settings["lab_table"] == "labevents_full" or settings["sql_paths"]
    return settings["chunk_table"] == "note_chunks_v2"


def _exists(conn, table: str) -> bool:
    return bool(conn.execute(text("SELECT to_regclass(:name)"), {"name": table}).scalar())


def retriever_problem(settings: Optional[dict] = None, supported: Optional[tuple] = None) -> Optional[str]:
    """Why the retriever cannot serve the profile's chunk table, or None. No query:
    a build can be complete and present and still not be what the retriever reads."""
    settings = PROFILE_SETTINGS if settings is None else settings
    supported = RETRIEVER_CHUNK_TABLES if supported is None else supported
    wanted = settings["chunk_table"]
    if wanted in supported:
        return None
    return f"the profile reads {wanted}, but the retriever queries {', '.join(supported)} only"


def structured_problem(conn) -> Optional[str]:
    """Why the structured tables cannot be served, or None when they can.
    Cheap by design: it checks that tables exist, are non-empty and match a
    completed run that recorded rows. It does not recount rows; exact counts are
    the loader's and the audit's job."""
    missing = [t for t in (*STRUCTURED_TABLES, "structured_load_runs") if not _exists(conn, t)]
    if missing:
        return f"missing table(s): {', '.join(missing)}"
    run = conn.execute(text(
        "SELECT status, row_counts FROM structured_load_runs ORDER BY started_at DESC LIMIT 1")).first()
    if run is None:
        return "no structured load has been run"
    status, counts = run[0], run[1] or {}
    if status != "completed":
        return f"the latest structured load is {status}"
    for table in STRUCTURED_TABLES:
        if not counts.get(table):
            return f"the completed load recorded no rows for {table}"
        if not conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM {table})")).scalar():    # table names are the constants above
            return f"{table} is empty although the load recorded {counts[table]} rows"
    if not conn.execute(text("SELECT to_regclass('idx_labfull_subject_item_time')")).scalar():
        return "index idx_labfull_subject_item_time is missing"
    return None


def chunk_build_problem(conn, build_id: Optional[str]) -> Optional[str]:
    """Why the selected v2 note-index build cannot be served, or None when it can."""
    if not build_id:
        return "no build selected: set LUMEN_CHUNK_BUILD to a completed build id"
    missing = [t for t in ("note_chunks_v2", "note_index_runs") if not _exists(conn, t)]
    if missing:
        return f"missing table(s): {', '.join(missing)}"
    status = conn.execute(text("SELECT status FROM note_index_runs WHERE run_id = :b"), {"b": build_id}).scalar()
    if status is None:
        return f"selected build {build_id} does not exist"
    if status != "completed":
        return f"selected build {build_id} is {status}"
    if not conn.execute(text("SELECT EXISTS (SELECT 1 FROM note_chunks_v2 WHERE build_id = :b)"), {"b": build_id}).scalar():
        return f"selected build {build_id} is completed but has no chunks"
    return None


def problems(conn_factory=None, settings: Optional[dict] = None, build_id: Optional[str] = CHUNK_BUILD,
             supported: Optional[tuple] = None) -> list[dict]:
    """Everything that stops the active profile from serving. Empty for control
    and scoped, without opening a connection."""
    found = []
    wanted = [c for c in (STRUCTURED, CHUNKS) if needs(c, settings)]
    if wanted:
        if conn_factory is None:
            from src.storage import engine
            conn_factory = engine.connect
        with conn_factory() as conn:
            for component in wanted:
                reason = structured_problem(conn) if component == STRUCTURED else chunk_build_problem(conn, build_id)
                if reason:
                    found.append({"component": component, "reason": reason})
    mismatch = retriever_problem(settings, supported)
    if mismatch:
        found.append({"component": RETRIEVER, "reason": mismatch})
    return found


def require(component: str, conn_factory=None, settings: Optional[dict] = None,
            build_id: Optional[str] = CHUNK_BUILD, supported: Optional[tuple] = None) -> None:
    """Raise DataSourceNotReady when the active profile reads `component` and it
    is not usable. A no-op, with no query, for profiles that do not read it.
    For the chunk index this also refuses a table the retriever cannot query."""
    if component == CHUNKS:
        mismatch = retriever_problem(settings, supported)
        if mismatch:
            raise DataSourceNotReady(RETRIEVER, mismatch)
    if not needs(component, settings):
        return
    if conn_factory is None:
        from src.storage import engine
        conn_factory = engine.connect
    with conn_factory() as conn:
        reason = structured_problem(conn) if component == STRUCTURED else chunk_build_problem(conn, build_id)
    if reason:
        raise DataSourceNotReady(component, reason)


__all__ = ["CHUNKS", "DATA_PROFILE", "DataSourceNotReady", "RETRIEVER", "STRUCTURED", "chunk_build_problem",
           "needs", "problems", "require", "retriever_problem", "structured_problem"]
