"""Structured-data loader (data-foundation plan, E8).

Loads three tables beside the ingested ones, for the patients already in
`patients`:

    labevents_full     every lab row, with no per-patient cap
    d_icd_diagnoses    ICD diagnosis titles
    d_icd_procedures   ICD procedure titles

It is its own entry point on purpose. It never imports or calls
src/storage/ingest.py (whose loaders TRUNCATE and DELETE the ingested tables),
and the only tables it writes are the three above plus its own run log.

    python -m src.storage.load_structured        (scripts/lumen research load-structured)

Safe to rerun: each run replaces the three tables inside one transaction, so a
failed run leaves the previous contents in place. Every run is recorded in
`structured_load_runs` as running, then completed or failed.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Iterable, Iterator

from sqlalchemy import text

from src.config import DATA_PLANE, MIMIC_IV_DIR
from src.storage.schema import LAB_FULL_INDEX_SQL, STRUCTURED_SQL

logger = logging.getLogger(__name__)

OWN_TABLES = ("labevents_full", "d_icd_diagnoses", "d_icd_procedures")
LAB_COLUMNS = ("labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid", "order_provider_id", "charttime",
               "storetime", "value", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag",
               "priority", "comments")
_LAB_INTS = frozenset({"labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid"})
ICD_COLUMNS = ("icd_code", "icd_version", "long_title")
COPY_BATCH = 200_000


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else \
        open(path, "r", encoding="utf-8", newline="")


def _source(hosp_dir: Path, name: str) -> Path:
    for candidate in (hosp_dir / f"{name}.csv.gz", hosp_dir / f"{name}.csv"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name}.csv(.gz) not found in {hosp_dir}")


def lab_rows(reader: Iterable[dict], subject_ids: set[int]) -> Iterator[list]:
    """Every lab row of the given patients, in file order. There is no cap."""
    for row in reader:
        if int(row["subject_id"]) not in subject_ids:
            continue
        # ids are written as integers; a stray "123.0" is normalised rather than rejected
        yield [str(int(float(row[c]))) if c in _LAB_INTS and row[c] else row[c] for c in LAB_COLUMNS]


def icd_rows(reader: Iterable[dict]) -> Iterator[list]:
    for row in reader:
        yield [row["icd_code"].strip(), row["icd_version"], row["long_title"]]


def _copy(cur, table: str, columns: tuple, rows: Iterable[list]) -> int:
    """COPY rows into `table` in batches. An empty field is NULL."""
    statement = f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv)"
    total, buffer = 0, io.StringIO()
    writer, pending = csv.writer(buffer), 0
    for row in rows:
        writer.writerow(row)
        pending += 1
        if pending == COPY_BATCH:
            buffer.seek(0)
            cur.copy_expert(statement, buffer)
            total, pending, buffer = total + pending, 0, io.StringIO()
            writer = csv.writer(buffer)
    if pending:
        buffer.seek(0)
        cur.copy_expert(statement, buffer)
    return total + pending


def load_tables(cur, hosp_dir: Path, subject_ids: set[int]) -> dict:
    """Replace the three tables through one cursor. The caller owns the
    transaction: commit on return, roll back on any exception."""
    cur.execute(f"TRUNCATE {', '.join(OWN_TABLES)}")
    cur.execute("DROP INDEX IF EXISTS idx_labfull_subject_item_time")     # rebuilt below; faster than maintaining it
    counts = {}
    for table in ("d_icd_diagnoses", "d_icd_procedures"):
        with _open(_source(hosp_dir, table)) as f:
            counts[table] = _copy(cur, table, ICD_COLUMNS, icd_rows(csv.DictReader(f)))
    with _open(_source(hosp_dir, "labevents")) as f:
        counts["labevents_full"] = _copy(cur, "labevents_full", LAB_COLUMNS, lab_rows(csv.DictReader(f), subject_ids))
    cur.execute(LAB_FULL_INDEX_SQL)
    for table in OWN_TABLES:
        cur.execute(f"ANALYZE {table}")
    return counts


def _finish(engine, run_id: str, status: str, counts: dict | None = None, error: str | None = None) -> None:
    with engine.begin() as c:
        c.execute(text("""UPDATE structured_load_runs SET status = :status, completed_at = NOW(),
                              row_counts = CAST(:counts AS jsonb), error_message = :error
                          WHERE run_id = :run_id"""),
                  {"run_id": run_id, "status": status, "counts": json.dumps(counts or {}), "error": error})


def run_load(engine=None, hosp_dir: Path | None = None) -> dict:
    """Create the tables if needed, record the run, and load. Returns row counts."""
    if DATA_PLANE != "research":
        raise RuntimeError(f"the structured loader reads MIMIC files and runs on the research plane only (got {DATA_PLANE!r})")
    if engine is None:
        from src.storage import engine
    hosp_dir = hosp_dir or MIMIC_IV_DIR / "hosp"
    for name in ("labevents", "d_icd_diagnoses", "d_icd_procedures"):
        _source(hosp_dir, name)                                           # fail before writing anything
    with engine.begin() as c:
        for statement in STRUCTURED_SQL.split(";"):
            c.execute(text(statement))
    with engine.connect() as c:
        subject_ids = {int(r[0]) for r in c.execute(text("SELECT subject_id FROM patients"))}
    run_id = str(uuid.uuid4())
    with engine.begin() as c:
        c.execute(text("INSERT INTO structured_load_runs (run_id, status) VALUES (:run_id, 'running')"), {"run_id": run_id})

    raw = engine.raw_connection()
    try:
        counts = load_tables(raw.cursor(), hosp_dir, subject_ids)
        raw.commit()
    except BaseException as exc:
        raw.rollback()
        _finish(engine, run_id, "failed", error=type(exc).__name__)
        raise
    finally:
        raw.close()
    _finish(engine, run_id, "completed", counts)
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    loaded = run_load()
    for name, n in loaded.items():
        print(f"  {name:<18} {n:>12,} rows")
    print(f"  loaded in {time.time() - t0:.0f}s")
