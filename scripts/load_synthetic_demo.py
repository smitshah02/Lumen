"""
Load the synthetic demo corpus into the DEMO database
=====================================================
Never touches the research (MIMIC) database:
  * refuses to run unless LUMEN_DATA_PLANE=demo
  * src.storage resolves the demo plane to its own database (default lumen_demo)
    and raises if that would be the research database
  * refuses to modify a database that holds any non-synthetic subject

Creates the demo database if missing, applies the normal Lumen schema, then
replaces every synthetic row (idempotent). Indexing is the normal pipeline:

    LUMEN_DATA_PLANE=demo python scripts/load_synthetic_demo.py
    LUMEN_DATA_PLANE=demo python -m src.retrieval.index_notes

Source files: src/demo_data/*.json (see src/demo_data/README.md).
"""

from __future__ import annotations

import os
import re
import sys
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "src" / "demo_data"

SYN_SUBJECTS = (90000000, 90999999)
SYN_ITEMIDS = (990000, 990999)


def _load(name: str):
    return json.loads((DATA / name).read_text())


def _verify_manifest() -> dict:
    manifest = _load("manifest.json")
    for name, digest in manifest["sha256"].items():
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"refusing: {name} does not match manifest.json (regenerate with python -m src.demo_data.generate)")
    if not manifest.get("synthetic"):
        raise SystemExit("refusing: manifest is not marked synthetic")
    return manifest


def main() -> int:
    import src  # noqa: F401  (loads .env before the plane check)

    plane = os.environ.get("LUMEN_DATA_PLANE", "research").strip().lower()
    if plane != "demo":
        print(f"refusing: LUMEN_DATA_PLANE={plane!r}. The synthetic loader only runs with LUMEN_DATA_PLANE=demo "
              f"and never writes to the research database.", file=sys.stderr)
        return 2

    from sqlalchemy import create_engine, text
    from src import storage
    from src.storage.schema import create_schema
    from src.storage.load_d_labitems import DDL as D_LABITEMS_DDL

    url = storage.engine.url
    db = url.database
    if db == storage.RESEARCH_DB_NAME or not re.fullmatch(r"[a-z_][a-z0-9_]*", db or ""):
        print(f"refusing: target database {db!r} is not a valid demo database", file=sys.stderr)
        return 2
    manifest = _verify_manifest()

    # 1. create the demo database on the same server if it does not exist
    server = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with server.connect() as c:
        if not c.execute(text("SELECT 1 FROM pg_database WHERE datname = :d"), {"d": db}).scalar():
            c.execute(text(f'CREATE DATABASE "{db}"'))
            print(f"created database {db}")
    server.dispose()

    # 2. normal Lumen schema (extensions, tables, HNSW + GIN indexes)
    create_schema()
    with storage.engine.begin() as c:
        for stmt in filter(None, (s.strip() for s in D_LABITEMS_DDL.split(";"))):
            c.execute(text(stmt))

    lo, hi = SYN_SUBJECTS
    with storage.engine.begin() as c:
        # 3. never touch a database holding real patients
        foreign = sum(c.execute(text(f"SELECT COUNT(*) FROM {t} WHERE subject_id NOT BETWEEN :lo AND :hi"),
                                {"lo": lo, "hi": hi}).scalar()
                      for t in ("patients", "admissions", "clinical_notes", "note_chunks", "labevents", "prescriptions", "diagnoses_icd"))
        foreign += c.execute(text("SELECT COUNT(*) FROM d_labitems WHERE itemid NOT BETWEEN :a AND :b"),
                             {"a": SYN_ITEMIDS[0], "b": SYN_ITEMIDS[1]}).scalar()
        if foreign:
            raise SystemExit(f"refusing: {db} contains {foreign} non-synthetic rows; not a demo database")

        # 4. replace all synthetic rows (children first)
        for t in ("note_chunks", "clinical_notes", "labevents", "prescriptions", "diagnoses_icd", "admissions", "patients"):
            c.execute(text(f"DELETE FROM {t} WHERE subject_id BETWEEN :lo AND :hi"), {"lo": lo, "hi": hi})
        c.execute(text("DELETE FROM d_labitems WHERE itemid BETWEEN :a AND :b"), {"a": SYN_ITEMIDS[0], "b": SYN_ITEMIDS[1]})

        c.execute(text("INSERT INTO patients (subject_id, gender, anchor_age, anchor_year, anchor_year_group) "
                       "VALUES (:subject_id, :gender, :anchor_age, :anchor_year, :anchor_year_group)"),
                  _load("synthetic_patients.json"))
        c.execute(text("INSERT INTO admissions (hadm_id, subject_id, admittime, dischtime, admission_type, insurance, "
                       "discharge_location, hospital_expire_flag) VALUES (:hadm_id, :subject_id, :admittime, :dischtime, "
                       ":admission_type, :insurance, :discharge_location, :hospital_expire_flag)"),
                  _load("synthetic_admissions.json"))
        c.execute(text("INSERT INTO diagnoses_icd (subject_id, hadm_id, seq_num, icd_code, icd_version) "
                       "VALUES (:subject_id, :hadm_id, :seq_num, :icd_code, :icd_version)"),
                  _load("synthetic_diagnoses.json"))
        c.execute(text("INSERT INTO d_labitems (itemid, label, fluid, category) VALUES (:itemid, :label, :fluid, :category)"),
                  _load("synthetic_lab_items.json"))
        c.execute(text("INSERT INTO labevents (labevent_id, subject_id, hadm_id, itemid, charttime, value, valuenum, valueuom) "
                       "VALUES (:labevent_id, :subject_id, :hadm_id, :itemid, :charttime, :value, :valuenum, :valueuom)"),
                  _load("synthetic_labs.json"))
        c.execute(text("INSERT INTO prescriptions (subject_id, hadm_id, pharmacy_id, starttime, stoptime, drug_type, drug, "
                       "dose_val_rx, dose_unit_rx, route) VALUES (:subject_id, :hadm_id, :pharmacy_id, :starttime, :stoptime, "
                       ":drug_type, :drug, :dose_val_rx, :dose_unit_rx, :route)"),
                  _load("synthetic_prescriptions.json"))
        # Synthetic text has no PHI, so it goes straight into text_deid (the column
        # every reader uses); text_original stays NULL. phi_entities carries the marker.
        marker = json.dumps({"synthetic": True, "version": manifest["version"]})
        c.execute(text("INSERT INTO clinical_notes (note_id, subject_id, hadm_id, note_type, charttime, text_original, "
                       "text_deid, phi_entities) VALUES (:note_id, :subject_id, :hadm_id, :note_type, :charttime, NULL, "
                       ":text, CAST(:marker AS JSONB))"),
                  [{**n, "marker": marker} for n in _load("synthetic_notes.json")])
        c.execute(text("UPDATE clinical_notes SET text_search = to_tsvector('english', text_deid) "
                       "WHERE subject_id BETWEEN :lo AND :hi"), {"lo": lo, "hi": hi})
        c.execute(text("SELECT setval(pg_get_serial_sequence('clinical_notes', 'note_id'), "
                       "(SELECT MAX(note_id) FROM clinical_notes))"))

        counts = {t: c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                  for t in ("patients", "admissions", "diagnoses_icd", "labevents", "d_labitems", "prescriptions", "clinical_notes")}

    print(json.dumps({"database": db, "version": manifest["version"], "counts": counts}))
    print("next: LUMEN_DATA_PLANE=demo python -m src.retrieval.index_notes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
