"""
MIMIC-IV Data Ingestion Pipeline
==================================
Loads MIMIC-IV structured data and clinical notes into Postgres.

Handles:
  1. Core tables: patients, admissions, diagnoses, labs, prescriptions, procedures
  2. Clinical notes: discharge summaries + radiology reports (with de-identification)

Uses 5,000-patient subset by default for fast iteration.

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    python -m src.storage.ingest

    # Or with custom options:
    python -m src.storage.ingest --patients 1000 --skip-notes
    python -m src.storage.ingest --patients 10000 --skip-labs
"""

from __future__ import annotations

import csv
import gzip
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.storage import engine, execute_sql
from src.storage.schema import create_schema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — matches your directory structure
# ---------------------------------------------------------------------------
DATA_DIR = Path.home() / "Lumen" / "data"
MIMIC_IV_DIR = DATA_DIR / "mimiciv"
MIMIC_NOTE_DIR = DATA_DIR / "mimic-iv-note"

HOSP_DIR = MIMIC_IV_DIR / "hosp"
ICU_DIR = MIMIC_IV_DIR / "icu"
NOTE_DIR = MIMIC_NOTE_DIR / "note"


def _open_csv(path: Path):
    """Open a .csv or .csv.gz file and return a csv.DictReader."""
    if path.suffix == ".gz":
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = open(path, "r", encoding="utf-8")
    return f, csv.DictReader(f)


def _find_file(directory: Path, name: str) -> Optional[Path]:
    """Find a csv.gz or csv file by name."""
    for ext in [".csv.gz", ".csv"]:
        p = directory / f"{name}{ext}"
        if p.exists():
            return p
    return None


def _safe_int(val: str) -> Optional[int]:
    if not val or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_float(val: str) -> Optional[float]:
    if not val or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_ts(val: str) -> Optional[str]:
    """Return timestamp string or None."""
    if not val or val == "":
        return None
    return val


def _log_ingestion(table: str, source: str, rows: int, status: str, error: str = None):
    """Log ingestion result to the ingestion_log table."""
    try:
        execute_sql(
            """INSERT INTO ingestion_log (table_name, source_file, rows_loaded, completed_at, status, error_message)
               VALUES (:table, :source, :rows, NOW(), :status, :error)""",
            {"table": table, "source": source, "rows": rows, "status": status, "error": error},
        )
    except Exception:
        pass  # Don't fail the pipeline over logging


# ---------------------------------------------------------------------------
# Cohort selection — runs before any DB writes
# ---------------------------------------------------------------------------

# ICD-10 prefixes for AHA / ADA guideline conditions
_ICD10_GUIDELINE_PREFIXES = (
    "I50",                                # CHF / heart failure   (AHA)
    "I48",                                # AFib                  (AHA)
    "I10", "I11", "I12", "I13", "I15",   # Hypertension          (AHA)
    "E11",                                # Type 2 diabetes       (ADA)
    "N18",                                # CKD
    "J44",                                # COPD
)

# ICD-9 prefixes for the same conditions (MIMIC stores codes without decimals)
_ICD9_GUIDELINE_PREFIXES = (
    "428",                                # CHF
    "42731",                              # AFib (427.31)
    "401", "402", "403", "404", "405",   # Hypertension
    "250",                                # Diabetes (all; ICD-10 E11 narrows to T2D)
    "585",                                # CKD
    "491", "492", "496",                  # COPD
)


def _is_guideline_dx(icd_code: str, icd_version: Optional[int]) -> bool:
    code = icd_code.strip().upper()
    if icd_version == 10:
        return code.startswith(_ICD10_GUIDELINE_PREFIXES)
    return code.startswith(_ICD9_GUIDELINE_PREFIXES)


def select_cohort(
    limit: int = 5000,
    min_admissions: int = 3,
    min_age: int = 18,
    require_discharge_note: bool = True,
    require_guideline_dx: bool = True,
) -> set[int]:
    """
    Scan raw CSVs and return up to `limit` subject_ids satisfying all of:
      1. Has at least one discharge summary (if require_discharge_note)
      2. anchor_age >= min_age
      3. >= min_admissions hospital admissions
      4. At least one AHA/ADA guideline-relevant diagnosis (if require_guideline_dx)
    Results are sorted by subject_id for reproducibility.
    """
    logger.info("Selecting patient cohort from raw CSVs...")
    t0 = time.time()

    # 1. Patients with a discharge summary
    discharge_ids: set[int] = set()
    if require_discharge_note:
        path = _find_file(NOTE_DIR, "discharge")
        if not path:
            raise FileNotFoundError(
                f"discharge.csv.gz not found in {NOTE_DIR}. "
                "Pass require_discharge_note=False or --no-filter-notes to skip."
            )
        logger.info(f"  Scanning discharge notes from {path.name}...")
        f, reader = _open_csv(path)
        try:
            for row in reader:
                sid = _safe_int(row.get("subject_id"))
                if sid is not None:
                    discharge_ids.add(sid)
        finally:
            f.close()
        logger.info(f"  Discharge notes cover {len(discharge_ids):,} patients")

    # 2. Adults
    path = _find_file(HOSP_DIR, "patients")
    if not path:
        raise FileNotFoundError(f"patients.csv.gz not found in {HOSP_DIR}")
    logger.info(f"  Scanning patients for age >= {min_age}...")
    adult_ids: set[int] = set()
    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            age = _safe_int(row.get("anchor_age"))
            if sid is not None and age is not None and age >= min_age:
                adult_ids.add(sid)
    finally:
        f.close()
    logger.info(f"  Adults: {len(adult_ids):,} patients")

    # 3. Multiple admissions
    path = _find_file(HOSP_DIR, "admissions")
    if not path:
        raise FileNotFoundError(f"admissions.csv.gz not found in {HOSP_DIR}")
    logger.info(f"  Scanning admissions for >= {min_admissions} admissions per patient...")
    admission_counts: dict[int, int] = {}
    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            if sid is not None:
                admission_counts[sid] = admission_counts.get(sid, 0) + 1
    finally:
        f.close()
    multi_admit_ids = {sid for sid, n in admission_counts.items() if n >= min_admissions}
    logger.info(f"  Patients with >= {min_admissions} admissions: {len(multi_admit_ids):,}")

    # 4. Guideline-relevant diagnoses
    guideline_dx_ids: set[int] = set()
    if require_guideline_dx:
        path = _find_file(HOSP_DIR, "diagnoses_icd")
        if not path:
            logger.warning("diagnoses_icd not found; skipping guideline diagnosis filter")
        else:
            logger.info(f"  Scanning diagnoses_icd for guideline ICD codes ({path.name})...")
            f, reader = _open_csv(path)
            try:
                for row in reader:
                    sid = _safe_int(row.get("subject_id"))
                    if sid is None:
                        continue
                    icd_code = row.get("icd_code", "")
                    icd_version = _safe_int(row.get("icd_version"))
                    if icd_code and _is_guideline_dx(icd_code, icd_version):
                        guideline_dx_ids.add(sid)
            finally:
                f.close()
            logger.info(f"  Patients with guideline diagnosis: {len(guideline_dx_ids):,}")

    # 5. Intersect all filters
    cohort = adult_ids & multi_admit_ids
    if require_discharge_note:
        cohort &= discharge_ids
    if require_guideline_dx and guideline_dx_ids:
        cohort &= guideline_dx_ids

    result = set(sorted(cohort)[:limit])
    elapsed = time.time() - t0
    logger.info(
        f"Cohort: {len(result):,} patients selected "
        f"(from {len(cohort):,} qualifying) in {elapsed:.1f}s"
    )
    return result


# ---------------------------------------------------------------------------
# Core table loaders
# ---------------------------------------------------------------------------

def load_patients(subject_ids: set[int]) -> set[int]:
    """
    Insert the pre-selected cohort into the patients table.
    Reads patients.csv to pull full metadata for each subject_id in the set.
    """
    path = _find_file(HOSP_DIR, "patients")
    if not path:
        raise FileNotFoundError(f"patients.csv.gz not found in {HOSP_DIR}")

    logger.info(f"Loading {len(subject_ids):,} patients from {path.name}...")
    t0 = time.time()

    rows = []
    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            if sid not in subject_ids:
                continue
            rows.append({
                "subject_id": sid,
                "gender": row.get("gender", ""),
                "anchor_age": _safe_int(row.get("anchor_age")),
                "anchor_year": _safe_int(row.get("anchor_year")),
                "anchor_year_group": row.get("anchor_year_group", ""),
                "dod": _safe_ts(row.get("dod")),
            })
    finally:
        f.close()

    from sqlalchemy import text as sa_text
    with engine.connect() as conn:
        # CASCADE clears admissions → diagnoses_icd in FK order automatically
        conn.execute(sa_text("TRUNCATE patients CASCADE"))
        for i in range(0, len(rows), 500):
            batch = rows[i:i + 500]
            conn.execute(
                sa_text("""
                    INSERT INTO patients (subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod)
                    VALUES (:subject_id, :gender, :anchor_age, :anchor_year, :anchor_year_group, :dod)
                    ON CONFLICT (subject_id) DO NOTHING
                """),
                batch,
            )
        conn.commit()

    elapsed = time.time() - t0
    logger.info(f"  ✅ patients: {len(rows)} rows in {elapsed:.1f}s")
    _log_ingestion("patients", path.name, len(rows), "completed")
    return subject_ids


def load_admissions(subject_ids: set[int]):
    """Load admissions for the selected patients."""
    path = _find_file(HOSP_DIR, "admissions")
    if not path:
        raise FileNotFoundError(f"admissions.csv.gz not found in {HOSP_DIR}")

    logger.info(f"Loading admissions from {path.name}...")
    t0 = time.time()

    rows = []
    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            if sid not in subject_ids:
                continue
            rows.append({
                "hadm_id": _safe_int(row.get("hadm_id")),
                "subject_id": sid,
                "admittime": _safe_ts(row.get("admittime")),
                "dischtime": _safe_ts(row.get("dischtime")),
                "deathtime": _safe_ts(row.get("deathtime")),
                "admission_type": row.get("admission_type", ""),
                "admit_provider_id": row.get("admit_provider_id", ""),
                "admission_location": row.get("admission_location", ""),
                "discharge_location": row.get("discharge_location", ""),
                "insurance": row.get("insurance", ""),
                "language": row.get("language", ""),
                "marital_status": row.get("marital_status", ""),
                "race": row.get("race", ""),
                "edregtime": _safe_ts(row.get("edregtime")),
                "edouttime": _safe_ts(row.get("edouttime")),
                "hospital_expire_flag": _safe_int(row.get("hospital_expire_flag")),
            })
    finally:
        f.close()

    from sqlalchemy import text as sa_text
    with engine.connect() as conn:
        conn.execute(sa_text("DELETE FROM admissions"))
        for i in range(0, len(rows), 500):
            batch = rows[i:i + 500]
            conn.execute(
                sa_text("""
                    INSERT INTO admissions (hadm_id, subject_id, admittime, dischtime, deathtime,
                        admission_type, admit_provider_id, admission_location, discharge_location,
                        insurance, language, marital_status, race, edregtime, edouttime, hospital_expire_flag)
                    VALUES (:hadm_id, :subject_id, :admittime, :dischtime, :deathtime,
                        :admission_type, :admit_provider_id, :admission_location, :discharge_location,
                        :insurance, :language, :marital_status, :race, :edregtime, :edouttime, :hospital_expire_flag)
                    ON CONFLICT (hadm_id) DO NOTHING
                """),
                batch,
            )
        conn.commit()

    elapsed = time.time() - t0
    logger.info(f"  ✅ admissions: {len(rows)} rows in {elapsed:.1f}s")
    _log_ingestion("admissions", path.name, len(rows), "completed")


def load_diagnoses(subject_ids: set[int]):
    """Load diagnoses_icd for selected patients."""
    path = _find_file(HOSP_DIR, "diagnoses_icd")
    if not path:
        logger.warning("diagnoses_icd not found, skipping")
        return

    logger.info(f"Loading diagnoses from {path.name}...")
    t0 = time.time()

    rows = []
    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            if sid not in subject_ids:
                continue
            rows.append({
                "subject_id": sid,
                "hadm_id": _safe_int(row.get("hadm_id")),
                "seq_num": _safe_int(row.get("seq_num")),
                "icd_code": row.get("icd_code", ""),
                "icd_version": _safe_int(row.get("icd_version")),
            })
    finally:
        f.close()

    from sqlalchemy import text as sa_text
    with engine.connect() as conn:
        conn.execute(sa_text("DELETE FROM diagnoses_icd"))
        for i in range(0, len(rows), 1000):
            batch = rows[i:i + 1000]
            conn.execute(
                sa_text("""
                    INSERT INTO diagnoses_icd (subject_id, hadm_id, seq_num, icd_code, icd_version)
                    VALUES (:subject_id, :hadm_id, :seq_num, :icd_code, :icd_version)
                """),
                batch,
            )
        conn.commit()

    elapsed = time.time() - t0
    logger.info(f"  ✅ diagnoses_icd: {len(rows)} rows in {elapsed:.1f}s")
    _log_ingestion("diagnoses_icd", path.name, len(rows), "completed")


def load_labevents(subject_ids: set[int], max_per_patient: int = 200):
    """
    Load lab events for selected patients.
    Caps per patient to keep DB size manageable during development.
    """
    path = _find_file(HOSP_DIR, "labevents")
    if not path:
        logger.warning("labevents not found, skipping")
        return

    logger.info(f"Loading labevents from {path.name} (max {max_per_patient}/patient)...")
    t0 = time.time()

    rows = []
    patient_counts: dict[int, int] = {}

    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            if sid not in subject_ids:
                continue
            count = patient_counts.get(sid, 0)
            if count >= max_per_patient:
                continue
            patient_counts[sid] = count + 1

            rows.append({
                "labevent_id": _safe_int(row.get("labevent_id")),
                "subject_id": sid,
                "hadm_id": _safe_int(row.get("hadm_id")),
                "specimen_id": _safe_int(row.get("specimen_id")),
                "itemid": _safe_int(row.get("itemid")),
                "order_provider_id": row.get("order_provider_id", ""),
                "charttime": _safe_ts(row.get("charttime")),
                "storetime": _safe_ts(row.get("storetime")),
                "value": row.get("value", ""),
                "valuenum": _safe_float(row.get("valuenum")),
                "valueuom": row.get("valueuom", ""),
                "ref_range_lower": _safe_float(row.get("ref_range_lower")),
                "ref_range_upper": _safe_float(row.get("ref_range_upper")),
                "flag": row.get("flag", ""),
                "priority": row.get("priority", ""),
                "comments": row.get("comments", ""),
            })
    finally:
        f.close()

    from sqlalchemy import text as sa_text
    with engine.connect() as conn:
        conn.execute(sa_text("DELETE FROM labevents"))
        for i in range(0, len(rows), 1000):
            batch = rows[i:i + 1000]
            conn.execute(
                sa_text("""
                    INSERT INTO labevents (labevent_id, subject_id, hadm_id, specimen_id, itemid,
                        order_provider_id, charttime, storetime, value, valuenum, valueuom,
                        ref_range_lower, ref_range_upper, flag, priority, comments)
                    VALUES (:labevent_id, :subject_id, :hadm_id, :specimen_id, :itemid,
                        :order_provider_id, :charttime, :storetime, :value, :valuenum, :valueuom,
                        :ref_range_lower, :ref_range_upper, :flag, :priority, :comments)
                    ON CONFLICT (labevent_id) DO NOTHING
                """),
                batch,
            )
        conn.commit()

    elapsed = time.time() - t0
    logger.info(f"  ✅ labevents: {len(rows)} rows in {elapsed:.1f}s")
    _log_ingestion("labevents", path.name, len(rows), "completed")


def load_prescriptions(subject_ids: set[int]):
    """Load prescriptions for selected patients."""
    path = _find_file(HOSP_DIR, "prescriptions")
    if not path:
        logger.warning("prescriptions not found, skipping")
        return

    logger.info(f"Loading prescriptions from {path.name}...")
    t0 = time.time()

    rows = []
    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            if sid not in subject_ids:
                continue
            rows.append({
                "subject_id": sid,
                "hadm_id": _safe_int(row.get("hadm_id")),
                "pharmacy_id": _safe_int(row.get("pharmacy_id")),
                "poe_id": row.get("poe_id", ""),
                "poe_seq": _safe_int(row.get("poe_seq")),
                "starttime": _safe_ts(row.get("starttime")),
                "stoptime": _safe_ts(row.get("stoptime")),
                "drug_type": row.get("drug_type", ""),
                "drug": row.get("drug", ""),
                "prod_strength": row.get("prod_strength", ""),
                "form_rx": row.get("form_rx", ""),
                "dose_val_rx": row.get("dose_val_rx", ""),
                "dose_unit_rx": row.get("dose_unit_rx", ""),
                "form_val_disp": row.get("form_val_disp", ""),
                "form_unit_disp": row.get("form_unit_disp", ""),
                "doses_per_24_hrs": _safe_float(row.get("doses_per_24_hrs")),
                "route": row.get("route", ""),
            })
    finally:
        f.close()

    from sqlalchemy import text as sa_text
    with engine.connect() as conn:
        conn.execute(sa_text("DELETE FROM prescriptions"))
        for i in range(0, len(rows), 1000):
            batch = rows[i:i + 1000]
            conn.execute(
                sa_text("""
                    INSERT INTO prescriptions (subject_id, hadm_id, pharmacy_id, poe_id, poe_seq,
                        starttime, stoptime, drug_type, drug, prod_strength, form_rx,
                        dose_val_rx, dose_unit_rx, form_val_disp, form_unit_disp, doses_per_24_hrs, route)
                    VALUES (:subject_id, :hadm_id, :pharmacy_id, :poe_id, :poe_seq,
                        :starttime, :stoptime, :drug_type, :drug, :prod_strength, :form_rx,
                        :dose_val_rx, :dose_unit_rx, :form_val_disp, :form_unit_disp, :doses_per_24_hrs, :route)
                """),
                batch,
            )
        conn.commit()

    elapsed = time.time() - t0
    logger.info(f"  ✅ prescriptions: {len(rows)} rows in {elapsed:.1f}s")
    _log_ingestion("prescriptions", path.name, len(rows), "completed")


def load_procedures(subject_ids: set[int]):
    """Load procedures_icd for selected patients."""
    path = _find_file(HOSP_DIR, "procedures_icd")
    if not path:
        logger.warning("procedures_icd not found, skipping")
        return

    logger.info(f"Loading procedures from {path.name}...")
    t0 = time.time()

    rows = []
    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            if sid not in subject_ids:
                continue
            rows.append({
                "subject_id": sid,
                "hadm_id": _safe_int(row.get("hadm_id")),
                "seq_num": _safe_int(row.get("seq_num")),
                "chartdate": _safe_ts(row.get("chartdate")),
                "icd_code": row.get("icd_code", ""),
                "icd_version": _safe_int(row.get("icd_version")),
            })
    finally:
        f.close()

    from sqlalchemy import text as sa_text
    with engine.connect() as conn:
        conn.execute(sa_text("DELETE FROM procedures_icd"))
        for i in range(0, len(rows), 1000):
            batch = rows[i:i + 1000]
            conn.execute(
                sa_text("""
                    INSERT INTO procedures_icd (subject_id, hadm_id, seq_num, chartdate, icd_code, icd_version)
                    VALUES (:subject_id, :hadm_id, :seq_num, :chartdate, :icd_code, :icd_version)
                """),
                batch,
            )
        conn.commit()

    elapsed = time.time() - t0
    logger.info(f"  ✅ procedures_icd: {len(rows)} rows in {elapsed:.1f}s")
    _log_ingestion("procedures_icd", path.name, len(rows), "completed")


# ---------------------------------------------------------------------------
# Clinical Notes loader (with de-identification)
# ---------------------------------------------------------------------------

def load_clinical_notes(subject_ids: set[int], run_deid: bool = True):
    """
    Load discharge summaries and radiology reports from MIMIC-IV-Note.
    Optionally runs the de-identification pipeline on each note.
    """
    pipeline = None
    if run_deid:
        logger.info("Initializing de-identification pipeline...")
        from src.deid.pipeline import DeidentificationPipeline
        pipeline = DeidentificationPipeline(score_threshold=0.35)

    from sqlalchemy import text as sa_text

    for note_type in ["discharge", "radiology"]:
        path = _find_file(NOTE_DIR, note_type)
        if not path:
            logger.warning(f"{note_type} notes not found in {NOTE_DIR}, skipping")
            continue

        logger.info(f"Loading {note_type} notes from {path.name}...")
        t0 = time.time()

        rows = []
        f, reader = _open_csv(path)
        try:
            for row in reader:
                sid = _safe_int(row.get("subject_id"))
                if sid not in subject_ids:
                    continue

                text_original = row.get("text", "")
                if not text_original.strip():
                    continue

                text_deid = None
                phi_entities = None
                deid_recall = None

                if pipeline:
                    result = pipeline.deidentify(text_original)
                    text_deid = result.text
                    phi_entities = json.dumps(result.entities)
                    deid_recall = 1.0  # we don't have ground truth for MIMIC

                rows.append({
                    "subject_id": sid,
                    "hadm_id": _safe_int(row.get("hadm_id")),
                    "note_type": note_type,
                    "charttime": _safe_ts(row.get("charttime")),
                    "text_original": text_original,
                    "text_deid": text_deid,
                    "phi_entities": phi_entities,
                    "deid_recall": deid_recall,
                })

                # Progress logging
                if len(rows) % 500 == 0:
                    logger.info(f"  Processed {len(rows)} {note_type} notes...")

        finally:
            f.close()

        # Insert into database
        with engine.connect() as conn:
            # Clear chunks referencing these notes first (FK: note_chunks → clinical_notes)
            conn.execute(sa_text("""
                DELETE FROM note_chunks
                WHERE note_id IN (
                    SELECT note_id FROM clinical_notes WHERE note_type = :ntype
                )
            """), {"ntype": note_type})
            conn.execute(
                sa_text("DELETE FROM clinical_notes WHERE note_type = :ntype"),
                {"ntype": note_type},
            )
            for i in range(0, len(rows), 200):
                batch = rows[i:i + 200]
                conn.execute(
                    sa_text("""
                        INSERT INTO clinical_notes (subject_id, hadm_id, note_type, charttime,
                            text_original, text_deid, phi_entities, deid_recall)
                        VALUES (:subject_id, :hadm_id, :note_type, :charttime,
                            :text_original, :text_deid, :phi_entities, :deid_recall)
                    """),
                    batch,
                )
            conn.commit()

        # Update full-text search index
        with engine.connect() as conn:
            conn.execute(sa_text("""
                UPDATE clinical_notes
                SET text_search = to_tsvector('english', COALESCE(text_deid, text_original))
                WHERE note_type = :ntype AND text_search IS NULL
            """), {"ntype": note_type})
            conn.commit()

        elapsed = time.time() - t0
        logger.info(f"  ✅ {note_type} notes: {len(rows)} rows in {elapsed:.1f}s")
        _log_ingestion("clinical_notes", path.name, len(rows), "completed")


# ---------------------------------------------------------------------------
# Main ingestion orchestrator
# ---------------------------------------------------------------------------

def run_ingestion(
    patient_limit: int = 5000,
    skip_labs: bool = False,
    skip_notes: bool = False,
    skip_deid: bool = False,
    min_admissions: int = 3,
    min_age: int = 18,
    filter_notes: bool = True,
    filter_dx: bool = True,
):
    """Run the full ingestion pipeline."""
    print("=" * 70)
    print("  LUMEN DATA INGESTION PIPELINE")
    print("=" * 70)
    print()

    # Verify data directories exist
    if not HOSP_DIR.exists():
        raise FileNotFoundError(
            f"MIMIC-IV hosp directory not found at {HOSP_DIR}. "
            f"Expected: data/mimiciv/hosp/"
        )
    if not NOTE_DIR.exists() and (not skip_notes or filter_notes):
        raise FileNotFoundError(
            f"MIMIC-IV-Note directory not found at {NOTE_DIR}. "
            f"Expected: data/mimic-iv-note/note/"
        )

    # Step 1: Create schema
    print("Step 1: Creating database schema...")
    create_schema()
    print()

    # Step 2: Select cohort by scanning raw CSVs
    print(
        f"Step 2: Selecting cohort "
        f"(limit={patient_limit}, min_admissions={min_admissions}, "
        f"filter_notes={filter_notes}, filter_dx={filter_dx})..."
    )
    subject_ids = select_cohort(
        limit=patient_limit,
        min_admissions=min_admissions,
        min_age=min_age,
        require_discharge_note=filter_notes and not skip_notes,
        require_guideline_dx=filter_dx,
    )
    print()

    # Step 3: Load patients for selected cohort
    print(f"Step 3: Loading {len(subject_ids):,} patients into database...")
    load_patients(subject_ids)
    print()

    # Step 4: Load admissions
    print("Step 4: Loading admissions...")
    load_admissions(subject_ids)
    print()

    # Step 5: Load diagnoses
    print("Step 5: Loading diagnoses...")
    load_diagnoses(subject_ids)
    print()

    # Step 6: Load lab events
    if not skip_labs:
        print("Step 6: Loading lab events (this may take a few minutes)...")
        load_labevents(subject_ids, max_per_patient=200)
    else:
        print("Step 6: Skipping lab events (--skip-labs)")
    print()

    # Step 7: Load prescriptions
    print("Step 7: Loading prescriptions...")
    load_prescriptions(subject_ids)
    print()

    # Step 8: Load procedures
    print("Step 8: Loading procedures...")
    load_procedures(subject_ids)
    print()

    # Step 9: Load clinical notes with de-identification
    if not skip_notes:
        deid_label = "WITH de-identification" if not skip_deid else "WITHOUT de-identification"
        print(f"Step 9: Loading clinical notes ({deid_label})...")
        print("  (This is the slowest step — de-identifying each note)")
        load_clinical_notes(subject_ids, run_deid=not skip_deid)
    else:
        print("Step 9: Skipping clinical notes (--skip-notes)")
    print()

    # Summary
    print("=" * 70)
    print("  INGESTION COMPLETE")
    print("=" * 70)
    from src.storage.schema import get_table_counts
    counts = get_table_counts()
    print()
    for table, count in counts.items():
        status = "✅" if count > 0 else "⬜"
        print(f"  {status} {table:<20} {count:>10,} rows")
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Lumen MIMIC-IV Ingestion")
    parser.add_argument("--patients", type=int, default=5000, help="Max cohort size")
    parser.add_argument("--min-admissions", type=int, default=3, help="Min admissions per patient")
    parser.add_argument("--min-age", type=int, default=18, help="Minimum anchor_age")
    parser.add_argument("--no-filter-notes", action="store_true", help="Don't require discharge summary")
    parser.add_argument("--no-filter-dx", action="store_true", help="Don't require guideline ICD codes")
    parser.add_argument("--skip-labs", action="store_true", help="Skip lab events (fastest)")
    parser.add_argument("--skip-notes", action="store_true", help="Skip clinical notes")
    parser.add_argument("--skip-deid", action="store_true", help="Load notes without de-identification")
    args = parser.parse_args()

    run_ingestion(
        patient_limit=args.patients,
        skip_labs=args.skip_labs,
        skip_notes=args.skip_notes,
        skip_deid=args.skip_deid,
        min_admissions=args.min_admissions,
        min_age=args.min_age,
        filter_notes=not args.no_filter_notes,
        filter_dx=not args.no_filter_dx,
    )
