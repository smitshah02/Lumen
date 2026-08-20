"""
Lumen Database Schema
======================
Creates all tables for:
  - MIMIC-IV core data (patients, admissions, diagnoses, lab events)
  - Clinical notes (de-identified discharge summaries & radiology reports)
  - Vector embeddings (pgvector) for RAG retrieval
  - Guideline chunks with embeddings

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    python -m src.storage.schema
"""

from __future__ import annotations

import logging
from src.storage import execute_sql, check_connection

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- ============================================================
-- Enable extensions
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for BM25-style text search

-- ============================================================
-- MIMIC-IV Core Tables (subset we actually need)
-- ============================================================

CREATE TABLE IF NOT EXISTS patients (
    subject_id      INTEGER PRIMARY KEY,
    gender          VARCHAR(1),
    anchor_age      INTEGER,
    anchor_year     INTEGER,
    anchor_year_group VARCHAR(20),
    dod             TIMESTAMP          -- date of death (NULL if alive)
);

CREATE TABLE IF NOT EXISTS admissions (
    hadm_id         INTEGER PRIMARY KEY,
    subject_id      INTEGER NOT NULL REFERENCES patients(subject_id),
    admittime       TIMESTAMP,
    dischtime       TIMESTAMP,
    deathtime       TIMESTAMP,
    admission_type  VARCHAR(50),
    admit_provider_id VARCHAR(10),
    admission_location VARCHAR(60),
    discharge_location VARCHAR(60),
    insurance       VARCHAR(255),
    language        VARCHAR(10),
    marital_status  VARCHAR(30),
    race            VARCHAR(80),
    edregtime       TIMESTAMP,
    edouttime       TIMESTAMP,
    hospital_expire_flag INTEGER
);
CREATE INDEX IF NOT EXISTS idx_admissions_subject ON admissions(subject_id);

CREATE TABLE IF NOT EXISTS diagnoses_icd (
    subject_id      INTEGER NOT NULL REFERENCES patients(subject_id),
    hadm_id         INTEGER NOT NULL REFERENCES admissions(hadm_id),
    seq_num         INTEGER,
    icd_code        VARCHAR(10),
    icd_version     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_dx_subject ON diagnoses_icd(subject_id);
CREATE INDEX IF NOT EXISTS idx_dx_hadm ON diagnoses_icd(hadm_id);
CREATE INDEX IF NOT EXISTS idx_dx_icd ON diagnoses_icd(icd_code);

CREATE TABLE IF NOT EXISTS labevents (
    labevent_id     BIGINT PRIMARY KEY,
    subject_id      INTEGER NOT NULL,
    hadm_id         INTEGER,
    specimen_id     BIGINT,
    itemid          INTEGER,
    order_provider_id VARCHAR(10),
    charttime       TIMESTAMP,
    storetime       TIMESTAMP,
    value           TEXT,
    valuenum        DOUBLE PRECISION,
    valueuom        VARCHAR(20),
    ref_range_lower DOUBLE PRECISION,
    ref_range_upper DOUBLE PRECISION,
    flag            VARCHAR(10),
    priority        VARCHAR(10),
    comments        TEXT
);
CREATE INDEX IF NOT EXISTS idx_lab_subject ON labevents(subject_id);
CREATE INDEX IF NOT EXISTS idx_lab_hadm ON labevents(hadm_id);
CREATE INDEX IF NOT EXISTS idx_lab_itemid ON labevents(itemid);
CREATE INDEX IF NOT EXISTS idx_lab_charttime ON labevents(charttime);

CREATE TABLE IF NOT EXISTS prescriptions (
    subject_id      INTEGER NOT NULL,
    hadm_id         INTEGER NOT NULL,
    pharmacy_id     BIGINT,
    poe_id          VARCHAR(25),
    poe_seq         INTEGER,
    starttime       TIMESTAMP,
    stoptime        TIMESTAMP,
    drug_type       VARCHAR(20),
    drug            TEXT,
    prod_strength   TEXT,
    form_rx         VARCHAR(25),
    dose_val_rx     TEXT,
    dose_unit_rx    VARCHAR(50),
    form_val_disp   TEXT,
    form_unit_disp  VARCHAR(50),
    doses_per_24_hrs DOUBLE PRECISION,
    route           VARCHAR(50)
);
CREATE INDEX IF NOT EXISTS idx_rx_subject ON prescriptions(subject_id);
CREATE INDEX IF NOT EXISTS idx_rx_hadm ON prescriptions(hadm_id);

CREATE TABLE IF NOT EXISTS procedures_icd (
    subject_id      INTEGER NOT NULL,
    hadm_id         INTEGER NOT NULL,
    seq_num         INTEGER,
    chartdate       DATE,
    icd_code        VARCHAR(10),
    icd_version     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_proc_subject ON procedures_icd(subject_id);
CREATE INDEX IF NOT EXISTS idx_proc_hadm ON procedures_icd(hadm_id);

-- ============================================================
-- Clinical Notes (de-identified)
-- ============================================================

CREATE TABLE IF NOT EXISTS clinical_notes (
    note_id         SERIAL PRIMARY KEY,
    subject_id      INTEGER NOT NULL,
    hadm_id         INTEGER,
    note_type       VARCHAR(30) NOT NULL,  -- 'discharge' or 'radiology'
    charttime       TIMESTAMP,
    text_original   TEXT,                   -- raw MIMIC text
    text_deid       TEXT,                   -- de-identified text
    phi_entities    JSONB,                  -- audit trail of detected PHI
    deid_recall     DOUBLE PRECISION,       -- pipeline confidence
    created_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_notes_subject ON clinical_notes(subject_id);
CREATE INDEX IF NOT EXISTS idx_notes_hadm ON clinical_notes(hadm_id);
CREATE INDEX IF NOT EXISTS idx_notes_type ON clinical_notes(note_type);

-- Full-text search index for BM25-style retrieval
ALTER TABLE clinical_notes ADD COLUMN IF NOT EXISTS text_search tsvector;
CREATE INDEX IF NOT EXISTS idx_notes_fts ON clinical_notes USING GIN(text_search);

-- ============================================================
-- Vector Embeddings (for RAG retrieval)
-- ============================================================

CREATE TABLE IF NOT EXISTS note_chunks (
    chunk_id        SERIAL PRIMARY KEY,
    note_id         INTEGER NOT NULL REFERENCES clinical_notes(note_id),
    subject_id      INTEGER NOT NULL,
    hadm_id         INTEGER,
    note_type       VARCHAR(30),
    chunk_index     INTEGER NOT NULL,       -- position within the note
    chunk_text      TEXT NOT NULL,
    token_count     INTEGER,
    embedding       vector(768),            -- MedCPT output dimension
    text_search     tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    created_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chunks_note ON note_chunks(note_id);
CREATE INDEX IF NOT EXISTS idx_chunks_subject ON note_chunks(subject_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hadm ON note_chunks(hadm_id);

CREATE INDEX IF NOT EXISTS idx_chunks_fts ON note_chunks USING GIN(text_search);
-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON note_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- Guideline Chunks (for clinical guideline retrieval)
-- ============================================================

CREATE TABLE IF NOT EXISTS guideline_chunks (
    chunk_id        SERIAL PRIMARY KEY,
    source_file     VARCHAR(255) NOT NULL,  -- PDF filename
    section_title   VARCHAR(500),
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    token_count     INTEGER,
    embedding       vector(768),
    created_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_guideline_embedding ON guideline_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- Ingestion tracking
-- ============================================================

CREATE TABLE IF NOT EXISTS ingestion_log (
    id              SERIAL PRIMARY KEY,
    table_name      VARCHAR(50) NOT NULL,
    source_file     VARCHAR(255),
    rows_loaded     INTEGER,
    started_at      TIMESTAMP DEFAULT NOW(),
    completed_at    TIMESTAMP,
    status          VARCHAR(20) DEFAULT 'running',  -- running, completed, failed
    error_message   TEXT
);
"""


def create_schema():
    """Create all database tables."""
    if not check_connection():
        raise ConnectionError("Cannot connect to database. Is Docker running?")

    logger.info("Creating database schema...")

    # Split and execute statements individually for better error reporting
    statements = [s.strip() for s in SCHEMA_SQL.split(";") if s.strip()]
    for i, stmt in enumerate(statements):
        try:
            execute_sql(stmt)
        except Exception as e:
            # Skip "already exists" errors gracefully
            if "already exists" in str(e).lower():
                continue
            logger.error(f"Error in statement {i + 1}: {e}")
            raise

    logger.info("Schema created successfully — all tables ready.")


def drop_all_tables():
    """Drop all project tables (for fresh start). Use with caution."""
    tables = [
        "ingestion_log",
        "guideline_chunks",
        "note_chunks",
        "clinical_notes",
        "procedures_icd",
        "prescriptions",
        "labevents",
        "diagnoses_icd",
        "admissions",
        "patients",
    ]
    for table in tables:
        try:
            execute_sql(f"DROP TABLE IF EXISTS {table} CASCADE")
        except Exception as e:
            logger.warning(f"Could not drop {table}: {e}")
    logger.info("All tables dropped.")


def get_table_counts() -> dict[str, int]:
    """Get row counts for all tables."""
    tables = [
        "patients", "admissions", "diagnoses_icd", "labevents",
        "prescriptions", "procedures_icd", "clinical_notes",
        "note_chunks", "guideline_chunks",
    ]
    counts = {}
    for table in tables:
        try:
            from sqlalchemy import text as sa_text
            from src.storage import engine
            with engine.connect() as conn:
                result = conn.execute(sa_text(f"SELECT COUNT(*) FROM {table}"))
                counts[table] = result.scalar()
        except Exception:
            counts[table] = -1  # table doesn't exist yet
    return counts


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    create_schema()
    print("\nTable row counts:")
    for table, count in get_table_counts().items():
        print(f"  {table:<20} {count}")
