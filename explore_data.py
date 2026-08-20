"""
Lumen Dataset Explorer — Live Database Analysis
==================================================
Connects to your Postgres instance and prints comprehensive stats
about every table, column distributions, sample data, and data quality.

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    python -m src.storage.explore_data
"""

from __future__ import annotations

import logging
from collections import defaultdict

from sqlalchemy import text as sa_text
from src.storage import engine, check_connection

logging.basicConfig(level=logging.WARNING)


def q(sql: str, params: dict = None) -> list[dict]:
    """Run a query and return list of dicts."""
    with engine.connect() as conn:
        result = conn.execute(sa_text(sql), params or {})
        return [dict(r) for r in result.mappings().all()]


def q1(sql: str, params: dict = None):
    """Run a query and return a single scalar."""
    with engine.connect() as conn:
        result = conn.execute(sa_text(sql), params or {})
        return result.scalar()


def div(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def sub(title: str, width: int = 70):
    print(f"\n  {'─' * (width - 4)}")
    print(f"  {title}")
    print(f"  {'─' * (width - 4)}")


def main():
    print("Connecting to Lumen database...")
    if not check_connection():
        print("ERROR: Cannot connect. Is Docker running? Is lumen-pg started?")
        return

    # ==================================================================
    # TABLE OVERVIEW
    # ==================================================================
    div("TABLE OVERVIEW")

    tables = [
        "patients", "admissions", "diagnoses_icd", "labevents",
        "prescriptions", "procedures_icd", "clinical_notes",
        "note_chunks", "guideline_chunks", "ingestion_log",
    ]
    print(f"\n  {'Table':<22} {'Rows':>12}  {'Status'}")
    print(f"  {'─' * 50}")
    for table in tables:
        try:
            count = q1(f"SELECT COUNT(*) FROM {table}")
            status = "✅" if count > 0 else "⬜ empty"
            print(f"  {table:<22} {count:>12,}  {status}")
        except Exception:
            print(f"  {table:<22} {'—':>12}  ❌ not found")

    # ==================================================================
    # PATIENTS
    # ==================================================================
    div("PATIENTS")

    total = q1("SELECT COUNT(*) FROM patients")
    print(f"\n  Total patients: {total:,}")

    sub("Gender Distribution")
    for row in q("SELECT gender, COUNT(*) as cnt FROM patients GROUP BY gender ORDER BY cnt DESC"):
        pct = row["cnt"] / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {row['gender']:>3}  {row['cnt']:>6,}  ({pct:5.1f}%)  {bar}")

    sub("Age Distribution (anchor_age)")
    for row in q("""
        SELECT
            CASE
                WHEN anchor_age < 30 THEN '18-29'
                WHEN anchor_age < 50 THEN '30-49'
                WHEN anchor_age < 70 THEN '50-69'
                WHEN anchor_age < 90 THEN '70-89'
                ELSE '90+'
            END as age_group,
            COUNT(*) as cnt
        FROM patients
        GROUP BY 1 ORDER BY 1
    """):
        pct = row["cnt"] / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {row['age_group']:>6}  {row['cnt']:>6,}  ({pct:5.1f}%)  {bar}")

    sub("Mortality")
    dead = q1("SELECT COUNT(*) FROM patients WHERE dod IS NOT NULL")
    alive = total - dead
    print(f"    Alive at last contact:  {alive:>6,}  ({alive/total*100:.1f}%)")
    print(f"    Deceased (dod not null): {dead:>6,}  ({dead/total*100:.1f}%)")

    sub("Sample Rows (first 5)")
    for row in q("SELECT subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod FROM patients ORDER BY subject_id LIMIT 5"):
        dod = str(row["dod"])[:10] if row["dod"] else "NULL"
        print(f"    subject={row['subject_id']}  gender={row['gender']}  age={row['anchor_age']}  year={row['anchor_year']}  group={row['anchor_year_group']}  dod={dod}")

    # ==================================================================
    # ADMISSIONS
    # ==================================================================
    div("ADMISSIONS")

    total_adm = q1("SELECT COUNT(*) FROM admissions")
    print(f"\n  Total admissions: {total_adm:,}")
    print(f"  Avg per patient: {total_adm/total:.1f}")

    sub("Admission Type")
    for row in q("SELECT admission_type, COUNT(*) as cnt FROM admissions GROUP BY 1 ORDER BY 2 DESC LIMIT 8"):
        pct = row["cnt"] / total_adm * 100
        print(f"    {row['admission_type']:<30} {row['cnt']:>6,}  ({pct:5.1f}%)")

    sub("Admission Location")
    for row in q("SELECT admission_location, COUNT(*) as cnt FROM admissions GROUP BY 1 ORDER BY 2 DESC LIMIT 6"):
        pct = row["cnt"] / total_adm * 100
        print(f"    {row['admission_location']:<35} {row['cnt']:>6,}  ({pct:5.1f}%)")

    sub("Discharge Location")
    for row in q("SELECT discharge_location, COUNT(*) as cnt FROM admissions WHERE discharge_location != '' GROUP BY 1 ORDER BY 2 DESC LIMIT 8"):
        pct = row["cnt"] / total_adm * 100
        print(f"    {row['discharge_location']:<30} {row['cnt']:>6,}  ({pct:5.1f}%)")

    sub("Insurance")
    for row in q("SELECT insurance, COUNT(*) as cnt FROM admissions WHERE insurance != '' GROUP BY 1 ORDER BY 2 DESC"):
        pct = row["cnt"] / total_adm * 100
        print(f"    {row['insurance']:<20} {row['cnt']:>6,}  ({pct:5.1f}%)")

    sub("In-Hospital Mortality")
    expired = q1("SELECT COUNT(*) FROM admissions WHERE hospital_expire_flag = 1")
    print(f"    Died during admission:  {expired:>6,}  ({expired/total_adm*100:.1f}%)")
    print(f"    Survived:               {total_adm - expired:>6,}  ({(total_adm-expired)/total_adm*100:.1f}%)")

    sub("Patients with Most Admissions")
    for row in q("SELECT subject_id, COUNT(*) as visits FROM admissions GROUP BY 1 ORDER BY 2 DESC LIMIT 5"):
        print(f"    subject_id={row['subject_id']}  → {row['visits']} admissions")

    # ==================================================================
    # DIAGNOSES
    # ==================================================================
    div("DIAGNOSES")

    total_dx = q1("SELECT COUNT(*) FROM diagnoses_icd")
    print(f"\n  Total diagnosis records: {total_dx:,}")
    print(f"  Avg per admission: {total_dx/total_adm:.1f}")

    sub("ICD Version Distribution")
    for row in q("SELECT icd_version, COUNT(*) as cnt FROM diagnoses_icd GROUP BY 1 ORDER BY 1"):
        pct = row["cnt"] / total_dx * 100
        print(f"    ICD-{row['icd_version']}:  {row['cnt']:>8,}  ({pct:.1f}%)")

    sub("Top 15 ICD Codes")
    for row in q("SELECT icd_code, icd_version, COUNT(*) as cnt FROM diagnoses_icd GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 15"):
        print(f"    {row['icd_code']:<10} (ICD-{row['icd_version']})  {row['cnt']:>6,}")

    sub("Principal Diagnosis (seq_num=1) — Top 10")
    for row in q("SELECT icd_code, icd_version, COUNT(*) as cnt FROM diagnoses_icd WHERE seq_num = 1 GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10"):
        print(f"    {row['icd_code']:<10} (ICD-{row['icd_version']})  {row['cnt']:>6,}")

    # ==================================================================
    # LAB EVENTS
    # ==================================================================
    div("LAB EVENTS")

    total_labs = q1("SELECT COUNT(*) FROM labevents")
    print(f"\n  Total lab events: {total_labs:,}")

    sub("Top 10 Lab Tests (by itemid)")
    for row in q("""
        SELECT itemid, COUNT(*) as cnt,
               ROUND(AVG(valuenum)::numeric, 2) as avg_val,
               valueuom
        FROM labevents
        WHERE valuenum IS NOT NULL
        GROUP BY itemid, valueuom
        ORDER BY cnt DESC
        LIMIT 10
    """):
        print(f"    itemid={row['itemid']}  count={row['cnt']:>8,}  avg={row['avg_val']}  unit={row['valueuom']}")

    sub("Abnormal Labs")
    abnormal = q1("SELECT COUNT(*) FROM labevents WHERE flag = 'abnormal'")
    if abnormal:
        print(f"    Flagged abnormal: {abnormal:>10,}  ({abnormal/total_labs*100:.1f}% of all labs)")
    else:
        print(f"    No 'abnormal' flags found (flag column may use different values)")

    sub("Null vs Numeric Results")
    numeric = q1("SELECT COUNT(*) FROM labevents WHERE valuenum IS NOT NULL")
    text_only = total_labs - numeric
    print(f"    Numeric (valuenum):   {numeric:>10,}  ({numeric/total_labs*100:.1f}%)")
    print(f"    Text-only (no num):   {text_only:>10,}  ({text_only/total_labs*100:.1f}%)")

    # ==================================================================
    # PRESCRIPTIONS
    # ==================================================================
    div("PRESCRIPTIONS")

    total_rx = q1("SELECT COUNT(*) FROM prescriptions")
    print(f"\n  Total prescriptions: {total_rx:,}")
    print(f"  Avg per admission: {total_rx/total_adm:.1f}")

    sub("Top 15 Drugs")
    for row in q("SELECT drug, COUNT(*) as cnt FROM prescriptions GROUP BY 1 ORDER BY 2 DESC LIMIT 15"):
        print(f"    {row['drug']:<40} {row['cnt']:>6,}")

    sub("Routes of Administration")
    for row in q("SELECT route, COUNT(*) as cnt FROM prescriptions WHERE route != '' GROUP BY 1 ORDER BY 2 DESC LIMIT 8"):
        pct = row["cnt"] / total_rx * 100
        print(f"    {row['route']:<15} {row['cnt']:>8,}  ({pct:.1f}%)")

    sub("Drug Types")
    for row in q("SELECT drug_type, COUNT(*) as cnt FROM prescriptions GROUP BY 1 ORDER BY 2 DESC"):
        print(f"    {row['drug_type']:<15} {row['cnt']:>8,}")

    # ==================================================================
    # PROCEDURES
    # ==================================================================
    div("PROCEDURES")

    total_proc = q1("SELECT COUNT(*) FROM procedures_icd")
    print(f"\n  Total procedures: {total_proc:,}")

    sub("Top 10 Procedure Codes")
    for row in q("SELECT icd_code, icd_version, COUNT(*) as cnt FROM procedures_icd GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10"):
        print(f"    {row['icd_code']:<10} (ICD-{row['icd_version']})  {row['cnt']:>6,}")

    # ==================================================================
    # CLINICAL NOTES
    # ==================================================================
    div("CLINICAL NOTES")

    total_notes = q1("SELECT COUNT(*) FROM clinical_notes")
    print(f"\n  Total notes: {total_notes:,}")

    sub("By Note Type")
    for row in q("SELECT note_type, COUNT(*) as cnt FROM clinical_notes GROUP BY 1 ORDER BY 2 DESC"):
        pct = row["cnt"] / total_notes * 100
        print(f"    {row['note_type']:<15} {row['cnt']:>8,}  ({pct:.1f}%)")

    sub("Text Length Stats (de-identified text)")
    for row in q("""
        SELECT note_type,
               COUNT(*) as cnt,
               ROUND(AVG(LENGTH(COALESCE(text_deid, text_original)))) as avg_len,
               MIN(LENGTH(COALESCE(text_deid, text_original))) as min_len,
               MAX(LENGTH(COALESCE(text_deid, text_original))) as max_len
        FROM clinical_notes
        GROUP BY 1
    """):
        print(f"    {row['note_type']:<12}  avg={row['avg_len']:>6} chars  min={row['min_len']}  max={row['max_len']:>6}")

    sub("De-identification Coverage")
    deid_count = q1("SELECT COUNT(*) FROM clinical_notes WHERE text_deid IS NOT NULL")
    print(f"    Notes with de-id text:    {deid_count:>8,}  ({deid_count/total_notes*100:.1f}%)")
    print(f"    Notes without de-id text: {total_notes - deid_count:>8,}")

    sub("PHI Entities Detected (sample)")
    for row in q("""
        SELECT note_id, note_type,
               jsonb_array_length(phi_entities) as phi_count
        FROM clinical_notes
        WHERE phi_entities IS NOT NULL AND phi_entities != 'null'::jsonb
        ORDER BY jsonb_array_length(phi_entities) DESC
        LIMIT 5
    """):
        print(f"    note_id={row['note_id']}  type={row['note_type']:<12}  PHI entities detected: {row['phi_count']}")

    sub("Sample Note (first discharge, truncated)")
    for row in q("""
        SELECT note_id, subject_id, hadm_id, note_type,
               LEFT(COALESCE(text_deid, text_original), 500) as preview
        FROM clinical_notes
        WHERE note_type = 'discharge'
        LIMIT 1
    """):
        print(f"    note_id={row['note_id']}  subject={row['subject_id']}  hadm={row['hadm_id']}")
        print(f"    type={row['note_type']}")
        print(f"    text preview:")
        for line in row["preview"].split("\n")[:15]:
            print(f"      {line}")
        print(f"      ...")

    # ==================================================================
    # NOTE CHUNKS
    # ==================================================================
    div("NOTE CHUNKS (Embeddings)")

    total_chunks = q1("SELECT COUNT(*) FROM note_chunks")
    print(f"\n  Total chunks: {total_chunks:,}")
    if total_notes > 0:
        print(f"  Avg chunks per note: {total_chunks/total_notes:.1f}")

    sub("Token Count Distribution")
    for row in q("""
        SELECT
            CASE
                WHEN token_count < 50 THEN 'tiny (<50)'
                WHEN token_count < 100 THEN 'small (50-99)'
                WHEN token_count < 200 THEN 'medium (100-199)'
                WHEN token_count < 300 THEN 'large (200-299)'
                ELSE 'xlarge (300+)'
            END as size_bucket,
            COUNT(*) as cnt
        FROM note_chunks
        GROUP BY 1 ORDER BY 1
    """):
        pct = row["cnt"] / total_chunks * 100 if total_chunks > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {row['size_bucket']:<20} {row['cnt']:>8,}  ({pct:5.1f}%)  {bar}")

    sub("Chunks by Note Type")
    for row in q("SELECT note_type, COUNT(*) as cnt FROM note_chunks GROUP BY 1 ORDER BY 2 DESC"):
        print(f"    {row['note_type']:<15} {row['cnt']:>8,}")

    sub("Embedding Coverage")
    with_emb = q1("SELECT COUNT(*) FROM note_chunks WHERE embedding IS NOT NULL")
    print(f"    With embeddings:    {with_emb:>8,}")
    print(f"    Without embeddings: {total_chunks - with_emb:>8,}")

    # ==================================================================
    # DATA RELATIONSHIPS
    # ==================================================================
    div("DATA RELATIONSHIPS")

    sub("Patients → Admissions Coverage")
    pts_with_adm = q1("SELECT COUNT(DISTINCT subject_id) FROM admissions")
    print(f"    Patients with admissions: {pts_with_adm:>6,} / {total:,}")

    sub("Admissions → Notes Coverage")
    adm_with_notes = q1("SELECT COUNT(DISTINCT hadm_id) FROM clinical_notes WHERE hadm_id IS NOT NULL")
    print(f"    Admissions with notes: {adm_with_notes:>6,} / {total_adm:,}")

    sub("Admissions → Diagnoses Coverage")
    adm_with_dx = q1("SELECT COUNT(DISTINCT hadm_id) FROM diagnoses_icd")
    print(f"    Admissions with diagnoses: {adm_with_dx:>6,} / {total_adm:,}")

    sub("Admissions → Labs Coverage")
    adm_with_labs = q1("SELECT COUNT(DISTINCT hadm_id) FROM labevents WHERE hadm_id IS NOT NULL")
    print(f"    Admissions with lab events: {adm_with_labs:>6,} / {total_adm:,}")

    print(f"\n{'=' * 70}")
    print(f"  EXPLORATION COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
