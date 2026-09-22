"""
MIMIC-IV Reranker Training Data Pipeline — Approach 1 (CSV.GZ Version)
======================================================================

Reads raw MIMIC-IV csv.gz files, EXCLUDES the patients already in your
RAG PostgreSQL database, and generates reranker training data from the
remaining patients.

Prerequisites:
    pip install psycopg2-binary tqdm

Usage:
    1. Configure paths and DB connection below
    2. python mimic_reranker_csv_pipeline.py
"""

import csv
import gzip
import json
import random
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, IO

import psycopg2
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION — UPDATE THESE
# =============================================================================

# --- MIMIC-IV csv.gz directories ---
HOSP_DIR = Path("/Users/smitshah/Lumen/data/mimiciv/hosp")       # <-- CHANGE: contains patients.csv.gz, diagnoses_icd.csv.gz, etc.
NOTE_DIR = Path("/Users/smitshah/Lumen/data/mimic-iv-note/note")        # <-- CHANGE: contains discharge.csv.gz, radiology.csv.gz

# --- PostgreSQL (only used to get the 5000 patient IDs to EXCLUDE) ---
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "lumen",
    "user": "postgres",        # <-- CHANGE
    "password": "lumen",    # <-- CHANGE
}
RAG_PATIENTS_TABLE = "patients"     # <-- CHANGE if your table name differs
RAG_PATIENTS_ID_COL = "subject_id"  # <-- CHANGE if column name differs

# Alternatively, if you'd rather not query the DB, set this to a file path
# containing one subject_id per line, and set DB_CONFIG = None
RAG_PATIENTS_FILE = None  # e.g. Path("./rag_patient_ids.txt")

# --- Output ---
OUTPUT_DIR = Path("./reranker_training_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Sampling ---
MAX_PATIENTS_FOR_TRAINING = 10000  # set to None to use ALL remaining patients
NEGATIVES_PER_QUERY = 5
MIN_CHUNK_LENGTH = 100              # min characters for a note/chunk to be usable
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# =============================================================================
# HELPERS
# =============================================================================

def _find_file(directory: Path, prefix: str) -> Optional[Path]:
    """Find a csv.gz or csv file matching a prefix in a directory."""
    for ext in [".csv.gz", ".csv"]:
        p = directory / f"{prefix}{ext}"
        if p.exists():
            return p
    # Try glob
    matches = list(directory.glob(f"{prefix}*"))
    return matches[0] if matches else None


def _open_csv(path: Path) -> tuple[IO, csv.DictReader]:
    """Open a csv or csv.gz and return (file_handle, DictReader)."""
    if path.suffix == ".gz":
        import io
        f = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        f = open(path, "r", encoding="utf-8", errors="replace")
    reader = csv.DictReader(f)
    return f, reader


def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# =============================================================================
# STEP 1: Get the RAG patient IDs to exclude
# =============================================================================

def get_rag_patient_ids() -> set[int]:
    """Get the set of subject_ids already in your RAG database — we exclude these."""

    if RAG_PATIENTS_FILE and Path(RAG_PATIENTS_FILE).exists():
        logger.info(f"Loading RAG patient IDs from file: {RAG_PATIENTS_FILE}")
        with open(RAG_PATIENTS_FILE) as f:
            return {int(line.strip()) for line in f if line.strip().isdigit()}

    if DB_CONFIG:
        logger.info("Querying RAG database for patient IDs to exclude...")
        conn = psycopg2.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT DISTINCT {RAG_PATIENTS_ID_COL} FROM {RAG_PATIENTS_TABLE}")
                ids = {row[0] for row in cur.fetchall()}
                logger.info(f"  Found {len(ids)} patient IDs to exclude")
                return ids
        finally:
            conn.close()

    raise ValueError("Set either DB_CONFIG or RAG_PATIENTS_FILE to identify RAG patients")


def get_training_patient_ids(rag_ids: set[int]) -> set[int]:
    """
    Read patients.csv.gz, remove RAG patient IDs, and randomly sample
    up to MAX_PATIENTS_FOR_TRAINING patients for reranker training.
    """
    path = _find_file(HOSP_DIR, "patients")
    if not path:
        raise FileNotFoundError(f"patients.csv.gz not found in {HOSP_DIR}")

    logger.info(f"Reading all patient IDs from {path.name}...")
    all_ids = []

    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            if sid is not None and sid not in rag_ids:
                all_ids.append(sid)
    finally:
        f.close()

    logger.info(f"  Total non-RAG patients available: {len(all_ids)}")

    if MAX_PATIENTS_FOR_TRAINING and len(all_ids) > MAX_PATIENTS_FOR_TRAINING:
        random.shuffle(all_ids)
        selected = set(all_ids[:MAX_PATIENTS_FOR_TRAINING])
        logger.info(f"  Randomly selected {len(selected)} patients for training")
    else:
        selected = set(all_ids)
        logger.info(f"  Using all {len(selected)} available patients")

    return selected


# =============================================================================
# STEP 2: Load structured data from csv.gz (for training patients)
# =============================================================================

def load_diagnoses(include_ids: set[int]) -> dict[int, list[dict]]:
    """
    Load diagnoses_icd + d_icd_diagnoses, return {hadm_id: [diagnosis_info]}.
    Only includes patients IN include_ids.
    """
    # First load ICD code descriptions
    icd_lookup = {}
    path = _find_file(HOSP_DIR, "d_icd_diagnoses")
    if not path:
        logger.warning("d_icd_diagnoses not found, skipping diagnoses")
        return {}

    logger.info(f"Loading ICD diagnosis descriptions from {path.name}...")
    f, reader = _open_csv(path)
    try:
        for row in reader:
            key = (row.get("icd_code", ""), row.get("icd_version", ""))
            icd_lookup[key] = row.get("long_title", "")
    finally:
        f.close()
    logger.info(f"  Loaded {len(icd_lookup)} ICD diagnosis codes")

    # Now load diagnoses_icd, filtering out RAG patients
    path = _find_file(HOSP_DIR, "diagnoses_icd")
    if not path:
        logger.warning("diagnoses_icd not found, skipping")
        return {}

    logger.info(f"Loading diagnoses from {path.name} (for {len(include_ids)} training patients)...")
    diagnoses_by_hadm = defaultdict(list)
    kept, skipped = 0, 0

    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            if sid is None or sid not in include_ids:
                skipped += 1
                continue

            hadm_id = _safe_int(row.get("hadm_id"))
            icd_code = row.get("icd_code", "")
            icd_version = row.get("icd_version", "")
            seq_num = _safe_int(row.get("seq_num")) or 99

            if seq_num > 5:  # only top 5 diagnoses per admission
                continue

            title = icd_lookup.get((icd_code, icd_version), "")
            if not title:
                continue

            diagnoses_by_hadm[hadm_id].append({
                "subject_id": sid,
                "hadm_id": hadm_id,
                "icd_code": icd_code,
                "icd_version": int(icd_version) if icd_version else 10,
                "long_title": title,
                "seq_num": seq_num,
            })
            kept += 1
    finally:
        f.close()

    logger.info(f"  Kept {kept} diagnoses across {len(diagnoses_by_hadm)} admissions (skipped {skipped})")
    return dict(diagnoses_by_hadm)


def load_abnormal_labs(include_ids: set[int]) -> dict[int, list[dict]]:
    """Load abnormal lab results, return {hadm_id: [lab_info]}."""
    # Load lab item descriptions
    lab_lookup = {}
    path = _find_file(HOSP_DIR, "d_labitems")
    if path:
        f, reader = _open_csv(path)
        try:
            for row in reader:
                itemid = _safe_int(row.get("itemid"))
                if itemid:
                    lab_lookup[itemid] = row.get("label", "")
        finally:
            f.close()
        logger.info(f"  Loaded {len(lab_lookup)} lab item descriptions")

    path = _find_file(HOSP_DIR, "labevents")
    if not path:
        logger.warning("labevents not found, skipping labs")
        return {}

    logger.info(f"Loading abnormal labs from {path.name}...")
    labs_by_hadm = defaultdict(list)
    kept, skipped = 0, 0
    seen_per_hadm = defaultdict(set)  # deduplicate same lab per admission

    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            if sid is None or sid not in include_ids:
                skipped += 1
                continue

            # Only abnormal results
            flag = (row.get("flag") or "").strip().lower()
            if flag != "abnormal":
                continue

            hadm_id = _safe_int(row.get("hadm_id"))
            if hadm_id is None:
                continue

            itemid = _safe_int(row.get("itemid"))
            lab_name = lab_lookup.get(itemid, "")
            if not lab_name:
                continue

            # Deduplicate: one entry per lab type per admission
            if lab_name in seen_per_hadm[hadm_id]:
                continue
            seen_per_hadm[hadm_id].add(lab_name)

            labs_by_hadm[hadm_id].append({
                "subject_id": sid,
                "hadm_id": hadm_id,
                "lab_name": lab_name,
                "valuenum": _safe_float(row.get("valuenum")),
                "valueuom": row.get("valueuom", ""),
                "ref_range_lower": _safe_float(row.get("ref_range_lower")),
                "ref_range_upper": _safe_float(row.get("ref_range_upper")),
                "flag": flag,
            })
            kept += 1
    finally:
        f.close()

    logger.info(f"  Kept {kept} abnormal labs across {len(labs_by_hadm)} admissions")
    return dict(labs_by_hadm)


def load_procedures(include_ids: set[int]) -> dict[int, list[dict]]:
    """Load procedures_icd, return {hadm_id: [procedure_info]}."""
    # Load procedure descriptions
    proc_lookup = {}
    path = _find_file(HOSP_DIR, "d_icd_procedures")
    if path:
        f, reader = _open_csv(path)
        try:
            for row in reader:
                key = (row.get("icd_code", ""), row.get("icd_version", ""))
                proc_lookup[key] = row.get("long_title", "")
        finally:
            f.close()
        logger.info(f"  Loaded {len(proc_lookup)} procedure codes")

    path = _find_file(HOSP_DIR, "procedures_icd")
    if not path:
        logger.warning("procedures_icd not found, skipping")
        return {}

    logger.info(f"Loading procedures from {path.name}...")
    procs_by_hadm = defaultdict(list)
    kept = 0

    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            if sid is None or sid not in include_ids:
                continue

            hadm_id = _safe_int(row.get("hadm_id"))
            icd_code = row.get("icd_code", "")
            icd_version = row.get("icd_version", "")
            title = proc_lookup.get((icd_code, icd_version), "")
            if not title:
                continue

            procs_by_hadm[hadm_id].append({
                "subject_id": sid,
                "hadm_id": hadm_id,
                "icd_code": icd_code,
                "proc_title": title,
            })
            kept += 1
    finally:
        f.close()

    logger.info(f"  Kept {kept} procedures across {len(procs_by_hadm)} admissions")
    return dict(procs_by_hadm)


def load_medications(include_ids: set[int]) -> dict[int, list[dict]]:
    """Load prescriptions, return {hadm_id: [medication_info]}."""
    path = _find_file(HOSP_DIR, "prescriptions")
    if not path:
        logger.warning("prescriptions not found, skipping")
        return {}

    logger.info(f"Loading prescriptions from {path.name}...")
    meds_by_hadm = defaultdict(list)
    kept = 0
    seen_per_hadm = defaultdict(set)

    f, reader = _open_csv(path)
    try:
        for row in reader:
            sid = _safe_int(row.get("subject_id"))
            if sid is None or sid not in include_ids:
                continue

            hadm_id = _safe_int(row.get("hadm_id"))
            if hadm_id is None:
                continue

            drug = (row.get("drug") or "").strip()
            if not drug:
                continue

            # Deduplicate same drug per admission
            if drug in seen_per_hadm[hadm_id]:
                continue
            seen_per_hadm[hadm_id].add(drug)

            meds_by_hadm[hadm_id].append({
                "subject_id": sid,
                "hadm_id": hadm_id,
                "drug": drug,
                "route": (row.get("route") or "").strip(),
            })
            kept += 1
    finally:
        f.close()

    logger.info(f"  Kept {kept} prescriptions across {len(meds_by_hadm)} admissions")
    return dict(meds_by_hadm)


def load_notes(include_ids: set[int]) -> dict[int, list[dict]]:
    """
    Load clinical notes (discharge + radiology), return {hadm_id: [note_chunks]}.
    
    Since we're working from raw files (not pre-chunked), we split notes into
    sections using common clinical note headers as delimiters.
    """
    notes_by_hadm = defaultdict(list)

    for note_type in ["discharge", "radiology"]:
        path = _find_file(NOTE_DIR, note_type)
        if not path:
            logger.warning(f"{note_type} notes not found, skipping")
            continue

        logger.info(f"Loading {note_type} notes from {path.name}...")
        kept, skipped = 0, 0

        f, reader = _open_csv(path)
        try:
            for row in reader:
                sid = _safe_int(row.get("subject_id"))
                if sid is None or sid not in include_ids:
                    skipped += 1
                    continue

                hadm_id = _safe_int(row.get("hadm_id"))
                if hadm_id is None:
                    continue

                text = (row.get("text") or "").strip()
                if len(text) < MIN_CHUNK_LENGTH:
                    continue

                # Split into sections for better matching
                sections = split_note_into_sections(text, note_type)
                for i, section in enumerate(sections):
                    if len(section["text"]) < MIN_CHUNK_LENGTH:
                        continue
                    note_id = row.get("note_id", "")
                    notes_by_hadm[hadm_id].append({
                        "chunk_id": f"{note_type}_{note_id}_{i}",
                        "chunk_text": section["text"],
                        "section_name": section["header"],
                        "note_type": note_type,
                        "subject_id": sid,
                        "hadm_id": hadm_id,
                    })
                kept += 1
        finally:
            f.close()

        logger.info(f"  Loaded {kept} {note_type} notes (skipped {skipped})")

    total_chunks = sum(len(v) for v in notes_by_hadm.values())
    logger.info(f"  Total: {total_chunks} note sections across {len(notes_by_hadm)} admissions")
    return dict(notes_by_hadm)


def split_note_into_sections(text: str, note_type: str) -> list[dict]:
    """
    Split a clinical note into sections based on common headers.
    Returns list of {header, text} dicts.
    """
    import re

    if note_type == "discharge":
        # Common discharge summary section headers
        header_pattern = re.compile(
            r'^(Name|Sex|Service|Allergies|Attending|Chief Complaint|'
            r'Major Surgical or Invasive Procedure|'
            r'History of Present Illness|Past Medical History|'
            r'Social History|Family History|Physical Exam|'
            r'Pertinent Results|Brief Hospital Course|'
            r'Medications on Admission|Discharge Medications|'
            r'Discharge Disposition|Discharge Diagnosis|'
            r'Discharge Condition|Discharge Instructions|'
            r'Followup Instructions|Assessment and Plan|'
            r'Hospital Course|Review of Systems|'
            r'Admission Date|Discharge Date|Date of Birth)\s*:',
            re.MULTILINE | re.IGNORECASE
        )
    elif note_type == "radiology":
        header_pattern = re.compile(
            r'^(EXAMINATION|INDICATION|TECHNIQUE|COMPARISON|'
            r'FINDINGS|IMPRESSION|CONCLUSION|HISTORY|'
            r'CLINICAL INFORMATION|CLINICAL HISTORY|WET READ)\s*:',
            re.MULTILINE | re.IGNORECASE
        )
    else:
        # Fallback: split on lines that look like headers (ALL CAPS followed by colon)
        header_pattern = re.compile(r'^([A-Z][A-Z\s]{2,})\s*:', re.MULTILINE)

    splits = list(header_pattern.finditer(text))

    if not splits:
        # No headers found — return whole note as one section
        return [{"header": "full_note", "text": text}]

    sections = []

    # Content before first header
    if splits[0].start() > 0:
        pre_text = text[:splits[0].start()].strip()
        if pre_text:
            sections.append({"header": "preamble", "text": pre_text})

    # Each header section
    for i, match in enumerate(splits):
        header = match.group(1).strip()
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append({"header": header, "text": section_text})

    return sections


# =============================================================================
# QUERY BUILDERS (same logic as before)
# =============================================================================

def build_diagnosis_queries(icd_code: str, icd_title: str, icd_version: int) -> list[str]:
    title_lower = icd_title.lower()
    queries = [
        f"{title_lower} clinical course and management",
        f"what treatment was given for {title_lower}",
        f"{title_lower} assessment and plan",
    ]
    prefix = "ICD-10" if icd_version == 10 else "ICD-9"
    queries.append(f"{prefix} {icd_code} {title_lower}")
    return queries


def build_lab_queries(lab_name: str, value: Optional[float], unit: str,
                      ref_low: Optional[float], ref_high: Optional[float]) -> list[str]:
    queries = []
    lab_lower = lab_name.lower()

    if value is not None and ref_high is not None and value > ref_high:
        queries.append(f"elevated {lab_lower} workup and management")
        queries.append(f"high {lab_lower} clinical significance")
    elif value is not None and ref_low is not None and value < ref_low:
        queries.append(f"low {lab_lower} evaluation and treatment")
        queries.append(f"decreased {lab_lower} clinical significance")
    else:
        queries.append(f"{lab_lower} abnormal result clinical context")

    return queries


def build_procedure_queries(proc_title: str) -> list[str]:
    title_lower = proc_title.lower()
    return [
        f"{title_lower} procedure details and indication",
        f"why was {title_lower} performed",
        f"{title_lower} post-procedure outcome",
    ]


def build_medication_queries(drug_name: str, route: str) -> list[str]:
    drug_lower = drug_name.lower()
    queries = [
        f"why was {drug_lower} prescribed",
        f"{drug_lower} indication and treatment rationale",
    ]
    if route:
        queries.append(f"{drug_lower} {route.lower()} clinical context")
    return queries


# =============================================================================
# RELEVANCE SCORING
# =============================================================================

def score_chunk_relevance(query: str, chunk_text: str) -> float:
    """Keyword overlap relevance — picks the best note section for a query."""
    stopwords = {
        "the", "a", "an", "is", "was", "were", "and", "or", "for", "of",
        "in", "to", "with", "on", "at", "by", "from", "as", "that", "this",
        "patient", "clinical", "assessment", "plan", "what", "why", "how",
    }
    query_tokens = set(query.lower().split()) - stopwords
    chunk_tokens = set(chunk_text.lower().split())

    if not query_tokens:
        return 0.0
    return len(query_tokens & chunk_tokens) / len(query_tokens)


def find_best_chunk(query: str, chunks: list[dict], min_score: float = 0.2) -> Optional[dict]:
    """Pick the most relevant chunk for a query."""
    if not chunks:
        return None

    scored = [(score_chunk_relevance(query, c["chunk_text"]), c) for c in chunks
              if len(c["chunk_text"]) >= MIN_CHUNK_LENGTH]
    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored[0][0] >= min_score else None


# =============================================================================
# NEGATIVE SAMPLER
# =============================================================================

class NegativeSampler:
    """Builds negatives from the in-memory note data."""

    def __init__(self, notes_by_hadm: dict[int, list[dict]],
                 diagnoses_by_hadm: dict[int, list[dict]]):
        self.notes_by_hadm = notes_by_hadm

        # Build indices
        # All chunks flattened (for random sampling)
        self.all_chunks = []
        for chunks in notes_by_hadm.values():
            self.all_chunks.extend(chunks)

        # subject_id -> list of hadm_ids (for same-patient negatives)
        self.subject_hadms = defaultdict(set)
        for hadm_id, chunks in notes_by_hadm.items():
            for c in chunks:
                self.subject_hadms[c["subject_id"]].add(hadm_id)

        # icd_code -> list of hadm_ids (for same-diagnosis negatives)
        self.icd_hadms = defaultdict(set)
        for hadm_id, diags in diagnoses_by_hadm.items():
            for d in diags:
                self.icd_hadms[d["icd_code"]].add(hadm_id)

        logger.info(f"  NegativeSampler: {len(self.all_chunks)} total chunks, "
                     f"{len(self.subject_hadms)} patients, "
                     f"{len(self.icd_hadms)} ICD codes indexed")

    def get_negatives(self, positive_chunk_id: str, hadm_id: int,
                      subject_id: int, icd_code: Optional[str] = None) -> list[dict]:
        exclude_ids = {positive_chunk_id}
        negatives = []

        # Tier 1 (hardest): Same patient, different admission
        other_hadms = self.subject_hadms.get(subject_id, set()) - {hadm_id}
        if other_hadms:
            sampled_hadms = random.sample(list(other_hadms), min(2, len(other_hadms)))
            for h in sampled_hadms:
                candidates = [c for c in self.notes_by_hadm.get(h, [])
                              if c["chunk_id"] not in exclude_ids
                              and len(c["chunk_text"]) >= MIN_CHUNK_LENGTH]
                if candidates:
                    c = random.choice(candidates)
                    negatives.append({
                        "text": c["chunk_text"], "chunk_id": c["chunk_id"],
                        "neg_type": "same_patient_diff_admission"
                    })
                    exclude_ids.add(c["chunk_id"])
                if len(negatives) >= 1:
                    break

        # Tier 2 (hard): Different patient, same diagnosis
        if icd_code:
            other_hadms = self.icd_hadms.get(icd_code, set()) - {hadm_id}
            if other_hadms:
                sampled_hadms = random.sample(list(other_hadms), min(4, len(other_hadms)))
                for h in sampled_hadms:
                    candidates = [c for c in self.notes_by_hadm.get(h, [])
                                  if c["chunk_id"] not in exclude_ids
                                  and len(c["chunk_text"]) >= MIN_CHUNK_LENGTH]
                    if candidates:
                        c = random.choice(candidates)
                        negatives.append({
                            "text": c["chunk_text"], "chunk_id": c["chunk_id"],
                            "neg_type": "same_diagnosis_diff_patient"
                        })
                        exclude_ids.add(c["chunk_id"])
                    if len(negatives) >= 3:
                        break

        # Tier 3 (easy): Random chunks
        remaining = NEGATIVES_PER_QUERY - len(negatives)
        if remaining > 0:
            candidates = [c for c in random.sample(self.all_chunks,
                          min(remaining * 10, len(self.all_chunks)))
                          if c["chunk_id"] not in exclude_ids
                          and len(c["chunk_text"]) >= MIN_CHUNK_LENGTH]
            for c in candidates[:remaining]:
                negatives.append({
                    "text": c["chunk_text"], "chunk_id": c["chunk_id"],
                    "neg_type": "random"
                })

        return negatives[:NEGATIVES_PER_QUERY]


# =============================================================================
# TRAINING EXAMPLE GENERATION
# =============================================================================

@dataclass
class TrainingExample:
    query: str
    query_type: str
    positive: str
    positive_chunk_id: str
    negatives: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def generate_diagnosis_examples(diagnoses_by_hadm: dict, notes_by_hadm: dict,
                                 sampler: NegativeSampler) -> list[TrainingExample]:
    logger.info("Generating diagnosis → note pairs...")
    examples = []

    hadm_ids = list(diagnoses_by_hadm.keys())
    random.shuffle(hadm_ids)

    for hadm_id in tqdm(hadm_ids, desc="  Diagnoses"):
        chunks = notes_by_hadm.get(hadm_id, [])
        if not chunks:
            continue

        for diag in diagnoses_by_hadm[hadm_id]:
            queries = build_diagnosis_queries(
                diag["icd_code"], diag["long_title"], diag["icd_version"])

            for query in queries[:2]:
                positive = find_best_chunk(query, chunks)
                if not positive:
                    continue

                negatives = sampler.get_negatives(
                    positive["chunk_id"], hadm_id,
                    diag["subject_id"], diag["icd_code"])

                if len(negatives) < 2:
                    continue

                examples.append(TrainingExample(
                    query=query, query_type="diagnosis",
                    positive=positive["chunk_text"],
                    positive_chunk_id=positive["chunk_id"],
                    negatives=negatives,
                    metadata={"hadm_id": hadm_id, "icd_code": diag["icd_code"],
                              "icd_title": diag["long_title"]}
                ))

    logger.info(f"  → {len(examples)} diagnosis examples")
    return examples


def generate_lab_examples(labs_by_hadm: dict, notes_by_hadm: dict,
                           sampler: NegativeSampler) -> list[TrainingExample]:
    logger.info("Generating abnormal lab → note pairs...")
    examples = []

    hadm_ids = list(labs_by_hadm.keys())
    random.shuffle(hadm_ids)

    for hadm_id in tqdm(hadm_ids, desc="  Labs"):
        chunks = notes_by_hadm.get(hadm_id, [])
        if not chunks:
            continue

        for lab in labs_by_hadm[hadm_id]:
            queries = build_lab_queries(
                lab["lab_name"], lab["valuenum"], lab["valueuom"],
                lab["ref_range_lower"], lab["ref_range_upper"])

            for query in queries[:1]:
                positive = find_best_chunk(query, chunks)
                if not positive:
                    continue

                negatives = sampler.get_negatives(
                    positive["chunk_id"], hadm_id, lab["subject_id"])

                if len(negatives) < 2:
                    continue

                examples.append(TrainingExample(
                    query=query, query_type="lab",
                    positive=positive["chunk_text"],
                    positive_chunk_id=positive["chunk_id"],
                    negatives=negatives,
                    metadata={"hadm_id": hadm_id, "lab_name": lab["lab_name"],
                              "value": lab["valuenum"], "flag": lab["flag"]}
                ))

    logger.info(f"  → {len(examples)} lab examples")
    return examples


def generate_procedure_examples(procs_by_hadm: dict, notes_by_hadm: dict,
                                 sampler: NegativeSampler) -> list[TrainingExample]:
    logger.info("Generating procedure → note pairs...")
    examples = []

    for hadm_id in tqdm(list(procs_by_hadm.keys()), desc="  Procedures"):
        chunks = notes_by_hadm.get(hadm_id, [])
        if not chunks:
            continue

        for proc in procs_by_hadm[hadm_id]:
            queries = build_procedure_queries(proc["proc_title"])

            for query in queries[:2]:
                positive = find_best_chunk(query, chunks)
                if not positive:
                    continue

                negatives = sampler.get_negatives(
                    positive["chunk_id"], hadm_id, proc["subject_id"])

                if len(negatives) < 2:
                    continue

                examples.append(TrainingExample(
                    query=query, query_type="procedure",
                    positive=positive["chunk_text"],
                    positive_chunk_id=positive["chunk_id"],
                    negatives=negatives,
                    metadata={"hadm_id": hadm_id, "proc_title": proc["proc_title"]}
                ))

    logger.info(f"  → {len(examples)} procedure examples")
    return examples


def generate_medication_examples(meds_by_hadm: dict, notes_by_hadm: dict,
                                  sampler: NegativeSampler) -> list[TrainingExample]:
    logger.info("Generating medication → note pairs...")
    examples = []

    for hadm_id in tqdm(list(meds_by_hadm.keys()), desc="  Medications"):
        chunks = notes_by_hadm.get(hadm_id, [])
        if not chunks:
            continue

        for med in meds_by_hadm[hadm_id]:
            queries = build_medication_queries(med["drug"], med["route"])

            for query in queries[:1]:
                positive = find_best_chunk(query, chunks)
                if not positive:
                    continue

                negatives = sampler.get_negatives(
                    positive["chunk_id"], hadm_id, med["subject_id"])

                if len(negatives) < 2:
                    continue

                examples.append(TrainingExample(
                    query=query, query_type="medication",
                    positive=positive["chunk_text"],
                    positive_chunk_id=positive["chunk_id"],
                    negatives=negatives,
                    metadata={"hadm_id": hadm_id, "drug": med["drug"]}
                ))

    logger.info(f"  → {len(examples)} medication examples")
    return examples


# =============================================================================
# OUTPUT
# =============================================================================

def save_jsonl(examples: list[TrainingExample], filepath: Path):
    with open(filepath, "w") as f:
        for ex in examples:
            f.write(json.dumps({
                "query": ex.query, "query_type": ex.query_type,
                "positive": ex.positive, "positive_chunk_id": ex.positive_chunk_id,
                "negatives": ex.negatives, "metadata": ex.metadata,
            }) + "\n")
    logger.info(f"Saved {len(examples)} examples → {filepath}")


def save_triplets(examples: list[TrainingExample], filepath: Path):
    count = 0
    with open(filepath, "w") as f:
        for ex in examples:
            for neg in ex.negatives:
                f.write(json.dumps({
                    "query": ex.query,
                    "positive": ex.positive,
                    "negative": neg["text"],
                    "neg_type": neg["neg_type"],
                }) + "\n")
                count += 1
    logger.info(f"Saved {count} triplets → {filepath}")


def save_pairs(examples: list[TrainingExample], filepath: Path):
    count = 0
    with open(filepath, "w") as f:
        for ex in examples:
            f.write(json.dumps({"query": ex.query, "document": ex.positive, "label": 1}) + "\n")
            count += 1
            for neg in ex.negatives:
                f.write(json.dumps({"query": ex.query, "document": neg["text"], "label": 0}) + "\n")
                count += 1
    logger.info(f"Saved {count} pairs → {filepath}")


def print_stats(examples: list[TrainingExample]):
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    by_type = defaultdict(int)
    neg_types = defaultdict(int)
    total_neg = 0

    for ex in examples:
        by_type[ex.query_type] += 1
        for neg in ex.negatives:
            neg_types[neg["neg_type"]] += 1
            total_neg += 1

    print(f"\nTotal examples:   {len(examples)}")
    print(f"Total triplets:   {total_neg}")

    print(f"\nBy query type:")
    for t, c in sorted(by_type.items()):
        print(f"  {t:15s}: {c:,}")

    print(f"\nBy negative type:")
    for t, c in sorted(neg_types.items()):
        pct = c / total_neg * 100 if total_neg else 0
        print(f"  {t:35s}: {c:,} ({pct:.1f}%)")

    print(f"\n{'=' * 60}")
    print("SAMPLE EXAMPLES")
    print("=" * 60)
    for ex in random.sample(examples, min(3, len(examples))):
        print(f"\n  Query [{ex.query_type}]: {ex.query}")
        print(f"  Positive: {ex.positive[:120]}...")
        for neg in ex.negatives[:2]:
            print(f"  Neg [{neg['neg_type'][:20]}]: {neg['text'][:80]}...")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("MIMIC-IV Reranker Training Data Pipeline (CSV.GZ)")
    print("=" * 60)

    # Step 1: Get RAG patient IDs to exclude
    rag_ids = get_rag_patient_ids()
    print(f"\nExcluding {len(rag_ids)} RAG patients from training data")

    # Step 2: Select training patients from patients.csv.gz
    training_ids = get_training_patient_ids(rag_ids)
    print(f"Using {len(training_ids)} patients for reranker training\n")

    # Step 3: Load structured data (only for training patients)
    print("--- Loading structured data from csv.gz ---")
    diagnoses_by_hadm = load_diagnoses(training_ids)
    labs_by_hadm = load_abnormal_labs(training_ids)
    procs_by_hadm = load_procedures(training_ids)
    meds_by_hadm = load_medications(training_ids)

    # Step 4: Load clinical notes (only for training patients)
    print("\n--- Loading clinical notes from csv.gz ---")
    notes_by_hadm = load_notes(training_ids)

    if not notes_by_hadm:
        logger.error("No notes loaded! Check NOTE_DIR path and file names.")
        return

    # Step 5: Build negative sampler
    print("\n--- Building negative sampler indices ---")
    sampler = NegativeSampler(notes_by_hadm, diagnoses_by_hadm)

    # Step 6: Generate training examples
    print("\n--- Generating training examples ---")
    all_examples = []
    all_examples.extend(generate_diagnosis_examples(diagnoses_by_hadm, notes_by_hadm, sampler))
    all_examples.extend(generate_lab_examples(labs_by_hadm, notes_by_hadm, sampler))
    all_examples.extend(generate_procedure_examples(procs_by_hadm, notes_by_hadm, sampler))
    all_examples.extend(generate_medication_examples(meds_by_hadm, notes_by_hadm, sampler))

    if not all_examples:
        logger.error("No training examples generated! Check data overlap between structured and notes.")
        return

    random.shuffle(all_examples)

    # Step 6: Train/val split and save
    split_idx = int(len(all_examples) * 0.9)
    train = all_examples[:split_idx]
    val = all_examples[split_idx:]

    print(f"\n--- Saving outputs (train={len(train)}, val={len(val)}) ---")
    save_jsonl(train, OUTPUT_DIR / "train_full.jsonl")
    save_jsonl(val, OUTPUT_DIR / "val_full.jsonl")
    save_triplets(train, OUTPUT_DIR / "train_triplets.jsonl")
    save_triplets(val, OUTPUT_DIR / "val_triplets.jsonl")
    save_pairs(train, OUTPUT_DIR / "train_pairs.jsonl")
    save_pairs(val, OUTPUT_DIR / "val_pairs.jsonl")

    print_stats(all_examples)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()