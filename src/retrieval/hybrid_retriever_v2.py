"""
Hybrid Clinical Retriever v2
===============================
Production-grade retrieval combining BM25 + MedCPT vector + BGE reranker
with clinical query expansion, context window assembly, and smart filtering.

Pipeline:
  1. Query Expansion — add clinical synonyms/abbreviations to boost recall
  2. BM25 search (Postgres tsvector) — expanded query, OR + AND combo
  3. MedCPT vector search (pgvector cosine) — semantic matching
  4. Minimum quality filter — drop tiny/empty chunks
  5. Reciprocal Rank Fusion with overlap bonus
  6. Context Window Expansion — pull adjacent chunks for full picture
  7. Deduplication — collapse same-note chunks into richest context
  8. Temporal filter/boost
  9. BGE cross-encoder reranker on assembled context → final top-K

Improvements over v1:
  - Query expansion bridges vocabulary gap (plain language → clinical terms)
  - Min token filter removes useless 1-2 sentence radiology indications
  - Context windows give the reranker full clinical picture, not fragments
  - Overlap bonus in RRF rewards chunks found by both BM25 and vector
  - Note-level dedup prevents 3 chunks from the same note dominating results
  - Expanded BM25 with OR matching catches partial keyword hits

Usage:
    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    results = retriever.search("fluid overload swollen legs", subject_id=10000032)
"""

from __future__ import annotations

import re
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import torch
from sqlalchemy import text as sa_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.storage import engine
from src.retrieval.embeddings import MedCPTEmbedder

logger = logging.getLogger(__name__)

from pathlib import Path

DEFAULT_RERANKER_MODEL = str(Path.home() / "Lumen" / "models" / "bge-reranker")


# ===========================================================================
# Clinical Query Expansion
# ===========================================================================

# Maps plain-language / partial terms → clinical synonyms and abbreviations.
# This is the key fix for BM25 recall — "swollen legs" now also searches
# for "edema", "lower extremity", "fluid overload", etc.
CLINICAL_SYNONYMS = {
    # Symptoms → clinical terms
    "swollen legs": ["edema", "lower extremity edema", "peripheral edema", "LE edema", "anasarca"],
    "swollen": ["edema", "swelling", "distended", "enlarged"],
    "fluid overload": ["volume overload", "hypervolemia", "fluid retention", "pulmonary edema", "CHF exacerbation", "decompensated heart failure"],
    "shortness of breath": ["dyspnea", "SOB", "respiratory distress", "tachypnea"],
    "chest pain": ["angina", "ACS", "acute coronary syndrome", "substernal chest pain", "pleuritic"],
    "heart attack": ["myocardial infarction", "MI", "STEMI", "NSTEMI", "troponin elevation"],
    "heart failure": ["CHF", "congestive heart failure", "HF", "HFrEF", "HFpEF", "decompensated", "EF"],
    "blood clot": ["DVT", "deep vein thrombosis", "PE", "pulmonary embolism", "thrombosis", "VTE"],
    "kidney failure": ["renal failure", "AKI", "acute kidney injury", "CKD", "creatinine elevated", "dialysis"],
    "blood sugar": ["glucose", "hyperglycemia", "hypoglycemia", "DKA", "diabetic ketoacidosis", "HbA1c", "A1c"],
    "confused": ["altered mental status", "AMS", "delirium", "encephalopathy", "disoriented"],
    "infection": ["sepsis", "bacteremia", "fever", "leukocytosis", "WBC elevated", "cultures"],
    "bleeding": ["hemorrhage", "GI bleed", "hematemesis", "melena", "hematochezia", "transfusion"],
    "stroke": ["CVA", "cerebrovascular accident", "TIA", "hemiparesis", "aphasia", "thrombolysis"],
    "seizure": ["epilepsy", "convulsion", "status epilepticus", "anticonvulsant"],
    "fell": ["fall", "mechanical fall", "syncope", "LOC", "loss of consciousness"],
    "breathing tube": ["intubation", "mechanical ventilation", "ventilator", "endotracheal"],
    "stopped breathing": ["respiratory arrest", "apnea", "respiratory failure", "intubated"],
    "high blood pressure": ["hypertension", "HTN", "hypertensive urgency", "hypertensive emergency"],
    "low blood pressure": ["hypotension", "shock", "vasopressors", "MAP"],
    "cancer": ["malignancy", "neoplasm", "tumor", "metastatic", "oncology", "chemotherapy"],

    # Drug classes
    "blood thinner": ["anticoagulation", "heparin", "warfarin", "enoxaparin", "apixaban", "rivaroxaban", "coumadin"],
    "pain medication": ["analgesic", "opioid", "morphine", "oxycodone", "hydromorphone", "acetaminophen", "NSAID"],
    "water pill": ["diuretic", "furosemide", "lasix", "bumetanide", "spironolactone", "torsemide"],
    "insulin": ["glargine", "lantus", "humalog", "lispro", "sliding scale", "basal bolus"],
    "antibiotic": ["antimicrobial", "vancomycin", "zosyn", "piperacillin", "ceftriaxone", "meropenem", "cipro"],

    # Labs
    "potassium": ["K+", "hyperkalemia", "hypokalemia", "potassium repletion"],
    "sodium": ["Na+", "hyponatremia", "hypernatremia", "SIADH"],
    "hemoglobin": ["Hgb", "Hb", "anemia", "transfusion", "hematocrit", "HCT"],
    "kidney labs": ["creatinine", "BUN", "GFR", "eGFR", "renal function"],
}


def expand_query(query: str) -> tuple[str, list[str]]:
    """
    Expand a clinical query with synonyms and abbreviations.

    Returns:
        (original_query, list_of_expansion_terms)
    """
    query_lower = query.lower()
    expansions = set()

    # Check multi-word phrases first, then single words
    for trigger, synonyms in CLINICAL_SYNONYMS.items():
        if trigger in query_lower:
            expansions.update(synonyms)

    # Also check individual words
    for word in query_lower.split():
        if word in CLINICAL_SYNONYMS:
            expansions.update(CLINICAL_SYNONYMS[word])

    return query, list(expansions)


# ===========================================================================
# Result container
# ===========================================================================

@dataclass
class RetrievalResult:
    """A single retrieved chunk with scores and metadata."""
    chunk_id: int
    note_id: int
    subject_id: int
    hadm_id: Optional[int]
    note_type: str
    chunk_index: int
    chunk_text: str
    token_count: int
    charttime: Optional[str] = None

    # Scores from each stage
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0

    # Which retrieval methods found this chunk
    sources: list[str] = field(default_factory=list)

    # Context: assembled text including adjacent chunks
    context_text: Optional[str] = None


# ===========================================================================
# BM25 Search — expanded with OR matching
# ===========================================================================

def bm25_search(
    query: str,
    expansions: list[str],
    subject_id: Optional[int] = None,
    hadm_id: Optional[int] = None,
    note_type: Optional[str] = None,
    top_n: int = 60,
    min_tokens: int = 40,
) -> list[dict]:
    """
    Chunk-level full-text search with query expansion.

    Each chunk is ranked on its OWN tsvector (note_chunks.text_search), so
    different chunks of the same note get different scores. This fixes the
    previous note-level behavior where every chunk of a matching note shared
    one identical rank and intra-note order was arbitrary.

    Two passes:
      1. Strict match on the original query, ranked per-chunk, boosted 1.5x
      2. OR match on expansion terms (high recall)
    Merged AND-first, deduped, min-max normalized (display only; RRF uses rank).
    """
    # ---- Pass 1: original query, per-chunk rank, boosted ----
    sql_and = """
        SELECT
            nc.chunk_id, nc.note_id, nc.subject_id, nc.hadm_id,
            nc.note_type, nc.chunk_index, nc.chunk_text, nc.token_count,
            cn.charttime,
            ts_rank_cd(nc.text_search, plainto_tsquery('english', :query)) * 1.5 AS bm25_score
        FROM note_chunks nc
        JOIN clinical_notes cn ON nc.note_id = cn.note_id
        WHERE nc.text_search @@ plainto_tsquery('english', :query)
          AND nc.token_count >= :min_tokens
          AND nc.chunk_text NOT LIKE '%Unit No:%'
    """
    params_and = {"query": query, "min_tokens": min_tokens}

    if subject_id:
        sql_and += " AND nc.subject_id = :subject_id"
        params_and["subject_id"] = subject_id
    if hadm_id:
        sql_and += " AND nc.hadm_id = :hadm_id"
        params_and["hadm_id"] = hadm_id
    if note_type:
        sql_and += " AND nc.note_type = :note_type"
        params_and["note_type"] = note_type

    sql_and += " ORDER BY bm25_score DESC LIMIT :top_n"
    params_and["top_n"] = top_n

    with engine.connect() as conn:
        result = conn.execute(sa_text(sql_and), params_and)
        and_results = [dict(r) for r in result.mappings().all()]

    # ---- Pass 2: OR match on expansion terms, per-chunk ----
    or_results = []
    if expansions:
        clean_terms = []
        for term in expansions[:8]:
            words = re.findall(r"[a-zA-Z0-9]+", term)
            if len(words) == 1:
                clean_terms.append(words[0])
            elif len(words) > 1:
                clean_terms.append(" & ".join(words))

        if clean_terms:
            expansion_query = " | ".join(clean_terms)

            sql_or = """
                SELECT
                    nc.chunk_id, nc.note_id, nc.subject_id, nc.hadm_id,
                    nc.note_type, nc.chunk_index, nc.chunk_text, nc.token_count,
                    cn.charttime,
                    ts_rank_cd(nc.text_search, to_tsquery('english', :exp_query)) AS bm25_score
                FROM note_chunks nc
                JOIN clinical_notes cn ON nc.note_id = cn.note_id
                WHERE nc.text_search @@ to_tsquery('english', :exp_query)
                  AND nc.token_count >= :min_tokens
                  AND nc.chunk_text NOT LIKE '%Unit No:%'
            """
            params_or = {"exp_query": expansion_query, "min_tokens": min_tokens}

            if subject_id:
                sql_or += " AND nc.subject_id = :subject_id"
                params_or["subject_id"] = subject_id
            if hadm_id:
                sql_or += " AND nc.hadm_id = :hadm_id"
                params_or["hadm_id"] = hadm_id
            if note_type:
                sql_or += " AND nc.note_type = :note_type"
                params_or["note_type"] = note_type

            sql_or += " ORDER BY bm25_score DESC LIMIT :top_n"
            params_or["top_n"] = top_n

            try:
                with engine.connect() as conn:
                    result = conn.execute(sa_text(sql_or), params_or)
                    or_results = [dict(r) for r in result.mappings().all()]
            except Exception as e:
                logger.debug(f"Expansion query failed (non-critical): {e}")
                or_results = []

    # ---- Merge: AND results first (boosted), then unseen OR results ----
    seen_ids = set()
    merged = []
    for row in and_results:
        seen_ids.add(row["chunk_id"])
        merged.append(row)
    for row in or_results:
        if row["chunk_id"] not in seen_ids:
            seen_ids.add(row["chunk_id"])
            merged.append(row)

    merged.sort(key=lambda r: r["bm25_score"], reverse=True)

    # Min-max normalize to 0-1 (cosmetic; RRF fuses on rank, not score value)
    if merged:
        scores = [r["bm25_score"] for r in merged]
        min_s, max_s = min(scores), max(scores)
        rng = max_s - min_s if max_s > min_s else 1.0
        for r in merged:
            r["bm25_score"] = (r["bm25_score"] - min_s) / rng

    return merged[:top_n]


# ===========================================================================
# Vector Search — with min token filter
# ===========================================================================

def vector_search(
    query_embedding: np.ndarray,
    subject_id: Optional[int] = None,
    hadm_id: Optional[int] = None,
    note_type: Optional[str] = None,
    top_n: int = 60,
    min_tokens: int = 40,
) -> list[dict]:
    """
    Vector similarity search with minimum chunk size filter.
    Filters out tiny chunks that score high on similarity but carry no substance.
    """
    # Guard: MPS can produce NaN embeddings for certain queries
    if np.isnan(query_embedding).any() or np.all(query_embedding == 0):
        logger.warning("Invalid query embedding (NaN or all-zero) — skipping vector search")
        return []

    vec_str = f"[{','.join(str(float(x)) for x in query_embedding)}]"

    sql = """
        SELECT
            nc.chunk_id, nc.note_id, nc.subject_id, nc.hadm_id,
            nc.note_type, nc.chunk_index, nc.chunk_text, nc.token_count,
            cn.charttime,
            1 - (nc.embedding <=> CAST(:query_vec AS vector)) as vector_score
        FROM note_chunks nc
        JOIN clinical_notes cn ON nc.note_id = cn.note_id
        WHERE nc.embedding IS NOT NULL
          AND nc.token_count >= :min_tokens
          AND nc.chunk_text NOT LIKE '%Unit No:%'
    """
    params = {"query_vec": vec_str, "min_tokens": min_tokens}

    if subject_id:
        sql += " AND nc.subject_id = :subject_id"
        params["subject_id"] = subject_id
    if hadm_id:
        sql += " AND nc.hadm_id = :hadm_id"
        params["hadm_id"] = hadm_id
    if note_type:
        sql += " AND nc.note_type = :note_type"
        params["note_type"] = note_type

    sql += " ORDER BY nc.embedding <=> CAST(:query_vec AS vector) LIMIT :top_n"
    params["top_n"] = top_n

    try:
        with engine.connect() as conn:
            result = conn.execute(sa_text(sql), params)
            rows = result.mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"Vector search failed (pgvector error): {e}")
        return []


# ===========================================================================
# Context Window Expansion
# ===========================================================================

def fetch_adjacent_chunks(note_id: int, chunk_index: int, window: int = 1) -> list[dict]:
    """
    Fetch chunks adjacent to the matched chunk from the same note.
    This provides full clinical context instead of isolated fragments.
    """
    sql = """
        SELECT chunk_id, chunk_index, chunk_text, token_count
        FROM note_chunks
        WHERE note_id = :note_id
          AND chunk_index BETWEEN :start_idx AND :end_idx
          AND chunk_text NOT LIKE '%Unit No:%'
        ORDER BY chunk_index
    """
    params = {
        "note_id": note_id,
        "start_idx": max(0, chunk_index - window),
        "end_idx": chunk_index + window,
    }

    with engine.connect() as conn:
        result = conn.execute(sa_text(sql), params)
        return [dict(r) for r in result.mappings().all()]


def expand_context(
    results: list[RetrievalResult],
    window: int = 1,
    max_context_tokens: int = 600,
) -> list[RetrievalResult]:
    """
    For each result, fetch adjacent chunks and assemble a context window.
    Stores the assembled text in result.context_text.
    """
    for r in results:
        adjacent = fetch_adjacent_chunks(r.note_id, r.chunk_index, window=window)

        context_parts = []
        total_tokens = 0
        for chunk in adjacent:
            if total_tokens + chunk["token_count"] > max_context_tokens:
                break
            context_parts.append(chunk["chunk_text"])
            total_tokens += chunk["token_count"]

        r.context_text = "\n".join(context_parts) if context_parts else r.chunk_text
        r.token_count = total_tokens

    return results


# ===========================================================================
# Reciprocal Rank Fusion with overlap bonus
# ===========================================================================

def reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.2,
    overlap_bonus: float = 0.5,
) -> list[RetrievalResult]:
    """
    Merge BM25 and vector results using RRF with an overlap bonus.

    Chunks found by BOTH methods get an additional bonus — agreement
    between keyword and semantic search is a strong relevance signal.

    vector_weight is slightly higher (1.2) because MedCPT's biomedical
    embeddings are generally more reliable than generic BM25 for clinical text.
    """
    merged: dict[int, RetrievalResult] = {}

    # Process BM25 results
    for rank, row in enumerate(bm25_results):
        cid = row["chunk_id"]
        if cid not in merged:
            merged[cid] = RetrievalResult(
                chunk_id=cid,
                note_id=row["note_id"],
                subject_id=row["subject_id"],
                hadm_id=row["hadm_id"],
                note_type=row["note_type"],
                chunk_index=row["chunk_index"],
                chunk_text=row["chunk_text"],
                token_count=row["token_count"],
                charttime=str(row["charttime"]) if row.get("charttime") else None,
            )
        merged[cid].bm25_score = float(row.get("bm25_score", 0))
        merged[cid].rrf_score += bm25_weight / (k + rank + 1)
        if "bm25" not in merged[cid].sources:
            merged[cid].sources.append("bm25")

    # Process vector results
    for rank, row in enumerate(vector_results):
        cid = row["chunk_id"]
        if cid not in merged:
            merged[cid] = RetrievalResult(
                chunk_id=cid,
                note_id=row["note_id"],
                subject_id=row["subject_id"],
                hadm_id=row["hadm_id"],
                note_type=row["note_type"],
                chunk_index=row["chunk_index"],
                chunk_text=row["chunk_text"],
                token_count=row["token_count"],
                charttime=str(row["charttime"]) if row.get("charttime") else None,
            )
        merged[cid].vector_score = float(row.get("vector_score", 0))
        merged[cid].rrf_score += vector_weight / (k + rank + 1)
        if "vector" not in merged[cid].sources:
            merged[cid].sources.append("vector")

    # Apply overlap bonus — chunks found by both methods
    for result in merged.values():
        if "bm25" in result.sources and "vector" in result.sources:
            result.rrf_score += overlap_bonus / (k + 1)
            result.sources.append("both")

    results = sorted(merged.values(), key=lambda r: r.rrf_score, reverse=True)

    if results:
        scores = [r.rrf_score for r in results]
        min_s, max_s = min(scores), max(scores)
        rng = max_s - min_s if max_s > min_s else 1.0
        for r in results:
            r.rrf_score = (r.rrf_score - min_s) / rng

    return results


# ===========================================================================
# Note-level deduplication
# ===========================================================================

def deduplicate_by_note(
    results: list[RetrievalResult],
    max_per_note: int = 2,
) -> list[RetrievalResult]:
    """
    Limit results to max_per_note chunks per clinical note.
    Keeps the highest-scoring chunk(s) from each note so one
    verbose discharge summary doesn't dominate all result slots.
    """
    note_counts: dict[int, int] = {}
    deduped = []

    for r in results:
        count = note_counts.get(r.note_id, 0)
        if count < max_per_note:
            deduped.append(r)
            note_counts[r.note_id] = count + 1

    return deduped




def _parse_charttime(value) -> Optional[datetime]:
    """Parse a MIMIC charttime (str or datetime) into a datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip().replace("Z", "")
    if not s or s.lower() in ("none", "nat", "nan"):
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# Conservative query-intent detection — returns "all" unless a clear signal.
_TEMPORAL_PATTERNS = [
    ("latest", [r"\bmost recent\b", r"\blatest\b", r"\bnewest\b", r"\bcurrent(?:ly)?\b",
                r"\blast (?:recorded|known|measured|documented|value)\b", r"\bas of (?:now|today)\b"]),
    ("trend",  [r"\btrend(?:ing|ed)?\b", r"\bover time\b", r"\bprogression\b", r"\bserial\b",
                r"\bevolution\b", r"\bchang(?:e|ed|ing) over\b",
                r"\bover the (?:past|last) \w+ (?:days|weeks|months|years)\b"]),
    ("recent", [r"\bin the (?:past|last) \d+ (?:days|weeks|months|years)\b",
                r"\brecent(?:ly)?\b", r"\bthis admission\b", r"\bduring (?:this|the current)\b"]),
]


def detect_temporal_mode(query: str) -> str:
    q = query.lower()
    for mode, patterns in _TEMPORAL_PATTERNS:   # latest > trend > recent
        if any(re.search(p, q) for p in patterns):
            return mode
    return "all"



# ===========================================================================
# Temporal Filter
# ===========================================================================

def apply_temporal_filter(
    results: list[RetrievalResult],
    mode: str = "all",
    boost_recent: bool = True,
    recency_days: int = 365,
    boost_weight: float = 0.20,
    halflife_days: float = 180.0,
    reference_times: Optional[dict[int, datetime]] = None,
) -> list[RetrievalResult]:
    """
    Temporal reweighting/filtering that is correct for MIMIC's shifted dates.

    MIMIC-IV shifts each patient's dates into 2100-2200 with a *per-patient*
    offset. Absolute dates are meaningless across patients, but the offset is
    identical for all of one patient's records, so intervals WITHIN a patient
    are real. We anchor recency to each subject's OWN latest retrieved record,
    never a global wall-clock.

    Modes:
      "all"                  -> no temporal effect (relevance order kept)
      "recent"               -> drop records older than recency_days before the
                                subject's anchor; boost survivors by recency
      "latest"/"most_recent" -> boost toward newest per subject; keep all
      "trend"/"oldest_first" -> chronological ascending (undated sink last)

    Recency is a half-life decay on the real intra-patient interval, applied
    additively to rrf_score (additive avoids the min-maxed "0 stays 0" trap).
    Pass `reference_times` to anchor on the patient's true latest encounter
    (e.g. supplied by the patient-context agent) instead of the result set.
    """
    mode = (mode or "all").lower()
    if mode == "oldest_first":
        mode = "trend"
    if mode == "most_recent":
        mode = "latest"
    if mode == "all":
        return results

    refs: dict[int, datetime] = dict(reference_times) if reference_times else {}
    if not reference_times:
        for r in results:
            ct = _parse_charttime(r.charttime)
            if ct is None:
                continue
            if r.subject_id not in refs or ct > refs[r.subject_id]:
                refs[r.subject_id] = ct

    kept: list[RetrievalResult] = []
    for r in results:
        ct = _parse_charttime(r.charttime)
        ref = refs.get(r.subject_id)

        # Undated records keep their relevance but get no recency signal and
        # are not dropped by "recent" (we can't prove they're old).
        if ct is None or ref is None:
            kept.append(r)
            continue

        days_ago = max(0.0, (ref - ct).total_seconds() / 86400.0)

        if mode == "recent" and days_ago > recency_days:
            continue

        if boost_recent and mode in ("recent", "latest"):
            r.rrf_score += boost_weight * (0.5 ** (days_ago / halflife_days))

        kept.append(r)

    if mode == "trend":
        kept.sort(key=lambda x: _parse_charttime(x.charttime) or datetime.max)
    else:
        kept.sort(key=lambda x: x.rrf_score, reverse=True)

    return kept


# ===========================================================================
# BGE Cross-Encoder Reranker
# ===========================================================================

class BGEReranker:
    """
    Cross-encoder reranker using BGE-reranker-v2-m3.
    Now reranks on the assembled context_text (with adjacent chunks)
    instead of isolated chunk fragments — gives much better scoring.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Loading BGE Reranker from {model_path}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        logger.info(f"  Reranker loaded in {time.time() - t0:.1f}s on {self.device}")

    @torch.no_grad()
    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        """
        Rerank using context_text (expanded window) if available,
        otherwise falls back to chunk_text.
        """
        if not results:
            return []

        pairs = [
            [query, r.context_text or r.chunk_text]
            for r in results
        ]

        # Batch processing
        batch_size = 16
        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            scores = self.model(**encoded).logits.squeeze(-1)
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            all_scores.extend(scores.cpu().numpy().tolist())

        import math
        for i, result in enumerate(results):
            raw = float(all_scores[i])
            # Sigmoid normalization: maps (-∞, +∞) → (0, 1)
            normalized = 1 / (1 + math.exp(-raw))
            result.rerank_score = normalized
            result.final_score = normalized

        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_k]


# ===========================================================================
# Main Hybrid Retriever v2
# ===========================================================================

class HybridRetriever:
    """
    Full hybrid retrieval pipeline v2:
      1. Query expansion (clinical synonyms)
      2. BM25 full-text search (AND + OR with expansions)
      3. MedCPT vector similarity search
      4. Min-token quality filter
      5. Reciprocal Rank Fusion with overlap bonus
      6. Note-level deduplication
      7. Context window expansion (adjacent chunks)
      8. Temporal filtering/boosting
      9. BGE cross-encoder reranking on full context
    """

    def __init__(
        self,
        use_reranker: bool = True,
        use_query_expansion: bool = True,
        use_context_window: bool = True,
        bm25_top_n: int = 60,
        vector_top_n: int = 60,
        rerank_candidates: int = 40,
        rerank_top_k: int = 10,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.2,
        overlap_bonus: float = 0.5,
        min_chunk_tokens: int = 40,
        context_window: int = 1,
        max_per_note: int = 2,
    ):
        self.use_reranker = use_reranker
        self.use_query_expansion = use_query_expansion
        self.use_context_window = use_context_window
        self.bm25_top_n = bm25_top_n
        self.vector_top_n = vector_top_n
        self.rerank_candidates = rerank_candidates
        self.rerank_top_k = rerank_top_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.overlap_bonus = overlap_bonus
        self.min_chunk_tokens = min_chunk_tokens
        self.context_window = context_window
        self.max_per_note = max_per_note

        logger.info("Initializing HybridRetriever v2...")
        self.embedder = MedCPTEmbedder()

        if use_reranker:
            self.reranker = BGEReranker()
        else:
            self.reranker = None

        logger.info("HybridRetriever v2 ready.")

    def search(
        self,
        query: str,
        subject_id: Optional[int] = None,
        hadm_id: Optional[int] = None,
        note_type: Optional[str] = None,
        temporal_filter: str = "auto",
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        Run the full hybrid retrieval pipeline.
        """
        t0 = time.time()

        # Stage 1: Query expansion
        expansions = []
        if self.use_query_expansion:
            query, expansions = expand_query(query)
            if expansions:
                logger.debug(f"  Expanded: +{len(expansions)} terms")
        
        # Resolve temporal intent from the query when set to "auto"
        resolved_temporal = (
            detect_temporal_mode(query) if temporal_filter == "auto" else temporal_filter
        )

        # Stage 2: BM25 search (with expansions)
        bm25_results = bm25_search(
            query=query,
            expansions=expansions,
            subject_id=subject_id,
            hadm_id=hadm_id,
            note_type=note_type,
            top_n=self.bm25_top_n,
            min_tokens=self.min_chunk_tokens,
        )

        # Stage 3: Vector search
        query_vec = self.embedder.embed_query(query)
        vec_results = vector_search(
            query_embedding=query_vec,
            subject_id=subject_id,
            hadm_id=hadm_id,
            note_type=note_type,
            top_n=self.vector_top_n,
            min_tokens=self.min_chunk_tokens,
        )

        # Stage 4: Reciprocal Rank Fusion with overlap bonus
        merged = reciprocal_rank_fusion(
            bm25_results=bm25_results,
            vector_results=vec_results,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
            overlap_bonus=self.overlap_bonus,
        )

        # Stage 5: Note-level deduplication
        merged = deduplicate_by_note(merged, max_per_note=self.max_per_note)

        # Stage 6: Temporal filter (MIMIC-correct, per-patient anchored)
        merged = apply_temporal_filter(merged, mode=resolved_temporal)

        # Stage 7: Context window expansion (on top candidates only)
        candidates = merged[:self.rerank_candidates]
        if self.use_context_window:
            candidates = expand_context(
                candidates,
                window=self.context_window,
                max_context_tokens=600,
            )

        # Stage 8: Reranking on assembled context
        if self.reranker and candidates:
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            # Fall back to RRF order if reranker confidence is too low — this
            # happens when the query type (lab values, culture results) doesn't
            # match the cross-encoder's training distribution well
            max_score = max((r.rerank_score for r in reranked), default=0.0)
            if max_score < 0.35:
                logger.debug(
                    f"Reranker max score {max_score:.3f} < 0.35 — falling back to RRF order"
                )
                for r in candidates:
                    r.final_score = r.rrf_score
                results = sorted(candidates, key=lambda r: r.rrf_score, reverse=True)[:top_k]
            else:
                results = reranked
        else:
            for r in candidates:
                r.final_score = r.rrf_score
            results = candidates[:top_k]

        elapsed = time.time() - t0
        exp_str = f", +{len(expansions)} expanded" if expansions else ""
        logger.info(
            f"Search '{query[:50]}' → {len(results)} results "
            f"(bm25={len(bm25_results)}, vec={len(vec_results)}, "
            f"merged={len(merged)}{exp_str}) in {elapsed:.2f}s"
        )

        return results

    def search_simple(
        self,
        query: str,
        subject_id: Optional[int] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Simplified search interface — returns plain dicts."""
        results = self.search(query=query, subject_id=subject_id, top_k=top_k)
        return [
            {
                "chunk_text": r.context_text or r.chunk_text,
                "score": round(r.final_score, 4),
                "note_type": r.note_type,
                "subject_id": r.subject_id,
                "hadm_id": r.hadm_id,
                "charttime": r.charttime,
                "sources": r.sources,
            }
            for r in results
        ]
