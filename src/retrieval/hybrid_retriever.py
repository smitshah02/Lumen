"""
Hybrid Clinical Retriever
===========================
Combines BM25 full-text search + MedCPT vector similarity + BGE reranker
for clinical note retrieval.

Pipeline:
  1. BM25 search (Postgres tsvector) → top-N candidates
  2. MedCPT vector search (pgvector cosine similarity) → top-N candidates
  3. Reciprocal Rank Fusion (RRF) → merge & re-rank
  4. Temporal filter (optional) — filter/boost by time window
  5. BGE cross-encoder reranker → final top-K

This is the core RAG retrieval layer for Lumen.

Usage:
    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()

    # Simple query
    results = retriever.search("What medications is the patient on?", subject_id=10000032)

    # Temporal query
    results = retriever.search(
        "most recent HbA1c results",
        subject_id=10000032,
        temporal_filter="recent",  # "recent", "all", or a date range
    )

    # Each result has: chunk_text, score, note_type, subject_id, hadm_id, metadata
"""

from __future__ import annotations

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

# Default reranker model path
from pathlib import Path

DEFAULT_RERANKER_MODEL = str(Path.home() / "Lumen" / "models" / "bge-reranker")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BM25 Search (Postgres full-text search)
# ---------------------------------------------------------------------------

def bm25_search(
    query: str,
    subject_id: Optional[int] = None,
    hadm_id: Optional[int] = None,
    note_type: Optional[str] = None,
    top_n: int = 50,
) -> list[dict]:
    """
    Full-text search using Postgres tsvector + ts_rank_cd.
    Uses the text_search column on clinical_notes joined with note_chunks.
    """
    # Build the tsquery — split on spaces, join with &
    terms = query.strip().split()
    tsquery = " & ".join(terms)

    sql = """
        SELECT
            nc.chunk_id, nc.note_id, nc.subject_id, nc.hadm_id,
            nc.note_type, nc.chunk_index, nc.chunk_text, nc.token_count,
            cn.charttime,
            ts_rank_cd(cn.text_search, plainto_tsquery('english', :query)) as bm25_score
        FROM note_chunks nc
        JOIN clinical_notes cn ON nc.note_id = cn.note_id
        WHERE cn.text_search @@ plainto_tsquery('english', :query)
    """
    params = {"query": query}

    if subject_id:
        sql += " AND nc.subject_id = :subject_id"
        params["subject_id"] = subject_id

    if hadm_id:
        sql += " AND nc.hadm_id = :hadm_id"
        params["hadm_id"] = hadm_id

    if note_type:
        sql += " AND nc.note_type = :note_type"
        params["note_type"] = note_type

    sql += " ORDER BY bm25_score DESC LIMIT :top_n"
    params["top_n"] = top_n

    with engine.connect() as conn:
        result = conn.execute(sa_text(sql), params)
        rows = result.mappings().all()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Vector Search (pgvector cosine similarity)
# ---------------------------------------------------------------------------

def vector_search(
    query_embedding: np.ndarray,
    subject_id: Optional[int] = None,
    hadm_id: Optional[int] = None,
    note_type: Optional[str] = None,
    top_n: int = 50,
) -> list[dict]:
    """
    Vector similarity search using pgvector's cosine distance operator.
    Uses the HNSW index on note_chunks.embedding.
    """
    # Convert numpy array to Postgres vector string
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
    """
    params = {"query_vec": vec_str}

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

    with engine.connect() as conn:
        result = conn.execute(sa_text(sql), params)
        rows = result.mappings().all()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[RetrievalResult]:
    """
    Merge BM25 and vector results using Reciprocal Rank Fusion (RRF).

    RRF score = sum( weight / (k + rank) ) across all result lists.
    k=60 is the standard constant from the original RRF paper.
    """
    # Build a map of chunk_id → RetrievalResult
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

    # Sort by RRF score
    results = sorted(merged.values(), key=lambda r: r.rrf_score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Temporal Filter
# ---------------------------------------------------------------------------

def apply_temporal_filter(
    results: list[RetrievalResult],
    mode: str = "all",
    boost_recent: bool = True,
    recency_days: int = 365,
) -> list[RetrievalResult]:
    """
    Filter or boost results based on temporal context.

    Modes:
      - "all": no filtering, optional recency boost
      - "recent": only notes from the last recency_days, boosted by recency
      - "oldest_first": reverse chronological (for trending/history queries)
    """
    if mode == "all" and not boost_recent:
        return results

    now = datetime(2200, 1, 1)  # MIMIC uses shifted dates in the 2100-2200 range
    cutoff = now - timedelta(days=recency_days)

    filtered = []
    for r in results:
        if r.charttime:
            try:
                ct = datetime.fromisoformat(str(r.charttime).replace("Z", ""))
            except (ValueError, TypeError):
                ct = None
        else:
            ct = None

        if mode == "recent" and ct and ct < cutoff:
            continue  # Skip old notes

        # Recency boost: notes closer to "now" get a small score boost
        if boost_recent and ct:
            days_ago = (now - ct).days
            # Decay: recent notes get up to 20% boost
            recency_boost = 0.2 * max(0, 1 - (days_ago / (recency_days * 2)))
            r.rrf_score *= (1 + recency_boost)

        filtered.append(r)

    if mode == "oldest_first":
        # Sort by charttime ascending for trend queries
        filtered.sort(
            key=lambda r: r.charttime or "9999",
            reverse=False,
        )
    else:
        # Re-sort by adjusted RRF score
        filtered.sort(key=lambda r: r.rrf_score, reverse=True)

    return filtered


# ---------------------------------------------------------------------------
# BGE Cross-Encoder Reranker
# ---------------------------------------------------------------------------

class BGEReranker:
    """
    Cross-encoder reranker using BGE-reranker-v2-m3.

    Takes (query, document) pairs and produces a relevance score.
    Much more accurate than bi-encoder similarity but too slow for
    first-stage retrieval — used only on the top candidates after RRF.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        # Auto-detect device
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
        Rerank results using the cross-encoder.
        Returns the top_k results sorted by reranker score.
        """
        if not results:
            return []

        # Build (query, chunk) pairs
        pairs = [[query, r.chunk_text] for r in results]

        # Tokenize
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Score
        scores = self.model(**encoded).logits.squeeze(-1)

        # Handle single result case
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)

        scores = scores.cpu().numpy()

        # Attach scores
        for i, result in enumerate(results):
            result.rerank_score = float(scores[i])
            result.final_score = float(scores[i])

        # Sort by reranker score and take top_k
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Main Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Full hybrid retrieval pipeline:
      1. BM25 full-text search
      2. MedCPT vector similarity search
      3. Reciprocal Rank Fusion
      4. Temporal filtering/boosting
      5. BGE cross-encoder reranking

    Usage:
        retriever = HybridRetriever()
        results = retriever.search("heart failure medications", subject_id=10000032)
    """

    def __init__(
        self,
        use_reranker: bool = True,
        bm25_top_n: int = 50,
        vector_top_n: int = 50,
        rerank_top_k: int = 10,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
    ):
        self.use_reranker = use_reranker
        self.bm25_top_n = bm25_top_n
        self.vector_top_n = vector_top_n
        self.rerank_top_k = rerank_top_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # Load MedCPT for query embedding
        logger.info("Initializing HybridRetriever...")
        self.embedder = MedCPTEmbedder()

        # Load BGE reranker
        if use_reranker:
            self.reranker = BGEReranker()
        else:
            self.reranker = None

        logger.info("HybridRetriever ready.")

    def search(
        self,
        query: str,
        subject_id: Optional[int] = None,
        hadm_id: Optional[int] = None,
        note_type: Optional[str] = None,
        temporal_filter: str = "all",
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        Run the full hybrid retrieval pipeline.

        Args:
            query: Natural language clinical query
            subject_id: Filter to a specific patient
            hadm_id: Filter to a specific admission
            note_type: Filter by note type ("discharge" or "radiology")
            temporal_filter: "all", "recent", or "oldest_first"
            top_k: Number of final results to return

        Returns:
            List of RetrievalResult sorted by final_score
        """
        t0 = time.time()

        # Stage 1: BM25 search
        logger.debug(f"BM25 search: '{query}'")
        bm25_results = bm25_search(
            query=query,
            subject_id=subject_id,
            hadm_id=hadm_id,
            note_type=note_type,
            top_n=self.bm25_top_n,
        )
        logger.debug(f"  BM25 returned {len(bm25_results)} results")

        # Stage 2: Vector search
        logger.debug("Vector search...")
        query_vec = self.embedder.embed_query(query)
        vec_results = vector_search(
            query_embedding=query_vec,
            subject_id=subject_id,
            hadm_id=hadm_id,
            note_type=note_type,
            top_n=self.vector_top_n,
        )
        logger.debug(f"  Vector returned {len(vec_results)} results")

        # Stage 3: Reciprocal Rank Fusion
        merged = reciprocal_rank_fusion(
            bm25_results=bm25_results,
            vector_results=vec_results,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
        )
        logger.debug(f"  RRF merged: {len(merged)} unique chunks")

        # Stage 4: Temporal filter
        if temporal_filter != "all":
            merged = apply_temporal_filter(
                merged, mode=temporal_filter, boost_recent=True
            )
            logger.debug(f"  After temporal filter: {len(merged)} chunks")
        else:
            merged = apply_temporal_filter(
                merged, mode="all", boost_recent=True
            )

        # Stage 5: Reranking (on top candidates only)
        if self.reranker and merged:
            # Rerank top candidates (limit to save compute)
            candidates = merged[: self.rerank_top_k * 3]  # rerank 3x the final top_k
            results = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            # No reranker — use RRF score as final
            for r in merged:
                r.final_score = r.rrf_score
            results = merged[:top_k]

        elapsed = time.time() - t0
        logger.info(
            f"Search '{query[:50]}...' → {len(results)} results "
            f"(bm25={len(bm25_results)}, vec={len(vec_results)}, "
            f"merged={len(merged)}) in {elapsed:.2f}s"
        )

        return results

    def search_simple(
        self,
        query: str,
        subject_id: Optional[int] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Simplified search interface — returns plain dicts for easy consumption.
        """
        results = self.search(query=query, subject_id=subject_id, top_k=top_k)
        return [
            {
                "chunk_text": r.chunk_text,
                "score": round(r.final_score, 4),
                "note_type": r.note_type,
                "subject_id": r.subject_id,
                "hadm_id": r.hadm_id,
                "charttime": r.charttime,
                "sources": r.sources,
            }
            for r in results
        ]
