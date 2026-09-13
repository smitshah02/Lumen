"""
Retrieval configurations under evaluation
=========================================
The five retriever configurations the eval compares, and nothing else. Each
`run_*` returns a ranked list of RetrievalResult for one query.

Moved out of eval_retrieval.py so that phase 1 (retrieve_pool.py) can depend on
the configurations WITHOUT dragging in that module's keyword judge and its
broken nDCG. The configurations were always the sound half of that file; the
measurement half is what was retired. See AUDIT_NOTES.md.

These mirror HybridRetriever.search() so the eval grades the system as shipped.
The one deliberate difference: temporal mode is pinned to "all" here, because
these run corpus-wide and unscoped. That is behaviourally identical for the
current golden set (detect_temporal_mode returns "all" for all 28 queries) —
but it does mean this harness never exercises the temporal path. That is what
eval_temporal.py is for.

Used by:
    src/evals/retrieve_pool.py   (phase 1: retrieve -> pooled.json)
"""

from __future__ import annotations

from pathlib import Path

from src.retrieval.embeddings import MedCPTEmbedder
from src.retrieval.hybrid_retriever_v2 import (
    bm25_search,
    vector_search,
    reciprocal_rank_fusion,
    deduplicate_by_note,
    apply_temporal_filter,
    expand_context,
    expand_query,
    BGEReranker,
    RetrievalResult,
)

# Reranker model paths
BGE_RERANKER_PATH = str(Path.home() / "Lumen" / "models" / "bge-reranker")
MEDCPT_RERANKER_PATH = str(Path.home() / "Lumen" / "models" / "medcpt-cross-encoder")
# Same reranker-confidence gate as HybridRetriever.search() — keep in sync.
RERANK_FALLBACK_THRESHOLD = 0.35


def run_bm25_only(query: str, embedder: MedCPTEmbedder, top_k: int = 5) -> list[RetrievalResult]:
    """BM25 only — no vector search, no reranking."""
    _, expansions = expand_query(query)
    results = bm25_search(query=query, expansions=expansions, top_n=60, min_tokens=40)

    output = []
    for i, row in enumerate(results[:top_k]):
        output.append(RetrievalResult(
            chunk_id=row["chunk_id"],
            note_id=row["note_id"],
            subject_id=row["subject_id"],
            hadm_id=row["hadm_id"],
            note_type=row["note_type"],
            chunk_index=row["chunk_index"],
            chunk_text=row["chunk_text"],
            token_count=row["token_count"],
            bm25_score=float(row.get("bm25_score", 0)),
            final_score=float(row.get("bm25_score", 0)),
            sources=["bm25"],
        ))
    return output


def run_vector_only(query: str, embedder: MedCPTEmbedder, top_k: int = 5) -> list[RetrievalResult]:
    """Vector only — no BM25, no reranking."""
    query_vec = embedder.embed_query(query)
    results = vector_search(query_embedding=query_vec, top_n=60, min_tokens=40)

    output = []
    for i, row in enumerate(results[:top_k]):
        output.append(RetrievalResult(
            chunk_id=row["chunk_id"],
            note_id=row["note_id"],
            subject_id=row["subject_id"],
            hadm_id=row["hadm_id"],
            note_type=row["note_type"],
            chunk_index=row["chunk_index"],
            chunk_text=row["chunk_text"],
            token_count=row["token_count"],
            vector_score=float(row.get("vector_score", 0)),
            final_score=float(row.get("vector_score", 0)),
            sources=["vector"],
        ))
    return output


def run_rrf_only(query: str, embedder: MedCPTEmbedder, top_k: int = 5) -> list[RetrievalResult]:
    """Hybrid RRF — BM25 + Vector fused, no reranking."""
    _, expansions = expand_query(query)
    bm25_results = bm25_search(query=query, expansions=expansions, top_n=60, min_tokens=40)
    query_vec = embedder.embed_query(query)
    vec_results = vector_search(query_embedding=query_vec, top_n=60, min_tokens=40)

    merged = reciprocal_rank_fusion(bm25_results=bm25_results, vector_results=vec_results)
    merged = deduplicate_by_note(merged, max_per_note=2)
    merged = apply_temporal_filter(merged, mode="all", boost_recent=True)

    for r in merged:
        r.final_score = r.rrf_score
    return merged[:top_k]


def run_rrf_plus_reranker(
    query: str,
    embedder: MedCPTEmbedder,
    reranker: BGEReranker,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Hybrid RRF + reranker with context windows.

    Mirrors HybridRetriever.search(): if the reranker's top confidence is below
    RERANK_FALLBACK_THRESHOLD, fall back to RRF order over the FULL candidate set
    (not just the reranked top_k) — so the eval grades the system as shipped.
    """
    _, expansions = expand_query(query)
    bm25_results = bm25_search(query=query, expansions=expansions, top_n=60, min_tokens=40)
    query_vec = embedder.embed_query(query)
    vec_results = vector_search(query_embedding=query_vec, top_n=60, min_tokens=40)

    merged = reciprocal_rank_fusion(bm25_results=bm25_results, vector_results=vec_results)
    merged = deduplicate_by_note(merged, max_per_note=2)
    merged = apply_temporal_filter(merged, mode="all", boost_recent=True)

    candidates = merged[:40]
    candidates = expand_context(candidates, window=1, max_context_tokens=600)
    reranked = reranker.rerank(query, candidates, top_k=top_k)

    # Low reranker confidence (e.g. lab values, culture results — query types
    # outside the cross-encoder's training distribution) → fall back to RRF order
    # over all candidates, exactly as the production retriever does.
    max_score = max((r.rerank_score for r in reranked), default=0.0)
    if max_score < RERANK_FALLBACK_THRESHOLD:
        for r in candidates:
            r.final_score = r.rrf_score
        return sorted(candidates, key=lambda r: r.rrf_score, reverse=True)[:top_k]

    return reranked
