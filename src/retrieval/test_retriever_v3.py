"""
Hybrid Retriever v2 Test Runner
=================================
Shows clean side-by-side comparison:
  - Hybrid (RRF) only
  - Hybrid + BGE rerank

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    python -m src.retrieval.test_retriever

    # Interactive mode:
    python -m src.retrieval.test_retriever -i

    # Skip reranker (fast):
    python -m src.retrieval.test_retriever --no-reranker
"""

from __future__ import annotations

import sys
import time
import logging
import argparse

from sqlalchemy import text as sa_text
from src.storage import engine
from src.retrieval.hybrid_retriever_v2 import (
    HybridRetriever,
    RetrievalResult,
    expand_query,
    bm25_search,
    vector_search,
    reciprocal_rank_fusion,
    deduplicate_by_note,
    apply_temporal_filter,
    expand_context,
)
from src.retrieval.embeddings import MedCPTEmbedder

logger = logging.getLogger(__name__)


SAMPLE_QUERIES = [
    "fluid overload swollen legs",
    "heart failure medications",
    "patient who stopped breathing and needed a tube",
    "blood sugar out of control",
    "sepsis antibiotics blood cultures",
    "chest pain differential diagnosis",
    "abnormal potassium lab results",
    "diabetes management insulin dosing",
    "acute kidney injury creatinine elevated",
    "GI bleeding hemoglobin transfusion",
]


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate text to max_len, replace newlines with spaces."""
    clean = " ".join(text.split())
    if len(clean) > max_len:
        return clean[:max_len] + "..."
    return clean


def run_comparison(retriever: HybridRetriever, query: str, subject_id: int = None, top_k: int = 5):
    """
    Run a query and show both RRF-only and RRF+Reranker results side by side.
    """
    # Show query
    print(f"\nQuery: {query}")

    # Show expansions
    _, expansions = expand_query(query)
    if expansions:
        exp_str = ", ".join(expansions[:6])
        if len(expansions) > 6:
            exp_str += f", +{len(expansions) - 6} more"
        print(f"Expanded: {exp_str}")

    if subject_id:
        print(f"Patient: {subject_id}")

    # Run the pipeline up to RRF (before reranking)
    query_text, exps = expand_query(query)

    bm25_results = bm25_search(
        query=query_text,
        expansions=exps,
        subject_id=subject_id,
        top_n=60,
        min_tokens=40,
    )

    query_vec = retriever.embedder.embed_query(query_text)
    vec_results = vector_search(
        query_embedding=query_vec,
        subject_id=subject_id,
        top_n=60,
        min_tokens=40,
    )

    merged = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        vector_results=vec_results,
    )
    merged = deduplicate_by_note(merged, max_per_note=2)
    merged = apply_temporal_filter(merged, mode="all", boost_recent=True)

    # ── RRF Only ──
    rrf_results = sorted(merged, key=lambda r: r.rrf_score, reverse=True)[:top_k]

    print(f"\nHybrid (RRF) only  [bm25={len(bm25_results)}, vec={len(vec_results)}, merged={len(merged)}]")
    for i, r in enumerate(rrf_results):
        src = "+".join(r.sources)
        text = truncate(r.chunk_text, 75)
        print(f"  {i+1}. [{r.rrf_score:.4f}] {r.chunk_id:<8} [{src:<12}]  {text}")

    # ── RRF + Reranker ──
    if retriever.reranker:
        # Expand context for reranker
        candidates = merged[:40]
        candidates = expand_context(candidates, window=1, max_context_tokens=600)

        reranked = retriever.reranker.rerank(query, candidates, top_k=top_k)

        # Fall back to RRF order when reranker confidence is too low —
        # cross-encoder struggles with lab/culture queries
        max_score = max((r.rerank_score for r in reranked), default=0.0)
        if max_score < 0.35:
            for r in candidates:
                r.final_score = r.rrf_score
            reranked = sorted(candidates, key=lambda r: r.rrf_score, reverse=True)[:top_k]
            label = f"Hybrid + MedCPT Cross-Encoder (RRF fallback — low confidence {max_score:.2f})"
        else:
            label = "Hybrid + MedCPT Cross-Encoder"

        print(f"\n{label}")
        for i, r in enumerate(reranked):
            src = "+".join(r.sources)
            text = truncate(r.context_text or r.chunk_text, 75)
            score = r.final_score if max_score < 0.35 else r.rerank_score
            print(f"  {i+1}. [{score:.4f}] {r.chunk_id:<8} [{src:<12}]  {text}")
    else:
        print(f"\n  (reranker disabled)")

    print()


def run_sample_queries(retriever: HybridRetriever):
    """Run all sample queries with comparison output."""
    print("=" * 90)
    print("  LUMEN HYBRID RETRIEVER v2 — RRF vs RRF + BMedCPT Cross-Encoder")
    print("=" * 90)

    for query in SAMPLE_QUERIES:
        run_comparison(retriever, query)
        print("─" * 90)


def run_patient_query(retriever: HybridRetriever):
    """Run patient-specific queries."""
    with engine.connect() as conn:
        result = conn.execute(sa_text(
            "SELECT subject_id FROM clinical_notes GROUP BY subject_id HAVING COUNT(*) > 2 LIMIT 1"
        ))
        row = result.fetchone()
        if not row:
            print("No patients with multiple notes found.")
            return
        subject_id = row[0]

    print("\n" + "=" * 90)
    print(f"  PATIENT-SPECIFIC QUERIES (subject_id={subject_id})")
    print("=" * 90)

    patient_queries = [
        "medications prescribed on discharge",
        "history of present illness",
        "lab results during hospitalization",
    ]
    for query in patient_queries:
        run_comparison(retriever, query, subject_id=subject_id)
        print("─" * 90)


def run_interactive(retriever: HybridRetriever):
    """Interactive mode."""
    print("=" * 90)
    print("  INTERACTIVE MODE — RRF vs RRF + MedCPT Cross-Encoder")
    print("  Type a query. Prefix 'p:12345 ' for patient filter. 'quit' to exit.")
    print("=" * 90)

    while True:
        try:
            user_input = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        subject_id = None
        query = user_input

        if query.startswith("p:"):
            parts = query.split(" ", 1)
            try:
                subject_id = int(parts[0][2:])
                query = parts[1] if len(parts) > 1 else ""
            except ValueError:
                print("Invalid patient ID. Use 'p:12345 your query'")
                continue

        if not query:
            print("Please enter a query.")
            continue

        t0 = time.time()
        run_comparison(retriever, query, subject_id=subject_id)
        print(f"  ⏱ {time.time() - t0:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Test Hybrid Retriever v2")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--no-reranker", action="store_true", help="Skip MedCPT Cross-Encoder")
    parser.add_argument("--patient", "-p", action="store_true", help="Run patient queries")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("Loading models (MedCPT + MedCPT Cross-Encoder)...\n")
    retriever = HybridRetriever(use_reranker=not args.no_reranker)
    print()

    if args.interactive:
        run_interactive(retriever)
    elif args.patient:
        run_patient_query(retriever)
    else:
        run_sample_queries(retriever)
        run_patient_query(retriever)


if __name__ == "__main__":
    main()
