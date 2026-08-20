"""
Hybrid Retriever v2 Test Runner
=================================
Tests the improved retrieval pipeline with clinical queries.

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    python -m src.retrieval.test_retriever

    # Interactive mode:
    python -m src.retrieval.test_retriever --interactive

    # Without reranker (faster):
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
)

logger = logging.getLogger(__name__)


SAMPLE_QUERIES = [
    {
        "query": "fluid overload swollen legs",
        "description": "Tests query expansion: 'swollen legs' → edema, peripheral edema, etc.",
    },
    {
        "query": "heart failure medications",
        "description": "Should find HF drugs: furosemide, lisinopril, carvedilol, etc.",
    },
    {
        "query": "patient who stopped breathing and needed a tube",
        "description": "Plain language → should find intubation/ventilator notes via expansion",
    },
    {
        "query": "blood sugar out of control",
        "description": "Tests expansion: 'blood sugar' → glucose, DKA, HbA1c, etc.",
    },
    {
        "query": "sepsis antibiotics blood cultures",
        "description": "Clinical jargon query — BM25 should dominate, vector confirms",
    },
]


def get_sample_patient() -> int | None:
    """Get a subject_id that has notes in the database."""
    with engine.connect() as conn:
        result = conn.execute(sa_text(
            "SELECT subject_id FROM clinical_notes GROUP BY subject_id HAVING COUNT(*) > 2 LIMIT 1"
        ))
        row = result.fetchone()
        return row[0] if row else None


def print_result(i: int, r: RetrievalResult, show_context: bool = True):
    """Pretty-print a single retrieval result."""
    source_str = ", ".join(r.sources)
    print(f"\n  Result #{i + 1}  [sources: {source_str}]")
    print(f"  {'─' * 64}")
    print(f"  Scores → final={r.final_score:.3f}  rerank={r.rerank_score:.3f}  rrf={r.rrf_score:.4f}  bm25={r.bm25_score:.3f}  vec={r.vector_score:.3f}")
    print(f"  Note   → {r.note_type} | subject={r.subject_id} | hadm={r.hadm_id}")
    print(f"  Time   → {r.charttime}")
    print(f"  Tokens → {r.token_count}")

    # Show context text (assembled from adjacent chunks)
    display_text = r.context_text or r.chunk_text
    if show_context:
        preview = display_text[:500]
        if len(display_text) > 500:
            preview += "\n  [...truncated...]"
        print(f"  Context:")
        for line in preview.split("\n"):
            print(f"    {line}")


def run_sample_queries(retriever: HybridRetriever):
    """Run sample queries and display results."""
    print("=" * 70)
    print("  HYBRID RETRIEVER v2 — SAMPLE QUERIES")
    print("=" * 70)

    for i, sample in enumerate(SAMPLE_QUERIES):
        # Show query expansion
        _, expansions = expand_query(sample["query"])

        print(f"\n{'━' * 70}")
        print(f"  Query {i + 1}: \"{sample['query']}\"")
        print(f"  Expected: {sample['description']}")
        if expansions:
            exp_preview = ", ".join(expansions[:6])
            if len(expansions) > 6:
                exp_preview += f", ... (+{len(expansions) - 6} more)"
            print(f"  Expanded: {exp_preview}")
        print(f"{'━' * 70}")

        t0 = time.time()
        results = retriever.search(
            query=sample["query"],
            top_k=3,
        )
        elapsed = time.time() - t0

        print(f"  Found {len(results)} results in {elapsed:.2f}s")

        for j, r in enumerate(results):
            print_result(j, r)

    print(f"\n{'=' * 70}")


def run_patient_query(retriever: HybridRetriever):
    """Run a query filtered to a specific patient."""
    subject_id = get_sample_patient()
    if not subject_id:
        print("No patients found in database.")
        return

    print(f"\n{'━' * 70}")
    print(f"  PATIENT-SPECIFIC QUERY (subject_id={subject_id})")
    print(f"{'━' * 70}")

    query = "medications prescribed on discharge"
    print(f"  Query: \"{query}\"")

    t0 = time.time()
    results = retriever.search(
        query=query,
        subject_id=subject_id,
        top_k=5,
    )
    elapsed = time.time() - t0

    print(f"  Found {len(results)} results in {elapsed:.2f}s")
    for j, r in enumerate(results):
        print_result(j, r)


def run_interactive(retriever: HybridRetriever):
    """Interactive mode — type queries and see results."""
    print("=" * 70)
    print("  INTERACTIVE RETRIEVER v2")
    print("  Type a query and press Enter. Type 'quit' to exit.")
    print("  Prefix with 'p:12345 ' to filter by patient subject_id.")
    print("  Suffix with ' /recent' for temporal recency filter.")
    print("=" * 70)

    while True:
        print()
        try:
            user_input = input("  Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        # Parse patient filter
        subject_id = None
        temporal = "all"
        query = user_input

        if query.startswith("p:"):
            parts = query.split(" ", 1)
            try:
                subject_id = int(parts[0][2:])
                query = parts[1] if len(parts) > 1 else ""
            except ValueError:
                print("  Invalid patient ID. Use 'p:12345 your query'")
                continue

        # Parse temporal filter
        if query.endswith("/recent"):
            temporal = "recent"
            query = query[:-7].strip()
        elif query.endswith("/oldest"):
            temporal = "oldest_first"
            query = query[:-7].strip()

        if not query:
            print("  Please enter a query.")
            continue

        # Show expansions
        _, expansions = expand_query(query)
        if expansions:
            exp_preview = ", ".join(expansions[:6])
            if len(expansions) > 6:
                exp_preview += f", ... (+{len(expansions) - 6} more)"
            print(f"  📚 Expanded: {exp_preview}")

        t0 = time.time()
        results = retriever.search(
            query=query,
            subject_id=subject_id,
            temporal_filter=temporal,
            top_k=5,
        )
        elapsed = time.time() - t0

        if subject_id:
            print(f"  [Patient {subject_id}] ", end="")
        print(f"Found {len(results)} results in {elapsed:.2f}s")

        for j, r in enumerate(results):
            print_result(j, r)


def main():
    parser = argparse.ArgumentParser(description="Test Hybrid Retriever v2")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive query mode")
    parser.add_argument("--no-reranker", action="store_true", help="Skip BGE reranker (faster)")
    parser.add_argument("--patient", action="store_true", help="Run patient-specific query test")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("Loading retriever v2 (MedCPT + BGE models, ~15s)...\n")
    retriever = HybridRetriever(use_reranker=not args.no_reranker)

    if args.interactive:
        run_interactive(retriever)
    elif args.patient:
        run_patient_query(retriever)
    else:
        run_sample_queries(retriever)
        print()
        run_patient_query(retriever)


if __name__ == "__main__":
    main()
