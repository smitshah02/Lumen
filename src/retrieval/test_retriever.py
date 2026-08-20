"""
Hybrid Retriever Test Runner
==============================
Tests the full retrieval pipeline with sample clinical queries.

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    python -m src.retrieval.test_retriever

    # Interactive mode — type your own queries:
    python -m src.retrieval.test_retriever --interactive

    # Without reranker (faster, for quick testing):
    python -m src.retrieval.test_retriever --no-reranker
"""

from __future__ import annotations

import sys
import time
import logging
import argparse

from sqlalchemy import text as sa_text
from src.storage import engine
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


# Sample clinical queries for testing
SAMPLE_QUERIES = [
    {
        "query": "heart failure medications",
        "description": "Should find discharge notes mentioning HF drugs (furosemide, lisinopril, carvedilol, etc.)",
    },
    {
        "query": "abnormal potassium lab results",
        "description": "Should find notes discussing hyperkalemia or hypokalemia",
    },
    {
        "query": "diabetes management insulin dosing",
        "description": "Should find notes about diabetic patients and their insulin regimens",
    },
    {
        "query": "chest pain differential diagnosis",
        "description": "Should find ED notes or discharge summaries discussing chest pain workup",
    },
    {
        "query": "ventilator settings respiratory failure",
        "description": "Should find ICU-related notes about mechanical ventilation",
    },
]


def get_sample_patient() -> int | None:
    """Get a subject_id that has notes in the database."""
    with engine.connect() as conn:
        result = conn.execute(sa_text(
            "SELECT subject_id FROM clinical_notes LIMIT 1"
        ))
        row = result.fetchone()
        return row[0] if row else None


def print_result(i: int, r: RetrievalResult, show_text: bool = True):
    """Pretty-print a single retrieval result."""
    print(f"\n  Result #{i + 1}")
    print(f"  {'─' * 60}")
    print(f"  Score:     {r.final_score:.4f}  (bm25={r.bm25_score:.3f}, vec={r.vector_score:.3f}, rrf={r.rrf_score:.4f}, rerank={r.rerank_score:.3f})")
    print(f"  Sources:   {', '.join(r.sources)}")
    print(f"  Note:      {r.note_type} | subject={r.subject_id} | hadm={r.hadm_id} | charttime={r.charttime}")
    print(f"  Chunk:     #{r.chunk_index} ({r.token_count} tokens)")
    if show_text:
        # Truncate long texts
        text = r.chunk_text[:400]
        if len(r.chunk_text) > 400:
            text += "..."
        print(f"  Text:      {text}")


def run_sample_queries(retriever: HybridRetriever):
    """Run sample queries and display results."""
    print("=" * 70)
    print("  HYBRID RETRIEVER — SAMPLE QUERIES")
    print("=" * 70)

    for i, sample in enumerate(SAMPLE_QUERIES):
        print(f"\n{'━' * 70}")
        print(f"  Query {i + 1}: \"{sample['query']}\"")
        print(f"  Expected: {sample['description']}")
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
    print("  INTERACTIVE RETRIEVER")
    print("  Type a query and press Enter. Type 'quit' to exit.")
    print("  Prefix with 'p:12345 ' to filter by patient subject_id.")
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
        query = user_input
        if user_input.startswith("p:"):
            parts = user_input.split(" ", 1)
            try:
                subject_id = int(parts[0][2:])
                query = parts[1] if len(parts) > 1 else ""
            except ValueError:
                print("  Invalid patient ID format. Use 'p:12345 your query'")
                continue

        if not query:
            print("  Please enter a query.")
            continue

        t0 = time.time()
        results = retriever.search(
            query=query,
            subject_id=subject_id,
            top_k=5,
        )
        elapsed = time.time() - t0

        if subject_id:
            print(f"  [Patient {subject_id}] ", end="")
        print(f"Found {len(results)} results in {elapsed:.2f}s")

        for j, r in enumerate(results):
            print_result(j, r)


def main():
    parser = argparse.ArgumentParser(description="Test the Hybrid Retriever")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive query mode")
    parser.add_argument("--no-reranker", action="store_true", help="Skip BGE reranker (faster)")
    parser.add_argument("--patient", action="store_true", help="Run patient-specific query test")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("Loading retriever (this loads MedCPT + BGE models, ~15s)...\n")
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
