"""
Lumen Retrieval Evaluation Harness
=====================================
Runs the golden dataset against multiple retriever configurations
and computes standard IR metrics for comparison.

Metrics computed:
  - Precision@K: fraction of top-K results that are relevant
  - Recall@K: fraction of relevant results found in top-K
  - MRR (Mean Reciprocal Rank): 1/rank of first relevant result
  - nDCG@K (Normalized Discounted Cumulative Gain): position-weighted relevance

Configurations compared:
  1. BM25 only
  2. Vector only
  3. Hybrid (RRF) — no reranker
  4. Hybrid + BGE reranker
  5. Hybrid + MedCPT cross-encoder

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    python -m src.evals.eval_retrieval

    # Test with fewer queries:
    python -m src.evals.eval_retrieval --limit 5

    # Specific config only:
    python -m src.evals.eval_retrieval --config rrf_only

    # Export results to JSON:
    python -m src.evals.eval_retrieval --export results.json
"""

from __future__ import annotations

import json
import time
import math
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
from sqlalchemy import text as sa_text

from src.storage import engine
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
from src.evals.golden_dataset import GOLDEN_QUERIES

logger = logging.getLogger(__name__)

# Reranker model paths
BGE_RERANKER_PATH = str(Path.home() / "Lumen" / "models" / "bge-reranker")
MEDCPT_RERANKER_PATH = str(Path.home() / "Lumen" / "models" / "medcpt-cross-encoder")
# Same reranker-confidence gate as HybridRetriever.search() — keep in sync.
RERANK_FALLBACK_THRESHOLD = 0.35


# ===========================================================================
# Relevance Judging
# ===========================================================================

def judge_relevance(chunk_text: str, criteria_groups: list[list[str]], irrelevance_signals: list[str] = None) -> bool:
    """
    Judge whether a chunk is relevant based on keyword criteria.

    A chunk is relevant if it contains at least one keyword from
    ANY criteria group. It's marked irrelevant if it matches
    an irrelevance signal AND has no criteria match.

    This is a heuristic — not perfect, but scalable and reproducible.
    """
    text_lower = chunk_text.lower()

    # Check irrelevance signals first
    if irrelevance_signals:
        for signal in irrelevance_signals:
            if signal.lower() in text_lower:
                # Only reject if no positive criteria match
                has_positive = False
                for group in criteria_groups:
                    for keyword in group:
                        if keyword.lower() in text_lower:
                            has_positive = True
                            break
                    if has_positive:
                        break
                if not has_positive:
                    return False

    # Check criteria: must match at least one keyword from any group
    for group in criteria_groups:
        for keyword in group:
            if keyword.lower() in text_lower:
                return True

    return False


# ===========================================================================
# Metrics
# ===========================================================================

def precision_at_k(relevance_list: list[bool], k: int) -> float:
    """Fraction of top-K results that are relevant."""
    top_k = relevance_list[:k]
    if not top_k:
        return 0.0
    return sum(top_k) / len(top_k)


def recall_at_k(relevance_list: list[bool], min_relevant: int, k: int) -> float:
    """Fraction of expected relevant results found in top-K."""
    found = sum(relevance_list[:k])
    if min_relevant <= 0:
        return 1.0
    return min(found / min_relevant, 1.0)


def mrr(relevance_list: list[bool]) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant result."""
    for i, rel in enumerate(relevance_list):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevance_list: list[bool], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    top_k = relevance_list[:k]
    if not top_k:
        return 0.0

    # DCG
    dcg = 0.0
    for i, rel in enumerate(top_k):
        if rel:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG (all relevant results at top)
    n_relevant = sum(top_k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0:
        return 0.0
    return dcg / idcg


@dataclass
class QueryResult:
    """Evaluation result for a single query."""
    query_id: str
    query: str
    category: str
    config: str
    num_results: int
    relevance: list[bool]
    precision_5: float
    recall_5: float
    mrr_score: float
    ndcg_5: float
    time_sec: float
    top_results: list[dict] = field(default_factory=list)


# ===========================================================================
# Retrieval Configurations
# ===========================================================================

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


# ===========================================================================
# Main Eval Runner
# ===========================================================================

def evaluate_config(
    config_name: str,
    run_fn,
    queries: list[dict],
    top_k: int = 5,
) -> list[QueryResult]:
    """Run evaluation for a single config across all golden queries."""
    results = []

    for q in queries:
        t0 = time.time()
        retrieved = run_fn(q["query"])
        elapsed = time.time() - t0

        # Judge relevance of each result
        relevance = []
        top_results = []
        for r in retrieved:
            text = r.context_text or r.chunk_text
            is_relevant = judge_relevance(
                text,
                q["relevance_criteria"],
                q.get("irrelevance_signals", []),
            )
            relevance.append(is_relevant)
            top_results.append({
                "chunk_id": r.chunk_id,
                "relevant": is_relevant,
                "score": round(r.final_score, 4),
                "sources": r.sources,
                "text_preview": " ".join(text.split())[:100],
            })

        # Compute metrics
        qr = QueryResult(
            query_id=q["id"],
            query=q["query"],
            category=q["category"],
            config=config_name,
            num_results=len(retrieved),
            relevance=relevance,
            precision_5=precision_at_k(relevance, top_k),
            recall_5=recall_at_k(relevance, q["min_relevant"], top_k),
            mrr_score=mrr(relevance),
            ndcg_5=ndcg_at_k(relevance, top_k),
            time_sec=elapsed,
            top_results=top_results,
        )
        results.append(qr)

    return results


def print_query_detail(qr: QueryResult):
    """Print detailed results for a single query."""
    print(f"\n  {qr.query_id}: \"{qr.query}\"")
    print(f"  P@5={qr.precision_5:.2f}  R@5={qr.recall_5:.2f}  MRR={qr.mrr_score:.2f}  nDCG@5={qr.ndcg_5:.2f}  ⏱{qr.time_sec:.1f}s")

    for i, r in enumerate(qr.top_results):
        rel_mark = "✅" if r["relevant"] else "❌"
        src = "+".join(r["sources"])
        print(f"    {i+1}. {rel_mark} [{r['score']:.4f}] [{src:<12}] {r['text_preview'][:70]}...")


def print_summary(all_results: dict[str, list[QueryResult]], top_k: int = 5):
    """Print the final comparison summary table."""
    print("\n" + "=" * 95)
    print("  EVALUATION SUMMARY")
    print("=" * 95)

    # Overall metrics per config
    print(f"\n  {'Config':<30} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'nDCG@5':>8} {'Avg Time':>9} {'Queries':>8}")
    print(f"  {'─' * 85}")

    for config_name, results in all_results.items():
        avg_p = np.mean([r.precision_5 for r in results])
        avg_r = np.mean([r.recall_5 for r in results])
        avg_mrr = np.mean([r.mrr_score for r in results])
        avg_ndcg = np.mean([r.ndcg_5 for r in results])
        avg_time = np.mean([r.time_sec for r in results])
        n = len(results)

        print(f"  {config_name:<30} {avg_p:>5.2f}  {avg_r:>5.2f}  {avg_mrr:>5.2f}  {avg_ndcg:>7.2f}  {avg_time:>7.2f}s  {n:>7}")

    # Per-category breakdown for each config
    categories = sorted(set(r.category for results in all_results.values() for r in results))

    for config_name, results in all_results.items():
        print(f"\n  {config_name} — by category:")
        print(f"    {'Category':<20} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'nDCG@5':>8} {'n':>4}")
        print(f"    {'─' * 55}")

        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            if not cat_results:
                continue
            avg_p = np.mean([r.precision_5 for r in cat_results])
            avg_r = np.mean([r.recall_5 for r in cat_results])
            avg_mrr = np.mean([r.mrr_score for r in cat_results])
            avg_ndcg = np.mean([r.ndcg_5 for r in cat_results])
            n = len(cat_results)
            print(f"    {cat:<20} {avg_p:>5.2f}  {avg_r:>5.2f}  {avg_mrr:>5.2f}  {avg_ndcg:>7.2f}  {n:>3}")

    # Winner analysis
    print(f"\n  {'─' * 85}")
    print(f"  HEAD-TO-HEAD: Per-query best config (by nDCG@5)")
    print(f"    {'Config':<30} {'Wins':>6}")
    print(f"    {'─' * 40}")

    wins = {name: 0 for name in all_results}
    query_ids = set(r.query_id for results in all_results.values() for r in results)
    for qid in query_ids:
        best_config = None
        best_ndcg = -1
        for config_name, results in all_results.items():
            for r in results:
                if r.query_id == qid and r.ndcg_5 > best_ndcg:
                    best_ndcg = r.ndcg_5
                    best_config = config_name
        if best_config:
            wins[best_config] += 1

    for config_name, win_count in sorted(wins.items(), key=lambda x: -x[1]):
        bar = "█" * win_count
        print(f"    {config_name:<30} {win_count:>5}  {bar}")

    print()


def run_eval(
    configs: list[str] = None,
    limit: Optional[int] = None,
    top_k: int = 5,
    verbose: bool = True,
    export_path: Optional[str] = None,
):
    """Main evaluation runner."""
    print("=" * 95)
    print("  LUMEN RETRIEVAL EVALUATION HARNESS")
    print("=" * 95)

    available_configs = ["bm25_only", "vector_only", "rrf_only", "rrf_bge", "rrf_medcpt"]
    if configs:
        active_configs = [c for c in configs if c in available_configs]
    else:
        active_configs = available_configs

    queries = GOLDEN_QUERIES[:limit] if limit else GOLDEN_QUERIES
    print(f"\n  Queries: {len(queries)}")
    print(f"  Configs: {', '.join(active_configs)}")
    print(f"  Top-K:   {top_k}")

    # Load models
    print("\n  Loading models...")
    embedder = MedCPTEmbedder()

    bge_reranker = None
    medcpt_reranker = None

    if "rrf_bge" in active_configs:
        try:
            bge_reranker = BGEReranker(model_path=BGE_RERANKER_PATH)
        except Exception as e:
            print(f"  ⚠ BGE reranker not available: {e}")
            active_configs.remove("rrf_bge")

    if "rrf_medcpt" in active_configs:
        try:
            medcpt_reranker = BGEReranker(model_path=MEDCPT_RERANKER_PATH)
        except Exception as e:
            print(f"  ⚠ MedCPT cross-encoder not available: {e}")
            active_configs.remove("rrf_medcpt")

    # Build run functions
    run_fns = {}
    if "bm25_only" in active_configs:
        run_fns["BM25 Only"] = lambda q: run_bm25_only(q, embedder, top_k)
    if "vector_only" in active_configs:
        run_fns["Vector Only (MedCPT)"] = lambda q: run_vector_only(q, embedder, top_k)
    if "rrf_only" in active_configs:
        run_fns["Hybrid RRF"] = lambda q: run_rrf_only(q, embedder, top_k)
    if "rrf_bge" in active_configs and bge_reranker:
        run_fns["Hybrid + BGE Reranker"] = lambda q: run_rrf_plus_reranker(q, embedder, bge_reranker, top_k)
    if "rrf_medcpt" in active_configs and medcpt_reranker:
        run_fns["Hybrid + MedCPT Cross-Enc"] = lambda q: run_rrf_plus_reranker(q, embedder, medcpt_reranker, top_k)

    # Run evaluations
    all_results: dict[str, list[QueryResult]] = {}

    for config_name, run_fn in run_fns.items():
        print(f"\n{'─' * 95}")
        print(f"  Evaluating: {config_name}")
        print(f"{'─' * 95}")

        results = evaluate_config(config_name, run_fn, queries, top_k)
        all_results[config_name] = results

        if verbose:
            for qr in results:
                print_query_detail(qr)

        # Print config summary
        avg_p = np.mean([r.precision_5 for r in results])
        avg_mrr = np.mean([r.mrr_score for r in results])
        print(f"\n  → {config_name}: avg P@5={avg_p:.2f}, avg MRR={avg_mrr:.2f}")

    # Print final comparison
    print_summary(all_results, top_k)

    # Export
    if export_path:
        export_data = {}
        for config_name, results in all_results.items():
            export_data[config_name] = {
                "overall": {
                    "precision_5": float(np.mean([r.precision_5 for r in results])),
                    "recall_5": float(np.mean([r.recall_5 for r in results])),
                    "mrr": float(np.mean([r.mrr_score for r in results])),
                    "ndcg_5": float(np.mean([r.ndcg_5 for r in results])),
                    "avg_time": float(np.mean([r.time_sec for r in results])),
                },
                "queries": [
                    {
                        "id": r.query_id,
                        "query": r.query,
                        "category": r.category,
                        "precision_5": r.precision_5,
                        "recall_5": r.recall_5,
                        "mrr": r.mrr_score,
                        "ndcg_5": r.ndcg_5,
                        "time_sec": r.time_sec,
                        "results": r.top_results,
                    }
                    for r in results
                ],
            }

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"  Results exported to {export_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Lumen Retrieval Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--config", type=str, nargs="+", default=None,
                        help="Configs to test: bm25_only vector_only rrf_only rrf_bge rrf_medcpt")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for metrics (default 5)")
    parser.add_argument("--export", type=str, default=None, help="Export results to JSON file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Hide per-query details")
    args = parser.parse_args()

    run_eval(
        configs=args.config,
        limit=args.limit,
        top_k=args.top_k,
        verbose=not args.quiet,
        export_path=args.export,
    )
