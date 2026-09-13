"""
RRF weight sweep
================
Why this exists: `vector_weight=1.2 > bm25_weight=1.0` was tuned while the
vector branch was returning ~1 result per query (pgvector's default
hnsw.ef_search=40 combined with HNSW post-filtering). With that fixed, the
vector arm returns a full 60 and the old weights over-trust it — the baseline
put Hybrid RRF at P@5 0.60 against BM25-only's 0.73.

Method. bm25_search and vector_search do not depend on the fusion weights, so
each query is retrieved ONCE and every weight combination is fused in memory
from those cached candidate lists. Relevance is pooled TREC-style across all
combinations and judged once, so no combination is scored against a pool it
alone defined.

    python -m src.evals.sweep_rrf                    # default grid
    python -m src.evals.sweep_rrf --vector-weights 0.4 0.8 1.2 --top-k 5

Needs Postgres (retrieval) AND Ollama (judging) in one process, unlike the
two-phase pipeline. The candidate cache keeps it to one retrieval pass.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from src.retrieval.embeddings import MedCPTEmbedder
from src.retrieval.hybrid_retriever_v2 import (
    bm25_search,
    vector_search,
    reciprocal_rank_fusion,
    deduplicate_by_note,
    apply_temporal_filter,
    expand_query,
)
from src.evals.golden_dataset import GOLDEN_QUERIES
from src.evals.llm_judge import (
    build_pooled_relevance,
    ndcg_at_k_graded,
    recall_at_k_pooled,
)
from src.evals.ollama_backend import make_ollama_judge, DEFAULT_OLLAMA_MODEL

logger = logging.getLogger(__name__)


def precision_at_k(binary, k):
    top = binary[:k]
    return (sum(1 for x in top if x) / len(top)) if top else 0.0


def mrr(binary):
    for i, x in enumerate(binary):
        if x:
            return 1.0 / (i + 1)
    return 0.0


def retrieve_once(queries, top_n: int, min_tokens: int):
    """One retrieval pass per query; both arms cached for every weight combo."""
    embedder = MedCPTEmbedder()
    cache = {}
    for i, q in enumerate(queries, 1):
        _, expansions = expand_query(q["query"])
        bm25 = bm25_search(query=q["query"], expansions=expansions,
                           top_n=top_n, min_tokens=min_tokens)
        vec = vector_search(query_embedding=embedder.embed_query(q["query"]),
                            top_n=top_n, min_tokens=min_tokens)
        cache[q["id"]] = (bm25, vec)
        print(f"  [{i:>2}/{len(queries)}] {q['id']:<14} bm25={len(bm25):>3} vec={len(vec):>3}")
    return cache


def fuse(bm25, vec, bm25_w, vec_w, overlap, max_per_note, top_k):
    merged = reciprocal_rank_fusion(
        bm25_results=bm25, vector_results=vec,
        bm25_weight=bm25_w, vector_weight=vec_w, overlap_bonus=overlap,
    )
    merged = deduplicate_by_note(merged, max_per_note=max_per_note)
    merged = apply_temporal_filter(merged, mode="all")
    for r in merged:
        r.final_score = r.rrf_score
    return merged[:top_k]


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep RRF fusion weights")
    ap.add_argument("--vector-weights", type=float, nargs="+",
                    default=[0.4, 0.6, 0.8, 1.0, 1.2, 1.5])
    ap.add_argument("--bm25-weight", type=float, default=1.0)
    ap.add_argument("--overlap-bonus", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--min-tokens", type=int, default=40)
    ap.add_argument("--max-per-note", type=int, default=2)
    ap.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    ap.add_argument("--threshold", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    queries = GOLDEN_QUERIES[:args.limit] if args.limit else GOLDEN_QUERIES

    print("=" * 88)
    print("  RRF WEIGHT SWEEP")
    print("=" * 88)
    print(f"  queries={len(queries)}  bm25_weight={args.bm25_weight}  "
          f"overlap_bonus={args.overlap_bonus}  top_k={args.top_k}")
    print(f"  vector weights: {args.vector_weights}\n")

    print("Retrieval pass (once per query; weights are applied afterwards):")
    cache = retrieve_once(queries, args.top_n, args.min_tokens)

    judge = make_ollama_judge(model=args.model)
    combos = [(args.bm25_weight, vw, args.overlap_bonus) for vw in args.vector_weights]

    agg = {vw: {"p": [], "r": [], "m": [], "n": []} for _, vw, _ in combos}
    print("\nFusing + judging:")
    for qi, q in enumerate(queries, 1):
        bm25, vec = cache[q["id"]]
        per_combo = {}
        for bw, vw, ob in combos:
            per_combo[vw] = fuse(bm25, vec, bw, vw, ob, args.max_per_note, args.top_k)

        # Pool across every combination, judge the union once.
        relevant_ids, grades, per_config = build_pooled_relevance(
            q["query"], {str(vw): res for vw, res in per_combo.items()},
            judge, threshold=args.threshold,
        )
        pool_grades = list(grades.values())
        n_rel = len(relevant_ids)
        for vw in per_combo:
            b = per_config[str(vw)]["binary"]
            g = per_config[str(vw)]["graded"]
            agg[vw]["p"].append(precision_at_k(b, args.top_k))
            agg[vw]["r"].append(recall_at_k_pooled(b, n_rel, args.top_k))
            agg[vw]["m"].append(mrr(b[:args.top_k]))
            agg[vw]["n"].append(ndcg_at_k_graded(g, pool_grades, args.top_k))
        print(f"  [{qi:>2}/{len(queries)}] {q['id']:<14} pool_rel={n_rel}")

    print("\n" + "=" * 88)
    print("  RESULTS  (bm25_weight fixed at %.2f)" % args.bm25_weight)
    print("=" * 88)
    print(f"\n  {'vector_weight':>14} {'P@k':>7} {'R@k':>7} {'MRR':>7} {'nDCG@k':>8}")
    print(f"  {'-' * 50}")
    best, best_nd = None, -1.0
    for vw in args.vector_weights:
        a = agg[vw]
        nd = float(np.mean(a["n"]))
        marker = ""
        if nd > best_nd:
            best_nd, best = nd, vw
        print(f"  {vw:>14.2f} {np.mean(a['p']):>7.3f} {np.mean(a['r']):>7.3f} "
              f"{np.mean(a['m']):>7.3f} {nd:>8.3f}{marker}")
    print(f"\n  best nDCG@{args.top_k}: vector_weight={best} ({best_nd:.3f})")
    print(f"  shipped default is 1.2 -> {np.mean(agg[1.2]['n']):.3f}"
          if 1.2 in agg else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
