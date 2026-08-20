"""
Two-phase eval — PHASE 1: RETRIEVAL ONLY
=========================================
Loads MedCPT + the reranker(s), runs every config over the golden queries to
`pool_k` depth, and writes the ranked chunks to a JSON file. Then it exits,
freeing all torch models — so PHASE 2 (judging) can load the 14B judge alone.
This split is what makes a 14B judge viable on a 16 GB machine: the retrieval
models and the judge never sit in memory at the same time.

Usage:
    cd ~/Lumen
    source .venv/bin/activate

    python -m src.evals.retrieve_pool --out pooled.json --pool-k 10
    python -m src.evals.retrieve_pool --out pooled.json --config rrf_only rrf_bge --limit 5
"""
from __future__ import annotations

import json
import time
import logging
import argparse
from pathlib import Path

from src.retrieval.embeddings import MedCPTEmbedder
from src.retrieval.hybrid_retriever_v2 import BGEReranker
from src.evals.eval_retrieval import (
    run_bm25_only,
    run_vector_only,
    run_rrf_only,
    run_rrf_plus_reranker,
    BGE_RERANKER_PATH,
    MEDCPT_RERANKER_PATH,
)
from src.evals.golden_dataset import GOLDEN_QUERIES

logger = logging.getLogger(__name__)


def _serialize(r) -> dict:
    """Everything phase 2 needs: chunk_text (judged) + display/metric fields."""
    return {
        "chunk_id": r.chunk_id,
        "note_id": r.note_id,
        "subject_id": r.subject_id,
        "hadm_id": r.hadm_id,
        "note_type": r.note_type,
        "chunk_index": r.chunk_index,
        "charttime": r.charttime,
        "chunk_text": r.chunk_text,          # bare chunk -> this is what gets judged
        "context_text": r.context_text,      # assembled window (optional/display)
        "final_score": round(float(r.final_score), 6),
        "sources": list(r.sources),
    }


def run_phase1(out_path: str, configs=None, limit=None, pool_k: int = 10):
    available = ["bm25_only", "vector_only", "rrf_only", "rrf_bge", "rrf_medcpt"]
    active = [c for c in (configs or available) if c in available]
    queries = GOLDEN_QUERIES[:limit] if limit else GOLDEN_QUERIES

    print("=" * 90)
    print(f"  PHASE 1 — retrieval to depth {pool_k}")
    print("=" * 90)
    print(f"  queries: {len(queries)}   configs: {', '.join(active)}")

    print("\n  Loading retrieval models...")
    embedder = MedCPTEmbedder()

    bge = medcpt = None
    if "rrf_bge" in active:
        try:
            bge = BGEReranker(model_path=BGE_RERANKER_PATH)
        except Exception as e:
            print(f"  ⚠ BGE reranker unavailable: {e}")
            active.remove("rrf_bge")
    if "rrf_medcpt" in active:
        try:
            medcpt = BGEReranker(model_path=MEDCPT_RERANKER_PATH)
        except Exception as e:
            print(f"  ⚠ MedCPT cross-encoder unavailable: {e}")
            active.remove("rrf_medcpt")

    run_fns = {}
    if "bm25_only" in active:
        run_fns["BM25 Only"] = lambda q: run_bm25_only(q, embedder, pool_k)
    if "vector_only" in active:
        run_fns["Vector Only (MedCPT)"] = lambda q: run_vector_only(q, embedder, pool_k)
    if "rrf_only" in active:
        run_fns["Hybrid RRF"] = lambda q: run_rrf_only(q, embedder, pool_k)
    if "rrf_bge" in active and bge:
        run_fns["Hybrid + BGE Reranker"] = lambda q: run_rrf_plus_reranker(q, embedder, bge, pool_k)
    if "rrf_medcpt" in active and medcpt:
        run_fns["Hybrid + MedCPT Cross-Enc"] = lambda q: run_rrf_plus_reranker(q, embedder, medcpt, pool_k)

    payload = {"_meta": {"pool_k": pool_k, "configs": list(run_fns.keys())}, "queries": []}

    t0 = time.time()
    for qi, q in enumerate(queries):
        entry = {
            "query_id": q["id"], "query": q["query"], "category": q["category"],
            "configs": {}, "timings": {},
        }
        for name, fn in run_fns.items():
            ts = time.time()
            results = fn(q["query"])
            entry["timings"][name] = round(time.time() - ts, 3)
            entry["configs"][name] = [_serialize(r) for r in results]
        payload["queries"].append(entry)
        print(f"  [{qi + 1:>2}/{len(queries)}] {q['id']}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Wrote {out}  ({time.time() - t0:.1f}s).")
    print("  Retrieval models can now be released — quit this process before phase 2.")
    print(f"  Next:  python -m src.evals.judge_and_score --in {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Phase 1 — retrieval only, writes pooled chunks to JSON")
    ap.add_argument("--out", type=str, default="pooled.json", help="Output JSON path")
    ap.add_argument("--config", type=str, nargs="+", default=None,
                    help="Configs: bm25_only vector_only rrf_only rrf_bge rrf_medcpt")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    ap.add_argument("--pool-k", type=int, default=10, help="Retrieval/pool depth (default 10)")
    args = ap.parse_args()
    run_phase1(args.out, configs=args.config, limit=args.limit, pool_k=args.pool_k)
