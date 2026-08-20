"""
Two-phase eval — PHASE 2: JUDGING + SCORING  (local Ollama, no torch)
=====================================================================
Reads the JSON written by retrieve_pool.py, pools each query's chunks, grades
the union ONCE with a local Qwen2.5-14B judge via Ollama, and scores every
config against the shared relevant-set (pooled recall, graded nDCG).

This module deliberately imports NO torch / transformers — it only needs the
pure-Python judge + metrics — so while it runs, Qwen is the only large thing in
memory. Run it as a separate process, AFTER retrieve_pool.py has exited.

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    ollama serve            # (in another terminal, if not already running)

    python -m src.evals.judge_and_score --in pooled.json
    python -m src.evals.judge_and_score --in pooled.json --model qwen2.5:14b \
        --threshold 2 --top-k 5 --export results.json
"""
from __future__ import annotations

import json
import argparse
import logging
from types import SimpleNamespace

import numpy as np

from src.evals.llm_judge import (
    build_pooled_relevance,
    ndcg_at_k_graded,
    recall_at_k_pooled,
)
from src.evals.ollama_backend import make_ollama_judge, DEFAULT_OLLAMA_MODEL

logger = logging.getLogger(__name__)


# --- tiny metrics, re-defined here so this file never imports torch ---
def precision_at_k(binary: list, k: int) -> float:
    top = binary[:k]
    return (sum(1 for x in top if x) / len(top)) if top else 0.0


def mrr(binary: list) -> float:
    for i, x in enumerate(binary):
        if x:
            return 1.0 / (i + 1)
    return 0.0


def load_configs_results(entry: dict) -> dict:
    """Rebuild lightweight records; build_pooled_relevance only needs
    .chunk_id and .chunk_text, but we keep the rest for display."""
    out = {}
    for name, chunks in entry["configs"].items():
        out[name] = [SimpleNamespace(**c) for c in chunks]
    return out


def score_query(entry: dict, judge, threshold: int, top_k: int) -> tuple[dict, int]:
    configs_results = load_configs_results(entry)
    relevant_ids, grades, per_config = build_pooled_relevance(
        entry["query"], configs_results, judge, threshold=threshold
    )
    pool_grades = list(grades.values())
    n_rel = len(relevant_ids)

    scored = {}
    for name in configs_results:
        binary = per_config[name]["binary"]
        graded = per_config[name]["graded"]
        scored[name] = {
            "precision": precision_at_k(binary, top_k),
            "recall": recall_at_k_pooled(binary, n_rel, top_k),
            "mrr": mrr(binary[:top_k]),
            "ndcg": ndcg_at_k_graded(graded, pool_grades, top_k),
            "binary": binary[:top_k],
            "graded": graded[:top_k],
        }
    return scored, n_rel


def run_phase2(in_path: str, model: str, threshold: int, top_k: int,
               export_path: str = None, verbose: bool = True):
    with open(in_path) as f:
        payload = json.load(f)
    queries = payload["queries"]
    meta = payload.get("_meta", {})

    print("=" * 90)
    print("  PHASE 2 — local judge + scoring")
    print("=" * 90)
    print(f"  input:  {in_path}   ({len(queries)} queries, pool_k={meta.get('pool_k')})")
    print(f"  judge:  Ollama {model}, threshold>={threshold}, metrics@{top_k}")

    judge = make_ollama_judge(model=model)

    agg: dict[str, list] = {name: [] for name in meta.get("configs", [])}
    for qi, entry in enumerate(queries):
        scored, n_rel = score_query(entry, judge, threshold, top_k)
        for name, s in scored.items():
            agg.setdefault(name, []).append({
                "query_id": entry["query_id"], "category": entry["category"],
                "n_relevant_pool": n_rel,
                "precision": s["precision"], "recall": s["recall"],
                "mrr": s["mrr"], "ndcg": s["ndcg"],
                "binary": s["binary"], "graded": s["graded"],
            })
        if verbose:
            print(f"  [{qi + 1:>2}/{len(queries)}] {entry['query_id']:<12} pool_rel={n_rel}")

    # ---- summary ----
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print(f"\n  {'Config':<30} {'P@k':>6} {'R@k':>6} {'MRR':>6} {'nDCG@k':>8} {'Queries':>8}")
    print(f"  {'-' * 76}")
    for name, rows in agg.items():
        if not rows:
            continue
        p = np.mean([r["precision"] for r in rows])
        rc = np.mean([r["recall"] for r in rows])
        mr = np.mean([r["mrr"] for r in rows])
        nd = np.mean([r["ndcg"] for r in rows])
        print(f"  {name:<30} {p:>5.2f}  {rc:>5.2f}  {mr:>5.2f}  {nd:>7.2f}  {len(rows):>7}")

    # per-category, per config
    cats = sorted({r["category"] for rows in agg.values() for r in rows})
    for name, rows in agg.items():
        if not rows:
            continue
        print(f"\n  {name} — by category:")
        print(f"    {'Category':<20} {'P@k':>6} {'R@k':>6} {'MRR':>6} {'nDCG@k':>8} {'n':>4}")
        print(f"    {'-' * 55}")
        for cat in cats:
            cr = [r for r in rows if r["category"] == cat]
            if not cr:
                continue
            print(f"    {cat:<20} {np.mean([r['precision'] for r in cr]):>5.2f}  "
                  f"{np.mean([r['recall'] for r in cr]):>5.2f}  "
                  f"{np.mean([r['mrr'] for r in cr]):>5.2f}  "
                  f"{np.mean([r['ndcg'] for r in cr]):>7.2f}  {len(cr):>3}")

    if export_path:
        with open(export_path, "w") as f:
            json.dump({"_meta": {"model": model, "threshold": threshold, "top_k": top_k},
                       "configs": agg}, f, indent=2)
        print(f"\n  exported -> {export_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Phase 2 — pooled local-LLM judging + scoring")
    ap.add_argument("--in", dest="in_path", type=str, default="pooled.json", help="Input JSON from phase 1")
    ap.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL, help="Ollama model tag")
    ap.add_argument("--threshold", type=int, default=2, help="Min grade (0-3) counted relevant")
    ap.add_argument("--top-k", type=int, default=5, help="Top-K for metrics")
    ap.add_argument("--export", type=str, default=None, help="Export results to JSON")
    ap.add_argument("--quiet", "-q", action="store_true", help="Hide per-query progress")
    args = ap.parse_args()
    run_phase2(args.in_path, model=args.model, threshold=args.threshold,
               top_k=args.top_k, export_path=args.export, verbose=not args.quiet)
