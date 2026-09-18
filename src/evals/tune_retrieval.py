"""
Retrieval tuning harness — dev/test split, two-phase (retrieve, then judge)
===========================================================================
Tunes fusion/rerank parameters on a DEVELOPMENT subset of the golden set and
evaluates one frozen choice on a HELD-OUT subset. Same judge, threshold, k and
pooled-relevance method as the canonical two-phase eval (retrieve_pool.py ->
judge_and_score.py); nothing here changes a metric definition.

Phase 1 `retrieve` runs BM25 (with and without query expansion) and MedCPT
vector search ONCE per query, then builds every variant in memory by re-fusing
those cached arms and reranking with BGE exactly as
retrieval_configs.run_rrf_plus_reranker does. Its output holds chunk_text for
judging — it is raw note text, so write it OUTSIDE the repo.

Phase 2 `score` pools every variant's top-`pool_depth` with the canonical five
configs' top-10 for the same query, judges the union once with the local
Ollama judge, and writes aggregate-only JSON (query ids, no text).

    python -m src.evals.tune_retrieval split    --out results/interview_tuning/split_manifest.json
    python -m src.evals.tune_retrieval retrieve --split results/interview_tuning/split_manifest.json \
        --set dev --out <scratch>/tune_dev.json
    python -m src.evals.tune_retrieval score    --in <scratch>/tune_dev.json \
        --canonical <scratch>/pooled.json --out <scratch>/tune_dev_scores.json

The test set can only be retrieved with --variants (the frozen selection plus
baselines); the dev grid is refused on it.
"""

from __future__ import annotations

import sys
import json
import math
import random
import argparse
import logging
from collections import defaultdict
from types import SimpleNamespace

logger = logging.getLogger(__name__)

SEED = 42
TEST_FRAC = 0.3
POOL_K = 10            # canonical pool depth / list length written per variant
TOP_K = 5
THRESHOLD = 2

# Current production / canonical-eval settings.
CURRENT = {"bm25_w": 1.0, "vec_w": 1.2, "k": 60, "ob": 0.5, "C": 40, "exp": True}

GRID_WEIGHTS = [(1.0, 1.2), (1.0, 1.0), (1.2, 0.8), (1.5, 0.75), (1.0, 0.5), (1.5, 0.5), (1.0, 0.0)]
GRID_DEPTHS = [10, 20, 30, 40, 50, 60]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def make_split(seed: int = SEED, test_frac: float = TEST_FRAC) -> dict:
    """Stratified by category; largest-remainder quotas; seeded tie-breaks.
    random.Random(str) seeding is independent of PYTHONHASHSEED."""
    from src.evals.golden_dataset import GOLDEN_QUERIES

    by_cat = defaultdict(list)
    for q in GOLDEN_QUERIES:
        by_cat[q["category"]].append(q["id"])
    n_test = round(test_frac * len(GOLDEN_QUERIES))

    quota = {c: test_frac * len(ids) for c, ids in by_cat.items()}
    alloc = {c: math.floor(v) for c, v in quota.items()}
    cats = sorted(by_cat)
    random.Random(f"{seed}:tiebreak").shuffle(cats)
    for c in sorted(cats, key=lambda c: -(quota[c] - alloc[c]))[: n_test - sum(alloc.values())]:
        alloc[c] += 1

    dev, test = [], []
    for c in sorted(by_cat):
        ids = sorted(by_cat[c])
        random.Random(f"{seed}:{c}").shuffle(ids)
        test += [{"query_id": i, "category": c} for i in sorted(ids[: alloc[c]])]
        dev += [{"query_id": i, "category": c} for i in sorted(ids[alloc[c]:])]
    return {
        "seed": seed, "test_frac": test_frac,
        "method": "stratified by category, largest-remainder quotas, seeded shuffle within category",
        "n_dev": len(dev), "n_test": len(test),
        "dev": dev, "test": test,
        "rule": "test queries are never used to select parameters; evaluated once with the frozen selection",
    }


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def vname(method: str, bm25_w=None, vec_w=None, k=None, ob=None, C=None, exp=True) -> str:
    e = "exp" if exp else "noexp"
    if method in ("bm25", "vector"):
        return f"{method}|{e}" if method == "bm25" else "vector"
    base = f"{method}|w{bm25_w:g}/{vec_w:g}|k{k}|ob{ob:g}|{e}"
    return base + (f"|C{C}" if method == "bge" else "")


def parse_variant(name: str) -> dict:
    parts = name.split("|")
    v = {"name": name, "method": parts[0], "exp": "noexp" not in parts}
    for p in parts[1:]:
        if p.startswith("w"):
            a, b = p[1:].split("/")
            v["bm25_w"], v["vec_w"] = float(a), float(b)
        elif p.startswith("k"):
            v["k"] = int(p[1:])
        elif p.startswith("ob"):
            v["ob"] = float(p[2:])
        elif p.startswith("C"):
            v["C"] = int(p[1:])
    return v


def dev_grid() -> list[str]:
    c = CURRENT
    names = [vname("bm25", exp=True), vname("bm25", exp=False), vname("vector")]
    for bw, vw in GRID_WEIGHTS:                                     # RRF weights
        names.append(vname("rrf", bw, vw, c["k"], c["ob"], exp=True))
        for C in GRID_DEPTHS:                                       # x rerank depth
            names.append(vname("bge", bw, vw, c["k"], c["ob"], C, exp=True))
    # expansion ablation at the current config
    names.append(vname("rrf", c["bm25_w"], c["vec_w"], c["k"], c["ob"], exp=False))
    names.append(vname("bge", c["bm25_w"], c["vec_w"], c["k"], c["ob"], c["C"], exp=False))
    return names


# ---------------------------------------------------------------------------
# Phase 1 — retrieve
# ---------------------------------------------------------------------------
def run_retrieve(split_path: str, which: str, out_path: str, variants: list[str] | None):
    from src.evals.golden_dataset import GOLDEN_QUERIES
    from src.retrieval.embeddings import MedCPTEmbedder
    from src.retrieval.hybrid_retriever_v2 import (
        BGEReranker, bm25_search, vector_search, reciprocal_rank_fusion,
        deduplicate_by_note, apply_temporal_filter, expand_context, expand_query,
    )
    from src.evals.retrieval_configs import BGE_RERANKER_PATH, RERANK_FALLBACK_THRESHOLD

    split = json.load(open(split_path))
    ids = {q["query_id"] for q in split[which]}
    if which == "test" and not variants:
        raise SystemExit("test set requires --variants (frozen selection + baselines); refusing the dev grid")
    names = variants or dev_grid()
    specs = [parse_variant(n) for n in names]
    queries = [q for q in GOLDEN_QUERIES if q["id"] in ids]

    embedder = MedCPTEmbedder()
    reranker = BGEReranker(model_path=BGE_RERANKER_PATH) if any(s["method"] == "bge" for s in specs) else None

    payload = {"_meta": {"set": which, "split": split_path, "pool_k": POOL_K, "variants": names},
               "queries": []}
    for qi, q in enumerate(queries, 1):
        _, expansions = expand_query(q["query"])
        arms = {
            True: bm25_search(query=q["query"], expansions=expansions, top_n=60, min_tokens=40),
            False: bm25_search(query=q["query"], expansions=[], top_n=60, min_tokens=40),
        }
        vec = vector_search(query_embedding=embedder.embed_query(q["query"]), top_n=60, min_tokens=40)

        texts, ranked, fallback = {}, {}, {}
        ctx_cache: dict = {}
        rerank_cache: dict = {}

        for s in specs:
            if s["method"] == "bm25":
                rows = arms[s["exp"]][:POOL_K]
                ranked[s["name"]] = [r["chunk_id"] for r in rows]
                texts.update({r["chunk_id"]: r["chunk_text"] for r in rows})
                continue
            if s["method"] == "vector":
                rows = vec[:POOL_K]
                ranked[s["name"]] = [r["chunk_id"] for r in rows]
                texts.update({r["chunk_id"]: r["chunk_text"] for r in rows})
                continue

            # Mirrors retrieval_configs.run_rrf_only / run_rrf_plus_reranker.
            merged = reciprocal_rank_fusion(
                bm25_results=arms[s["exp"]], vector_results=vec,
                k=s["k"], bm25_weight=s["bm25_w"], vector_weight=s["vec_w"], overlap_bonus=s["ob"],
            )
            merged = deduplicate_by_note(merged, max_per_note=2)
            merged = apply_temporal_filter(merged, mode="all", boost_recent=True)

            if s["method"] == "rrf":
                out = merged[:POOL_K]
            else:
                candidates = merged[: s["C"]]
                # expand_context depends only on the chunk; cache its result.
                todo = [r for r in candidates if r.chunk_id not in ctx_cache]
                if todo:
                    expand_context(todo, window=1, max_context_tokens=600)
                    ctx_cache.update({r.chunk_id: (r.context_text, r.token_count) for r in todo})
                for r in candidates:
                    r.context_text, r.token_count = ctx_cache[r.chunk_id]
                key = tuple(r.chunk_id for r in candidates)
                if key not in rerank_cache:
                    reranker.rerank(q["query"], candidates, top_k=POOL_K)   # sorts candidates in place
                    rerank_cache[key] = {r.chunk_id: r.rerank_score for r in candidates}
                scores = rerank_cache[key]
                for r in candidates:
                    r.rerank_score = r.final_score = scores[r.chunk_id]
                candidates.sort(key=lambda r: r.rerank_score, reverse=True)
                reranked = candidates[:POOL_K]
                max_score = max((r.rerank_score for r in reranked), default=0.0)
                fallback[s["name"]] = max_score < RERANK_FALLBACK_THRESHOLD
                if fallback[s["name"]]:
                    out = sorted(candidates, key=lambda r: r.rrf_score, reverse=True)[:POOL_K]
                else:
                    out = reranked
            ranked[s["name"]] = [r.chunk_id for r in out]
            texts.update({r.chunk_id: r.chunk_text for r in out})

        payload["queries"].append({"query_id": q["id"], "category": q["category"], "query": q["query"],
                                   "ranked": ranked, "fallback": fallback,
                                   "texts": {str(k): v for k, v in texts.items()}})
        print(f"  [{qi:>2}/{len(queries)}] {q['id']:<12} bm25={len(arms[True])} vec={len(vec)}", flush=True)

    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"  wrote {out_path}  ({len(names)} variants x {len(queries)} queries)")


# ---------------------------------------------------------------------------
# Phase 2 — judge + score
# ---------------------------------------------------------------------------
def _pools(entry: dict, canon: dict | None, pool_depth: int) -> dict:
    texts = {int(k): v for k, v in entry["texts"].items()}
    configs = {}
    if canon is not None:
        for name, rows in canon["configs"].items():
            configs[f"canonical:{name}"] = [SimpleNamespace(chunk_id=r["chunk_id"], chunk_text=r["chunk_text"])
                                            for r in rows[:POOL_K]]
    for name, ids in entry["ranked"].items():
        configs[name] = [SimpleNamespace(chunk_id=c, chunk_text=texts[c]) for c in ids[:pool_depth]]
    return configs


def run_score(in_path: str, canonical_path: str, out_path: str, pool_depth: int, estimate_only: bool):
    from src.evals.llm_judge import build_pooled_relevance, ndcg_at_k_graded, recall_at_k_pooled
    from src.evals.judge_and_score import precision_at_k, mrr
    from src.evals.ollama_backend import make_ollama_judge, DEFAULT_OLLAMA_MODEL

    payload = json.load(open(in_path))
    canon = {q["query_id"]: q for q in json.load(open(canonical_path))["queries"]}
    judge = make_ollama_judge(model=DEFAULT_OLLAMA_MODEL)

    if estimate_only:
        cache = judge._load_cache() if hasattr(judge, "_load_cache") else {}
        miss = 0
        for e in payload["queries"]:
            seen = set()
            for rows in _pools(e, canon.get(e["query_id"]), pool_depth).values():
                for r in rows:
                    if r.chunk_id not in seen:
                        seen.add(r.chunk_id)
                        miss += judge._cache_key(e["query"], r.chunk_text) not in cache
        print(f"uncached judgements needed at pool_depth={pool_depth}: {miss}")
        return

    per_query = defaultdict(dict)       # variant -> query_id -> metrics
    for qi, e in enumerate(payload["queries"], 1):
        configs = _pools(e, canon.get(e["query_id"]), pool_depth)
        relevant, grades, per = build_pooled_relevance(e["query"], configs, judge, threshold=THRESHOLD)
        pool_grades, n_rel = list(grades.values()), len(relevant)
        for name, d in per.items():
            b, g = d["binary"], d["graded"]
            per_query[name][e["query_id"]] = {
                "category": e["category"], "n_relevant_pool": n_rel,
                "precision": precision_at_k(b, TOP_K), "recall": recall_at_k_pooled(b, n_rel, TOP_K),
                "mrr": mrr(b[:TOP_K]), "ndcg": ndcg_at_k_graded(g, pool_grades, TOP_K),
                "fallback": e["fallback"].get(name),
            }
        print(f"  [{qi:>2}/{len(payload['queries'])}] {e['query_id']:<12} pool_rel={n_rel}", flush=True)

    def agg(rows):
        n = len(rows)
        out = {"n": n}
        for k, key in (("precision@5", "precision"), ("recall@5", "recall"), ("mrr", "mrr"), ("ndcg@5", "ndcg")):
            out[k] = round(sum(r[key] for r in rows) / n, 4)
        fb = [r["fallback"] for r in rows if r["fallback"] is not None]
        if fb:
            out["rerank_fallback_rate"] = round(sum(fb) / len(fb), 4)
        return out

    result = {"_meta": {**payload["_meta"], "judge_model": DEFAULT_OLLAMA_MODEL, "threshold": THRESHOLD,
                        "top_k": TOP_K, "pool_depth_variants": pool_depth,
                        "pool": "union of canonical 5 configs top-10 + every variant top-pool_depth"},
              "variants": {}}
    for name, qs in per_query.items():
        rows = list(qs.values())
        cats = sorted({r["category"] for r in rows})
        result["variants"][name] = {
            **agg(rows),
            "by_category": {c: agg([r for r in rows if r["category"] == c]) for c in cats},
            "per_query_ndcg": {qid: round(r["ndcg"], 4) for qid, r in sorted(qs.items())},
        }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Lumen retrieval tuning (dev/test split)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--out", required=True)
    r = sub.add_parser("retrieve")
    r.add_argument("--split", required=True)
    r.add_argument("--set", choices=["dev", "test"], required=True)
    r.add_argument("--variants", nargs="+", default=None, help="explicit variant names (required for test)")
    r.add_argument("--out", required=True)
    c = sub.add_parser("score")
    c.add_argument("--in", dest="in_path", required=True)
    c.add_argument("--canonical", required=True, help="canonical pooled JSON from retrieve_pool.py")
    c.add_argument("--out", required=True)
    c.add_argument("--pool-depth", type=int, default=POOL_K)
    c.add_argument("--estimate", action="store_true", help="only count uncached judgements")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s", stream=sys.stderr)
    if args.cmd == "split":
        with open(args.out, "w") as f:
            json.dump(make_split(), f, indent=2)
        print(f"  wrote {args.out}")
    elif args.cmd == "retrieve":
        run_retrieve(args.split, args.set, args.out, args.variants)
    else:
        run_score(args.in_path, args.canonical, args.out, args.pool_depth, args.estimate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
