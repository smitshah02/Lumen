"""Patient-scoped retrieval latency benchmark, by stage.

Runs a small fixed question set against several patients through
HybridRetriever.search exactly as the patient_retrieval node calls it, and
reports where the time goes.

    ./scripts/lumen research retrieval-bench --out ~/Lumen_local_results/retrieval/before.json
    ./scripts/lumen research retrieval-bench --out ~/Lumen_local_results/retrieval/after.json \\
        --compare ~/Lumen_local_results/retrieval/before.json

Prints aggregates only. The --out file holds per-query stage timings, chunk
ids and scores — no note text — and stays on this machine.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTIONS = {   # one per retrieval shape the graph actually sends
    "factual": "What medications was the patient discharged on?",
    "temporal_latest": "What did the most recent chest imaging show?",
    "longitudinal": "How did the patient's kidney function change across admissions?",
    "multi_evidence": "Summarize the main diagnoses and the treatments given during the hospital stays.",
}
STAGES = ("lexical_search", "query_embedding", "vector_search", "fusion", "temporal_processing",
          "context_expansion", "reranking")
TOP_K = 5        # graph.PATIENT_TOP_K


def _pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--subjects-file", default="~/Lumen_local_results/p0/manual_subjects.json",
                   help="local JSON list of subject ids")
    p.add_argument("--passes", type=int, default=2, help="warm passes over the whole set")
    p.add_argument("--out", required=True, help="local JSON file for raw results")
    p.add_argument("--compare", help="an earlier --out file: report how the top-5 changed")
    args = p.parse_args(argv)

    from src import storage
    if storage.DATA_PLANE != "research":
        print(f"refusing: this benchmark runs on the research plane, not {storage.DATA_PLANE!r}", file=sys.stderr)
        return 2
    import logging
    logging.disable(logging.INFO)
    from src.retrieval.hybrid_retriever_v2 import HybridRetriever, _parse_charttime, detect_temporal_mode

    sids = json.loads(Path(args.subjects_file).expanduser().read_text())
    t = time.perf_counter()
    retriever = HybridRetriever()
    load_s = time.perf_counter() - t

    def run(alias, sid, kind, pass_no):
        results = retriever.search(query=QUESTIONS[kind], subject_id=sid,
                                   temporal_filter=detect_temporal_mode(QUESTIONS[kind]), top_k=TOP_K)
        times = [_parse_charttime(r.charttime) for r in results]
        return {"patient": alias, "kind": kind, "pass": pass_no, **retriever.last_stages,
                "chunk_ids": [r.chunk_id for r in results],
                "scores": [round(float(r.final_score), 4) for r in results],
                "isolated": all(r.subject_id == sid for r in results),
                # "latest": newest first among the returned rows
                "temporal_ok": (kind != "temporal_latest"
                                or all(a and b and a >= b for a, b in zip(times, times[1:])))}

    rows = [dict(run("P1", sids[0], "factual", 0), cold=True)]          # first call after load
    for n in range(1, args.passes + 1):
        for i, sid in enumerate(sids, 1):
            rows += [run(f"P{i}", sid, kind, n) for kind in QUESTIONS]
    warm = [r for r in rows if not r.get("cold")]

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model_load_s": round(load_s, 2), "rows": rows}, indent=1))

    total = statistics.median(r["total"] for r in warm)
    print(f"model load {load_s:.1f}s | cold first search {rows[0]['total']:.0f} ms | "
          f"{len(warm)} warm searches over {len(sids)} patients")
    print(f"{'stage':20s} {'median':>8s} {'min':>8s} {'p95':>8s} {'max':>8s} {'% of total':>10s}")
    for name in STAGES + ("total",):
        v = [r.get(name, 0.0) for r in warm]
        print(f"{name:20s} {statistics.median(v):8.0f} {min(v):8.0f} {_pct(v, 0.95):8.0f} {max(v):8.0f} "
              f"{100 * statistics.median(v) / total:9.0f}%")
    print(f"candidates reranked: median {statistics.median(r['reranked'] for r in warm):.0f} "
          f"(min {min(r['reranked'] for r in warm)}, max {max(r['reranked'] for r in warm)}) | top_k {TOP_K}")
    for kind in QUESTIONS:
        v = [r["total"] for r in warm if r["kind"] == kind]
        print(f"  {kind:17s} median {statistics.median(v):6.0f} ms")
    print(f"patient isolation: {'PASS' if all(r['isolated'] for r in rows) else 'FAIL'} | "
          f"temporal order (latest): {'PASS' if all(r['temporal_ok'] for r in rows) else 'FAIL'}")

    if args.compare:
        before = {(r["patient"], r["kind"]): r for r in json.loads(Path(args.compare).expanduser().read_text())["rows"]
                  if r.get("pass") == 1}
        after = {(r["patient"], r["kind"]): r for r in rows if r.get("pass") == 1}
        keys = sorted(set(before) & set(after))
        same_order = sum(before[k]["chunk_ids"] == after[k]["chunk_ids"] for k in keys)
        same_top1 = sum(before[k]["chunk_ids"][:1] == after[k]["chunk_ids"][:1] for k in keys)
        overlap = statistics.mean(len(set(before[k]["chunk_ids"]) & set(after[k]["chunk_ids"])) / TOP_K for k in keys)
        drift = max((abs(a - b) for k in keys for a, b in zip(before[k]["scores"], after[k]["scores"])
                     if before[k]["chunk_ids"] == after[k]["chunk_ids"]), default=0.0)
        print(f"vs {Path(args.compare).name}: identical top-5 order {same_order}/{len(keys)} | "
              f"same top-1 {same_top1}/{len(keys)} | mean top-5 overlap {overlap:.2f} | max score drift {drift:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
