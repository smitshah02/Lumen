"""
Temporal Golden Cases + Temporal-Accuracy Check
===============================================
Scores the temporal differentiator objectively.

Two separate questions, two separate measurements:
  - "Is this chunk about creatinine?"        -> LLM judge (topical relevance)
  - "Did we return the RIGHT time-windowed
     creatinine?"                            -> charttime math (this module)

For each temporal query we run retrieval twice — temporal mode ON ("auto",
which resolves to latest/trend/recent) and OFF ("all") — judge topical
relevance once (pooled + cached), then measure the temporal dimension on the
relevant subset and report the LIFT (temporal vs all). The lift is the proof
the temporal logic is doing real work.

These queries are PATIENT-SCOPED: recency only has meaning within one patient's
timeline (MIMIC shifts each patient's dates independently). Patients are auto-
selected for rich longitudinal records, so you don't hand-pick subject_ids.

Usage (real eval, from the Lumen root):
    python -m src.evals.eval_temporal

Self-test of the metric math (no DB / models / network):
    python src/evals/eval_temporal.py --selftest
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal golden cases — query text + which temporal property we assert.
# `assertion` drives which metric is applied; the retriever's mode is
# auto-detected from the query phrasing (detect_temporal_mode).
# ---------------------------------------------------------------------------
TEMPORAL_CASES = [
    {"id": "temp_latest_creatinine", "query": "most recent creatinine value",        "assertion": "latest"},
    {"id": "temp_latest_hgb",        "query": "latest hemoglobin result",            "assertion": "latest"},
    {"id": "temp_latest_meds",       "query": "current medications",                 "assertion": "latest"},
    {"id": "temp_trend_creatinine",  "query": "creatinine trend over time",          "assertion": "trend"},
    {"id": "temp_trend_potassium",   "query": "potassium values over time",          "assertion": "trend"},
    {"id": "temp_window_labs",       "query": "lab results this admission",          "assertion": "window"},
]


# ---------------------------------------------------------------------------
# Small local charttime parser (kept self-contained so the metrics are
# testable without importing the retriever, which pulls in torch).
# ---------------------------------------------------------------------------
def _to_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip().replace("Z", "")
    if not s or s.lower() in ("none", "nat", "nan"):
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Temporal-accuracy metrics (pure, unit-testable) — operate on the relevant
# chunks IN THE SYSTEM'S RETURNED ORDER.
# ---------------------------------------------------------------------------
def latest_metrics(charttimes: list[datetime]) -> dict:
    """For 'most recent X': is the newest relevant record ranked first?"""
    cts = [c for c in charttimes if c is not None]
    if not cts:
        return {"measurable": False, "n": 0}
    newest = max(cts)
    rank_of_newest = cts.index(newest) + 1
    return {
        "measurable": True,
        "n": len(cts),
        "hit@1": cts[0] == newest,
        "hit@3": rank_of_newest <= 3,
        "rank_of_newest": rank_of_newest,
    }


def trend_metrics(charttimes: list[datetime]) -> dict:
    """For 'trend over time': are results chronological and multi-timepoint?"""
    cts = [c for c in charttimes if c is not None]
    if len(cts) < 2:
        return {"measurable": False, "n": len(cts)}
    ascending = sum(1 for i in range(len(cts) - 1) if cts[i] <= cts[i + 1])
    return {
        "measurable": True,
        "n": len(cts),
        "monotonicity": ascending / (len(cts) - 1),     # 1.0 = perfectly chronological
        "distinct_dates": len({c.date() for c in cts}),  # a trend needs >= 2
    }


def window_metrics(pairs: list[tuple], latest_hadm_id) -> dict:
    """For 'this admission': what fraction of relevant results are from the
    patient's most recent admission?  pairs = [(charttime, hadm_id), ...]."""
    pairs = [(c, h) for c, h in pairs if c is not None]
    if not pairs or latest_hadm_id is None:
        return {"measurable": False, "n": len(pairs)}
    same = sum(1 for _, h in pairs if h == latest_hadm_id)
    return {
        "measurable": True,
        "n": len(pairs),
        "frac_same_admission": same / len(pairs),
    }


def _score(assertion: str, seq: list[tuple], latest_hadm_id) -> dict:
    cts = [c for c, _ in seq]
    if assertion == "latest":
        return latest_metrics(cts)
    if assertion == "trend":
        return trend_metrics(cts)
    return window_metrics(seq, latest_hadm_id)


# ---------------------------------------------------------------------------
# Patient selection — richest longitudinal records
# ---------------------------------------------------------------------------
def select_temporal_patients(limit: int = 2, min_notes: int = 4, min_times: int = 3) -> list[tuple]:
    from sqlalchemy import text as sa_text
    from src.storage import engine
    sql = """
        SELECT subject_id,
               COUNT(*)                  AS n_notes,
               COUNT(DISTINCT charttime) AS n_times
        FROM clinical_notes
        WHERE charttime IS NOT NULL
        GROUP BY subject_id
        HAVING COUNT(*) >= :min_notes AND COUNT(DISTINCT charttime) >= :min_times
        ORDER BY n_times DESC, n_notes DESC
        LIMIT :limit
    """
    with engine.connect() as conn:
        rows = conn.execute(sa_text(sql),
                            {"min_notes": min_notes, "min_times": min_times, "limit": limit}).mappings().all()
    return [(r["subject_id"], r["n_notes"], r["n_times"]) for r in rows]


# ---------------------------------------------------------------------------
# One case: temporal mode vs all, with the relevance pool judged once
# ---------------------------------------------------------------------------
def temporal_accuracy_for_case(retriever, judge, subject_id, query, assertion,
                               top_k: int = 10, threshold: int = 2) -> dict:
    res_t = retriever.search(query, subject_id=subject_id, top_k=top_k, temporal_filter="auto")
    res_a = retriever.search(query, subject_id=subject_id, top_k=top_k, temporal_filter="all")

    pool = {}
    for r in list(res_t) + list(res_a):
        pool.setdefault(r.chunk_id, r.chunk_text)
    judged = judge.judge_pool(query, list(pool.items()))
    relevant = {cid for cid, jr in judged.items() if jr.score >= threshold}

    def rel_seq(results):
        out = []
        for r in results:
            if r.chunk_id in relevant:
                ct = _to_dt(r.charttime)
                if ct is not None:
                    out.append((ct, getattr(r, "hadm_id", None)))
        return out

    seq_t, seq_a = rel_seq(res_t), rel_seq(res_a)

    latest_hadm = None
    all_pairs = seq_t + seq_a
    if all_pairs:
        latest_hadm = max(all_pairs, key=lambda p: p[0])[1]

    return {
        "assertion": assertion,
        "n_relevant": len(relevant),
        "temporal": _score(assertion, seq_t, latest_hadm),
        "all": _score(assertion, seq_a, latest_hadm),
    }


def run_temporal_eval(top_k: int = 10, threshold: int = 2, patients: Optional[list] = None):
    from src.retrieval.hybrid_retriever_v2 import HybridRetriever
    from src.evals.ollama_backend import make_ollama_judge

    print("=" * 74)
    print("  LUMEN TEMPORAL-ACCURACY EVAL  (temporal mode vs all, per patient)")
    print("=" * 74)

    retriever = HybridRetriever(use_reranker=True)
    judge = make_ollama_judge()

    if patients is None:
        picks = select_temporal_patients(limit=2)
        if not picks:
            print("No patients with enough longitudinal notes found.")
            return
        print("\nSelected patients (subject_id, n_notes, n_times):", picks)
        patients = [p[0] for p in picks]

    agg = {"latest": [], "trend": [], "window": []}
    for sid in patients:
        print(f"\n--- patient {sid} ---")
        for case in TEMPORAL_CASES:
            r = temporal_accuracy_for_case(retriever, judge, sid, case["query"], case["assertion"],
                                           top_k=top_k, threshold=threshold)
            t, a = r["temporal"], r["all"]
            if not t.get("measurable"):
                print(f"  {case['id']:<26} n/a (no relevant timepoints retrieved)")
                continue
            agg[case["assertion"]].append((t, a))
            if case["assertion"] == "latest":
                print(f"  {case['id']:<26} hit@1 temporal={t['hit@1']}  all={a.get('hit@1')}  "
                      f"(newest at rank {t['rank_of_newest']} / {t['n']})")
            elif case["assertion"] == "trend":
                print(f"  {case['id']:<26} monotonicity temporal={t['monotonicity']:.2f} "
                      f"all={a.get('monotonicity', 0):.2f}  ({t['distinct_dates']} timepoints)")
            else:
                print(f"  {case['id']:<26} same-admission temporal={t['frac_same_admission']:.2f} "
                      f"all={a.get('frac_same_admission', 0):.2f}  (n={t['n']})")

    # ---- aggregate lift ----
    print("\n" + "=" * 74)
    print("  TEMPORAL LIFT (mean temporal vs mean all)")
    print("=" * 74)
    if agg["latest"]:
        th = sum(1 for t, _ in agg["latest"] if t["hit@1"]) / len(agg["latest"])
        ah = sum(1 for _, a in agg["latest"] if a.get("hit@1")) / len(agg["latest"])
        print(f"  latest   hit@1:        {ah:.2f} -> {th:.2f}   (+{th - ah:+.2f})")
    if agg["trend"]:
        tm = sum(t["monotonicity"] for t, _ in agg["trend"]) / len(agg["trend"])
        am = sum(a.get("monotonicity", 0) for _, a in agg["trend"]) / len(agg["trend"])
        print(f"  trend    monotonicity: {am:.2f} -> {tm:.2f}   (+{tm - am:+.2f})")
    if agg["window"]:
        tw = sum(t["frac_same_admission"] for t, _ in agg["window"]) / len(agg["window"])
        aw = sum(a.get("frac_same_admission", 0) for _, a in agg["window"]) / len(agg["window"])
        print(f"  window   same-adm:     {aw:.2f} -> {tw:.2f}   (+{tw - aw:+.2f})")


# ===========================================================================
# Synthetic self-test of the metric math (no DB / models / network)
# ===========================================================================
def _selftest():
    print("=" * 74)
    print("  Temporal-accuracy metrics — synthetic self-test (MIMIC-style 2150 dates)")
    print("=" * 74)

    d = lambda s: datetime.fromisoformat(s)
    # 5 relevant creatinine timepoints; first 3 in admission A, last 2 in admission B (latest).
    jan, mar, jun, sep, dec = (d("2150-01-10"), d("2150-03-15"), d("2150-06-20"),
                               d("2150-09-25"), d("2150-12-30"))
    hadm = {jan: "A", mar: "A", jun: "A", sep: "B", dec: "B"}
    latest_hadm = "B"

    order_latest = [dec, sep, jun, mar, jan]        # temporal "latest" mode: newest first
    order_trend  = [jan, mar, jun, sep, dec]        # temporal "trend" mode: chronological
    order_all    = [jun, dec, jan, sep, mar]        # "all" mode: relevance order, time-agnostic

    print("\n[latest]  query 'most recent creatinine value'")
    print("   temporal:", latest_metrics(order_latest))
    print("   all     :", latest_metrics(order_all))

    print("\n[trend]   query 'creatinine trend over time'")
    print("   temporal:", trend_metrics(order_trend))
    print("   all     :", trend_metrics(order_all))

    print("\n[window]  query 'lab results this admission'  (latest admission = B)")
    only_b   = [(dec, "B"), (sep, "B")]                                  # temporal 'recent' keeps latest stay
    all_mix  = [(jun, "A"), (dec, "B"), (jan, "A"), (sep, "B"), (mar, "A")]
    print("   temporal:", window_metrics(only_b, latest_hadm))
    print("   all     :", window_metrics(all_mix, latest_hadm))

    # quick assertions so this fails loudly if the math regresses
    assert latest_metrics(order_latest)["hit@1"] is True
    assert latest_metrics(order_all)["hit@1"] is False
    assert trend_metrics(order_trend)["monotonicity"] == 1.0
    assert trend_metrics(order_all)["monotonicity"] < 1.0
    assert window_metrics(only_b, latest_hadm)["frac_same_admission"] == 1.0
    assert window_metrics(all_mix, latest_hadm)["frac_same_admission"] < 1.0
    print("\nAll metric assertions passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Lumen temporal-accuracy eval")
    parser.add_argument("--selftest", action="store_true", help="Run metric self-test (no DB/models)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=2)
    args = parser.parse_args()

    if args.selftest:
        _selftest()
    else:
        run_temporal_eval(top_k=args.top_k, threshold=args.threshold)
