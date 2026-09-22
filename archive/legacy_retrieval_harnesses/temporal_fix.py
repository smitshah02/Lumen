"""
Regression test for the MIMIC-correct temporal logic.
======================================================
Proves that anchoring recency to each patient's OWN latest record beats
anchoring to a global wall-clock, which is the single correctness fix that
makes temporal retrieval work at all on MIMIC-IV.

Why this exists: MIMIC-IV shifts every patient's dates into 2100-2200 with a
*per-patient* offset. Absolute dates are meaningless across patients, but
intervals WITHIN one patient are real. Code that picks any fixed "now" is
wrong for every patient simultaneously — and wrong silently, because it still
returns plausible-looking results.

The live implementation is imported from hybrid_retriever_v2 — this file holds
no copy of it, so the two cannot drift. `old_apply` below is a deliberately
frozen replica of the pre-fix behaviour, kept only as the contrast baseline
the assertions measure against. Do not "fix" it.

No DB, no models, no network — synthetic dates only.

Usage:
    python -m src.retrieval.temporal_fix           # run assertions
    python -m src.retrieval.temporal_fix --show    # + side-by-side old/new output
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from src.retrieval.hybrid_retriever_v2 import (
    RetrievalResult,
    apply_temporal_filter,
    detect_temporal_mode,
    _parse_charttime,
)


def _mk(chunk_id: int, subject_id: int, charttime, rrf_score: float) -> RetrievalResult:
    """Build a RetrievalResult with only the fields the temporal logic reads."""
    return RetrievalResult(
        chunk_id=chunk_id,
        note_id=chunk_id,
        subject_id=subject_id,
        hadm_id=None,
        note_type="discharge",
        chunk_index=0,
        chunk_text="",
        token_count=0,
        charttime=charttime,
        rrf_score=rrf_score,
    )


# ===========================================================================
# The pre-fix behaviour, frozen for contrast. NOT the live implementation.
# ===========================================================================
def old_apply(results, mode="all", boost_recent=True, recency_days=365):
    """
    How temporal filtering worked before the per-patient anchor.

    Two defects, both reproduced faithfully:
      1. `now` is a hardcoded global date, so a patient shifted to 2150 looks
         ~18,000 days stale and every record is "old".
      2. The boost is multiplicative on rrf_score, so a min-maxed score of 0
         stays 0 no matter how recent the record is.
    """
    if mode == "all" and not boost_recent:
        return results
    now = datetime(2200, 1, 1)
    cutoff = now - timedelta(days=recency_days)
    out = []
    for r in results:
        ct = _parse_charttime(r.charttime)
        if mode == "recent" and ct and ct < cutoff:
            continue
        if boost_recent and ct:
            days_ago = (now - ct).days
            r.rrf_score *= (1 + 0.15 * max(0, 1 - days_ago / (recency_days * 2)))
        out.append(r)
    return out


# ===========================================================================
# Scenarios — each returns fresh objects, since the filters mutate rrf_score
# ===========================================================================
def s1():
    """One patient shifted to 2150. Latest = Dec; older = Jan, 334 days before."""
    return [
        _mk(1, 100, "2150-12-01 09:00:00", 0.90),
        _mk(2, 100, "2150-01-01 09:00:00", 0.85),
    ]


def s2():
    """Two patients, offsets 35 years apart. Each has a note ~1 month before
    THEIR OWN latest, so both should earn a near-identical recency boost."""
    return [
        _mk(10, 100, "2150-12-01", 0.50),   # patient 100's latest
        _mk(11, 100, "2150-11-01", 0.50),   # 30d before it
        _mk(20, 200, "2185-06-01", 0.50),   # patient 200's latest
        _mk(21, 200, "2185-05-01", 0.50),   # 31d before it
    ]


def s3():
    """Out-of-order dates plus one undated record, for trend mode."""
    return [
        _mk(31, 100, "2150-09-15", 0.7),
        _mk(32, 100, "2150-03-15", 0.7),
        _mk(33, 100, "2150-12-20", 0.7),
        _mk(34, 100, None, 0.7),
    ]


INTENT_CASES = [
    ("most recent HbA1c",                        "latest"),
    ("current medications",                      "latest"),
    ("trend in HbA1c over the last 12 months",   "trend"),
    ("creatinine progression",                   "trend"),
    ("potassium in the last 7 days",             "recent"),
    ("lab results this admission",               "recent"),
    ("any historical mention of penicillin reaction", "all"),
    ("abnormal potassium lab results",           "all"),
]


# ===========================================================================
# Assertions
# ===========================================================================
def test_intra_patient_window_uses_real_intervals():
    """The old global anchor drops a patient's ENTIRE record as stale."""
    old = old_apply(s1(), mode="recent", recency_days=365)
    assert len(old) == 0, f"expected old anchor to drop everything, kept {len(old)}"

    new = apply_temporal_filter(s1(), mode="recent", recency_days=365)
    assert [r.chunk_id for r in new] == [1, 2], "both notes are within 365d of the patient's own latest"

    tight = apply_temporal_filter(s1(), mode="recent", recency_days=180)
    assert [r.chunk_id for r in tight] == [1], "the 334-day-old note should fall outside a 180d window"


def test_per_patient_anchoring_survives_different_offsets():
    """A 35-year difference in date shift must not change relative recency."""
    old = old_apply(s2(), mode="all", boost_recent=True)
    assert all(abs(r.rrf_score - 0.50) < 1e-9 for r in old), \
        "old anchor makes every record look ancient, so the boost collapses to zero"

    new = {r.chunk_id: r.rrf_score for r in apply_temporal_filter(s2(), mode="latest")}
    assert new[10] > new[11], "patient 100's latest must outrank its own older note"
    assert new[20] > new[21], "patient 200's latest must outrank its own older note"
    assert abs(new[11] - new[21]) < 0.01, \
        f"~30d-old notes should score alike across patients, got {new[11]:.4f} vs {new[21]:.4f}"
    assert new[10] > 0.50, "a boost must actually be applied, not multiplied into nothing"


def test_trend_mode_is_chronological_with_undated_last():
    order = [r.chunk_id for r in apply_temporal_filter(s3(), mode="trend")]
    assert order == [32, 31, 33, 34], f"expected ascending by charttime, undated last; got {order}"


def test_undated_records_survive_recent_mode():
    """We can't prove an undated record is old, so it must not be dropped."""
    results = s3() + [_mk(35, 100, None, 0.9)]
    kept = {r.chunk_id for r in apply_temporal_filter(results, mode="recent", recency_days=30)}
    assert 34 in kept and 35 in kept, "undated records must never be dropped by 'recent'"


def test_query_intent_detection():
    for query, expected in INTENT_CASES:
        got = detect_temporal_mode(query)
        assert got == expected, f"{query!r}: expected {expected}, got {got}"


TESTS = [
    test_intra_patient_window_uses_real_intervals,
    test_per_patient_anchoring_survives_different_offsets,
    test_trend_mode_is_chronological_with_undated_last,
    test_undated_records_survive_recent_mode,
    test_query_intent_detection,
]


# ===========================================================================
# Optional side-by-side display (--show)
# ===========================================================================
def banner(t):
    print("\n" + "=" * 74 + f"\n  {t}\n" + "=" * 74)


def show(rs, label):
    print(f"  {label}")
    for r in rs:
        print(f"    chunk {r.chunk_id}  subj {r.subject_id}  {r.charttime}  "
              f"score={r.rrf_score:.4f}")
    if not rs:
        print("    (empty)")


def demo():
    banner("Scenario 1 — intra-patient window uses REAL intervals, not 2200 distance")
    print("\n  OLD (anchor=2200): 'recent' within 365d — the whole record vanishes")
    show(old_apply(s1(), mode="recent", recency_days=365), "->")
    print("\n  NEW (anchor=patient's own latest): 'recent' within 365d")
    show(apply_temporal_filter(s1(), mode="recent", recency_days=365), "->")
    print("\n  NEW: 'recent' within 180d  (the Jan note is 334d old -> dropped)")
    show(apply_temporal_filter(s1(), mode="recent", recency_days=180), "->")

    banner("Scenario 2 — per-patient anchoring works across patients with different shifts")
    print("\n  OLD (anchor=2200): every record looks ancient -> ~0 boost")
    show(old_apply(s2(), mode="all", boost_recent=True), "->")
    print("\n  NEW (mode=latest): each note boosted vs its OWN patient's latest")
    show(apply_temporal_filter(s2(), mode="latest"), "->")

    banner("Scenario 3 — trend mode = chronological ascending (for 'over time' queries)")
    show(apply_temporal_filter(s3(), mode="trend"), "trend ->")

    banner("Scenario 4 — query-intent auto-detection")
    for query, expected in INTENT_CASES:
        got = detect_temporal_mode(query)
        flag = " " if got == expected else "  <-- MISMATCH"
        print(f"    {got:8s}  <-  \"{query}\"{flag}")


def main():
    parser = argparse.ArgumentParser(description="Temporal logic regression test (no DB/models)")
    parser.add_argument("--show", action="store_true", help="Print side-by-side old vs new output")
    args = parser.parse_args()

    if args.show:
        demo()

    print("\n" + "=" * 74)
    failures = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {fn.__name__}\n          {e}")

    print("=" * 74)
    if failures:
        print(f"\n{failures} of {len(TESTS)} checks FAILED.")
        raise SystemExit(1)
    print(f"\nAll {len(TESTS)} temporal checks passed.")


if __name__ == "__main__":
    main()
