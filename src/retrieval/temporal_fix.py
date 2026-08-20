"""
Self-contained validation of the MIMIC-correct temporal logic.
The functions below are EXACTLY what goes into hybrid_retriever_v2.py
(with the real RetrievalResult). Minimal stand-in here so it runs alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RetrievalResult:  # minimal stand-in for the test
    chunk_id: int
    subject_id: int
    charttime: Optional[str]
    rrf_score: float = 0.0
    sources: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Charttime parsing (MIMIC charttime is "YYYY-MM-DD HH:MM:SS")
# ---------------------------------------------------------------------------
def _parse_charttime(value) -> Optional[datetime]:
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
# Optional: infer temporal intent straight from the query phrasing.
# Conservative — returns "all" unless there's a clear signal.
# ---------------------------------------------------------------------------
_TEMPORAL_PATTERNS = [
    ("latest", [r"\bmost recent\b", r"\blatest\b", r"\bnewest\b", r"\bcurrent(?:ly)?\b",
                r"\blast (?:recorded|known|measured|documented|value)\b", r"\bas of (?:now|today)\b"]),
    ("trend",  [r"\btrend(?:ing|ed)?\b", r"\bover time\b", r"\bprogression\b", r"\bserial\b",
                r"\bevolution\b", r"\bchang(?:e|ed|ing) over\b",
                r"\bover the (?:past|last) \w+ (?:days|weeks|months|years)\b"]),
    ("recent", [r"\bin the (?:past|last) \d+ (?:days|weeks|months|years)\b",
                r"\brecent(?:ly)?\b", r"\bthis admission\b", r"\bduring (?:this|the current)\b"]),
]


def detect_temporal_mode(query: str) -> str:
    q = query.lower()
    for mode, patterns in _TEMPORAL_PATTERNS:   # latest > trend > recent
        if any(re.search(p, q) for p in patterns):
            return mode
    return "all"


# ---------------------------------------------------------------------------
# THE FIX
# ---------------------------------------------------------------------------
def apply_temporal_filter(
    results: list[RetrievalResult],
    mode: str = "all",
    boost_recent: bool = True,
    recency_days: int = 365,
    boost_weight: float = 0.20,
    halflife_days: float = 180.0,
    reference_times: Optional[dict] = None,
) -> list[RetrievalResult]:
    """
    Temporal reweighting/filtering that is correct for MIMIC's shifted dates.

    MIMIC-IV shifts each patient's dates into 2100-2200 with a *per-patient*
    offset. Absolute dates are meaningless across patients, but the offset is
    identical for all of one patient's records, so intervals WITHIN a patient
    are real. We therefore anchor recency to each subject's OWN latest retrieved
    record, never a global wall-clock.

    Modes:
      "all"                    -> no temporal effect (relevance order kept)
      "recent"                 -> drop records older than recency_days before the
                                  subject's anchor; boost survivors by recency
      "latest"/"most_recent"   -> boost toward newest per subject; keep all
      "trend"/"oldest_first"   -> chronological ascending (undated sink last)
    """
    mode = (mode or "all").lower()
    if mode == "oldest_first":
        mode = "trend"
    if mode == "most_recent":
        mode = "latest"
    if mode == "all":
        return results

    # Per-subject anchor = that subject's latest retrieved charttime
    # (unless the caller supplies the patient's true latest encounter).
    refs: dict = dict(reference_times) if reference_times else {}
    if not reference_times:
        for r in results:
            ct = _parse_charttime(r.charttime)
            if ct is None:
                continue
            if r.subject_id not in refs or ct > refs[r.subject_id]:
                refs[r.subject_id] = ct

    kept: list[RetrievalResult] = []
    for r in results:
        ct = _parse_charttime(r.charttime)
        ref = refs.get(r.subject_id)

        # Undated records keep their relevance but get no recency signal and
        # are not dropped by "recent" (we can't prove they're old).
        if ct is None or ref is None:
            kept.append(r)
            continue

        days_ago = max(0.0, (ref - ct).total_seconds() / 86400.0)

        if mode == "recent" and days_ago > recency_days:
            continue

        if boost_recent and mode in ("recent", "latest"):
            r.rrf_score += boost_weight * (0.5 ** (days_ago / halflife_days))

        kept.append(r)

    if mode == "trend":
        kept.sort(key=lambda x: _parse_charttime(x.charttime) or datetime.max)
    else:
        kept.sort(key=lambda x: x.rrf_score, reverse=True)

    return kept


# ---------------------------------------------------------------------------
# The OLD broken behaviour, reproduced for contrast
# ---------------------------------------------------------------------------
def old_apply(results, mode="all", boost_recent=True, recency_days=365):
    from datetime import timedelta
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
# Scenarios
# ===========================================================================
def banner(t): print("\n" + "=" * 74 + f"\n  {t}\n" + "=" * 74)


def show(rs, label):
    print(f"  {label}")
    for r in rs:
        print(f"    chunk {r.chunk_id}  subj {r.subject_id}  {r.charttime}  "
              f"score={r.rrf_score:.4f}")
    if not rs:
        print("    (empty)")


banner("Scenario 1 — intra-patient window uses REAL intervals, not 2200 distance")
# One patient, dates shifted to 2150. Latest = Dec; older = Jan (~334 days before).
def s1():
    return [
        RetrievalResult(1, 100, "2150-12-01 09:00:00", rrf_score=0.90),
        RetrievalResult(2, 100, "2150-01-01 09:00:00", rrf_score=0.85),  # 334d before
    ]
print("\n  OLD (anchor=2200): 'recent' within 365d")
show(old_apply(s1(), mode="recent", recency_days=365), "->")
print("\n  NEW (anchor=patient's own latest): 'recent' within 365d")
show(apply_temporal_filter(s1(), mode="recent", recency_days=365), "->")
print("\n  NEW: 'recent' within 180d  (the Jan note is 334d old -> dropped)")
show(apply_temporal_filter(s1(), mode="recent", recency_days=180), "->")


banner("Scenario 2 — per-patient anchoring works across patients with different shifts")
# Patient 100 lives in 2150, patient 200 in 2185. Each has a note ~1 month before
# THEIR OWN latest. Both should get a strong, comparable recency boost.
def s2():
    return [
        RetrievalResult(10, 100, "2150-12-01", rrf_score=0.50),  # 100's latest
        RetrievalResult(11, 100, "2150-11-01", rrf_score=0.50),  # ~30d before
        RetrievalResult(20, 200, "2185-06-01", rrf_score=0.50),  # 200's latest
        RetrievalResult(21, 200, "2185-05-01", rrf_score=0.50),  # ~31d before
    ]
print("\n  OLD (anchor=2200): every record looks ancient -> ~0 boost")
show(old_apply(s2(), mode="latest" if False else "all", boost_recent=True), "->")
print("\n  NEW (mode=latest): each note boosted vs its OWN patient's latest")
show(apply_temporal_filter(s2(), mode="latest"), "->")


banner("Scenario 3 — trend mode = chronological ascending (for 'over time' queries)")
def s3():
    return [
        RetrievalResult(31, 100, "2150-09-15", rrf_score=0.7),
        RetrievalResult(32, 100, "2150-03-15", rrf_score=0.7),
        RetrievalResult(33, 100, "2150-12-20", rrf_score=0.7),
        RetrievalResult(34, 100, None,          rrf_score=0.7),  # undated -> last
    ]
show(apply_temporal_filter(s3(), mode="trend"), "trend ->")


banner("Scenario 4 — query-intent auto-detection")
for q in [
    "most recent HbA1c",
    "trend in HbA1c over the last 12 months",
    "potassium in the last 7 days",
    "any historical mention of penicillin reaction",
    "abnormal potassium lab results",
    "current medications",
]:
    print(f"    {detect_temporal_mode(q):8s}  <-  \"{q}\"")

print("\nAll scenarios ran.")
