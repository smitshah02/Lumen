"""
Phase 10 — run comparison
=========================
Strictly read-only. It opens existing artifacts, compares what is genuinely
comparable, and REFUSES the rest by name.

The asymmetry that makes this necessary
---------------------------------------
There is no previous answer-level benchmark. The preserved cloud artifacts
(scripts/performance_eval.py -> performance.json) measure latency, call budget,
escalation behaviour and HTTP success. They contain no fact matching, no
citation validity, no temporal scoring and no judge dimension — those metrics
did not exist when they were produced.

So a delta like "fact recall improved 12 points" cannot be computed against
them. Not because the numbers are hard to line up, but because one side of the
subtraction does not exist. Inventing a zero, or silently comparing against the
subset that happens to overlap, would manufacture a result. This module reports
such a comparison as REFUSED, with the reason, and computes the metrics that do
exist on both sides.

Two modes
---------
  legacy     a final_eval run vs a cloud_eval performance.json. Operational
             metrics only, restricted to the 15 shared cases.
  baseline   two final_eval runs. Full comparison, gated on compatibility:
             the same eval-set hash, the same results schema and — for judge
             dimensions only — the same judge prompt version, because aj1 and
             aj2 are not the same measurement.

The 40-case run is `Lumen Final Answer-Level Quality Baseline v1`. It is the
first of its kind; the honest comparison for it is against itself over time.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.evals.final_eval import EVALUATOR_VERSION, cases as case_mod
from src.evals.final_eval import manifest as man


class IncomparableRuns(RuntimeError):
    """The two artifacts cannot be compared at all. Never downgraded into a
    partial number."""


# Metrics that exist in a cloud_eval performance.json summary, and the path to
# the same quantity in a final_eval summary. Anything not on this list is,
# by construction, absent from the legacy side.
LEGACY_SHARED = {
    "n": ("operational", "n"),
    "successful": ("operational", "successful"),
    "mean_ms": ("operational", "latency_ms", "mean"),
    "p50_ms": ("operational", "latency_ms", "p50"),
    "p95_ms": ("operational", "latency_ms", "p95"),
    "mean_llm_calls": ("operational", "mean_llm_calls"),
    "mean_main_calls": ("operational", "mean_main_calls"),
    "mean_fast_calls": ("operational", "mean_fast_calls"),
    "requests_with_zero_llm_calls": ("operational", "requests_with_zero_llm_calls"),
    "human_review_required": ("_review_count",),
}

# Answer-quality metrics. Present in a final_eval summary, absent from every
# legacy artifact — this list is what the refusal names.
QUALITY_ONLY = (
    "deterministic case pass", "fact recall", "factual contradictions",
    "strict-unmatched facts", "citation validity", "citation coverage",
    "temporal accuracy", "unsupported abstention", "ambiguity handling",
    "admission scope", "review recall", "auto-approval precision",
    "every judge dimension",
)


def _dig(d: dict, path: tuple):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _delta(new, old):
    if isinstance(new, (int, float)) and isinstance(old, (int, float)):
        return {"new": new, "old": old, "delta": round(new - old, 4)}
    return {"new": new, "old": old, "delta": None}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_run(run_dir) -> dict:
    """A final_eval run's manifest, summary and raw response rows.

    The rows are carried so the operational section can be recomputed over just
    the shared cases. That is a re-projection of this run's own raw artifacts
    through the same aggregate.operational_section the run itself used — not a
    new measurement, and not a second definition of "mean latency" that could
    drift from the first.
    """
    if not run_dir.file("summary").exists():
        raise IncomparableRuns(
            f"{run_dir.path} has no summary.json; run `score --run-id {run_dir.run_id}` "
            f"first. Comparison never recomputes a measurement.")
    return {"kind": "final_eval",
            "manifest": run_dir.read_json("manifest"),
            "summary": run_dir.read_json("summary"),
            "responses": run_dir.read_jsonl("responses"),
            "path": str(run_dir.path)}


def load_legacy(path: Path | str) -> dict:
    """A cloud_eval performance.json. Identified by shape, not by filename."""
    p = Path(path)
    data = json.loads(p.read_text())
    if "summary" not in data or "requests" not in data:
        raise IncomparableRuns(
            f"{p} is not a cloud_eval performance artifact (no `summary`/`requests`). "
            f"To compare two final_eval runs, pass --baseline-run-id instead.")
    return {"kind": "cloud_eval", "data": data, "path": str(p)}


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------
def compatibility(new: dict, old: dict) -> dict:
    """What may be compared, and what is refused, with a reason for each."""
    refused, notes = [], []
    nm, ns = new["manifest"], new["summary"]

    if old["kind"] == "cloud_eval":
        refused.append({
            "comparison": "answer quality",
            "metrics": list(QUALITY_ONLY),
            "reason": ("the baseline is a cloud_eval performance artifact. It measures "
                       "latency, call budget and escalation only — it contains no "
                       "answer-level quality metric, so no delta exists to compute. "
                       "This is the first answer-level benchmark."),
        })
        legacy_ids = set(case_mod.LEGACY_IDS)
        evaluated = set((nm.get("eval_set") or {}).get("evaluated_ids") or [])
        shared = sorted(legacy_ids & evaluated)
        if not shared:
            raise IncomparableRuns(
                "the two runs share no case ids: the legacy artifact covers "
                f"{sorted(legacy_ids)[:3]}… and this run evaluated "
                f"{sorted(evaluated)[:3]}…")
        notes.append(f"restricted to the {len(shared)} shared legacy case(s)")
        return {"mode": "legacy", "shared_ids": shared, "refused": refused,
                "notes": notes, "quality_comparable": False,
                "judge_comparable": False}

    om, os_ = old["manifest"], old["summary"]
    new_sha = (nm.get("eval_set") or {}).get("sha256")
    old_sha = (om.get("eval_set") or {}).get("sha256")
    quality_ok = True
    if new_sha != old_sha:
        quality_ok = False
        refused.append({
            "comparison": "answer quality",
            "metrics": list(QUALITY_ONLY),
            "reason": (f"the runs used different evaluation sets "
                       f"({str(new_sha)[:12]} vs {str(old_sha)[:12]}). Different "
                       f"questions produce different numbers for reasons that have "
                       f"nothing to do with the system."),
        })
    if nm.get("results_schema_version") != om.get("results_schema_version"):
        notes.append(f"results schema {om.get('results_schema_version')} -> "
                     f"{nm.get('results_schema_version')}: row-level fields differ")

    new_j = (nm.get("judge") or {}).get("prompt_version")
    old_j = (om.get("judge") or {}).get("prompt_version")
    judge_ok = quality_ok and bool(new_j) and new_j == old_j
    if not judge_ok:
        refused.append({
            "comparison": "judge dimensions",
            "metrics": ["every judge dimension mean, median and distribution"],
            "reason": (f"judge prompt versions differ ({old_j} vs {new_j}); the rubric and "
                       f"the schema changed, so the scores are not the same measurement"
                       if new_j != old_j else
                       "the eval sets differ, so judge scores are not comparable"),
        })

    new_ids = set((nm.get("eval_set") or {}).get("evaluated_ids") or [])
    old_ids = set((om.get("eval_set") or {}).get("evaluated_ids") or [])
    shared = sorted(new_ids & old_ids)
    if not shared:
        raise IncomparableRuns("the two runs share no case ids")
    if new_ids != old_ids:
        notes.append(f"case sets differ: {len(shared)} shared, "
                     f"{len(new_ids - old_ids)} new-only, {len(old_ids - new_ids)} old-only")
        # summary.json's quality section is whole-run, so a delta across
        # different case sets would partly measure the case sets.
        if quality_ok:
            quality_ok = judge_ok = False
            refused.append({
                "comparison": "answer quality",
                "metrics": list(QUALITY_ONLY),
                "reason": (f"the runs evaluated different case sets ({len(new_ids)} vs "
                           f"{len(old_ids)}, {len(shared)} shared). The quality section is "
                           f"computed over a whole run, so a delta would partly measure "
                           f"which questions each run asked."),
            })
    return {"mode": "baseline", "shared_ids": shared, "refused": refused, "notes": notes,
            "quality_comparable": quality_ok, "judge_comparable": judge_ok}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _legacy_summary(old: dict, shared: list):
    """The legacy side restricted to exactly the shared cases, or None.

    cloud_eval publishes a `legacy_subset` summary and a whole-run summary. One
    of them must cover exactly the shared ids — otherwise the two sides of every
    subtraction would be computed over different questions, and a latency
    "delta" would mostly measure which cases each run happened to include. There
    is no honest fallback, so the operational comparison is refused instead.
    """
    d, want = old["data"], set(shared)
    sub = d.get("legacy_subset") or {}
    if sub.get("summary") and set(sub.get("evaluated") or []) == want:
        return {**sub["summary"], "_scope": f"legacy_subset, exactly the {len(want)} shared ids"}
    whole = {r.get("query_id") for r in (d.get("requests") or [])}
    if d.get("summary") and whole == want:
        return {**d["summary"], "_scope": f"whole legacy run, exactly the {len(want)} shared ids"}
    return None


def _project(run: dict, shared: list):
    """This run's operational metrics over just the shared cases, or None when
    its raw rows were not carried."""
    rows = run.get("responses")
    if rows is None:
        return None
    want = set(shared)
    subset = [r for r in rows if r.get("query_id") in want]
    if len(subset) != len(want):
        return None
    from src.evals.final_eval.aggregate import operational_section
    o = operational_section(subset)
    return {"operational": o, "_scope": f"recomputed over the {len(want)} shared ids"}


def _operational_of(summary: dict, review_count: int) -> dict:
    out = {}
    for name, path in LEGACY_SHARED.items():
        out[name] = review_count if path == ("_review_count",) else _dig(summary, path)
    return out


def _review_count(summary: dict) -> int:
    d = _dig(summary, ("operational", "review_status")) or {}
    return sum(v for k, v in d.items() if k and "review" in str(k))


def compare(new: dict, old: dict) -> dict:
    """The comparison record. Never raises for a missing metric — it refuses."""
    compat = compatibility(new, old)
    shared = compat["shared_ids"]
    ns, nm = new["summary"], new["manifest"]

    # Both sides must describe the SAME cases or the operational deltas are not
    # about the system at all.
    new_proj = _project(new, shared)
    new_scope = (new_proj or {}).get("_scope") or "whole run"
    new_src = new_proj if new_proj else {"operational": ns.get("operational") or {}}
    new_ops = _operational_of(new_src, _review_count(new_src))

    if compat["mode"] == "legacy":
        old_sum = _legacy_summary(old, shared)
        old_scope = (old_sum or {}).get("_scope")
        old_ops = {k: (old_sum or {}).get(k) for k in LEGACY_SHARED}
        baseline_ref = {"kind": "cloud_eval", "path": old["path"],
                        "run_id": old["data"].get("run_id"),
                        "generated_at": old["data"].get("generated_at"),
                        "models": old["data"].get("models"),
                        "scope": old_scope}
        quality = None
    else:
        os_, om = old["summary"], old["manifest"]
        old_proj = _project(old, shared)
        old_scope = (old_proj or {}).get("_scope") or "whole run"
        old_src = old_proj if old_proj else {"operational": os_.get("operational") or {}}
        old_ops = _operational_of(old_src, _review_count(old_src))
        baseline_ref = {"kind": "final_eval", "path": old["path"],
                        "run_id": om.get("run_id"), "git_sha": om.get("git_sha"),
                        "models": om.get("models", {}).get("main"),
                        "judge_prompt_version": (om.get("judge") or {}).get("prompt_version"),
                        "scope": old_scope}
        quality = _quality_delta(ns, os_, compat) if compat["quality_comparable"] else None

    scopes_ok = old_ops and all(v is not None for v in old_ops.values()) \
        and new_proj is not None and old_scope is not None
    if not scopes_ok:
        compat["refused"].append({
            "comparison": "operational metrics",
            "metrics": list(LEGACY_SHARED),
            "reason": (f"neither artifact can be restricted to exactly the "
                       f"{len(shared)} shared case(s) "
                       f"(new scope: {new_scope}; baseline scope: {old_scope or 'unavailable'}). "
                       f"Subtracting metrics computed over different question sets would "
                       f"mostly measure which cases each run included."),
        })
        operational = None
    else:
        operational = {k: _delta(new_ops.get(k), old_ops.get(k)) for k in LEGACY_SHARED}

    return {
        "scope": {"shared_cases": len(shared), "new": new_scope, "baseline": old_scope},
        "generated_at": man.utc_now(),
        "evaluator_version": EVALUATOR_VERSION,
        "read_only": True,
        "mode": compat["mode"],
        "new_run": {"run_id": nm.get("run_id"), "git_sha": nm.get("git_sha"),
                    "benchmark": nm.get("benchmark"),
                    "judge_prompt_version": (nm.get("judge") or {}).get("prompt_version"),
                    "path": new["path"]},
        "baseline": baseline_ref,
        "shared_case_ids": compat["shared_ids"],
        "notes": compat["notes"],
        "operational": operational,
        "quality": quality,
        "refused": compat["refused"],
        "configuration_differences": _config_diff(new, old),
        "interpretation": (
            "Operational deltas across different hardware or runtimes are descriptive, "
            "not causal. A refused comparison means the metric does not exist on one "
            "side; it is never reported as an unchanged or zero delta."),
    }


_QUALITY_PATHS = {
    "case_pass_rate": ("deterministic_quality", "case_pass_rate", "rate"),
    "fact_recall": ("deterministic_quality", "facts", "fact_recall", "rate"),
    "contradictions": ("deterministic_quality", "facts", "contradictions"),
    "strict_unmatched": ("deterministic_quality", "facts", "strict_unmatched"),
    "citation_validity": ("deterministic_quality", "citations", "citation_validity_rate", "rate"),
    "citation_coverage": ("deterministic_quality", "citations", "citation_coverage", "rate"),
    "temporal_accuracy": ("deterministic_quality", "temporal_accuracy", "rate"),
    "review_recall": ("deterministic_quality", "routing", "review_recall", "rate"),
    "auto_approval_precision": ("deterministic_quality", "routing",
                                "auto_approval_precision", "rate"),
    "review_burden": ("deterministic_quality", "routing", "review_burden", "rate"),
}


def _quality_delta(ns: dict, os_: dict, compat: dict) -> dict:
    out = {k: _delta(_dig(ns, p), _dig(os_, p)) for k, p in _QUALITY_PATHS.items()}
    if compat["judge_comparable"]:
        from src.evals.final_eval.judge import DIMENSIONS
        out["judge"] = {
            d: _delta(_dig(ns, ("judge_quality", "dimensions", d, "mean")),
                      _dig(os_, ("judge_quality", "dimensions", d, "mean")))
            for d in DIMENSIONS}
    else:
        out["judge"] = None
    return out


def _config_diff(new: dict, old: dict) -> dict:
    nm = new["manifest"]
    if old["kind"] == "cloud_eval":
        om = {"models": old["data"].get("models") or {}}
    else:
        om = old["manifest"]
    out = {}
    for role in ("main", "fast"):
        a, b = (nm.get("models") or {}).get(role), (om.get("models") or {}).get(role)
        if a != b:
            out[f"model_{role}"] = {"new": a, "old": b}
    if old["kind"] == "final_eval":
        for k in ("prompt_versions", "retrieval", "git_sha", "branch"):
            if nm.get(k) != om.get(k):
                out[k] = {"new": nm.get(k), "old": om.get(k)}
    return out


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render(cmp: dict) -> str:
    L = [f"=== comparison ({cmp['mode']}) ===", "",
         f"  new      {cmp['new_run']['run_id']}  ({str(cmp['new_run']['git_sha'])[:12]})",
         f"  baseline {cmp['baseline'].get('run_id')}  [{cmp['baseline']['kind']}] "
         f"{cmp['baseline']['path']}",
         f"  shared   {len(cmp['shared_case_ids'])} case(s)", ""]
    for n in cmp["notes"]:
        L.append(f"  note: {n}")
    L.append(f"  scope: new={cmp['scope']['new']}, baseline={cmp['scope']['baseline']}")
    L += ["", "  operational", "  " + "-" * 60]
    if cmp["operational"] is None:
        L.append("    REFUSED — see below")
    else:
        for k, v in cmp["operational"].items():
            d = "" if v["delta"] is None else f"  ({v['delta']:+})"
            L.append(f"    {k:<30} {v['old']} -> {v['new']}{d}")
    if cmp["quality"]:
        L += ["", "  answer quality", "  " + "-" * 60]
        for k, v in cmp["quality"].items():
            if k == "judge" or v is None:
                continue
            d = "" if v["delta"] is None else f"  ({v['delta']:+})"
            L.append(f"    {k:<30} {v['old']} -> {v['new']}{d}")
        if cmp["quality"].get("judge"):
            L.append("    judge dimension means")
            for k, v in cmp["quality"]["judge"].items():
                d = "" if v["delta"] is None else f"  ({v['delta']:+})"
                L.append(f"      {k:<28} {v['old']} -> {v['new']}{d}")
    if cmp["configuration_differences"]:
        L += ["", "  configuration differences", "  " + "-" * 60]
        for k, v in cmp["configuration_differences"].items():
            L.append(f"    {k}: {str(v['old'])[:60]} -> {str(v['new'])[:60]}")
    if cmp["refused"]:
        L += ["", "  REFUSED", "  " + "-" * 60]
        for r in cmp["refused"]:
            L.append(f"    {r['comparison']}: {r['reason']}")
            L.append(f"      not computed: {', '.join(r['metrics'][:6])}"
                     + (" …" if len(r["metrics"]) > 6 else ""))
    L += ["", f"  {cmp['interpretation']}", ""]
    return "\n".join(L)
