"""
Phase 5 — calibration
=====================
Where the three independent opinions on a case disagree, and what a human needs
to decide about it.

The three opinions are:

  deterministic   gold-anchored code. Certain about numbers, units, dates and
                  provenance; deliberately unable to resolve paraphrase.
  runtime         the frozen system's own routing decision (auto-approved or
                  escalated). Measured, never used to define correctness.
  judge           an independent model's per-criterion assessment and scores.
                  A second opinion, explicitly NOT ground truth.

This module reads a completed run's artifacts and nothing else. It calls no
model, touches no database, re-runs no case and writes nothing outside
calibration.json / calibration.md. Running it after the 15-case cloud run
therefore cannot change a single measured number — which is the point of
building it BEFORE those answers exist.

What it does not do
-------------------
It does not decide who is right. Every finding carries a `suggested_category`
derived from a rule, and an `adjudication` field left null for a human. An LLM
judge is not ground truth, and neither is a regex: a disagreement is a question,
and answering it automatically is how a calibration step turns into a way of
making the numbers agree with whichever side is louder.
"""

from __future__ import annotations

import json
from collections import Counter

from src.evals.final_eval import EVALUATOR_VERSION

# The adjudication vocabulary. Rules below may SUGGEST one; only a human may
# record one. `deterministic_evaluator_bug` is never suggested — code cannot
# diagnose its own bug, it can only report that it disagreed with something.
CATEGORIES = (
    "deterministic_evaluator_too_strict",
    "deterministic_evaluator_bug",
    "judge_too_lenient",
    "judge_too_harsh",
    "judge_completeness_inflation",
    "judge_temporal_misunderstanding",
    "judge_grounding_error",
    "legitimate_system_failure",
    "human_adjudication_required",
)

TOP = 4          # the 0-4 rubric's ceiling
EXCERPT = 160


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dim(judge: dict, name: str):
    d = ((judge or {}).get("dimensions") or {}).get(name) or {}
    s = d.get("score")
    return s if isinstance(s, int) and not isinstance(s, bool) else None


def _judge_usable(judge: dict) -> bool:
    return bool(judge) and judge.get("status") in ("ok", "partial")


def _assessments_by_criterion(judge: dict) -> dict:
    return {a["criterion_id"]: a for a in (judge or {}).get("criterion_assessments") or []}


def _fact_criterion_ids(judge: dict) -> list:
    """Criterion ids for the gold expected_facts, in gold order.

    The judge row carries the criteria it was actually given, so the mapping
    from fact index to criterion id is read from the artifact rather than
    recomputed — a run judged under a different criterion layout still lines up.
    """
    return [c["id"] for c in (judge or {}).get("criteria") or [] if c.get("kind") == "fact"]


def _finding(kind: str, detail: str, category: str, det_evidence, judge_evidence) -> dict:
    return {
        "kind": kind,
        "detail": detail,
        "suggested_category": category,
        "deterministic_evidence": det_evidence,
        "judge_evidence": judge_evidence,
        # Filled in by a human. The code never writes here.
        "adjudication": None,
        "adjudication_note": None,
    }


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def disagreements(det: dict, judge: dict) -> list:
    """Every disagreement for one case. Order is stable, so two calibration
    runs over the same artifacts produce identical files."""
    out: list = []
    facts = det["facts"]
    judge_present = bool(judge)

    # --- judge could not be used ----------------------------------------
    if judge_present and not _judge_usable(judge):
        out.append(_finding(
            "judge_unusable",
            f"the judge returned status {judge['status']!r}; its scores cannot be "
            f"compared against the deterministic findings",
            "human_adjudication_required",
            {"case_pass": det["case_pass"], "failure_tags": det["failure_tags"]},
            {"status": judge["status"], "error": (judge.get("error") or "")[:EXCERPT],
             "consistency_violations": judge.get("consistency_violations") or [],
             "repair_attempted": bool(judge.get("repair_attempted"))}))
        return out                      # nothing else about this judge row means anything

    by_crit = _assessments_by_criterion(judge)
    fact_ids = _fact_criterion_ids(judge)

    # --- per-fact: strict matching vs the judge's own assessment ---------
    for i, r in enumerate(facts["results"]):
        cid = fact_ids[i] if i < len(fact_ids) else None
        a = by_crit.get(cid) if cid else None
        if not a:
            continue
        if r["outcome"] == "strict_unmatched" and a["status"] == "supported":
            out.append(_finding(
                "strict_unmatched_but_judge_supported",
                f"deterministic matching could not confirm {r['fact']!r}; the judge "
                f"reports it supported. Either the answer paraphrases it (matching is "
                f"too strict) or the judge accepted something the answer does not say.",
                "deterministic_evaluator_too_strict",
                {"fact": r["fact"], "kind": r["kind"], "detail": r["detail"][:EXCERPT]},
                {"criterion_id": cid, "status": a["status"],
                 "reason": (a.get("reason") or "")[:EXCERPT],
                 "evidence_labels": a.get("evidence_labels")}))
        elif r["outcome"] == "strict_match" and a["status"] in ("missing", "contradicted"):
            out.append(_finding(
                "strict_match_but_judge_says_absent",
                f"the answer contains every anchor of {r['fact']!r}, yet the judge "
                f"marked the criterion {a['status']}.",
                "judge_too_harsh",
                {"fact": r["fact"], "detail": r["detail"][:EXCERPT]},
                {"criterion_id": cid, "status": a["status"],
                 "reason": (a.get("reason") or "")[:EXCERPT]}))

    # --- dimension-level contradictions ----------------------------------
    if facts["n_contradictions"] and _dim(judge, "factual_correctness") == TOP:
        bad = [r for r in facts["results"] if r["outcome"] == "contradiction"]
        out.append(_finding(
            "contradiction_but_judge_fully_correct",
            f"{len(bad)} deterministic contradiction(s) against pinned gold values, "
            f"yet factual_correctness=4.",
            "judge_too_lenient",
            {"contradictions": [{"fact": r["fact"], "detail": r["detail"][:EXCERPT]}
                                for r in bad]},
            {"factual_correctness": TOP,
             "reason": (judge["dimensions"]["factual_correctness"].get("reason") or "")[:EXCERPT]}))

    if not facts["threshold_met"] and _dim(judge, "completeness") == TOP:
        out.append(_finding(
            "incomplete_but_judge_completeness_top",
            f"only {facts['matched']}/{facts['n_expected']} required facts matched "
            f"(min_facts={facts['min_facts']}), yet completeness=4. This is the aj1 "
            f"failure mode aj2's consistency check exists to catch; seeing it here "
            f"means the judge marked the unmatched fact supported.",
            "judge_completeness_inflation",
            {"matched": facts["matched"], "n_expected": facts["n_expected"],
             "min_facts": facts["min_facts"],
             "unmatched": [r["fact"] for r in facts["results"]
                           if r["outcome"] != "strict_match"]},
            {"completeness": TOP,
             "criterion_status_counts": judge.get("criterion_status_counts")}))

    if det["temporal"]["pass"] is False and _dim(judge, "temporal_correctness") == TOP:
        out.append(_finding(
            "temporal_failure_but_judge_temporal_top",
            f"the {det['temporal']['mode']} check failed deterministically, yet "
            f"temporal_correctness=4.",
            "judge_temporal_misunderstanding",
            {"mode": det["temporal"]["mode"], "detail": det["temporal"]["detail"][:EXCERPT]},
            {"temporal_correctness": TOP,
             "reason": (judge["dimensions"]["temporal_correctness"].get("reason") or "")[:EXCERPT]}))

    if det["citations"]["hallucinated_labels_pre_strip"] and _dim(judge, "groundedness") == TOP:
        out.append(_finding(
            "hallucinated_citation_but_judge_fully_grounded",
            "the model emitted citation label(s) that do not exist, yet groundedness=4. "
            "The judge sees the repaired answer, so this is expected to need a human "
            "read rather than a judge correction.",
            "judge_grounding_error",
            {"labels": det["citations"]["hallucinated_labels_pre_strip"],
             "labels_available": det["citations"]["labels_available"]},
            {"groundedness": TOP,
             "unsupported_content": judge.get("unsupported_content") or []}))

    if det["abstention"]["pass"] is False and _dim(judge, "abstention_quality") == TOP:
        out.append(_finding(
            "missed_abstention_but_judge_abstention_top",
            "an unsupported case was not declined cleanly, yet abstention_quality=4.",
            "judge_too_lenient",
            {k: det["abstention"][k] for k in
             ("refusal_detected", "must_not_contain_violations", "fabricated_citations")},
            {"abstention_quality": TOP}))

    if det["ambiguity"]["pass"] is False and _dim(judge, "abstention_quality") == TOP:
        out.append(_finding(
            "mishandled_ambiguity_but_judge_abstention_top",
            "an ambiguous case was answered with false certainty, yet "
            "abstention_quality=4.",
            "judge_too_lenient",
            {"markers": det["ambiguity"].get("markers"),
             "hard_refusal": det["ambiguity"].get("hard_refusal")},
            {"abstention_quality": TOP}))

    # --- routing ---------------------------------------------------------
    if det["review_worthy"]["is_review_worthy"] and \
            not det["routing_observed"]["needs_human_review"]:
        out.append(_finding(
            "auto_approved_despite_review_worthy_finding",
            "the run auto-approved an answer carrying an independently detected, "
            "gold-derived finding. This is a finding about the SYSTEM, not about "
            "either evaluator.",
            "legitimate_system_failure",
            {"reasons": det["review_worthy"]["reasons"],
             "routing": det["routing_observed"]},
            {"dimensions": {k: _dim(judge, k) for k in
                            (judge.get("applicable_dimensions") or [])} if judge else None}))

    # --- whole-case shape -------------------------------------------------
    applicable = judge.get("applicable_dimensions") or [] if judge else []
    if applicable and not det["case_pass"] and \
            all(_dim(judge, k) == TOP for k in applicable):
        out.append(_finding(
            "case_failed_deterministically_but_judge_scored_perfect",
            "the deterministic case pass failed while every applicable judge dimension "
            "scored 4. One of the two is measuring the wrong thing.",
            "human_adjudication_required",
            {"failed_components": [k for k, v in det["case_pass_components"].items() if not v],
             "failure_tags": det["failure_tags"]},
            {"dimensions": {k: TOP for k in applicable}}))
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(run_dir, manifest: dict, progress=None) -> dict:
    """Read a run's artifacts and produce the calibration record.

    Raises nothing on a judge-less run: the judge-side detectors simply do not
    fire, and the artifact says so rather than implying agreement.
    """
    say = progress or (lambda *_a, **_k: None)
    dets = run_dir.read_jsonl("deterministic")
    if not dets:
        raise SystemExit(
            f"no deterministic.jsonl in {run_dir.path}; run `deterministic` (or `run`) "
            f"first. Calibration only ever reads existing artifacts.")
    judges = {j["query_id"]: j for j in run_dir.read_jsonl("judge")}

    cases, kinds, cats = [], Counter(), Counter()
    for det in sorted(dets, key=lambda d: d["query_id"]):
        j = judges.get(det["query_id"])
        found = disagreements(det, j)
        for f in found:
            kinds[f["kind"]] += 1
            cats[f["suggested_category"]] += 1
        cases.append({
            "query_id": det["query_id"],
            "category": det["category"],
            "answer_type": det["answer_type"],
            "difficulty": det["difficulty"],
            "gold_criteria": [c for c in (j or {}).get("criteria") or []],
            "deterministic": {
                "case_pass": det["case_pass"],
                "failed_components": [k for k, v in det["case_pass_components"].items() if not v],
                "matched_facts": f"{det['facts']['matched']}/{det['facts']['n_expected']}",
                "min_facts": det["facts"]["min_facts"],
                "contradictions": det["facts"]["n_contradictions"],
                "strict_unmatched": det["facts"]["n_strict_unmatched"],
                "temporal": det["temporal"]["pass"],
                "abstention": det["abstention"]["pass"],
                "ambiguity": det["ambiguity"]["pass"],
                "hallucinated_labels_pre_strip": det["citations"]["hallucinated_labels_pre_strip"],
                "failure_tags": det["failure_tags"],
                "review_worthy": det["review_worthy"]["is_review_worthy"],
                "review_worthy_reasons": det["review_worthy"]["reasons"],
            },
            "runtime": det["routing_observed"],
            "judge": ({
                "status": j["status"],
                "dimensions": {k: _dim(j, k) for k in (j.get("applicable_dimensions") or [])},
                "criterion_status_counts": j.get("criterion_status_counts"),
                "criterion_assessments": [
                    {"criterion_id": a["criterion_id"], "status": a["status"],
                     "reason": (a.get("reason") or "")[:EXCERPT]}
                    for a in (j.get("criterion_assessments") or [])],
                "unsupported_content": j.get("unsupported_content") or [],
                "consistency_violations": j.get("consistency_violations") or [],
            } if j else {"status": "absent",
                         "note": "no judge row for this case; judge-side comparisons "
                                 "were not evaluated"}),
            "disagreements": found,
        })
        say(f"  {det['query_id']:<10} {len(found)} disagreement(s) "
            f"{[f['kind'] for f in found] or ''}")

    n_with = sum(1 for c in cases if c["disagreements"])
    return {
        "run_id": manifest.get("run_id"),
        "generated_at": _utc(),
        # Calibration is a DERIVED view over preserved raw artifacts, so it may be
        # produced (or reproduced) for a sealed run. Recording that here keeps an
        # artifact from ever silently post-dating the run it describes.
        "generated_after_seal": run_dir.is_complete(),
        "evaluator_version": EVALUATOR_VERSION,
        "judge_prompt_version": (manifest.get("judge") or {}).get("prompt_version"),
        "judge_model": (manifest.get("judge") or {}).get("model"),
        "source": ("this run's existing artifacts only (deterministic.jsonl, judge.jsonl). "
                   "No model was called, no case was re-run, no database was read."),
        "n_cases": len(cases),
        "n_cases_with_disagreements": n_with,
        "n_cases_judged": sum(1 for c in cases if c["judge"]["status"] != "absent"),
        "by_kind": dict(kinds.most_common()),
        "by_suggested_category": dict(cats.most_common()),
        "adjudication": {
            "status": "pending",
            "categories": list(CATEGORIES),
            "note": ("`suggested_category` is a rule's guess. Set `adjudication` on each "
                     "finding by hand. An LLM judge is not ground truth, and neither is "
                     "the deterministic matcher on a paraphrase — the point of this file "
                     "is to make a human choose, per case, which one was wrong."),
        },
        "cases": cases,
    }


def _utc() -> str:
    from src.evals.final_eval.manifest import utc_now
    return utc_now()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(cal: dict) -> str:
    L = [
        f"# Calibration — {cal['run_id']}",
        "",
        f"- judge `{cal['judge_model']}` (prompt {cal['judge_prompt_version']})",
        f"- {cal['n_cases']} cases, {cal['n_cases_judged']} judged, "
        f"**{cal['n_cases_with_disagreements']} with at least one disagreement**",
        f"- source: {cal['source']}",
        "",
        "> " + cal["adjudication"]["note"],
        "",
        "## Disagreements by kind",
        "",
    ]
    if cal["by_kind"]:
        L += ["| Kind | Count | Suggested category |", "|---|---|---|"]
        seen = {}
        for c in cal["cases"]:
            for f in c["disagreements"]:
                seen[f["kind"]] = f["suggested_category"]
        L += [f"| {k} | {n} | {seen.get(k, '—')} |" for k, n in cal["by_kind"].items()]
    else:
        L.append("None. The deterministic checks, the runtime routing and the judge "
                 "agreed on every case.")

    L += ["", "## Per case", ""]
    for c in cal["cases"]:
        d, j = c["deterministic"], c["judge"]
        flag = "⚠︎" if c["disagreements"] else "·"
        L += [f"### {flag} {c['query_id']} — {c['category']} / {c['answer_type']} "
              f"({c['difficulty']})", ""]
        L.append(f"- deterministic: pass={d['case_pass']} facts={d['matched_facts']} "
                 f"(min {d['min_facts']}) contradictions={d['contradictions']} "
                 f"strict_unmatched={d['strict_unmatched']} tags={d['failure_tags'] or 'none'}")
        L.append(f"- runtime: review={c['runtime']['needs_human_review']} "
                 f"status={c['runtime']['review_status']} "
                 f"reason={c['runtime']['escalation_reason']}")
        if j["status"] == "absent":
            L.append("- judge: absent")
        else:
            L.append(f"- judge: {j['status']} {j['dimensions']} "
                     f"criteria={ {k: v for k, v in (j['criterion_status_counts'] or {}).items() if v} }")
        if c["disagreements"]:
            L.append("")
            for f in c["disagreements"]:
                L += [f"  - **{f['kind']}** → suggested `{f['suggested_category']}`, "
                      f"adjudication: `{f['adjudication'] or 'PENDING'}`",
                      f"    - {f['detail']}"]
        L.append("")
    L += ["---", "",
          "Adjudication vocabulary: " + ", ".join(f"`{c}`" for c in cal["adjudication"]["categories"]),
          "", "Record the decision in `calibration.json` (`adjudication` on each finding). "
              "Nothing in this file changes a measured number.", ""]
    return "\n".join(L)


def run(run_dir, manifest: dict, progress=None) -> dict:
    cal = build(run_dir, manifest, progress=progress)
    run_dir.write_json("calibration", cal)
    run_dir.write_text("calibration_report", build_report(cal))
    return cal
