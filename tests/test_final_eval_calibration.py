"""Calibration: disagreement detection between the three opinions on a case.

No model, no database. Deterministic rows and judge rows are synthesised, so
every detector is pinned to an exact input.
"""

import json

import pytest

from src.evals.final_eval import cases as case_mod
from src.evals.final_eval import manifest as man
from src.evals.final_eval import deterministic as D
from src.evals.final_eval import judge as J
from src.evals.final_eval import calibration as C


@pytest.fixture(scope="module")
def gold():
    return {c.query_id: c for c in case_mod.load_cases()}


@pytest.fixture
def run(tmp_path):
    return man.RunDir("cal-run", tmp_path)


MANIFEST = {"run_id": "cal-run", "judge": {"model": "judge-x", "prompt_version": "aj2"}}


def response(qid, answer, subject, **over):
    row = {"query_id": qid, "subject_id": subject, "status": "completed", "answer": answer,
           "citations": [{"label": "S1", "labels": ["S1"], "chunk_id": 1, "claim": answer,
                          "verified": True}],
           "sources": [{"label": "S1", "chunk_id": 1, "subject_id": subject,
                        "hadm_id": 91000013, "resolution": "note_chunks"}],
           "node_trail": ["triage", "synthesis"], "citation_report": {"bad_labels": []},
           "verification_summary": {"synthesis_failed": False},
           "needs_human_review": False, "review_status": "auto_approved",
           "escalation_reason": "none", "deterministic_lab_path": False,
           "graph_errors": [], "timings": {"total_ms": 100.0}}
    row.update(over)
    return row


def judge_row(case, *, status="ok", statuses=None, dims=None, **over):
    crits = J.criteria_for(case)
    applicable = J.applicable_dimensions(case)
    row = {
        "query_id": case.query_id,
        "status": status,
        "judge_model": "judge-x",
        "judge_prompt_version": "aj2",
        "applicable_dimensions": applicable,
        "n_criteria": len(crits),
        "criteria": [{"id": c.id, "kind": c.kind, "text": c.text} for c in crits],
        "criterion_assessments": [
            {"criterion_id": c.id, "criterion": c.text,
             "status": (statuses or {}).get(c.id, "supported"),
             "evidence_labels": ["S1"], "reason": "stated"} for c in crits],
        "unsupported_content": [],
        "consistency_violations": [],
        "repair_attempted": False,
        "dimensions": {d: {"score": ((dims or {}).get(d, 4) if d in applicable else None),
                           "applicable": d in applicable, "reason": "",
                           "criterion": None, "confidence": None} for d in J.DIMENSIONS},
    }
    row["criterion_status_counts"] = J.criterion_status_counts(row["criterion_assessments"])
    row.update(over)
    return row


def seed(run, gold, pairs):
    """pairs: [(response row, judge row or None)]"""
    run.path.mkdir(parents=True, exist_ok=True)
    index = {}
    for resp, jrow in pairs:
        run.append("responses", resp)
        index[resp["query_id"]] = gold[resp["query_id"]]
        if jrow is not None:
            run.append("judge", jrow)
    D.run(run, index)
    return C.build(run, MANIFEST)


def kinds(cal, qid):
    case = next(c for c in cal["cases"] if c["query_id"] == qid)
    return [d["kind"] for d in case["disagreements"]]


# --- detectors -------------------------------------------------------------
def test_strict_unmatched_but_judge_supported_is_flagged(run, gold):
    """demo_q19's second fact is text-only; an answer that paraphrases it goes
    strict_unmatched while an honest judge may still call it supported."""
    case = gold["demo_q19"]
    answer = ("The dose was intensified from 5 mg/kg because her infusions were "
              "delayed by a coverage problem [S1].")
    cal = seed(run, gold, [(response("demo_q19", answer, 90000015), judge_row(case))])
    assert "strict_unmatched_but_judge_supported" in kinds(cal, "demo_q19")
    f = next(d for d in cal["cases"][0]["disagreements"]
             if d["kind"] == "strict_unmatched_but_judge_supported")
    assert f["suggested_category"] == "deterministic_evaluator_too_strict"
    assert f["adjudication"] is None            # never decided by code


def test_strict_match_but_judge_says_missing_is_flagged_as_too_harsh(run, gold):
    case = gold["demo_q19"]
    answer = ("The dose intensified from 5 mg/kg after an insurance interruption [S1].")
    first = J.criteria_for(case)[0].id
    j = judge_row(case, statuses={first: "missing"}, dims={"completeness": 2})
    cal = seed(run, gold, [(response("demo_q19", answer, 90000015), j)])
    assert "strict_match_but_judge_says_absent" in kinds(cal, "demo_q19")


def test_incomplete_but_judge_completeness_top(run, gold):
    """The demo_q19 aj1 failure: required fact absent, completeness still 4."""
    case = gold["demo_q19"]
    answer = "The dose was increased [S1]."          # matches neither required fact
    cal = seed(run, gold, [(response("demo_q19", answer, 90000015), judge_row(case))])
    ks = kinds(cal, "demo_q19")
    assert "incomplete_but_judge_completeness_top" in ks
    f = next(d for d in cal["cases"][0]["disagreements"]
             if d["kind"] == "incomplete_but_judge_completeness_top")
    assert f["suggested_category"] == "judge_completeness_inflation"


def test_deterministic_contradiction_but_judge_fully_correct(run, gold):
    case = gold["demo_q01"]
    answer = "The most recent creatinine was 2.9 mg/dL [S1]."      # gold is 1.4
    cal = seed(run, gold, [(response("demo_q01", answer, 90000001), judge_row(case))])
    ks = kinds(cal, "demo_q01")
    assert "contradiction_but_judge_fully_correct" in ks
    f = next(d for d in cal["cases"][0]["disagreements"]
             if d["kind"] == "contradiction_but_judge_fully_correct")
    assert f["suggested_category"] == "judge_too_lenient"


def test_temporal_failure_but_judge_temporal_top(run, gold):
    case = gold["demo_q02"]
    answer = "Creatinine was 1.8 mg/dL then 2.1 mg/dL [S1]."       # 1.4 omitted
    cal = seed(run, gold, [(response("demo_q02", answer, 90000001), judge_row(case))])
    ks = kinds(cal, "demo_q02")
    assert "temporal_failure_but_judge_temporal_top" in ks
    f = next(d for d in cal["cases"][0]["disagreements"]
             if d["kind"] == "temporal_failure_but_judge_temporal_top")
    assert f["suggested_category"] == "judge_temporal_misunderstanding"


def test_hallucinated_citation_but_judge_fully_grounded(run, gold):
    case = gold["demo_q01"]
    answer = "The most recent creatinine was 1.4 mg/dL [S1]."
    resp = response("demo_q01", answer, 90000001,
                    citation_report={"bad_labels": ["S7"]})
    cal = seed(run, gold, [(resp, judge_row(case))])
    ks = kinds(cal, "demo_q01")
    assert "hallucinated_citation_but_judge_fully_grounded" in ks
    assert next(d for d in cal["cases"][0]["disagreements"]
                if d["kind"] == "hallucinated_citation_but_judge_fully_grounded"
                )["suggested_category"] == "judge_grounding_error"


def test_auto_approved_despite_review_worthy_finding_is_a_system_failure(run, gold):
    case = gold["demo_q01"]
    answer = "The most recent creatinine was 1.4 mg/dL [S1]."
    resp = response("demo_q01", answer, 90000001,
                    citation_report={"bad_labels": ["S9"]}, needs_human_review=False)
    cal = seed(run, gold, [(resp, judge_row(case))])
    f = next(d for d in cal["cases"][0]["disagreements"]
             if d["kind"] == "auto_approved_despite_review_worthy_finding")
    assert f["suggested_category"] == "legitimate_system_failure"


def test_judge_failure_short_circuits_every_other_judge_comparison(run, gold):
    """An unusable verdict must not also generate 'the judge said X' findings."""
    case = gold["demo_q01"]
    j = judge_row(case, status="judge_inconsistent",
                  consistency_violations=[{"rule": "completeness_inflated", "detail": "d"}])
    answer = "The most recent creatinine was 2.9 mg/dL [S1]."      # would contradict
    cal = seed(run, gold, [(response("demo_q01", answer, 90000001), j)])
    assert kinds(cal, "demo_q01") == ["judge_unusable"]
    f = cal["cases"][0]["disagreements"][0]
    assert f["suggested_category"] == "human_adjudication_required"
    assert f["judge_evidence"]["consistency_violations"]


def test_agreeing_case_produces_no_disagreement(run, gold):
    case = gold["demo_q01"]
    answer = "The most recent creatinine was 1.4 mg/dL [S1]."
    cal = seed(run, gold, [(response("demo_q01", answer, 90000001), judge_row(case))])
    assert kinds(cal, "demo_q01") == []
    assert cal["n_cases_with_disagreements"] == 0


def test_a_run_without_a_judge_reports_absent_not_agreement(run, gold):
    """No judge row means the judge-side comparisons were not evaluated — never
    that the judge agreed. The routing detector compares deterministic findings
    against runtime routing and is judge-independent, so it still fires."""
    case = gold["demo_q01"]
    answer = "The most recent creatinine was 2.9 mg/dL [S1]."      # contradicts gold
    cal = seed(run, gold, [(response("demo_q01", answer, 90000001), None)])
    assert cal["cases"][0]["judge"]["status"] == "absent"
    assert "not evaluated" in cal["cases"][0]["judge"]["note"]
    assert cal["n_cases_judged"] == 0
    ks = kinds(cal, "demo_q01")
    assert ks == ["auto_approved_despite_review_worthy_finding"]   # judge-free detector
    assert not any(k.startswith(("strict_", "contradiction_", "incomplete_", "temporal_"))
                   for k in ks)


# --- contract --------------------------------------------------------------
def test_calibration_never_decides_the_adjudication(run, gold):
    case = gold["demo_q19"]
    cal = seed(run, gold, [(response("demo_q19", "The dose was increased [S1].", 90000015),
                            judge_row(case))])
    for c in cal["cases"]:
        for d in c["disagreements"]:
            assert d["adjudication"] is None
            assert d["suggested_category"] in C.CATEGORIES
    assert cal["adjudication"]["status"] == "pending"
    # A rule can never blame the deterministic evaluator for a bug: code cannot
    # diagnose its own defect, only report that it disagreed.
    assert "deterministic_evaluator_bug" in C.CATEGORIES
    suggested = {d["suggested_category"] for c in cal["cases"] for d in c["disagreements"]}
    assert "deterministic_evaluator_bug" not in suggested


def test_calibration_calls_no_model_and_reads_only_artifacts(run, gold, monkeypatch):
    """Guard rail: if calibration ever grew a model call, this fails."""
    import src.evals.final_eval.judge_backend as jb

    def explode(*a, **k):
        raise AssertionError("calibration must never build a judge backend")

    monkeypatch.setattr(jb, "build_backend", explode)
    monkeypatch.setattr(jb.OllamaJudgeBackend, "complete", explode)
    case = gold["demo_q01"]
    cal = seed(run, gold, [(response("demo_q01", "creatinine 1.4 mg/dL [S1].", 90000001),
                            judge_row(case))])
    assert cal["source"].startswith("this run's existing artifacts only")


def test_calibration_requires_deterministic_results(run, gold):
    run.path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit, match="no deterministic.jsonl"):
        C.build(run, MANIFEST)


def test_artifacts_are_written_and_readable(run, gold):
    case = gold["demo_q19"]
    run.path.mkdir(parents=True, exist_ok=True)
    run.append("responses", response("demo_q19", "The dose was increased [S1].", 90000015))
    run.append("judge", judge_row(case))
    D.run(run, {"demo_q19": case})
    cal = C.run(run, MANIFEST)
    assert run.file("calibration").exists() and run.file("calibration_report").exists()
    assert json.loads(run.file("calibration").read_text())["run_id"] == "cal-run"
    md = run.file("calibration_report").read_text()
    assert "Calibration — cal-run" in md
    assert "PENDING" in md                       # adjudication is visibly unresolved


def test_calibration_output_carries_no_raw_note_text(run, gold):
    case = gold["demo_q01"]
    cal = seed(run, gold, [(response("demo_q01", "creatinine 1.4 mg/dL [S1].", 90000001),
                            judge_row(case))])
    blob = json.dumps(cal)
    assert "DISCHARGE SUMMARY" not in blob.upper()
    assert "text_sha256" not in blob
