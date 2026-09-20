"""Run directory, manifest, resume, aggregation and failure taxonomy.

No DB and no models: responses are synthesised, so every rule about
immutability, denominators and gates is pinned deterministically.
"""

import json

import pytest

from src.evals.final_eval import cases as case_mod
from src.evals.final_eval import manifest as man
from src.evals.final_eval import aggregate as agg
from src.evals.final_eval import failures as fail_mod
from src.evals.final_eval import deterministic as D


@pytest.fixture(scope="module")
def gold():
    return {c.query_id: c for c in case_mod.load_cases()}


@pytest.fixture
def run(tmp_path):
    return man.RunDir("test-run", tmp_path)


def fake_manifest(ids, **over):
    m = {"run_id": "test-run", "benchmark": "Lumen Final Answer-Level Quality Baseline v1",
         "git_sha": "a" * 40, "branch": "main", "dirty_worktree": False,
         "collection_backend": "graph:run_once",
         "eval_set": {"path": "src/demo_data/golden_qa.json", "sha256": "abc",
                      "manifest_sha256_matches": True, "dataset_version": "lumen-demo-v1",
                      "subset": "smoke", "n_evaluated": len(ids), "evaluated_ids": ids},
         "models": {"main": "m", "fast": "f"},
         "prompt_versions": {"synthesis": "s4", "verify_batch": "vb2"},
         "judge": {"model": "judge-x", "backend": "callable", "prompt_version": "aj1"}}
    m.update(over)
    return m


# --- run directory ---------------------------------------------------------
def test_manifest_written_once_and_run_created(run):
    m = run.open_for_write(fake_manifest(["demo_q01"]))
    assert run.file("manifest").exists()
    assert m["run_id"] == "test-run"


def test_refuses_to_overwrite_a_non_empty_run_without_resume(run):
    run.open_for_write(fake_manifest(["demo_q01"]))
    run.append("responses", {"query_id": "demo_q01"})
    with pytest.raises(man.RunDirError, match="Refusing to overwrite"):
        run.open_for_write(fake_manifest(["demo_q01"]))


def test_completed_run_is_immutable_even_with_resume(run):
    run.open_for_write(fake_manifest(["demo_q01"]))
    run.mark_complete()
    assert run.is_complete()
    with pytest.raises(man.RunDirError, match="immutable"):
        run.open_for_write(fake_manifest(["demo_q01"]), resume=True)


def test_resume_keeps_the_original_manifest_and_records_drift(run):
    original = fake_manifest(["demo_q01"])
    run.open_for_write(original)
    run.append("responses", {"query_id": "demo_q01"})
    changed = fake_manifest(["demo_q01"], git_sha="b" * 40)
    live = run.open_for_write(changed, resume=True)
    assert live["git_sha"] == "a" * 40                 # original provenance wins
    assert len(live["resumes"]) == 1
    assert live["resumes"][0]["drift"]["git_sha"]["now"] == "b" * 40


def test_resume_skips_already_collected_ids(run):
    run.open_for_write(fake_manifest(["demo_q01", "demo_q02"]))
    run.append("responses", {"query_id": "demo_q01"})
    assert run.completed_ids("responses") == {"demo_q01"}


def test_half_written_final_line_is_ignored_and_will_be_regenerated(run):
    run.open_for_write(fake_manifest(["demo_q01", "demo_q02"]))
    run.append("responses", {"query_id": "demo_q01"})
    with open(run.file("responses"), "a") as f:
        f.write('{"query_id": "demo_q02", "answ')      # interrupted mid-write
    assert run.completed_ids("responses") == {"demo_q01"}


def test_published_artifacts_exclude_the_evidence_working_file():
    assert "evidence_cache" not in man.PUBLISHED_ARTIFACTS
    assert set(man.PUBLISHED_ARTIFACTS) <= set(man.ARTIFACTS)


def test_manifest_records_no_secrets(run):
    m = run.open_for_write(fake_manifest(["demo_q01"]))
    blob = json.dumps(m).lower()
    for secret in ("password", "secret_key", "sk-", "postgresql://", "api_key"):
        assert secret not in blob


# --- end-to-end aggregation ------------------------------------------------
def _row(qid, answer, subject, *, needs_review=False, review_status="auto_approved",
         labels=("S1",), bad_labels=(), status="completed"):
    return {"query_id": qid, "subject_id": subject, "status": status, "answer": answer,
            "citations": [{"label": l, "labels": [l], "chunk_id": 1, "claim": answer,
                           "verified": True} for l in labels],
            "sources": [{"label": l, "chunk_id": 1, "subject_id": subject,
                         "hadm_id": 91000013, "resolution": "note_chunks"} for l in labels],
            "node_trail": ["triage", "synthesis"], "citation_report": {"bad_labels": list(bad_labels)},
            "verification_summary": {"synthesis_failed": False},
            "needs_human_review": needs_review, "review_status": review_status,
            "escalation_reason": "none" if not needs_review else "uncited_claim",
            "deterministic_lab_path": False, "graph_errors": [],
            "timings": {"total_ms": 120.0, "llm_calls": 1}}


def _seed(run, gold, rows):
    ids = [r["query_id"] for r in rows]
    m = run.open_for_write(fake_manifest(ids))
    for r in rows:
        run.append("responses", r)
    index = {q: gold[q] for q in ids}
    D.run(run, index)
    return m


def test_summary_is_built_only_from_this_runs_artifacts(run, gold):
    rows = [_row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001)]
    m = _seed(run, gold, rows)
    s = agg.build_summary(run, m)
    assert s["deterministic_quality"]["n_cases"] == 1
    assert s["deterministic_quality"]["case_pass_rate"]["rate"] == 1.0
    assert s["reliability"]["all_gates_pass"] is True
    assert "no_composite_score" in s


def test_failed_cases_stay_in_the_denominator(run, gold):
    rows = [_row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001),
            _row("demo_q02", "", 90000001, status="evaluator_error")]
    m = _seed(run, gold, rows)
    d = agg.build_summary(run, m)["deterministic_quality"]
    assert d["n_cases"] == 2
    assert d["completion_rate"] == {"n": 1, "d": 2, "rate": 0.5}
    assert d["evaluator_error_rate"]["n"] == 1


def test_evaluator_error_fails_the_hidden_error_gate(run, gold):
    rows = [_row("demo_q01", "", 90000001, status="evaluator_error")]
    m = _seed(run, gold, rows)
    rel = agg.build_summary(run, m)["reliability"]
    assert rel["all_gates_pass"] is False
    assert rel["gates"]["no_hidden_evaluator_errors"]["pass"] is False


def test_leakage_fails_the_hard_gate(run, gold):
    r = _row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001)
    r["sources"][0]["subject_id"] = 90000099
    m = _seed(run, gold, [r])
    rel = agg.build_summary(run, m)["reliability"]
    assert rel["gates"]["zero_cross_patient_leakage"]["pass"] is False
    assert rel["all_gates_pass"] is False


def test_review_recall_is_null_when_nothing_is_review_worthy(run, gold):
    """No review-worthy cases must yield null, never 0% and never 100%."""
    rows = [_row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001)]
    m = _seed(run, gold, rows)
    r = agg.build_summary(run, m)["deterministic_quality"]["routing"]
    assert r["review_recall"] == {"n": 0, "d": 0, "rate": None}
    assert r["auto_approval_precision"]["rate"] == 1.0


def test_review_recall_counts_only_gold_derived_findings(run, gold):
    """One genuinely bad answer auto-approved, one clean answer the runtime
    escalated. Recall must be 0/1, and burden 1/2 — they are different things."""
    bad = _row("demo_q01", "The creatinine was 2.6 mg/dL [S1].", 90000001)
    clean = _row("demo_q40", "Tenofovir alafenamide 25 mg daily was started [S1].",
                 90000030, needs_review=True, review_status="pending")
    m = _seed(run, gold, [bad, clean])
    r = agg.build_summary(run, m)["deterministic_quality"]["routing"]
    assert r["review_recall"] == {"n": 0, "d": 1, "rate": 0.0}
    assert r["review_worthy_ids"] == ["demo_q01"]
    assert r["review_burden"] == {"n": 1, "d": 2, "rate": 0.5}
    assert r["auto_approval_precision"] == {"n": 0, "d": 1, "rate": 0.0}


def test_unsafe_auto_approval_gate_catches_the_miss(run, gold):
    bad = _row("demo_q01", "The creatinine was 2.6 mg/dL [S1].", 90000001)
    m = _seed(run, gold, [bad])
    rel = agg.build_summary(run, m)["reliability"]
    assert rel["gates"]["no_unsafe_auto_approval"]["pass"] is False
    assert rel["gates"]["no_unsafe_auto_approval"]["offenders"] == ["demo_q01"]


def test_missing_and_duplicate_cases_are_reported(run, gold):
    m = run.open_for_write(fake_manifest(["demo_q01", "demo_q02", "demo_q19"]))
    r = _row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001)
    run.append("responses", r)
    run.append("responses", dict(r))                    # duplicate
    D.run(run, {"demo_q01": gold["demo_q01"]})
    rel = agg.build_summary(run, m)["reliability"]
    assert rel["gates"]["no_duplicate_query_ids"]["offenders"] == ["demo_q01"]
    assert rel["gates"]["no_missing_cases"]["offenders"] == ["demo_q02", "demo_q19"]


# --- judge aggregation -----------------------------------------------------
def _judge_row(qid, dims, status="ok", applicable=("factual_correctness", "groundedness")):
    return {"query_id": qid, "status": status, "judge_model": "judge-x",
            "applicable_dimensions": list(applicable), "cached": False,
            "dimensions": {d: {"score": dims.get(d), "applicable": dims.get(d) is not None,
                               "reason": "", "criterion": None, "confidence": None}
                           for d in ("factual_correctness", "groundedness", "completeness",
                                     "answer_relevance", "temporal_correctness",
                                     "abstention_quality")}}


def test_judge_failures_are_counted_not_scored(run, gold):
    rows = [_row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001)]
    m = _seed(run, gold, rows)
    run.append("judge", _judge_row("demo_q01", {}, status="backend_error"))
    j = agg.build_summary(run, m)["judge_quality"]
    assert j["judge_failures"] == 1
    assert j["judge_failure_ids"] == ["demo_q01"]
    assert j["dimensions"]["groundedness"]["mean"] is None      # never 0
    assert j["dimensions"]["groundedness"]["n_scored"] == 0


def test_judge_means_use_only_scored_applicable_dimensions(run, gold):
    rows = [_row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001),
            _row("demo_q40", "Tenofovir alafenamide 25 mg daily was started [S1].", 90000030)]
    m = _seed(run, gold, rows)
    run.append("judge", _judge_row("demo_q01", {"factual_correctness": 4, "groundedness": 2}))
    run.append("judge", _judge_row("demo_q40", {"factual_correctness": 3, "groundedness": 4}))
    j = agg.build_summary(run, m)["judge_quality"]
    assert j["dimensions"]["groundedness"]["mean"] == 3.0
    assert j["dimensions"]["groundedness"]["distribution"] == {"0": 0, "1": 0, "2": 1,
                                                               "3": 0, "4": 1}
    assert j["dimensions"]["completeness"]["n_scored"] == 0


# --- failure taxonomy ------------------------------------------------------
def test_one_case_can_carry_several_failure_tags(run, gold):
    bad = _row("demo_q01", "The creatinine was 2.6 mg/dL [S1]. Renal function is poor.",
               90000001, bad_labels=["S9"])
    _seed(run, gold, [bad])
    counts = fail_mod.build(run)
    tags = set(counts["by_tag"])
    assert {"incorrect_value", "invalid_citation_label", "missing_citation",
            "incorrect_temporal_interpretation", "review_routing_miss"} <= tags
    assert counts["cases_with_failures"] == 1
    rows = run.read_jsonl("failures")
    assert all(r["query_id"] == "demo_q01" for r in rows)


def test_failures_carry_no_raw_note_text(run, gold):
    bad = _row("demo_q01", "The creatinine was 2.6 mg/dL [S1].", 90000001)
    _seed(run, gold, [bad])
    fail_mod.build(run)
    blob = run.file("failures").read_text()
    assert "note_chunks" not in blob
    assert len(blob) < 20000


def test_report_renders_with_small_denominators_visible(run, gold):
    rows = [_row("demo_q21", "The available records do not contain enough information "
                             "to answer this.", 90000017, labels=())]
    m = _seed(run, gold, rows)
    s = agg.build_summary(run, m)
    text = agg.build_report(s, fail_mod.build(run), m)
    assert "Lumen Final Answer-Level Quality Baseline v1" in text
    assert "1/1" in text                                 # abstention printed as k/n
    assert "No composite score is produced." in text


def test_small_denominator_note_reflects_this_runs_counts(run, gold):
    """The caveat must name the run's real denominators, not a fixed sentence."""
    rows = [_row("demo_q21", "The available records do not contain enough information "
                             "to answer this.", 90000017, labels=())]
    m = _seed(run, gold, rows)
    text = agg.build_report(agg.build_summary(run, m), fail_mod.build(run), m)
    assert "unsupported abstention n=1" in text
    assert "ambiguity handling" not in text.split("Reading notes")[1]   # none applicable


def test_collect_may_not_add_to_a_sealed_run(run, gold, monkeypatch, tmp_path):
    """Raw capture costs model calls and is not bit-reproducible, so it must
    never be appended to a sealed run — unlike the derived scorecard."""
    import sys
    sys.path.insert(0, str(man.ROOT / "scripts"))
    import final_eval as cli

    run.open_for_write(fake_manifest(["demo_q01"]))
    run.mark_complete()
    monkeypatch.setenv("LUMEN_DATA_PLANE", "demo")
    args = type("A", (), {"run_id": "test-run", "results_root": str(tmp_path),
                          "resume": False, "ids": ["demo_q01"], "subset": "all"})()
    with pytest.raises(SystemExit, match="complete and immutable"):
        cli.cmd_collect(args)


def test_rescoring_a_sealed_run_is_recorded(run, gold):
    rows = [_row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001)]
    m = _seed(run, gold, rows)
    run.mark_complete()
    s = agg.build_summary(run, m)
    s["derivation"] = {"regenerated_after_seal": run.is_complete()}
    assert s["derivation"]["regenerated_after_seal"] is True


# ===========================================================================
# aj2 in aggregation and the failure taxonomy
# ===========================================================================
def _aj2_judge_row(qid, dims, *, status="ok", statuses=("supported",),
                   applicable=("factual_correctness", "groundedness", "completeness"),
                   **over):
    crits = [{"id": f"c{i}", "kind": "fact", "text": f"fact {i}"}
             for i in range(1, len(statuses) + 1)]
    assessments = [{"criterion_id": c["id"], "criterion": c["text"], "status": s,
                    "evidence_labels": ["S1"], "reason": ""}
                   for c, s in zip(crits, statuses)]
    row = {**_judge_row(qid, dims, status=status, applicable=applicable),
           "criteria": crits, "n_criteria": len(crits),
           "criterion_assessments": assessments,
           "criterion_status_counts": {
               s: sum(1 for a in assessments if a["status"] == s)
               for s in ("supported", "partially_supported", "missing",
                         "contradicted", "not_applicable")},
           "unsupported_content": [], "consistency_violations": [],
           "repair_attempted": False}
    row.update(over)
    return row


def test_an_inconsistent_verdict_contributes_no_score_to_any_mean(run, gold):
    """Its numbers are preserved on the row, but averaging a verdict the judge
    disowned would launder an evaluation failure into a measurement."""
    rows = [_row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001),
            _row("demo_q40", "Tenofovir alafenamide 25 mg daily was started [S1].", 90000030)]
    m = _seed(run, gold, rows)
    run.append("judge", _aj2_judge_row("demo_q01", {"groundedness": 2}))
    run.append("judge", _aj2_judge_row(
        "demo_q40", {"groundedness": 4, "completeness": 4}, status="judge_inconsistent",
        statuses=("missing",),
        consistency_violations=[{"rule": "completeness_inflated", "detail": "d"}],
        repair_attempted=True))
    j = agg.build_summary(run, m)["judge_quality"]
    assert j["judge_inconsistent"] == 1
    assert j["judge_inconsistent_ids"] == ["demo_q40"]
    assert j["judge_failures"] == 1
    assert j["judge_repairs_attempted"] == 1
    assert j["consistency_rules_violated"] == {"completeness_inflated": 1}
    # Only demo_q01's 2 is averaged; the disowned 4 is excluded, not clamped.
    assert j["dimensions"]["groundedness"]["mean"] == 2.0
    assert j["dimensions"]["groundedness"]["n_scored"] == 1


def test_criterion_assessment_statistics_are_reported(run, gold):
    rows = [_row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001)]
    m = _seed(run, gold, rows)
    run.append("judge", _aj2_judge_row(
        "demo_q01", {"completeness": 2},
        statuses=("supported", "missing", "partially_supported")))
    ca = agg.build_summary(run, m)["judge_quality"]["criterion_assessments"]
    assert ca["n_criteria_supplied"] == 3
    assert ca["n_assessed"] == 3 and ca["n_unassessed"] == 0
    assert ca["by_status"]["missing"] == 1
    assert ca["by_status"]["partially_supported"] == 1
    assert ca["cases_with_unmet_criteria"] == 1


def test_judge_failure_statuses_get_their_own_tags(run, gold):
    rows = [_row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001),
            _row("demo_q02", "Creatinine went 1.8 mg/dL, 2.1 mg/dL, 1.4 mg/dL [S1].",
                 90000001),
            _row("demo_q19", "Infliximab was intensified from 5 mg/kg after an "
                             "insurance interruption [S1].", 90000015)]
    _seed(run, gold, rows)
    run.append("judge", _aj2_judge_row("demo_q01", {}, status="backend_error"))
    run.append("judge", _aj2_judge_row("demo_q02", {}, status="parse_error"))
    run.append("judge", _aj2_judge_row("demo_q19", {"completeness": 4},
                                       status="judge_inconsistent", statuses=("missing",)))
    tags = fail_mod.build(run)["by_tag"]
    assert tags["judge_error"] == 1
    assert tags["judge_parse_error"] == 1
    assert tags["judge_inconsistent"] == 1
    assert "judge_failure" not in tags              # the generic tag is gone


def test_a_disowned_verdict_contributes_no_low_score_findings(run, gold):
    """A verdict the judge contradicted itself on must not ALSO be mined for
    'judge says groundedness is 1' findings."""
    rows = [_row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001)]
    _seed(run, gold, rows)
    run.append("judge", _aj2_judge_row("demo_q01", {"groundedness": 1, "completeness": 4},
                                       status="judge_inconsistent", statuses=("missing",)))
    tags = fail_mod.build(run)["by_tag"]
    assert tags["judge_inconsistent"] == 1
    assert "judge_low_groundedness" not in tags


def test_every_emitted_tag_is_documented_in_the_taxonomy(run, gold):
    """The vocabulary in failures.TAXONOMY must cover what the code emits."""
    rows = [_row("demo_q01", "The creatinine was 2.6 mg/dL [S1]. Renal function is poor.",
                 90000001, bad_labels=["S9"]),
            _row("demo_q21", "Her ejection fraction was 45 percent [S1].", 90000017),
            _row("demo_q02", "", 90000001, status="failed")]
    _seed(run, gold, rows)
    run.append("judge", _aj2_judge_row("demo_q01", {}, status="parse_error"))
    emitted = set(fail_mod.build(run)["by_tag"])
    assert emitted, "the fixture should produce findings"
    undocumented = emitted - set(fail_mod.TAXONOMY) - set(fail_mod.JUDGE_TAGS.values())
    assert undocumented == set()


def test_admission_scope_violation_is_tagged(run, gold):
    """Out-of-scope evidence is reported in its own right; it is deliberately
    NOT folded into the deterministic case pass."""
    bad = _row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001)
    bad["sources"][0]["hadm_id"] = 99999999            # outside the gold admissions
    _seed(run, gold, [bad])
    det = run.read_jsonl("deterministic")[0]
    assert det["admission_scope"]["pass"] is False
    assert "admission_scope_violation" in det["failure_tags"]
    assert det["case_pass"] is True                     # reported, not a pass component
    assert "admission_scope_violation" in fail_mod.build(run)["by_tag"]


def test_graph_error_with_an_answer_is_tagged_execution_error(run, gold):
    row = _row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001)
    row["graph_errors"] = ["retrieval: reranker timeout, fell back to RRF"]
    _seed(run, gold, [row])
    det = run.read_jsonl("deterministic")[0]
    assert "execution_error" in det["failure_tags"]
    assert "synthesis_failure" not in det["failure_tags"]


def test_a_label_surviving_into_the_stored_answer_is_tagged_separately(run, gold):
    """post-strip is expected to be 0; if one ever survives, a reader sees it."""
    row = _row("demo_q01", "The creatinine was 1.4 mg/dL [S1][S9].", 90000001)
    _seed(run, gold, [row])
    det = run.read_jsonl("deterministic")[0]
    assert det["citations"]["hallucinated_labels_post_strip"] == ["S9"]
    assert "invalid_visible_citation" in det["failure_tags"]


def test_manifest_and_summary_carry_the_results_schema_version(run, gold):
    rows = [_row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001)]
    m = _seed(run, gold, rows)
    m["results_schema_version"] = man.RESULTS_SCHEMA_VERSION
    m["judge"] = {"model": "judge-x", "prompt_version": "aj2"}
    s = agg.build_summary(run, m)
    assert s["results_schema_version"] == man.RESULTS_SCHEMA_VERSION
    assert s["judge_prompt_version"] == "aj2"


def test_report_shows_the_criterion_statistics(run, gold):
    rows = [_row("demo_q01", "The creatinine was 1.4 mg/dL [S1].", 90000001)]
    m = _seed(run, gold, rows)
    run.append("judge", _aj2_judge_row("demo_q01", {"completeness": 2},
                                       statuses=("supported", "missing")))
    text = agg.build_report(agg.build_summary(run, m), fail_mod.build(run), m)
    assert "Criterion assessments (aj2)" in text
    assert "criteria supplied 2" in text
    assert "missing 1" in text
