"""Preflight (doctor), comparison and API contract cross-check.

Every external dependency is injected, so the whole file runs with no GPU, no
Ollama, no database and no network.
"""

import json

import pytest

from src.evals.final_eval import manifest as man
from src.evals.final_eval import doctor as doc
from src.evals.final_eval import compare as CMP
from src.evals.final_eval import api_crosscheck as X


# ===========================================================================
# doctor
# ===========================================================================
DB_OK = {"connected": True, "database": "lumen_demo", "detail": "connected",
         "counts": {"note_chunks": 464, "labevents": 1359}}
DB_DOWN = {"connected": False, "detail": "OperationalError: refused",
           "database": None, "counts": {}}


def ollama(*models):
    def probe(host):
        return {"reachable": True,
                "models": [{"name": m, "digest": "d" * 64,
                            "details": {"parameter_size": "8B", "quantization_level": "Q4"}}
                           for m in models]}
    return probe


def by_name(rep):
    return {c.name: c for c in rep.checks}


@pytest.fixture
def demo_plane(monkeypatch):
    monkeypatch.setenv("LUMEN_DATA_PLANE", "demo")


def test_doctor_reports_the_dataset_contract(demo_plane, tmp_path):
    rep = doc.run_checks(profile="local", results_root=tmp_path,
                         db_probe=lambda: DB_OK, ollama_probe=ollama())
    c = by_name(rep)
    assert c["eval_set_case_count"].status == doc.PASS
    assert c["eval_set_case_count"].value == 40
    assert c["legacy_subset_count"].value == list(man.case_mod.LEGACY_IDS)
    assert c["demo_manifest_hash_matches"].status == doc.PASS
    assert len(c["eval_set_sha256"].value) == 64


def test_doctor_names_the_legacy_subset_as_historical(demo_plane, tmp_path):
    rep = doc.run_checks(profile="local", results_root=tmp_path,
                         db_probe=lambda: DB_OK, ollama_probe=ollama())
    assert "not a designed statistical sample" in by_name(rep)["legacy_subset_count"].detail


def test_wrong_data_plane_is_a_blocking_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMEN_DATA_PLANE", "research")
    rep = doc.run_checks(profile="final", results_root=tmp_path,
                         db_probe=lambda: DB_OK, ollama_probe=ollama())
    assert by_name(rep)["data_plane_is_demo"].status == doc.FAIL
    assert rep.exit_code == 1
    # Even under --profile local, the plane is never advisory.
    rep = doc.run_checks(profile="local", results_root=tmp_path,
                         db_probe=lambda: DB_OK, ollama_probe=ollama())
    assert by_name(rep)["data_plane_is_demo"].status == doc.FAIL
    assert rep.exit_code == 1


def test_missing_runtime_model_blocks_the_final_profile(demo_plane, tmp_path, monkeypatch):
    from src.llm import local_client
    rep = doc.run_checks(profile="final", results_root=tmp_path, db_probe=lambda: DB_OK,
                         ollama_probe=ollama("qwen2.5:14b"))     # no runtime tiers
    c = by_name(rep)
    assert c["runtime_model_main"].status == doc.FAIL
    assert c["judge_model_installed"].status == doc.PASS
    assert rep.exit_code == 1


def test_local_profile_demotes_machine_dependent_failures(demo_plane, tmp_path):
    rep = doc.run_checks(profile="local", results_root=tmp_path, db_probe=lambda: DB_DOWN,
                         ollama_probe=lambda host: {"reachable": False, "models": [],
                                                    "error": "ConnectionError"})
    c = by_name(rep)
    assert c["database_connectivity"].status == doc.WARN
    assert c["runtime_model_main"].status == doc.WARN
    assert "advisory under --profile local" in c["runtime_model_main"].detail
    # The repository's own contract is still enforced.
    assert c["eval_set_case_count"].status == doc.PASS
    assert rep.exit_code == 0


def test_database_failure_blocks_the_final_profile(demo_plane, tmp_path):
    rep = doc.run_checks(profile="final", results_root=tmp_path, db_probe=lambda: DB_DOWN,
                         ollama_probe=ollama())
    assert by_name(rep)["database_connectivity"].status == doc.FAIL
    assert rep.exit_code == 1


def test_judge_independence_is_checked_against_the_live_runtime(demo_plane, tmp_path,
                                                                monkeypatch):
    """Configuring the judge as the runtime main model is refused, whatever the
    model happens to be called today."""
    from src.llm import local_client
    rep = doc.run_checks(profile="final", judge_model=local_client.MAIN_MODEL,
                         results_root=tmp_path, db_probe=lambda: DB_OK,
                         ollama_probe=ollama(local_client.MAIN_MODEL, local_client.FAST_MODEL))
    c = by_name(rep)
    assert c["judge_independence"].status == doc.FAIL
    assert "runtime" in c["judge_independence"].detail
    assert rep.exit_code == 1


def test_judge_independence_failure_is_never_advisory(demo_plane, tmp_path):
    from src.llm import local_client
    rep = doc.run_checks(profile="local", judge_model=local_client.FAST_MODEL,
                         results_root=tmp_path, db_probe=lambda: DB_OK,
                         ollama_probe=ollama(local_client.FAST_MODEL))
    assert by_name(rep)["judge_independence"].status == doc.FAIL
    assert rep.exit_code == 1


def test_sealed_run_id_is_refused(demo_plane, tmp_path):
    rd = man.RunDir("taken", tmp_path)
    rd.path.mkdir(parents=True)
    rd.mark_complete()
    rep = doc.run_checks(profile="local", results_root=tmp_path, run_id="taken",
                         db_probe=lambda: DB_OK, ollama_probe=ollama())
    c = by_name(rep)["run_id_available"]
    assert c.status == doc.FAIL and "immutable" in c.detail
    assert rep.exit_code == 1


def test_unsealed_run_id_is_a_warning_about_resume(demo_plane, tmp_path):
    rd = man.RunDir("partial", tmp_path)
    rd.path.mkdir(parents=True)
    rd.append("responses", {"query_id": "demo_q01"})
    rep = doc.run_checks(profile="local", results_root=tmp_path, run_id="partial",
                         db_probe=lambda: DB_OK, ollama_probe=ollama())
    c = by_name(rep)["run_id_available"]
    assert c.status == doc.WARN and "--resume" in c.detail
    assert rep.exit_code == 0


def test_free_run_id_passes(demo_plane, tmp_path):
    rep = doc.run_checks(profile="local", results_root=tmp_path, run_id="brand-new",
                         db_probe=lambda: DB_OK, ollama_probe=ollama())
    assert by_name(rep)["run_id_available"].status == doc.PASS


def test_doctor_reports_versions_and_renders(demo_plane, tmp_path):
    rep = doc.run_checks(profile="local", results_root=tmp_path,
                         db_probe=lambda: DB_OK, ollama_probe=ollama())
    c = by_name(rep)
    assert c["judge_prompt_version"].value == "aj2"
    assert c["results_schema_version"].value == man.RESULTS_SCHEMA_VERSION
    text = doc.render(rep)
    assert "final_eval doctor" in text and ("READY" in text or "NOT READY" in text)
    assert json.loads(json.dumps(rep.to_dict()))["profile"] == "local"


def test_doctor_writes_nothing_but_the_write_probe(demo_plane, tmp_path):
    before = set(tmp_path.rglob("*"))
    doc.run_checks(profile="local", results_root=tmp_path,
                   db_probe=lambda: DB_OK, ollama_probe=ollama())
    after = set(tmp_path.rglob("*"))
    # Only the results root itself may be created; no artifact, no run directory.
    assert after - before <= {tmp_path}
    assert not list(tmp_path.glob("*/manifest.json"))


def test_porcelain_paths_survive_the_stripped_first_line():
    """git status output is stripped, so the first line loses its leading blank
    status column. A naive slice would record a filename that does not exist."""
    assert man.porcelain_paths("M src/a.py\n M src/b.py\n?? src/c.py") == \
        ["src/a.py", "src/b.py", "src/c.py"]
    assert man.porcelain_paths("R  old.py -> new.py") == ["new.py"]


# ===========================================================================
# compare
# ===========================================================================
def _summary(**over):
    s = {
        "deterministic_quality": {
            "case_pass_rate": {"n": 8, "d": 10, "rate": 0.8},
            "facts": {"fact_recall": {"n": 20, "d": 25, "rate": 0.8},
                      "contradictions": 1, "strict_unmatched": 4},
            "citations": {"citation_validity_rate": {"rate": 0.95},
                          "citation_coverage": {"rate": 0.9}},
            "temporal_accuracy": {"rate": 0.75},
            "routing": {"review_recall": {"rate": 1.0},
                        "auto_approval_precision": {"rate": 0.9},
                        "review_burden": {"rate": 0.2}},
        },
        "judge_quality": {"dimensions": {"completeness": {"mean": 3.4},
                                         "groundedness": {"mean": 3.8}}},
        "operational": {"n": 15, "successful": 15,
                        "latency_ms": {"mean": 2000.0, "p50": 1800, "p95": 3000},
                        "mean_llm_calls": 2.0, "mean_main_calls": 1.0,
                        "mean_fast_calls": 1.0, "requests_with_zero_llm_calls": 1,
                        "review_status": {"auto_approved": 12, "human_review_required": 3}},
    }
    s.update(over)
    return s


def _man(**over):
    m = {"run_id": "new", "git_sha": "a" * 40, "branch": "eval/final-framework",
         "benchmark": "Lumen Final Answer-Level Quality Baseline v1",
         "results_schema_version": 2,
         "eval_set": {"sha256": "sha-40", "evaluated_ids": [f"demo_q{i:02d}" for i in range(1, 16)]},
         "models": {"main": "qwen3:30b", "fast": "qwen3:4b"},
         "prompt_versions": {"synthesis": "s4"},
         "judge": {"model": "qwen2.5:14b", "prompt_version": "aj2"}}
    m.update(over)
    return m


def _responses(ids, *, total_ms=2000.0, llm_calls=2, reviews=()):
    """Minimal rows in the shape aggregate.operational_section consumes."""
    return [{"query_id": i, "status": "completed",
             "node_trail": ["triage", "synthesis"], "query_complexity": "simple",
             "classified_by": "deterministic", "synthesis_role": "main",
             "review_status": ("human_review_required" if i in reviews else "auto_approved"),
             "timings": {"total_ms": total_ms, "llm_calls": llm_calls,
                         "llm_main_calls": 1, "llm_fast_calls": llm_calls - 1}}
            for i in ids]


LEGACY_15 = list(man.case_mod.LEGACY_IDS)


def _run(**over):
    m = _man(**over.pop("manifest", {}))
    rows = over.pop("responses", None)
    if rows is None:
        rows = _responses((m.get("eval_set") or {}).get("evaluated_ids") or [])
    return {"kind": "final_eval", "manifest": m,
            "summary": _summary(**over.pop("summary", {})),
            "responses": rows, "path": "/runs/new"}


LEGACY = {"kind": "cloud_eval", "path": "/old/performance.json", "data": {
    "run_id": "cloud-1", "generated_at": "2026-01-01T00:00:00+00:00",
    "requests": [], "summary": {"n": 15, "successful": 15, "mean_ms": 3500.0,
                                "p50_ms": 3200, "p95_ms": 5000, "mean_llm_calls": 4.0,
                                "mean_main_calls": 2.0, "mean_fast_calls": 2.0,
                                "requests_with_zero_llm_calls": 0,
                                "human_review_required": 2},
    "legacy_subset": {"ids": list(man.case_mod.LEGACY_IDS),
                      "evaluated": list(man.case_mod.LEGACY_IDS),
                      "summary": {"n": 15, "successful": 15, "mean_ms": 3500.0,
                                  "p50_ms": 3200, "p95_ms": 5000, "mean_llm_calls": 4.0,
                                  "mean_main_calls": 2.0, "mean_fast_calls": 2.0,
                                  "requests_with_zero_llm_calls": 0,
                                  "human_review_required": 2}},
    "models": {"main": "qwen3:14b", "fast": "qwen3:4b"}}}


def test_legacy_comparison_refuses_every_quality_delta():
    c = CMP.compare(_run(), LEGACY)
    assert c["mode"] == "legacy"
    assert c["quality"] is None
    refusal = next(r for r in c["refused"] if r["comparison"] == "answer quality")
    assert "no answer-level quality metric" in refusal["reason"]
    for metric in ("fact recall", "citation validity", "every judge dimension"):
        assert metric in refusal["metrics"]


def test_legacy_comparison_still_computes_the_shared_operational_metrics():
    c = CMP.compare(_run(), LEGACY)
    assert len(c["shared_case_ids"]) == 15
    assert "15 shared ids" in c["scope"]["new"]
    assert "15 shared ids" in c["scope"]["baseline"]
    assert c["operational"]["mean_ms"] == {"new": 2000.0, "old": 3500.0, "delta": -1500.0}
    assert c["operational"]["mean_llm_calls"]["delta"] == -2.0


def test_operational_comparison_is_refused_when_the_scopes_cannot_be_aligned():
    """The smoke case: a 3-case run against a 15-case legacy artifact. The two
    sides would be computed over different questions, so no delta is printed."""
    new = _run(manifest={"eval_set": {"sha256": "sha-40",
                                      "evaluated_ids": ["demo_q01", "demo_q02", "demo_q19"]}})
    c = CMP.compare(new, LEGACY)
    assert c["operational"] is None
    r = next(r for r in c["refused"] if r["comparison"] == "operational metrics")
    assert "different question sets" in r["reason"]
    assert "REFUSED" in CMP.render(c)


def test_operational_comparison_is_refused_when_raw_rows_are_unavailable():
    new = _run(responses=[])                       # nothing to re-project
    c = CMP.compare(new, LEGACY)
    assert c["operational"] is None


def test_legacy_comparison_records_the_configuration_difference():
    c = CMP.compare(_run(), LEGACY)
    assert c["configuration_differences"]["model_main"] == \
        {"new": "qwen3:30b", "old": "qwen3:14b"}


def test_comparison_is_refused_outright_when_no_case_is_shared():
    new = _run(manifest={"eval_set": {"sha256": "x", "evaluated_ids": ["other_q01"]}})
    with pytest.raises(CMP.IncomparableRuns, match="share no case ids"):
        CMP.compare(new, LEGACY)


def test_a_different_eval_set_refuses_quality_deltas():
    old = {"kind": "final_eval", "path": "/runs/old",
           "manifest": _man(run_id="old", eval_set={"sha256": "sha-15",
                                                    "evaluated_ids": ["demo_q01"]}),
           "summary": _summary(), "responses": _responses(["demo_q01"])}
    c = CMP.compare(_run(), old)
    assert c["quality"] is None
    assert any("different evaluation sets" in r["reason"] for r in c["refused"])


def test_a_different_judge_prompt_version_refuses_only_the_judge_deltas():
    old = {"kind": "final_eval", "path": "/runs/old",
           "manifest": _man(run_id="old", judge={"model": "qwen2.5:14b",
                                                 "prompt_version": "aj1"}),
           "summary": _summary(), "responses": _responses(LEGACY_15)}
    c = CMP.compare(_run(), old)
    assert c["quality"] is not None                     # deterministic deltas survive
    assert c["quality"]["case_pass_rate"]["delta"] == 0.0
    assert c["quality"]["judge"] is None                # judge deltas do not
    assert any(r["comparison"] == "judge dimensions" for r in c["refused"])


def test_two_compatible_runs_compare_fully():
    old = {"kind": "final_eval", "path": "/runs/old", "manifest": _man(run_id="old"),
           "summary": _summary(),
           "responses": _responses(LEGACY_15, total_ms=2500.0, llm_calls=3)}
    c = CMP.compare(_run(), old)
    assert c["mode"] == "baseline"
    assert c["quality"]["judge"]["completeness"]["delta"] == 0.0
    assert c["operational"]["mean_ms"]["delta"] == -500.0
    assert c["operational"]["mean_llm_calls"]["delta"] == -1.0
    assert c["refused"] == []
    assert c["read_only"] is True


def test_different_case_sets_refuse_quality_deltas_between_two_final_runs():
    """summary.json's quality section is whole-run, so a delta across different
    case sets would partly measure which questions each run asked."""
    old = {"kind": "final_eval", "path": "/runs/old",
           "manifest": _man(run_id="old",
                            eval_set={"sha256": "sha-40",
                                      "evaluated_ids": LEGACY_15 + ["demo_q16"]}),
           "summary": _summary(), "responses": _responses(LEGACY_15 + ["demo_q16"])}
    c = CMP.compare(_run(), old)
    assert c["quality"] is None
    assert any("different case sets" in r["reason"] for r in c["refused"])
    assert c["operational"] is not None          # still aligned on the 15 shared ids


def test_comparison_refuses_an_unscored_run(tmp_path):
    rd = man.RunDir("unscored", tmp_path)
    rd.path.mkdir(parents=True)
    with pytest.raises(CMP.IncomparableRuns, match="no summary.json"):
        CMP.load_run(rd)


def test_a_non_cloud_eval_json_is_rejected(tmp_path):
    p = tmp_path / "something.json"
    p.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(CMP.IncomparableRuns, match="not a cloud_eval performance artifact"):
        CMP.load_legacy(p)


def test_render_names_the_refusals():
    text = CMP.render(CMP.compare(_run(), LEGACY))
    assert "REFUSED" in text and "not computed" in text


# ===========================================================================
# api cross-check
# ===========================================================================
def graph_row(qid="demo_q01", answer="Creatinine was 1.4 mg/dL on 2024-03-18 [S1].",
              **over):
    row = {"query_id": qid, "subject_id": 90000001, "query": "q", "status": "completed",
           "answer": answer,
           "citations": [{"label": "S1", "labels": ["S1"], "claim": answer, "verified": True}],
           "needs_human_review": False}
    row.update(over)
    return row


def api_payload(answer="Creatinine was 1.4 mg/dL on 2024-03-18 [S1].", **over):
    p = {k: None for k in X.REQUIRED_RESPONSE_KEYS}
    p.update({"status": "completed", "answer": answer,
              "citations": [{"label": "S1", "labels": ["S1"]}],
              "needs_human_review": False, "sources": [], "node_trail": [],
              "models": {}, "timings": {}, "latency_ms": 1.0})
    p.update(over)
    return p


def test_identical_contract_passes():
    out = X.compare_one(graph_row(), 200, api_payload())
    assert out["contract_ok"] and out["findings"] == []
    assert out["checks"]["citation_labels_match"] is True


def test_reworded_answer_with_the_same_facts_is_not_a_failure():
    """Byte-identical prose is never required."""
    out = X.compare_one(
        graph_row(),
        200,
        api_payload("On 2024-03-18 the creatinine measured 1.4 mg/dL [S1]."))
    assert out["contract_ok"]
    assert out["checks"]["numeric_anchors_match"] is True
    assert out["answer_consistency"]["content_word_overlap"] is not None


def test_a_different_value_is_a_finding():
    out = X.compare_one(graph_row(), 200,
                        api_payload("Creatinine was 2.9 mg/dL on 2024-03-18 [S1]."))
    assert not out["contract_ok"]
    assert any("numeric anchors differ" in f for f in out["findings"])


def test_citation_label_divergence_is_a_finding():
    out = X.compare_one(graph_row(), 200,
                        api_payload(citations=[{"label": "S2", "labels": ["S2"]}]))
    assert out["checks"]["citation_labels_match"] is False


def test_missing_schema_keys_are_reported():
    payload = api_payload()
    payload.pop("review_status")
    out = X.compare_one(graph_row(), 200, payload)
    assert out["missing_response_keys"] == ["review_status"]
    assert not out["contract_ok"]


def test_status_mismatch_is_a_finding():
    out = X.compare_one(graph_row(), 200, api_payload(status="refused"))
    assert out["checks"]["status_matches"] is False


def test_review_routing_divergence_is_a_finding():
    out = X.compare_one(graph_row(needs_human_review=True), 200,
                        api_payload(needs_human_review=False))
    assert out["checks"]["review_status_matches"] is False


def test_a_failed_graph_status_must_surface_as_5xx():
    ok = X.compare_one(graph_row(status="failed", answer=""), 500, {})
    assert ok["checks"]["http_semantics"] is True
    bad = X.compare_one(graph_row(status="failed", answer=""), 200, api_payload())
    assert bad["checks"]["http_semantics"] is False


def test_crosscheck_writes_its_own_artifact_and_never_a_score(tmp_path):
    rd = man.RunDir("x-run", tmp_path)
    rd.path.mkdir(parents=True)
    rd.append("responses", graph_row())
    rd.append("responses", graph_row("demo_q02", "Values 1.8 mg/dL then 2.1 mg/dL [S1]."))

    def caller(subject_id, query, request_id):
        return 200, api_payload()

    rep = X.run(rd, "http://api", caller=caller)
    assert rd.file("api_crosscheck").exists()
    assert rep["n_cases"] == 2 and rep["n_divergent"] == 1     # q02 text differs
    assert not rd.file("summary").exists()                    # never touches the scorecard
    assert "not a prerequisite" in rep["scope"]


def test_evaluator_errors_are_not_cross_checked(tmp_path):
    rd = man.RunDir("x-run2", tmp_path)
    rd.path.mkdir(parents=True)
    rd.append("responses", graph_row(status="evaluator_error", answer=""))
    rep = X.run(rd, "http://api", caller=lambda *a: (200, api_payload()))
    assert rep["n_cases"] == 0


def test_transport_failure_is_recorded_not_raised(tmp_path):
    rd = man.RunDir("x-run3", tmp_path)
    rd.path.mkdir(parents=True)
    rd.append("responses", graph_row())

    def boom(*a):
        raise ConnectionError("no route to host")

    rep = X.run(rd, "http://api", caller=boom)
    assert rep["transport_errors"][0]["query_id"] == "demo_q01"
    assert rep["n_cases"] == 0
