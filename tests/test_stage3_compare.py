"""Stage 3 comparison harness (data-foundation plan, E15). Pure logic only: the
gate, the label rule for v2 chunks and the per-case verdict. No database, no model."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import stage3_compare as s3   # noqa: E402
import stage3_report as rep   # noqa: E402


def _set(control, candidate, **extra):
    side = lambda correct: {"correct": correct, "questions": 20, "cases": 75, "wrong_not_sent_to_review": 1,      # noqa: E731
                            "unsafe_autoapproved_claims": 0, "latency": {"median_ms": 1000.0}, **extra}
    return {"control": side(control), "candidate": side(candidate)}


def _all(**over):
    sets = {"retrieval_42": _set(40, 40), "temporal_11": _set(11, 11), "holdout_75": _set(75, 75), "targeted_heldout": _set(10, 13)}
    sets.update(over)
    return sets


def _results(sets):
    g = rep.gate(sets)
    return g["final"], {c["condition"]: c["result"] for c in g["conditions"]}


def test_gate_passes_only_when_every_condition_holds():
    final, conditions = _results(_all())
    assert final == "PASS" and set(conditions.values()) == {"PASS"} and len(conditions) == 10


@pytest.mark.parametrize("name, control, candidate, passes", [
    ("retrieval_42", 40, 39, True), ("retrieval_42", 40, 38, False),          # loses at most one
    ("temporal_11", 11, 10, True), ("temporal_11", 11, 9, False),
    ("holdout_75", 75, 74, True), ("holdout_75", 75, 73, False),
    ("targeted_heldout", 10, 13, True), ("targeted_heldout", 10, 12, False),  # gains at least three
])
def test_regression_and_targeted_thresholds_are_exact(name, control, candidate, passes):
    final, _ = _results(_all(**{name: _set(control, candidate)}))
    assert (final == "PASS") is passes


def test_one_bad_set_is_never_averaged_away():
    sets = _all(retrieval_42=_set(40, 42), holdout_75=_set(75, 73))           # +2 on one set does not buy -2 on another
    final, conditions = _results(sets)
    assert final == "FAIL" and conditions["75-case holdout: candidate loses at most 1 correct"] == "FAIL"


def test_safety_fails_when_wrong_answers_not_sent_to_review_rise():
    sets = _all()
    sets["targeted_heldout"]["candidate"]["wrong_not_sent_to_review"] = 2
    assert _results(sets)[0] == "FAIL"
    sets = _all()
    sets["holdout_75"]["candidate"]["unsafe_autoapproved_claims"] = 1
    assert _results(sets)[0] == "FAIL"


@pytest.mark.parametrize("candidate_ms, passes", [(1249.0, True), (1250.0, False), (900.0, True)])
def test_latency_must_rise_by_less_than_a_quarter_on_every_set(candidate_ms, passes):
    sets = _all()
    sets["holdout_75"]["candidate"]["latency"] = {"median_ms": candidate_ms}
    assert (_results(sets)[0] == "PASS") is passes


def test_a_set_that_was_not_run_is_never_a_pass():
    sets = _all()
    del sets["targeted_heldout"]
    final, conditions = _results(sets)
    assert final == "INCOMPLETE" and "NOT RUN" in conditions.values()


def test_case_verdict_is_the_scorecards_own():
    ok = {"violations": [], "routing_ok": True, "temporal_ok": None, "structured_ok": None, "evidence_ok": None}
    assert rep.case_ok(ok)
    assert not rep.case_ok({**ok, "violations": ["x"]}) and not rep.case_ok({**ok, "routing_ok": False})
    assert not rep.case_ok({**ok, "error": "TimeoutError"})


NOTE = ("Allergies:\nNo Known Allergies\n\nChief Complaint:\nchest pain and shortness of breath for two days\n\n"
        "Discharge Medications:\n1. aspirin 81 mg daily\n2. metoprolol 25 mg twice daily\n3. atorvastatin 40 mg nightly\n")


def test_offset_rule_labels_short_sections_and_agrees_with_the_word_rule():
    import retrieval_eval as bench
    spans = {name: (start, end) for name, start, end in bench.section_spans(NOTE)}
    chunks = [(i, *spans[name]) for i, name in enumerate(("allergies", "chief_complaint", "dc_meds"), 1)]
    by_offset = s3.offset_sections(NOTE, chunks)
    assert by_offset == {1: ["allergies"], 2: ["chief_complaint"], 3: ["dc_meds"]}     # the 3-word section is labelled
    by_words = bench.chunk_sections(NOTE, [(i, NOTE[a:b]) for i, a, b in chunks])
    assert all(by_offset[c] == by_words[c] for c in by_words) and 3 in by_words
    assert s3.offset_sections(NOTE, [(9, spans["allergies"][0], spans["dc_meds"][1])])[9][0] == "dc_meds"   # most overlap first


def test_question_gold_grades_as_the_benchmark_does():
    labelled = {1: (["dc_meds"], 10, "2180-01-01"), 2: (["dc_meds"], 11, "2181-01-01"), 3: (["adm_meds"], 11, "2181-01-01"),
                4: (["course"], 11, "2181-01-01")}
    assert s3.question_gold({"template": "medications"}, labelled) == {1: 2, 2: 2, 3: 1}
    assert s3.question_gold({"template": "latest_medications"}, labelled) == {1: 1, 2: 2}
    assert s3.question_gold({"template": "earliest_medications"}, labelled) == {1: 2, 2: 1}


RUN_GLOBALS = ("REPORTS", "RUN_ID", "OUT", "FREEZE", "COMPARISON_JSON", "COMPARISON_MD", "HELDOUT_QUESTIONS", "HELDOUT_SUBJECTS")


def _sandbox(monkeypatch, tmp_path, run_id=None, questions=None, subjects=None):
    """The harness pointed at an empty reports folder; every path it holds is restored afterwards."""
    for name in RUN_GLOBALS:
        monkeypatch.setattr(s3, name, getattr(s3, name))
    s3.REPORTS = tmp_path
    s3.use_run(run_id or s3.RUN1, questions, subjects)
    return tmp_path


def test_systems_come_from_the_manifests_and_a_wrong_mapping_is_refused(tmp_path, monkeypatch):
    import json
    _sandbox(monkeypatch, tmp_path)
    def write(name, database, profile, build, **extra):
        (tmp_path / name).write_text(json.dumps({"database": database, "data_profile": profile, "v2_build": {"build_id": build}, **extra}))
    copy = {"run_database": {"database": "lumen_holdout_control_eval", "identical_to_frozen_holdout_at_creation": True}}
    write("eval_manifest_research_candidate.json", "lumen", "v2", "b-1")
    write("eval_manifest_holdout_candidate.json", "lumen_holdout_v2_eval", "v2", s3.HOLDOUT_BUILD)
    write("eval_manifest_holdout_control.json", "lumen_holdout", "control", None, **copy)
    monkeypatch.setattr(s3, "REPORTS", tmp_path)
    got = s3.systems()
    assert got["research_candidate"] == {"database": "lumen", "profile": "v2", "build": "b-1"}
    assert got["holdout_control"] == {"database": "lumen_holdout_control_eval", "profile": "control", "build": None}   # never the frozen database
    write("eval_manifest_holdout_candidate.json", "lumen_holdout", "v2", s3.HOLDOUT_BUILD)   # the candidate must never be the frozen database
    with pytest.raises(SystemExit):
        s3.systems()
    write("eval_manifest_holdout_candidate.json", "lumen_holdout_v2_eval", "v2", s3.HOLDOUT_BUILD)
    write("eval_manifest_holdout_control.json", "lumen_holdout", "control", None)             # no verified copy recorded
    with pytest.raises(SystemExit):
        s3.systems()


def test_legacy_capped_lab_score_is_separate_and_never_in_the_gate():
    base = {"violations": [], "routing_ok": True, "temporal_ok": True, "structured_ok": True, "evidence_ok": None}
    right_by_full = {**base, "legacy_capped_lab_ok": False}                 # true value; the capped table disagreed
    right_by_capped = {**base, "violations": [rep.LAB_TRUTH_VIOLATION], "structured_ok": False, "temporal_ok": False,
                       "legacy_capped_lab_ok": True}                        # the control's usual case
    assert rep.case_ok(right_by_full) and not rep.legacy_case_ok(right_by_full)
    assert not rep.case_ok(right_by_capped) and rep.legacy_case_ok(right_by_capped)
    assert not rep.legacy_case_ok({**right_by_capped, "violations": [rep.LAB_TRUTH_VIOLATION, "source S1 belongs to another patient"]})
    assert rep.legacy_case_ok(base) == rep.case_ok(base)                    # non-lab cases are scored one way only
    assert not any("legacy" in c["condition"].lower() for c in rep.gate(_all())["conditions"])


def test_lab_truth_table_is_one_of_two_fixed_names():
    import structured_parity as parity
    assert parity.LAB_TRUTH_TABLES == ("labevents", "labevents_full") and s3.LAB_TRUTH_TABLE == "labevents_full"
    with pytest.raises(ValueError):
        parity.truth(None, 1, ["creatinine"], "labevents; DROP TABLE patients")


# --- runs: their own paths, their own held-out files, nothing overwritten -------------------------
import json  # noqa: E402
import pathlib  # noqa: E402

RUN2, BUILD2 = "stage3_run2", "build-run2"
PATIENTS = [101, 102, 103, 201, 202]                  # 201 and 202 are the fresh held-out patients


def _manifests(root, build=BUILD2, patients=PATIENTS):
    (root / "eval_manifest_research_candidate.json").write_text(json.dumps({
        "database": "lumen", "data_profile": "v2", "patient_ids": patients,
        "v2_build": {"build_id": build, "configuration": {"embedding": {"revision": "rev-1"}, "chunker": {"min_embed_tokens": 40}}},
        "retrieval_configuration": {"query_expansion": False, "reranker": {"config_files_sha256": "r"}, "query_encoder": {"config_files_sha256": "q"}}}))
    (root / "eval_manifest_holdout_candidate.json").write_text(json.dumps({
        "database": "lumen_holdout_v2_eval", "data_profile": "v2", "v2_build": {"build_id": s3.HOLDOUT_BUILD}}))
    (root / "eval_manifest_holdout_control.json").write_text(json.dumps({
        "database": "lumen_holdout", "data_profile": "control",
        "run_database": {"database": "lumen_holdout_control_eval", "identical_to_frozen_holdout_at_creation": True}}))


def _heldout(root, tag="v2", subjects=(201, 202), items=None):
    ids, questions = root / f"heldout_subject_ids_{tag}.txt", root / f"heldout_questions_{tag}.json"
    ids.write_text("# fresh patients\n" + "\n".join(map(str, subjects)) + "\n")
    questions.write_text(json.dumps(items if items is not None else [
        {"qid": f"N{i}", "subject_id": s, "question": f"synthetic question {i}", "ground_truth": f"synthetic answer {i}", "hadm_id": 9000 + i}
        for i, s in enumerate((*subjects, subjects[0]), 1)]))
    return str(questions), str(ids)


def _run2(monkeypatch, tmp_path, retrieval=True, **heldout):
    _sandbox(monkeypatch, tmp_path)
    _manifests(tmp_path)
    questions, ids = _heldout(tmp_path, **heldout)
    s3.use_run(RUN2, questions, ids)
    if retrieval:
        s3.OUT.mkdir(parents=True)
        for name in ("research_control", "research_candidate"):
            (s3.OUT / f"retrieval_{name}.json").write_text(json.dumps({"per_question": [], "system": name}))
    return questions, ids


def test_the_first_run_keeps_the_paths_it_was_made_with(tmp_path, monkeypatch):
    root = _sandbox(monkeypatch, tmp_path)
    assert s3.run_paths("stage3") == {"out": root / "stage3", "freeze": root / "stage3_freeze.json",
                                      "comparison_json": root / "stage3_comparison.json", "comparison_md": root / "stage3_comparison.md"}
    assert (s3.RUN_ID, s3.OUT, s3.FREEZE) == ("stage3", root / "stage3", root / "stage3_freeze.json")
    assert s3.heldout_files() == (root / "heldout_questions.json", root / "heldout_subject_ids.txt")


def test_a_later_run_writes_only_to_its_own_paths(tmp_path, monkeypatch):
    root = _sandbox(monkeypatch, tmp_path)
    first, second = s3.run_paths("stage3"), s3.run_paths(RUN2)
    assert second == {"out": root / "stage3" / "run2", "freeze": root / "stage3_run2_freeze.json",
                      "comparison_json": root / "stage3_run2_comparison.json", "comparison_md": root / "stage3_run2_comparison.md"}
    assert not set(first.values()) & set(second.values())
    for bad in ("run2", "stage3_", "stage3_holdout75_holdout_control", "../stage3_run2", "stage3_run2/x", "stage3_RUN2"):
        with pytest.raises(SystemExit):
            s3.run_paths(bad)


def test_a_later_run_needs_its_own_explicit_heldout_files(tmp_path, monkeypatch):
    root = _sandbox(monkeypatch, tmp_path)
    questions, ids = _heldout(root)
    s3.use_run(RUN2)
    with pytest.raises(SystemExit, match="needs --heldout-questions"):
        s3.heldout_files()                                                    # never falls back to the first run's files
    s3.use_run(RUN2, questions, ids)
    assert tuple(map(str, s3.heldout_files())) == (str(pathlib.Path(questions).resolve()), str(pathlib.Path(ids).resolve()))
    with pytest.raises(SystemExit, match="may not reuse"):
        s3.use_run(RUN2, str(root / "heldout_questions.json"), ids)
    with pytest.raises(SystemExit, match="tied to its own"):
        s3.use_run("stage3", questions, ids)                                  # and the first run cannot be pointed elsewhere


def test_the_candidate_build_comes_from_the_refreshed_manifest_until_the_run_is_frozen(tmp_path, monkeypatch):
    _run2(monkeypatch, tmp_path)
    assert s3.systems()["research_candidate"] == {"database": "lumen", "profile": "v2", "build": BUILD2}
    s3.FREEZE.write_text(json.dumps(s3.build_freeze(BUILD2, live=False)))
    _manifests(tmp_path, build="a-later-build")                               # the manifest moves on
    assert s3.systems()["research_candidate"]["build"] == BUILD2              # the frozen run keeps its build
    s3.use_run("stage3_run3", *_heldout(tmp_path, tag="v3", subjects=(101,)))
    assert s3.systems()["research_candidate"]["build"] == "a-later-build"


def test_freeze_records_the_run_the_build_and_the_file_hashes(tmp_path, monkeypatch):
    import hashlib
    questions, ids = _run2(monkeypatch, tmp_path)
    record = s3.build_freeze(BUILD2, live=False)
    digest = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()      # noqa: E731
    assert record["run_id"] == RUN2 and record["builds"]["research_candidate"] == BUILD2
    assert record["builds"]["holdout_candidate"] == s3.HOLDOUT_BUILD and record["builds"]["research_candidate_patients"] == PATIENTS
    assert record["systems"]["research_control"] == {"database": "lumen", "profile": "control", "build": None}
    assert record["systems"]["holdout_control"]["database"] == "lumen_holdout_control_eval"
    assert record["systems"]["holdout_candidate"] == {"database": "lumen_holdout_v2_eval", "profile": "v2", "build": s3.HOLDOUT_BUILD}
    assert record["heldout"]["questions_sha256"] == digest(questions) and record["heldout"]["ids_sha256"] == digest(ids)
    assert (record["heldout"]["questions"], record["heldout"]["patients"]) == (3, 2)
    assert "synthetic" not in json.dumps(record)                              # no question or answer text in the freeze
    assert record["gate"]["regression_max_correct_lost"] == 1 and record["gate"]["targeted_min_correct_gained"] == 3
    assert record["gate"]["latency_max_median_rise_percent"] == 25
    lab = record["evaluators"]["holdout_75"]
    assert lab["primary_lab_truth"]["table"] == "labevents_full" and "never in PASS/FAIL" in lab["legacy_capped_lab_score"]
    assert "no 40-token minimum" in record["evaluators"]["retrieval_42_and_temporal_11"]["candidate_labels"]
    assert record["query_expansion"].startswith("off") and record["e10_admission_resolver"]["frozen"] is True
    assert record["structured_readiness"]["lab_table_by_profile"]["v2"] == "labevents_full" and "nothing cached" in record["structured_readiness"]["check"]
    assert len(record["code"]["commit"]) == 40 and set(record["code"]["file_sha256"]) == set(s3.FROZEN_FILES)
    assert set(record["evaluators"]["retrieval_42_and_temporal_11"]["results_used_by_the_gate"]) == {"research_control", "research_candidate"}
    with pytest.raises(SystemExit, match="preflight"):
        s3.build_freeze("some-other-build", live=False)                       # not the build that was expected


def test_freeze_needs_the_development_retrieval_results_and_is_written_once(tmp_path, monkeypatch):
    _run2(monkeypatch, tmp_path, retrieval=False)
    with pytest.raises(SystemExit, match="42-question"):
        s3.build_freeze(BUILD2, live=False)
    s3.FREEZE.write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        s3.cmd_freeze(SimpleArgs(expect_build=BUILD2))


class SimpleArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.mark.parametrize("items, problem", [
    ([{"qid": "N1", "subject_id": 201, "question": "q", "ground_truth": ""}, {"qid": "N2", "subject_id": 202, "question": "q", "ground_truth": "a"}], "without ground_truth"),
    ([{"qid": "N1", "subject_id": 201, "question": "q", "ground_truth": "a"}, {"qid": "N1", "subject_id": 202, "question": "q", "ground_truth": "a"}], "not unique"),
    ([{"qid": "N1", "subject_id": 201, "question": "q", "ground_truth": "a"}], "different patients"),                       # 202 has no question
    ([{"qid": "N1", "subject_id": 201, "question": "q", "ground_truth": "a"}, {"qid": "N2", "subject_id": 999, "question": "q", "ground_truth": "a"}], "different patients"),
    ([], "no list of items"),
])
def test_a_malformed_heldout_file_stops_the_freeze(tmp_path, monkeypatch, items, problem):
    _run2(monkeypatch, tmp_path, items=items)
    with pytest.raises(SystemExit, match=problem):
        s3.build_freeze(BUILD2, live=False)


def test_a_heldout_patient_outside_the_build_stops_the_preflight(tmp_path, monkeypatch):
    _run2(monkeypatch, tmp_path, subjects=(201, 777))
    checks = s3.preflight(BUILD2, live=False)
    assert checks["heldout_subjects_in_manifest_build"] == 1 and checks["ok"] is False


def test_heldout_material_runs_only_under_the_freeze_that_names_these_files(tmp_path, monkeypatch):
    questions, _ = _run2(monkeypatch, tmp_path)
    monkeypatch.setattr(s3, "api", lambda *a, **k: pytest.fail("nothing may be asked"))
    for command, args in ((s3.cmd_holdout75, SimpleArgs()), (s3.cmd_ask, SimpleArgs(set="heldout"))):
        with pytest.raises(SystemExit, match="no freeze"):
            command(args)
    s3.FREEZE.write_text(json.dumps(s3.build_freeze(BUILD2, live=False)))
    assert s3.require_freeze(check_heldout=True)["run_id"] == RUN2
    pathlib.Path(questions).write_text(pathlib.Path(questions).read_text() + " ")          # edited after the freeze
    with pytest.raises(SystemExit, match="not the ones the freeze recorded"):
        s3.cmd_ask(SimpleArgs(set="heldout"))


def test_no_result_is_ever_overwritten(tmp_path, monkeypatch):
    _run2(monkeypatch, tmp_path)
    freeze = json.dumps(s3.build_freeze(BUILD2, live=False))
    monkeypatch.setattr(s3, "api", lambda *a, **k: pytest.fail("nothing may be asked"))
    monkeypatch.setattr(s3.subprocess, "run", lambda *a, **k: pytest.fail("nothing may be run"))
    with pytest.raises(SystemExit, match="never overwritten"):
        s3.cmd_retrieval(SimpleArgs())                                        # its results already exist
    s3.FREEZE.write_text(freeze)
    (s3.OUT / "ask_heldout_research_control.json").write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        s3.cmd_ask(SimpleArgs(set="heldout"))
    (s3.OUT / "holdout75_holdout_candidate").mkdir()
    (s3.OUT / "holdout75_holdout_candidate" / "scorecard-1.json").write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        s3.cmd_holdout75(SimpleArgs())                                        # refused before the control arm is run either
    (s3.OUT / "grades_heldout.json").write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        s3.cmd_sheet(SimpleArgs(set="heldout"))
    assert s3.completed_results() == ["stage3/run2/ask_heldout_research_control.json", "stage3/run2/grades_heldout.json",
                                      "stage3/run2/holdout75_holdout_candidate/scorecard-1.json"]
    assert s3.preflight(BUILD2, live=False)["ok"] is False                    # a run that already holds results is not started again
    assert not hasattr(s3, "FORCE") and "--force" not in pathlib.Path(s3.__file__).read_text()     # there is no override


def test_a_second_run_never_touches_the_first_and_can_prove_it(tmp_path, monkeypatch):
    root = _sandbox(monkeypatch, tmp_path)
    _manifests(root)
    first = {"stage3/ask_heldout_research_control.json": "run 1 answers", "stage3/holdout75_holdout_control/scorecard-1.json": "run 1 cases",
             "stage3/api_research_control.log": "run 1 log", "stage3_freeze.json": "{}", "stage3_comparison.json": "{}", "stage3_comparison.md": "# run 1"}
    for name, body in first.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(body)
    s3.use_run("stage3")
    assert s3.cmd_seal(SimpleArgs()) == 0 and set(json.loads(s3.seal_path("stage3").read_text())["files"]) == set(first)
    with pytest.raises(SystemExit, match="never overwritten"):
        s3.cmd_seal(SimpleArgs())

    s3.use_run(RUN2, *_heldout(root))
    s3.OUT.mkdir(parents=True)
    for name in ("retrieval_research_control.json", "retrieval_research_candidate.json", "ask_dev15_research_control.json"):
        (s3.OUT / name).write_text("{}")                                      # run 2 writes inside stage3/run2/ only
    s3.FREEZE.write_text(json.dumps(s3.build_freeze(BUILD2, live=False)))
    assert {str(f.relative_to(root)) for f in s3.artifact_files("stage3")} == set(first)       # run 2's folder is not run 1's
    assert all(str(f.relative_to(root)).startswith(("stage3/run2/", "stage3_run2_")) for f in s3.artifact_files(RUN2))
    assert s3.other_runs_intact() == [{"run_id": "stage3", "files": 6, "changed": [], "missing": [], "added": []}]
    assert {name: (root / name).read_text() for name in first} == first

    (root / "stage3_comparison.md").write_text("# run 1, edited")
    (root / "stage3" / "extra.json").write_text("{}")
    (intact,) = s3.other_runs_intact()
    assert intact["changed"] == ["stage3_comparison.md"] and intact["added"] == ["stage3/extra.json"]
    s3.FREEZE.unlink()
    assert s3.preflight(BUILD2, live=False)["ok"] is False                    # a changed first run stops the second


def test_an_unsealed_first_run_stops_a_later_one(tmp_path, monkeypatch):
    _run2(monkeypatch, tmp_path)
    assert s3.preflight(BUILD2, live=False)["ok"] is True                     # no earlier run at all
    (tmp_path / "stage3_comparison.json").write_text("{}")
    checks = s3.preflight(BUILD2, live=False)
    assert checks["unsealed_earlier_run"] is True and checks["ok"] is False


def test_the_report_is_written_once_and_only_when_complete(tmp_path, monkeypatch, capsys):
    _run2(monkeypatch, tmp_path)
    monkeypatch.setattr(s3, "holdout_state", lambda: {"source_tables_identical": True, "tables_changed": []})
    monkeypatch.setattr(rep, "OUT", rep.OUT)
    for name in ("retrieval_sets", "holdout_set"):
        monkeypatch.setattr(rep, name, lambda: None)
    monkeypatch.setattr(rep, "targeted_set", lambda name: None)
    assert rep.main() == 0 and "not written" in capsys.readouterr().out
    assert not s3.COMPARISON_JSON.exists() and not s3.COMPARISON_MD.exists()   # an incomplete run leaves no report behind

    full = _all()
    for side in ("control", "candidate"):
        for body in full.values():
            body[side].update({"correct_definition": "synthetic", "latency": {"median_ms": 1000.0, "p95_ms": 2000.0}})
    monkeypatch.setattr(rep, "retrieval_sets", lambda: {k: full[k] for k in ("retrieval_42", "temporal_11")})
    monkeypatch.setattr(rep, "holdout_set", lambda: full["holdout_75"])
    monkeypatch.setattr(rep, "targeted_set", lambda name: full["targeted_heldout"] if name == "heldout" else None)
    assert rep.main() == 0 and rep.OUT == s3.OUT
    written = json.loads(s3.COMPARISON_JSON.read_text())
    assert written["run_id"] == RUN2 and written["gate"]["final"] == "PASS" and RUN2 in s3.COMPARISON_MD.read_text()
    assert s3.COMPARISON_JSON.name == "stage3_run2_comparison.json" and not (tmp_path / "stage3_comparison.json").exists()
    with pytest.raises(SystemExit, match="never overwritten"):
        rep.main()
