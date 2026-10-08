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


def test_systems_come_from_the_manifests_and_a_wrong_mapping_is_refused(tmp_path, monkeypatch):
    import json
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
