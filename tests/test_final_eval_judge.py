"""Independent judge: prompt independence, parsing, cache keys, failure states.

No network. The backend is an injected callable, so every path is exercised
deterministically.
"""

import json

import pytest

from src.evals.final_eval import cases as case_mod
from src.evals.final_eval import judge as J
from src.evals.final_eval import judge_backend as JB


@pytest.fixture(scope="module")
def gold():
    return {c.query_id: c for c in case_mod.load_cases()}


def response_row(qid="demo_q01", answer="The creatinine was 1.4 mg/dL [S1]."):
    """A row carrying plenty of runtime-verifier state, so the independence
    tests have something real to fail on."""
    return {
        "query_id": qid, "subject_id": 90000001, "answer": answer, "status": "completed",
        "citations": [{"label": "S1", "labels": ["S1"], "claim": answer, "verified": True,
                       "verification_note": "deterministic: 1 anchor matched verbatim in S1"}],
        "sources": [{"label": "S1", "chunk_id": 101, "subject_id": 90000001}],
        "needs_human_review": False, "review_status": "auto_approved",
        "escalation_reason": "none",
        "verification_trace": [{"i": 0, "stage": "deterministic", "verdict": "supported",
                                "final": "supported"}],
    }


EVIDENCE = [{"label": "S1", "text": "Discharge labs: creatinine 1.4 mg/dL on 2024-03-18."},
            {"label": "S2", "text": "Unrelated chunk about physiotherapy."}]


def verdict_json(score=4, **over):
    d = {k: {"score": score, "applicable": True, "reason": "ok", "criterion": None,
             "confidence": "high"} for k in J.DIMENSIONS}
    d.update(over)
    return json.dumps(d)


# --- independence ----------------------------------------------------------
def test_prompt_contains_no_runtime_verifier_state(gold):
    system, user, _ = J.build_prompt(gold["demo_q01"], response_row(), EVIDENCE)
    assert J.forbidden_keys_present(system) == []
    assert J.forbidden_keys_present(J.prompt_without_evidence(user)) == []
    for leak in ("auto_approved", "needs_human_review", "verification_note",
                 "deterministic: 1 anchor", "supported"):
        assert leak not in user, leak


def test_subject_id_is_not_supplied_as_a_field(gold):
    _, user, _ = J.build_prompt(gold["demo_q01"], response_row(), EVIDENCE)
    assert "90000001" not in J.prompt_without_evidence(user)


def test_judge_model_may_not_be_a_runtime_tier():
    runtime = {"main": "qwen3:30b-a3b-instruct-2507-q4_K_M", "fast": "qwen3:4b-instruct-2507-q4_K_M"}
    for tag in (runtime["main"], runtime["fast"], "qwen3:4b-instruct-2507-q4_K_M:latest"):
        with pytest.raises(JB.JudgeNotIndependent):
            JB.assert_independent(tag, runtime)
    JB.assert_independent("qwen2.5:14b", runtime)          # a third model is fine


def test_backend_refuses_to_construct_with_a_runtime_model():
    runtime = {"main": "m-model", "fast": "f-model"}
    with pytest.raises(JB.JudgeNotIndependent):
        JB.CallableJudgeBackend(lambda *a: "", model="f-model", runtime=runtime)


# --- evidence selection ----------------------------------------------------
def test_only_cited_evidence_is_shown(gold):
    _, user, _ = J.build_prompt(gold["demo_q01"], response_row(), EVIDENCE)
    assert "creatinine 1.4 mg/dL on 2024-03-18" in user
    assert "physiotherapy" not in user


def test_uncited_answer_falls_back_to_all_retrieved_evidence(gold):
    row = response_row(answer="Renal function is stable.")
    row["citations"] = []
    ev = J.cited_evidence(row, EVIDENCE)
    assert {e["label"] for e in ev} == {"S1", "S2"}


# --- applicability ---------------------------------------------------------
BASE_DIMS = {"factual_correctness", "groundedness", "completeness", "answer_relevance"}


@pytest.mark.parametrize("qid, extra", [
    ("demo_q01", {"temporal_correctness"}),                  # latest
    ("demo_q02", {"temporal_correctness"}),                  # trend
    ("demo_q19", set()),                                     # temporal 'all', answerable
    ("demo_q21", {"temporal_correctness", "abstention_quality"}),   # unsupported, latest
    ("demo_q28", {"temporal_correctness", "abstention_quality"}),   # ambiguous, latest
    ("demo_q35", {"abstention_quality"}),                    # ambiguous, temporal 'all'
])
def test_applicable_dimensions_come_from_gold(gold, qid, extra):
    """Applicability is decided by the evaluator from the gold case, never by
    the judge, so the applicable counts are identical across runs."""
    assert set(J.applicable_dimensions(gold[qid])) == BASE_DIMS | extra


# --- parsing ---------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    verdict_json(3),
    "```json\n" + verdict_json(3) + "\n```",
    "Here is my assessment:\n" + verdict_json(3),
    verdict_json(3) + "\n\nLet me know if you need more.",
])
def test_robust_json_extraction(raw):
    v, err = J.parse_verdict(raw)
    assert err is None and v.groundedness.score == 3


@pytest.mark.parametrize("raw", ["", "no json at all", "{broken", "[1,2,3]"])
def test_unparseable_response_is_an_error_not_a_score(raw):
    v, err = J.parse_verdict(raw)
    assert v is None and err


def test_out_of_range_and_non_integer_scores_are_discarded():
    raw = verdict_json(4, groundedness={"score": 9, "applicable": True, "reason": "x"},
                       completeness={"score": "not a number", "applicable": True, "reason": "x"})
    v, err = J.parse_verdict(raw)
    assert v.groundedness.score is None
    assert v.completeness.score is None
    assert v.factual_correctness.score == 4


def test_string_digit_score_is_coerced():
    raw = verdict_json(4, groundedness={"score": "2", "applicable": True, "reason": "x"})
    v, _ = J.parse_verdict(raw)
    assert v.groundedness.score == 2


# --- failure handling ------------------------------------------------------
def test_backend_failure_is_recorded_never_scored(gold, tmp_path):
    def boom(*_a, **_k):
        raise JB.JudgeUnavailable("connection refused")

    backend = JB.CallableJudgeBackend(boom, model="judge-x", check_independence=False)
    cache = J.JudgeCache(str(tmp_path / "c.json"))
    out = J.judge_case(backend, cache, gold["demo_q01"], response_row(), EVIDENCE)
    assert out["status"] == "backend_error"
    assert all(out["dimensions"][d]["score"] is None for d in J.DIMENSIONS)
    assert cache.get(out["cache_key"]) is None          # a failure is never cached


def test_parse_failure_is_recorded_never_scored(gold, tmp_path):
    backend = JB.CallableJudgeBackend(lambda *a, **k: "garbage", model="judge-x",
                                      check_independence=False)
    out = J.judge_case(backend, J.JudgeCache(str(tmp_path / "c.json")),
                       gold["demo_q01"], response_row(), EVIDENCE)
    assert out["status"] == "parse_error"
    assert all(out["dimensions"][d]["score"] is None for d in J.DIMENSIONS)


def test_scores_for_inapplicable_dimensions_are_discarded(gold, tmp_path):
    """demo_q01 does not exercise abstention; a score for it must not survive
    into the aggregate."""
    backend = JB.CallableJudgeBackend(lambda *a, **k: verdict_json(4), model="judge-x",
                                      check_independence=False)
    out = J.judge_case(backend, J.JudgeCache(str(tmp_path / "c.json")),
                       gold["demo_q01"], response_row(), EVIDENCE)
    assert out["status"] == "ok"
    assert out["dimensions"]["abstention_quality"]["score"] is None
    assert out["dimensions"]["abstention_quality"]["applicable"] is False
    assert out["dimensions"]["temporal_correctness"]["score"] == 4


# --- cache -----------------------------------------------------------------
def _key(**over):
    base = dict(prompt_version="aj1", model="judge-x", query_id="demo_q01",
                query="q", answer="a", evidence=[{"label": "S1", "text": "t"}],
                criteria={"expected_facts": ["f"]})
    base.update(over)
    return J.JudgeCache.key(**base)


def test_cache_key_is_stable_for_identical_input():
    assert _key() == _key()


@pytest.mark.parametrize("field, value", [
    ("prompt_version", "aj2"),
    ("model", "other-judge"),
    ("query", "different question"),
    ("answer", "different answer"),
    ("evidence", [{"label": "S1", "text": "different evidence"}]),
    ("criteria", {"expected_facts": ["different gold"]}),
])
def test_cache_key_changes_when_anything_that_matters_changes(field, value):
    assert _key(**{field: value}) != _key()


def test_cache_round_trip_and_hit(gold, tmp_path):
    calls = []

    def fn(*a, **k):
        calls.append(1)
        return verdict_json(3)

    backend = JB.CallableJudgeBackend(fn, model="judge-x", check_independence=False)
    cache = J.JudgeCache(str(tmp_path / "c.json"))
    first = J.judge_case(backend, cache, gold["demo_q01"], response_row(), EVIDENCE)
    second = J.judge_case(backend, cache, gold["demo_q01"], response_row(), EVIDENCE)
    assert len(calls) == 1
    assert first["cached"] is False and second["cached"] is True
    assert second["dimensions"]["groundedness"]["score"] == 3
    # survives a fresh process
    assert J.JudgeCache(str(tmp_path / "c.json")).get(first["cache_key"]) is not None
