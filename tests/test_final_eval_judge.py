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


def verdict_json(score=4, case=None, statuses=None, unsupported=None, **over):
    """An aj2-shaped verdict that is internally consistent by default.

    `statuses` overrides the per-criterion status by criterion id, which is how
    the consistency tests build a verdict that contradicts itself.
    """
    d = {k: {"score": score, "applicable": True, "reason": "ok", "criterion": None,
             "confidence": "high"} for k in J.DIMENSIONS}
    crits = J.criteria_for(case) if case is not None else []
    statuses = statuses or {}
    d["criterion_assessments"] = [
        {"criterion_id": c.id, "criterion": c.text,
         "status": statuses.get(c.id, "supported"),
         "evidence_labels": ["S1"], "reason": "stated in the answer"}
        for c in crits]
    d["unsupported_content"] = list(unsupported or [])
    for k, v in over.items():
        # A bare int overrides just that dimension's score; a dict replaces it.
        d[k] = v if isinstance(v, dict) else {"score": v, "applicable": True,
                                              "reason": "ok", "criterion": None,
                                              "confidence": "high"}
    return json.dumps(d)


def judge_fn(payload):
    """A backend callable returning a fixed payload, recording every call."""
    calls = []

    def fn(system, user, schema=None):
        calls.append(user)
        return payload(len(calls)) if callable(payload) else payload

    return fn, calls


# --- independence ----------------------------------------------------------
def test_prompt_contains_no_runtime_verifier_state(gold):
    system, user, _, _ = J.build_prompt(gold["demo_q01"], response_row(), EVIDENCE)
    assert J.forbidden_keys_present(system) == []
    assert J.forbidden_keys_present(J.prompt_without_evidence(user)) == []
    for leak in ("auto_approved", "needs_human_review", "verification_note",
                 "deterministic: 1 anchor", "supported"):
        assert leak not in user, leak


def test_subject_id_is_not_supplied_as_a_field(gold):
    _, user, _, _ = J.build_prompt(gold["demo_q01"], response_row(), EVIDENCE)
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
    _, user, _, _ = J.build_prompt(gold["demo_q01"], response_row(), EVIDENCE)
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
    backend = JB.CallableJudgeBackend(
        lambda *a, **k: verdict_json(4, case=gold["demo_q01"]), model="judge-x",
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
        return verdict_json(3, case=gold["demo_q01"])

    backend = JB.CallableJudgeBackend(fn, model="judge-x", check_independence=False)
    cache = J.JudgeCache(str(tmp_path / "c.json"))
    first = J.judge_case(backend, cache, gold["demo_q01"], response_row(), EVIDENCE)
    second = J.judge_case(backend, cache, gold["demo_q01"], response_row(), EVIDENCE)
    assert len(calls) == 1
    assert first["cached"] is False and second["cached"] is True
    assert second["dimensions"]["groundedness"]["score"] == 3
    # survives a fresh process
    assert J.JudgeCache(str(tmp_path / "c.json")).get(first["cache_key"]) is not None


# ===========================================================================
# aj2 — per-criterion assessment and consistency validation
# ===========================================================================
def test_every_gold_expectation_becomes_a_criterion(gold):
    """Criteria come from the gold case, one per expected fact, plus a criterion
    for each applicable behavioural expectation."""
    c1 = gold["demo_q01"]                         # latest-value case
    crits = J.criteria_for(c1)
    assert [c.text for c in crits if c.kind == J.KIND_FACT] == [f.text for f in c1.expected_facts]
    assert [c.kind for c in crits].count(J.KIND_TEMPORAL) == 1
    assert len({c.id for c in crits}) == len(crits)      # ids are unique

    trend = gold["demo_q02"]
    assert any(c.kind == J.KIND_TEMPORAL for c in J.criteria_for(trend))

    unsupported = next(c for c in gold.values() if c.expects_abstention)
    assert any(c.kind == J.KIND_ABSTENTION for c in J.criteria_for(unsupported))

    ambiguous = next(c for c in gold.values() if c.expects_ambiguity)
    assert any(c.kind == J.KIND_AMBIGUITY for c in J.criteria_for(ambiguous))


def test_must_not_contain_is_not_a_criterion(gold):
    """A prohibition would invert every status word, so it stays out of the
    criterion list and is stated in the prompt instead."""
    case = next((c for c in gold.values() if c.must_not_contain), None)
    if case is None:
        pytest.skip("no gold case carries must_not_contain")
    crits = J.criteria_for(case)
    for term in case.must_not_contain:
        assert all(term not in c.text for c in crits)
    _, user, _, _ = J.build_prompt(case, response_row(case.query_id), EVIDENCE)
    assert "MUST NOT ASSERT" in user


def test_prompt_lists_every_criterion_with_its_id(gold):
    case = gold["demo_q02"]
    _, user, _, crits = J.build_prompt(case, response_row("demo_q02"), EVIDENCE)
    assert "REQUIRED CRITERIA" in user
    for c in crits:
        assert c.id in user and c.text[:40] in user


def test_criterion_assessments_are_parsed_and_counted(gold):
    case = gold["demo_q01"]
    raw = verdict_json(4, case=case)
    verdict, err = J.parse_verdict(raw)
    assert err is None
    ids = [a.criterion_id for a in verdict.criterion_assessments]
    assert ids == [c.id for c in J.criteria_for(case)]
    counts = J.criterion_status_counts([a.model_dump() for a in verdict.criterion_assessments])
    assert counts["supported"] == len(ids)


def test_assessment_with_an_unknown_status_is_dropped_not_coerced(gold):
    """An unrecognised status leaves the criterion unassessed — which the
    consistency check catches — rather than being guessed into a pass."""
    case = gold["demo_q01"]
    payload = json.loads(verdict_json(4, case=case))
    payload["criterion_assessments"][0]["status"] = "probably_fine"
    verdict, err = J.parse_verdict(json.dumps(payload))
    assert err is None
    assert len(verdict.criterion_assessments) == len(J.criteria_for(case)) - 1


# --- the consistency rules -------------------------------------------------
def _consistency(case, statuses=None, unsupported=None, **scores):
    """Violations for a verdict built from `case`, with score overrides."""
    crits = J.criteria_for(case)
    dims = {d: {"score": 4, "applicable": True} for d in J.DIMENSIONS}
    for k, v in scores.items():
        dims[k] = {"score": v, "applicable": True}
    assessments = [{"criterion_id": c.id, "criterion": c.text,
                    "status": (statuses or {}).get(c.id, "supported"),
                    "evidence_labels": [], "reason": ""} for c in crits]
    return J.check_consistency(assessments, dims, crits, list(unsupported or []),
                               J.applicable_dimensions(case))


@pytest.mark.parametrize("status", ["missing", "partially_supported", "contradicted"])
def test_completeness_four_is_impossible_with_an_unmet_criterion(gold, status):
    """The demo_q19 failure mode: a fluent answer that omits a required fact
    can no longer be scored completeness=4."""
    case = gold["demo_q19"]
    first = J.criteria_for(case)[0].id
    rules = [v["rule"] for v in _consistency(case, statuses={first: status})]
    assert "completeness_inflated" in rules


def test_completeness_four_is_fine_when_every_criterion_is_met(gold):
    assert _consistency(gold["demo_q19"]) == []


def test_not_applicable_criterion_does_not_block_completeness(gold):
    case = gold["demo_q19"]
    first = J.criteria_for(case)[0].id
    assert _consistency(case, statuses={first: "not_applicable"}) == []


def test_factual_correctness_four_is_impossible_with_a_contradicted_criterion(gold):
    case = gold["demo_q19"]
    first = J.criteria_for(case)[0].id
    rules = [v["rule"] for v in
             _consistency(case, statuses={first: "contradicted"}, completeness=2)]
    assert "factual_correctness_inflated" in rules
    assert "completeness_inflated" not in rules       # completeness was lowered honestly


def test_groundedness_four_is_impossible_with_unsupported_content(gold):
    rules = [v["rule"] for v in
             _consistency(gold["demo_q19"], unsupported=["the patient was started on X"])]
    assert "groundedness_inflated" in rules


def test_temporal_four_is_impossible_with_a_missing_temporal_criterion(gold):
    case = gold["demo_q02"]
    tid = next(c.id for c in J.criteria_for(case) if c.kind == J.KIND_TEMPORAL)
    rules = [v["rule"] for v in
             _consistency(case, statuses={tid: "missing"}, completeness=2)]
    assert "temporal_correctness_inflated" in rules


def test_abstention_four_is_impossible_with_an_unmet_abstention_criterion(gold):
    case = next(c for c in gold.values() if c.expects_abstention)
    aid = next(c.id for c in J.criteria_for(case) if c.kind == J.KIND_ABSTENTION)
    rules = [v["rule"] for v in
             _consistency(case, statuses={aid: "missing"}, completeness=2)]
    assert "abstention_quality_inflated" in rules


def test_an_unassessed_criterion_is_itself_a_violation(gold):
    """Omitting a criterion must not be cheaper than assessing it honestly."""
    case = gold["demo_q19"]
    crits = J.criteria_for(case)
    dims = {d: {"score": 3, "applicable": True} for d in J.DIMENSIONS}
    v = J.check_consistency([], dims, crits, [], J.applicable_dimensions(case))
    assert [x["rule"] for x in v] == ["criterion_not_assessed"]
    assert all(c.id in v[0]["detail"] for c in crits)


def test_consistency_ignores_dimensions_the_gold_marks_inapplicable(gold):
    """demo_q01 has no abstention expectation, so an abstention score cannot
    generate a violation for it."""
    case = gold["demo_q01"]
    crits = J.criteria_for(case)
    dims = {d: {"score": 4, "applicable": True} for d in J.DIMENSIONS}
    assessments = [{"criterion_id": c.id, "criterion": "", "status": "supported",
                    "evidence_labels": [], "reason": ""} for c in crits]
    assert J.check_consistency(assessments, dims, crits, [],
                               J.applicable_dimensions(case)) == []


# --- repair and judge_inconsistent ----------------------------------------
def test_inconsistent_verdict_is_repaired_once_and_accepted(gold, tmp_path):
    case = gold["demo_q19"]
    first = J.criteria_for(case)[0].id
    bad = verdict_json(4, case=case, statuses={first: "missing"})
    good = verdict_json(4, case=case, statuses={first: "missing"}, completeness=2)

    fn, calls = judge_fn(lambda n: bad if n == 1 else good)
    backend = JB.CallableJudgeBackend(fn, model="judge-x", check_independence=False)
    out = J.judge_case(backend, J.JudgeCache(str(tmp_path / "c.json")),
                       case, response_row("demo_q19"), EVIDENCE)
    assert len(calls) == 2
    assert out["status"] == "ok"
    assert out["repair_attempted"] is True
    assert out["dimensions"]["completeness"]["score"] == 2
    assert out["consistency_violations"] == []


def test_repair_prompt_names_the_specific_contradiction(gold, tmp_path):
    case = gold["demo_q19"]
    first = J.criteria_for(case)[0].id
    bad = verdict_json(4, case=case, statuses={first: "missing"})
    fn, calls = judge_fn(bad)
    backend = JB.CallableJudgeBackend(fn, model="judge-x", check_independence=False)
    J.judge_case(backend, J.JudgeCache(str(tmp_path / "c.json")),
                 case, response_row("demo_q19"), EVIDENCE)
    repair = calls[1]
    assert "inconsistent" in repair.lower()
    assert "completeness=4" in repair and f"{first}=missing" in repair
    # The repair instruction must not tell it which way to move the score.
    assert "raise" not in repair.lower().split("do not change")[0]


def test_persistently_inconsistent_verdict_is_a_judge_failure(gold, tmp_path):
    """Scores are preserved exactly, not clamped, not zeroed — and the case is
    not counted as passed."""
    case = gold["demo_q19"]
    first = J.criteria_for(case)[0].id
    bad = verdict_json(4, case=case, statuses={first: "missing"})
    fn, calls = judge_fn(bad)
    backend = JB.CallableJudgeBackend(fn, model="judge-x", check_independence=False)
    out = J.judge_case(backend, J.JudgeCache(str(tmp_path / "c.json")),
                       case, response_row("demo_q19"), EVIDENCE)
    assert len(calls) == 2                       # exactly one repair, never a loop
    assert out["status"] == "judge_inconsistent"
    assert out["dimensions"]["completeness"]["score"] == 4      # untouched
    assert [v["rule"] for v in out["consistency_violations"]] == ["completeness_inflated"]
    assert out["criterion_status_counts"]["missing"] == 1


def test_unparseable_repair_is_recorded_as_a_parse_error(gold, tmp_path):
    case = gold["demo_q19"]
    first = J.criteria_for(case)[0].id
    bad = verdict_json(4, case=case, statuses={first: "missing"})
    fn, _ = judge_fn(lambda n: bad if n == 1 else "not json at all")
    backend = JB.CallableJudgeBackend(fn, model="judge-x", check_independence=False)
    out = J.judge_case(backend, J.JudgeCache(str(tmp_path / "c.json")),
                       case, response_row("demo_q19"), EVIDENCE)
    assert out["status"] == "parse_error"
    assert "repair" in out["error"]
    assert all(d["score"] is None for d in out["dimensions"].values())


def test_backend_failure_during_repair_never_becomes_a_score(gold, tmp_path):
    case = gold["demo_q19"]
    first = J.criteria_for(case)[0].id
    bad = verdict_json(4, case=case, statuses={first: "missing"})

    def fn(system, user, schema=None):
        if "inconsistent" in user.lower():
            raise JB.JudgeUnavailable("connection reset")
        return bad

    backend = JB.CallableJudgeBackend(fn, model="judge-x", check_independence=False)
    out = J.judge_case(backend, J.JudgeCache(str(tmp_path / "c.json")),
                       case, response_row("demo_q19"), EVIDENCE)
    assert out["status"] == "backend_error"
    assert all(d["score"] is None for d in out["dimensions"].values())


def test_a_stray_evidence_label_is_reported_not_treated_as_inconsistency(gold, tmp_path):
    case = gold["demo_q01"]
    payload = json.loads(verdict_json(4, case=case))
    for a in payload["criterion_assessments"]:
        a["evidence_labels"] = ["S9"]
    fn, calls = judge_fn(json.dumps(payload))
    backend = JB.CallableJudgeBackend(fn, model="judge-x", check_independence=False)
    out = J.judge_case(backend, J.JudgeCache(str(tmp_path / "c.json")),
                       case, response_row(), EVIDENCE)
    assert len(calls) == 1                       # no repair consumed
    assert out["status"] == "ok"
    assert out["unknown_evidence_labels"] == ["S9"]


# --- cache -----------------------------------------------------------------
def test_cache_key_changes_with_the_judge_prompt_version():
    """A cached aj1 verdict must never be served for an aj2 request."""
    common = dict(model="m", query_id="q", query="Q", answer="A",
                  evidence=[{"label": "S1", "text": "t"}], criteria={"a": 1})
    assert J.JudgeCache.key(prompt_version="aj1", **common) != \
        J.JudgeCache.key(prompt_version="aj2", **common)


def test_cache_key_changes_with_the_evidence_text():
    common = dict(prompt_version=J.JUDGE_PROMPT_VERSION, model="m", query_id="q",
                  query="Q", answer="A", criteria={})
    a = J.JudgeCache.key(evidence=[{"label": "S1", "text": "creatinine 1.4"}], **common)
    b = J.JudgeCache.key(evidence=[{"label": "S1", "text": "creatinine 2.1"}], **common)
    assert a != b


def test_cache_key_changes_when_the_gold_criteria_change(gold):
    """criteria_of carries the aj2 criterion list, so editing the gold facts
    invalidates the cached verdict instead of serving a stale one."""
    case = gold["demo_q01"]
    before = J.criteria_of(case)
    assert [c["id"] for c in before["criteria"]] == [c.id for c in J.criteria_for(case)]
    after = {**before, "criteria": before["criteria"][:-1]}
    common = dict(prompt_version=J.JUDGE_PROMPT_VERSION, model="m", query_id="q",
                  query="Q", answer="A", evidence=[])
    assert J.JudgeCache.key(criteria=before, **common) != \
        J.JudgeCache.key(criteria=after, **common)


def test_an_inconsistent_verdict_is_cached_so_it_is_not_re_paid_for(gold, tmp_path):
    case = gold["demo_q19"]
    first = J.criteria_for(case)[0].id
    fn, calls = judge_fn(verdict_json(4, case=case, statuses={first: "missing"}))
    backend = JB.CallableJudgeBackend(fn, model="judge-x", check_independence=False)
    cache = J.JudgeCache(str(tmp_path / "c.json"))
    a = J.judge_case(backend, cache, case, response_row("demo_q19"), EVIDENCE)
    b = J.judge_case(backend, cache, case, response_row("demo_q19"), EVIDENCE)
    assert a["status"] == b["status"] == "judge_inconsistent"
    assert len(calls) == 2                       # the second case was served from cache
    assert b["cached"] is True


def test_the_prompt_orders_criterion_assessment_before_scoring(gold):
    """aj2's whole premise: the model must rule on each criterion before it is
    allowed to produce a holistic score."""
    system, user, _, _ = J.build_prompt(gold["demo_q19"], response_row("demo_q19"), EVIDENCE)
    assert system.index("STEP 1") < system.index("STEP 3")
    assert "Do not skip step 1" in system
    assert system.index("criterion_assessments") < system.index("ONLY NOW ASSIGN")
    # The consistency rules are stated to the model, not only enforced after.
    assert "completeness = 4 is only valid when NO criterion is missing" in system
    assert "will be rejected and sent back to you" in system
    assert user.rstrip().endswith("Assess every criterion first, then return the JSON object.")


def test_the_schema_requires_criterion_assessments(gold):
    req = J.OLLAMA_SCHEMA["required"]
    assert "criterion_assessments" in req and "unsupported_content" in req
    statuses = J.OLLAMA_SCHEMA["properties"]["criterion_assessments"]["items"][
        "properties"]["status"]["enum"]
    assert set(statuses) == set(J.CRITERION_STATUSES)
