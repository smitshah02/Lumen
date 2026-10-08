"""Admission-scoped lab questions (Stage 3 routing diagnostic, lab part).

A lab question about ONE resolved admission is answered from the lab table for
that admission: the question is read without its admission phrase, the resolved
hadm_id constrains the SQL, and the admissions path does not claim it for
containing "admission" next to "latest", "last", "first" or "how many".
Whole-patient lab questions, and every profile without admission scope, behave
exactly as before. No database, no model: synthetic patient, stub resolver.
"""
from datetime import datetime

import pytest

import src.agents.graph as g
from src.agents.admission_scope import resolve_admission
from src.agents.classify import classify, scoped_lab_mode
from src.retrieval.hybrid_retriever_v2 import detect_temporal_mode
from src.storage.readiness import STRUCTURED, DataSourceNotReady
from tests.test_routing import QUALIFIED_ITEMS, _lab, _resolver, graph_mod, spy  # noqa: F401

SID, FIRST, SECOND = 7, 91000001, 91000002
ADMISSIONS = [(FIRST, datetime(2190, 3, 1, 8, 0), datetime(2190, 3, 6, 15, 0)),
              (SECOND, datetime(2191, 9, 10, 9, 0), datetime(2191, 9, 14, 12, 0))]
# Creatinine, deliberately not monotonic. The patient's overall latest, lowest and highest
# are all in the SECOND admission, so an answer for FIRST that used the whole patient is wrong.
BY_ADMISSION = {
    FIRST: [("2190-03-01", 1.4, None), ("2190-03-03", 2.2, None), ("2190-03-04", 1.1, None), ("2190-03-06", 1.3, None)],
    SECOND: [("2191-09-10", 0.7, None), ("2191-09-12", 3.9, None), ("2191-09-14", 0.8, None)],
}
WHOLE = BY_ADMISSION[FIRST] + BY_ADMISSION[SECOND]


def _state(question, request_hadm_id=None, scope=True):
    """What triage leaves in the state, built with the real classifier and resolver."""
    temporal = detect_temporal_mode(question)
    d = classify(question, temporal)
    state = {"query": question, "subject_id": SID, "query_type": d.query_type, "query_complexity": d.complexity,
             "temporal_mode": temporal, "classified_by": "rules" if d.confident else "fast_model"}
    if scope:
        state["admission_scope"] = resolve_admission(question, ADMISSIONS, request_hadm_id).as_state()
    return state


@pytest.fixture
def labs(monkeypatch):
    """A resolver whose fetch behaves like the SQL: one admission's rows when hadm_id is given."""
    calls = []
    resolver = _resolver(*QUALIFIED_ITEMS)

    def fetch(sid, itemids, **kw):
        calls.append({"sid": sid, **kw})
        points = BY_ADMISSION.get(kw["hadm_id"], []) if "hadm_id" in kw else WHOLE
        return [_lab("Creatinine", *points)] if points else []
    resolver.fetch = fetch
    monkeypatch.setattr(g, "get_lab_resolver", lambda: resolver)
    return calls


def _answer(question, labs, request_hadm_id=None):
    state = _state(question, request_hadm_id)
    assert g.route_from_triage(state) == "lab_lookup"
    out = g.lab_lookup(state)
    assert g.route_after_lab_lookup(out) == "finalize" and out["review_status"] == "auto_approved"
    assert out["admission_scope_applied"] is True and all(c["verified"] for c in out["citations"])
    return out["final_answer"]


# --- the answer is that admission's, by each kind of question ---------------------------------------

def test_latest_lab_in_an_admission_named_by_date(labs, spy):
    answer = _answer("What was the latest creatinine during the admission that ended on 2190-03-06?", labs)
    assert answer == f"The most recent creatinine during admission {FIRST} was 1.3 mg/dL on 2190-03-06 [L1]."
    assert labs == [{"sid": SID, "per_lab_cap": g.LAB_SERIES_CAP, "hadm_id": FIRST}] and spy == []     # zero model calls


def test_latest_lab_with_an_explicit_hadm_id_in_the_question_or_on_the_request(labs, spy):
    answer = _answer(f"What was the most recent creatinine value in admission {FIRST}?", labs)
    assert "was 1.3 mg/dL on 2190-03-06" in answer and labs[-1]["hadm_id"] == FIRST
    answer = _answer("What was the most recent creatinine?", labs, request_hadm_id=FIRST)       # stated on the request
    assert "was 1.3 mg/dL on 2190-03-06" in answer and labs[-1]["hadm_id"] == FIRST


def test_last_measured_wording(labs, spy):
    answer = _answer("During the hospital stay ending 2190-03-06, what was the last creatinine measured?", labs)
    assert answer == f"The most recent creatinine during admission {FIRST} was 1.3 mg/dL on 2190-03-06 [L1]."


@pytest.mark.parametrize("question", [
    "What was the lowest creatinine during the admission that ended on 2190-03-06?",
    f"What was the minimum creatinine recorded in admission {FIRST}?",
])
def test_lowest_value_in_an_admission(labs, spy, question):
    assert _answer(question, labs) == f"The lowest creatinine during admission {FIRST} was 1.1 mg/dL on 2190-03-04 [L1]."


@pytest.mark.parametrize("question", [
    "What was the highest creatinine during the admission that ended on 2190-03-06?",
    f"During admission {FIRST}, what was the maximum creatinine?",
])
def test_highest_value_in_an_admission(labs, spy, question):
    assert _answer(question, labs) == f"The highest creatinine during admission {FIRST} was 2.2 mg/dL on 2190-03-03 [L1]."


def test_a_repeated_extreme_says_so(monkeypatch, labs, spy):
    monkeypatch.setitem(BY_ADMISSION, FIRST, [("2190-03-01", 1.1, None), ("2190-03-03", 2.2, None), ("2190-03-04", 1.1, None)])
    answer = _answer(f"What was the lowest creatinine in admission {FIRST}?", labs)
    assert "was 1.1 mg/dL on 2190-03-01, the first of 2 measurements at that value" in answer


@pytest.mark.parametrize("question", [
    "How did creatinine change during the admission that ended on 2190-03-06?",
    f"What was the creatinine trend in admission {FIRST}?",
    "How did the creatinine change from the first to the last measurement during the admission that ended on 2190-03-06?",
])
def test_trend_and_first_to_last_wording_in_an_admission(labs, spy, question):
    answer = _answer(question, labs)
    assert f"measured 4 times during admission {FIRST} between 2190-03-01 and 2190-03-06" in answer
    assert "first value was 1.4 mg/dL on 2190-03-01 and the most recent was 1.3 mg/dL on 2190-03-06" in answer
    assert "lowest value was 1.1 mg/dL on 2190-03-04 and the highest was 2.2 mg/dL on 2190-03-03" in answer
    assert "fluctuated" in answer and "3.9" not in answer and "0.7" not in answer


def test_no_value_from_another_admission_reaches_the_answer_or_the_evidence(labs, spy):
    state = _state(f"What was the highest creatinine in admission {FIRST}?")
    out = g.lab_lookup(state)
    text = out["final_answer"] + out["lab_evidence"][0]["text"]
    assert "3.9" not in text and "2191" not in text and f"admission {FIRST}" in out["lab_evidence"][0]["text"]
    other = g.lab_lookup(_state(f"What was the highest creatinine in admission {SECOND}?"))
    assert f"during admission {SECOND} was 3.9 mg/dL on 2191-09-12" in other["final_answer"]


# --- the admissions path no longer takes these ------------------------------------------------------

@pytest.mark.parametrize("question", [
    "What was the latest creatinine during the admission that ended on 2190-03-06?",
    f"What was the most recent creatinine value in admission {FIRST}?",
    "How did the creatinine change from the first to the last measurement during the admission that ended on 2190-03-06?",
    f"How many times did the creatinine change in admission {FIRST}?",
])
def test_encounter_path_does_not_claim_a_scoped_lab_question(labs, question):
    state = _state(question)
    assert g.wants_encounter_lookup(g._decision_of(state), question, SID)        # its word list still matches, as before
    assert g.route_from_triage(state) == "lab_lookup"                            # the lab intent goes first
    assert g.route_from_triage(_state(question, scope=False)) == "encounter_lookup"   # without a resolved admission: unchanged


@pytest.mark.parametrize("question", [
    "How many hospital admissions does the patient have?",
    "When was the most recent admission?",
])
def test_real_admission_questions_still_go_to_the_admissions_path(labs, question):
    assert g.route_from_triage(_state(question)) == "encounter_lookup"
    assert g.route_from_triage(_state(question, request_hadm_id=FIRST)) == "encounter_lookup"   # even with a stated admission


# --- the SQL is constrained, not filtered afterwards ------------------------------------------------

def test_the_resolved_hadm_id_reaches_the_sql_as_a_where_clause(monkeypatch):
    import src.generation.lab_query as lab_query
    seen = []

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def execute(self, stmt, params):
            seen.append((" ".join(str(stmt).split()), dict(params)))
            return self

        def fetchall(self): return []
    monkeypatch.setattr(lab_query, "engine", type("E", (), {"connect": lambda self: _Conn()})())
    monkeypatch.setattr(lab_query.readiness, "require", lambda component: None)
    resolver = _resolver(*QUALIFIED_ITEMS)
    monkeypatch.setattr(g, "get_lab_resolver", lambda: resolver)
    out = g.lab_lookup(_state(f"What was the lowest creatinine in admission {SECOND}?"))
    sql, params = seen[-1]
    assert "WHERE l.subject_id = :sid" in sql and "AND l.hadm_id = :hadm" in sql
    assert params["sid"] == SID and params["hadm"] == SECOND
    assert "lab_evidence" not in out and g.route_after_lab_lookup(out) == "patient_retrieval"    # no rows there: a miss


# --- what must not change ---------------------------------------------------------------------------

def test_whole_patient_lab_questions_are_answered_as_before(labs, spy):
    for scope in (False, True):          # no admission scope at all (control), and scope enabled but none named
        state = _state("What was the most recent creatinine?", scope=scope)
        assert g._scoped_lab_request(state) is None and g.route_from_triage(state) == "lab_lookup"
        out = g.lab_lookup(state)
        assert out["final_answer"] == "The most recent creatinine was 0.8 mg/dL on 2191-09-14 [L1]."
        assert "admission_scope_applied" not in out and labs[-1] == {"sid": SID, "per_lab_cap": g.LAB_SERIES_CAP}
        assert out["lab_evidence"][0]["text"].endswith(f"Source: labevents table, subject {SID}.")


def test_whole_patient_questions_get_no_new_modes(labs, spy):
    for question in ("What was the lowest creatinine?", "What was the highest creatinine recorded?"):
        for scope in (False, True):
            state = _state(question, scope=scope)
            assert g.route_from_triage(state) == "patient_retrieval" and labs == []


def test_control_profile_never_scopes_a_lab_question(labs, spy):
    """Control records no admission scope, so even a question that names an admission is handled as it always was."""
    question = "What was the latest creatinine during the admission that ended on 2190-03-06?"
    state = _state(question, scope=False)
    assert g._scoped_lab_request(state) is None
    assert g.route_from_triage(state) == "encounter_lookup"                      # the route it took before this change


@pytest.mark.parametrize("question", [
    "What was the creatinine during the admission that ended on 2190-03-06?",                 # asks for no particular value
    "What was the lowest and the highest creatinine during the admission that ended on 2190-03-06?",   # two answers
    "What was the latest urine creatinine during the admission that ended on 2190-03-06?",    # a specimen not fetched
    "What was the latest creatinine and BUN during the admission that ended on 2190-03-06?",  # two analytes
    "Why was the creatinine highest during the admission that ended on 2190-03-06?",          # a reason, not a value
])
def test_a_scoped_lab_question_it_does_not_fully_understand_is_not_answered(labs, spy, question):
    state = _state(question)
    if g.route_from_triage(state) == "lab_lookup":
        out = g.lab_lookup(state)
        assert "lab_evidence" not in out and g.route_after_lab_lookup(out) == "patient_retrieval"


def test_a_non_numeric_result_or_mixed_units_makes_an_extreme_unsure(monkeypatch, spy):
    resolver = _resolver(*QUALIFIED_ITEMS)
    monkeypatch.setattr(g, "get_lab_resolver", lambda: resolver)
    state = _state(f"What was the lowest creatinine in admission {FIRST}?")
    resolver.fetch = lambda sid, ids, **kw: [{**_lab("Creatinine", *BY_ADMISSION[FIRST]), "n_non_numeric": 1}]
    assert "lab_evidence" not in g.lab_lookup(state)
    resolver.fetch = lambda sid, ids, **kw: [_lab("Creatinine", ("2190-03-01", 1.4, "mg/dL"), ("2190-03-03", 120.0, "umol/L"))]
    assert "lab_evidence" not in g.lab_lookup(state)


def test_an_unresolved_or_ambiguous_admission_is_never_scoped(labs, spy):
    state = _state("What was the latest creatinine during the admission that ended on 2199-01-01?")   # no such admission
    assert state["admission_scope"]["status"] != "resolved" and g._scoped_lab_request(state) is None
    if g.route_from_triage(state) == "lab_lookup":
        assert "lab_evidence" not in g.lab_lookup(state)


def test_readiness_errors_still_propagate_from_a_scoped_lookup(monkeypatch):
    class _Resolver:
        labels = []

        def match(self, query):
            return [50912], ["creatinine"]

        def fetch(self, *a, **k):
            raise DataSourceNotReady(STRUCTURED, "the latest structured load is running")
    monkeypatch.setattr(g, "get_lab_resolver", lambda: _Resolver())
    state = _state(f"What was the latest creatinine in admission {FIRST}?")
    assert g._scoped_lab_request(state) is not None
    with pytest.raises(DataSourceNotReady):
        g.lab_lookup(state)


@pytest.mark.parametrize("question, mode", [
    ("What was the latest creatinine?", "latest"), ("what was the last sodium measured?", "latest"),
    ("What was the first potassium?", "earliest"), ("What was the lowest potassium?", "min"),
    ("what was the minimum hemoglobin?", "min"), ("What was the highest creatinine recorded?", "max"),
    ("How did creatinine change?", "trend"), ("What was the creatinine trend?", "trend"),
    ("How did the sodium change from the first to the last measurement?", "trend"), ("What was the creatinine?", None),
])
def test_scoped_lab_modes(question, mode):
    assert scoped_lab_mode(question, detect_temporal_mode(question)) == mode
