"""One named drug's inpatient order (Stage 3 routing diagnostic, medication part).

With a resolved admission, a question about ONE drug's dose, route, doses per
24 hours or whole inpatient order is answered from that admission's rows in the
prescriptions table. The drug must match exactly one drug there; a similar name
is never offered instead. Discharge, home and current medication questions, the
whole-list path and every profile without SQL paths are unchanged.
No database, no model: synthetic patient and orders.
"""
from datetime import datetime

import pytest

import src.agents.graph as graph
from src.agents.admission_scope import resolve_admission
from src.agents.classify import classify
from src.generation import structured_lookup as sl
from src.retrieval.hybrid_retriever_v2 import detect_temporal_mode
from src.storage.readiness import STRUCTURED, DataSourceNotReady

SID, FIRST, SECOND = 7, 91000001, 91000002
ADMISSIONS = [(FIRST, datetime(2190, 3, 1, 8, 0), datetime(2190, 3, 6, 15, 0)),
              (SECOND, datetime(2191, 9, 10, 9, 0), datetime(2191, 9, 14, 12, 0))]
NAMED = "during the admission that ended on 2190-03-06"


def _rx(drug, dose, unit, route, rate, start="2190-03-01 10:00:00", stop="2190-03-06 09:00:00"):
    return {"drug": drug, "dose_val_rx": dose, "dose_unit_rx": unit, "route": route, "doses_per_24_hrs": rate,
            "starttime": start, "stoptime": stop}


ORDERS = {
    FIRST: [
        _rx("Carvedilol", "6.25", "mg", "PO", 2),
        _rx("Carvedilol", "6.25", "mg", "PO", 2, start="2190-03-03 10:00:00"),          # the same order, renewed
        _rx("Furosemide", "40", "mg", "IV", 2),
        _rx("Furosemide", "80", "mg", "PO", 1, start="2190-03-04 08:00:00"),            # a second, different order
        _rx("CefTAZidime", "1", "g", "IV", 3),
        _rx("Heparin", "5000", "UNIT", "SC", 3),
        _rx("Heparin Flush (10 units/ml)", "2", "mL", "IV", None),
        _rx("Ampicillin Sodium", "2", "g", "IV", 6),
        _rx("Metoprolol Tartrate", "25", "mg", "PO", 2),
        _rx("Metoprolol Succinate XL", "50", "mg", "PO", 1),
        _rx("Aspirin", "81", "mg", "PO", 1),
    ],
    SECOND: [_rx("Carvedilol", "25", "mg", "PO", 2, start="2191-09-10 12:00:00", stop="2191-09-14 09:00:00"),
             _rx("Warfarin", "5", "mg", "PO", 1, start="2191-09-10 12:00:00", stop="2191-09-14 09:00:00")],
}


def _state(question, request_hadm_id=None, scope=True):
    temporal = detect_temporal_mode(question)
    d = classify(question, temporal)
    state = {"query": question, "subject_id": SID, "query_type": d.query_type, "query_complexity": d.complexity,
             "temporal_mode": temporal, "classified_by": "rules" if d.confident else "fast_model"}
    if scope:
        state["admission_scope"] = resolve_admission(question, ADMISSIONS, request_hadm_id).as_state()
    return state


@pytest.fixture
def g(monkeypatch):
    """The structured profile, with a prescriptions table that answers per admission, as the SQL does."""
    asked = []
    monkeypatch.setattr(graph, "PROFILE_SETTINGS", {"admission_scope": True, "sql_paths": True, "lab_table": "labevents_full"})
    monkeypatch.setattr(graph.structured, "fetch",
                        lambda kind, sid, hadm: asked.append((kind, sid, hadm)) or list(ORDERS.get(hadm, [])))
    graph.asked = asked
    return graph


def _answer(g, question, request_hadm_id=None):
    state = _state(question, request_hadm_id)
    assert g.route_from_triage(state) == "structured_lookup"
    out = g.structured_lookup(state)
    assert g.route_after_structured_lookup(out) == "finalize" and out["review_status"] == "auto_approved"
    assert out["admission_scope_applied"] is True and all(c["verified"] and c["label"] == "R1" for c in out["citations"])
    assert "they are not a record of what was administered and not the discharge medication list" in out["final_answer"]
    assert f"Inpatient medication orders for" in out["final_answer"] and f"in admission {FIRST}" in out["final_answer"]
    return out["final_answer"]


# --- dose, route, doses per 24 hours, whole order --------------------------------------------------

@pytest.mark.parametrize("question, drug, line", [
    (f"What dose of carvedilol was ordered {NAMED}?", "Carvedilol", "Carvedilol 6.25 mg PO 2 dose(s) per 24 hours"),
    (f"By what route was ceftazidime given {NAMED}?", "CefTAZidime", "CefTAZidime 1 g IV 3 dose(s) per 24 hours"),
    (f"What was the route of the inpatient heparin order in admission {FIRST}?", "Heparin", "Heparin 5000 UNIT SC 3 dose(s) per 24 hours"),
    (f"How many doses per 24 hours of aspirin were ordered {NAMED}?", "Aspirin", "Aspirin 81 mg PO 1 dose(s) per 24 hours"),
    (f"How many times a day was aspirin ordered in admission {FIRST}?", "Aspirin", "Aspirin 81 mg PO 1 dose(s) per 24 hours"),
    (f"What was the inpatient ampicillin order (dose, route and doses per 24 hours) {NAMED}?", "Ampicillin Sodium",
     "Ampicillin Sodium 2 g IV 6 dose(s) per 24 hours"),
])
def test_one_drugs_order_is_read_from_the_admissions_rows(g, question, drug, line):
    answer = _answer(g, question)
    assert f"Inpatient medication orders for {drug} in admission {FIRST}, from the prescriptions table" in answer
    assert line in answer and g.asked == [("prescriptions", SID, FIRST)]
    others = {r["drug"] for r in ORDERS[FIRST]} - {drug}
    assert not any(f"\n{other} " in answer for other in others)                  # the requested drug only


def test_a_request_level_admission_works_the_same(g):
    answer = _answer(g, "What was the inpatient carvedilol dose?", request_hadm_id=FIRST)
    assert "Carvedilol 6.25 mg PO 2 dose(s) per 24 hours" in answer and g.asked == [("prescriptions", SID, FIRST)]


def test_a_repeated_order_is_listed_once_and_distinct_orders_are_all_listed(g):
    answer = _answer(g, f"What dose of carvedilol was ordered {NAMED}?")
    assert "2 order(s), 1 distinct" in answer and answer.count("Carvedilol 6.25 mg PO") == 1
    answer = _answer(g, f"What was the inpatient furosemide dose in admission {FIRST}?")
    assert "2 order(s), 2 distinct" in answer                                    # both, never a guess at one
    assert "Furosemide 40 mg IV 2 dose(s) per 24 hours" in answer and "Furosemide 80 mg PO 1 dose(s) per 24 hours" in answer


def test_a_missing_field_is_said_to_be_missing(g):
    answer = _answer(g, f"How many doses per 24 hours of heparin flush were ordered {NAMED}?")
    assert "Heparin Flush (10 units/ml) 2 mL IV (doses per 24 hours not recorded)" in answer and "Heparin 5000" not in answer


# --- the right admission, and only it ---------------------------------------------------------------

def test_orders_from_another_admission_are_never_in_the_answer(g):
    answer = _answer(g, f"What dose of carvedilol was ordered {NAMED}?")
    assert "Carvedilol 25 mg" not in answer and "2191" not in answer             # the SECOND admission's carvedilol
    state = _state(f"What dose of carvedilol was ordered in admission {SECOND}?")
    out = g.structured_lookup(state)
    assert g.asked[-1] == ("prescriptions", SID, SECOND) and "Carvedilol 25 mg PO" in out["final_answer"] and "6.25" not in out["final_answer"]
    assert ":sid" in sl.SQL["prescriptions"] and "hadm_id = :hadm" in sl.SQL["prescriptions"]    # constrained in SQL


def test_a_drug_not_ordered_in_that_admission_is_a_miss(g):
    state = _state(f"What dose of warfarin was ordered {NAMED}?")                # warfarin is in the SECOND admission only
    out = g.structured_lookup(state)
    assert out == {"node_trail": ["structured_lookup"]} and g.route_after_structured_lookup(out) == "patient_retrieval"


# --- the drug named, never a similar one -------------------------------------------------------------

def test_an_exact_name_is_not_widened_to_a_longer_one(g):
    answer = _answer(g, f"What was the inpatient heparin dose in admission {FIRST}?")
    assert "Heparin 5000 UNIT SC" in answer and "Flush" not in answer and "1 order(s), 1 distinct" in answer


def test_an_ambiguous_name_matches_nothing_and_arms_the_refusal_guard(g):
    state = _state(f"What dose of metoprolol was ordered {NAMED}?")              # tartrate and succinate are both there
    assert g.route_from_triage(state) == "structured_lookup"
    out = g.structured_lookup(state)
    assert "structured_evidence" not in out and "final_answer" not in out and out["structured_rows"] == 2
    assert g.route_after_structured_lookup(out) == "patient_retrieval"
    assert sl.match_drug("metoprolol", ORDERS[FIRST]) == ([], ["Metoprolol Succinate XL", "Metoprolol Tartrate"])
    rows, _ = sl.match_drug("metoprolol tartrate", ORDERS[FIRST])
    assert [r["drug"] for r in rows] == ["Metoprolol Tartrate"]


def test_matching_is_whole_words_with_no_spelling_tolerance():
    rows = ORDERS[FIRST]
    assert sl.match_drug("ceftazidime", rows)[0][0]["drug"] == "CefTAZidime"     # case does not matter
    assert sl.match_drug("ampicillin", rows)[0][0]["drug"] == "Ampicillin Sodium"   # the only drug with that word
    for name in ("hepari", "heparins", "carvedilo", "amp", "sodium chloride", ""):
        assert sl.match_drug(name, rows) == ([], [])


# --- what must stay on note retrieval ----------------------------------------------------------------

@pytest.mark.parametrize("question", [
    f"What dose of carvedilol was the patient discharged on {NAMED}?",
    f"What carvedilol dose was she sent home on after the admission that ended on 2190-03-06?",
    f"What is the patient's current carvedilol dose?",
    f"What home dose of carvedilol was she taking before admission {FIRST}?",
    f"What dose of carvedilol was prescribed {NAMED}?",
    f"Was she given heparin {NAMED}?",                                           # yes/no: an order is not proof it was given
    f"Why was the carvedilol dose changed {NAMED}?",
    f"What dose of carvedilol and furosemide was ordered {NAMED}?",              # two drugs
])
def test_discharge_home_current_and_narrative_questions_are_not_routed(g, question):
    state = _state(question, request_hadm_id=FIRST)                              # an admission is resolved either way
    assert state["admission_scope"]["status"] == "resolved"
    assert g._drug_order_request(state) is None and g.route_from_triage(state) != "structured_lookup" and g.asked == []


@pytest.mark.parametrize("question", [
    "What dose of carvedilol was ordered?",                                      # no admission named
    "What dose of carvedilol was ordered during the admission that ended on 2199-01-01?",   # no such admission
])
def test_without_a_resolved_admission_nothing_is_queried(g, question):
    state = _state(question)
    assert state["admission_scope"]["status"] != "resolved"
    assert g._drug_order_request(state) is None and g.route_from_triage(state) == "patient_retrieval" and g.asked == []


@pytest.mark.parametrize("settings", [{"admission_scope": False, "sql_paths": False},     # control
                                      {"admission_scope": True, "sql_paths": False}])     # scoped
def test_control_and_scoped_never_use_the_drug_path(monkeypatch, settings):
    monkeypatch.setattr(graph, "PROFILE_SETTINGS", settings)
    monkeypatch.setattr(graph.structured, "fetch", lambda *a, **k: pytest.fail("disabled profile: must not query"))
    state = _state(f"What dose of carvedilol was ordered {NAMED}?")
    assert graph._drug_order_request(state) is None and graph.route_from_triage(state) != "structured_lookup"


# --- neighbours: the admissions path, the whole list, the lab path ----------------------------------

@pytest.mark.parametrize("question", [
    f"How many doses per 24 hours of aspirin were ordered {NAMED}?",
    f"How many times a day was aspirin ordered in admission {FIRST}?",
])
def test_the_admissions_path_does_not_claim_a_drug_order_question(g, question):
    state = _state(question)
    assert g.wants_encounter_lookup(g._decision_of(state), question, SID)        # "how many" + "admission" still matches its word list
    assert g.route_from_triage(state) == "structured_lookup"
    assert g.route_from_triage(_state("How many hospital admissions does the patient have?")) == "encounter_lookup"


def test_the_whole_list_question_is_answered_as_before(g):
    state = _state(f"What inpatient medications were ordered {NAMED}?")
    assert g._structured_request(state) == ("prescriptions", FIRST) and g._drug_order_request(state) is None
    out = g.structured_lookup(state)
    answer = out["final_answer"]
    assert answer.startswith(f"Inpatient medication orders for admission {FIRST}, from the prescriptions table: 11 order(s), 10 distinct.")
    assert all(drug in answer for drug in ("Carvedilol", "Furosemide", "Heparin Flush", "Metoprolol Succinate XL", "Aspirin"))
    assert "not recorded)" not in answer                                         # the list keeps its own wording
    assert sl.structured_kind("Was she given heparin?") is None                  # and its gate is unchanged


def test_lab_questions_are_not_taken_by_the_drug_path(g):
    for question in (f"What was the lowest potassium {NAMED}?", f"What was the latest creatinine in admission {FIRST}?",
                     f"How did the sodium change from the first to the last measurement {NAMED}?"):
        state = _state(question)
        assert g._drug_order_request(state) is None and g.route_from_triage(state) == "lab_lookup"


def test_readiness_errors_still_propagate(g, monkeypatch):
    def not_ready(*a):
        raise DataSourceNotReady(STRUCTURED, "the latest structured load is running")
    monkeypatch.setattr(g.structured, "fetch", not_ready)
    with pytest.raises(DataSourceNotReady):
        g.structured_lookup(_state(f"What dose of carvedilol was ordered {NAMED}?"))


@pytest.mark.parametrize("question, drug", [
    ("What dose of carvedilol was ordered?", "carvedilol"),
    ("What was the inpatient furosemide dose?", "furosemide"),
    ("By what route was ceftazidime given?", "ceftazidime"),
    ("How many times a day was lisinopril ordered?", "lisinopril"),
    ("What dose of potassium chloride was ordered?", "potassium chloride"),
    ("What was the inpatient heparin flush order?", "heparin flush"),
    ("Was heparin ordered?", None),                       # asks for no dose, route, rate or order
    ("What was the dose?", None),                         # no drug
    ("What medications were given?", None),               # the whole-list question
    ("What was the lowest potassium?", None),
    ("", None),
])
def test_the_single_drug_wording_gate(question, drug):
    assert sl.drug_order_question(question) == drug
