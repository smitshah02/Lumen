"""Structured-truth regressions, by QUESTION CLASS.

The black-box failures these cover were invisible to tests that only asserted
routing: a trend answered from five note chunks and an admission count read
out of whichever notes retrieval kept both "passed". Every test here asserts
the answer against the rows the structured table holds.
"""
import pytest

from src.agents import citations
from src.agents.classify import classify, encounter_intents, wants_deterministic_lab, wants_encounter_lookup
from src.retrieval.hybrid_retriever_v2 import detect_temporal_mode, strip_temporal_intent
from tests.test_routing import QUALIFIED_ITEMS, REFUSAL, _Rows, _cite, _ev, _lab, _mimic_lookup, _resolver, graph_mod, spy  # noqa: F401

# deliberately out of value order, and not monotonic: 1.1 -> 2.4 -> 0.9 -> 1.6
SERIES = [("2150-01-02", 1.1, None), ("2150-03-05", 2.4, None), ("2151-06-01", 0.9, None),
          ("2152-02-10", 1.6, None)]
TREND_QUESTIONS = ["How did creatinine change over the patient's available record?",
                   "What was the creatinine trend?", "How has creatinine changed over time?"]


def _routes_to_lab(question):
    mode = detect_temporal_mode(question)
    return wants_deterministic_lab(classify(question, mode), mode, 1)


# --- lab -------------------------------------------------------------------
def test_latest_single_analyte(graph_mod, monkeypatch, spy):
    out = _mimic_lookup(graph_mod, monkeypatch, "What was the most recent creatinine?", [_lab("Creatinine", *SERIES)])
    assert spy == [] and out["final_answer"] == "The most recent creatinine was 1.6 mg/dL on 2152-02-10 [L1]."


@pytest.mark.parametrize("question", ["What was the earliest creatinine?", "What was the first recorded creatinine value?"])
def test_earliest_single_analyte(graph_mod, monkeypatch, spy, question):
    assert _routes_to_lab(question)
    out = _mimic_lookup(graph_mod, monkeypatch, question, [_lab("Creatinine", *SERIES)])
    assert spy == [] and out["final_answer"] == "The earliest creatinine was 1.1 mg/dL on 2150-01-02 [L1]."
    assert out["review_status"] == "auto_approved"


@pytest.mark.parametrize("question", TREND_QUESTIONS)
def test_trend_reports_the_series_from_labevents(graph_mod, monkeypatch, spy, question):
    assert _routes_to_lab(question)
    out = _mimic_lookup(graph_mod, monkeypatch, question, [_lab("Creatinine", *SERIES)])
    answer = out["final_answer"]
    assert spy == []                                              # zero model calls
    assert "measured 4 times between 2150-01-02 and 2152-02-10" in answer
    assert "first value was 1.1 mg/dL on 2150-01-02 and the most recent was 1.6 mg/dL on 2152-02-10" in answer
    assert "lowest value was 0.9 mg/dL on 2151-06-01 and the highest was 2.4 mg/dL on 2150-03-05" in answer
    # 1.1 -> 1.6 is not "increased": it went to 2.4 and down to 0.9 on the way
    assert "fluctuated" in answer and "higher than the first" in answer and " rose" not in answer
    assert all(c["verified"] for c in out["citations"]) and graph_mod.route_after_lab_lookup(out) == "finalize"
    report = citations.validate(answer, out["lab_evidence"])
    assert report["bad_labels"] == [] and report["cite_rate"] == 1.0


@pytest.mark.parametrize("values,phrase", [
    ([1.0, 1.2, 1.2, 1.9], "rose, with no decrease"), ([1.9, 1.2, 1.0], "fell, with no increase"),
    ([1.0, 1.0], "unchanged"), ([1.0, 3.0, 1.0], "the same as the first"), ([2.0, 3.0, 1.0], "lower than the first"),
])
def test_direction_is_named_only_when_every_step_agrees(graph_mod, values, phrase):
    assert phrase in graph_mod._direction(values)


def test_a_single_value_is_not_a_trend(graph_mod, monkeypatch, spy):
    out = _mimic_lookup(graph_mod, monkeypatch, TREND_QUESTIONS[1], [_lab("Creatinine", SERIES[0])])
    assert "Only one creatinine value" in out["final_answer"] and "no trend" in out["final_answer"]


def test_fetch_returns_the_series_in_time_order_with_both_ends_described(monkeypatch):
    """The SQL orders by charttime; the trend's endpoints depend on it surviving grouping."""
    import src.generation.lab_query as lab_query
    rows = [("Creatinine", 50912, f"2150-{m:02d}-01 06:00", v, "mg/dL", None, "Blood")
            for m, v in ((1, 1.1), (2, 2.4), (3, 0.9))] + [("Creatinine", 50912, "2149-12-01 06:00", None, "mg/dL", None, "Blood")]
    monkeypatch.setattr(lab_query, "engine", type("E", (), {"connect": lambda self: _Rows(rows)})())
    (grp,) = _resolver(*QUALIFIED_ITEMS).fetch(1, [50912], per_lab_cap=100_000)
    assert [v["date"] for v in grp["values"]] == ["2150-01-01", "2150-02-01", "2150-03-01"]
    assert grp["older_non_numeric"] is True and grp["newer_non_numeric"] is False
    assert grp["n_non_numeric"] == 1 and grp["conflicting_earliest"] is False


@pytest.mark.parametrize("question", [
    "How did urine creatinine change over time?",                 # qualifier: another specimen
    "What was the creatinine clearance trend?",                   # qualifier: another analyte
    "What was the earliest urine creatinine?",
    "How did creatinine and BUN change over time?",               # two analytes
    "How did creatinine change over time and why?",               # extra clause
    "How did creatinine change over the last 3 months?",          # a window the template ignores
    "What was the latest creatinine trend?",                      # two temporal intents
])
def test_qualified_or_two_analyte_questions_fall_through(graph_mod, monkeypatch, spy, question):
    out = _mimic_lookup(graph_mod, monkeypatch, question, [_lab("Creatinine", *SERIES)])
    assert "lab_evidence" not in out and graph_mod.route_after_lab_lookup(out) == "patient_retrieval"
    assert out["structured_rows"] == 4                            # ...but the rows were seen


@pytest.mark.parametrize("question,series", [
    # ambiguous analyte: "cholesterol" is total, HDL and LDL in the dictionary
    ("How did cholesterol change over time?", [_lab("Cholesterol, HDL", *SERIES)]),
    # ambiguous specimen: one label drawn from two fluids
    ("How did creatinine change over time?", [_lab("Creatinine", *SERIES, fluids=("blood", "urine"))]),
    # mixed units: min and max across them would be meaningless
    ("How did creatinine change over time?", [_lab("Creatinine", SERIES[0], ("2150-03-05", 97.0, "umol/L"))]),
    # a non-numeric result older than the first number: that number is not the earliest result
    ("What was the earliest creatinine?", [{**_lab("Creatinine", *SERIES), "older_non_numeric": True}]),
    ("How did creatinine change over time?", [{**_lab("Creatinine", *SERIES), "conflicting_earliest": True}]),
    ("How did creatinine change over time?", [_lab("Creatinine", *SERIES, newer_non_numeric=True)]),
])
def test_ambiguous_series_fall_through(graph_mod, monkeypatch, spy, question, series):
    assert "lab_evidence" not in _mimic_lookup(graph_mod, monkeypatch, question, series)


# --- admissions ------------------------------------------------------------
# 12 admissions: more than PATIENT_TOP_K, so no set of retrieved chunks could hold the count.
ADMISSIONS = [(f"21{40 + i}-03-0{1 + i % 9} 10:00:00", f"21{40 + i}-03-1{i % 9} 15:00:00") for i in range(12)]


def _encounter(graph_mod, monkeypatch, question, rows=ADMISSIONS):
    monkeypatch.setattr(graph_mod, "_admissions", lambda sid: list(rows))
    return graph_mod.encounter_lookup({"query": question, "subject_id": 1})


def test_admission_count_is_the_row_count(graph_mod, monkeypatch, spy):
    assert len(ADMISSIONS) > graph_mod.PATIENT_TOP_K
    out = _encounter(graph_mod, monkeypatch, "How many hospital admissions does the patient have?")
    assert spy == [] and out["final_answer"] == "The patient has 12 recorded hospital admissions [A1]."
    assert out["review_status"] == "auto_approved" and graph_mod.route_after_encounter_lookup(out) == "finalize"
    report = citations.validate(out["final_answer"], out["encounter_evidence"])
    assert report["bad_labels"] == [] and report["cite_rate"] == 1.0          # [A1] parses and resolves


def test_earliest_and_latest_admission_dates(graph_mod, monkeypatch, spy):
    first = _encounter(graph_mod, monkeypatch, "When was the earliest admission?")["final_answer"]
    last = _encounter(graph_mod, monkeypatch, "When was the most recent admission?")["final_answer"]
    assert "earliest admission began on 2140-03-01" in first and "2151" not in first
    assert "most recent admission began on 2151-03-03" in last and "2140" not in last


def test_admission_date_is_not_the_discharge_date(graph_mod, monkeypatch, spy):
    answer = _encounter(graph_mod, monkeypatch, "When was the most recent admission?")["final_answer"]
    assert answer == ("The most recent admission began on 2151-03-03; "
                      "that stay's discharge date was 2151-03-12 [A1].")
    open_stay = _encounter(graph_mod, monkeypatch, "When was the most recent admission?",
                           ADMISSIONS + [("2160-01-01 08:00:00", None)])["final_answer"]
    assert "began on 2160-01-01" in open_stay and "discharge date was not recorded" in open_stay


def test_count_plus_most_recent(graph_mod, monkeypatch, spy):
    q = "How many hospital admissions does the patient have, and when was the most recent one?"
    assert wants_encounter_lookup(classify(q, detect_temporal_mode(q)), q, 1)
    out = _encounter(graph_mod, monkeypatch, q)
    assert spy == [] and "12 recorded hospital admissions" in out["final_answer"]
    assert "most recent admission began on 2151-03-03" in out["final_answer"] and len(out["citations"]) == 2


@pytest.mark.parametrize("question", [
    "How many admissions does the patient have, and why was each one needed?",
    "How many admissions were for heart failure?",
    "When was the most recent admission and what medications were started?",
    "When was the patient last discharged?",                       # not an admission question at all
])
def test_an_extra_clause_falls_back(graph_mod, monkeypatch, spy, question):
    out = _encounter(graph_mod, monkeypatch, question)
    assert "encounter_evidence" not in out and "final_answer" not in out
    assert graph_mod.route_after_encounter_lookup(out) == "patient_retrieval"


def test_encounter_routing_needs_a_patient_and_an_intent():
    q = "How many hospital admissions does the patient have?"
    assert not wants_encounter_lookup(classify(q), q, None)
    for other in ("Why was the patient admitted?", "What was the most recent creatinine?"):
        assert not wants_encounter_lookup(classify(other), other, 1), other
    assert encounter_intents("When was the earliest admission?") == ({"earliest"}, True)


def test_lookup_errors_and_empty_tables_fall_back(graph_mod, monkeypatch, spy):
    assert "encounter_evidence" not in _encounter(graph_mod, monkeypatch, "How many admissions does the patient have?", [])
    monkeypatch.setattr(graph_mod, "_admissions", lambda sid: 1 / 0)
    out = graph_mod.encounter_lookup({"query": "How many admissions does the patient have?", "subject_id": 1})
    assert graph_mod.route_after_encounter_lookup(out) == "patient_retrieval"


# --- verification ----------------------------------------------------------
@pytest.mark.parametrize("cites", [[_cite(REFUSAL, [])], []])        # with evidence, and the no-evidence path
def test_refusal_is_not_auto_approved_when_structured_rows_exist(graph_mod, spy, cites):
    state = {"patient_evidence": [_ev()], "draft_answer": REFUSAL, "citations": cites, "structured_rows": 12}
    out = graph_mod.verification(state)
    assert spy == []
    assert out["needs_human_review"] is True and out["review_status"] == "pending"
    assert out["verification"]["unsupported"] == 1 and out["citations"][0]["verified"] is False
    assert graph_mod.route_after_verification(out) == "human_review"
    # without structured rows the same refusal is still a correct, approved decline
    out = graph_mod.verification({**state, "citations": [_cite(REFUSAL, [])], "structured_rows": 0})
    assert out["review_status"] == "auto_approved"


# --- temporal cleanup ------------------------------------------------------
@pytest.mark.parametrize("question", TREND_QUESTIONS)
def test_trend_question_stays_meaningful_after_stripping(question):
    stripped = strip_temporal_intent(question)
    assert "creatinine" in stripped and " ?" not in stripped
    assert "creatinine the patient" not in stripped               # the verb was not cut out
