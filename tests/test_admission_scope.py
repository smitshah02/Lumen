"""Deterministic admission resolver (data-foundation plan, E3). Pure: no database."""
from datetime import datetime

import pytest

from src.agents.admission_scope import resolve_admission

# (hadm_id, admittime, dischtime), oldest first; long numeric ids, as in MIMIC. 2000300 and 2000400 touch on 2180-08-07:
# one is discharged the morning the next is admitted.
ADMISSIONS = [
    (2000100, datetime(2180, 5, 6, 22, 0), datetime(2180, 5, 7, 17, 0)),
    (2000200, datetime(2180, 6, 26, 18, 0), datetime(2180, 6, 27, 18, 0)),
    (2000300, datetime(2180, 7, 23, 12, 0), datetime(2180, 8, 7, 9, 0)),
    (2000400, datetime(2180, 8, 7, 20, 0), datetime(2180, 8, 9, 11, 0)),
]

RESOLVED = [
    # question, hadm_id, source, phrase, retrieval text
    ("For hadm_id 2000200, what was the discharge diagnosis?", 2000200, "rule:hadm_id",
     "For hadm_id 2000200", "what was the discharge diagnosis?"),
    ("During the admission that ended on 2180-05-07, why was spironolactone 50 mg chosen?", 2000100, "rule:date",
     "During the admission that ended on 2180-05-07", "why was spironolactone 50 mg chosen?"),
    ("What was the lipase value in the results of the admission that ended on 2180-05-07?", 2000100, "rule:date",
     "of the admission that ended on 2180-05-07", "What was the lipase value in the results?"),
    ("What dose of torsemide was the patient discharged on at the discharge on 2180-06-27?", 2000200, "rule:date",
     "at the discharge on 2180-06-27", "What dose of torsemide was the patient discharged on?"),
    ("What happened during the 2180-06-26 admission to her potassium?", 2000200, "rule:date",
     "during the 2180-06-26 admission", "What happened to her potassium?"),
    ("In the admission that started on 2180-08-07, what imaging was done?", 2000400, "rule:date",
     "In the admission that started on 2180-08-07", "what imaging was done?"),
    ("During her first admission, what was the chief complaint?", 2000100, "rule:first",
     "During her first admission", "what was the chief complaint?"),
    ("During her last admission, what happened to her kidney function?", 2000400, "rule:last",
     "During her last admission", "what happened to her kidney function?"),
    ("What antibiotics were given in the most recent hospitalization?", 2000400, "rule:last",
     "in the most recent hospitalization", "What antibiotics were given?"),
    # admitted on / discharged on <date>: the verb stays in the question
    ("What was the plan when she was admitted on 2180-06-26?", 2000200, "rule:date",
     "admitted on 2180-06-26", "What was the plan when she was admitted?"),
    ("What medications was she taking when discharged on 2180-05-07?", 2000100, "rule:date",
     "discharged on 2180-05-07", "What medications was she taking when discharged?"),
    # nth admission, by admit time
    ("During the second admission, what was the chief complaint?", 2000200, "rule:ordinal",
     "During the second admission", "what was the chief complaint?"),
    ("What imaging was done in her 3rd hospitalization?", 2000300, "rule:ordinal",
     "in her 3rd hospitalization", "What imaging was done?"),
    ("What happened in the fourth admission to her sodium?", 2000400, "rule:ordinal",
     "in the fourth admission", "What happened to her sodium?"),
    # previous admission, anchored by one other explicit reference
    ("What was the creatinine in the admission before her last admission?", 2000300, "rule:previous",
     "in the admission before; her last admission", "What was the creatinine?"),
    ("In the hospitalization prior to hadm_id 2000300, what was the discharge diagnosis?", 2000200, "rule:previous",
     "In the hospitalization prior to; hadm_id 2000300", "what was the discharge diagnosis?"),
    ("What was done in the admission before the admission that ended on 2180-06-27?", 2000100, "rule:previous",
     "in the admission before; the admission that ended on 2180-06-27", "What was done?"),
]


@pytest.mark.parametrize("question,hadm_id,source,phrase,retrieval", RESOLVED)
def test_resolves_one_admission_and_removes_only_its_phrase(question, hadm_id, source, phrase, retrieval):
    r = resolve_admission(question, ADMISSIONS)
    assert (r.status, r.hadm_id, r.source, r.phrase, r.retrieval_query) == ("resolved", hadm_id, source, phrase, retrieval)
    assert r.reason is None


NOT_APPLIED = [
    # question, status, source, reason fragment
    ("During the admission that ended on 2181-01-01, what was the plan?", "unresolved", "rule:date", "no admission matches 2181-01-01"),
    ("For hadm_id 999999, what was the plan?", "unresolved", "rule:hadm_id", "999999 is not an admission of this patient"),
    # 2180-08-07 is inside two stays: never pick one
    ("What happened during the admission on 2180-08-07?", "ambiguous", "rule:date", "2 admissions match: [2000300, 2000400]"),
    ("Compare her first admission with her last admission.", "ambiguous", None, "more than one admission"),
    # admitted on / discharged on a date that matches nothing, or the wrong end of a stay
    ("What was the plan when she was admitted on 2180-05-07?", "unresolved", "rule:date", "no admission matches 2180-05-07"),
    ("What was she taking when discharged on 2180-05-06?", "unresolved", "rule:date", "no admission matches 2180-05-06"),
    # nth beyond the record
    ("What happened in the seventh admission?", "unresolved", "rule:ordinal", "has 4 admissions on record, not 7"),
    # previous admission is never guessed
    ("What happened in the previous admission?", "unresolved", "rule:previous", "no anchor admission"),
    ("What happened in the admission before her first admission?", "unresolved", "rule:previous", "no admission precedes hadm_id 2000100"),
    ("What happened in the admission before the admission on 2180-08-07?", "ambiguous", "rule:previous", "anchor admission is not unique"),
    ("What happened in the admission before the admission that ended on 2181-01-01?", "unresolved", "rule:previous", "anchor admission is unknown"),
    ("In the previous admission, before the second admission and the last admission, what happened?", "ambiguous", "rule:previous", "more than one admission"),
]


@pytest.mark.parametrize("question,status,source,reason", NOT_APPLIED)
def test_zero_or_several_matches_never_choose_and_leave_the_question_whole(question, status, source, reason):
    r = resolve_admission(question, ADMISSIONS)
    assert (r.status, r.hadm_id, r.source) == (status, None, source)
    assert reason in r.reason and r.phrase
    assert r.retrieval_query == question


NO_REFERENCE = [
    "What medications is the patient taking?",
    "What was the creatinine on 2180-05-06?",                   # a clinical date, not an admission
    "What did the chest x-ray from 2180-07-24 show?",
    "What were the admission labs?",                            # "admission" with no date or ordinal
    "What was her last creatinine?",                            # "last" without an admission word
    "Was she taking aspirin for the first 3 days after the stent?",
    "What was she discharged on?",                              # "discharged on" with no date
    "Was the second dose of vancomycin given?",                 # an ordinal with no admission word
    "What did the previous chest x-ray show?",                  # "previous" with no admission word
    "Was she admitted on aspirin?",
    "What was she taking on admission before the stent was placed?",
]


@pytest.mark.parametrize("question", NO_REFERENCE)
def test_questions_that_name_no_admission_are_untouched(question):
    r = resolve_admission(question, ADMISSIONS)
    assert (r.status, r.hadm_id, r.source, r.reason, r.phrase, r.retrieval_query) == ("none", None, None, None, None, question)


def test_request_hadm_id_wins_over_the_text_and_is_never_trusted_blindly():
    q = "During her last admission, what happened to her kidney function?"
    r = resolve_admission(q, ADMISSIONS, request_hadm_id=2000200)
    assert (r.status, r.hadm_id, r.source, r.retrieval_query) == ("resolved", 2000200, "request", q)
    bad = resolve_admission(q, ADMISSIONS, request_hadm_id=555)
    assert (bad.status, bad.hadm_id, bad.source) == ("unresolved", None, "request")
    assert "555 is not an admission of this patient" in bad.reason


def test_last_and_first_with_tied_or_missing_admissions_do_not_guess():
    tied = [(1, datetime(2180, 1, 1, 8), None), (2, datetime(2180, 1, 1, 8), None)]
    assert resolve_admission("During her last admission, what was done?", tied).status == "ambiguous"
    none = resolve_admission("During her last admission, what was done?", [])
    assert (none.status, none.hadm_id) == ("unresolved", None) and "no admissions" in none.reason


def test_a_question_that_is_only_the_reference_keeps_its_words():
    r = resolve_admission("When was her last admission?", ADMISSIONS)
    assert (r.status, r.hadm_id, r.retrieval_query) == ("resolved", 2000400, "When was her last admission?")


def test_triage_records_scope_only_for_profiles_that_enable_it(monkeypatch):
    import src.agents.graph as g
    calls = []
    monkeypatch.setattr(g, "load_admissions", lambda sid: calls.append(sid) or ADMISSIONS)
    monkeypatch.setattr(g, "chat_for", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model in tests")))
    state = {"query": "During her last admission, what happened to her kidney function?", "subject_id": 7}

    monkeypatch.setattr(g, "PROFILE_SETTINGS", {"admission_scope": False})
    assert "admission_scope" not in g.triage(state) and calls == []          # control: unchanged, no query

    monkeypatch.setattr(g, "PROFILE_SETTINGS", {"admission_scope": True})
    scope = g.triage(state)["admission_scope"]
    assert calls == [7] and (scope["status"], scope["hadm_id"], scope["source"]) == ("resolved", 2000400, "rule:last")
    assert scope["retrieval_query"] == "what happened to her kidney function?"
    assert "admission_scope" not in g.triage({"query": state["query"], "subject_id": None})   # no patient, no scope


def test_request_hadm_id_wins_even_when_the_question_says_previous_admission():
    q = "What was the creatinine in the previous admission?"
    r = resolve_admission(q, ADMISSIONS, request_hadm_id=2000300)
    assert (r.status, r.hadm_id, r.source, r.phrase, r.retrieval_query) == ("resolved", 2000300, "request", None, q)
    # also when the question carries its own anchor: the request still wins
    anchored = "What was the creatinine in the admission before her last admission?"
    r = resolve_admission(anchored, ADMISSIONS, request_hadm_id=2000100)
    assert (r.status, r.hadm_id, r.source, r.retrieval_query) == ("resolved", 2000100, "request", anchored)
    bad = resolve_admission(q, ADMISSIONS, request_hadm_id=555)
    assert (bad.status, bad.hadm_id, bad.source) == ("unresolved", None, "request")


def test_nth_and_previous_do_not_guess_across_tied_admit_times():
    tied = [(1, datetime(2180, 1, 1, 8), None), (2, datetime(2180, 2, 1, 8), None), (3, datetime(2180, 2, 1, 8), None),
            (4, datetime(2180, 3, 1, 8), None)]
    second = resolve_admission("What happened in the second admission?", tied)
    assert (second.status, second.hadm_id) == ("ambiguous", None) and "[2, 3]" in second.reason
    assert resolve_admission("What happened in the first admission?", tied).hadm_id == 1
    prev = resolve_admission("What happened in the admission before hadm_id 4000004?",
                             [(i + 4000000, t, d) for i, t, d in tied])
    assert (prev.status, prev.hadm_id, prev.source) == ("ambiguous", None, "rule:previous")
