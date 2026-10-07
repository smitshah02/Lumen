"""Optional hadm_id on /ask (data-foundation plan, E5 / decision R3). No services:
the subject check, the admission check and the graph are replaced per test."""
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import src.api.app as api
from tests.test_api import STATE

client = TestClient(api.app)
SUBJECT, MINE, THEIRS = 90000001, 91000001, 91000777
ADMISSIONS = [(MINE - 1, datetime(2024, 1, 5, 9), datetime(2024, 1, 9, 12)),
              (MINE, datetime(2024, 3, 14, 9), datetime(2024, 3, 18, 12))]


@pytest.fixture
def calls(monkeypatch):
    """Real handler; fake database lookups and graph. Records what reached the graph."""
    seen = []
    monkeypatch.setattr(api, "_ensure_subject", lambda sid: None)

    def ensure_admission(sid, hadm):
        if (sid, hadm) != (SUBJECT, MINE):
            raise api.AdmissionNotFound(hadm)
    monkeypatch.setattr(api, "_ensure_admission", ensure_admission)
    monkeypatch.setattr(api, "_run_ask", lambda *a, **k: seen.append((a, k)) or ({}, STATE))
    return seen


def test_valid_hadm_id_reaches_the_graph(calls):
    r = client.post("/ask", json={"subject_id": SUBJECT, "query": "what was the plan?", "hadm_id": MINE})
    assert r.status_code == 200 and r.json()["status"] == "completed"
    (args, kwargs), = calls
    assert args[:2] == ("what was the plan?", SUBJECT) and kwargs == {"hadm_id": MINE}


@pytest.mark.parametrize("hadm_id", [THEIRS, 99999999])     # another patient's admission; one that does not exist
def test_hadm_id_that_is_not_this_patients_is_422_and_never_runs(calls, hadm_id):
    r = client.post("/ask", json={"subject_id": SUBJECT, "query": "what was the plan?", "hadm_id": hadm_id})
    body = r.json()
    assert r.status_code == 422 and body["error"] == "validation_error"
    assert body["detail"] == [{"loc": ["body", "hadm_id"], "msg": "hadm_id is not an admission of this subject",
                               "type": "value_error"}]
    assert str(hadm_id) not in r.text and calls == []           # value not echoed; the graph never ran


@pytest.mark.parametrize("hadm_id", [0, -4, "abc", 1.5, 2_147_483_648])
def test_malformed_hadm_id_is_422_before_any_lookup(calls, hadm_id):
    r = client.post("/ask", json={"subject_id": SUBJECT, "query": "what was the plan?", "hadm_id": hadm_id})
    assert r.status_code == 422 and r.json()["error"] == "validation_error" and calls == []


def test_no_hadm_id_makes_the_call_it_always_made(calls):
    for payload in ({"subject_id": SUBJECT, "query": "what was the plan?"},
                    {"subject_id": SUBJECT, "query": "what was the plan?", "hadm_id": None}):
        assert client.post("/ask", json=payload).status_code == 200
    assert [(len(a), k) for a, k in calls] == [(4, {}), (4, {})]


def test_admission_check_asks_for_this_subject_and_this_admission(monkeypatch):
    asked = []

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, stmt, params):
            asked.append((str(stmt), params))
            return type("R", (), {"scalar": lambda self: params == {"h": MINE, "s": SUBJECT}})()
    monkeypatch.setattr(api.storage, "engine", type("E", (), {"connect": lambda self: _Conn()})())
    api._ensure_admission(SUBJECT, MINE)
    with pytest.raises(api.AdmissionNotFound):
        api._ensure_admission(SUBJECT, THEIRS)
    assert "hadm_id = :h AND subject_id = :s" in asked[0][0]


def test_run_once_puts_the_request_admission_in_state_only_when_given(monkeypatch):
    from src.agents import run_graph

    class _Graph:
        def invoke(self, state, config=None):
            self.state = state
            return {}
    g = _Graph()
    run_graph.run_once(g, "q", SUBJECT, "t-1")
    assert g.state == {"query": "q", "subject_id": SUBJECT, "thread_id": "t-1"}
    run_graph.run_once(g, "q", SUBJECT, "t-2", hadm_id=MINE)
    assert g.state == {"query": "q", "subject_id": SUBJECT, "thread_id": "t-2", "request_hadm_id": MINE}


def _triage(monkeypatch, scoping, state):
    import src.agents.graph as g
    monkeypatch.setattr(g, "load_admissions", lambda sid: ADMISSIONS)
    monkeypatch.setattr(g, "chat_for", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model in tests")))
    monkeypatch.setattr(g, "PROFILE_SETTINGS", {"admission_scope": scoping})
    return g.triage(state)


CONFLICT = "During her first admission, what happened to her kidney function?"


def test_request_hadm_id_wins_over_a_conflicting_question_reference(monkeypatch):
    out = _triage(monkeypatch, True, {"query": CONFLICT, "subject_id": SUBJECT, "request_hadm_id": MINE})
    scope = out["admission_scope"]
    assert (scope["status"], scope["hadm_id"], scope["source"]) == ("resolved", MINE, "request")
    assert scope["retrieval_query"] == CONFLICT                 # the question is not reinterpreted or stripped


def test_without_a_request_hadm_id_the_question_resolver_decides(monkeypatch):
    scope = _triage(monkeypatch, True, {"query": CONFLICT, "subject_id": SUBJECT})["admission_scope"]
    assert (scope["status"], scope["hadm_id"], scope["source"]) == ("resolved", MINE - 1, "rule:first")


def test_control_profile_computes_no_scope_even_with_a_request_hadm_id(monkeypatch):
    out = _triage(monkeypatch, False, {"query": CONFLICT, "subject_id": SUBJECT, "request_hadm_id": MINE})
    assert "admission_scope" not in out
