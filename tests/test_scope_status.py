"""Admission-scope status in /ask responses, logs and the UI (data-foundation plan,
E6 / decision G1). No services: the graph is replaced by the state it would return."""
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.api.app as api
from src.api.schemas import AdmissionScope, AskResponse
from tests.test_api import STATE

client = TestClient(api.app)
JS = (Path(api.__file__).parent / "ui" / "app.js").read_text(encoding="utf-8")
HADM = 91000001


def _scope(status, hadm_id=None, source=None, reason=None, phrase=None):
    return {"status": status, "hadm_id": hadm_id, "source": source, "reason": reason, "phrase": phrase,
            "retrieval_query": "q"}


# name, graph state added to STATE, request hadm_id, expected admission_scope, UI warns
CASES = [
    ("scoped by a rule", {"admission_scope": _scope("resolved", HADM, "rule:last", phrase="During her last admission"),
                          "admission_scope_applied": True}, None,
     {"applied": True, "requested": True, "status": "resolved", "hadm_id": HADM, "source": "rule:last", "reason": None}, False),
    ("scoped by the request", {"admission_scope": _scope("resolved", HADM, "request"), "admission_scope_applied": True}, HADM,
     {"applied": True, "requested": True, "status": "resolved", "hadm_id": HADM, "source": "request", "reason": None}, False),
    ("named but unresolved", {"admission_scope": _scope("unresolved", None, "rule:date", "no admission matches 2181-01-01", "x")}, None,
     {"applied": False, "requested": True, "status": "unresolved", "hadm_id": None, "source": "rule:date",
      "reason": "no admission matches 2181-01-01"}, True),
    ("named but ambiguous", {"admission_scope": _scope("ambiguous", None, "rule:date", "2 admissions match: [1, 2]", "x")}, None,
     {"applied": False, "requested": True, "status": "ambiguous", "hadm_id": None, "source": "rule:date",
      "reason": "2 admissions match: [1, 2]"}, True),
    ("no admission named", {"admission_scope": _scope("none")}, None,
     {"applied": False, "requested": False, "status": "none", "hadm_id": None, "source": None, "reason": None}, False),
    ("control profile", {}, None,
     {"applied": False, "requested": False, "status": "not_enabled", "hadm_id": None, "source": None,
      "reason": "admission scoping is not enabled in the control data profile"}, False),
    ("control profile with a request admission", {}, HADM,
     {"applied": False, "requested": True, "status": "not_enabled", "hadm_id": None, "source": None,
      "reason": "admission scoping is not enabled in the control data profile"}, True),
    # resolved, but a structured path answered and never searched notes: not applied, and said so
    ("resolved but answered elsewhere", {"admission_scope": _scope("resolved", HADM, "rule:last", phrase="x")}, None,
     {"applied": False, "requested": True, "status": "resolved", "hadm_id": HADM, "source": "rule:last",
      "reason": "the answer did not come from note retrieval, so the admission scope was not applied"}, True),
]


def _ask(monkeypatch, extra, hadm_id):
    monkeypatch.setattr(api, "_ensure_subject", lambda sid: None)
    monkeypatch.setattr(api, "_ensure_admission", lambda sid, hadm: None)
    monkeypatch.setattr(api, "_run_ask", lambda *a, **k: ({}, {**STATE, **extra}))
    payload = {"subject_id": 90000001, "query": "what happened?"}
    if hadm_id is not None:
        payload["hadm_id"] = hadm_id
    return client.post("/ask", json=payload)


@pytest.mark.parametrize("name,extra,hadm_id,expected,warns", CASES, ids=[c[0] for c in CASES])
def test_ask_reports_scope_truthfully_and_leaves_the_answer_alone(monkeypatch, caplog, name, extra, hadm_id, expected, warns):
    with caplog.at_level(logging.INFO, logger="lumen.api"):
        r = _ask(monkeypatch, extra, hadm_id)
    body = r.json()
    assert r.status_code == 200 and body["admission_scope"] == expected
    assert body["answer"] == STATE["final_answer"]                    # metadata only; the answer text is untouched
    # the same facts in the structured log
    (fields,) = [rec.lumen_fields for rec in caplog.records if getattr(rec, "lumen_event", "") == "ask_completed"]
    assert {k[len("scope_"):]: v for k, v in fields.items() if k.startswith("scope_")} == expected
    # internally consistent
    scope = AdmissionScope(**body["admission_scope"])
    assert not scope.applied or (scope.status == "resolved" and scope.hadm_id is not None)
    assert scope.requested or (scope.status in ("none", "not_enabled") and not scope.applied)
    assert (scope.reason is None) == (scope.applied or scope.status == "none")
    # what the page does with it: warn exactly when an admission was asked for and not applied
    assert (scope.requested and not scope.applied) is warns


def test_page_warns_only_on_requested_and_not_applied_and_adds_no_admission_control():
    assert "admission_scope" in AskResponse.model_fields
    assert "scope.requested && !scope.applied" in JS
    assert '"Admission scope was not applied to this answer."' in JS    # true for retrieval fallback and structured answers alike
    assert "whole patient record" not in JS
    assert 'el("div", "scope-reason", scope.reason)' in JS              # the specific reason sits underneath
    assert "hadm_id:" not in JS                                       # the page still sends no admission
    assert "{ subject_id: msg.subject, query }" in JS


def test_patient_retrieval_marks_scope_applied_only_when_it_scoped(monkeypatch):
    import src.agents.graph as g

    class _Retriever:
        def search(self, **kwargs):
            return []
    monkeypatch.setattr(g, "get_retrievers", lambda: (_Retriever(), None))
    monkeypatch.setattr(g, "load_stay_window", lambda hadm: ("s", "e"))
    resolved = {"query": "q", "subject_id": 7, "temporal_mode": "all",
                "admission_scope": _scope("resolved", HADM, "rule:last")}
    monkeypatch.setattr(g, "PROFILE_SETTINGS", {"admission_scope": True})
    assert g.patient_retrieval(resolved)["admission_scope_applied"] is True
    assert "admission_scope_applied" not in g.patient_retrieval({**resolved, "admission_scope": _scope("unresolved")})
    monkeypatch.setattr(g, "PROFILE_SETTINGS", {"admission_scope": False})
    assert "admission_scope_applied" not in g.patient_retrieval(resolved)


# --- the same facts on the log line as it is written --------------------------------------------
# The fields above are what the handler passes to the logger; only whitelisted names are written.
# Admission ids and the free-text reason are deliberately not logged: the reason can quote a date
# or an admission id from the question, and can list the patient's admission ids.
LOGGED = ("applied", "requested", "status", "source")


@pytest.mark.parametrize("name,extra,hadm_id,expected,warns", CASES, ids=[c[0] for c in CASES])
def test_scope_status_reaches_the_written_log_line(monkeypatch, caplog, name, extra, hadm_id, expected, warns):
    import json
    from src.obs.logging import JsonFormatter
    with caplog.at_level(logging.INFO, logger="lumen.api"):
        _ask(monkeypatch, extra, hadm_id)
    (record,) = [rec for rec in caplog.records if getattr(rec, "lumen_event", "") == "ask_completed"]
    raw = JsonFormatter().format(record)
    line = json.loads(raw)
    for field in LOGGED:
        if expected[field] is None:
            assert f"scope_{field}" not in line                        # no value: left out, like every other field
        else:
            assert line[f"scope_{field}"] == expected[field]
    assert "scope_hadm_id" not in line and "scope_reason" not in line
    assert str(HADM) not in raw and STATE["final_answer"] not in raw and "what happened?" not in raw
