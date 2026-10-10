"""Request logs name the note-index build a request read (Stage 4, pre-switch).

`build_id` on a request's completion and access log lines is the build the API
gated that request on: the id in the query that decided the patient is servable.
It is null, explicitly, for a request that read no build, and it is never the
configured id under a profile that does not read note_chunks_v2.

These tests read the line as the JSON formatter writes it, not the fields the
caller passed: a field that is not whitelisted never reaches the log.
"""
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api import app as api
from src.obs.logging import JsonFormatter

client = TestClient(api.app)

BUILD = "11111111-2222-3333-4444-555555555555"
OTHER = "99999999-8888-7777-6666-555555555555"
QUERY = "what happened during the stay?"
STATE = {"query_type": "chart_review", "temporal_mode": "all", "review_status": "auto_approved",
         "draft_answer": "SYNTHETIC ANSWER [S1].", "final_answer": "SYNTHETIC ANSWER [S1].",
         "citations": [{"claim": "SYNTHETIC ANSWER [S1].", "label": "S1", "chunk_id": 7, "verified": True}],
         "patient_evidence": [], "node_trail": ["triage", "patient_retrieval", "synthesis", "verification", "finalize"],
         "errors": []}


class _Engine:
    """storage.engine for _ensure_subject: answers its one EXISTS and keeps the parameters it was asked with."""
    def __init__(self, found=True):
        self.found, self.params = found, []

    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        self.params.append(dict(params or {}))
        return SimpleNamespace(scalar=lambda: self.found)


def _profile(monkeypatch, chunk_table, build, found=True):
    engine = _Engine(found)
    monkeypatch.setattr(api, "PROFILE_SETTINGS", {**api.PROFILE_SETTINGS, "chunk_table": chunk_table})
    monkeypatch.setattr(api, "CHUNK_BUILD", build)
    monkeypatch.setattr(api.storage, "engine", engine)
    monkeypatch.setattr(api, "_run_ask", lambda *a, **k: ({}, STATE))
    monkeypatch.setattr(api, "_run_retrieve", lambda *a, **k: ("all", []))
    return engine


def _lines(caplog, event):
    """The log lines for one event, exactly as written."""
    return [json.loads(JsonFormatter().format(r)) for r in caplog.records if getattr(r, "lumen_event", "") == event]


def _ask(caplog):
    with caplog.at_level(logging.INFO, logger="lumen.api"):
        return client.post("/ask", json={"subject_id": 90000001, "query": QUERY})


def test_v2_ask_logs_the_build_that_gated_the_request(monkeypatch, caplog):
    engine = _profile(monkeypatch, "note_chunks_v2", BUILD)
    assert _ask(caplog).status_code == 200
    (done,), (access,) = _lines(caplog, "ask_completed"), _lines(caplog, "http_request")
    assert done["build_id"] == BUILD and access["build_id"] == BUILD
    assert engine.params == [{"b": BUILD, "s": 90000001}]              # the id logged is the id that was queried


def test_v2_retrieve_logs_the_build(monkeypatch, caplog):
    _profile(monkeypatch, "note_chunks_v2", BUILD)
    with caplog.at_level(logging.INFO, logger="lumen.api"):
        r = client.post("/retrieve", json={"subject_id": 90000001, "query": QUERY})
    assert r.status_code == 200
    (done,), (access,) = _lines(caplog, "retrieve_completed"), _lines(caplog, "http_request")
    assert done["build_id"] == BUILD and access["build_id"] == BUILD


def test_control_reports_no_build_even_when_one_is_configured(monkeypatch, caplog):
    _profile(monkeypatch, "note_chunks", BUILD)                        # LUMEN_CHUNK_BUILD set, but control reads note_chunks
    assert _ask(caplog).status_code == 200
    (done,), (access,) = _lines(caplog, "ask_completed"), _lines(caplog, "http_request")
    assert "build_id" in done and done["build_id"] is None             # an explicit null, not a missing key
    assert "build_id" in access and access["build_id"] is None
    assert BUILD not in json.dumps([done, access])


def test_v2_with_no_build_selected_logs_null_and_serves_nothing(monkeypatch, caplog):
    _profile(monkeypatch, "note_chunks_v2", None, found=False)
    assert _ask(caplog).status_code == 404
    (access,) = _lines(caplog, "http_request")
    assert "build_id" in access and access["build_id"] is None
    assert not _lines(caplog, "ask_completed")


def test_a_patient_outside_the_build_is_404_and_names_the_build_consulted(monkeypatch, caplog):
    _profile(monkeypatch, "note_chunks_v2", BUILD, found=False)
    r = _ask(caplog)
    assert r.status_code == 404 and r.json()["error"] == "subject_not_found"
    (access,) = _lines(caplog, "http_request")
    assert access["build_id"] == BUILD and access["status"] == "subject_not_found"


def test_a_failure_after_the_build_was_resolved_still_names_it(monkeypatch, caplog):
    _profile(monkeypatch, "note_chunks_v2", BUILD)

    def crash(*a, **k):
        raise RuntimeError("SYNTHETIC failure text that must not be logged")
    monkeypatch.setattr(api, "_run_ask", crash)
    assert _ask(caplog).status_code == 500
    (access,) = _lines(caplog, "http_request")
    assert access["build_id"] == BUILD and access["status_code"] == 500


@pytest.mark.parametrize("call", [lambda: client.get("/health"),
                                  lambda: client.post("/ask", json={"subject_id": "not-a-number", "query": QUERY})],
                         ids=["health", "validation error"])
def test_a_request_that_read_no_build_logs_null(monkeypatch, caplog, call):
    _profile(monkeypatch, "note_chunks_v2", BUILD)
    _ask(caplog)                                                       # a request that did read the build, first
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="lumen.api"):
        call()
    (access,) = _lines(caplog, "http_request")
    assert "build_id" in access and access["build_id"] is None         # nothing carried over from the request before


def test_the_build_is_resolved_per_request_not_remembered(monkeypatch, caplog):
    _profile(monkeypatch, "note_chunks_v2", BUILD)
    _ask(caplog)
    monkeypatch.setattr(api, "CHUNK_BUILD", OTHER)
    caplog.clear()
    _ask(caplog)
    assert [l["build_id"] for l in _lines(caplog, "ask_completed")] == [OTHER]


def test_the_new_field_adds_nothing_else_to_the_line(monkeypatch, caplog):
    _profile(monkeypatch, "note_chunks", None)
    _ask(caplog)
    control = _lines(caplog, "ask_completed")[0]
    caplog.clear()
    _profile(monkeypatch, "note_chunks_v2", BUILD)
    _ask(caplog)
    v2 = _lines(caplog, "ask_completed")[0]
    assert set(v2) == set(control)                                     # same keys; only the value of build_id differs
    assert v2["build_id"] == BUILD and control["build_id"] is None
    raw = json.dumps(_lines(caplog, "ask_completed") + _lines(caplog, "http_request"))
    assert QUERY not in raw and "SYNTHETIC" not in raw                 # no question, answer or claim text


def test_formatter_writes_null_only_for_fields_where_null_is_information():
    record = logging.LogRecord("lumen.api", logging.INFO, __file__, 1, "ask_completed", None, None)
    record.lumen_event, record.lumen_fields = "ask_completed", {"build_id": None, "llm_ms": None, "answer": "SYNTHETIC"}
    line = json.loads(JsonFormatter().format(record))
    assert line["build_id"] is None and "llm_ms" not in line and "answer" not in line
