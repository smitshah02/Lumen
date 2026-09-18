"""API tests — no Postgres, Ollama, MedCPT, BGE or qwen: dependencies are mocked."""

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

import src.api.app as api

client = TestClient(api.app)


def _no_subject_check(monkeypatch):
    monkeypatch.setattr(api, "_ensure_subject", lambda sid: None)


# --- /health + request ids ---------------------------------------------------------
def test_health_ok_with_request_id():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert len(r.headers["X-Request-ID"]) == 32


def test_client_request_id_echoed_and_unsafe_one_replaced():
    assert client.get("/health", headers={"X-Request-ID": "demo-req-0001"}).headers["X-Request-ID"] == "demo-req-0001"
    bad = client.get("/health", headers={"X-Request-ID": "x; rm -rf /"}).headers["X-Request-ID"]
    assert bad != "x; rm -rf /" and len(bad) == 32


# --- validation ------------------------------------------------------------------------
@pytest.mark.parametrize("payload", [
    {"subject_id": 90000001, "query": ""},
    {"subject_id": 90000001, "query": "   "},
    {"query": "most recent creatinine"},
    {"subject_id": 0, "query": "most recent creatinine"},
    {"subject_id": "abc", "query": "most recent creatinine"},
    {"subject_id": 90000001, "query": "x" * 501},
    {"subject_id": 90000001, "query": "ok", "unexpected": 1},
])
def test_ask_validation_error(payload):
    r = client.post("/ask", json=payload)
    body = r.json()
    assert r.status_code == 422 and body["error"] == "validation_error"
    assert body["request_id"] == r.headers["X-Request-ID"]
    assert "most recent creatinine" not in r.text          # inputs are not echoed back


@pytest.mark.parametrize("payload", [
    {"subject_id": 90000001, "query": "creatinine", "top_k": 0},
    {"subject_id": 90000001, "query": "creatinine", "top_k": 21},
    {"subject_id": 90000001, "query": "creatinine", "temporal_filter": "yesterday"},
    {"subject_id": -5, "query": "creatinine"},
])
def test_retrieve_validation_error(payload):
    r = client.post("/retrieve", json=payload)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


# --- /ready ------------------------------------------------------------------------------
def _ok_db():
    return {"database": "ok", "corpus": "ok"}


def _ok_llm():
    return {"ollama": "ok", "model": "ok"}


def test_ready_ok(monkeypatch):
    monkeypatch.setattr(api, "_check_database", _ok_db)
    monkeypatch.setattr(api, "_check_ollama", _ok_llm)
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["status"] == "ready"
    assert r.json()["data_plane"] == "demo" and r.json()["database"] == "lumen_demo"


def test_ready_database_failure_is_503_and_leaks_nothing(monkeypatch):
    def boom():
        raise OperationalError("SELECT 1", {}, Exception("password=hunter2 host=db.internal"))
    monkeypatch.setattr(api, "_check_database", boom)
    monkeypatch.setattr(api, "_check_ollama", _ok_llm)
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["dependencies"]["database"] == "unavailable"
    assert "hunter2" not in r.text and "db.internal" not in r.text


def test_ready_ollama_and_timeout_failures(monkeypatch):
    monkeypatch.setattr(api, "READY_TIMEOUT_S", 0.05)
    monkeypatch.setattr(api, "_check_database", lambda: time.sleep(0.5) or _ok_db())
    monkeypatch.setattr(api, "_check_ollama", lambda: {"ollama": "unavailable", "model": "unknown"})
    deps = client.get("/ready").json()["dependencies"]
    assert deps["database"] == "timeout" and deps["ollama"] == "unavailable"


# --- /ask mapping (graph mocked) ------------------------------------------------------------
STATE = {"query_type": "lab_trend", "temporal_mode": "latest", "review_status": "auto_approved",
         "draft_answer": "Creatinine 1.4 mg/dL [S1].", "final_answer": "Creatinine 1.4 mg/dL [S1].",
         "citations": [{"claim": "Creatinine 1.4 mg/dL [S1].", "label": "S1", "chunk_id": 7, "verified": True,
                        "verification_note": "supported"}],
         "patient_evidence": [{"label": "S1", "chunk_id": 7, "source_type": "note", "note_type": "discharge",
                               "charttime": "2024-03-18 10:40:00", "text": "SYNTHETIC", "score": 0.9}],
         "node_trail": ["triage", "patient_retrieval", "synthesis", "verification", "finalize"], "errors": []}


def test_ask_completed(monkeypatch):
    _no_subject_check(monkeypatch)
    monkeypatch.setattr(api, "_run_ask", lambda q, s, t, rid: ({}, STATE))
    body = client.post("/ask", json={"subject_id": 90000001, "query": "most recent creatinine?"}).json()
    assert body["status"] == "completed" and not body["needs_human_review"] and not body["answer_is_draft"]
    assert body["citations"][0]["label"] == "S1" and body["sources"][0]["chunk_id"] == 7
    assert "text" not in body["sources"][0] and body["thread_id"] == f"api-{body['request_id']}"


def test_ask_human_review_is_not_an_error(monkeypatch):
    _no_subject_check(monkeypatch)
    paused = {**STATE, "review_status": "pending", "final_answer": "",
              "citations": [{**STATE["citations"][0], "verified": False}]}
    out = {"__interrupt__": [SimpleNamespace(value={"flagged": [{"index": 0}]})]}
    monkeypatch.setattr(api, "_run_ask", lambda q, s, t, rid: (out, paused))
    r = client.post("/ask", json={"subject_id": 90000001, "query": "why short of breath?"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "human_review_required"
    assert body["needs_human_review"] and body["answer_is_draft"] and body["flagged_claims"] == 1


def test_ask_subject_not_found(monkeypatch):
    def missing(sid):
        raise api.SubjectNotFound(sid)
    monkeypatch.setattr(api, "_ensure_subject", missing)
    r = client.post("/ask", json={"subject_id": 12345, "query": "anything"})
    assert r.status_code == 404 and r.json()["error"] == "subject_not_found"


def test_ask_model_unavailable(monkeypatch):
    _no_subject_check(monkeypatch)
    failed = {**STATE, "review_status": "failed", "draft_answer": "", "final_answer": "", "citations": [],
              "errors": ["synthesis: local LLM call failed after 4 attempts: connection refused"]}
    monkeypatch.setattr(api, "_run_ask", lambda q, s, t, rid: ({}, failed))
    r = client.post("/ask", json={"subject_id": 90000001, "query": "anything"})
    assert r.status_code == 503 and r.json()["error"] == "model_unavailable"


def test_ask_database_error(monkeypatch):
    def down(sid):
        raise OperationalError("SELECT", {}, Exception("password=hunter2"))
    monkeypatch.setattr(api, "_ensure_subject", down)
    r = client.post("/ask", json={"subject_id": 90000001, "query": "anything"})
    assert r.status_code == 503 and r.json()["error"] == "database_unavailable" and "hunter2" not in r.text


def test_internal_error_is_generic(monkeypatch):
    _no_subject_check(monkeypatch)
    def crash(*a):
        raise RuntimeError("patient note text that must not leak")
    monkeypatch.setattr(api, "_run_ask", crash)
    r = client.post("/ask", json={"subject_id": 90000001, "query": "anything"})
    assert r.status_code == 500 and r.json() == {"error": "internal_error", "request_id": r.headers["X-Request-ID"]}


# --- /retrieve mapping (retriever mocked) ----------------------------------------------------
def test_retrieve_maps_results(monkeypatch):
    _no_subject_check(monkeypatch)
    res = [SimpleNamespace(chunk_id=11, note_id=92000003, note_type="discharge", charttime="2024-03-18 10:40:00",
                           final_score=0.91234, sources=["bm25", "vector", "both"], chunk_text="SYNTHETIC chunk")]
    monkeypatch.setattr(api, "_run_retrieve", lambda q, s, tf, k: ("latest", res))
    body = client.post("/retrieve", json={"subject_id": 90000001, "query": "most recent creatinine"}).json()
    assert body["results"][0] == {"rank": 1, "chunk_id": 11, "note_id": 92000003, "note_type": "discharge",
                                  "charttime": "2024-03-18 10:40:00", "score": 0.9123, "sources": ["bm25", "vector"],
                                  "text": "SYNTHETIC chunk"}
    assert body["temporal_mode"] == "latest" and body["data_plane"] == "demo"


# --- data plane --------------------------------------------------------------------------------
def test_research_plane_refuses_non_loopback_clients(monkeypatch):
    monkeypatch.setattr(api, "DATA_PLANE", "research")
    r = client.get("/health")                      # TestClient's client host is "testclient"
    assert r.status_code == 403 and r.json()["error"] == "forbidden"
