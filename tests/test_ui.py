"""The local UI: served by the API, self-contained, and in step with the API contract."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.api.app as api
from src.api.schemas import AskRequest, AskResponse, Citation, ReviewDecision, Source

client = TestClient(api.app)
UI = Path(api.UI_DIR)
JS = (UI / "app.js").read_text()
HTML = (UI / "index.html").read_text()


def test_ui_page_and_assets_are_served():
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (302, 307) and root.headers["location"] == "/ui/"
    page = client.get("/ui/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "LUMEN" in page.text and "Longitudinal Clinical RAG" in page.text
    assert client.get("/ui/app.js").status_code == 200
    assert "text/css" in client.get("/ui/style.css").headers["content-type"]
    assert client.get("/ui/../app.py").status_code == 404           # only the ui directory is exposed


def test_api_routes_are_not_shadowed_by_the_ui_mount():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/ask", json={}).status_code == 422


def test_ui_loads_nothing_from_outside_this_server():
    for name in ("index.html", "app.js", "style.css"):
        text = (UI / name).read_text()
        assert not re.search(r"https?://|//cdn|@import|fonts\.googleapis", text), name
    assert "script-src 'self'" in HTML and "connect-src 'self'" in HTML and "default-src 'none'" in HTML
    assert "<script>" not in HTML and "style=" not in HTML and "innerHTML" not in JS


def test_research_plane_serves_the_ui_to_loopback_only(monkeypatch):
    monkeypatch.setattr(api, "DATA_PLANE", "research")
    assert client.get("/ui/").status_code == 403                     # TestClient is not a loopback client


def test_ui_reads_only_fields_the_api_returns():
    """Every response field the page reads exists in the schema it reads it from."""
    ask = set(AskResponse.model_fields)
    for field in ("answer", "answer_is_draft", "status", "review_status", "citations", "sources", "flagged_claims",
                  "node_trail", "models", "latency_ms", "timings", "thread_id", "request_id", "query_type",
                  "temporal_mode", "needs_human_review"):
        assert field in ask and re.search(rf"\b{field}\b", JS), field
    for field in ("label", "claim", "verified"):
        assert field in Citation.model_fields and field in JS
    for field in ("label", "source_type", "note_type", "charttime", "hadm_id", "note_id", "chunk_id"):
        assert field in Source.model_fields and field in JS
    assert "text" not in Source.model_fields                         # /ask returns no note text, and the UI asks for none


def test_ui_requests_match_the_request_schemas():
    assert set(AskRequest.model_fields) == {"subject_id", "query"}
    assert "{ subject_id: msg.subject, query }" in JS
    assert set(ReviewDecision.model_fields) == {"decision", "reviewer_note"}
    assert "{ decision, reviewer_note:" in JS and '"approve"' in JS and '"reject"' in JS
    assert '"/review/" + encodeURIComponent(msg.resp.thread_id)' in JS
    assert JS.count('api("POST", "/ask"') == 1                       # a review never re-asks the question


@pytest.mark.parametrize("error", ["subject_not_found", "model_unavailable", "database_unavailable",
                                   "generation_failed", "validation_error", "review_not_found",
                                   "review_not_pending", "internal_error", "forbidden"])
def test_every_api_error_code_has_a_message(error):
    assert re.search(rf"\b{error}:", JS), error
