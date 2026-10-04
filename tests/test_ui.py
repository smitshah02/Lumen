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
ALL_JS = JS + (UI / "review.js").read_text()
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
    for name in ("index.html", "app.js", "review.js", "style.css"):
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
        assert field in ask and re.search(rf"\b{field}\b", ALL_JS), field
    for field in ("label", "claim", "verified"):
        assert field in Citation.model_fields and field in JS
    for field in ("label", "source_type", "note_type", "charttime", "hadm_id", "note_id", "chunk_id"):
        assert field in Source.model_fields and field in JS
    assert "text" not in Source.model_fields                         # /ask returns no note text, and the UI asks for none


def test_ui_requests_match_the_request_schemas():
    assert set(AskRequest.model_fields) == {"subject_id", "query"}
    assert "{ subject_id: msg.subject, query }" in JS
    assert set(ReviewDecision.model_fields) == {"decision", "reviewer_note"}
    assert "{ decision, reviewer_note:" in JS and '"approve"' in ALL_JS and '"reject"' in ALL_JS
    assert '"/review/" + encodeURIComponent(msg.resp.thread_id)' in JS
    assert JS.count('api("POST", "/ask"') == 1                       # a review never re-asks the question


@pytest.mark.parametrize("error", ["subject_not_found", "model_unavailable", "database_unavailable",
                                   "generation_failed", "validation_error", "review_not_found",
                                   "review_not_pending", "internal_error", "forbidden"])
def test_every_api_error_code_has_a_message(error):
    assert re.search(rf"\b{error}:", JS), error


# --- approving a flagged draft is a two-step, explicit override --------------------
REVIEW_JS = UI / "review.js"
NODE = __import__("shutil").which("node")


def _review(script: str) -> dict:
    """Run a snippet against review.js under Node and return what it prints as JSON."""
    import json
    import subprocess
    code = f"const R = require({json.dumps(str(REVIEW_JS))});\n{script}"
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _paused(n):
    return f'{{resp: {{status: "human_review_required", flagged_claims: {n}}}, pending: {{flagged: new Array({n}).fill({{}})}}}}'


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("n,warning,question", [
    (1, "This draft contains 1 flagged claim. Approving will release it unchanged.",
     "Release this draft despite 1 verifier flag?"),
    (2, "This draft contains 2 flagged claims. Approving will release them unchanged.",
     "Release this draft despite 2 verifier flags?"),
])
def test_flagged_draft_needs_an_explicit_confirmation(n, warning, question):
    r = _review(f"""
        const m = {_paused(n)}; const sent = [];
        const first = R.view(m);
        sent.push(R.act(m, "approve"));                      // first click
        const asking = R.view(m);
        sent.push(R.act(m, "cancel"));                       // back out
        const after_cancel = R.view(m);
        sent.push(R.act(m, "approve")); sent.push(R.act(m, "confirm"));
        console.log(JSON.stringify({{first, asking, after_cancel, sent, overridden: m.overridden, note: R.overrideNote(m)}}));
    """)
    assert [b["label"] for b in r["first"]["buttons"]] == ["Approve draft as-is", "Reject"]
    assert r["first"]["warning"] == warning and r["first"]["count"] == n
    assert [b["label"] for b in r["asking"]["buttons"]] == ["Confirm approval", "Cancel"]
    assert r["asking"]["question"] == question
    assert [b["label"] for b in r["after_cancel"]["buttons"]] == ["Approve draft as-is", "Reject"]
    assert r["sent"] == [None, None, None, "approve"]         # exactly one submission, on the confirm
    assert r["overridden"] == n and "approved as-is" in r["note"] and "judgement is unchanged" in r["note"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_reject_is_one_click_and_closed_or_busy_reviews_send_nothing():
    r = _review(f"""
        const a = {_paused(2)}; const b = {_paused(2)}; b.deciding = true;
        const c = {_paused(2)}; c.closed = true; const d = {_paused(2)};
        console.log(JSON.stringify({{reject: R.act(a, "reject"), busy: R.act(b, "reject"), closed: R.act(c, "approve"),
                                     confirm_without_asking: R.act(d, "confirm"), closed_view: R.view(c)}}));
    """)
    assert r == {"reject": "reject", "busy": None, "closed": None, "confirm_without_asking": None,
                 "closed_view": {"open": False, "buttons": []}}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_answers_that_need_no_review_get_no_controls():
    r = _review("""
        const done = {resp: {status: "completed", flagged_claims: 0}};
        const unflagged = {resp: {status: "human_review_required", flagged_claims: 0}, pending: {flagged: []}};
        console.log(JSON.stringify({done: R.view(done), act: R.act(done, "approve"),
                                    labels: R.view(unflagged).buttons.map(b => b.label), direct: R.act(unflagged, "approve")}));
    """)
    assert r["done"] == {"open": False, "buttons": []} and r["act"] is None
    assert r["labels"] == ["Approve", "Reject"] and r["direct"] == "approve"


def test_the_page_uses_the_review_logic_and_posts_only_through_decide():
    assert HTML.index('src="review.js"') < HTML.index('src="app.js"')
    assert "LumenReview.act(msg, spec.action)" in JS and "if (decision) decide(msg, decision)" in JS
    assert JS.count('api("POST", "/review/"') == 1 and "alert(" not in JS and "confirm(" not in JS
    assert client.get("/ui/review.js").status_code == 200
