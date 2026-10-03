"""Human-in-the-loop: the real graph, really interrupted and really resumed.

Retrieval and the model are faked; the graph wiring, `interrupt()`, the
checkpointer and `Command(resume=...)` are LangGraph's own, against an
in-memory saver instead of Postgres.
"""
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

import src.agents.graph as g
import src.api.app as api
from src.agents import review

QUESTION = "What medications is the patient taking?"
EVIDENCE = [{"chunk_id": 7, "source_type": "note", "note_type": "discharge", "charttime": "2150-01-02",
             "score": 0.9, "label": "S1", "text": "Discharge medications: metformin 500 mg twice daily."}]
UNSUPPORTED = "The patient takes warfarin 5 mg daily [S1]."
SUPPORTED = "The patient takes metformin 500 mg twice daily [S1]."


@pytest.fixture
def graph(monkeypatch):
    """The production wiring, compiled on an in-memory checkpointer. `graph.draft`
    is what the fake model writes; the verifier never returns a verdict, so a
    claim the deterministic check cannot anchor stays unsupported."""
    monkeypatch.setattr(g, "patient_retrieval",
                        lambda state: {"patient_evidence": EVIDENCE, "node_trail": ["patient_retrieval"]})
    box = {"draft": UNSUPPORTED, "calls": []}

    def chat_for(role, messages, **kw):
        box["calls"].append(role)
        return "{}" if role == "verify" else box["draft"]

    monkeypatch.setattr(g, "chat_for", chat_for)
    compiled = g._builder().compile(checkpointer=InMemorySaver())
    compiled.box = box
    return compiled


def _ask(graph, thread):
    return graph.invoke({"query": QUESTION, "subject_id": 1, "thread_id": thread},
                        config={"configurable": {"thread_id": thread}})


def test_unsupported_claim_interrupts_and_the_state_is_checkpointed(graph):
    out = _ask(graph, "t1")
    assert "__interrupt__" in out                                   # the graph actually stopped
    snap = graph.get_state({"configurable": {"thread_id": "t1"}})
    assert snap.next == ("human_review",) and snap.values["review_status"] == "pending"
    assert snap.values["needs_human_review"] is True and not snap.values.get("final_answer")
    waiting = review.pending(graph, "t1")
    assert waiting["draft_answer"] == UNSUPPORTED and waiting["subject_id"] == 1
    assert [f["claim"] for f in waiting["flagged"]] == [UNSUPPORTED]


def test_approve_resumes_from_the_checkpoint_and_finalizes(graph):
    _ask(graph, "t2")
    calls_before = list(graph.box["calls"])
    st = review.submit(graph, "t2", "approve", "confirmed against the chart")
    assert st["review_status"] == "reviewed" and st["final_answer"] == UNSUPPORTED
    assert st["needs_human_review"] is False and st["citations"][0]["verified"] is True
    assert st["human_decisions"] == [{"index": 0, "action": "approve", "note": "confirmed against the chart"}]
    assert st["node_trail"][-2:] == ["human_review", "finalize"]
    assert st["node_trail"].count("synthesis") == 1                 # resumed, not re-run
    assert graph.box["calls"] == calls_before                       # and no new model call
    assert graph.get_state({"configurable": {"thread_id": "t2"}}).next == ()


def test_reject_resumes_and_releases_nothing(graph):
    _ask(graph, "t3")
    st = review.submit(graph, "t3", "reject", "wrong drug")
    assert st["review_status"] == "rejected" and st["final_answer"] == g.REJECTED_ANSWER
    assert "warfarin" not in st["final_answer"]
    assert st["citations"][0]["verified"] is False
    assert st["human_decisions"] == [{"index": 0, "action": "reject", "note": "wrong drug"}]
    assert st["node_trail"][-2:] == ["human_review", "finalize"]
    assert st["draft_answer"] == UNSUPPORTED                        # the audit trail keeps the draft


def test_invalid_decision_is_refused_and_the_run_stays_paused(graph):
    _ask(graph, "t4")
    for bad in ("", "APPROVE", "strike", "yes"):
        with pytest.raises(ValueError):
            review.submit(graph, "t4", bad)
    assert review.pending(graph, "t4")["review_status"] == "pending"


def test_unknown_thread_is_not_found(graph):
    with pytest.raises(review.ReviewNotFound):
        review.pending(graph, "never-ran")
    with pytest.raises(review.ReviewNotFound):
        review.submit(graph, "never-ran", "approve")


@pytest.mark.parametrize("first,second", [("approve", "reject"), ("reject", "approve"), ("approve", "approve")])
def test_a_decided_review_cannot_be_submitted_again(graph, first, second):
    _ask(graph, "t5")
    decided = review.submit(graph, "t5", first)
    with pytest.raises(review.ReviewNotPending) as e:
        review.submit(graph, "t5", second)
    assert e.value.review_status == decided["review_status"]
    after = graph.get_state({"configurable": {"thread_id": "t5"}}).values
    assert after["final_answer"] == decided["final_answer"] and after["node_trail"] == decided["node_trail"]
    assert len(after["human_decisions"]) == 1                       # nothing ran twice


def test_a_supported_answer_never_pauses(graph):
    graph.box["draft"] = SUPPORTED
    out = _ask(graph, "t6")
    assert "__interrupt__" not in out and out["review_status"] == "auto_approved"
    assert out["final_answer"] == SUPPORTED and "human_review" not in out["node_trail"]
    with pytest.raises(review.ReviewNotPending):
        review.submit(graph, "t6", "approve")


def test_per_claim_resume_still_works_for_the_cli(graph):
    _ask(graph, "t7")
    st = review.resume(graph, "t7", [{"action": "strike", "note": "not in chart"}])
    # every claim struck: nothing is released (finalize used to put the draft back)
    assert st["review_status"] == "reviewed" and st["final_answer"] == ""
    assert st["node_trail"][-1] == "finalize"


# --- API ---------------------------------------------------------------------
@pytest.fixture
def client(graph, monkeypatch):
    monkeypatch.setattr(api, "_get_graph", lambda: graph)
    return TestClient(api.app)


def test_api_pending_then_approve(graph, client):
    _ask(graph, "api-a")
    waiting = client.get("/review/api-a")
    assert waiting.status_code == 200 and waiting.json()["review_status"] == "pending"
    assert len(waiting.json()["flagged"]) == 1
    r = client.post("/review/api-a", json={"decision": "approve", "reviewer_note": "ok"})
    assert r.status_code == 200
    assert r.json() | {"human_decisions": None, "node_trail": None} == {
        "thread_id": "api-a", "status": "completed", "decision": "approve", "review_status": "reviewed",
        "answer": UNSUPPORTED, "needs_human_review": False, "human_decisions": None, "node_trail": None}
    assert client.get("/review/api-a").status_code == 409            # no longer pending


def test_api_reject(graph, client):
    _ask(graph, "api-b")
    r = client.post("/review/api-b", json={"decision": "reject"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    assert r.json()["review_status"] == "rejected" and r.json()["answer"] == g.REJECTED_ANSWER


@pytest.mark.parametrize("body", [{}, {"decision": "maybe"}, {"decision": "approve", "answer": "edited"},
                                  {"decision": "approve", "reviewer_note": "x" * 1001}])
def test_api_bad_request_approves_nothing(graph, client, body):
    _ask(graph, "api-c")
    assert client.post("/review/api-c", json=body).status_code == 422
    assert client.get("/review/api-c").json()["review_status"] == "pending"


def test_api_unknown_and_repeated_submissions_fail_safely(graph, client):
    assert client.get("/review/api-missing").status_code == 404
    assert client.post("/review/api-missing", json={"decision": "approve"}).status_code == 404
    assert client.post("/review/bad%20id!", json={"decision": "approve"}).status_code == 422
    _ask(graph, "api-d")
    assert client.post("/review/api-d", json={"decision": "reject"}).status_code == 200
    again = client.post("/review/api-d", json={"decision": "approve"})
    assert again.status_code == 409 and again.json()["error"] == "review_not_pending"
    assert again.json()["detail"] == {"review_status": "rejected"}


# --- one continuous CLI interaction ---------------------------------------------
def _answers(*replies):
    it = iter(replies)
    return lambda prompt="": next(it)


@pytest.mark.parametrize("typed,status,answer", [
    (("approve", "looks right"), "reviewed", UNSUPPORTED),
    (("pass", ""), "reviewed", UNSUPPORTED),
    (("nonsense", "reject", "wrong drug"), "rejected", g.REJECTED_ANSWER),   # re-asked after bad input
    (("s", ""), "reviewed", ""),
])
def test_cli_prompts_and_resumes_the_same_run(graph, capsys, typed, status, answer):
    from src.agents.review_cli import review_interactively
    _ask(graph, "cli-1")
    calls = list(graph.box["calls"])
    st = review_interactively(graph, "cli-1", ask=_answers(*typed))
    shown = capsys.readouterr().out
    assert "HUMAN REVIEW REQUIRED" in shown and UNSUPPORTED in shown and "Resuming workflow" in shown
    assert EVIDENCE[0]["text"] in shown                              # the cited source is on screen
    assert st["review_status"] == status and st["final_answer"] == answer
    assert st["thread_id"] == "cli-1" and st["node_trail"][-2:] == ["human_review", "finalize"]
    assert st["node_trail"].count("patient_retrieval") == 1 and graph.box["calls"] == calls   # nothing re-run


def test_run_graph_main_reviews_inline_and_prints_the_final_answer(graph, monkeypatch, capsys):
    import sys
    import src.agents.run_graph as rg
    monkeypatch.setattr(rg, "build_graph", lambda: (graph, None))
    monkeypatch.setattr(sys, "argv", ["run_graph", "--query", QUESTION, "--subject", "1", "--thread", "cli-main"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", _answers("approve", "ok"))
    assert rg.main() == 0
    shown = capsys.readouterr().out
    assert "PAUSED for human review" in shown and "Resuming workflow" in shown
    assert "review       reviewed" in shown and "human_review -> finalize" in shown
    assert graph.get_state({"configurable": {"thread_id": "cli-main"}}).next == ()


def test_run_graph_main_does_not_prompt_without_a_terminal(graph, monkeypatch, capsys):
    import sys
    import src.agents.run_graph as rg
    monkeypatch.setattr(rg, "build_graph", lambda: (graph, None))
    monkeypatch.setattr(sys, "argv", ["run_graph", "--query", QUESTION, "--subject", "1", "--thread", "cli-pipe"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("prompted without a terminal"))
    assert rg.main() == 0 and "resume with" in capsys.readouterr().out
    assert review.pending(graph, "cli-pipe")["review_status"] == "pending"   # still safely paused


@pytest.mark.parametrize("typed,shown", [(("approve", ""), "verified     1/1"), (("reject", ""), "verified     0/1")])
def test_cli_verified_count_is_never_negative_on_the_refusal_guard_path(graph, monkeypatch, capsys, typed, shown):
    """A refusal after a declined structured lookup is unsupported with no model
    check: checked=0, unsupported=1, which the CLI printed as "verified -1/0"."""
    import sys
    import src.agents.run_graph as rg
    refusal = "The available records do not contain enough information to answer this."
    monkeypatch.setattr(g, "patient_retrieval", lambda state: {
        "patient_evidence": EVIDENCE, "structured_rows": 11, "node_trail": ["patient_retrieval"]})
    graph = g._builder().compile(checkpointer=InMemorySaver())
    monkeypatch.setattr(g, "chat_for", lambda role, messages, **kw: refusal)
    monkeypatch.setattr(rg, "build_graph", lambda: (graph, None))
    monkeypatch.setattr(sys, "argv", ["run_graph", "--query", QUESTION, "--subject", "1", "--thread", "cli-neg"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", _answers(*typed))
    assert rg.main() == 0
    out = capsys.readouterr().out
    assert "structured lookup found 11 row(s)" in out and shown in out and "-1/" not in out
