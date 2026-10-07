"""Human review of a paused run — the one mechanism the CLI and the API share.

A run that verification could not fully support stops inside the
`human_review` node on LangGraph's `interrupt()`, and the checkpointer keeps
the whole state under its thread_id. Nothing here rebuilds that state: a
decision is handed back with `Command(resume=...)` and the graph continues
from its own checkpoint.

    pending(graph, thread_id)                    what is waiting, or raise
    submit(graph, thread_id, "approve"|"reject") one decision for the whole draft
    resume(graph, thread_id, decisions)          per-claim decisions (the CLI)

A thread that does not exist raises ReviewNotFound; one that is not paused
(never needed review, or already decided) raises ReviewNotPending. The check
and the resume run under one lock, so a repeated submission cannot resume a
run twice.
"""

from __future__ import annotations

import threading

from langgraph.types import Command

DECISIONS = ("approve", "reject")
_lock = threading.Lock()


class ReviewNotFound(Exception):
    """No checkpoint exists for this thread."""


class ReviewNotPending(Exception):
    """The thread exists but is not paused at human review."""

    def __init__(self, review_status: str | None):
        super().__init__(review_status)
        self.review_status = review_status


def _config(thread_id: str) -> dict:
    from src.agents.run_graph import build_config
    return build_config(thread_id, tags=("lumen", "human-review"))


def _paused(graph, config: dict):
    snap = graph.get_state(config)
    if not snap.values:
        raise ReviewNotFound(config["configurable"]["thread_id"])
    interrupts = [i for t in snap.tasks for i in (t.interrupts or [])]
    if not snap.next or not interrupts:
        raise ReviewNotPending(snap.values.get("review_status"))
    return snap, interrupts[0].value


def pending(graph, thread_id: str) -> dict:
    """The draft and the flagged claims awaiting a decision."""
    snap, payload = _paused(graph, _config(thread_id))
    return {"thread_id": thread_id, "subject_id": snap.values.get("subject_id"),
            "review_status": "pending", "query": payload.get("query", ""),
            "draft_answer": payload.get("answer", ""), "flagged": payload.get("flagged", [])}


def resume(graph, thread_id: str, decisions: list[dict]) -> dict:
    """Resume a paused run with one decision per flagged claim. Returns the final state."""
    config = _config(thread_id)
    with _lock:
        _paused(graph, config)                       # raises unless it is paused right now
        graph.invoke(Command(resume=decisions), config=config)
        return dict(graph.get_state(config).values)


def submit(graph, thread_id: str, decision: str, note: str = "") -> dict:
    """Approve or reject the whole draft."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    config = _config(thread_id)
    with _lock:
        _, payload = _paused(graph, config)
        decisions = [{"action": decision, "note": note} for _ in payload.get("flagged", [])]
        graph.invoke(Command(resume=decisions), config=config)
        return dict(graph.get_state(config).values)
