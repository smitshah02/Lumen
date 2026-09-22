"""Performance evaluation evidence: deterministic human-review routing + source provenance.

The routing test compiles the PRODUCTION graph (src.agents.graph.build_graph)
with an in-memory checkpointer instead of Postgres; no model or LLM is loaded
(the human_review / finalize nodes make no LLM calls)."""

import sys
import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import performance_eval  # noqa: E402


@pytest.fixture
def production_graph(monkeypatch):
    import src.agents.graph as G

    class _Mem(MemorySaver):
        def __init__(self, pool):
            super().__init__()

        def setup(self):
            pass

    class _Pool:
        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    monkeypatch.setattr(G, "ConnectionPool", _Pool)
    monkeypatch.setattr(G, "PostgresSaver", _Mem)
    graph, _ = G.build_graph(validate_checkpoints=False)
    return graph


def test_review_routing_check_passes_on_production_graph(production_graph):
    r = performance_eval.review_routing_check(production_graph)
    assert r["result"] == "PASS", r
    c = r["cases"]
    assert c["unsupported_claim"]["route"] == "human_review" and c["unsupported_claim"]["paused"]
    assert c["unsupported_claim"]["api_status_equivalent"] == "human_review_required"
    assert c["unsupported_claim"]["flagged"] == 1
    assert c["all_verified"]["route"] == "finalize" and not c["all_verified"]["paused"]
    assert c["synthesis_failed"]["review_status"] == "failed" and not c["synthesis_failed"]["paused"]


def test_review_routing_check_detects_a_broken_route(production_graph, monkeypatch):
    import src.agents.graph as G
    monkeypatch.setattr(G, "route_after_verification", lambda state: "finalize")   # simulated regression
    assert performance_eval.review_routing_check(production_graph)["result"] == "FAIL"


SHA = "a" * 40


@pytest.mark.parametrize("content, expected", [
    (None, {"git_sha": None, "branch": None, "dirty_worktree": "unknown", "provenance_source": "missing"}),
    ("{not json", {"git_sha": None, "dirty_worktree": "unknown", "provenance_source": "unreadable"}),
    (json.dumps({"git_sha": SHA, "branch": "main", "dirty_worktree": False, "generated_at": "2026-09-19T00:00:00+00:00"}),
     {"git_sha": SHA, "branch": "main", "dirty_worktree": False, "provenance_source": "sync_metadata"}),
    (json.dumps({"git_sha": SHA, "branch": None, "dirty_worktree": True}),
     {"git_sha": SHA, "branch": None, "dirty_worktree": True}),
    (json.dumps({"git_sha": "abc123", "branch": "main", "dirty_worktree": "yes"}),      # never trust malformed values
     {"git_sha": None, "dirty_worktree": "unknown", "provenance_source": "sync_metadata"}),
])
def test_read_provenance(tmp_path, content, expected):
    f = tmp_path / ".deployment_source.json"
    if content is not None:
        f.write_text(content)
    got = performance_eval.read_provenance(f)
    for k, v in expected.items():
        assert got[k] == v, (k, got)
