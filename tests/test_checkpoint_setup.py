"""Checkpoint initialization is explicit; graph construction stays read-only."""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.agents import graph
from src.storage import checkpoints


class FakePool:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def close(self):
        self.closed = True


class FakeSaver(MemorySaver):
    setup_calls = 0

    def __init__(self, pool):
        super().__init__()
        self.pool = pool

    def setup(self):
        type(self).setup_calls += 1


def test_explicit_schema_pool_has_valid_bounds_and_closes(monkeypatch):
    seen = {}

    class RecordingPool(FakePool):
        def __init__(self, *args, **kwargs):
            super().__init__()
            seen.update(kwargs)

    class RecordingSaver:
        def __init__(self, pool):
            seen["pool"] = pool

        def setup(self):
            seen["setup"] = True

    monkeypatch.setattr(checkpoints, "ConnectionPool", RecordingPool)
    monkeypatch.setattr(checkpoints, "PostgresSaver", RecordingSaver)
    monkeypatch.setattr(checkpoints, "checkpoint_schema_status",
                        lambda: {"ready": True, "missing": []})

    checkpoints.initialize_checkpoint_schema()

    assert seen["min_size"] == 1
    assert seen["max_size"] == 2
    assert seen["min_size"] <= seen["max_size"]
    assert seen["setup"] is True
    assert seen["pool"].closed is True


def test_graph_refuses_missing_checkpoint_schema_without_mutating(monkeypatch):
    FakeSaver.setup_calls = 0
    monkeypatch.setattr(graph, "checkpoint_schema_status",
                        lambda: {"ready": False, "missing": ["checkpoints"]})
    monkeypatch.setattr(graph, "ConnectionPool", FakePool)
    monkeypatch.setattr(graph, "PostgresSaver", FakeSaver)
    with pytest.raises(RuntimeError, match="src.storage.checkpoints"):
        graph.build_graph()
    assert FakeSaver.setup_calls == 0


def test_explicit_defensive_setup_remains_available(monkeypatch):
    FakeSaver.setup_calls = 0
    monkeypatch.setattr(graph, "ConnectionPool", FakePool)
    monkeypatch.setattr(graph, "PostgresSaver", FakeSaver)
    compiled, saver = graph.build_graph(setup=True)
    assert compiled is not None and isinstance(saver, FakeSaver)
    assert FakeSaver.setup_calls == 1
    graph.close_pools()
