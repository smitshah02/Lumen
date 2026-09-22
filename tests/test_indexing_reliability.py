"""Ingestion/index run-state and provenance tests require no database/models."""

import pytest

from src.retrieval import index_notes
from src.retrieval import index_provenance as provenance
from src.storage import ingest


def test_index_provenance_is_stable_and_tracks_output_configuration():
    config = provenance.index_configuration()
    assert config["embedding"]["vector_dimension"] == 768
    assert len(config["embedding"]["revision"]) == 40
    assert provenance.configuration_hash() == provenance.configuration_hash(config)
    changed = {**config, "chunker": {**config["chunker"], "max_tokens": 385}}
    assert provenance.configuration_hash(changed) != provenance.configuration_hash(config)


def test_index_run_records_completion(monkeypatch):
    events = []
    monkeypatch.setattr(index_notes, "_start_index_run", lambda *a: events.append("running"))
    monkeypatch.setattr(index_notes, "_run_indexing_steps",
                        lambda *a, **k: {"notes": 2, "chunks": 4})
    monkeypatch.setattr(index_notes, "_finish_index_run",
                        lambda _id, status, error_type=None: events.append((status, error_type)))
    assert index_notes.run_indexing() == {"notes": 2, "chunks": 4}
    assert events == ["running", ("completed", None)]


def test_index_run_records_failure(monkeypatch):
    events = []
    monkeypatch.setattr(index_notes, "_start_index_run", lambda *a: events.append("running"))
    monkeypatch.setattr(index_notes, "_run_indexing_steps",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(index_notes, "_finish_index_run",
                        lambda _id, status, error_type=None: events.append((status, error_type)))
    with pytest.raises(RuntimeError):
        index_notes.run_indexing()
    assert events == ["running", ("failed", "RuntimeError")]


def test_ingestion_run_records_failure_without_error_content(monkeypatch):
    events = []
    monkeypatch.setattr(ingest, "create_schema", lambda: None)
    monkeypatch.setattr(ingest, "_start_ingestion_run", lambda *a: events.append("running"))
    monkeypatch.setattr(ingest, "_run_ingestion_steps",
                        lambda **k: (_ for _ in ()).throw(ValueError("sensitive row")))
    monkeypatch.setattr(ingest, "_finish_ingestion_run",
                        lambda _id, status, error_type=None: events.append((status, error_type)))
    with pytest.raises(ValueError):
        ingest.run_ingestion()
    assert events == ["running", ("failed", "ValueError")]
