"""Ingestion/index run-state and provenance tests require no database/models."""

import re

import pytest

from src.retrieval import index_notes
from src.retrieval import index_provenance as provenance
from src.retrieval.chunker import ClinicalNoteChunker
from src.storage import ingest


def test_index_provenance_is_stable_and_tracks_output_configuration():
    config = provenance.index_configuration()
    assert config["embedding"]["vector_dimension"] == 768
    assert len(config["embedding"]["revision"]) == 40
    assert provenance.configuration_hash() == provenance.configuration_hash(config)
    changed = {**config, "chunker": {**config["chunker"], "max_tokens": 385}}
    assert provenance.configuration_hash(changed) != provenance.configuration_hash(config)


def test_canonical_chunker_completely_handles_106467_character_note():
    facts = "\n".join(
        f"- Date=2025-01-01 | Code=SENTINEL{i:04d} | Description=Source fact {i}"
        for i in range(700)
    )
    prefix = f"Encounter Summary\n\nEncounter:\nType: ambulatory\n\nObservations:\n{facts}\n"
    padding = ("deterministic-padding " * 10_000)[:106_467 - len(prefix)]
    text = prefix + padding
    assert len(text) == 106_467

    chunker = ClinicalNoteChunker(**provenance.CHUNKER_CONFIG)
    first = chunker.chunk_text(text, note_type="encounter_summary")
    second = chunker.chunk_text(text, note_type="encounter_summary")
    assert first == second
    assert [chunk.chunk_index for chunk in first] == list(range(len(first)))
    assert max(chunk.token_count for chunk in first) <= provenance.CHUNKER_CONFIG["max_tokens"]
    output = " ".join(chunk.text for chunk in first)
    assert set(re.findall(r"SENTINEL\d{4}", text)) <= set(
        re.findall(r"SENTINEL\d{4}", output)
    )


def test_index_run_records_completion(monkeypatch):
    events = []
    monkeypatch.setattr(index_notes, "_legacy_index_unadopted", lambda: False)   # no database in unit tests
    monkeypatch.setattr(index_notes, "_start_index_run", lambda *a: events.append("running"))
    monkeypatch.setattr(index_notes, "_run_indexing_steps",
                        lambda *a, **k: {"notes": 2, "chunks": 4})
    monkeypatch.setattr(index_notes, "_finish_index_run",
                        lambda _id, status, error_type=None: events.append((status, error_type)))
    assert index_notes.run_indexing() == {"notes": 2, "chunks": 4}
    assert events == ["running", ("completed", None)]


def test_index_run_records_failure(monkeypatch):
    events = []
    monkeypatch.setattr(index_notes, "_legacy_index_unadopted", lambda: False)   # no database in unit tests
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


# --- legacy adoption ---------------------------------------------------------
_FACTS = {"eligible_notes": 10, "eligible_notes_without_chunks": 0, "chunks": 40, "notes_with_chunk_gaps": 0,
          "chunks_without_embedding": 0, "vector_dimensions": [768],
          "duplicate_chunk_positions": 0, "max_token_count": 572,
          "chunks_over_current_max_tokens": 3, "existing_state_rows": 0}


def test_legacy_index_is_never_recorded_under_the_current_configuration_hash():
    from src.storage import adopt_legacy_index as adopt
    configuration = adopt.legacy_configuration(_FACTS)
    assert configuration["provenance"] == provenance.LEGACY_ADOPTED
    assert configuration["chunker"]["matches_current"] is False
    assert provenance.configuration_hash(configuration) != provenance.configuration_hash()


@pytest.mark.parametrize("change,fragment", [
    ({"eligible_notes_without_chunks": 2}, "no chunks"),
    ({"chunks_without_embedding": 1}, "no embedding"),
    ({"vector_dimensions": [384]}, "vector dimensions"),
    ({"duplicate_chunk_positions": 1}, "duplicate"),
    ({"eligible_notes": 0}, "no eligible notes"),
])
def test_an_incomplete_index_is_not_adopted(change, fragment):
    from src.storage import adopt_legacy_index as adopt
    assert adopt.problems(_FACTS) == []
    assert any(fragment in p for p in adopt.problems({**_FACTS, **change}))


def test_legacy_adoption_refuses_outside_the_research_plane(capsys):
    from src.storage import adopt_legacy_index as adopt
    assert adopt.storage.DATA_PLANE == "demo"          # the test environment
    assert adopt.main(["--apply", "--confirm-legacy-adoption"]) == 2
    assert "research plane" in capsys.readouterr().err


class _HashConn:
    """Stands in for a connection: returns the adopted hashes for any query."""
    def __init__(self, hashes): self.hashes, self.params = hashes, None
    def execute(self, statement, params=None):
        self.params = params
        return self
    def scalars(self): return iter(self.hashes)


class _AdoptConn:
    """Answers adopt()'s queries: any prior adoption hash, then insert rowcounts."""
    def __init__(self, prior=None, rowcount=7):
        self.prior, self.rowcount, self.calls = prior, rowcount, []
    def execute(self, statement, params=None):
        self.calls.append(params)
        return self
    def scalar(self): return self.prior
    @property
    def rowcount_(self): return self.rowcount


def _adopt(monkeypatch, conn, current):
    from src.storage import adopt_legacy_index as adopt
    monkeypatch.setattr(adopt, "legacy_adopted_hashes", lambda c: current)
    return adopt


def test_adoption_is_idempotent_once_a_legacy_run_exists(monkeypatch):
    conn = _AdoptConn(prior="legacyhash")
    adopt = _adopt(monkeypatch, conn, ["legacyhash"])
    assert adopt.adopt(conn, _FACTS) == {"status": "already_adopted", "config_hash": "legacyhash",
                                         "notes_recorded": 0}


def test_an_adoption_made_under_another_embedding_pin_is_not_repeated(monkeypatch):
    """After the pin changes the old adoption no longer counts; adopting again
    would stamp the new pin on vectors it never produced."""
    conn = _AdoptConn(prior="oldpinhash")
    adopt = _adopt(monkeypatch, conn, [])
    assert adopt.adopt(conn, _FACTS)["status"] == "stale_adoption_requires_reindex"
    assert len(conn.calls) == 1                         # nothing was written


def test_a_partly_tracked_index_is_not_adopted(monkeypatch):
    conn = _AdoptConn(prior=None)
    adopt = _adopt(monkeypatch, conn, [])
    assert adopt.adopt(conn, {**_FACTS, "existing_state_rows": 3})["status"] == "refused_partial_state"
    assert len(conn.calls) == 1


def test_a_fresh_adoption_records_the_rows_it_inserted(monkeypatch):
    import json as _json
    conn = _AdoptConn(prior=None)
    conn.rowcount = 7
    adopt = _adopt(monkeypatch, conn, [])
    result = adopt.adopt(conn, _FACTS)
    assert result["status"] == "adopted" and result["notes_recorded"] == 7
    run = conn.calls[-1]
    assert run["recorded"] == 7 and run["expected"] == _FACTS["eligible_notes"]
    assert _json.loads(run["configuration"])["provenance"] == provenance.LEGACY_ADOPTED


def test_adopted_hashes_are_only_accepted_while_the_embedding_pin_matches():
    conn = _HashConn([])
    provenance.legacy_adopted_hashes(conn)
    embedding = provenance.index_configuration()["embedding"]
    assert conn.params == {"provenance": provenance.LEGACY_ADOPTED,
                           "repo": embedding["repo"], "revision": embedding["revision"]}


def test_indexer_treats_adopted_notes_as_indexed(monkeypatch):
    seen = {}

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, stmt, params=None):
            seen["params"] = params
            return []

    monkeypatch.setattr(index_notes, "legacy_adopted_hashes", lambda conn: ["legacyhash"])
    monkeypatch.setattr(index_notes, "engine", type("E", (), {"connect": lambda self: _Conn()})())
    assert index_notes.fetch_unindexed_note_ids("current", note_type="DS") == []
    assert seen["params"] == {"note_type": "DS", "config_hashes": ["current", "legacyhash"]}


def test_indexer_refuses_to_re_embed_an_index_that_was_never_adopted(monkeypatch):
    """A legacy database after the schema upgrade has chunks but no state rows:
    without this guard 'research index' deletes and re-embeds every note."""
    monkeypatch.setattr(index_notes, "_legacy_index_unadopted", lambda: True)
    started = []
    monkeypatch.setattr(index_notes, "_start_index_run", lambda *a: started.append(a))
    with pytest.raises(RuntimeError, match="adopt_legacy_index"):
        index_notes.run_indexing()
    assert started == []


def test_adopt_main_never_writes_when_unconfirmed_or_incomplete(monkeypatch):
    from src.storage import adopt_legacy_index as adopt

    class _Begin:
        def __enter__(self): return object()
        def __exit__(self, *a): return False

    monkeypatch.setattr(adopt.storage, "engine", type("E", (), {"begin": lambda self: _Begin()})())
    monkeypatch.setattr(adopt.storage, "DATA_PLANE", "research")
    called = []
    monkeypatch.setattr(adopt, "adopt", lambda conn, facts: called.append(1) or
                        {"status": "adopted", "config_hash": "x" * 64, "notes_recorded": 0})
    monkeypatch.setattr(adopt, "inspect", lambda conn: {**_FACTS, "chunks_without_embedding": 1})
    assert adopt.main(["--apply"]) == 2
    assert adopt.main(["--apply", "--confirm-legacy-adoption"]) == 1
    assert called == []
    monkeypatch.setattr(adopt, "inspect", lambda conn: dict(_FACTS))
    assert adopt.main(["--apply", "--confirm-legacy-adoption"]) == 0 and called == [1]


def test_a_note_with_a_gap_in_its_chunk_positions_is_not_adopted():
    from src.storage import adopt_legacy_index as adopt
    assert any("chunk positions" in p for p in adopt.problems({**_FACTS, "notes_with_chunk_gaps": 2}))
