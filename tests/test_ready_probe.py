"""The /ready database probe (Stage 4, pre-switch).

The probe runs on every /ready call inside one three-second budget. "Is there a
corpus?" only needs to know whether a chunk exists, so it must not count all of
note_chunks to find out. What the probe reports does not change.
"""
import pytest

from src.api import app as api
from src.storage.schema import SCHEMA_VERSION

NOTES = 3


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value

    def scalars(self):
        return self.value


class _Db:
    """storage.engine for _check_database: a healthy database, answering each statement by what it reads."""
    def __init__(self, has_chunks=True, indexed=NOTES):
        self.sql = []
        self.answers = [                                       # first match wins: the most specific text first
            ("current_database", api.EXPECTED_DB), ("extversion", "0.8.6"), ("to_regclass", "present"),
            ("MAX(version)", SCHEMA_VERSION), ("pg_indexes", ["idx_chunks_fts", "idx_chunks_embedding", "idx_chunks_subject"]),
            ("FROM clinical_notes", NOTES), ("DISTINCT config_hash", []), ("FROM note_index_state nis", indexed),
            ("FROM ingestion_runs", "completed"),
            ("EXISTS (SELECT 1 FROM note_chunks)", has_chunks), ("COUNT(*) FROM note_chunks", 5 if has_chunks else 0),
        ]

    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        sql = " ".join(str(statement).split())
        self.sql.append(sql)
        for text, value in self.answers:
            if text in sql:
                return _Result(value)
        raise AssertionError(f"unexpected statement: {sql}")


def _probe(monkeypatch, **state):
    db = _Db(**state)
    monkeypatch.setattr(api.storage, "engine", db)
    monkeypatch.setattr(api, "DATA_PLANE", "research")
    return api._check_database(), db


def test_corpus_check_asks_whether_a_chunk_exists_instead_of_counting_them(monkeypatch):
    result, db = _probe(monkeypatch)
    assert "SELECT COUNT(*) FROM note_chunks" not in db.sql            # the full count read every chunk on every probe
    assert "SELECT EXISTS (SELECT 1 FROM note_chunks)" in db.sql
    assert result["corpus"] == "ok"


@pytest.mark.parametrize("has_chunks,corpus", [(True, "ok"), (False, "empty")])
def test_corpus_status_still_reports_an_empty_corpus(monkeypatch, has_chunks, corpus):
    result, _ = _probe(monkeypatch, has_chunks=has_chunks)
    assert result["corpus"] == corpus


def test_the_index_check_still_counts_every_note(monkeypatch):
    complete, db = _probe(monkeypatch)
    assert complete["index"] == "ok" and (complete["indexed_notes"], complete["eligible_notes"]) == (NOTES, NOTES)
    assert any("FROM clinical_notes" in s for s in db.sql) and any("FROM note_index_state nis" in s for s in db.sql)
    stale, _ = _probe(monkeypatch, indexed=NOTES - 1)                  # one note not indexed: still refused
    assert stale["index"] == "incomplete_or_stale"


# --- the index behind the eligible-notes count -------------------------------------------------
def _norm(sql: str) -> str:
    return " ".join(sql.replace("!=", "<>").split())


def test_eligible_notes_index_has_exactly_the_predicate_the_probe_counts_with():
    """A partial index answers the count only while its predicate is the query's.
    Change one without the other and /ready goes back to scanning the table."""
    import inspect
    from src.storage.schema import ELIGIBLE_NOTES_INDEX_SQL
    predicate = _norm(ELIGIBLE_NOTES_INDEX_SQL).split(" WHERE ", 1)[1]
    assert predicate == "COALESCE(text_deid, text_original) IS NOT NULL AND COALESCE(text_deid, text_original) <> ''"
    assert f"FROM clinical_notes WHERE {predicate}" in _norm(inspect.getsource(api._check_database))


def test_eligible_notes_index_is_additive_and_part_of_a_fresh_schema():
    from src.storage.schema import ELIGIBLE_NOTES_INDEX_SQL, MIGRATIONS, SCHEMA_SQL, SCHEMA_VERSION
    assert ELIGIBLE_NOTES_INDEX_SQL.startswith("CREATE INDEX IF NOT EXISTS idx_notes_eligible ON clinical_notes (note_id) WHERE ")
    assert not any(word in ELIGIBLE_NOTES_INDEX_SQL.upper() for word in ("DROP", "ALTER", "UNIQUE", ";"))
    assert ELIGIBLE_NOTES_INDEX_SQL in SCHEMA_SQL                      # a new database gets it
    assert SCHEMA_VERSION == 5 and not any("idx_notes_eligible" in s for v in MIGRATIONS.values() for s in v)   # no version bump:
    # /ready does not require the index, so a database without it (the holdout copies) is still a complete one
