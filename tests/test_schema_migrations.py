"""Static migration invariants; no database or protected data required."""

import pytest

from src.storage import schema


def test_every_schema_version_has_an_ordered_migration():
    assert schema.pending_migrations(0) == list(range(1, schema.SCHEMA_VERSION + 1))
    assert schema.pending_migrations(schema.SCHEMA_VERSION) == []
    assert set(schema.MIGRATIONS) == set(range(1, schema.SCHEMA_VERSION + 1))


def test_current_schema_contains_required_runtime_objects():
    sql = schema.SCHEMA_SQL.lower()
    assert "create table if not exists d_labitems" in sql
    assert "create table if not exists lumen_schema_version" in sql
    assert "create table if not exists ingestion_runs" in sql
    assert "create table if not exists note_index_runs" in sql
    assert "create table if not exists note_index_state" in sql
    assert "idx_chunks_note_position_unique" in sql
    assert "generated always as (to_tsvector('english', chunk_text))" in sql


@pytest.mark.parametrize("current,target", [(-1, 2), (2, 1), (0, 99)])
def test_invalid_migration_ranges_are_refused(current, target):
    with pytest.raises(ValueError):
        schema.pending_migrations(current, target)
