"""
Migration: Chunk-Level Full-Text Search
========================================
Adds a per-chunk tsvector column + GIN index to note_chunks so BM25 ranks
each chunk on its OWN text, not the whole note.

Before: bm25_search ranked on clinical_notes.text_search (note-level) — every
        chunk of a matching note shared one identical score, so a "social
        history" chunk and a "discharge meds" chunk from the same note were
        indistinguishable to lexical search.
After:  each chunk has note_chunks.text_search (chunk-level) — chunks compete
        on their own content.

The column is GENERATED ALWAYS ... STORED, so:
  - existing rows are backfilled automatically by the ALTER
  - new inserts populate it with NO change to index_notes.py
  - it can never drift out of sync with chunk_text

No re-embedding required. Safe to re-run (idempotent). Run once.

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    python -m src.storage.migrate_chunk_fts
    python -m src.storage.migrate_chunk_fts --verify-only
"""

from __future__ import annotations

import time
import logging
import argparse

from sqlalchemy import text as sa_text
from src.storage import engine, check_connection

logger = logging.getLogger(__name__)

# to_tsvector('english', ...) — the TWO-arg form with a constant config is
# IMMUTABLE, which is required for a GENERATED column. The one-arg form is only
# STABLE and would be rejected, so the explicit 'english' is load-bearing.
ADD_COLUMN_SQL = """
ALTER TABLE note_chunks
    ADD COLUMN IF NOT EXISTS text_search tsvector
    GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON note_chunks USING GIN(text_search)
"""


def column_exists() -> bool:
    sql = """
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'note_chunks' AND column_name = 'text_search'
    """
    with engine.connect() as conn:
        return conn.execute(sa_text(sql)).fetchone() is not None


def verify():
    """Confirm the column is populated and the chunk-level rank path works."""
    with engine.connect() as conn:
        total = conn.execute(sa_text("SELECT COUNT(*) FROM note_chunks")).scalar()
        populated = conn.execute(
            sa_text("SELECT COUNT(*) FROM note_chunks WHERE text_search IS NOT NULL")
        ).scalar()
        sample = conn.execute(sa_text("""
            SELECT chunk_id, note_id,
                   ts_rank_cd(text_search,
                              plainto_tsquery('english', 'heart failure')) AS rank
            FROM note_chunks
            WHERE text_search @@ plainto_tsquery('english', 'heart failure')
            ORDER BY rank DESC
            LIMIT 5
        """)).mappings().all()

    print(f"  Total chunks:        {total:,}")
    print(f"  With text_search:    {populated:,}")
    if total:
        print(f"  Coverage:            {populated / total:.1%}")
    print("  Sample 'heart failure' hits (top 5 by chunk-level rank):")
    if sample:
        for row in sample:
            print(f"    chunk {row['chunk_id']:<8} note {row['note_id']:<8} rank={row['rank']:.4f}")
        # If chunk-level worked, distinct chunks should show DISTINCT ranks here.
        distinct_ranks = len({round(r["rank"], 6) for r in sample})
        print(f"  Distinct ranks among top 5: {distinct_ranks}/{len(sample)} "
              f"({'good — per-chunk scoring' if distinct_ranks > 1 else 'all tied — check data'})")
    else:
        print("    (no matches — is note_chunks populated? run index_notes first)")


def run_migration():
    if not check_connection():
        raise ConnectionError("Cannot connect to database. Is Docker running?")

    print("=" * 70)
    print("  MIGRATION: chunk-level full-text search")
    print("=" * 70)
    print()

    if column_exists():
        print("note_chunks.text_search already exists — ensuring index, then verifying.")
    else:
        print("Adding generated tsvector column to note_chunks...")
        print("  (backfills existing rows; may take a moment on a large table)")

    # ADD COLUMN GENERATED is transactional in Postgres; commit per statement
    # for clear, isolated error reporting.
    t0 = time.time()
    with engine.connect() as conn:
        conn.execute(sa_text(ADD_COLUMN_SQL))
        conn.commit()
    print(f"  Column ready in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("Creating GIN index on note_chunks.text_search...")
    with engine.connect() as conn:
        conn.execute(sa_text(CREATE_INDEX_SQL))
        conn.commit()
    print(f"  Index ready in {time.time() - t0:.1f}s")
    print()

    print("Verifying:")
    verify()
    print()
    print("Done. Now update bm25_search to query nc.text_search (see the patch),")
    print("then re-run the eval to measure the chunk-level vs note-level gain.")
    print("Tip: run  ANALYZE note_chunks;  once, so the planner uses the new index.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Add chunk-level FTS to note_chunks")
    parser.add_argument("--verify-only", action="store_true", help="Only run verification")
    args = parser.parse_args()

    if args.verify_only:
        print("Verification:")
        verify()
    else:
        run_migration()
