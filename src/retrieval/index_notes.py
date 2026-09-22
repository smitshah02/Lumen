"""
Note Indexing Pipeline
=======================
Reads de-identified clinical notes from Postgres, chunks them,
embeds with MedCPT, and stores vectors in the note_chunks table.

This is the bridge between raw notes and searchable RAG retrieval.

Usage:
    source .venv/bin/activate
    python -m src.retrieval.index_notes

    # Process a smaller batch for testing:
    python -m src.retrieval.index_notes --limit 500

    # Reindex everything (atomically replaces each note):
    python -m src.retrieval.index_notes --reindex
"""

from __future__ import annotations

import json
import time
import uuid
import logging
import argparse
from typing import Optional

import numpy as np
from sqlalchemy import text as sa_text

from src.storage import engine
from src.retrieval.chunker import ClinicalNoteChunker
from src.retrieval.embeddings import MedCPTEmbedder
from src.retrieval.index_provenance import (CHUNKER_CONFIG, configuration_hash,
                                            index_configuration)

logger = logging.getLogger(__name__)

def fetch_notes(limit: Optional[int] = None, note_type: Optional[str] = None) -> list[dict]:
    """Fetch de-identified notes from the database."""
    query = """
        SELECT note_id, subject_id, hadm_id, note_type,
               COALESCE(text_deid, text_original) as text
        FROM clinical_notes
        WHERE COALESCE(text_deid, text_original) IS NOT NULL
          AND COALESCE(text_deid, text_original) != ''
    """
    params = {}

    if note_type:
        query += " AND note_type = :note_type"
        params["note_type"] = note_type

    query += " ORDER BY note_id"

    if limit:
        query += " LIMIT :limit"
        params["limit"] = limit

    with engine.connect() as conn:
        result = conn.execute(sa_text(query), params)
        rows = result.mappings().all()

    return [dict(r) for r in rows]


def get_existing_note_ids() -> set[int]:
    """Get note_ids that already have chunks (to skip re-processing)."""
    with engine.connect() as conn:
        result = conn.execute(sa_text("SELECT DISTINCT note_id FROM note_chunks"))
        return {row[0] for row in result}


def fetch_note_ids(note_type: Optional[str] = None) -> list[int]:
    query = """
        SELECT note_id FROM clinical_notes
        WHERE COALESCE(text_deid, text_original) IS NOT NULL
          AND COALESCE(text_deid, text_original) != ''
    """
    params = {}
    if note_type:
        query += " AND note_type = :note_type"
        params["note_type"] = note_type
    query += " ORDER BY note_id"
    with engine.connect() as conn:
        return list(conn.execute(sa_text(query), params).scalars())


def fetch_unindexed_note_ids(config_hash: str, note_type: Optional[str] = None) -> list[int]:
    """Notes are complete only when state, provenance, and chunk count agree."""
    query = """
        SELECT cn.note_id
        FROM clinical_notes cn
        WHERE NOT EXISTS (
            SELECT 1 FROM note_index_state nis
            WHERE nis.note_id = cn.note_id
              AND nis.status = 'completed'
              AND nis.config_hash = :config_hash
              AND nis.chunk_count = (
                  SELECT COUNT(*) FROM note_chunks nc WHERE nc.note_id = cn.note_id
              )
        )
          AND COALESCE(cn.text_deid, cn.text_original) IS NOT NULL
          AND COALESCE(cn.text_deid, cn.text_original) != ''
    """
    params = {"config_hash": config_hash}
    if note_type:
        query += " AND cn.note_type = :note_type"
        params["note_type"] = note_type
    query += " ORDER BY cn.note_id"
    with engine.connect() as conn:
        result = conn.execute(sa_text(query), params)
        return [row[0] for row in result]


def fetch_notes_by_ids(note_ids: list[int]) -> list[dict]:
    """Fetch full note text + metadata for a specific list of note_ids."""
    with engine.connect() as conn:
        result = conn.execute(
            sa_text("""
                SELECT note_id, subject_id, hadm_id, note_type,
                       COALESCE(text_deid, text_original) as text
                FROM clinical_notes
                WHERE note_id = ANY(:ids)
                ORDER BY note_id
            """),
            {"ids": note_ids},
        )
        return [dict(r) for r in result.mappings().all()]


def store_chunks_batch(chunks_data: list[dict], config_hash: str) -> tuple[int, int]:
    """Replace each note atomically and mark completion only after commit."""
    if not chunks_data:
        return 0, 0
    by_note: dict[int, list[dict]] = {}
    for chunk in chunks_data:
        by_note.setdefault(chunk["note_id"], []).append(chunk)
    completed = chunks = 0
    for note_id, records in by_note.items():
        try:
            with engine.begin() as conn:
                conn.execute(sa_text("""
                    INSERT INTO note_index_state
                        (note_id, status, config_hash, chunk_count, completed_at, error_message)
                    VALUES (:note_id, 'running', :config_hash, 0, NULL, NULL)
                    ON CONFLICT (note_id) DO UPDATE SET
                        status='running', config_hash=EXCLUDED.config_hash,
                        chunk_count=0, completed_at=NULL, error_message=NULL
                """), {"note_id": note_id, "config_hash": config_hash})
                conn.execute(sa_text("DELETE FROM note_chunks WHERE note_id = :note_id"),
                             {"note_id": note_id})
                for i in range(0, len(records), 100):
                    conn.execute(sa_text("""
                        INSERT INTO note_chunks
                            (note_id, subject_id, hadm_id, note_type,
                             chunk_index, chunk_text, token_count, embedding)
                        VALUES
                            (:note_id, :subject_id, :hadm_id, :note_type,
                             :chunk_index, :chunk_text, :token_count, :embedding)
                    """), records[i:i + 100])
                conn.execute(sa_text("""
                    UPDATE note_index_state SET status='completed',
                        chunk_count=:count, completed_at=NOW(), error_message=NULL
                    WHERE note_id=:note_id
                """), {"note_id": note_id, "count": len(records)})
            completed += 1
            chunks += len(records)
        except BaseException as exc:
            with engine.begin() as conn:
                conn.execute(sa_text("""
                    INSERT INTO note_index_state
                        (note_id, status, config_hash, chunk_count, completed_at, error_message)
                    VALUES (:note_id, 'failed', :config_hash, 0, NOW(), :error)
                    ON CONFLICT (note_id) DO UPDATE SET
                        status='failed', config_hash=EXCLUDED.config_hash,
                        completed_at=NOW(), error_message=EXCLUDED.error_message
                """), {"note_id": note_id, "config_hash": config_hash,
                         "error": type(exc).__name__})
            raise
    return completed, chunks


def _start_index_run(run_id: str, configuration: dict, config_hash: str) -> None:
    with engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO note_index_runs (run_id, status, configuration, config_hash)
            VALUES (:run_id, 'running', CAST(:configuration AS jsonb), :config_hash)
        """), {"run_id": run_id, "configuration": json.dumps(configuration, sort_keys=True),
                 "config_hash": config_hash})


def _update_index_run(run_id: str, *, expected: int | None = None,
                      completed: int | None = None, chunks: int | None = None) -> None:
    fields, params = [], {"run_id": run_id}
    for column, value in (("notes_expected", expected), ("notes_completed", completed),
                          ("chunks_created", chunks)):
        if value is not None:
            fields.append(f"{column} = :{column}")
            params[column] = value
    if fields:
        with engine.begin() as conn:
            conn.execute(sa_text(f"UPDATE note_index_runs SET {', '.join(fields)} WHERE run_id=:run_id"), params)


def _finish_index_run(run_id: str, status: str, error_type: str | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE note_index_runs SET status=:status, completed_at=NOW(), error_message=:error
            WHERE run_id=:run_id
        """), {"run_id": run_id, "status": status, "error": error_type})


def _run_indexing_steps(
    run_id: str,
    config_hash: str,
    limit: Optional[int] = None,
    note_type: Optional[str] = None,
    reindex: bool = False,
    embed_batch_size: int = 32,
    note_batch_size: int = 1000,
    cooldown_secs: int = 30,
):
    """
    Main indexing pipeline — processes notes in batches with a cooldown
    between each batch to prevent the machine from overheating.

      1. Fetch all unindexed note_ids from Postgres
      2. Process note_batch_size notes at a time:
           chunk → embed → store → sleep cooldown_secs
      3. Repeat until all notes are indexed
    """
    print("=" * 70)
    print("  LUMEN NOTE INDEXING PIPELINE")
    print("=" * 70)
    print()

    # Step 0: A reindex replaces each note atomically after its new embedding
    # is ready, so the old usable index remains until that note commits.
    if reindex:
        print("Rebuilding every note atomically (--reindex)...")
        print()

    # Step 1: Fetch all unindexed note IDs (lightweight — IDs only)
    print("Step 1: Finding unindexed notes...")
    all_ids = (fetch_note_ids(note_type=note_type) if reindex else
               fetch_unindexed_note_ids(config_hash, note_type=note_type))
    if limit:
        all_ids = all_ids[:limit]

    total_notes = len(all_ids)
    _update_index_run(run_id, expected=total_notes)
    if not total_notes:
        print("  All notes already indexed. Use --reindex to rebuild.")
        return {"notes": 0, "chunks": 0}

    n_batches = (total_notes + note_batch_size - 1) // note_batch_size
    print(f"  {total_notes:,} notes to index in {n_batches} batches of {note_batch_size}")
    print()

    # Step 2: Load models once — keep alive across batches
    print("Step 2: Loading chunker and MedCPT embedder...")
    chunker = ClinicalNoteChunker(**CHUNKER_CONFIG)
    embedder = MedCPTEmbedder(batch_size=embed_batch_size)
    print()

    # Step 3: Batch loop
    total_chunks_created = 0
    total_notes_done = 0
    pipeline_start = time.time()

    for batch_num, batch_start in enumerate(range(0, total_notes, note_batch_size), start=1):
        batch_ids = all_ids[batch_start : batch_start + note_batch_size]
        batch_actual = len(batch_ids)

        print(
            f"── Batch {batch_num}/{n_batches}  "
            f"(notes {batch_start + 1}–{batch_start + batch_actual} of {total_notes}) ──"
        )

        # Fetch full text for this batch only
        notes = fetch_notes_by_ids(batch_ids)

        # Chunk
        t0 = time.time()
        chunk_records = []
        chunk_texts = []
        for note in notes:
            for chunk in chunker.chunk_text(note["text"], note_type=note.get("note_type", "discharge")):
                chunk_records.append({
                    "note_id":      note["note_id"],
                    "subject_id":   note["subject_id"],
                    "hadm_id":      note["hadm_id"],
                    "note_type":    note.get("note_type", ""),
                    "chunk_index":  chunk.chunk_index,
                    "chunk_text":   chunk.text,
                    "token_count":  chunk.token_count,
                })
                chunk_texts.append(chunk.text)
        print(f"  Chunked  : {batch_actual} notes → {len(chunk_texts)} chunks ({time.time()-t0:.1f}s)")

        # Embed
        t0 = time.time()
        embeddings = embedder.embed_documents(chunk_texts, show_progress=True)
        embed_time = time.time() - t0
        rate = len(chunk_texts) / embed_time if embed_time > 0 else 0
        print(f"  Embedded : {len(chunk_texts)} chunks ({embed_time:.1f}s, {rate:.0f} chunks/sec)")

        # Attach vectors and store
        for i, record in enumerate(chunk_records):
            vec = embeddings[i]
            record["embedding"] = f"[{','.join(str(float(x)) for x in vec)}]"
        notes_stored, chunks_stored = store_chunks_batch(chunk_records, config_hash)

        total_chunks_created += chunks_stored
        total_notes_done += notes_stored
        _update_index_run(run_id, completed=total_notes_done, chunks=total_chunks_created)
        elapsed = time.time() - pipeline_start
        pct = total_notes_done / total_notes * 100
        print(
            f"  Stored   : {len(chunk_records)} chunks | "
            f"Progress: {total_notes_done:,}/{total_notes:,} ({pct:.1f}%) | "
            f"Elapsed: {elapsed:.0f}s"
        )

        # Cooldown between batches (skip after the last one)
        if batch_num < n_batches:
            for remaining in range(cooldown_secs, 0, -1):
                print(f"\r  Cooling down... {remaining}s ", end="", flush=True)
                time.sleep(1)
            print("\r  Cooling down... done.          ")

        print()

    # Summary
    print("=" * 70)
    print("  INDEXING COMPLETE")
    print("=" * 70)
    print()
    print(f"  Notes processed:  {total_notes_done:,}")
    print(f"  Chunks created:   {total_chunks_created:,}")
    print(f"  Total time:       {time.time() - pipeline_start:.1f}s")
    print()

    with engine.connect() as conn:
        result = conn.execute(sa_text("SELECT COUNT(*) FROM note_chunks"))
        print(f"  Total chunks in database: {result.scalar():,}")
    print()
    if total_notes_done != total_notes:
        raise RuntimeError(
            f"index incomplete: completed {total_notes_done} of {total_notes} expected notes"
        )
    return {"notes": total_notes_done, "chunks": total_chunks_created}


def run_indexing(
    limit: Optional[int] = None,
    note_type: Optional[str] = None,
    reindex: bool = False,
    embed_batch_size: int = 32,
    note_batch_size: int = 1000,
    cooldown_secs: int = 30,
):
    """Index notes with durable provenance and run completion accounting."""
    configuration = {
        **index_configuration(),
        "scope": {"limit": limit, "note_type": note_type, "reindex": reindex},
        "embedding_batch_size": embed_batch_size,
        "note_batch_size": note_batch_size,
    }
    config_hash = configuration_hash()
    run_id = str(uuid.uuid4())
    _start_index_run(run_id, configuration, config_hash)
    try:
        result = _run_indexing_steps(
            run_id, config_hash, limit=limit, note_type=note_type, reindex=reindex,
            embed_batch_size=embed_batch_size, note_batch_size=note_batch_size,
            cooldown_secs=cooldown_secs,
        )
    except BaseException as exc:
        _finish_index_run(run_id, "failed", type(exc).__name__)
        raise
    _finish_index_run(run_id, "completed")
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Lumen Note Indexing Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Max notes to process (default: all)")
    parser.add_argument("--note-type", type=str, default=None, help="Filter by note type (discharge/radiology)")
    parser.add_argument("--reindex", action="store_true", help="Atomically rebuild every note")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--note-batch-size", type=int, default=1000, help="Notes per batch before cooldown")
    parser.add_argument("--cooldown", type=int, default=30, help="Seconds to sleep between batches")
    args = parser.parse_args()

    run_indexing(
        limit=args.limit,
        note_type=args.note_type,
        reindex=args.reindex,
        embed_batch_size=args.batch_size,
        note_batch_size=args.note_batch_size,
        cooldown_secs=args.cooldown,
    )
