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
from src.retrieval.section_labels import label_chunks
from src.retrieval.index_provenance import (CHUNKER_CONFIG, configuration_hash,
                                            index_configuration,
                                            legacy_adopted_hashes)

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
              AND nis.config_hash = ANY(:config_hashes)
              AND nis.chunk_count = (
                  SELECT COUNT(*) FROM note_chunks nc WHERE nc.note_id = cn.note_id
              )
        )
          AND COALESCE(cn.text_deid, cn.text_original) IS NOT NULL
          AND COALESCE(cn.text_deid, cn.text_original) != ''
    """
    params = {}
    if note_type:
        query += " AND cn.note_type = :note_type"
        params["note_type"] = note_type
    query += " ORDER BY cn.note_id"
    with engine.connect() as conn:
        # A legacy-adopted index counts as indexed; only --reindex replaces it.
        params["config_hashes"] = [config_hash, *legacy_adopted_hashes(conn)]
        result = conn.execute(sa_text(query), params)
        return [row[0] for row in result]


def fetch_notes_by_ids(note_ids: list[int]) -> list[dict]:
    """Fetch full note text + metadata for a specific list of note_ids."""
    with engine.connect() as conn:
        result = conn.execute(
            sa_text("""
                SELECT note_id, subject_id, hadm_id, note_type,
                       COALESCE(text_deid, text_original) as text,
                       COALESCE(text_original, text_deid) as text_as_written
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
                # Search-only section names. The DELETE above cascaded the old ones away.
                labelled = [r for r in records if r.get("search_labels")]
                if labelled:
                    conn.execute(sa_text("""
                        INSERT INTO chunk_search_labels (chunk_id, labels)
                        SELECT chunk_id, :search_labels FROM note_chunks
                        WHERE note_id = :note_id AND chunk_index = :chunk_index
                    """), labelled)
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
            chunks = chunker.chunk_text(note["text"], note_type=note.get("note_type", "discharge"))
            # Search-only section names, read from the headers of the note as written.
            labels = (label_chunks(note.get("text_as_written") or note["text"], [c.text for c in chunks])
                      if note.get("note_type") == "discharge" else [None] * len(chunks))
            for chunk, label in zip(chunks, labels):
                chunk_records.append({
                    "search_labels": label,
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


def _legacy_index_unadopted() -> bool:
    """Some note has chunks but no index-state row: part of an index built
    before provenance tracking that was never adopted. (A note indexed by this
    module always gets its state row first, so this cannot be a run in
    progress.) One adopted or --limit-indexed note must not hide the rest."""
    with engine.connect() as conn:
        return bool(conn.execute(sa_text("""
            SELECT EXISTS (
                SELECT 1 FROM note_chunks nc
                WHERE NOT EXISTS (SELECT 1 FROM note_index_state nis WHERE nis.note_id = nc.note_id))
        """)).scalar())


def run_indexing(
    limit: Optional[int] = None,
    note_type: Optional[str] = None,
    reindex: bool = False,
    embed_batch_size: int = 32,
    note_batch_size: int = 1000,
    cooldown_secs: int = 30,
):
    """Index notes with durable provenance and run completion accounting."""
    if not reindex and _legacy_index_unadopted():
        # Without this, every note looks unindexed and is deleted and re-embedded.
        raise RuntimeError(
            "note_chunks holds an index with no provenance rows; record it with "
            "`python -m src.storage.adopt_legacy_index --apply --confirm-legacy-adoption` "
            "(scripts/lumen research adopt), or pass --reindex to rebuild it deliberately; "
            "adoption refuses an index that some notes already track, which leaves --reindex"
        )
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


# ===========================================================================
# v2 build (data-foundation plan, E13)
# ===========================================================================
# Builds note_chunks_v2 for a set of patients from the note as written, using
# parse_sections + chunk_note. Everything above is the control indexer and is
# untouched; this never reads or writes note_chunks.
#
#     python -m src.retrieval.index_notes --v2-build --subjects 10000032,10000980
#     python -m src.retrieval.index_notes --v2-build --subjects-file ids_a.txt --subjects-file ids_b.txt
#
# One build is one note_index_runs row (its run_id is the build_id that
# LUMEN_CHUNK_BUILD selects) with the full configuration, and its status is
# running until every chunk and embedding is stored and counted, then completed.
# Any error leaves it failed. A build only ever writes rows carrying its own
# build_id, so a failed or interrupted build cannot damage an earlier one.

def v2_configuration(subject_ids, chunk_config: dict) -> dict:
    """Everything that determines a v2 build's chunks: recorded, and hashed."""
    from src.retrieval.chunker import V2_CHUNKER_VERSION
    from src.retrieval.section_labels import PARSER_VERSION
    embedding = index_configuration()["embedding"]
    return {
        "provenance": "v2",
        "table": "note_chunks_v2",
        "source_text": "note as written (text_original, else text_deid)",
        "parser": {"version": PARSER_VERSION},
        "chunker": {"version": V2_CHUNKER_VERSION, **chunk_config, "embedded_text": "chunk_text, verbatim"},
        "tokenizer": {"repo": embedding["repo"], "revision": embedding["revision"]},
        "embedding": embedding,
        "scope": {"subject_ids": sorted(int(s) for s in subject_ids)},
    }


def recover_mimic_note_ids(notes: list[dict], note_dir=None) -> dict[int, str]:
    """clinical_notes.note_id -> MIMIC-IV-Note note_id ("10000032-DS-21").

    clinical_notes does not keep MIMIC's id, so it is recovered from the source
    files by patient, chart time and the exact text. Every note must match
    exactly one source row; anything else raises, because a guessed identity
    would make chunk keys meaningless."""
    import csv
    import gzip
    import hashlib
    from src.config import MIMIC_NOTE_DIR

    def key(subject_id, charttime, text) -> tuple:
        return int(subject_id), str(charttime)[:19], hashlib.md5((text or "").encode("utf-8")).hexdigest()

    wanted: dict[tuple, list[int]] = {}
    for n in notes:
        wanted.setdefault((n["note_type"],) + key(n["subject_id"], n["charttime"], n["text"]), []).append(n["note_id"])
    subjects = {n["subject_id"] for n in notes}
    note_dir = note_dir or MIMIC_NOTE_DIR / "note"
    found: dict[int, str] = {}
    for note_type in sorted({n["note_type"] for n in notes}):
        path = next((p for p in (note_dir / f"{note_type}.csv.gz", note_dir / f"{note_type}.csv") if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"{note_type}.csv(.gz) not found in {note_dir}")
        with (gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz"
              else open(path, "r", encoding="utf-8", newline="")) as f:
            for row in csv.DictReader(f):
                if int(row["subject_id"]) not in subjects:
                    continue
                ids = wanted.get((note_type,) + key(row["subject_id"], row["charttime"], row["text"]))
                if ids:
                    found[ids.pop(0)] = row["note_id"]          # identical duplicates pair off in file order
    missing = sorted(n["note_id"] for n in notes if n["note_id"] not in found)
    if missing:
        raise RuntimeError(f"{len(missing)} note(s) have no matching MIMIC source row (first note_id {missing[0]})")
    if len(set(found.values())) != len(found):
        raise RuntimeError("two notes resolved to the same MIMIC note id")
    return found


def v2_rows(notes: list[dict], mimic_ids: dict[int, str], tokenizer, chunk_config: dict, build_id: str) -> list[dict]:
    """The note_chunks_v2 rows for `notes`, without embeddings. Pure and
    deterministic: same notes, tokenizer and configuration, same keys and text."""
    from src.retrieval.chunker import chunk_note
    rows = []
    for note in sorted(notes, key=lambda n: mimic_ids[n["note_id"]]):
        for c in chunk_note(note["text"], note["note_type"], tokenizer, **chunk_config):
            rows.append({
                "build_id": build_id, "mimic_note_id": mimic_ids[note["note_id"]],
                "section_ord": c.section_ord, "chunk_ord": c.chunk_ord, "note_id": note["note_id"],
                "subject_id": note["subject_id"], "hadm_id": note["hadm_id"], "note_type": note["note_type"],
                "charttime": note["charttime"], "section_name": c.section_name, "char_start": c.start,
                "char_end": c.end, "chunk_text": c.text, "token_count": c.token_count, "embed": c.embed,
                "embedding": None,
            })
    return rows


def ensure_v2_table(db=None) -> None:
    """Create note_chunks_v2 and its indexes if missing. Additive and idempotent."""
    from src.storage.schema import NOTE_CHUNKS_V2_SQL
    with (db or engine).begin() as conn:
        for statement in NOTE_CHUNKS_V2_SQL.split(";"):
            conn.execute(sa_text(statement))


def _fetch_v2_notes(conn, subject_ids) -> list[dict]:
    return [dict(r) for r in conn.execute(sa_text("""
        SELECT note_id, subject_id, hadm_id, note_type, charttime, COALESCE(text_original, text_deid) AS text
        FROM clinical_notes
        WHERE subject_id = ANY(:subjects) AND COALESCE(text_original, text_deid) IS NOT NULL
          AND COALESCE(text_original, text_deid) != ''
        ORDER BY note_id"""), {"subjects": sorted(int(s) for s in subject_ids)}).mappings()]


_V2_INSERT = sa_text("""
    INSERT INTO note_chunks_v2 (build_id, mimic_note_id, section_ord, chunk_ord, note_id, subject_id, hadm_id,
                                note_type, charttime, section_name, char_start, char_end, chunk_text, token_count,
                                embed, embedding)
    VALUES (:build_id, :mimic_note_id, :section_ord, :chunk_ord, :note_id, :subject_id, :hadm_id,
            :note_type, :charttime, :section_name, :char_start, :char_end, :chunk_text, :token_count,
            :embed, CAST(:embedding AS vector))""")


def read_subject_ids(listed: str = "", path: Optional[str] = None) -> list[int]:
    """Patient ids for a build: a comma-separated list, a file, or both. The
    file holds ids only, one per line; blank lines and # comments are skipped,
    and anything else is refused, so a question or an answer can never ride in."""
    lines = listed.split(",")
    if path:
        with open(path, encoding="utf-8") as f:
            lines += [line.split("#", 1)[0] for line in f]
    ids = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not line.isdigit():
            raise ValueError("a patient id list may contain subject_ids only")     # never echo the line
        ids.add(int(line))
    return sorted(ids)


def run_v2_build(subject_ids, chunk_config: Optional[dict] = None, *, db=None, tokenizer=None, embedder=None,
                 mimic_ids: Optional[dict] = None, note_batch: int = 200) -> dict:
    """Build note_chunks_v2 for the given patients as one new build. Returns its
    id and statistics. `db`, `tokenizer`, `embedder` and `mimic_ids` default to
    the real ones and exist so the lifecycle can be tested without them."""
    from src.retrieval.chunker import V2_CHUNK_CONFIG, load_article_tokenizer
    db = db or engine
    chunk_config = dict(chunk_config or V2_CHUNK_CONFIG)
    configuration = v2_configuration(subject_ids, chunk_config)
    ensure_v2_table(db)
    with db.connect() as conn:
        notes = _fetch_v2_notes(conn, subject_ids)
    if not notes:
        raise RuntimeError("no notes found for the given patients")

    build_id = str(uuid.uuid4())
    with db.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO note_index_runs (run_id, status, configuration, config_hash, notes_expected)
            VALUES (:run_id, 'running', CAST(:configuration AS jsonb), :config_hash, :expected)"""),
            {"run_id": build_id, "configuration": json.dumps(configuration, sort_keys=True),
             "config_hash": configuration_hash({k: v for k, v in configuration.items() if k != "scope"}),
             "expected": len(notes)})

    def finish(status: str, error: Optional[str] = None, **counts) -> None:
        with db.begin() as conn:
            conn.execute(sa_text("""
                UPDATE note_index_runs SET status = :status, completed_at = NOW(), error_message = :error,
                       notes_completed = :notes, chunks_created = :chunks WHERE run_id = :run_id"""),
                {"run_id": build_id, "status": status, "error": error,
                 "notes": counts.get("notes", 0), "chunks": counts.get("chunks", 0)})

    t0, stats = time.time(), {"build_id": build_id, "patients": len({n["subject_id"] for n in notes}), "notes": len(notes),
                              "chunks": 0, "embedded": 0, "lexical_only": 0, "max_embedded_tokens": 0,
                              "chunk_seconds": 0.0, "embed_seconds": 0.0, "store_seconds": 0.0}
    try:
        mimic_ids = mimic_ids or recover_mimic_note_ids(notes)
        tokenizer = tokenizer or load_article_tokenizer()
        embedder = embedder or MedCPTEmbedder(batch_size=32)
        done = 0
        for i in range(0, len(notes), note_batch):
            batch = notes[i:i + note_batch]
            t = time.time()
            rows = v2_rows(batch, mimic_ids, tokenizer, chunk_config, build_id)
            stats["chunk_seconds"] += time.time() - t
            to_embed = [r for r in rows if r["embed"]]
            t = time.time()
            vectors = embedder.embed_documents([r["chunk_text"] for r in to_embed], show_progress=False) if to_embed else []
            stats["embed_seconds"] += time.time() - t
            if len(vectors) != len(to_embed):
                raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(to_embed)} chunks")
            for row, vec in zip(to_embed, vectors):
                row["embedding"] = f"[{','.join(str(float(x)) for x in vec)}]"
            t = time.time()
            with db.begin() as conn:                           # a batch of notes is stored whole or not at all
                for j in range(0, len(rows), 200):
                    conn.execute(_V2_INSERT, rows[j:j + 200])
            stats["store_seconds"] += time.time() - t
            done += len(batch)
            stats["chunks"] += len(rows)
            stats["embedded"] += len(to_embed)
            stats["lexical_only"] += len(rows) - len(to_embed)
            stats["max_embedded_tokens"] = max([stats["max_embedded_tokens"]] + [r["token_count"] for r in to_embed])
        # Completed only when what is stored is what was built.
        with db.connect() as conn:
            stored, with_vector = conn.execute(sa_text(
                "SELECT COUNT(*), COUNT(embedding) FROM note_chunks_v2 WHERE build_id = :b"), {"b": build_id}).first()
        if (stored, with_vector) != (stats["chunks"], stats["embedded"]):
            raise RuntimeError(f"stored {stored} chunks / {with_vector} embeddings, built {stats['chunks']} / {stats['embedded']}")
    except BaseException as exc:
        finish("failed", type(exc).__name__, notes=0, chunks=stats["chunks"])
        raise
    finish("completed", notes=done, chunks=stats["chunks"])
    stats["seconds"] = round(time.time() - t0, 1)
    for k in ("chunk_seconds", "embed_seconds", "store_seconds"):
        stats[k] = round(stats[k], 1)
    return stats


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
    parser.add_argument("--v2-build", action="store_true", help="Build note_chunks_v2 for --subjects as a new build")
    parser.add_argument("--subjects", type=str, default="", help="Comma-separated subject_ids for --v2-build")
    parser.add_argument("--subjects-file", action="append", default=[],
                        help="File of subject_ids, one per line, for --v2-build; may be given more than once")
    args = parser.parse_args()
    if args.v2_build:
        subjects = sorted({s for path in (args.subjects_file or [None]) for s in read_subject_ids(args.subjects, path)})
        if not subjects:
            parser.error("--v2-build needs --subjects or --subjects-file")
        built = run_v2_build(subjects)
        for name, value in built.items():
            print(f"  {name:<22} {value}")
        raise SystemExit(0)

    run_indexing(
        limit=args.limit,
        note_type=args.note_type,
        reindex=args.reindex,
        embed_batch_size=args.batch_size,
        note_batch_size=args.note_batch_size,
        cooldown_secs=args.cooldown,
    )
