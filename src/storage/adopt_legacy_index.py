"""Adopt a note index that predates provenance tracking — metadata only.

An index built before ``note_index_state`` existed has no record of which
chunker or model revision produced it. Re-embedding it just to obtain that
record would throw away a working index, and stamping it with the current
configuration hash would record something nobody can verify. This command
does neither: after read-only checks that the index is complete and usable it
records the index under its own ``legacy-adopted`` hash, which readiness and
the indexer accept alongside the current one
(``src.retrieval.index_provenance.legacy_adopted_hashes``).

It writes ``note_index_runs`` and ``note_index_state`` only. It never reads
note text and never touches ``clinical_notes``, ``note_chunks`` or any index.

    LUMEN_DATA_PLANE=research python -m src.storage.adopt_legacy_index --check
    LUMEN_DATA_PLANE=research python -m src.storage.adopt_legacy_index \\
        --apply --confirm-legacy-adoption
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from sqlalchemy import text

from src import storage
from src.retrieval.index_provenance import (CHUNKER_CONFIG, LEGACY_ADOPTED, VECTOR_DIMENSION,
                                            configuration_hash, index_configuration,
                                            legacy_adopted_hashes)

_ELIGIBLE = "COALESCE(cn.text_deid, cn.text_original) IS NOT NULL AND COALESCE(cn.text_deid, cn.text_original) != ''"


def inspect(conn) -> dict:
    """Aggregate facts about the existing index. Counts only; no note text."""
    def one(sql: str):
        return conn.execute(text(sql)).scalar()

    return {
        "eligible_notes": one(f"SELECT COUNT(*) FROM clinical_notes cn WHERE {_ELIGIBLE}"),
        "eligible_notes_without_chunks": one(f"""
            SELECT COUNT(*) FROM clinical_notes cn WHERE {_ELIGIBLE}
              AND NOT EXISTS (SELECT 1 FROM note_chunks nc WHERE nc.note_id = cn.note_id)"""),
        "chunks": one("SELECT COUNT(*) FROM note_chunks"),
        "chunks_without_embedding": one("SELECT COUNT(*) FROM note_chunks WHERE embedding IS NULL"),
        "vector_dimensions": sorted(conn.execute(text(
            "SELECT DISTINCT vector_dims(embedding) FROM note_chunks WHERE embedding IS NOT NULL"
        )).scalars()),
        "notes_with_chunk_gaps": one("""
            SELECT COUNT(*) FROM (SELECT 1 FROM note_chunks GROUP BY note_id
                                  HAVING MIN(chunk_index) <> 0
                                      OR MAX(chunk_index) + 1 <> COUNT(*)) g"""),
        "duplicate_chunk_positions": one("""
            SELECT COUNT(*) FROM (SELECT 1 FROM note_chunks
                                  GROUP BY note_id, chunk_index HAVING COUNT(*) > 1) d"""),
        "max_token_count": one("SELECT MAX(token_count) FROM note_chunks"),
        "chunks_over_current_max_tokens": one(
            f"SELECT COUNT(*) FROM note_chunks WHERE token_count > {int(CHUNKER_CONFIG['max_tokens'])}"),
        "existing_state_rows": one("SELECT COUNT(*) FROM note_index_state"),
    }


def problems(facts: dict) -> list[str]:
    """Why this index must not be adopted. Empty means it is complete and usable."""
    out = []
    if not facts["eligible_notes"]:
        out.append("no eligible notes")
    if facts["eligible_notes_without_chunks"]:
        out.append(f"{facts['eligible_notes_without_chunks']} eligible note(s) have no chunks")
    if facts["chunks_without_embedding"]:
        out.append(f"{facts['chunks_without_embedding']} chunk(s) have no embedding")
    if facts["vector_dimensions"] != [VECTOR_DIMENSION]:
        out.append(f"vector dimensions {facts['vector_dimensions']} != [{VECTOR_DIMENSION}]")
    if facts["notes_with_chunk_gaps"]:
        out.append(f"{facts['notes_with_chunk_gaps']} note(s) have gaps in their chunk positions")
    if facts["duplicate_chunk_positions"]:
        out.append(f"{facts['duplicate_chunk_positions']} duplicate (note_id, chunk_index) position(s)")
    return out


def legacy_configuration(facts: dict) -> dict:
    """What is known about the adopted index, and what is not."""
    return {
        "provenance": LEGACY_ADOPTED,
        "embedding": {**index_configuration()["embedding"],
                      "basis": "pinned revision assumed; the index build was not recorded"},
        "chunker": {"version": "unrecorded",
                    "matches_current": facts["chunks_over_current_max_tokens"] == 0,
                    "current_max_tokens": CHUNKER_CONFIG["max_tokens"],
                    "observed_max_token_count": facts["max_token_count"],
                    "chunks_over_current_max_tokens": facts["chunks_over_current_max_tokens"]},
        "observed": {"eligible_notes": facts["eligible_notes"], "chunks": facts["chunks"]},
    }


def adopt(conn, facts: dict) -> dict:
    """Record the index. Idempotent: a second call changes nothing.

    Refuses, writing nothing, when an earlier adoption was made under a
    different embedding pin (stamping the new pin on old vectors would be
    false; that index needs --reindex) or when some notes already carry
    provenance rows (a partly tracked index is not a legacy one)."""
    prior = conn.execute(text("""
        SELECT config_hash FROM note_index_runs
        WHERE status = 'completed' AND configuration->>'provenance' = :provenance
        ORDER BY completed_at DESC LIMIT 1
    """), {"provenance": LEGACY_ADOPTED}).scalar()
    if prior:
        status = ("already_adopted" if prior in legacy_adopted_hashes(conn)
                  else "stale_adoption_requires_reindex")
        return {"status": status, "config_hash": prior, "notes_recorded": 0}
    if facts["existing_state_rows"]:
        return {"status": "refused_partial_state", "config_hash": "", "notes_recorded": 0}
    configuration = legacy_configuration(facts)
    config_hash = configuration_hash(configuration)
    recorded = conn.execute(text("""
        INSERT INTO note_index_state (note_id, status, config_hash, chunk_count, completed_at)
        SELECT nc.note_id, 'completed', :config_hash, COUNT(*), NOW()
        FROM note_chunks nc GROUP BY nc.note_id
        ON CONFLICT (note_id) DO NOTHING
    """), {"config_hash": config_hash}).rowcount
    conn.execute(text("""
        INSERT INTO note_index_runs (run_id, status, configuration, config_hash,
                                     notes_expected, notes_completed, chunks_created, completed_at)
        VALUES (:run_id, 'completed', CAST(:configuration AS jsonb), :config_hash,
                :expected, :recorded, 0, NOW())
    """), {"run_id": str(uuid.uuid4()), "configuration": json.dumps(configuration),
           "config_hash": config_hash, "expected": facts["eligible_notes"], "recorded": recorded})
    return {"status": "adopted", "config_hash": config_hash, "notes_recorded": recorded}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="read-only report")
    parser.add_argument("--apply", action="store_true", help="record the adoption")
    parser.add_argument("--confirm-legacy-adoption", action="store_true")
    args = parser.parse_args(argv)

    if storage.DATA_PLANE != "research":
        print(f"refusing: legacy adoption applies to the research plane, not {storage.DATA_PLANE!r}",
              file=sys.stderr)
        return 2
    if args.apply == args.check:
        parser.error("choose exactly one of --check or --apply")
    if args.apply and not args.confirm_legacy_adoption:
        print("refusing: --apply requires --confirm-legacy-adoption", file=sys.stderr)
        return 2

    with storage.engine.begin() as conn:
        facts = inspect(conn)
        blocking = problems(facts)
        report = {"facts": facts, "problems": blocking,
                  "current_config_hash": configuration_hash()[:12]}
        if args.apply and not blocking:
            report["result"] = adopt(conn, facts)
            report["result"]["config_hash"] = report["result"]["config_hash"][:12]
    print(json.dumps(report, indent=2))
    refused = report.get("result", {}).get("status") not in (None, "adopted", "already_adopted")
    return 1 if blocking or refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
