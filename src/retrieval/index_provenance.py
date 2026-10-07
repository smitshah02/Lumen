"""Lightweight canonical provenance for the note-vector index."""

from __future__ import annotations

import hashlib
import json

from src.config import MODELS_CONFIG

# The chunk tables the retriever's SQL can read. This is the single statement of
# that fact: readiness refuses a profile whose chunk table is not listed, and a
# test holds the list to the tables the retriever's queries actually name. It
# gained "note_chunks_v2" in the change that taught the retriever to query it.
RETRIEVER_CHUNK_TABLES = ("note_chunks", "note_chunks_v2")

CHUNKER_VERSION = "clinical-note-chunker-v2"
CHUNKER_CONFIG = {"max_tokens": 384, "overlap_tokens": 64, "min_chunk_tokens": 50}
VECTOR_DIMENSION = 768


def index_configuration() -> dict:
    article = MODELS_CONFIG["hugging_face"]["medcpt-article"]
    return {
        "chunker": {"version": CHUNKER_VERSION, **CHUNKER_CONFIG},
        "embedding": {"repo": article["repo"], "revision": article["revision"],
                      "vector_dimension": VECTOR_DIMENSION},
    }


def configuration_hash(configuration: dict | None = None) -> str:
    payload = json.dumps(configuration or index_configuration(),
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# An index built before provenance was recorded cannot be given the current
# configuration hash truthfully: nothing proves which chunker produced it. It is
# adopted under its own hash instead (src/storage/adopt_legacy_index.py), and
# that hash is accepted alongside the current one wherever "is this note
# indexed?" is asked — so a usable index reads as ready and is never silently
# re-embedded, while its provenance still says what it is.
LEGACY_ADOPTED = "legacy-adopted"


def legacy_adopted_hashes(conn) -> list[str]:
    """Adopted hashes that still match the current embedding pin. When the
    pinned embedding model changes, an adopted index stops counting as indexed:
    otherwise new notes would be embedded with one model and the adopted ones
    queried as if they came from it too."""
    from sqlalchemy import text
    embedding = index_configuration()["embedding"]
    return list(conn.execute(text("""
        SELECT DISTINCT config_hash FROM note_index_runs
        WHERE status = 'completed' AND configuration->>'provenance' = :provenance
          AND configuration->'embedding'->>'repo' = :repo
          AND configuration->'embedding'->>'revision' = :revision
    """), {"provenance": LEGACY_ADOPTED, "repo": embedding["repo"],
           "revision": embedding["revision"]}).scalars())

