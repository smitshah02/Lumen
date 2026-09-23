"""Lightweight canonical provenance for the note-vector index."""

from __future__ import annotations

import hashlib
import json

from src.config import MODELS_CONFIG

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
