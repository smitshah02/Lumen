"""
Evidence provenance resolver
============================
src/agents/graph.py:_to_evidence() copies chunk_id, text, charttime, note_type,
score and label out of a RetrievalResult — but drops `subject_id` and
`hadm_id`, which the retriever does carry. Two required checks need them:

    cross-patient leakage   every cited chunk must belong to the case's subject
    admission scope         cited chunks should fall in the gold admissions

Rather than change the frozen online path to carry two more fields, this module
resolves them READ-ONLY from note_chunks at evaluation time. It issues a single
parameterised SELECT and writes nothing.

Chunk id -1 is the deterministic lab path's sentinel (src/agents/graph.py:
lab_lookup) — no note chunk backs a labevents row. Such evidence is resolved
from the lab row's own text, which states the subject, not from note_chunks.
"""

from __future__ import annotations

import re
import hashlib
import logging

logger = logging.getLogger(__name__)

LAB_SENTINEL_CHUNK_ID = -1
_LAB_SUBJECT_RE = re.compile(r"subject\s+(\d+)", re.I)


def evidence_digest(ev: dict) -> dict:
    """Privacy-preserving record of one evidence item.

    The retrieved note text is NEVER written to a result artifact: only its
    label, id, length and SHA-256. Two runs can be compared chunk-for-chunk
    without either file containing clinical prose.
    """
    text = ev.get("text") or ""
    return {
        "label": ev.get("label"),
        "chunk_id": ev.get("chunk_id"),
        "source_type": ev.get("source_type"),
        "note_type": ev.get("note_type"),
        "charttime": ev.get("charttime"),
        "score": ev.get("score"),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_chars": len(text),
    }


def resolve_provenance(evidence: list[dict], subject_id: int | None) -> dict:
    """label -> {subject_id, hadm_id, resolution}.

    `resolution` says how each answer was reached, so a check can never mistake
    "could not resolve" for "resolved and clean":
        note_chunks     read from the database
        lab_row         parsed from the deterministic lab evidence text
        unresolved      no row found, or the database was unreachable
    """
    out: dict = {}
    to_lookup: dict = {}

    for ev in evidence or []:
        label, cid = ev.get("label"), ev.get("chunk_id")
        if not label:
            continue
        if cid is None or int(cid) == LAB_SENTINEL_CHUNK_ID:
            m = _LAB_SUBJECT_RE.search(ev.get("text") or "")
            out[label] = {
                "subject_id": int(m.group(1)) if m else subject_id,
                "hadm_id": None,
                "resolution": "lab_row" if m else "unresolved",
            }
            continue
        to_lookup.setdefault(int(cid), []).append(label)

    if to_lookup:
        rows = _lookup_chunks(sorted(to_lookup))
        for cid, labels in to_lookup.items():
            row = rows.get(cid)
            for label in labels:
                out[label] = ({"subject_id": row["subject_id"], "hadm_id": row["hadm_id"],
                               "resolution": "note_chunks"} if row else
                              {"subject_id": None, "hadm_id": None, "resolution": "unresolved"})
    return out


def _lookup_chunks(chunk_ids: list[int]) -> dict:
    """One read-only SELECT. A database failure yields no rows rather than an
    exception: the caller reports every affected label as `unresolved`, which
    fails the leakage check open (visible) instead of closed (silent pass)."""
    try:
        from sqlalchemy import text as sa_text
        from src.storage import engine
        with engine.connect() as c:
            rows = c.execute(sa_text(
                "SELECT chunk_id, subject_id, hadm_id FROM note_chunks "
                "WHERE chunk_id = ANY(:ids)"), {"ids": list(chunk_ids)}).mappings().all()
        return {int(r["chunk_id"]): {"subject_id": r["subject_id"], "hadm_id": r["hadm_id"]}
                for r in rows}
    except Exception as e:
        logger.warning("evidence provenance lookup failed (%s); labels will be "
                       "reported unresolved", type(e).__name__)
        return {}
