"""
Lumen MCP Server
================
Exposes the clinical retrieval layer as MCP tools so any host can query it.

    LUMEN_DATA_PLANE=demo     python -m src.mcp_server.server
    LUMEN_DATA_PLANE=research python -m src.mcp_server.server

CRITICAL — stdio transport uses stdout as the JSON-RPC channel. Anything
written to stdout corrupts the protocol, and the failure looks like the host
silently refusing to connect. Model loading emits progress output, so stdout
is redirected to stderr for the whole process before any of it happens.

Tools return structured rows that preserve chunk_id / charttime / note_type.
Those ids are what make an answer traceable back to a source; losing them at
the tool boundary breaks the citation chain the rest of Lumen depends on.
"""

from __future__ import annotations

import sys
import logging

# Must happen before torch/transformers import anything.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(levelname)s | %(message)s")

from typing import Optional                                    # noqa: E402
from sqlalchemy import text                                    # noqa: E402
from mcp.server.mcpserver import MCPServer                     # noqa: E402

from src.storage import engine                                 # noqa: E402
from src.mcp_server import planes                              # noqa: E402

logger = logging.getLogger(__name__)

mcp = MCPServer("lumen", version="0.4.0")

_retriever = None
_guidelines = None

import threading
_load_lock = threading.Lock()


def _load():
    """Lazy — the demo plane never needs the models.
    v2 runs sync handlers on worker threads, so this must be guarded:
    two concurrent first-calls would otherwise load MedCPT + BGE twice."""
    global _retriever, _guidelines
    with _load_lock:
        if _retriever is None:
            from src.retrieval.hybrid_retriever_v2 import HybridRetriever
            from src.retrieval.guideline_retriever import GuidelineRetriever
            _retriever = HybridRetriever()
            _guidelines = GuidelineRetriever(
                embedder=_retriever.embedder,
                reranker=getattr(_retriever, "reranker", None),
            )
    return _retriever, _guidelines


# MIMIC-IV labevents itemids. There is no d_labitems table in this schema,
# so names are mapped here. VERIFY against your own data before trusting.
LAB_ITEMIDS = {
    "potassium": 50971, "sodium": 50983, "chloride": 50902, "bicarbonate": 50882,
    "creatinine": 50912, "bun": 51006, "urea nitrogen": 51006, "glucose": 50931,
    "calcium": 50893, "magnesium": 50960, "phosphate": 50970,
    "hemoglobin": 51222, "hematocrit": 51221, "wbc": 51301,
    "white blood cells": 51301, "platelet": 51265, "platelets": 51265,
    "alt": 50861, "ast": 50878, "alkaline phosphatase": 50863,
    "bilirubin": 50885, "albumin": 50862, "lactate": 50813,
    "inr": 51237, "pt": 51274, "ptt": 51275,
}


# ===========================================================================
# Tools
# ===========================================================================

@mcp.tool()
def search_patient_notes(query: str, subject_id: Optional[int] = None,
                         top_k: int = 5) -> dict:
    """Search de-identified clinical notes using hybrid retrieval (BM25 +
    MedCPT vectors + cross-encoder reranking). Returns passages with the ids
    needed to cite them.

    Args:
        query: clinical question or topic, e.g. "recent potassium results"
        subject_id: restrict to one patient; omit to search the whole corpus
        top_k: number of passages (1-10)
    """
    top_k = max(1, min(top_k, 10))

    if planes.is_demo():
        rows = [n for n in planes.DEMO_NOTES
                if subject_id is None or n["subject_id"] == subject_id][:top_k]
        return {"notice": planes.banner(), "count": len(rows),
                "results": [{**r, "score": 1.0} for r in rows]}

    retriever, _ = _load()
    results = retriever.search(query=query, subject_id=subject_id,
                               temporal_filter="auto", top_k=top_k)
    return {
        "notice": planes.banner(),
        "count": len(results),
        "results": [{
            "chunk_id": r.chunk_id,
            "subject_id": r.subject_id,
            "note_type": r.note_type,
            "charttime": r.charttime,
            "score": round(float(r.final_score), 4),
            # context_text/chunk_text derive from note_chunks, which is built
            # from text_deid. text_original is never read here.
            "chunk_text": (r.context_text or r.chunk_text),
        } for r in results],
    }


@mcp.tool()
def search_guidelines(query: str, top_k: int = 3) -> dict:
    """Search indexed clinical practice guidelines (ADA Standards of Care,
    AHA/ACC, GOLD). Returns general recommendations, NOT patient-specific
    facts — never present these as something a patient received.

    Args:
        query: clinical topic, e.g. "potassium monitoring in CKD"
        top_k: number of passages (1-8)
    """
    top_k = max(1, min(top_k, 8))
    _, gr = _load()
    results = gr.search(query, top_k=top_k)

    out = []
    for r in results:
        meta = gr.source_of(r.chunk_id)
        out.append({
            "chunk_id": r.chunk_id,
            "source_file": meta.get("source_file"),
            "section_title": meta.get("section_title"),
            "score": round(float(r.final_score), 4),
            "chunk_text": r.chunk_text,
        })
    return {"notice": "General guidance, not patient-specific.",
            "count": len(out), "results": out}


@mcp.tool()
def get_lab_trend(subject_id: int, lab_name: str, limit: int = 20) -> dict:
    """Return a patient's values for one lab test over time, oldest first.

    Args:
        subject_id: patient identifier
        lab_name: e.g. "potassium", "creatinine", "hemoglobin"
        limit: maximum values to return (1-100)
    """
    limit = max(1, min(limit, 100))
    key = lab_name.strip().lower()

    if planes.is_demo():
        rows = planes.DEMO_LABS.get((subject_id, key), [])
        return {"notice": planes.banner(), "lab": lab_name, "count": len(rows),
                "values": [{"charttime": t, "value": v, "unit": u, "flag": f}
                           for t, v, u, f in rows[:limit]]}

    itemid = LAB_ITEMIDS.get(key)
    if itemid is None:
        return {"error": f"unknown lab {lab_name!r}",
                "known_labs": sorted(LAB_ITEMIDS.keys())}

    with engine.connect() as c:
        rows = c.execute(text("""
            SELECT charttime, valuenum, value, valueuom, flag,
                   ref_range_lower, ref_range_upper
            FROM labevents
            WHERE subject_id = :sid AND itemid = :iid AND valuenum IS NOT NULL
            ORDER BY charttime
            LIMIT :lim
        """), {"sid": subject_id, "iid": itemid, "lim": limit}).fetchall()

    if not rows:
        with engine.connect() as c:
            avail = c.execute(text("""
                SELECT itemid, count(*) ct FROM labevents WHERE subject_id = :sid
                GROUP BY itemid ORDER BY ct DESC LIMIT 10
            """), {"sid": subject_id}).fetchall()
        return {"notice": planes.banner(), "lab": lab_name, "count": 0, "values": [],
                "hint": f"no {lab_name} (itemid {itemid}) for subject {subject_id}",
                "available_itemids": [{"itemid": r[0], "n": r[1]} for r in avail]}

    return {
        "notice": planes.banner(), "lab": lab_name, "itemid": itemid, "count": len(rows),
        "values": [{
            "charttime": str(r[0]) if r[0] else None,
            "value": r[1], "raw": r[2], "unit": r[3], "flag": r[4],
            "ref_low": r[5], "ref_high": r[6],
        } for r in rows],
    }


@mcp.tool()
def get_patient_timeline(subject_id: int, limit: int = 40) -> dict:
    """Chronological events for one patient: admissions, notes, and abnormal
    labs. Use this to orient before searching for details.

    Args:
        subject_id: patient identifier
        limit: maximum events (1-200)
    """
    limit = max(1, min(limit, 200))

    if planes.is_demo():
        ev = planes.DEMO_TIMELINE.get(subject_id, [])[:limit]
        return {"notice": planes.banner(), "count": len(ev), "events": ev}

    with engine.connect() as c:
        rows = c.execute(text("""
            SELECT admittime AS when_, 'admission' AS kind,
                   COALESCE(admission_type, 'admission') AS detail
            FROM admissions WHERE subject_id = :sid AND admittime IS NOT NULL
            UNION ALL
            SELECT charttime, 'note', note_type
            FROM clinical_notes WHERE subject_id = :sid AND charttime IS NOT NULL
            UNION ALL
            SELECT charttime, 'lab',
                   'itemid ' || itemid || ' = ' || valuenum || ' ' ||
                   COALESCE(valueuom, '') || ' (' || flag || ')'
            FROM labevents
            WHERE subject_id = :sid AND flag IS NOT NULL AND valuenum IS NOT NULL
            ORDER BY 1
            LIMIT :lim
        """), {"sid": subject_id, "lim": limit}).fetchall()

    return {
        "notice": planes.banner(), "count": len(rows),
        "events": [{"when": str(r[0]), "kind": r[1], "detail": r[2]} for r in rows],
    }


# ===========================================================================

def main() -> None:
    transport = "stdio"
    planes.enforce_transport(transport)          # raises before binding
    logger.info(f"lumen mcp server starting — plane={planes.PLANE}")
    sys.stdout = _REAL_STDOUT                    # hand stdout to the protocol
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()