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

mcp = MCPServer("lumen", version="0.5.0")

_retriever = None
_guidelines = None
_lab_resolver = None

import threading
_load_lock = threading.Lock()


def _load():
    """Lazy model loading for either isolated data plane.
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


def _load_lab_resolver():
    """Reuse the deterministic lab service used by the LangGraph path."""
    global _lab_resolver
    with _load_lock:
        if _lab_resolver is None:
            from src.generation.lab_query import LabResolver
            _lab_resolver = LabResolver()
    return _lab_resolver


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
    resolver = _load_lab_resolver()
    itemids, matched = resolver.match(lab_name.strip())
    if not itemids:
        return {"error": f"unknown lab {lab_name!r}",
                "known_labs": resolver.labels}

    series = resolver.fetch(subject_id, itemids, per_lab_cap=limit)
    values = [
        {
            "label": group["label"],
            "charttime": value["charttime"],
            "value": value["valuenum"],
            "unit": value["uom"],
            "abnormal": value["abnormal"],
        }
        for group in series
        for value in group["values"]
    ]
    values.sort(key=lambda value: value["charttime"])
    values = values[:limit]

    return {
        "notice": planes.banner(), "lab": lab_name, "matched": matched,
        "itemids": itemids, "count": len(values), "values": values,
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
