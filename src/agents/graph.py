"""
Lumen Agent Graph
=================
Day 2: real node bodies and conditional routing.

    triage ──> patient_retrieval ──> guideline_retrieval ──> synthesis ──> verification
       │              │                       ▲
       │              └───────────────────────┘ (skipped unless needed)
       └──> refuse ──> END

Retrievers are module-level singletons: MedCPT + BGE are ~3.5GB on MPS
and must be loaded exactly once per process.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from src.storage import engine
from src.agents.state import AgentState
from src.agents import prompts, citations
from src.llm.local_client import chat
from src.retrieval.hybrid_retriever_v2 import HybridRetriever, detect_temporal_mode
from src.retrieval.guideline_retriever import GuidelineRetriever
from langgraph.types import Command, interrupt

logger = logging.getLogger(__name__)

PATIENT_TOP_K = 5      # keep the synthesis prompt inside num_ctx on 16GB
GUIDELINE_TOP_K = 3

_retriever: Optional[HybridRetriever] = None
_guidelines: Optional[GuidelineRetriever] = None


def get_retrievers() -> tuple[HybridRetriever, GuidelineRetriever]:
    """Lazy singletons. The guideline retriever SHARES the embedder and
    reranker — loading a second pair would blow a 16GB machine."""
    global _retriever, _guidelines
    if _retriever is None:
        logger.info("loading retrievers (one-time, ~20s)...")
        _retriever = HybridRetriever()
        _guidelines = GuidelineRetriever(
            embedder=_retriever.embedder,
            reranker=getattr(_retriever, "reranker", None),
            top_k=GUIDELINE_TOP_K,
        )
    return _retriever, _guidelines


def _dsn() -> str:
    return engine.url.set(drivername="postgresql").render_as_string(hide_password=False)


def _trail(state: AgentState, name: str) -> list[str]:
    return list(state.get("node_trail", [])) + [name]


def _to_evidence(results, prefix: str, source_type: str) -> list[dict]:
    out = []
    for i, r in enumerate(results, 1):
        out.append({
            "chunk_id": r.chunk_id,
            "source_type": source_type,
            "text": (r.context_text or r.chunk_text),
            "charttime": r.charttime,
            "note_type": r.note_type,
            "score": round(float(r.final_score), 4),
            "label": f"{prefix}{i}",
        })
    return out


# ===========================================================================
# Nodes
# ===========================================================================

def triage(state: AgentState) -> dict:
    query = state["query"]
    temporal = detect_temporal_mode(query)

    qtype, target = "chart_review", ""
    try:
        raw = chat(
            [{"role": "system", "content": prompts.TRIAGE_SYSTEM},
             {"role": "user", "content": query}],
            tier="fast", json_mode=True, max_tokens=120,
        )
        parsed = json.loads(raw)
        cand = parsed.get("query_type", "")
        if cand in {"chart_review", "guideline_check", "lab_trend", "literature", "unsupported"}:
            qtype = cand
        target = parsed.get("target", "")
    except Exception as e:
        # Heuristic fallback — never let triage take the graph down.
        logger.warning(f"triage LLM failed ({e}); falling back to heuristics")
        q = query.lower()
        if any(w in q for w in ("should", "recommend", "guideline", "indicated")):
            qtype = "guideline_check"
        elif temporal in ("latest", "trend"):
            qtype = "lab_trend"

    logger.info(f"[triage] type={qtype} temporal={temporal} target={target!r}")
    return {"query_type": qtype, "temporal_mode": temporal,
            "node_trail": _trail(state, "triage")}


def patient_retrieval(state: AgentState) -> dict:
    retriever, _ = get_retrievers()
    results = retriever.search(
        query=state["query"],
        subject_id=state.get("subject_id"),
        temporal_filter=state.get("temporal_mode") or "auto",
        top_k=PATIENT_TOP_K,
    )
    ev = _to_evidence(results, "S", "note")
    logger.info(f"[patient_retrieval] {len(ev)} chunks")
    return {"patient_evidence": ev, "node_trail": _trail(state, "patient_retrieval")}


def guideline_retrieval(state: AgentState) -> dict:
    _, gr = get_retrievers()
    results = gr.search(state["query"], top_k=GUIDELINE_TOP_K)
    ev = _to_evidence(results, "G", "guideline")
    logger.info(f"[guideline_retrieval] {len(ev)} chunks")
    return {"guideline_evidence": ev, "node_trail": _trail(state, "guideline_retrieval")}


def synthesis(state: AgentState) -> dict:
    pt = state.get("patient_evidence", []) or []
    gl = state.get("guideline_evidence", []) or []

    if not pt and not gl:
        return {"draft_answer": "The available records do not contain enough information to answer this.",
                "citations": [], "node_trail": _trail(state, "synthesis")}

    user = prompts.build_synthesis_prompt(state["query"], pt, gl)
    try:
        answer = chat(
            [{"role": "system", "content": prompts.SYNTHESIS_SYSTEM},
             {"role": "user", "content": user}],
            tier="main", max_tokens=500,
        )
    except Exception as e:
        logger.error(f"synthesis failed: {e}")
        return {"draft_answer": "", "citations": [],
                "errors": list(state.get("errors", [])) + [f"synthesis: {e}"],
                "node_trail": _trail(state, "synthesis")}

    report = citations.validate(answer, pt + gl)
    if report["bad_labels"]:
        logger.warning(f"[synthesis] hallucinated labels {report['bad_labels']} — stripped")
        answer = citations.strip_bad_labels(answer, pt + gl)

    cites = [{
        "claim": c["claim"],
        "label": c["valid_labels"][0] if c["valid_labels"] else "",
        "chunk_id": next((e["chunk_id"] for e in pt + gl if e["label"] == (c["valid_labels"] or [None])[0]), -1),
        "verified": False,
        "verification_note": "",
    } for c in report["claims"]]

    logger.info(f"[synthesis] {report['n_claims']} claims, "
                f"cite_rate={report['cite_rate']:.0%}, bad={report['bad_labels']}")
    return {"draft_answer": answer, "citations": cites,
            "verification": {"citation_report": {k: v for k, v in report.items() if k != "claims"}},
            "node_trail": _trail(state, "synthesis")}


def verification(state: AgentState) -> dict:
    ev = {e["label"]: e for e in (state.get("patient_evidence", []) or []) +
                                  (state.get("guideline_evidence", []) or [])}
    cites = state.get("citations", []) or []
    checked = unsupported = 0
    out = []

    for c in cites:
        src = ev.get(c["label"])
        if not src:
            c = {**c, "verified": False, "verification_note": "no valid citation"}
            unsupported += 1
            out.append(c)
            continue
        try:
            raw = chat(
                [{"role": "system", "content": prompts.VERIFY_SYSTEM},
                 {"role": "user", "content": f"CLAIM: {c['claim']}\n\nSOURCE TEXT:\n{src['text'][:3000]}"}],
                tier="main", json_mode=True, max_tokens=150,
            )
            parsed = json.loads(raw)
            verdict = parsed.get("verdict", "unsupported")
            note = parsed.get("reason", "")
        except Exception as e:
            verdict, note = "unsupported", f"verify error: {e}"

        checked += 1
        ok = verdict == "supported"
        if not ok:
            unsupported += 1
        out.append({**c, "verified": ok, "verification_note": f"{verdict}: {note}"})

        prior = state.get("verification", {}) or {}
    logger.info(f"[verification] {checked} checked, {unsupported} unsupported")
    return {
        "citations": out,
        "verification": {**prior, "checked": checked, "unsupported": unsupported},
        "needs_human_review": unsupported > 0,
        "review_status": "pending" if unsupported > 0 else "auto_approved",
        "node_trail": _trail(state, "verification"),
    }

def human_review(state: AgentState) -> dict:
    """
    Pause for clinician adjudication of unsupported claims.

    This node is deliberately cheap: it reads state and interrupts, nothing
    more. LangGraph re-executes a node from the top on resume, so any LLM
    work here would repeat on every decision — which is why the verification
    calls live in the previous node.
    """
    cites = state.get("citations", []) or []
    flagged = [(i, c) for i, c in enumerate(cites) if not c.get("verified")]

    if not flagged:
        return {"review_status": "auto_approved", "node_trail": _trail(state, "human_review")}

    ev = {e["label"]: e for e in (state.get("patient_evidence", []) or []) +
                                  (state.get("guideline_evidence", []) or [])}

    payload = {
        "query": state.get("query", ""),
        "answer": state.get("draft_answer", ""),
        "flagged": [{
            "index": i,
            "claim": c["claim"],
            "label": c.get("label", ""),
            "note": c.get("verification_note", ""),
            "source_text": (ev.get(c.get("label", ""), {}) or {}).get("text", "")[:3000],
        } for i, c in flagged],
    }

    logger.info(f"[human_review] pausing on {len(flagged)} flagged claim(s)")
    decisions = interrupt(payload)   # <-- graph halts here; state is checkpointed

    # ---- resumed from here ----
    if isinstance(decisions, dict):
        decisions = [decisions]
    if not isinstance(decisions, list):
        decisions = [{"action": "escalate", "note": "malformed resume payload"}]

    updated = [dict(c) for c in cites]
    struck, escalated, recorded = set(), False, []

    for (idx, _), d in zip(flagged, decisions):
        action = (d or {}).get("action", "escalate")
        note = (d or {}).get("note", "")
        recorded.append({"index": idx, "action": action, "note": note})

        if action == "approve":
            updated[idx]["verified"] = True
            updated[idx]["verification_note"] = f"human approved: {note}" if note else "human approved"
        elif action == "strike":
            struck.add(idx)
            updated[idx]["verification_note"] = f"struck by reviewer: {note}" if note else "struck by reviewer"
        else:
            escalated = True
            updated[idx]["verification_note"] = f"escalated: {note}" if note else "escalated"

    keep = {i for i in range(len(updated)) if i not in struck}
    final = citations.rebuild_answer(updated, keep)

    logger.info(f"[human_review] {len(recorded)} decision(s); struck={len(struck)} escalated={escalated}")
    return {
        "citations": updated,
        "human_decisions": list(state.get("human_decisions", [])) + recorded,
        "final_answer": final,
        "review_status": "escalated" if escalated else "reviewed",
        "needs_human_review": escalated,
        "node_trail": _trail(state, "human_review"),
    }


def finalize(state: AgentState) -> dict:
    """Set final_answer when no review was needed."""
    if state.get("final_answer"):
        return {"node_trail": _trail(state, "finalize")}
    return {"final_answer": state.get("draft_answer", ""),
            "node_trail": _trail(state, "finalize")}


def refuse(state: AgentState) -> dict:
    logger.info("[refuse] out of scope")
    return {"draft_answer": "This question is outside what the clinical record can support.",
            "citations": [], "needs_human_review": False,
            "node_trail": _trail(state, "refuse")}


# ===========================================================================
# Routing
# ===========================================================================

def route_from_triage(state: AgentState) -> str:
    qt = state.get("query_type", "chart_review")
    if qt == "unsupported":
        return "refuse"
    if qt in ("guideline_check", "literature"):
        return "guideline_retrieval" if state.get("subject_id") is None else "patient_retrieval"
    return "patient_retrieval"


def route_after_patient(state: AgentState) -> str:
    # Only fetch guidelines when the question is actually asking what
    # should be done. A plain chart lookup does not need them.
    return "guideline_retrieval" if state.get("query_type") in ("guideline_check", "literature") else "synthesis"

def route_after_verification(state: AgentState) -> str:
    return "human_review" if state.get("needs_human_review") else "finalize"

def build_graph(setup: bool = True):
    pool = ConnectionPool(
        conninfo=_dsn(), max_size=5,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    checkpointer = PostgresSaver(pool)
    _pools.append(pool)
    if setup:
        checkpointer.setup()

    b = StateGraph(AgentState)
    b.add_node("triage", triage)
    b.add_node("patient_retrieval", patient_retrieval)
    b.add_node("guideline_retrieval", guideline_retrieval)
    b.add_node("synthesis", synthesis)
    b.add_node("verification", verification)
    b.add_node("refuse", refuse)

    b.add_edge(START, "triage")
    b.add_conditional_edges("triage", route_from_triage,
                            ["patient_retrieval", "guideline_retrieval", "refuse"])
    b.add_conditional_edges("patient_retrieval", route_after_patient,
                            ["guideline_retrieval", "synthesis"])
    b.add_edge("guideline_retrieval", "synthesis")
    b.add_edge("synthesis", "verification")
    b.add_edge("verification", END)
    b.add_node("human_review", human_review)
    b.add_node("finalize", finalize)
    b.add_conditional_edges("verification", route_after_verification,
                            ["human_review", "finalize"])
    b.add_edge("human_review", "finalize")
    b.add_edge("finalize", END)

    return b.compile(checkpointer=checkpointer), checkpointer

import atexit

_pools: list[ConnectionPool] = []

def close_pools() -> None:
    """Close pools before interpreter shutdown. Python 3.14 forbids joining
    threads during finalization, so relying on __del__ raises a (harmless but
    noisy) PythonFinalizationError."""
    while _pools:
        try:
            _pools.pop().close()
        except Exception:
            pass


atexit.register(close_pools)