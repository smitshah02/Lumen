"""
Lumen Agent Graph (skeleton)
============================
Day 1: five nodes wired end to end, all stubs, backed by a durable
Postgres checkpointer. Day 2 fills the node bodies.

The checkpointer connection is derived from the SQLAlchemy engine you
already configure in src/storage, so there is exactly one place the DB
is configured.

Usage:
    from src.agents.graph import build_graph

    graph, checkpointer = build_graph()
    out = graph.invoke(
        {"query": "most recent creatinine", "subject_id": 10000032},
        config={"configurable": {"thread_id": "demo-1"}},
    )
"""

from __future__ import annotations

import logging

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from src.storage import engine
from src.agents.state import AgentState

logger = logging.getLogger(__name__)


def _dsn() -> str:
    """SQLAlchemy URL -> plain libpq DSN (strip the +psycopg2 driver suffix)."""
    return engine.url.set(drivername="postgresql").render_as_string(hide_password=False)


def _trail(state: AgentState, name: str) -> list[str]:
    return list(state.get("node_trail", [])) + [name]


# ===========================================================================
# Nodes — Day 1 stubs. Day 2 replaces the bodies, not the signatures.
# ===========================================================================

def triage(state: AgentState) -> dict:
    # DAY 2: classify query_type + reuse detect_temporal_mode() from the retriever
    logger.info(f"[triage] {state['query']!r}")
    return {"query_type": "chart_review", "temporal_mode": "auto",
            "node_trail": _trail(state, "triage")}


def patient_retrieval(state: AgentState) -> dict:
    # DAY 2: thin wrapper over HybridRetriever.search(); no new retrieval logic
    logger.info(f"[patient_retrieval] subject_id={state.get('subject_id')}")
    return {"patient_evidence": [], "node_trail": _trail(state, "patient_retrieval")}


def guideline_retrieval(state: AgentState) -> dict:
    # DAY 2: THE NEW CODE — guideline_chunks has no retrieval path today
    logger.info("[guideline_retrieval]")
    return {"guideline_evidence": [], "node_trail": _trail(state, "guideline_retrieval")}


def synthesis(state: AgentState) -> dict:
    # DAY 2: local_client.chat(tier="main"), cited draft, [S#]/[L#]/[G#]
    logger.info("[synthesis]")
    return {"draft_answer": "", "citations": [], "node_trail": _trail(state, "synthesis")}


def verification(state: AgentState) -> dict:
    # DAY 2: claim-vs-span check.  DAY 3: interrupt() when unsupported.
    logger.info("[verification]")
    return {"verification": {"checked": 0, "unsupported": 0},
            "needs_human_review": False,
            "node_trail": _trail(state, "verification")}


# ===========================================================================
# Assembly
# ===========================================================================

def build_graph(setup: bool = True):
    """Returns (compiled_graph, checkpointer). Keep the pool alive for the
    process lifetime — do not use the context-manager form for a server."""
    pool = ConnectionPool(
        conninfo=_dsn(),
        max_size=5,
        # Both of these are REQUIRED by PostgresSaver. Omitting either gives
        # errors that do not point at the cause.
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    checkpointer = PostgresSaver(pool)
    if setup:
        checkpointer.setup()   # idempotent; creates the checkpoint tables

    builder = StateGraph(AgentState)
    builder.add_node("triage", triage)
    builder.add_node("patient_retrieval", patient_retrieval)
    builder.add_node("guideline_retrieval", guideline_retrieval)
    builder.add_node("synthesis", synthesis)
    builder.add_node("verification", verification)

    # DAY 2: replace this linear chain with conditional edges off triage,
    # so guideline_check skips patient_retrieval and vice versa.
    builder.add_edge(START, "triage")
    builder.add_edge("triage", "patient_retrieval")
    builder.add_edge("patient_retrieval", "guideline_retrieval")
    builder.add_edge("guideline_retrieval", "synthesis")
    builder.add_edge("synthesis", "verification")
    builder.add_edge("verification", END)

    return builder.compile(checkpointer=checkpointer), checkpointer