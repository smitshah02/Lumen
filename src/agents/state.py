"""
Agent State
===========
The typed state that flows through the LangGraph. Every node reads it
and returns a partial update; LangGraph merges and checkpoints.

Design note: evidence entries keep chunk_id and charttime all the way
through. The verifier and the citation validator both need to resolve a
claim back to an exact source span, and losing the ids at any hop breaks
the grounding chain.
"""

from __future__ import annotations

from typing import Optional, Literal
from typing_extensions import TypedDict


QueryType = Literal[
    "chart_review",     # what happened to this patient
    "guideline_check",  # what does the guideline recommend
    "lab_trend",        # values over time
    "literature",       # published evidence (Day 5)
    "unsupported",      # out of scope -> refuse
]


class Evidence(TypedDict):
    """One retrieved passage, with everything needed to cite it."""
    chunk_id: int
    source_type: str          # "note" | "guideline" | "lab" | "literature"
    text: str
    charttime: Optional[str]
    note_type: Optional[str]
    score: float
    label: str                # the citation marker: "S1", "L1", "G1"


class Citation(TypedDict):
    """A claim linked to the evidence that should support it."""
    claim: str
    label: str
    chunk_id: int
    verified: bool
    verification_note: str


class EgressRecord(TypedDict):
    """One attempted outbound call to an external tool (Day 5)."""
    tool: str
    allowed: bool
    rule: Optional[str]
    payload_sha256: str


class AgentState(TypedDict, total=False):
    # --- input ---
    query: str
    subject_id: Optional[int]
    thread_id: str

    # --- triage ---
    query_type: QueryType
    temporal_mode: str        # from detect_temporal_mode()

    # --- evidence ---
    patient_evidence: list[Evidence]
    guideline_evidence: list[Evidence]
    literature_evidence: list[Evidence]

    # --- generation ---
    draft_answer: str
    citations: list[Citation]

    # --- verification ---
    verification: dict
    needs_human_review: bool
    human_decisions: list[dict]

    # --- observability / safety ---
    egress_log: list[EgressRecord]
    node_trail: list[str]     # which nodes ran, in order
    errors: list[str]