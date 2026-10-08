"""
Lumen Agent Graph
=================
Day 2: real node bodies and conditional routing.

    triage ──> patient_retrieval ──> guideline_retrieval ──> synthesis ──> verification
       │  │           │                       ▲
       │  │           └───────────────────────┘ (skipped unless needed)
       │  └──> lab_lookup / encounter_lookup ──> finalize   (structured hit: no LLM at all)
       │              └──────> patient_retrieval            (miss: normal path)
       └──> refuse ──> END

Latency shape: triage classifies in code and only calls the FAST model on an
unrecognised phrasing; synthesis picks FAST or MAIN from the query's
complexity; verification settles what it can in code and batches the rest into
ONE FAST call instead of one MAIN call per citation.

Retrievers are module-level singletons: MedCPT + BGE are ~3.5GB on MPS
and must be loaded exactly once per process.
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
from typing import Optional

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError

from src.storage import engine
from src.storage.checkpoints import checkpoint_schema_status
from src.agents.state import AgentState
from src.agents.admission_scope import (AdmissionResolution, descriptive_reference, load_admissions,
                                        load_candidates, load_stay_window, resolve_admission,
                                        resolve_with_model)
from src.config import PROFILE_SETTINGS
from src.storage.readiness import DataSourceNotReady
from src.generation import structured_lookup as structured
from src.agents import prompts, citations, verify as verify_util
from src.agents.classify import (classify, encounter_intents, lab_mode, scoped_lab_mode, structured_admission_clause,
                                 wants_deterministic_lab, wants_encounter_lookup)
from src.generation.lab_query import SYNONYMS, label_tokens
from src.llm.local_client import chat_for   # every node call goes through a ROLE
from src.obs.logging import add_timing, bump
from src.retrieval.hybrid_retriever_v2 import HybridRetriever, detect_temporal_mode
from src.retrieval.guideline_retriever import GuidelineRetriever
from langgraph.types import Command, interrupt
from src.safety.egress_gate import EgressGate, call_external
from src.safety import pubmed
from src.obs import tracing

logger = logging.getLogger(__name__)

PATIENT_TOP_K = 5      # keep the synthesis prompt inside num_ctx on 16GB
GUIDELINE_TOP_K = 3

# The structured shortcuts answer "most recent / earliest / trend of <analyte>"
# straight from labevents, and admission counts and dates from admissions. Set LUMEN_DETERMINISTIC_LABS=0 to force every
# question back through retrieval + synthesis (used for A/B latency runs).
DETERMINISTIC_LABS = os.environ.get("LUMEN_DETERMINISTIC_LABS", "1").strip() not in ("0", "false", "no")
LAB_RECENT_POINTS = 4   # values rendered per analyte as citable evidence
LAB_SERIES_CAP = 100_000  # a trend's first/min/max need the whole series, not the newest 80

_retriever: Optional[HybridRetriever] = None
_guidelines: Optional[GuidelineRetriever] = None
_labs = None


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


def get_lab_resolver():
    """Lazy singleton. Reads d_labitems once; no model weights involved."""
    global _labs
    if _labs is None:
        from src.generation.lab_query import LabResolver
        _labs = LabResolver()
    return _labs


def _dsn() -> str:
    return engine.url.set(drivername="postgresql").render_as_string(hide_password=False)


def _trail(state: AgentState, name: str) -> list[str]:
    """Return ONLY this node's entry — AgentState.node_trail carries an
    operator.add reducer, so LangGraph appends it to what is already there.
    Returning the merged list here would duplicate the whole trail."""
    return [name]


def _to_evidence(results, prefix: str, source_type: str) -> list[dict]:
    out = []
    for i, r in enumerate(results, 1):
        out.append({
            "chunk_id": r.chunk_id,
            "note_id": r.note_id,
            "subject_id": r.subject_id,
            "hadm_id": r.hadm_id,
            "chunk_index": r.chunk_index,
            "source_type": source_type,
            "text": (r.context_text or r.chunk_text),
            "charttime": r.charttime,
            "note_type": r.note_type,
            "score": round(float(r.final_score), 4),
            "label": f"{prefix}{i}",
        })
        if getattr(r, "provenance", None):          # v2 chunks only; other evidence is unchanged
            out[-1]["provenance"] = r.provenance
    return out

def _gate_for(state: AgentState) -> EgressGate:
    """Build a gate loaded with everything the patient side has retrieved.

    Rebuilt per call rather than held on the graph: LangGraph re-executes
    nodes on resume, and a gate carrying stale corpus from a previous run
    would protect the wrong text. Shingling a handful of chunks is cheap.
    """
    gate = EgressGate()
    gate.load_evidence(state.get("patient_evidence", []) or [])
    gate.load_evidence(state.get("guideline_evidence", []) or [])
    gate.load_evidence(state.get("lab_evidence", []) or [])
    gate.load_evidence(state.get("encounter_evidence", []) or [])
    gate.load_evidence(state.get("structured_evidence", []) or [])
    return gate


# The external literature tool, or None when none is enabled — the default.
# LUMEN_LITERATURE_BACKEND=pubmed turns on src/safety/pubmed.py. The stand-in
# in src/safety/stub_tools.py exists to exercise the egress gate in its eval
# and must never be wired in here: its "studies" and ids are invented.
LITERATURE_TOOL = pubmed.configured_tool()
LITERATURE_UNAVAILABLE = ("External literature retrieval is not available in this local research build, "
                          "so no published studies were consulted for this answer.")
LITERATURE_EMPTY = ("The external literature search returned no usable results for this question, "
                    "so no published studies were consulted for this answer.")


def _is_stub(backend) -> bool:
    return not backend or str(backend).lower().startswith("stub")


def _citable_literature(state: AgentState) -> list[dict]:
    """Literature evidence that may be cited: only what a real backend returned.
    A second guard behind literature_retrieval, so an invented result cannot
    become a valid [P#] label even if one reaches the state some other way."""
    return [e for e in (state.get("literature_evidence", []) or []) if not _is_stub(e.get("backend"))]


def _with_literature_notice(state: AgentState, answer: str) -> str:
    """Say, in fixed words, that no literature was consulted. Attached to the
    answer rather than written by the model, and never treated as a claim."""
    notice = LITERATURE_UNAVAILABLE if LITERATURE_TOOL is None else LITERATURE_EMPTY
    if not state.get("literature_unavailable") or notice in (answer or ""):
        return answer
    return f"{answer}\n\n{notice}".strip()


def _to_literature_evidence(tool_output: dict) -> list[dict]:
    """Tool results as [P#] evidence. A stub backend yields none."""
    backend = (tool_output or {}).get("source")
    if _is_stub(backend):
        return []
    out = []
    for i, r in enumerate(tool_output.get("results", []), 1):
        out.append({
            "chunk_id": -1,
            "source_type": "literature",
            "backend": backend,
            "text": (f"{r.get('title', '')} {r.get('journal', '')} {r.get('year', '')}. "
                     f"PMID {r.get('pmid', '')}.\n{r.get('abstract', '')}").strip(),
            "charttime": r.get("year") or None,
            "note_type": f"PubMed PMID {r.get('pmid', '')}",
            "score": 0.0,
            "label": f"P{i}",
        })
    return out


# ===========================================================================
# Nodes
# ===========================================================================

def triage(state: AgentState) -> dict:
    """Classify the query and decide how much model to spend on it.

    The deterministic classifier settles the common cases with no model call at
    all. Only an unrecognised or safety-sensitive phrasing falls through to the
    FAST triage prompt, which is the same call this node always used to make.
    """
    query = state["query"]
    temporal = detect_temporal_mode(query)
    d = classify(query, temporal)
    qtype, complexity, target = d.query_type, d.complexity, ""

    if not d.confident:
        try:
            # One attempt: the deterministic decision is already a safe answer,
            # so a backoff loop here only adds latency to a degraded request.
            raw = chat_for("triage", [{"role": "system", "content": prompts.TRIAGE_SYSTEM},
                                      {"role": "user", "content": query}], max_retries=0)
            parsed = json.loads(raw)
            cand = parsed.get("query_type", "")
            if cand in {"chart_review", "guideline_check", "lab_trend", "literature", "unsupported"}:
                qtype = cand
            target = parsed.get("target", "")
        except Exception as e:
            # Heuristic fallback — never let triage take the graph down. The
            # deterministic decision is already a safe default, so keep it.
            logger.warning(f"triage LLM failed ({e}); keeping deterministic class {qtype}")
    else:
        bump("triage_deterministic")

    logger.info(f"[triage] type={qtype} complexity={complexity} temporal={temporal} "
                f"rule={d.reason} llm={not d.confident} target={target!r}")
    out = {"query_type": qtype, "temporal_mode": temporal, "query_complexity": complexity,
           "classified_by": "rules" if d.confident else "fast_model",
           "node_trail": _trail(state, "triage")}
    if PROFILE_SETTINGS["admission_scope"] and state.get("subject_id") is not None:
        out["admission_scope"] = _admission_scope(query, state["subject_id"],
                                                  state.get("request_hadm_id")).as_state()
    return out


def _admission_scope(query: str, subject_id: int, request_hadm_id: Optional[int] = None) -> AdmissionResolution:
    """Which admission the question names; patient_retrieval scopes to it."""
    try:
        admissions = load_admissions(subject_id)
    except SQLAlchemyError as e:
        logger.warning(f"[triage] admissions lookup failed ({type(e).__name__}); admission scope unresolved")
        return AdmissionResolution("unresolved", query.strip(), reason="admissions lookup failed")
    res = resolve_admission(query, admissions, request_hadm_id)
    # Model step: only for the structured and v2 profiles, only when the rules found
    # no admission reference at all, and only when the question describes one.
    if res.status == "none" and PROFILE_SETTINGS["sql_paths"] and descriptive_reference(query):
        res = resolve_with_model(query, load_candidates(subject_id), _ask_admission_model)
    logger.info(f"[triage] admission_scope status={res.status} hadm_id={res.hadm_id} source={res.source}")
    return res


def _ask_admission_model(prompt: str) -> str:
    # The verify role is the fast tier in JSON mode with room for a quoted line; one attempt, no retries.
    return chat_for("verify", [{"role": "system", "content": prompts.ADMISSION_SYSTEM},
                               {"role": "user", "content": prompt}], max_retries=0)


# What an admission-scoped lab question may say on top of _QUESTION_FILLER, by
# scoped_lab_mode(). Separate from _MODE_FILLER so whole-patient questions are
# understood exactly as before.
_SCOPED_MODE_FILLER = {
    "latest": frozenset({"final"}),
    "earliest": frozenset("earliest oldest first initial known".split()),
    "trend": frozenset("""
        how did does has have been change changed changes changing over time trend trends trending trended
        course whole entire increase increased increasing decrease decreased decreasing or
        from to first earliest initial final
    """.split()),
    "min": frozenset("lowest minimum min least".split()),
    "max": frozenset("highest maximum max peak".split()),
}


def _scoped_lab_request(state: AgentState) -> Optional[tuple[str, int, str]]:
    """(question without its admission phrase, hadm_id, mode) when this is a lab
    question about one resolved admission: the rules classified it as a lab
    question and it asks for a latest, earliest, lowest or highest value or a
    trend. Otherwise None. No query is made here.

    Such a question goes to lab_lookup ahead of the admissions path, whose word
    list ("latest", "last", "first", "how many" next to "admission") would
    otherwise claim it and, on its own miss, send it to note retrieval. A
    question the admissions table fully answers is never taken from it."""
    scope = state.get("admission_scope") or {}
    if (not DETERMINISTIC_LABS or scope.get("status") != "resolved" or state.get("subject_id") is None
            or state.get("query_type") != "lab_trend" or state.get("classified_by") != "rules"):
        return None
    question = scope.get("retrieval_query") or state.get("query") or ""
    if encounter_intents(question)[1]:
        return None
    mode = scoped_lab_mode(question, detect_temporal_mode(question))
    return (question, scope["hadm_id"], mode) if mode else None


def lab_lookup(state: AgentState) -> dict:
    """Answer a latest / earliest / trend lab question straight from `labevents`.

    This is not a shortcut around grounding: the values, units and dates are
    read from the structured table the notes are generated from, rendered as
    citable [L#] evidence, and the claims are marked verified because the
    numbers in the answer ARE the numbers in the rows. It is a shortcut around
    asking a language model to re-read numbers a query can select — and, for a
    trend, around guessing a series from the five note chunks retrieval kept.

    A miss — no matching analyte, no rows for this patient, a question that
    does not resolve to exactly one analyte or carries words beyond it, a
    non-numeric or conflicting result at the end of the series the question
    asks about, mixed units in a trend, or a series drawn from more than one
    specimen — answers nothing, and the router sends the question down the
    normal retrieval path. `structured_rows` records that rows existed, so
    verification will not auto-approve an "insufficient information" answer.
    """
    query, sid = state["query"], state.get("subject_id")
    # One resolved admission: read the question without its admission phrase and
    # ask the table for that admission's rows only. Otherwise exactly as before.
    scoped = _scoped_lab_request(state)
    if scoped:
        query, hadm_id, mode = scoped
        mode_filler, scope_args = _SCOPED_MODE_FILLER[mode], {"hadm_id": hadm_id}
    else:
        mode = lab_mode(query, state.get("temporal_mode") or detect_temporal_mode(query))
        if mode not in _MODE_FILLER:
            mode = "latest"
        mode_filler, scope_args = _MODE_FILLER[mode], {}
    try:
        resolver = get_lab_resolver()
        itemids, matched = resolver.match(query)
        series = resolver.fetch(sid, itemids, per_lab_cap=LAB_SERIES_CAP, **scope_args) if itemids else []
    except DataSourceNotReady:
        raise                # the configured lab source is unusable: an error, not a miss to retrieve around
    except Exception as e:
        logger.warning(f"[lab_lookup] structured lookup failed ({e}); falling back to retrieval")
        return {"node_trail": _trail(state, "lab_lookup")}

    series = [g for g in series if g.get("values")]
    miss = {"structured_rows": sum(len(g["values"]) for g in series),
            "node_trail": _trail(state, "lab_lookup")}
    series = _disambiguate(series, query, resolver.labels, matched, resolver.labels_for(itemids),
                           filler=_QUESTION_FILLER | mode_filler)
    if any(_endpoint_unclear(g, mode) for g in series):
        logger.info(f"[lab_lookup] no single {mode} numeric result — using retrieval")
        return miss
    if len(series) != 1:
        logger.info(f"[lab_lookup] {len(series)} analyte(s) for {matched or 'no match'} — using retrieval")
        return miss

    grp, label = series[0], "L1"
    vals, name = grp["values"], grp["label"].lower()

    def point(v: dict) -> str:
        return f"{_num(v['valuenum'])}{_uom(v.get('uom') or grp['uom'])} on {v['date']}"

    def history(points: list[dict]) -> str:
        return "; ".join(f"{v['date']} {_num(v['valuenum'])}{_uom(v.get('uom') or grp['uom'])}"
                         for v in points)

    source = f"Source: labevents table, subject {sid}."
    where = ""                                   # scoped answers say which admission the rows are from
    if scoped:
        where = f" during admission {hadm_id}"
        source = f"Source: {PROFILE_SETTINGS['lab_table']} table, subject {sid}, admission {hadm_id}."
    if mode == "latest":
        anchor = vals[-1]
        text = (f"{grp['label']} — {grp['n_total']} recorded value(s){where}, most recent first shown last: "
                f"{history(vals[-LAB_RECENT_POINTS:])}. {source}")
        sentences = [f"The most recent {name}{where} was {point(anchor)}"]
    elif mode == "earliest":
        anchor = vals[0]
        text = (f"{grp['label']} — {grp['n_total']} recorded value(s){where}, earliest shown first: "
                f"{history(vals[:LAB_RECENT_POINTS])}. {source}")
        sentences = [f"The earliest {name}{where} was {point(anchor)}"]
    elif mode in ("min", "max"):
        word = "lowest" if mode == "min" else "highest"
        anchor = (min if mode == "min" else max)(vals, key=lambda v: v["valuenum"])    # the first time it was reached
        times = sum(1 for v in vals if v["valuenum"] == anchor["valuenum"])
        text = (f"{grp['label']} — {len(vals)} numeric value(s){where}, {vals[0]['date']} to {vals[-1]['date']}. "
                f"{word.capitalize()}: {point(anchor)}. Last {min(len(vals), LAB_RECENT_POINTS)} value(s): "
                f"{history(vals[-LAB_RECENT_POINTS:])}. {source}")
        sentences = [f"The {word} {name}{where} was {point(anchor)}"
                     + (f", the first of {times} measurements at that value" if times > 1 else "")]
    else:
        anchor = vals[-1]
        low, high = min(vals, key=lambda v: v["valuenum"]), max(vals, key=lambda v: v["valuenum"])
        skipped = grp.get("n_non_numeric") or 0
        text = (f"{grp['label']} — {len(vals)} numeric value(s) in time order, {vals[0]['date']} to "
                f"{anchor['date']}. First: {point(vals[0])}. Most recent: {point(anchor)}. "
                f"Lowest: {point(low)}. Highest: {point(high)}. "
                f"Last {min(len(vals), LAB_RECENT_POINTS)} value(s): {history(vals[-LAB_RECENT_POINTS:])}. "
                f"{source}")
        if len(vals) == 1:
            sentences = [f"Only one {name} value is recorded{where} ({point(anchor)}), so no trend can be described"]
        else:
            sentences = [
                f"{grp['label']} was measured {len(vals)} times{where} between {vals[0]['date']} and {anchor['date']}"
                + (f" ({skipped} non-numeric result(s) are not included)" if skipped else ""),
                f"The first value was {point(vals[0])} and the most recent was {point(anchor)}",
                f"The lowest value was {point(low)} and the highest was {point(high)}",
                f"Overall, the values {_direction([v['valuenum'] for v in vals])}",
            ]

    ev = [{"chunk_id": -1, "source_type": "lab", "note_type": "lab",
           "charttime": anchor["charttime"], "score": 1.0, "label": label, "text": text}]
    claims = [{"claim": f"{sentence} [{label}].", "label": label, "chunk_id": -1, "verified": True,
               "verification_note": "deterministic: value read directly from labevents"}
              for sentence in sentences]

    answer = " ".join(c["claim"] for c in claims)
    bump("deterministic_answer")
    bump("deterministic_verified", len(claims))
    logger.info(f"[lab_lookup] answered {mode} deterministically, 0 LLM calls")
    return {
        "lab_evidence": ev, "draft_answer": answer, "final_answer": answer, "citations": claims,
        "verification": {"checked": 0, "unsupported": 0, "synthesis_failed": False,
                         "deterministic": len(claims), "llm_checked": 0},
        "needs_human_review": False, "review_status": "auto_approved", "structured_rows": 0,
        "node_trail": _trail(state, "lab_lookup"),
        **({"admission_scope_applied": True} if scoped else {}),      # the rows are that admission's and no other's
    }


def _endpoint_unclear(g: dict, mode: str) -> bool:
    """Is the end of the series this question asks about not one plain number?"""
    first = g.get("older_non_numeric") or g.get("conflicting_earliest")
    last = g.get("newer_non_numeric") or g.get("conflicting_latest")
    return bool(len(g.get("fluids") or ()) > 1
                or len(g["values"]) < g.get("n_total", 0)          # capped: not the whole series
                or (mode != "earliest" and last) or (mode != "latest" and first)
                # a lowest or highest value is only as sure as every result: one "<0.1" could be the real minimum
                or (mode in ("min", "max") and g.get("n_non_numeric"))
                or (mode in ("trend", "min", "max") and len({v.get("uom") for v in g["values"] if v.get("uom")}) > 1))


def _direction(nums: list) -> str:
    """Name a direction only when every consecutive step agrees with it. A
    series that went up and came back down "fluctuated", whatever its ends say."""
    steps = [b - a for a, b in zip(nums, nums[1:])]
    if all(s == 0 for s in steps):
        return "were unchanged across all measurements"
    if all(s >= 0 for s in steps):
        return "rose, with no decrease between consecutive measurements"
    if all(s <= 0 for s in steps):
        return "fell, with no increase between consecutive measurements"
    end = ("higher than" if nums[-1] > nums[0] else "lower than" if nums[-1] < nums[0] else "the same as")
    return (f"fluctuated, with both rises and falls between measurements; "
            f"the most recent value is {end} the first")


def _admissions(subject_id) -> list[tuple]:
    """Every admission for one patient as (admittime, dischtime), oldest first."""
    with engine.connect() as c:
        return [tuple(r) for r in c.execute(sa.text(
            "SELECT admittime, dischtime FROM admissions "
            "WHERE subject_id = :sid ORDER BY admittime, hadm_id"), {"sid": subject_id})]


def encounter_lookup(state: AgentState) -> dict:
    """Answer "how many admissions / when was the first / most recent one" from
    the `admissions` table. Retrieval sees five note chunks; the count of a
    patient's admissions is a row count, and no number of chunks contains it.

    Answers only a question every word of which it understands
    (classify.encounter_intents). Anything else — an extra clause, no rows, an
    admission with no admit time — falls through to retrieval.
    `structured_rows` is set only when the table had rows AND some clause of
    the question asked exactly what it answers; a refusal about an unrelated
    question that merely mentions admissions is left alone.
    """
    sid = state.get("subject_id")
    intents, understood = encounter_intents(state["query"])
    try:
        rows = _admissions(sid)
    except Exception as e:
        logger.warning(f"[encounter_lookup] structured lookup failed ({e}); falling back to retrieval")
        return {"node_trail": _trail(state, "encounter_lookup")}
    if not rows or not understood or any(admit is None for admit, _ in rows):
        logger.info(f"[encounter_lookup] {len(rows)} row(s), understood={understood} — using retrieval")
        declined = understood or structured_admission_clause(state["query"])
        return {"structured_rows": len(rows) if declined else 0,
                "node_trail": _trail(state, "encounter_lookup")}

    def day(t) -> str:
        return str(t)[:10]

    def stay(which: str, row: tuple) -> str:
        admit, disch = row
        return (f"The {which} admission began on {day(admit)}; that stay's discharge date was "
                f"{day(disch) if disch else 'not recorded'}")

    n, label = len(rows), "A1"
    sentences = []
    if "count" in intents:
        sentences.append(f"The patient has {n} recorded hospital admission{'' if n == 1 else 's'}")
    if "earliest" in intents:
        sentences.append(stay("earliest", rows[0]))
    if "latest" in intents:
        sentences.append(stay("most recent", rows[-1]))

    ev = [{"chunk_id": -1, "source_type": "admissions", "note_type": "admissions",
           "charttime": str(rows[-1][0]), "score": 1.0, "label": label,
           "text": (f"{n} admission(s), admitted -> discharged, oldest first: "
                    + "; ".join(f"{day(a)} -> {day(d) if d else 'not recorded'}" for a, d in rows)
                    + f". Source: admissions table, subject {sid}.")}]
    claims = [{"claim": f"{sentence} [{label}].", "label": label, "chunk_id": -1, "verified": True,
               "verification_note": "deterministic: read directly from the admissions table"}
              for sentence in sentences]
    answer = " ".join(c["claim"] for c in claims)
    bump("deterministic_answer")
    bump("deterministic_verified", len(claims))
    logger.info(f"[encounter_lookup] answered {sorted(intents)} deterministically, 0 LLM calls")
    return {
        "encounter_evidence": ev, "draft_answer": answer, "final_answer": answer, "citations": claims,
        "verification": {"checked": 0, "unsupported": 0, "synthesis_failed": False,
                         "deterministic": len(claims), "llm_checked": 0},
        "needs_human_review": False, "review_status": "auto_approved", "structured_rows": 0,
        "node_trail": _trail(state, "encounter_lookup"),
    }


def _structured_request(state: AgentState) -> Optional[tuple[str, int]]:
    """(kind, hadm_id) when this question goes to a structured table: the
    profile enables SQL paths, one admission was resolved, and the wording
    clearly asks for that table. Otherwise None, and nothing is queried."""
    scope = state.get("admission_scope") or {}
    if not PROFILE_SETTINGS["sql_paths"] or scope.get("status") != "resolved" or state.get("subject_id") is None:
        return None
    kind = structured.structured_kind(scope.get("retrieval_query") or state.get("query") or "")
    return (kind, scope["hadm_id"]) if kind else None


def _drug_order_request(state: AgentState) -> Optional[tuple[str, int]]:
    """(drug as the question names it, hadm_id) when the question asks for one
    named drug's inpatient order, dose, route or doses per 24 hours in a resolved
    admission, under a profile with SQL paths. Otherwise None; nothing is queried."""
    scope = state.get("admission_scope") or {}
    if not PROFILE_SETTINGS["sql_paths"] or scope.get("status") != "resolved" or state.get("subject_id") is None:
        return None
    drug = structured.drug_order_question(scope.get("retrieval_query") or state.get("query") or "")
    return (drug, scope["hadm_id"]) if drug else None


def structured_lookup(state: AgentState) -> dict:
    """Answer an inpatient-orders, coded-diagnoses or coded-procedures question
    for the resolved admission straight from its table. No rows is a miss and
    the question goes to note retrieval. Unusable structured data is an error
    and is raised, never retrieved around.

    A question about one named drug reads the same admission's orders and keeps
    only that drug's. A name that matches no drug there, or more than one, is a
    miss: no other drug is offered in its place."""
    request, drug = _structured_request(state), None
    if request:
        kind, hadm_id = request
    else:
        (drug, hadm_id), kind = _drug_order_request(state), "prescriptions"
    sid = state["subject_id"]
    try:
        rows = structured.fetch(kind, sid, hadm_id)
    except DataSourceNotReady:
        raise
    except SQLAlchemyError as e:
        logger.warning(f"[structured_lookup] {kind} lookup failed ({type(e).__name__}); falling back to retrieval")
        return {"node_trail": _trail(state, "structured_lookup")}
    if not rows:
        logger.info(f"[structured_lookup] no {kind} rows for hadm_id={hadm_id} — using retrieval")
        return {"node_trail": _trail(state, "structured_lookup")}
    if drug:
        rows, candidates = structured.match_drug(drug, rows)
        if not rows:
            # Several drugs fit the name: their orders exist, so a "no information" answer from the notes must not be auto-approved.
            logger.info(f"[structured_lookup] named drug matched {len(candidates)} drug(s) for hadm_id={hadm_id} — using retrieval")
            return {"node_trail": _trail(state, "structured_lookup"), **({"structured_rows": len(candidates)} if candidates else {})}
    sentences, evidence = structured.render(kind, rows, sid, hadm_id, **({"drug": drug} if drug else {}))
    label, table = evidence["label"], evidence["source_type"]
    claims = [{"claim": f"{sentence} [{label}].", "label": label, "chunk_id": -1, "verified": True,
               "verification_note": f"deterministic: read directly from the {table} table"}
              for sentence in sentences]
    answer = "\n".join(c["claim"] for c in claims)
    bump("deterministic_answer")
    bump("deterministic_verified", len(claims))
    logger.info(f"[structured_lookup] answered {kind} for hadm_id={hadm_id} from {len(rows)} row(s), 0 LLM calls")
    return {
        "structured_evidence": [evidence], "draft_answer": answer, "final_answer": answer, "citations": claims,
        "verification": {"checked": 0, "unsupported": 0, "synthesis_failed": False,
                         "deterministic": len(claims), "llm_checked": 0},
        "needs_human_review": False, "review_status": "auto_approved", "structured_rows": 0,
        "admission_scope_applied": True,             # the rows are that admission's and no other's
        "node_trail": _trail(state, "structured_lookup"),
    }


def _num(v) -> str:
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def _uom(unit: str) -> str:
    """Render a unit the way the notes render it, so a value quoted from the
    table reads identically to the same value quoted from a discharge summary
    ("7.1%", not "7.1 %")."""
    unit = (unit or "").strip()
    if not unit:
        return ""
    return unit if unit == "%" else f" {unit}"


# Words a plain "latest <analyte>" question may contain besides the analyte.
# Anything else ("urine", "clearance", "ionized", "and", "supplements") means
# the question asks something narrower or different, and the verified shortcut
# must not answer it. Unknown words fail toward retrieval, never toward a value.
_QUESTION_FILLER = frozenset("""
    what what's whats was were is are the a an his her their this that patient patient's patients s
    most recent recently latest last current currently newest value values level levels result results
    reading readings lab labs measured recorded documented measurement for of on in at me tell show give
    blood serum plasma
""".split())

# What each kind of question may say on top of that. Keyed by temporal mode, so
# a word that asks for something else ("latest creatinine trend") is not filler.
_MODE_FILLER = {
    "latest": frozenset(),
    "earliest": frozenset("earliest oldest first initial known".split()),
    "trend": frozenset("""
        how did does has have been change changed changes changing over time trend trends trending
        trended available record records course whole entire
        increase increased increasing decrease decreased decreasing or
    """.split()),
}


def _disambiguate(series: list[dict], query: str, known_labels: list[str],
                  concepts=(), resolved_labels=None, filler=_QUESTION_FILLER) -> list[dict]:
    """Narrow a resolver match down to the ONE analyte the question asked about,
    or to nothing when the question is not a plain request for one analyte.

    LabResolver.match is a substring matcher built for retrieval, where pulling
    in a neighbouring analyte only adds harmless context. As an *answer* path
    — one whose result is marked verified with no model in the loop — an
    over-match is a wrong answer. So the shortcut answers only a question it
    fully accounts for:

      every word is filler, a time word, or part of the analyte it named or a
      synonym the resolver matched ("blood sugar", "bun"). "urine creatinine",
      "creatinine clearance", "creatinine and BUN", "potassium supplements"
      all leave a word over and go to retrieval.

      every synonym the resolver matched must be in the answer: "INR, PTT"
      and "BUN/creatinine" ask for two analytes and get none, not one.

      a question that names no label is ambiguous when its synonym resolves
      to several dictionary analytes ("cholesterol": total, HDL, LDL), even if
      this patient only has rows for one of them.

      a dictionary label counts as named when all its words appear in the
      question, in any order ("urine creatinine" names "Creatinine, Urine").
      Labels made only of filler words (MIMIC's urine dipstick "Blood") name
      nothing.
      The most specific named label wins (A1c beats hemoglobin), and if this
      patient has no rows for it, return nothing — retrieval and synthesis
      will say it is not documented. A label of one or two letters ("H", "I",
      "CR") names nothing unless the question contains it as a word.
    """
    # drop the possessive first: "the patient's hemoglobin" must not name "Hemoglobin S"
    q = label_tokens(re.sub(r"['\u2019]s\b", "", query))
    keys = {label_tokens(label) for label in known_labels}
    named = {k for k in keys if k and k <= q and not k <= filler}
    named = {k for k in named if not any(o > k for o in named)}
    concepts = list(concepts or ())
    explained = set(filler).union(*named, *(label_tokens(c) for c in concepts))
    if not q <= explained:
        return []                      # a word the shortcut does not understand
    if not named:
        # The question used a synonym ("blood sugar"), not a label. Trust it
        # only when it resolves to exactly one analyte in the dictionary.
        if resolved_labels is not None and len({label_tokens(l) for l in resolved_labels}) != 1:
            return []
        chosen = series if len(series) == 1 else []
    else:
        have = {label_tokens(g["label"]) for g in series}
        if not named <= have:
            return []                  # asked for something this patient has no rows for
        chosen = [g for g in series if label_tokens(g["label"]) in named]
    for concept in concepts:           # every analyte asked for must be in the answer
        words = SYNONYMS.get(concept)
        if not all(any(kw in g["label"].lower() for kw in words) if words
                   else label_tokens(concept) <= label_tokens(g["label"]) for g in chosen):
            return []
    return chosen


def patient_retrieval(state: AgentState) -> dict:
    t0 = time.perf_counter()
    retriever, _ = get_retrievers()
    add_timing("retriever_load_ms", (time.perf_counter() - t0) * 1000)   # ~0 once loaded
    with tracing.span("patient_retrieval", subject_id=state.get("subject_id"),
                      query=state["query"]) as s:
        # Admission scope: only for profiles that enable it, and only when the
        # question resolved to exactly one admission. Otherwise the call below
        # is the patient-wide search it has always been.
        scope = state.get("admission_scope") or {}
        scoped = PROFILE_SETTINGS["admission_scope"] and scope.get("status") == "resolved"
        admission = ({"hadm_id": scope["hadm_id"], "stay_window": load_stay_window(scope["hadm_id"])}
                     if scoped else {})
        results = retriever.search(
            query=scope["retrieval_query"] if scoped else state["query"],
            subject_id=state.get("subject_id"),
            temporal_filter=state.get("temporal_mode") or "auto",
            top_k=PATIENT_TOP_K,
            **admission,
        )
        # The retriever chooses evidence by relevance plus a preference for the
        # newest (or oldest) date. The chosen few are then shown in time order,
        # so S1 is the newest record of a "latest" question (oldest for "earliest").
        mode = state.get("temporal_mode") or detect_temporal_mode(state["query"])
        if mode in ("latest", "earliest"):
            dated = sorted((r for r in results if r.charttime), key=lambda r: str(r.charttime),
                           reverse=(mode == "latest"))
            results = dated + [r for r in results if not r.charttime]
        ev = _to_evidence(results, "S", "note")
        if s is not None:
            s.update(output={"n": len(ev), "chunk_ids": [e["chunk_id"] for e in ev],
                             "top_score": ev[0]["score"] if ev else None})
    logger.info(f"[patient_retrieval] {len(ev)} chunks")
    out = {"patient_evidence": ev, "node_trail": _trail(state, "patient_retrieval")}
    if scoped:
        out["admission_scope_applied"] = True
    return out


def guideline_retrieval(state: AgentState) -> dict:
    _, gr = get_retrievers()
    with tracing.span("guideline_retrieval", query=state["query"]) as s:
        results = gr.search(state["query"], top_k=GUIDELINE_TOP_K)
        ev = _to_evidence(results, "G", "guideline")
        if s is not None:
            s.update(output={"n": len(ev), "chunk_ids": [e["chunk_id"] for e in ev],
                             "top_score": ev[0]["score"] if ev else None})
    logger.info(f"[guideline_retrieval] {len(ev)} chunks")
    return {"guideline_evidence": ev, "node_trail": _trail(state, "guideline_retrieval")}

def literature_retrieval(state: AgentState) -> dict:
    """Query external evidence sources. Every outbound call passes the gate.

    The query sent outward is NEVER the user's question verbatim — it is a
    concept query generated from it, then gate-checked. If the gate blocks,
    we retry once under a stricter instruction rather than failing the run.
    """
    if LITERATURE_TOOL is None:
        # Nothing to query, so nothing is sent anywhere and no model call is
        # spent on a search string. The answer says so only when literature is
        # what the question asked for; as a fallback for an empty guideline
        # search there is nothing to announce.
        logger.info("[literature_retrieval] no literature backend in this build — skipped")
        return {"literature_evidence": [],
                "literature_unavailable": state.get("query_type") == "literature",
                "node_trail": _trail(state, "literature_retrieval")}

    gate = _gate_for(state)
    query = state["query"]
    egress = list(state.get("egress_log", []) or [])

    def concept(strict: bool = False) -> str:
        sys_prompt = prompts.CONCEPT_SYSTEM
        if strict:
            sys_prompt += ("\n\nYour previous attempt was rejected for containing patient data. "
                           "Use ONLY the disease or drug name and one general qualifier.")
        try:
            raw = chat_for("concept", [{"role": "system", "content": sys_prompt},
                                       {"role": "user", "content": query}], max_retries=1)
            return (json.loads(raw).get("concept_query") or "").strip()
        except Exception as e:
            logger.warning(f"concept extraction failed: {e}")
            return ""

    tool_output, attempts = {}, []
    for strict in (False, True):
        cq = concept(strict)
        if not cq:
            continue
        attempts.append(cq)
        try:
            out = call_external(gate, "search_literature",
                                {"query": cq}, LITERATURE_TOOL)
        except Exception as e:
            # Offline, timed out, or an unreadable reply: no literature, and the
            # answer says so. The approved query is not retried or logged.
            logger.warning(f"[literature_retrieval] backend failed ({type(e).__name__}); continuing without it")
            egress.append(gate.log[-1])
            break
        egress.append(gate.log[-1])
        rec = gate.log[-1]
        # Hash and rule only. The gate's no-retention rule holds inside the
        # observability layer too — a blocked payload must not reappear here.
        with tracing.span("egress_check", tool=rec["tool"], rule=rec["rule"],
                          allowed=rec["allowed"], sha256=rec["payload_sha256"][:16]) as s:
            if s is not None:
                s.update(output={"allowed": rec["allowed"], "rule": rec["rule"]})
        if isinstance(out, dict) and out.get("blocked"):
            logger.warning(f"[literature_retrieval] egress blocked ({out['rule']}); "
                           f"{'giving up' if strict else 'retrying stricter'}")
            continue
        tool_output = out
        break

    ev = _to_literature_evidence(tool_output)
    logger.info(f"[literature_retrieval] {len(ev)} results; "
                f"{sum(1 for r in egress if not r['allowed'])} blocked so far")
    return {"literature_evidence": ev, "egress_log": egress,
            "literature_unavailable": not ev and state.get("query_type") == "literature",
            "node_trail": _trail(state, "literature_retrieval")}


def synthesis(state: AgentState) -> dict:
    pt = state.get("patient_evidence", []) or []
    gl = state.get("guideline_evidence", []) or []
    lit = _citable_literature(state)

    if not pt and not gl and not lit:
        return {"draft_answer": _with_literature_notice(
                    state, "The available records do not contain enough information to answer this."),
                "citations": [], "node_trail": _trail(state, "synthesis")}

    user = prompts.build_synthesis_prompt(state["query"], pt, gl, lit,
                                          literature_unavailable=bool(state.get("literature_unavailable")))
    # MAIN earns its cost on longitudinal reasoning, comparisons and anything
    # weighing general recommendations against this patient. A single-fact
    # lookup over a handful of chunks does not need it. Presence of guideline
    # or literature evidence forces MAIN: rule 5 (never present a guideline as
    # something the patient received) is the rule a small model drops first.
    simple = (state.get("query_complexity") == "simple"
              and state.get("classified_by") == "rules"
              and not gl and not lit)
    role = "synthesis_simple" if simple else "synthesis_complex"
    try:
        answer = chat_for(role, [{"role": "system", "content": prompts.SYNTHESIS_SYSTEM},
                                 {"role": "user", "content": user}])
    except Exception as e:
        logger.error(f"synthesis failed: {e}")
        return {"draft_answer": "", "citations": [],
                "errors": [f"synthesis: {e}"],   # reducer appends; see AgentState
                "node_trail": _trail(state, "synthesis")}

    ev_all = pt + gl + lit
    # Formatting repair first: a marker the model put on a line of its own gets
    # moved onto the sentence it was written for, so the splitter does not read
    # one claim as an uncited sentence plus a meaningless fragment. Then strip
    # hallucinated labels, THEN validate — so draft_answer and the claim list
    # describe the same text. Validating before stripping left the citations
    # carrying sentences that no longer matched the answer the user is shown.
    answer = citations.normalize_orphan_citations(answer)
    pre = citations.validate(answer, ev_all)
    if pre["bad_labels"]:
        logger.warning(f"[synthesis] hallucinated labels {pre['bad_labels']} — stripped")
        answer = citations.strip_bad_labels(answer, ev_all)
    report = citations.validate(answer, ev_all)
    report["bad_labels"] = pre["bad_labels"]

    # `label`/`chunk_id` stay single-valued for the API contract; `labels` keeps
    # EVERY citation the sentence carried, because verification must check a
    # claim against the union of its sources, not just the first one.
    cites = [{
        "claim": c["claim"],
        "label": c["valid_labels"][0] if c["valid_labels"] else "",
        "labels": list(c["valid_labels"]),
        "chunk_id": next((e["chunk_id"] for e in ev_all if e["label"] == (c["valid_labels"] or [None])[0]), -1),
        "verified": False,
        "verification_note": "",
    } for c in report["claims"]]

    logger.info(f"[synthesis] role={role} {report['n_claims']} claims, "
                f"cite_rate={report['cite_rate']:.0%}, bad={report['bad_labels']}")
    return {"draft_answer": _with_literature_notice(state, answer), "citations": cites,
            "verification": {"citation_report": {k: v for k, v in report.items() if k != "claims"},
                             "synthesis_role": role},
            "node_trail": _trail(state, "synthesis")}


def verification(state: AgentState) -> dict:
    """Deterministic checks first; one batched FAST call for what is left.

    This used to be one MAIN generation per citation, run serially. The code
    path that decides human review is unchanged — unsupported > 0 still routes
    to a clinician — only the way a verdict is reached has changed.

    Every claim ends in exactly one of three states, recorded per claim in
    `verification["claims"]` so an audit can say WHY a run escalated:
      supported    deterministic anchors matched, or the model said so
      unsupported  no citation, a model verdict of partial/unsupported, or no
                   verdict came back at all
    A claim never silently disappears, and nothing turns "unresolved" into
    "supported".
    """
    ev = {e["label"]: e for e in (state.get("patient_evidence", []) or []) +
                                  (state.get("guideline_evidence", []) or []) +
                                  _citable_literature(state) +
                                  (state.get("lab_evidence", []) or []) +
                                  (state.get("encounter_evidence", []) or []) +
                                  (state.get("structured_evidence", []) or [])}
    cites = state.get("citations", []) or []
    out: list[dict] = [dict(c) for c in cites]
    unsupported = 0
    pending: list[dict] = []          # claims the deterministic pass could not settle
    trace: list[dict] = []            # per-claim audit record (no source text)

    def _labels_of(c: dict) -> list[str]:
        """Every citation the claim carries. `label` stays the primary one for
        the API; `labels` is what verification must actually check against."""
        ls = [l for l in (c.get("labels") or []) if l in ev]
        if not ls and c.get("label") in ev:
            ls = [c["label"]]
        return ls

    # A correctly declined answer is not an unsupported claim. Synthesis is
    # instructed to emit this exact sentence, uncited, when the evidence does
    # not answer the question — and the uncited sentence was then failing
    # verification and escalating every "not documented" answer to a clinician.
    # The same sentence produced on the no-evidence path already auto-approved,
    # so the two routes disagreed about identical output.
    draft_now = (state.get("draft_answer") or "").strip()
    refusal = bool(draft_now) and citations.validate(draft_now, list(ev.values()))["is_refusal"]
    # "Pure" means every claim IS the decline. It used to mean "contains the
    # decline and cites nothing", which approved any uncited sentence the model
    # wrote next to it — a patient-specific statement finalized with no check.
    # Such an answer now takes the normal pass below, where an uncited claim is
    # unsupported and goes to a reviewer.
    pure_refusal = refusal and all(citations.is_refusal_claim(c.get("claim")) for c in out)
    # ...unless a structured lookup found rows for this very question and only
    # declined to phrase the answer. Then "the records do not contain enough
    # information" is a statement about five note chunks, not about the record,
    # and a clinician has to see it.
    contradicted = pure_refusal and bool(state.get("structured_rows"))

    if contradicted:
        note = (f"declined, but a structured lookup found {state['structured_rows']} row(s) "
                "for this question")
        out = out or [{"claim": draft_now, "label": "", "chunk_id": -1}]
        for i, c in enumerate(out):
            out[i] = {**c, "verified": False, "verification_note": note}
            trace.append({"i": i, "labels": [], "stage": "refusal", "verdict": "unsupported",
                          "reason": note, "final": "unsupported"})
        unsupported, deterministic, checked, pure_refusal = len(out), 0, 0, False
    elif pure_refusal:
        for i, c in enumerate(out):
            out[i] = {**c, "verified": True,
                      "verification_note": "declined: evidence does not answer the question"}
            trace.append({"i": i, "labels": [], "stage": "refusal", "verdict": "declined",
                          "reason": "answer correctly states the record does not support an answer",
                          "final": "supported"})
        deterministic, checked = len(out), 0
    else:
        # --- pass 1: no model -----------------------------------------------
        for i, c in enumerate(out):
            labels = _labels_of(c)
            if not labels:
                # Deterministically unsupported: nothing valid is cited.
                out[i] = {**c, "verified": False, "verification_note": "no valid citation"}
                unsupported += 1
                trace.append({"i": i, "labels": [], "stage": "deterministic",
                              "verdict": "unsupported", "reason": "no valid citation",
                              "final": "unsupported"})
                continue
            src_text = "\n\n".join(ev[l]["text"] for l in labels)
            verdict, note = verify_util.deterministic_verdict(c["claim"], src_text, labels)
            if verdict == "supported":
                out[i] = {**c, "verified": True, "verification_note": note}
                trace.append({"i": i, "labels": labels, "stage": "deterministic",
                              "verdict": "supported", "reason": note, "final": "supported"})
            elif verdict == "unsupported":
                # The claim contradicts itself; no source can support it.
                out[i] = {**c, "verified": False, "verification_note": note}
                unsupported += 1
                trace.append({"i": i, "labels": labels, "stage": "deterministic",
                              "verdict": "unsupported", "reason": note, "final": "unsupported"})
            else:
                pending.append({"i": i, "claim": c["claim"], "labels": labels,
                                "sources": [ev[l]["text"] for l in labels],
                                "det_reason": note})

        deterministic = sum(1 for t in trace if t["final"] == "supported")
        checked = len(pending)

        # --- pass 2: one batched call for the remainder ----------------------
        # Identical claims cited identically get one verdict, not one each.
        by_key: dict[tuple, list[dict]] = {}
        for it in pending:
            by_key.setdefault((it["claim"], tuple(it["labels"])), []).append(it)
        unique = [group[0] for group in by_key.values()]

        for batch in (unique[k:k + verify_util.MAX_BATCH]
                      for k in range(0, len(unique), verify_util.MAX_BATCH)):
            prompt, mapping = verify_util.build_batch_prompt(batch)
            try:
                raw = chat_for("verify", [
                    {"role": "system", "content": prompts.VERIFY_BATCH_SYSTEM},
                    {"role": "user", "content": prompt},
                    # room for one compact JSON verdict per claim, bounded so a
                    # large batch cannot blow past the role's budget
                ], max_tokens=min(480, max(160, 90 * len(batch))))
                verdicts = verify_util.parse_batch(raw, mapping)
            except Exception as e:
                logger.error(f"[verification] batch call failed: {e}")
                verdicts = {}
            for it in batch:
                verdict, note = verdicts.get(it["i"], ("unsupported", "no verdict returned"))
                ok = verdict == "supported"
                for twin in by_key[(it["claim"], tuple(it["labels"]))]:
                    if not ok:
                        unsupported += 1
                    out[twin["i"]] = {**out[twin["i"]], "verified": ok,
                                      "verification_note": f"{verdict}: {note}"}
                    trace.append({"i": twin["i"], "labels": twin["labels"], "stage": "fast_model",
                                  "verdict": verdict, "reason": note,
                                  "det_reason": twin["det_reason"],
                                  "final": "supported" if ok else "unsupported"})

    bump("deterministic_verified", deterministic)
    bump("llm_verified", checked)
    prior = state.get("verification", {}) or {}
    trace.sort(key=lambda t: t["i"])

    # An empty draft is a FAILURE, not a clean bill of health. Without this, a
    # synthesis outage produced zero citations, `unsupported > 0` was False, and
    # the run reported "auto_approved, 0/0 verified" with an empty answer —
    # indistinguishable from success to every caller.
    draft = (state.get("draft_answer") or "").strip()
    synthesis_failed = not draft and not state.get("final_answer")
    errors: list[str] = []          # reducer appends; see AgentState
    if synthesis_failed:
        logger.error("[verification] no draft answer to verify — failing the run")
        errors.append("verification: no draft answer produced (synthesis failed)")

    logger.info(f"[verification] {len(out)} claims: {deterministic} deterministic, "
                f"{checked} via model, {unsupported} unsupported"
                f"{' (declined answer)' if pure_refusal else ''}")
    return {
        "citations": out,
        "verification": {**prior, "checked": checked, "unsupported": unsupported,
                         "deterministic": deterministic, "llm_checked": checked,
                         "refusal": pure_refusal, "claims": trace,
                         "synthesis_failed": synthesis_failed},
        "errors": errors,
        "needs_human_review": unsupported > 0 or synthesis_failed,
        "review_status": ("failed" if synthesis_failed
                          else "pending" if unsupported > 0
                          else "auto_approved"),
        "node_trail": _trail(state, "verification"),
    }

REJECTED_ANSWER = ("A clinician reviewed the draft answer to this question and rejected it. "
                   "No answer is released.")


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

    # A failed synthesis reaches here with nothing to adjudicate. Returning
    # "auto_approved" would launder the failure back into a success.
    if (state.get("verification", {}) or {}).get("synthesis_failed"):
        logger.error("[human_review] synthesis failed upstream — nothing to review")
        return {"review_status": "failed", "node_trail": _trail(state, "human_review")}

    if not flagged:
        return {"review_status": "auto_approved", "node_trail": _trail(state, "human_review")}

    ev = {e["label"]: e for e in (state.get("patient_evidence", []) or []) +
                                  (state.get("guideline_evidence", []) or []) +
                                  (state.get("literature_evidence", []) or []) +
                                  (state.get("lab_evidence", []) or []) +
                                  (state.get("encounter_evidence", []) or []) +
                                  (state.get("structured_evidence", []) or [])}

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

    # A reviewer who rejects the draft rejects all of it: nothing the model
    # wrote is released, whatever was said about the individual claims.
    if any((d or {}).get("action") == "reject" for d in decisions):
        note = next(((d or {}).get("note", "") for d in decisions if (d or {}).get("action") == "reject"), "")
        for idx, _ in flagged:
            updated[idx]["verification_note"] = f"rejected by reviewer: {note}" if note else "rejected by reviewer"
        logger.info("[human_review] draft rejected by reviewer")
        return {
            "citations": updated,
            "human_decisions": list(state.get("human_decisions", []))
                               + [{"index": idx, "action": "reject", "note": note} for idx, _ in flagged],
            "final_answer": REJECTED_ANSWER, "review_status": "rejected", "needs_human_review": False,
            "node_trail": _trail(state, "human_review"),
        }

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
    """Set final_answer when no review was needed. After a human decision the
    reviewer's result stands even when it is empty: a draft whose every claim
    was struck must not be released because "" looks like "not set"."""
    decided = state.get("final_answer") or state.get("review_status") in ("reviewed", "escalated", "rejected")
    answer = (state.get("final_answer") or "") if decided else state.get("draft_answer", "")
    if state.get("review_status") != "rejected":       # a rejected draft releases nothing at all
        answer = _with_literature_notice(state, answer)
    if decided and answer == (state.get("final_answer") or ""):
        return {"node_trail": _trail(state, "finalize")}
    return {"final_answer": answer, "node_trail": _trail(state, "finalize")}


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
    if _structured_request(state) or _drug_order_request(state):     # before the admissions path: "how many doses" is not a count of admissions
        return "structured_lookup"
    if _scoped_lab_request(state):               # a lab question about one resolved admission, before the admissions path
        return "lab_lookup"
    if DETERMINISTIC_LABS and wants_encounter_lookup(
            _decision_of(state), state.get("query") or "", state.get("subject_id")):
        return "encounter_lookup"
    if DETERMINISTIC_LABS and wants_deterministic_lab(
            _decision_of(state), lab_mode(state.get("query") or "", state.get("temporal_mode") or "all"),
            state.get("subject_id")):
        return "lab_lookup"
    return "patient_retrieval"


def _decision_of(state: AgentState):
    """Rebuild the classifier Decision from what triage stored, so routing does
    not re-run (or second-guess) the classification."""
    from src.agents.classify import Decision
    return Decision(query_type=state.get("query_type", "chart_review"),
                    complexity=state.get("query_complexity", "complex"),
                    confident=state.get("classified_by") == "rules",
                    reason="from_state")


def route_after_lab_lookup(state: AgentState) -> str:
    """A structured hit is already a finished, cited, verified answer. A miss
    falls through to the normal retrieval path with nothing lost."""
    return "finalize" if state.get("lab_evidence") else "patient_retrieval"


def route_after_structured_lookup(state: AgentState) -> str:
    return "finalize" if state.get("structured_evidence") else "patient_retrieval"


def route_after_encounter_lookup(state: AgentState) -> str:
    return "finalize" if state.get("encounter_evidence") else "patient_retrieval"


def route_after_patient(state: AgentState) -> str:
    # Only fetch guidelines when the question is actually asking what
    # should be done. A plain chart lookup does not need them.
    return "guideline_retrieval" if state.get("query_type") in ("guideline_check", "literature") else "synthesis"

def route_after_verification(state: AgentState) -> str:
    return "human_review" if state.get("needs_human_review") else "finalize"

def route_after_guidelines(state: AgentState) -> str:
    # Literature is consulted when explicitly asked for, or as a fallback
    # when the local guideline corpus came back empty.
    if state.get("query_type") == "literature" or not state.get("guideline_evidence"):
        return "literature_retrieval"
    return "synthesis"

def _builder() -> StateGraph:
    """The graph's nodes and edges, uncompiled, so a test can compile it
    against an in-memory checkpointer and exercise interrupt/resume for real."""
    b = StateGraph(AgentState)
    b.add_node("triage", triage)
    b.add_node("patient_retrieval", patient_retrieval)
    b.add_node("guideline_retrieval", guideline_retrieval)
    b.add_node("synthesis", synthesis)
    b.add_node("verification", verification)
    b.add_node("refuse", refuse)
    b.add_node("literature_retrieval", literature_retrieval)
    b.add_node("lab_lookup", lab_lookup)
    b.add_node("encounter_lookup", encounter_lookup)
    b.add_node("structured_lookup", structured_lookup)

    b.add_edge(START, "triage")
    b.add_conditional_edges("triage", route_from_triage,
                            ["patient_retrieval", "guideline_retrieval", "lab_lookup",
                             "encounter_lookup", "structured_lookup", "refuse"])
    b.add_conditional_edges("structured_lookup", route_after_structured_lookup,
                            ["finalize", "patient_retrieval"])
    b.add_conditional_edges("encounter_lookup", route_after_encounter_lookup,
                            ["finalize", "patient_retrieval"])
    b.add_conditional_edges("lab_lookup", route_after_lab_lookup,
                            ["finalize", "patient_retrieval"])
    b.add_conditional_edges("patient_retrieval", route_after_patient,
                            ["guideline_retrieval", "synthesis"])
    b.add_conditional_edges("guideline_retrieval", route_after_guidelines,
                            ["literature_retrieval", "synthesis"])
    b.add_edge("literature_retrieval", "synthesis")
    b.add_edge("synthesis", "verification")
    b.add_node("human_review", human_review)
    b.add_node("finalize", finalize)
    # verification routes ONLY through the conditional edge. A static
    # `add_edge("verification", END)` used to sit alongside this, left over from
    # the pre-human-review topology, which made verification fan out to END and
    # to the branch at the same time — parallelism this state schema has no
    # reducers to survive.
    b.add_conditional_edges("verification", route_after_verification,
                            ["human_review", "finalize"])
    b.add_edge("human_review", "finalize")
    # refuse had no outgoing edge at all: it terminated implicitly and never set
    # final_answer, so callers reading state["final_answer"] got "" on refusal.
    b.add_edge("refuse", "finalize")
    b.add_edge("finalize", END)
    return b


def build_graph(setup: bool = False, validate_checkpoints: bool = True):
    """Compile the canonical graph without implicit schema mutation.

    Checkpoint tables are created by ``python -m src.storage.checkpoints``.
    ``setup=True`` and LUMEN_CHECKPOINT_AUTO_SETUP=1 remain explicit defensive
    fallbacks for controlled recovery, not the normal request path.
    """
    auto_setup = os.environ.get("LUMEN_CHECKPOINT_AUTO_SETUP", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if validate_checkpoints and not (setup or auto_setup):
        status = checkpoint_schema_status()
        if not status["ready"]:
            raise RuntimeError(
                "LangGraph checkpoint schema is missing "
                f"{status['missing']}; run `python -m src.storage.checkpoints`"
            )
    pool = ConnectionPool(
        conninfo=_dsn(), max_size=5,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    checkpointer = PostgresSaver(pool)
    _pools.append(pool)
    if setup or auto_setup:
        checkpointer.setup()

    return _builder().compile(checkpointer=checkpointer), checkpointer

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
