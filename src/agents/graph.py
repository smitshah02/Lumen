"""
Lumen Agent Graph
=================
Day 2: real node bodies and conditional routing.

    triage ──> patient_retrieval ──> guideline_retrieval ──> synthesis ──> verification
       │  │           │                       ▲
       │  │           └───────────────────────┘ (skipped unless needed)
       │  └──> lab_lookup ──> finalize            (structured hit: no LLM at all)
       │              └──────> patient_retrieval  (miss: normal path)
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
import json
import logging
from typing import Optional

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from src.storage import engine
from src.storage.checkpoints import checkpoint_schema_status
from src.agents.state import AgentState
from src.agents import prompts, citations, verify as verify_util
from src.agents.classify import classify, wants_deterministic_lab
from src.llm.local_client import chat_for   # every node call goes through a ROLE
from src.obs.logging import bump
from src.retrieval.hybrid_retriever_v2 import HybridRetriever, detect_temporal_mode
from src.retrieval.guideline_retriever import GuidelineRetriever
from langgraph.types import Command, interrupt
from src.safety.egress_gate import EgressGate, call_external
from src.safety import stub_tools
from src.obs import tracing

logger = logging.getLogger(__name__)

PATIENT_TOP_K = 5      # keep the synthesis prompt inside num_ctx on 16GB
GUIDELINE_TOP_K = 3

# The structured-lab shortcut answers "what was the most recent <analyte>"
# straight from labevents. Set LUMEN_DETERMINISTIC_LABS=0 to force every
# question back through retrieval + synthesis (used for A/B latency runs).
DETERMINISTIC_LABS = os.environ.get("LUMEN_DETERMINISTIC_LABS", "1").strip() not in ("0", "false", "no")
LAB_RECENT_POINTS = 4   # values rendered per analyte as citable evidence

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
            "source_type": source_type,
            "text": (r.context_text or r.chunk_text),
            "charttime": r.charttime,
            "note_type": r.note_type,
            "score": round(float(r.final_score), 4),
            "label": f"{prefix}{i}",
        })
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
    return gate


def _to_literature_evidence(results: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(results, 1):
        out.append({
            "chunk_id": -1,
            "source_type": "literature",
            "text": f"{r.get('title', '')} ({r.get('pmid') or r.get('nct_id', '')})",
            "charttime": None,
            "note_type": "literature",
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
    return {"query_type": qtype, "temporal_mode": temporal, "query_complexity": complexity,
            "classified_by": "rules" if d.confident else "fast_model",
            "node_trail": _trail(state, "triage")}


def lab_lookup(state: AgentState) -> dict:
    """Answer a latest-value lab question straight from `labevents`.

    This is not a shortcut around grounding: the value, its unit and its date
    are read from the structured table the notes are generated from, rendered
    as citable [L#] evidence, and the claim is marked verified because the
    number in the answer IS the number in the row. It is a shortcut around
    asking a language model to re-read a number a query can select.

    A miss — no matching analyte, no rows for this patient, or a question that
    does not resolve to exactly one analyte — returns nothing, and the router
    sends the question down the normal retrieval path with nothing lost.
    """
    query, sid = state["query"], state.get("subject_id")
    try:
        resolver = get_lab_resolver()
        itemids, matched = resolver.match(query)
        series = resolver.fetch(sid, itemids) if itemids else []
    except Exception as e:
        logger.warning(f"[lab_lookup] structured lookup failed ({e}); falling back to retrieval")
        return {"node_trail": _trail(state, "lab_lookup")}

    series = [g for g in series if g.get("values")]
    series = _disambiguate(series, query, resolver.labels)
    if len(series) != 1:
        logger.info(f"[lab_lookup] {len(series)} analyte(s) for {matched or 'no match'} — using retrieval")
        return {"node_trail": _trail(state, "lab_lookup")}

    ev, claims = [], []
    for i, grp in enumerate(series, 1):
        label = f"L{i}"
        recent = grp["values"][-LAB_RECENT_POINTS:]
        uom = _uom(grp["uom"])
        latest = recent[-1]
        history = "; ".join(f"{v['date']} {_num(v['valuenum'])}{uom}" for v in recent)
        ev.append({
            "chunk_id": -1, "source_type": "lab", "note_type": "lab",
            "charttime": latest["charttime"], "score": 1.0, "label": label,
            "text": (f"{grp['label']} — {grp['n_total']} recorded value(s), most recent first shown last: "
                     f"{history}. Source: labevents table, subject {sid}."),
        })
        claims.append({
            "claim": (f"The most recent {grp['label'].lower()} was {_num(latest['valuenum'])}{uom} "
                      f"on {latest['date']} [{label}]."),
            "label": label, "chunk_id": -1, "verified": True,
            "verification_note": "deterministic: value read directly from labevents",
        })

    answer = " ".join(c["claim"] for c in claims)
    bump("deterministic_answer")
    bump("deterministic_verified", len(claims))
    logger.info(f"[lab_lookup] answered deterministically from {len(series)} analyte(s), 0 LLM calls")
    return {
        "lab_evidence": ev, "draft_answer": answer, "final_answer": answer, "citations": claims,
        "verification": {"checked": 0, "unsupported": 0, "synthesis_failed": False,
                         "deterministic": len(claims), "llm_checked": 0},
        "needs_human_review": False, "review_status": "auto_approved",
        "node_trail": _trail(state, "lab_lookup"),
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


def _disambiguate(series: list[dict], query: str, known_labels: list[str]) -> list[dict]:
    """Narrow a resolver match down to the ONE analyte the question asked about.

    LabResolver.match is a substring matcher built for retrieval, where pulling
    in a neighbouring analyte only adds harmless context. As an *answer* path
    that over-match is wrong in two different ways:

      "most recent hemoglobin A1c"  -> resolves to Hemoglobin AND Hemoglobin A1c;
                                       answering with plain hemoglobin answers a
                                       question nobody asked.
      same question, patient has no A1c at all
                                    -> resolves to Hemoglobin alone, and a
                                       confident "hemoglobin was 13.1 g/dL"
                                       replaces the correct answer, which is that
                                       no A1c is documented.

    The second case is why a single result is not automatically safe. So: work
    out which known analyte names the question actually contains, let the most
    specific one win (A1c beats hemoglobin), and if this patient has no rows for
    the analyte that was named, return nothing — retrieval and synthesis will
    say it is not documented. Only the deterministic path is narrowed here; the
    resolver and retrieval are untouched.
    """
    q = (query or "").lower()
    named = {lab.lower() for lab in known_labels if lab and lab.lower() in q}
    named = {l for l in named if not any(o != l and l in o for o in named)}
    if not named:
        # The question used a synonym ("blood sugar"), not a label. Trust the
        # resolver only when it came back with exactly one analyte.
        return series if len(series) == 1 else []
    have = {g["label"].lower() for g in series}
    if not named <= have:
        return []                      # asked for something this patient has no rows for
    return [g for g in series if g["label"].lower() in named]


def patient_retrieval(state: AgentState) -> dict:
    retriever, _ = get_retrievers()
    with tracing.span("patient_retrieval", subject_id=state.get("subject_id"),
                      query=state["query"]) as s:
        results = retriever.search(
            query=state["query"],
            subject_id=state.get("subject_id"),
            temporal_filter=state.get("temporal_mode") or "auto",
            top_k=PATIENT_TOP_K,
        )
        ev = _to_evidence(results, "S", "note")
        if s is not None:
            s.update(output={"n": len(ev), "chunk_ids": [e["chunk_id"] for e in ev],
                             "top_score": ev[0]["score"] if ev else None})
    logger.info(f"[patient_retrieval] {len(ev)} chunks")
    return {"patient_evidence": ev, "node_trail": _trail(state, "patient_retrieval")}


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

    results, attempts = [], []
    for strict in (False, True):
        cq = concept(strict)
        if not cq:
            continue
        attempts.append(cq)
        out = call_external(gate, "search_literature",
                            {"query": cq}, stub_tools.search_literature)
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
        results = out.get("results", [])
        break

    ev = _to_literature_evidence(results)
    logger.info(f"[literature_retrieval] {len(ev)} results; "
                f"{sum(1 for r in egress if not r['allowed'])} blocked so far")
    return {"literature_evidence": ev, "egress_log": egress,
            "node_trail": _trail(state, "literature_retrieval")}


def synthesis(state: AgentState) -> dict:
    pt = state.get("patient_evidence", []) or []
    gl = state.get("guideline_evidence", []) or []
    lit = state.get("literature_evidence", []) or []

    if not pt and not gl and not lit:
        return {"draft_answer": "The available records do not contain enough information to answer this.",
                "citations": [], "node_trail": _trail(state, "synthesis")}

    user = prompts.build_synthesis_prompt(state["query"], pt, gl, lit)
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
    return {"draft_answer": answer, "citations": cites,
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
                                  (state.get("literature_evidence", []) or []) +
                                  (state.get("lab_evidence", []) or [])}
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
    pure_refusal = refusal and not any(_labels_of(c) for c in out)

    if pure_refusal:
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
                                  (state.get("lab_evidence", []) or [])}

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
    if DETERMINISTIC_LABS and wants_deterministic_lab(
            _decision_of(state), state.get("temporal_mode") or "all", state.get("subject_id")):
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

    b = StateGraph(AgentState)
    b.add_node("triage", triage)
    b.add_node("patient_retrieval", patient_retrieval)
    b.add_node("guideline_retrieval", guideline_retrieval)
    b.add_node("synthesis", synthesis)
    b.add_node("verification", verification)
    b.add_node("refuse", refuse)
    b.add_node("literature_retrieval", literature_retrieval)
    b.add_node("lab_lookup", lab_lookup)

    b.add_edge(START, "triage")
    b.add_conditional_edges("triage", route_from_triage,
                            ["patient_retrieval", "guideline_retrieval", "lab_lookup", "refuse"])
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
