"""
Phase 2 — answer collection against the real frozen execution path
==================================================================
Every case is executed by the PRODUCTION LangGraph via src.agents.run_graph.
run_once — the same function src/api/app.py:_run_ask calls. Nothing about
retrieval, routing, synthesis or verification is reimplemented here; this
module only runs the graph and records what came back.

Why in-process rather than over HTTP
------------------------------------
/ask cannot supply what Phases 3 and 4 require:
  * AskResponse carries no evidence TEXT, so the offline judge could not grade
    groundedness;
  * it carries no per-claim verifier trace (stage/verdict/reason), so review
    routing could not be measured;
  * a synthesis failure is raised as an HTTP 5xx (GenerationFailed), so the
    failing run's detail never reaches the client.
run_once is the identical code path with the full graph state still attached.
The HTTP contract is checked separately, by the `api-crosscheck` stage.

Isolation
---------
Every case gets a fresh, run-scoped, per-case thread id. /ask derives its
LangGraph thread from the request id, and the Postgres checkpointer persists
threads — a fixed id would RESUME a previous run's thread, accumulating
node_trail and errors through their operator.add reducers and feeding the next
invoke to a graph paused at human_review. This is the same trap
scripts/cloud_eval.py documents at its RUN_ID definition.

Failure policy
--------------
A single case can never abort the run: the exception type and message are
recorded and execution continues. But an evaluator failure stays VISIBLE — it
is written with status "evaluator_error", counted in the denominator, and
surfaced in the report. It is never quietly dropped.
"""

from __future__ import annotations

import sys
import uuid
import time
import logging
import traceback
from pathlib import Path

from src.evals.final_eval import evidence as ev_mod
from src.evals.final_eval.manifest import RunDir, ROOT, utc_now

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = ("completed", "human_review_required", "refused", "failed")


def _escalation_reason_fn():
    """Reuse scripts/cloud_eval.py's labelling so this run's escalation reasons
    are directly comparable with the preserved cloud artifacts. If it cannot be
    imported the reason is recorded as 'unavailable' rather than invented."""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from cloud_eval import _escalation_reason
        return _escalation_reason
    except Exception as e:
        logger.warning("cloud_eval._escalation_reason unavailable (%s)", type(e).__name__)
        return lambda st, trace: "unavailable"


def _status_of(out: dict, st: dict) -> str:
    """The status the API would report for this state.

    Mirrors src/api/app.py:ask — a failed synthesis is `failed` (the API turns
    it into a 5xx), an interrupt is `human_review_required`, an out-of-scope
    triage is `refused`, everything else completed.
    """
    if "__interrupt__" in out:
        return "human_review_required"
    v = st.get("verification") or {}
    if st.get("review_status") == "failed" or v.get("synthesis_failed"):
        return "failed"
    if any(str(e).startswith("synthesis") for e in (st.get("errors") or [])):
        return "failed"
    if st.get("query_type") == "unsupported":
        return "refused"
    return "completed"


def _all_evidence(st: dict) -> list:
    return ((st.get("patient_evidence") or []) + (st.get("guideline_evidence") or [])
            + (st.get("literature_evidence") or []) + (st.get("lab_evidence") or []))


def collect_one(graph, case, run_id: str) -> tuple[dict, list]:
    """Run one case. Returns (response row, evidence-with-text for the judge)."""
    from src.obs.logging import start_request, end_request, current_timings
    from src.agents.run_graph import run_once
    from src.llm import local_client

    # Run-scoped AND per-case unique: never resumes an earlier checkpoint.
    thread_id = f"feval-{run_id}-{case.query_id}-{uuid.uuid4().hex[:6]}"
    request_id = thread_id
    tokens = start_request(request_id)
    t0 = time.perf_counter()
    try:
        out, _cfg = run_once(graph, case.query, case.subject_id, thread_id,
                             tags=("lumen", "final-eval"),
                             metadata={"request_id": request_id, "query_id": case.query_id})
        st = graph.get_state({"configurable": {"thread_id": thread_id}}).values
        timings = current_timings()
    finally:
        end_request(tokens)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    status = _status_of(out, st)
    verification = st.get("verification") or {}
    trace = verification.get("claims") or []
    cites = st.get("citations") or []
    evidence = _all_evidence(st)
    prov = ev_mod.resolve_provenance(evidence, case.subject_id)

    needs_review = bool("__interrupt__" in out or st.get("needs_human_review"))
    escalation = ("none" if not needs_review
                  else _escalation_reason_fn()(st, trace))

    # The deterministic lab path (src/agents/graph.py:lab_lookup) short-circuits
    # to finalize and never enters the `verification` node, so it produces no
    # per-claim trace. That is expected, not missing data, and is labelled so
    # the routing metrics do not read it as an absent verdict.
    trail = st.get("node_trail") or []
    det_lab = "lab_lookup" in trail and bool(st.get("lab_evidence"))

    row = {
        "query_id": case.query_id,
        "subject_id": case.subject_id,
        "query": case.query,
        "category": case.category,
        "answer_type": case.answer_type,
        "temporal": case.temporal,
        "difficulty": case.difficulty,
        "status": status,
        "error_type": None,
        "error_message": None,
        "answer": (st.get("final_answer") or st.get("draft_answer") or ""),
        "answer_is_draft": "__interrupt__" in out,
        "citations": [{"label": c.get("label"), "labels": c.get("labels"),
                       "chunk_id": c.get("chunk_id"), "claim": c.get("claim"),
                       "verified": bool(c.get("verified")),
                       "verification_note": c.get("verification_note")} for c in cites],
        "sources": [{**ev_mod.evidence_digest(e),
                     **{k: v for k, v in (prov.get(e.get("label")) or {}).items()}}
                    for e in evidence],
        "verification_trace": trace,
        "verification_summary": {k: verification.get(k) for k in
                                 ("checked", "unsupported", "deterministic", "llm_checked",
                                  "refusal", "synthesis_failed", "synthesis_role")},
        # src/agents/graph.py:synthesis strips hallucinated labels OUT of
        # draft_answer but keeps the pre-strip list here. Re-validating the
        # stored answer therefore always reports zero bad labels — a property
        # of the repair, not of the model. The honest citation-validity metric
        # is this pre-strip record, so it is carried through.
        "citation_report": verification.get("citation_report") or {},
        "review_status": st.get("review_status"),
        "needs_human_review": needs_review,
        "escalation_reason": escalation,
        "deterministic_lab_path": det_lab,
        "query_type": st.get("query_type"),
        "temporal_mode": st.get("temporal_mode"),
        "query_complexity": st.get("query_complexity"),
        "classified_by": st.get("classified_by"),
        "synthesis_role": verification.get("synthesis_role"),
        "synthesis_model": (local_client.role_spec(verification["synthesis_role"])["model"]
                            if verification.get("synthesis_role") else None),
        "node_trail": trail,
        "graph_errors": st.get("errors") or [],
        "n_claims": len(cites),
        "n_flagged_claims": sum(1 for c in cites if not c.get("verified")),
        # Same zero-defaults src/api/app.py:ask applies, so a counter that never
        # fired reads 0 rather than absent — otherwise a deterministic run with
        # no fast-tier call reports "fast: None" instead of "fast: 0".
        "timings": {"llm_calls": 0, "llm_main_calls": 0, "llm_fast_calls": 0,
                    "deterministic_answer": 0, "deterministic_verified": 0,
                    "llm_verified": 0, **timings, "total_ms": elapsed_ms},
        "thread_id": thread_id,
        "collected_at": utc_now(),
    }
    judge_evidence = [{"label": e.get("label"), "text": e.get("text") or ""} for e in evidence]
    return row, judge_evidence


def _error_row(case, exc: BaseException, elapsed_ms: float) -> dict:
    """An evaluator/system failure, recorded so it counts against completeness.

    Only the exception type and message are stored — never a traceback, which
    would embed absolute filesystem paths in a published artifact.
    """
    return {
        "query_id": case.query_id, "subject_id": case.subject_id, "query": case.query,
        "category": case.category, "answer_type": case.answer_type,
        "temporal": case.temporal, "difficulty": case.difficulty,
        "status": "evaluator_error",
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:500],
        "answer": "", "answer_is_draft": False, "citations": [], "sources": [],
        "verification_trace": [], "verification_summary": {},
        "review_status": None, "needs_human_review": False, "escalation_reason": "evaluator_error",
        "deterministic_lab_path": False, "query_type": None, "temporal_mode": None,
        "query_complexity": None, "classified_by": None, "synthesis_role": None,
        "synthesis_model": None, "node_trail": [], "graph_errors": [],
        "n_claims": 0, "n_flagged_claims": 0,
        "timings": {"total_ms": elapsed_ms}, "thread_id": None, "collected_at": utc_now(),
    }


def collect(run: RunDir, cases: list, *, resume: bool = False, progress=None) -> dict:
    """Execute every case and append to responses.jsonl.

    Resume skips ids already present, so an interrupted run never re-pays for a
    case it already completed and never duplicates a query_id.
    """
    say = progress or (lambda *_a, **_k: None)
    done = run.completed_ids("responses") if resume else set()
    todo = [c for c in cases if c.query_id not in done]
    say(f"  collection: {len(todo)} to run, {len(done)} already present")

    from src.agents.graph import build_graph, close_pools
    graph, _ = build_graph()
    stats = {"attempted": 0, "errors": 0, "skipped_existing": len(done)}
    try:
        for case in todo:
            t0 = time.perf_counter()
            stats["attempted"] += 1
            try:
                row, judge_ev = collect_one(graph, case, run.run_id)
            except BaseException as e:            # noqa: BLE001 — isolation is the point
                if isinstance(e, KeyboardInterrupt):
                    raise
                elapsed = round((time.perf_counter() - t0) * 1000, 1)
                logger.error("[collect] %s failed: %s", case.query_id, e)
                logger.debug("%s", traceback.format_exc())
                row, judge_ev = _error_row(case, e, elapsed), []
                stats["errors"] += 1
            run.append("responses", row)
            if judge_ev:
                run.append("evidence_cache", {"query_id": case.query_id, "evidence": judge_ev})
            say(f"  {row['query_id']:<10} {row['status']:<22} "
                f"claims={row['n_claims']} flagged={row['n_flagged_claims']} "
                f"review={row['review_status']} {row['timings'].get('total_ms')}ms")
    finally:
        close_pools()
    return stats


def load_evidence_cache(run: RunDir) -> dict:
    """query_id -> [{label, text}] for the judge stage."""
    return {r["query_id"]: r.get("evidence") or [] for r in run.read_jsonl("evidence_cache")}
