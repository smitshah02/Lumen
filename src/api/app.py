"""
Lumen HTTP API
==============
Thin FastAPI layer over the EXISTING agent graph and retriever — no RAG logic
lives here. /ask runs src.agents.graph via run_graph.run_once (triage ->
hybrid retrieval -> local qwen synthesis -> verification -> human review), and
/retrieve calls the graph's own HybridRetriever singleton.

    LUMEN_DATA_PLANE=demo uvicorn src.api.app:app --host 127.0.0.1 --port 8000

Data planes: src.storage picks the database from LUMEN_DATA_PLANE (demo ->
lumen_demo, research -> lumen). The research plane holds real MIMIC-derived
text, so — like src.mcp_server.planes — it only answers loopback clients.
"""

from __future__ import annotations

import re
import time
import uuid
import asyncio
import logging
import threading
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src import storage
from src.llm import local_client
from src.obs import tracing
from src.mcp_server.planes import VALID_PLANES
from src.obs.logging import (configure_logging, log_event, obs_extra, start_request, end_request,
                             current_timings, Timer)
from src.api.schemas import (AskRequest, AskResponse, RetrieveRequest, RetrieveResponse, RetrievedChunk,
                             Citation, Source)

logger = logging.getLogger("lumen.api")

DATA_PLANE = storage.DATA_PLANE
EXPECTED_DB = storage.DEMO_DB_NAME if DATA_PLANE == "demo" else storage.RESEARCH_DB_NAME
READY_TIMEOUT_S = 3.0
_RID_RE = re.compile(r"[A-Za-z0-9._-]{8,64}")
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}

# The retriever/reranker share one accelerator and the graph singletons are not
# re-entrant, so inference is serialized. /health and /ready never take this lock.
_infer_lock = threading.Lock()
_graph = None
_graph_lock = threading.Lock()


class SubjectNotFound(Exception):
    pass


class DependencyUnavailable(Exception):
    def __init__(self, dependency: str):
        super().__init__(dependency)
        self.dependency = dependency


class GenerationFailed(Exception):
    pass


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    if DATA_PLANE not in VALID_PLANES:
        raise RuntimeError(f"LUMEN_DATA_PLANE={DATA_PLANE!r}; expected one of {sorted(VALID_PLANES)}")
    log_event(logger, "startup", database=EXPECTED_DB, model=local_client.MAIN_MODEL)
    ts = tracing.status()
    log_event(logger, "tracing_config", tracing_enabled=ts["enabled"], tracing_host=ts["host"],
              tracing_state=ts["state"], reason=ts.get("policy"))
    yield
    tracing.flush()                  # send buffered spans before the process exits
    if _graph is not None:
        from src.agents.graph import close_pools
        close_pools()
    log_event(logger, "shutdown")


app = FastAPI(title="Lumen", version="0.5.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request context: id, plane guard, access log, last-resort 500
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id", "")
    if not _RID_RE.fullmatch(rid):
        rid = uuid.uuid4().hex
    tokens = start_request(rid)
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        client = request.client.host if request.client else ""
        if DATA_PLANE == "research" and client not in _LOOPBACK:
            response = JSONResponse(status_code=403, content={
                "error": "forbidden", "detail": "research plane serves loopback clients only", "request_id": rid})
        else:
            response = await call_next(request)
    except Exception as e:
        logger.exception("unhandled_error", extra=obs_extra("unhandled_error", path=request.url.path,
                                                           error_type=type(e).__name__))
        response = JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})
    response.headers["X-Request-ID"] = rid
    log_event(logger, "http_request", method=request.method, path=request.url.path,
              status_code=response.status_code, status=getattr(request.state, "outcome", None),
              subject_id=getattr(request.state, "subject_id", None),
              duration_ms=round((time.perf_counter() - t0) * 1000, 1))
    end_request(tokens)
    return response


def _err(request: Request, code: int, error: str, detail=None, outcome=None) -> JSONResponse:
    request.state.outcome = outcome or error
    body = {"error": error, "request_id": getattr(request.state, "request_id", None)}
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(status_code=code, content=body)


@app.exception_handler(RequestValidationError)
async def _on_validation(request: Request, exc: RequestValidationError):
    # loc/msg/type only — never echo the submitted values back.
    detail = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
    return _err(request, 422, "validation_error", detail)


@app.exception_handler(SubjectNotFound)
async def _on_subject(request: Request, exc: SubjectNotFound):
    return _err(request, 404, "subject_not_found")


@app.exception_handler(DependencyUnavailable)
async def _on_dependency(request: Request, exc: DependencyUnavailable):
    log_event(logger, "dependency_unavailable", level=logging.ERROR, dependency=exc.dependency)
    return _err(request, 503, f"{exc.dependency}_unavailable")


@app.exception_handler(SQLAlchemyError)
@app.exception_handler(psycopg.Error)
async def _on_database(request: Request, exc: Exception):
    log_event(logger, "dependency_unavailable", level=logging.ERROR, dependency="database",
              error_type=type(exc).__name__)
    return _err(request, 503, "database_unavailable")


@app.exception_handler(GenerationFailed)
async def _on_generation(request: Request, exc: GenerationFailed):
    return _err(request, 500, "generation_failed")


# ---------------------------------------------------------------------------
# Dependencies (module-level so tests can replace them)
# ---------------------------------------------------------------------------
def _check_database() -> dict:
    with storage.engine.connect() as c:
        db = c.execute(text("SELECT current_database()")).scalar()
        indexed = c.execute(text("SELECT EXISTS (SELECT 1 FROM note_chunks)")).scalar()
    plane_ok = db == EXPECTED_DB and not (DATA_PLANE == "demo" and db == storage.RESEARCH_DB_NAME)
    return {"database": "ok" if plane_ok else "wrong_database", "corpus": "ok" if indexed else "empty"}


def _check_ollama() -> dict:
    h = local_client.health()
    if not h.get("reachable"):
        return {"ollama": "unavailable", "model": "unknown"}
    return {"ollama": "ok", "model": "ok" if h.get("main_ok") and h.get("fast_ok") else "missing"}


def _ensure_subject(subject_id: int) -> None:
    with storage.engine.connect() as c:
        if not c.execute(text("SELECT EXISTS (SELECT 1 FROM note_chunks WHERE subject_id = :s)"),
                         {"s": subject_id}).scalar():
            raise SubjectNotFound(subject_id)


def _get_graph():
    global _graph
    with _graph_lock:
        if _graph is None:
            from src.agents.graph import build_graph
            _graph, _ = build_graph()
        return _graph


def _run_retrieve(query: str, subject_id: int, temporal_filter: str, top_k: int):
    from src.agents.graph import get_retrievers
    from src.retrieval.hybrid_retriever_v2 import detect_temporal_mode
    mode = detect_temporal_mode(query) if temporal_filter == "auto" else temporal_filter
    retriever, _ = get_retrievers()
    with _infer_lock:
        results = retriever.search(query=query, subject_id=subject_id, temporal_filter=mode, top_k=top_k)
    return mode, results


def _run_ask(query: str, subject_id: int, thread_id: str, request_id: str) -> tuple[dict, dict]:
    from src.agents.run_graph import run_once
    graph = _get_graph()
    with _infer_lock:
        out, config = run_once(graph, query, subject_id, thread_id, tags=("lumen", "api", DATA_PLANE),
                               metadata={"request_id": request_id})
    return out, graph.get_state(config).values


async def _probe(fn, name: str) -> dict:
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn), READY_TIMEOUT_S)
    except asyncio.TimeoutError:
        log_event(logger, "ready_probe", level=logging.WARNING, dependency=name, error="timeout")
        return {name: "timeout"}
    except Exception as e:
        log_event(logger, "ready_probe", level=logging.WARNING, dependency=name, error_type=type(e).__name__)
        return {name: "unavailable"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Liveness only: no models, no database, no Ollama."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request):
    db, llm = await asyncio.gather(_probe(_check_database, "database"), _probe(_check_ollama, "ollama"))
    deps = {"database": db.get("database"), "corpus": db.get("corpus", "unknown"),
            "ollama": llm.get("ollama"), "model": llm.get("model", "unknown")}
    ok = all(v == "ok" for v in deps.values())
    request.state.outcome = "ready" if ok else "not_ready"
    body = {"status": "ready" if ok else "not_ready", "data_plane": DATA_PLANE, "database": EXPECTED_DB,
            "models": {"main": local_client.MAIN_MODEL, "fast": local_client.FAST_MODEL}, "dependencies": deps,
            # informational only: an unreachable observability backend never makes the API unready
            "tracing": tracing.status(),
            "request_id": request.state.request_id}
    return JSONResponse(status_code=200 if ok else 503, content=body)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest, request: Request):
    request.state.subject_id = req.subject_id

    def work():
        _ensure_subject(req.subject_id)
        return _run_retrieve(req.query, req.subject_id, req.temporal_filter, req.top_k)

    with Timer() as t:
        mode, results = await asyncio.to_thread(work)
    request.state.outcome = "ok"
    log_event(logger, "retrieve_completed", subject_id=req.subject_id, result_count=len(results),
              top_k=req.top_k, temporal_mode=mode, duration_ms=t.ms)
    return RetrieveResponse(
        request_id=request.state.request_id, data_plane=DATA_PLANE, subject_id=req.subject_id, query=req.query,
        temporal_mode=mode, latency_ms=t.ms,
        # chunk_text comes from note_chunks (de-identified in research, synthetic in demo);
        # clinical_notes.text_original is never read by the retriever.
        results=[RetrievedChunk(rank=i, chunk_id=r.chunk_id, note_id=r.note_id, note_type=r.note_type,
                                charttime=r.charttime, score=round(float(r.final_score), 4),
                                sources=[s for s in r.sources if s != "both"], text=r.chunk_text)
                 for i, r in enumerate(results, 1)],
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    rid = request.state.request_id
    thread_id = f"api-{rid}"
    request.state.subject_id = req.subject_id

    def work():
        _ensure_subject(req.subject_id)
        return _run_ask(req.query, req.subject_id, thread_id, rid)

    with Timer() as t:
        out, st = await asyncio.to_thread(work)

    errors = st.get("errors") or []
    if st.get("review_status") == "failed" or any(e.startswith("synthesis") for e in errors):
        request.state.outcome = "failed"
        if any("local LLM call failed" in e for e in errors):
            raise DependencyUnavailable("model")
        raise GenerationFailed()

    interrupted = "__interrupt__" in out
    cites = st.get("citations") or []
    if interrupted:
        status, answer = "human_review_required", st.get("draft_answer") or ""
        flagged = len(out["__interrupt__"][0].value.get("flagged", []))
    else:
        status = "refused" if st.get("query_type") == "unsupported" else "completed"
        answer = st.get("final_answer") or st.get("draft_answer") or ""
        flagged = sum(1 for c in cites if not c.get("verified"))
    evidence = (st.get("patient_evidence") or []) + (st.get("guideline_evidence") or []) + (st.get("literature_evidence") or [])
    timings = {**current_timings(), "total_ms": t.ms}
    request.state.outcome = status
    log_event(logger, "ask_completed", subject_id=req.subject_id, thread_id=thread_id, status=status,
              review_status=st.get("review_status"), query_type=st.get("query_type"),
              n_citations=sum(1 for c in cites if c.get("label")), needs_human_review=interrupted or bool(st.get("needs_human_review")),
              llm_calls=timings.get("llm_calls"), llm_ms=timings.get("llm_ms"),
              retrieval_ms=timings.get("retrieval_ms"), duration_ms=t.ms)
    return AskResponse(
        request_id=rid, thread_id=thread_id, data_plane=DATA_PLANE, status=status,
        review_status=st.get("review_status"), answer=answer, answer_is_draft=interrupted,
        citations=[Citation(label=c.get("label") or "", chunk_id=int(c.get("chunk_id", -1)), claim=c.get("claim", ""),
                            verified=bool(c.get("verified"))) for c in cites],
        sources=[Source(label=e["label"], chunk_id=int(e["chunk_id"]), source_type=e.get("source_type", ""),
                        note_type=e.get("note_type"), charttime=e.get("charttime")) for e in evidence],
        flagged_claims=flagged, needs_human_review=interrupted or bool(st.get("needs_human_review")),
        query_type=st.get("query_type"), temporal_mode=st.get("temporal_mode"),
        node_trail=st.get("node_trail") or [],
        models={"main": local_client.MAIN_MODEL, "fast": local_client.FAST_MODEL},
        latency_ms=t.ms, timings=timings,
    )
