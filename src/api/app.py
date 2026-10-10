"""
Lumen HTTP API
==============
Thin FastAPI layer over the EXISTING agent graph and retriever — no RAG logic
lives here. /ask runs src.agents.graph via run_graph.run_once (triage ->
hybrid retrieval -> local qwen synthesis -> verification -> human review), and
/retrieve calls the graph's own HybridRetriever singleton.

    LUMEN_DATA_PLANE=demo uvicorn src.api.app:app --host 127.0.0.1 --port 8000

Data planes: src.storage picks the isolated database from LUMEN_DATA_PLANE
(demo -> lumen_demo, research -> lumen). The research
plane holds real MIMIC-derived text, so it only answers loopback clients.
"""

from __future__ import annotations

import os
import json
import re
import time
import uuid
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import FastAPI, Path as PathParam, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src import storage
from src.config import CHUNK_BUILD, DATA_PROFILE, PROFILE_SETTINGS, MODELS_CONFIG, MODELS_DIR, VALID_PLANES, pgvector_status
from src.retrieval.index_provenance import (configuration_hash as index_configuration_hash,
                                            legacy_adopted_hashes)
from src.storage.schema import SCHEMA_VERSION
from src.storage import readiness
from src.agents import review
from src.safety import pubmed
from src.llm import local_client
from src.obs import tracing
from src.obs.logging import (configure_logging, log_event, obs_extra, start_request, end_request,
                             current_timings, note_build, current_build, Timer)
from src.api.schemas import (AdmissionScope, AskRequest, AskResponse, RetrieveRequest, RetrieveResponse,
                             RetrievedChunk, Citation, Source, ReviewDecision)

logger = logging.getLogger("lumen.api")

DATA_PLANE = storage.DATA_PLANE
_PLANE_DATABASES = {
    "demo": storage.DEMO_DB_NAME,
    "research": storage.RESEARCH_DB_NAME,
}
EXPECTED_DB = _PLANE_DATABASES[DATA_PLANE]
READY_TIMEOUT_S = 3.0
_RID_RE = re.compile(r"[A-Za-z0-9._-]{8,64}")
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}

# The retriever/reranker share one accelerator and the graph singletons are not
# re-entrant, so inference is serialized. /health and /ready never take this lock.
_infer_lock = threading.Lock()
# Load both model tiers at startup. Off by default so unit tests and local
# runs never reach for Ollama; the Pod env template turns it on.
WARMUP = os.environ.get("LUMEN_LLM_WARMUP", "0").strip() in ("1", "true", "yes")

_graph = None
_graph_lock = threading.Lock()


class SubjectNotFound(Exception):
    pass


class AdmissionNotFound(Exception):
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
    try:                                         # say it at startup too; /ready and every read enforce it
        for p in readiness.problems():
            log_event(logger, "data_source_not_ready", level=logging.ERROR, data_profile=DATA_PROFILE, **p)
    except Exception as e:                       # database not up yet: /ready reports it
        log_event(logger, "data_source_check_failed", level=logging.WARNING, error_type=type(e).__name__)
    ts = tracing.status()
    log_event(logger, "tracing_config", tracing_enabled=ts["enabled"], tracing_host=ts["host"],
              tracing_state=ts["state"], reason=ts.get("policy"))
    # Make both tiers resident before the first request rather than paying the
    # weight load inside it. Daemon thread: warmup must never delay /health or
    # /ready, and Ollama serialises its own model loads, so this cannot race a
    # real request into a double load.
    if WARMUP:
        threading.Thread(target=_warm_models, name="llm-warmup", daemon=True).start()
    yield
    tracing.flush()                  # send buffered spans before the process exits
    if _graph is not None:
        from src.agents.graph import close_pools
        close_pools()
    log_event(logger, "shutdown")


def _warm_models() -> None:
    try:
        local_client.warmup()
    except Exception as e:                               # never fatal
        log_event(logger, "llm_warmup", level=logging.WARNING, error_type=type(e).__name__)


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
              build_id=current_build(),
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


@app.exception_handler(AdmissionNotFound)
async def _on_admission(request: Request, exc: AdmissionNotFound):
    # One answer whether the admission is another patient's or does not exist,
    # so the response says nothing about other patients. The value is not echoed.
    return _err(request, 422, "validation_error", [{
        "loc": ["body", "hadm_id"], "msg": "hadm_id is not an admission of this subject", "type": "value_error"}])


@app.exception_handler(readiness.DataSourceNotReady)
async def _on_data_source(request: Request, exc: readiness.DataSourceNotReady):
    # The profile's data is unusable. Say which part and why; never answer from something else.
    log_event(logger, "data_source_not_ready", level=logging.ERROR, component=exc.component, reason=exc.reason,
              data_profile=DATA_PROFILE)
    return _err(request, 503, "data_source_not_ready", {"component": exc.component, "reason": exc.reason})


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


@app.exception_handler(review.ReviewNotFound)
async def _on_review_missing(request: Request, exc: review.ReviewNotFound):
    return _err(request, 404, "review_not_found")


@app.exception_handler(review.ReviewNotPending)
async def _on_review_not_pending(request: Request, exc: review.ReviewNotPending):
    # 409: the thread exists but has nothing to decide — it never needed review,
    # or a decision was already submitted. Nothing is resumed.
    return _err(request, 409, "review_not_pending", {"review_status": exc.review_status})


@app.exception_handler(GenerationFailed)
async def _on_generation(request: Request, exc: GenerationFailed):
    return _err(request, 500, "generation_failed")


# ---------------------------------------------------------------------------
# Dependencies (module-level so tests can replace them)
# ---------------------------------------------------------------------------
def _extension_status(installed) -> tuple[str, bool]:
    """(readiness status, patch mismatch) for an installed pgvector version."""
    status = pgvector_status(installed)
    return ("ok" if status in ("ok", "patch_mismatch") else "missing_or_wrong_version",
            status == "patch_mismatch")


def _check_database() -> dict:
    with storage.engine.connect() as c:
        db = c.execute(text("SELECT current_database()")).scalar()
        extension = c.execute(text(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )).scalar()
        required_tables = ("note_chunks", "clinical_notes", "d_labitems", "lumen_schema_version",
                           "ingestion_log", "ingestion_runs", "note_index_runs", "note_index_state")
        present = {
            table for table in required_tables
            if c.execute(text("SELECT to_regclass(:name)"), {"name": table}).scalar()
        }
        missing_tables = sorted(set(required_tables) - present)
        schema_version = None
        indexes = set()
        chunks = None
        eligible_notes = None
        indexed_notes = None
        ingestion_status = None
        index_provenance = None
        if not missing_tables:
            schema_version = c.execute(text(
                "SELECT COALESCE(MAX(version), 0) FROM lumen_schema_version"
            )).scalar()
            indexes = set(c.execute(text(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()"
            )).scalars())
            # Whether there is a corpus, not how big: counting it read every chunk on every probe.
            chunks = c.execute(text("SELECT EXISTS (SELECT 1 FROM note_chunks)")).scalar()
            eligible_notes = c.execute(text("""
                SELECT COUNT(*) FROM clinical_notes
                WHERE COALESCE(text_deid, text_original) IS NOT NULL
                  AND COALESCE(text_deid, text_original) != ''
            """)).scalar()
            # The current configuration, or an index adopted as-is from before
            # provenance was recorded (reported below, never hidden).
            legacy = legacy_adopted_hashes(c)
            indexed_notes = c.execute(text("""
                SELECT COUNT(*) FROM note_index_state nis
                WHERE nis.status='completed' AND nis.config_hash = ANY(:config_hashes)
                  AND nis.chunk_count=(
                      SELECT COUNT(*) FROM note_chunks nc WHERE nc.note_id=nis.note_id
                  )
            """), {"config_hashes": [index_configuration_hash(), *legacy]}).scalar()
            # From the state rows, not the run log: after a full --reindex the
            # adopted run row remains but no note carries its hash any more.
            index_provenance = "legacy_adopted" if legacy and c.execute(text("""
                SELECT EXISTS (SELECT 1 FROM note_index_state
                               WHERE status='completed' AND config_hash = ANY(:legacy))
            """), {"legacy": legacy}).scalar() else "current"
            if DATA_PLANE == "research":
                ingestion_status = c.execute(text("""
                    SELECT status FROM ingestion_runs ORDER BY started_at DESC LIMIT 1
                """)).scalar()
                if DATA_PLANE == "research" and ingestion_status is None:
                    legacy_completed = c.execute(text("""
                        SELECT COUNT(*) FROM (
                            SELECT DISTINCT ON (table_name) table_name, status
                            FROM ingestion_log
                            WHERE table_name IN ('patients', 'admissions', 'clinical_notes')
                            ORDER BY table_name, id DESC
                        ) latest WHERE status='completed'
                    """)).scalar()
                    if legacy_completed == 3:
                        ingestion_status = "legacy_completed"
    plane_ok = db == EXPECTED_DB
    required_indexes = {"idx_chunks_fts", "idx_chunks_embedding", "idx_chunks_subject"}
    schema_ok = (not missing_tables and schema_version == SCHEMA_VERSION
                 and required_indexes.issubset(indexes))
    return {
        "database": "ok" if plane_ok else "wrong_database",
        "schema": "ok" if schema_ok else "missing_or_outdated",
        "extension": _extension_status(extension)[0],
        "corpus": "ok" if chunks else "empty" if chunks is False else "unknown",
        "ingestion": ("ok" if DATA_PLANE == "demo" else
                      "ok" if ingestion_status in ("completed", "legacy_completed") else
                      ingestion_status or "untracked"),
        "index": ("ok" if eligible_notes and indexed_notes == eligible_notes else
                  "incomplete_or_stale" if eligible_notes is not None else "unknown"),
        "schema_version": schema_version,
        "pgvector_version": extension,
        "pgvector_patch_mismatch": _extension_status(extension)[1],
        "index_provenance": index_provenance,
        "missing_tables": missing_tables,
        "missing_indexes": sorted(required_indexes - indexes),
        "indexed_notes": indexed_notes,
        "eligible_notes": eligible_notes,
    }


def _check_retrieval_models() -> dict:
    """Cheap readiness check: pinned metadata and sizes, but no multi-GB hashing."""
    problems = []
    for name, spec in MODELS_CONFIG["hugging_face"].items():
        if spec["profile"] != "runtime":
            continue
        target = MODELS_DIR / name
        try:
            manifest = json.loads((target / ".lumen-model.json").read_text(encoding="utf-8"))
            if any(manifest.get(key) != expected for key, expected in
                   (("name", name), ("repo", spec["repo"]), ("revision", spec["revision"]))):
                problems.append(f"{name}:provenance")
                continue
            files = manifest.get("files") or {}
            for required in ("config.json", "model.safetensors"):
                path = target / required
                if required not in files or not path.is_file() or path.stat().st_size != files[required].get("size"):
                    problems.append(f"{name}:{required}")
        except (OSError, ValueError, TypeError):
            problems.append(f"{name}:manifest")
    return {"retrieval_models": "ok" if not problems else "missing_or_unverified",
            "model_problems": problems}


def _check_ollama() -> dict:
    h = local_client.health()
    if not h.get("reachable"):
        return {"ollama": "unavailable", "model": "unknown"}
    return {"ollama": "ok", "model": "ok" if h.get("main_ok") and h.get("fast_ok") else "missing"}


def _check_data_profile() -> dict:
    """Is the data the active profile reads usable? control and scoped need nothing more."""
    found = readiness.problems()
    return {"data_profile": "ok" if not found else f"{found[0]['component']}_not_ready", "problems": found}


def _ensure_subject(subject_id: int) -> None:
    with storage.engine.connect() as c:
        if PROFILE_SETTINGS["chunk_table"] == "note_chunks_v2":        # a patient outside the selected build is not searchable
            note_build(CHUNK_BUILD)                                    # the build this request is gated on, for its log lines
            found = c.execute(text("SELECT EXISTS (SELECT 1 FROM note_chunks_v2 WHERE build_id = :b AND subject_id = :s)"),
                              {"b": CHUNK_BUILD, "s": subject_id}).scalar()
            if not found:
                raise SubjectNotFound(subject_id)
            return
        if not c.execute(text("SELECT EXISTS (SELECT 1 FROM note_chunks WHERE subject_id = :s)"),
                         {"s": subject_id}).scalar():
            raise SubjectNotFound(subject_id)


def _ensure_admission(subject_id: int, hadm_id: int) -> None:
    with storage.engine.connect() as c:
        if not c.execute(text("SELECT EXISTS (SELECT 1 FROM admissions WHERE hadm_id = :h AND subject_id = :s)"),
                         {"h": hadm_id, "s": subject_id}).scalar():
            raise AdmissionNotFound(hadm_id)


def _scope_report(st: dict, request_hadm_id: int | None) -> AdmissionScope:
    """What happened to admission scope on this run, read from graph state.
    `applied` is true only when note retrieval really searched one admission."""
    scope = st.get("admission_scope")
    if scope is None:                                    # the active profile does not scope
        return AdmissionScope(applied=False, requested=request_hadm_id is not None, status="not_enabled",
                              hadm_id=None, source=None,
                              reason=f"admission scoping is not enabled in the {DATA_PROFILE} data profile")
    status, applied = scope["status"], bool(st.get("admission_scope_applied"))
    reason = scope.get("reason")
    if status == "resolved" and not applied:
        reason = "the answer did not come from note retrieval, so the admission scope was not applied"
    return AdmissionScope(applied=applied, requested=status != "none", status=status,
                          hadm_id=scope.get("hadm_id"), source=scope.get("source"), reason=reason)


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


def _run_ask(query: str, subject_id: int, thread_id: str, request_id: str,
             hadm_id: int | None = None) -> tuple[dict, dict]:
    from src.agents.run_graph import run_once
    graph = _get_graph()
    with _infer_lock:
        out, config = run_once(graph, query, subject_id, thread_id, tags=("lumen", "api", DATA_PLANE),
                               metadata={"request_id": request_id}, hadm_id=hadm_id)
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
    db, llm, retrieval = await asyncio.gather(
        _probe(_check_database, "database"), _probe(_check_ollama, "ollama"),
        _probe(_check_retrieval_models, "retrieval_models"),
    )
    profile = await _probe(_check_data_profile, "data_profile")
    deps = {"database": db.get("database"), "schema": db.get("schema", "unknown"),
            "extension": db.get("extension", "unknown"), "corpus": db.get("corpus", "unknown"),
            "ingestion": db.get("ingestion", "unknown"), "index": db.get("index", "unknown"),
            "retrieval_models": retrieval.get("retrieval_models", "unknown"),
            "ollama": llm.get("ollama"), "model": llm.get("model", "unknown"),
            "data_profile": profile.get("data_profile")}
    ok = all(v == "ok" for v in deps.values())
    request.state.outcome = "ready" if ok else "not_ready"
    body = {"status": "ready" if ok else "not_ready", "data_plane": DATA_PLANE, "data_profile": DATA_PROFILE, "database": EXPECTED_DB,
            "models": {"main": local_client.MAIN_MODEL, "fast": local_client.FAST_MODEL,
                       "roles": local_client.runtime_config()["roles"]}, "dependencies": deps,
            "database_details": {"schema_version": db.get("schema_version"),
                                 "pgvector_version": db.get("pgvector_version"),
                                 "pgvector_patch_mismatch": db.get("pgvector_patch_mismatch", False),
                                 "index_provenance": db.get("index_provenance"),
                                 "missing_tables": db.get("missing_tables", []),
                                 "missing_indexes": db.get("missing_indexes", []),
                                 "indexed_notes": db.get("indexed_notes"),
                                 "eligible_notes": db.get("eligible_notes")},
            "retrieval_model_problems": retrieval.get("model_problems", []),
            "data_profile_problems": profile.get("problems", []),
            # informational only: an unreachable observability backend never makes the API unready
            "tracing": tracing.status(),
            # "none" unless LUMEN_LITERATURE_BACKEND opts in; the only outbound integration
            "literature_backend": pubmed.backend_name(),
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
              top_k=req.top_k, temporal_mode=mode, build_id=current_build(), duration_ms=t.ms)
    return RetrieveResponse(
        request_id=request.state.request_id, data_plane=DATA_PLANE, data_profile=DATA_PROFILE,
        subject_id=req.subject_id, query=req.query,
        temporal_mode=mode, latency_ms=t.ms,
        # chunk_text comes from the profile's chunk table: synthetic in demo; in research it is
        # MIMIC text, DUA-restricted and local-only (loopback clients only, enforced above).
        results=[RetrievedChunk(rank=i, chunk_id=r.chunk_id, note_id=r.note_id,
                                subject_id=r.subject_id, hadm_id=r.hadm_id,
                                chunk_index=r.chunk_index, note_type=r.note_type,
                                charttime=r.charttime, score=round(float(r.final_score), 4),
                                sources=[s for s in r.sources if s != "both"], text=r.chunk_text,
                                provenance=getattr(r, "provenance", None))
                 for i, r in enumerate(results, 1)],
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    rid = request.state.request_id
    thread_id = f"api-{rid}"
    request.state.subject_id = req.subject_id

    def work():
        _ensure_subject(req.subject_id)
        if req.hadm_id is None:
            return _run_ask(req.query, req.subject_id, thread_id, rid)
        _ensure_admission(req.subject_id, req.hadm_id)
        return _run_ask(req.query, req.subject_id, thread_id, rid, hadm_id=req.hadm_id)

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
    evidence = ((st.get("patient_evidence") or []) + (st.get("guideline_evidence") or [])
                + (st.get("literature_evidence") or []) + (st.get("lab_evidence") or [])
                + (st.get("encounter_evidence") or []) + (st.get("structured_evidence") or []))
    timings = {"llm_calls": 0, "llm_main_calls": 0, "llm_fast_calls": 0,
               "deterministic_answer": 0, "deterministic_verified": 0, "llm_verified": 0,
               **current_timings(), "total_ms": t.ms,
               "query_complexity": st.get("query_complexity"), "classified_by": st.get("classified_by")}
    request.state.outcome = status
    scope = _scope_report(st, req.hadm_id)
    log_event(logger, "ask_completed", subject_id=req.subject_id, thread_id=thread_id, status=status,
              scope_applied=scope.applied, scope_requested=scope.requested, scope_status=scope.status,
              scope_hadm_id=scope.hadm_id, scope_source=scope.source, scope_reason=scope.reason,
              review_status=st.get("review_status"), query_type=st.get("query_type"),
              n_citations=sum(1 for c in cites if c.get("label")), needs_human_review=interrupted or bool(st.get("needs_human_review")),
              llm_calls=timings.get("llm_calls", 0), llm_ms=timings.get("llm_ms"),
              llm_main_calls=timings.get("llm_main_calls", 0), llm_fast_calls=timings.get("llm_fast_calls", 0),
              query_class=st.get("query_complexity"),
              deterministic_answer=timings.get("deterministic_answer", 0),
              deterministic_verified=timings.get("deterministic_verified", 0),
              llm_verified=timings.get("llm_verified", 0),
              retrieval_ms=timings.get("retrieval_ms"), build_id=current_build(), duration_ms=t.ms)
    return AskResponse(
        request_id=rid, thread_id=thread_id, data_plane=DATA_PLANE, data_profile=DATA_PROFILE, status=status,
        review_status=st.get("review_status"), answer=answer, answer_is_draft=interrupted,
        citations=[Citation(label=c.get("label") or "", chunk_id=int(c.get("chunk_id", -1)), claim=c.get("claim", ""),
                            verified=bool(c.get("verified"))) for c in cites],
        sources=[Source(label=e["label"], chunk_id=int(e["chunk_id"]), source_type=e.get("source_type", ""),
                        note_id=e.get("note_id"), subject_id=e.get("subject_id"),
                        hadm_id=e.get("hadm_id"), chunk_index=e.get("chunk_index"),
                        note_type=e.get("note_type"), charttime=e.get("charttime"),
                        provenance=e.get("provenance")) for e in evidence],
        flagged_claims=flagged, needs_human_review=interrupted or bool(st.get("needs_human_review")),
        query_type=st.get("query_type"), temporal_mode=st.get("temporal_mode"),
        node_trail=st.get("node_trail") or [], admission_scope=scope,
        models={"main": local_client.MAIN_MODEL, "fast": local_client.FAST_MODEL},
        latency_ms=t.ms, timings=timings,
    )


# ---------------------------------------------------------------------------
# Human review: inspect a paused run, then approve or reject its draft
# ---------------------------------------------------------------------------
_THREAD = PathParam(pattern=r"^[A-Za-z0-9._:-]{1,128}$")


@app.get("/review/{thread_id}")
async def review_pending(thread_id: str = _THREAD):
    """The draft and flagged claims of a run paused at human review."""
    return await asyncio.to_thread(lambda: review.pending(_get_graph(), thread_id))


@app.post("/review/{thread_id}")
async def review_submit(body: ReviewDecision, request: Request, thread_id: str = _THREAD):
    """Resume the paused run from its checkpoint with the reviewer's decision."""
    st = await asyncio.to_thread(
        lambda: review.submit(_get_graph(), thread_id, body.decision, body.reviewer_note))
    status = "rejected" if st.get("review_status") == "rejected" else "completed"
    request.state.subject_id = st.get("subject_id")
    request.state.outcome = status
    log_event(logger, "review_submitted", thread_id=thread_id, decision=body.decision,
              status=status, review_status=st.get("review_status"))
    return {"thread_id": thread_id, "status": status, "decision": body.decision,
            "review_status": st.get("review_status"), "answer": st.get("final_answer") or "",
            "needs_human_review": bool(st.get("needs_human_review")),
            "human_decisions": st.get("human_decisions") or [],
            "node_trail": st.get("node_trail") or []}


# ---------------------------------------------------------------------------
# Local UI: three static files that call the endpoints above. Served by this
# app so it inherits the request middleware — on the research plane that is the
# loopback-only guard — and needs no second process. Mounted last so it can
# never shadow an API route.
# ---------------------------------------------------------------------------
UI_DIR = Path(__file__).parent / "ui"


@app.get("/", include_in_schema=False)
async def ui_root():
    return RedirectResponse("/ui/")


app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
