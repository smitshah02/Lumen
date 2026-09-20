"""
Structured logging
==================
One JSON object per line on stderr, with the request id attached automatically
to every record emitted while a request is being served (including records
from the graph, retriever and LLM client, which run in worker threads).

Only whitelisted field names are ever emitted as structured data, so a caller
cannot accidentally log note text, prompts or credentials by passing them as a
field. Plain log messages from existing modules pass through as "message".

    from src.obs.logging import configure_logging, log_event, obs_extra
    configure_logging()
    log_event(logger, "retrieval", result_count=5, duration_ms=812.4)
    logger.info("free text", extra=obs_extra("retrieval", result_count=5))
"""

from __future__ import annotations

import os
import json
import time
import logging
import contextvars
from datetime import datetime, timezone

ALLOWED_FIELDS = frozenset({
    "request_id", "method", "path", "status", "status_code", "duration_ms", "subject_id",
    "data_plane", "thread_id", "result_count", "top_k", "temporal_mode", "model", "tier",
    "prompt_tokens", "completion_tokens", "query_type", "review_status", "needs_human_review",
    "n_citations", "llm_calls", "llm_ms", "retrieval_ms", "error", "error_type", "dependency",
    "database", "client", "reason", "tracing_host", "tracing_enabled", "tracing_state",
    # latency attribution: which model role/tier served a call, and how much of
    # the work the deterministic paths absorbed. Names and counts only.
    "llm_role", "llm_main_calls", "llm_fast_calls", "llm_main_ms", "llm_fast_ms",
    "query_class", "deterministic_answer", "deterministic_verified", "llm_verified",
})

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("lumen_request_id", default=None)
_timings: contextvars.ContextVar[dict | None] = contextvars.ContextVar("lumen_timings", default=None)


# --- request context -----------------------------------------------------------
def start_request(request_id: str) -> tuple:
    """Bind a request id and a fresh timings accumulator to the current context."""
    return _request_id.set(request_id), _timings.set({})


def end_request(tokens: tuple) -> None:
    rid_token, t_token = tokens
    _request_id.reset(rid_token)
    _timings.reset(t_token)


def current_request_id() -> str | None:
    return _request_id.get()


def add_timing(key: str, ms: float) -> None:
    """Accumulate a duration for the current request (no-op outside a request).
    The dict is shared by reference, so worker threads that received a copy of
    the context still write into the request's accumulator."""
    t = _timings.get()
    if t is not None:
        t[key] = round(t.get(key, 0.0) + ms, 1)
        t[key.replace("_ms", "_calls")] = t.get(key.replace("_ms", "_calls"), 0) + 1


def bump(key: str, n: int = 1) -> None:
    """Increment a plain counter on the current request (no-op outside a request).
    For things that are *not* durations — deterministic answers, claims resolved
    without a model call — so they ride along in the same /ask timings payload."""
    t = _timings.get()
    if t is not None:
        t[key] = t.get(key, 0) + n


def current_timings() -> dict:
    return dict(_timings.get() or {})


# --- emitting ---------------------------------------------------------------------
def obs_extra(event: str, **fields) -> dict:
    return {"lumen_event": event, "lumen_fields": fields}


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, event, extra=obs_extra(event, **fields))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "lumen_event", None)
        out = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": event or "log",
            "data_plane": os.environ.get("LUMEN_DATA_PLANE", "research").strip().lower(),
        }
        msg = record.getMessage()
        if msg and msg != event:
            out["message"] = msg[:500]
        rid = _request_id.get()
        if rid:
            out["request_id"] = rid
        for k, v in (getattr(record, "lumen_fields", None) or {}).items():
            if k in ALLOWED_FIELDS and v is not None:
                out[k] = v
        if record.exc_info:
            out["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            out["traceback"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: str | int | None = None) -> None:
    """Route the root logger (and uvicorn's error logger) through JSON on stderr.
    uvicorn's plain-text access log is silenced: the API logs every request itself."""
    level = level or os.environ.get("LUMEN_LOG_LEVEL", "INFO")
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers, lg.propagate = [], True
    access = logging.getLogger("uvicorn.access")
    access.handlers, access.propagate, access.disabled = [], False, True
    for noisy in ("httpx", "httpcore", "urllib3", "transformers", "sentence_transformers", "psycopg.pool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Timer:
    """with Timer() as t: ...; t.ms"""
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = round((time.perf_counter() - self._t0) * 1000, 1)
        return False
