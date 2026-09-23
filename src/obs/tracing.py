"""
Tracing
=======
Langfuse tracing, OFF unless LUMEN_TRACING=1. Everything degrades to a no-op
(with a structured warning) when tracing is off or misconfigured, so the graph
and the API behave identically with tracing on or off.

Egress policy — traces carry note text in span inputs and LLM prompts:
  * research plane (real MIMIC):  local Langfuse only (localhost / 127.0.0.1 /
    host.docker.internal). Any remote endpoint is refused.
  * demo/synthea planes (synthetic data): local, or a remote endpoint over
    https.

The endpoint is resolved exactly as the Langfuse SDK resolves it —
LANGFUSE_BASE_URL, then the deprecated LANGFUSE_HOST, then the SDK default
https://cloud.langfuse.com — so the policy checks the URL traces would really
go to (an unset URL means Langfuse Cloud, not localhost).

Keys (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY) are read by the SDK from the
environment and are never logged or returned by status().
"""

from __future__ import annotations

import os
import atexit
import logging
from contextlib import contextmanager, ExitStack
from urllib.parse import urlparse

from src.config import get_data_plane
from src.obs.logging import log_event

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("LUMEN_TRACING", "0") == "1"
PLANE = get_data_plane()
PROVIDER = "langfuse"
_SDK_DEFAULT_URL = "https://cloud.langfuse.com"
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal")

_client = None
_init_failed = False      # sticky: one connection attempt per process
_state = "off" if not ENABLED else "pending"


def _effective_url() -> str:
    """The URL the SDK will send to (same precedence as langfuse/_client/client.py)."""
    return os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or _SDK_DEFAULT_URL


def _hostname() -> str:
    return urlparse(_effective_url()).hostname or ""


def _policy() -> tuple[bool, str]:
    url = urlparse(_effective_url())
    if (url.hostname or "") in _LOCAL_HOSTS:
        return True, "local endpoint"
    if PLANE not in {"demo", "synthea"}:
        return False, "remote endpoint refused outside synthetic data planes"
    if url.scheme != "https":
        return False, "remote endpoint must use https"
    return True, f"remote endpoint ({PLANE} plane)"


def _keys_present() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(os.environ.get("LANGFUSE_SECRET_KEY"))


def _disable(state: str, reason: str, level: int = logging.WARNING) -> None:
    global _client, _init_failed, _state
    _client, _init_failed, _state = None, True, state
    log_event(logger, "tracing_disabled", level=level, reason=reason, tracing_host=_hostname())


def status() -> dict:
    """Configuration view for /ready and manifests. No network, no secrets."""
    if not ENABLED:
        return {"enabled": False, "provider": PROVIDER, "host": None, "state": "off"}
    allowed, reason = _policy()
    state = _state
    if state == "pending":
        state = "pending" if (allowed and _keys_present()) else "misconfigured"
    return {"enabled": True, "provider": PROVIDER, "host": _hostname(), "state": state,
            "policy": reason, "keys_configured": _keys_present()}


def client():
    """Cached client, or None. Initialisation is attempted exactly once.

    A failed attempt is sticky. An unreachable Langfuse makes auth_check
    block until it times out, and every span and every LLM call goes
    through here — without this, a paused backend would stall a whole
    graph run instead of degrading quietly. The tradeoff is deliberate:
    a Langfuse that comes up mid-run is not picked up until the next
    process starts.
    """
    global _client, _state
    if not ENABLED or _init_failed:
        return None
    if _client is None:
        allowed, reason = _policy()
        if not allowed:
            _disable("refused", reason, logging.ERROR)
            return None
        if not _keys_present():
            _disable("misconfigured", "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
            return None
        try:
            from langfuse import get_client
            lf = get_client()
            if not lf.auth_check():
                _disable("unavailable", "langfuse auth check failed")
                return None
        except Exception as e:
            _disable("unavailable", f"langfuse unavailable ({type(e).__name__})")
            return None
        _client, _state = lf, "active"
        atexit.register(flush)      # short-lived processes must not drop buffered spans
        log_event(logger, "tracing_enabled", tracing_host=_hostname(), reason=reason)
    return _client


def handler():
    """LangChain CallbackHandler for the graph, or None."""
    if not ENABLED or client() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception as e:
        log_event(logger, "tracing_degraded", level=logging.WARNING, reason=f"handler unavailable ({type(e).__name__})")
        return None


@contextmanager
def root_trace(name: str, *, session_id: str, metadata: dict, tags: list[str]):
    """One root span per graph run. Graph callbacks, node spans and LLM
    generations started inside it nest under it (they pick up the current
    OpenTelemetry context). `metadata` must be safe identifiers only — no
    note text. SDK errors here never reach the caller."""
    lf = client()
    if lf is None:
        yield None
        return
    stack = ExitStack()
    try:
        from langfuse import propagate_attributes
        span = stack.enter_context(lf.start_as_current_observation(as_type="span", name=name))
        stack.enter_context(propagate_attributes(
            session_id=session_id, trace_name=name, tags=[str(t) for t in tags],
            metadata={k: str(v)[:200] for k, v in metadata.items() if v is not None}))
    except Exception as e:
        stack.close()
        log_event(logger, "tracing_degraded", level=logging.WARNING, reason=f"root trace failed ({type(e).__name__})")
        yield None
        return
    try:
        yield span
    finally:
        try:
            stack.close()
        except Exception as e:
            log_event(logger, "tracing_degraded", level=logging.WARNING, reason=f"root trace close failed ({type(e).__name__})")


def annotate(**metadata) -> None:
    """Add safe metadata (ids, statuses) to the current span. No-op when off."""
    lf = client()
    if lf is None:
        return
    try:
        lf.update_current_span(metadata={k: v for k, v in metadata.items() if v is not None})
    except Exception as e:
        log_event(logger, "tracing_degraded", level=logging.WARNING, reason=f"annotate failed ({type(e).__name__})")


@contextmanager
def span(name: str, **attrs):
    lf = client()
    if lf is None:
        yield None
        return
    with lf.start_as_current_observation(as_type="span", name=name) as s:
        if attrs:
            s.update(input=attrs)
        yield s


@contextmanager
def generation(name: str, model: str, prompt=None):
    lf = client()
    if lf is None:
        yield None
        return
    with lf.start_as_current_observation(as_type="generation", name=name, model=model) as g:
        if prompt is not None:
            g.update(input=prompt)
        yield g


def flush() -> None:
    """Send buffered spans. Safe to call when tracing is off or already failed."""
    if _client is None:
        return
    try:
        _client.flush()
    except Exception as e:
        log_event(logger, "tracing_degraded", level=logging.WARNING, reason=f"flush failed ({type(e).__name__})")
