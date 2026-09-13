"""
Tracing
=======
Self-hosted Langfuse only. Traces carry note text in span inputs, so this
must never point at Langfuse Cloud — that would be the same DUA violation
the egress gate exists to prevent, arriving through the back door.

Everything degrades to a no-op when LUMEN_TRACING is unset, so the graph
runs identically with tracing off.
"""

from __future__ import annotations

import os
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("LUMEN_TRACING", "0") == "1"
_client = None
_init_failed = False      # sticky: one connection attempt per process

# The docstring above is a rule, so enforce it rather than trusting it: a
# hostname outside the machine would ship note text off-box the moment
# someone copies a cloud LANGFUSE_HOST into .env.
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal")


def _host_is_local() -> bool:
    from urllib.parse import urlparse
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    return (urlparse(host).hostname or "") in _LOCAL_HOSTS


def client():
    """Cached client, or None. Initialisation is attempted exactly once.

    A failed attempt is sticky. An unreachable Langfuse makes auth_check
    block until it times out, and every span and every LLM call goes
    through here — without this, a paused container would stall a whole
    graph run instead of degrading quietly. The tradeoff is deliberate:
    a Langfuse that comes up mid-run is not picked up until the next
    process starts.
    """
    global _client, _init_failed
    if not ENABLED or _init_failed:
        return None
    if _client is None:
        if not _host_is_local():
            logger.error("LANGFUSE_HOST is not local; tracing disabled (traces carry note text)")
            _init_failed = True
            return None
        try:
            from langfuse import get_client
            _client = get_client()
            if not _client.auth_check():
                logger.warning("langfuse auth failed; tracing disabled for this process")
                _client, _init_failed = None, True
        except Exception as e:
            logger.warning(f"langfuse unavailable ({e}); tracing disabled for this process")
            _client, _init_failed = None, True
    return _client


def handler():
    """LangChain CallbackHandler for the graph, or None."""
    if not ENABLED or client() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception as e:
        logger.warning(f"langfuse handler unavailable: {e}")
        return None


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
    lf = client()
    if lf is not None:
        lf.flush()
