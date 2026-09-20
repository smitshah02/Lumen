"""
Independent judge backend
=========================
The offline judge must not be the runtime verifier wearing a different hat.
This module is the only place a judge model is resolved, and it REFUSES to
return a backend whose model is either runtime tier:

    LUMEN_LLM_MAIN   the synthesis model — judging its own output is not an
                     independent assessment
    LUMEN_LLM_FAST   the runtime claim verifier — the thing whose routing
                     decisions the evaluation is trying to measure

There is deliberately no fallback. If the configured judge is unavailable the
call raises JudgeUnavailable, the affected cases are recorded as judge
failures, and the summary counts them. A judge failure never becomes a zero, a
passing score, or a silent substitution of the runtime model.
"""

from __future__ import annotations

import os
import json
import time
import random
import logging

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = os.environ.get("LUMEN_JUDGE_MODEL", "qwen2.5:14b")
DEFAULT_JUDGE_HOST = os.environ.get("LUMEN_JUDGE_HOST",
                                    os.environ.get("LUMEN_LLM_HOST", "http://localhost:11434"))
JUDGE_NUM_CTX = int(os.environ.get("LUMEN_JUDGE_NUM_CTX", "8192"))
JUDGE_MAX_TOKENS = int(os.environ.get("LUMEN_JUDGE_MAX_TOKENS", "900"))


class JudgeUnavailable(RuntimeError):
    """The configured independent judge could not be used. Never caught and
    converted into a score."""


class JudgeNotIndependent(RuntimeError):
    """The configured judge model is a runtime tier. Refused outright."""


def runtime_models() -> dict:
    """The models the online system uses, read live rather than hardcoded."""
    try:
        from src.llm import local_client
        return {"main": local_client.MAIN_MODEL, "fast": local_client.FAST_MODEL}
    except Exception as e:                      # pragma: no cover - import guard
        logger.warning("could not read runtime models (%s)", type(e).__name__)
        return {"main": None, "fast": None}


def _norm_tag(tag: str) -> str:
    """'qwen3:4b' and 'qwen3:4b:latest' are the same model."""
    t = (tag or "").strip().lower()
    return t[:-7] if t.endswith(":latest") else t


def assert_independent(model: str, runtime: dict | None = None) -> None:
    rt = runtime if runtime is not None else runtime_models()
    tag = _norm_tag(model)
    for role in ("main", "fast"):
        if rt.get(role) and _norm_tag(rt[role]) == tag:
            raise JudgeNotIndependent(
                f"judge model {model!r} is the runtime {role} model. The offline "
                f"judge must be independent of the system under evaluation. Set "
                f"LUMEN_JUDGE_MODEL to a different model.")


class OllamaJudgeBackend:
    """Deterministic (temperature 0) structured-output calls to an Ollama host."""

    kind = "ollama"

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, host: str = DEFAULT_JUDGE_HOST,
                 num_ctx: int = JUDGE_NUM_CTX, max_tokens: int = JUDGE_MAX_TOKENS,
                 retries: int = 3, timeout: int = 300, runtime: dict | None = None):
        assert_independent(model, runtime)
        self.model, self.host = model, host.rstrip("/")
        self.num_ctx, self.max_tokens = num_ctx, max_tokens
        self.retries, self.timeout = retries, timeout

    def describe(self) -> dict:
        from urllib.parse import urlparse
        return {"backend": self.kind, "model": self.model,
                "host": urlparse(self.host).netloc or self.host,   # no credentials
                "temperature": 0.0, "num_ctx": self.num_ctx,
                "max_tokens": self.max_tokens, "retries": self.retries}

    def digest(self) -> dict:
        try:
            import requests
            tags = requests.get(f"{self.host}/api/tags", timeout=10).json().get("models", [])
            m = next((m for m in tags if _norm_tag(m.get("name")) == _norm_tag(self.model)), {})
            return {"digest": (m.get("digest") or "")[:12] or None,
                    "parameter_size": (m.get("details") or {}).get("parameter_size"),
                    "quantization": (m.get("details") or {}).get("quantization_level"),
                    "installed": bool(m)}
        except Exception as e:
            return {"digest": None, "installed": None, "error": type(e).__name__}

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        import requests
        payload = {
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": 0.0, "top_p": 1.0, "seed": 0,
                        "num_ctx": self.num_ctx, "num_predict": self.max_tokens},
        }
        if schema:
            payload["format"] = schema
        last = None
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
                r.raise_for_status()
                return (r.json().get("message") or {}).get("content") or ""
            except Exception as e:
                last = e
                if attempt < self.retries:
                    time.sleep(min(8.0, 0.75 * 2 ** (attempt - 1)) + random.uniform(0, 0.4))
        raise JudgeUnavailable(f"{type(last).__name__}: {last}") from last


class CallableJudgeBackend:
    """A backend wrapping an injected callable. For tests and for plugging in a
    non-Ollama provider without touching any scoring logic."""

    kind = "callable"

    def __init__(self, fn, model: str = "injected", runtime: dict | None = None,
                 check_independence: bool = True):
        if check_independence:
            assert_independent(model, runtime)
        self.fn, self.model = fn, model

    def describe(self) -> dict:
        return {"backend": self.kind, "model": self.model, "temperature": 0.0}

    def digest(self) -> dict:
        return {"digest": None, "installed": True}

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        return self.fn(system, user, schema)


def build_backend(model: str | None = None, host: str | None = None, **kw):
    """The configured independent judge. Raises rather than degrading."""
    return OllamaJudgeBackend(model=model or DEFAULT_JUDGE_MODEL,
                              host=host or DEFAULT_JUDGE_HOST, **kw)
