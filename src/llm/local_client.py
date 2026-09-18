"""
Local LLM Client (Ollama)
=========================
Single inference path for every LLM job in Lumen: triage, judging,
synthesis, verification. Nothing here leaves the machine.

Two model tiers, set in .env (both default to qwen3:8b):
  LUMEN_LLM_FAST  — triage, judging
  LUMEN_LLM_MAIN  — synthesis, verification
  LUMEN_LLM_HOST  — Ollama base URL (default http://localhost:11434)

    python -m src.llm.local_client --config   # configured models + reachability, no prompts

Why native /api/chat and not the OpenAI-compat endpoint:
  we need `keep_alive` (to evict a model and free RAM) and `num_ctx`
  (Ollama's default context is far too small for clinical chunks).

Usage:
    from src.llm.local_client import chat, build_call_fn, unload

    text = chat([{"role": "user", "content": "hi"}], tier="fast")

    # Drop-in for LLMJudge's injectable call_fn:
    judge = LLMJudge(model=FAST_MODEL, call_fn=build_call_fn("fast"))
"""

from __future__ import annotations

import os
import time
import random
import logging
from typing import Callable, Optional

import requests
import re

from src.obs import tracing

logger = logging.getLogger(__name__)

HOST = os.environ.get("LUMEN_LLM_HOST", "http://localhost:11434")
MAIN_MODEL = os.environ.get("LUMEN_LLM_MAIN", "qwen3:8b")
FAST_MODEL = os.environ.get("LUMEN_LLM_FAST", "qwen3:8b")

# Context windows. Ollama defaults are far below what clinical work needs:
# a judged chunk runs ~1k tokens, a synthesis prompt with 8 chunks runs ~6k+.
# Raising num_ctx costs KV-cache RAM, so keep the fast tier modest.
CTX_FAST = 8192
CTX_MAIN = 12288

# Seconds a model stays resident after its last call. On a 16GB Mac set
# LUMEN_LLM_KEEPALIVE=30s so the 14B evicts before the reranker needs MPS.
KEEP_ALIVE = os.environ.get("LUMEN_LLM_KEEPALIVE", "10m")
_SUPPORTS_THINK_PARAM = True

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _resolve(tier: str, model: Optional[str]) -> tuple[str, int]:
    if model:
        return model, (CTX_MAIN if tier == "main" else CTX_FAST)
    if tier == "main":
        return MAIN_MODEL, CTX_MAIN
    return FAST_MODEL, CTX_FAST


def chat(
    messages: list[dict],
    tier: str = "fast",
    model: Optional[str] = None,
    json_mode: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 512,
    num_ctx: Optional[int] = None,
    keep_alive: Optional[str] = None,
    timeout: float = 300.0,
    max_retries: int = 3,
    think: bool = False,
) -> str:
    """Send a message list, get a string back. Raises on final failure."""
    resolved_model, default_ctx = _resolve(tier, model)

    global _SUPPORTS_THINK_PARAM

    payload = {
        "model": resolved_model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive or KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": num_ctx or default_ctx,
        },
    }
    if json_mode:
        payload["format"] = "json"
    if _SUPPORTS_THINK_PARAM:
        payload["think"] = think

    last_exc = None
    with tracing.generation(f"ollama:{tier}", resolved_model, prompt=messages) as gen:
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(f"{HOST}/api/chat", json=payload, timeout=timeout)
                # Older Ollama builds reject the `think` field; drop it and retry once.
                if resp.status_code == 400 and "think" in payload:
                    logger.warning("ollama rejected `think`; falling back to tag stripping")
                    _SUPPORTS_THINK_PARAM = False
                    payload.pop("think")
                    continue
                resp.raise_for_status()
                body = resp.json()
                text_out = _strip_thinking(body["message"]["content"])
                if gen is not None:
                    gen.update(output=text_out, usage_details={
                        "input": body.get("prompt_eval_count", 0),
                        "output": body.get("eval_count", 0),
                    })
                return text_out
            except Exception as e:
                last_exc = e
                if attempt == max_retries:
                    break
                delay = 1.5 * (2 ** attempt) + random.uniform(0, 1.0)
                logger.debug(f"local LLM call failed (attempt {attempt + 1}): {e}; retry in {delay:.1f}s")
                time.sleep(delay)
    raise RuntimeError(f"local LLM call failed after {max_retries + 1} attempts: {last_exc}")


def build_call_fn(tier: str = "fast", json_mode: bool = True, **kw) -> Callable[[list], str]:
    """
    Returns a `call_fn(messages) -> str` matching LLMJudge's injection point.
    JSON mode is baked in here because call_fn only receives messages.
    """
    def _fn(messages: list[dict]) -> str:
        return chat(messages, tier=tier, json_mode=json_mode, **kw)
    return _fn


def unload(tier: str = "main", model: Optional[str] = None) -> None:
    """Evict a model from memory immediately. Call before heavy MPS work."""
    resolved_model, _ = _resolve(tier, model)
    try:
        requests.post(
            f"{HOST}/api/chat",
            json={"model": resolved_model, "messages": [], "keep_alive": 0},
            timeout=30,
        )
        logger.info(f"unloaded {resolved_model}")
    except Exception as e:
        logger.warning(f"could not unload {resolved_model}: {e}")


def runtime_config() -> dict:
    """Configured host and model tags. Nothing here is secret."""
    return {"host": HOST, "main_model": MAIN_MODEL, "fast_model": FAST_MODEL,
            "keep_alive": KEEP_ALIVE, "ctx_main": CTX_MAIN, "ctx_fast": CTX_FAST}


def health() -> dict:
    """Check the server is up and both configured models exist."""
    out = {"host": HOST, "reachable": False, "models": [], "main_ok": False, "fast_ok": False}
    try:
        r = requests.get(f"{HOST}/api/tags", timeout=10)
        r.raise_for_status()
        out["reachable"] = True
        out["models"] = [m["name"] for m in r.json().get("models", [])]
        out["main_ok"] = any(m.split(":")[0] == MAIN_MODEL.split(":")[0] for m in out["models"])
        out["fast_ok"] = any(m.split(":")[0] == FAST_MODEL.split(":")[0] for m in out["models"])
    except Exception as e:
        out["error"] = str(e)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if "--config" in __import__("sys").argv:
        import json
        h = health()
        print(json.dumps({**runtime_config(), "reachable": h["reachable"],
                          "main_ok": h["main_ok"], "fast_ok": h["fast_ok"]}, indent=2))
        raise SystemExit(0 if h["reachable"] and h["main_ok"] and h["fast_ok"] else 1)
    h = health()
    print("=" * 60)
    print("  LOCAL LLM HEALTH")
    print("=" * 60)
    print(f"  host       {h['host']}  {'OK' if h['reachable'] else 'UNREACHABLE'}")
    print(f"  installed  {', '.join(h['models']) or '(none)'}")
    print(f"  main       {MAIN_MODEL}  {'OK' if h['main_ok'] else 'MISSING'}")
    print(f"  fast       {FAST_MODEL}  {'OK' if h['fast_ok'] else 'MISSING'}")
    if not h["reachable"]:
        raise SystemExit("start Ollama first:  brew services start ollama")

    print("\n  fast tier, JSON mode:")
    t0 = time.time()
    print("   ", chat(
        [{"role": "system", "content": 'Reply with only {"ok": true}.'},
         {"role": "user", "content": "ping"}],
        tier="fast", json_mode=True, max_tokens=32,
    ).strip(), f"({time.time() - t0:.1f}s)")

    print("\n  main tier, prose:")
    t0 = time.time()
    print("   ", chat(
        [{"role": "user", "content": "In one sentence, what is hyperkalemia?"}],
        tier="main", max_tokens=80,
    ).strip(), f"\n    ({time.time() - t0:.1f}s)")