"""
Local Ollama backend for the Lumen LLM judge
=============================================
Wires a local Ollama model (default Qwen2.5-14B-Instruct) into LLMJudge through
its `call_fn` hook, so no clinical text ever leaves the machine — which keeps
you inside the PhysioNet MIMIC Data Use Agreement (no third-party API, no
retention). When `call_fn` is set, LLMJudge never imports the Groq SDK or reads
GROQ_API_KEY.

Requires a running Ollama server (`ollama serve`) with the model pulled:
    ollama pull qwen2.5:14b

Usage:
    from src.evals.ollama_backend import make_ollama_judge
    judge = make_ollama_judge(model="qwen2.5:14b")
"""
from __future__ import annotations

import requests
from typing import Callable

from src.evals.llm_judge import LLMJudge

DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"          # Q4_K_M instruct, ~9 GB on disk
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def make_ollama_call_fn(
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    num_ctx: int = 2048,          # judge prompt is tiny -> small ctx = small KV cache
    temperature: float = 0.0,     # deterministic grading
    timeout: float = 180.0,
    keep_alive: str = "30m",      # keep the model warm between calls
) -> Callable[[list], str]:
    """Return a call_fn(messages) -> str that talks to a local Ollama server."""
    url = f"{host.rstrip('/')}/api/chat"

    def _call(messages: list) -> str:
        resp = requests.post(url, timeout=timeout, json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",          # force a JSON object out of the model
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": 256,
            },
        })
        resp.raise_for_status()        # raise -> LLMJudge retries, then degrades to 0
        return resp.json()["message"]["content"]

    return _call


def make_ollama_judge(
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    num_ctx: int = 2048,
    cache_path: str = ".cache/llm_judge_ollama.json",
    max_workers: int = 2,          # 14B on 16 GB: low concurrency avoids swap
    **judge_kwargs,
) -> LLMJudge:
    """
    Build an LLMJudge backed by local Ollama.

    `model` is also passed to LLMJudge so the disk cache keys stay honest — the
    cache is keyed on (prompt_version, model, query, chunk), so a Qwen run and a
    (hypothetical) Groq run won't collide.
    """
    call_fn = make_ollama_call_fn(model=model, host=host, num_ctx=num_ctx)
    return LLMJudge(
        model=model,
        call_fn=call_fn,
        cache_path=cache_path,
        max_workers=max_workers,
        **judge_kwargs,
    )
