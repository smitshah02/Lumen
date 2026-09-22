#!/usr/bin/env python3
"""Dependency-free repository checks for CI and clean-environment diagnosis."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def fail(message: str) -> None:
    raise SystemExit(f"offline check failed: {message}")


def check_python() -> None:
    if sys.version_info[:2] != (3, 12):
        if os.environ.get("LUMEN_OFFLINE_ALLOW_NONCANONICAL_PYTHON") == "1":
            return
        fail(f"CPython 3.12 required, got {sys.version.split()[0]}")


def check_python_syntax() -> None:
    roots = (ROOT / "src", ROOT / "scripts", ROOT / "tests")
    for base in roots:
        for path in base.rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeError) as exc:
                fail(f"{path.relative_to(ROOT)}: {exc}")


def check_json() -> None:
    paths = [ROOT / "configs/models.json", *sorted((ROOT / "src/demo_data").glob("*.json"))]
    for path in paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"{path.relative_to(ROOT)}: {exc}")


def check_model_defaults() -> None:
    registry = json.loads((ROOT / "configs/models.json").read_text())
    example = (ROOT / ".env.example").read_text()
    expected = {
        "LUMEN_LLM_MAIN": registry["ollama"]["runtime_main"],
        "LUMEN_LLM_FAST": registry["ollama"]["runtime_fast"],
        "LUMEN_JUDGE_MODEL": registry["ollama"]["independent_judge"],
    }
    for key, value in expected.items():
        if f"{key}={value}" not in example:
            fail(f".env.example {key} does not match configs/models.json")


def check_archived_code_is_inactive() -> None:
    for base in (ROOT / "src", ROOT / "scripts"):
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if re.search(r"(?:from|import)\s+archive(?:\.|\s)", text):
                fail(f"active import references archive: {path.relative_to(ROOT)}")


def check_markdown_links() -> None:
    docs = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md")), ROOT / "archive/README.md"]
    for document in docs:
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                fail(f"broken link in {document.relative_to(ROOT)}: {target}")


def check_documented_configuration() -> None:
    reference = (ROOT / "docs/configuration.md").read_text()
    supported = {
        "DATABASE_URL", "LUMEN_DATA_PLANE", "LUMEN_DEMO_DATABASE_URL",
        "LUMEN_DEMO_DB_NAME", "LUMEN_DB_CONNECT_TIMEOUT", "LUMEN_PG_PASSWORD",
        "LUMEN_PG_PORT", "LUMEN_PG_VOLUME", "LUMEN_DEMO_PG_PASSWORD",
        "LUMEN_CHECKPOINT_AUTO_SETUP", "LUMEN_LLM_HOST", "LUMEN_LLM_MAIN",
        "LUMEN_LLM_FAST", "LUMEN_LLM_KEEPALIVE", "LUMEN_LLM_WARMUP",
        "LUMEN_DETERMINISTIC_LABS", "LUMEN_RESEARCH_ALLOW_REMOTE_MODELS",
        "LUMEN_JUDGE_MODEL", "LUMEN_JUDGE_HOST", "LUMEN_JUDGE_NUM_CTX",
        "LUMEN_JUDGE_MAX_TOKENS", "LUMEN_JUDGE_CACHE", "LUMEN_DATA_DIR",
        "MIMIC_IV_DIR", "MIMIC_IV_NOTE_DIR", "LUMEN_GUIDELINES_DIR",
        "LUMEN_MODELS_DIR", "LUMEN_EVAL_RESULTS_ROOT", "LUMEN_RRF_BM25_WEIGHT",
        "LUMEN_RRF_VECTOR_WEIGHT", "LUMEN_QUERY_EXPANSION", "LUMEN_HNSW_EF_SEARCH",
        "LUMEN_API_PORT", "LUMEN_API_BIND", "LUMEN_API_ALLOW_NONLOCAL",
        "LUMEN_LOG_LEVEL", "LUMEN_TRACING", "LANGFUSE_BASE_URL", "LANGFUSE_HOST",
        "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LUMEN_PYTHON", "LUMEN_ROOT",
        "LUMEN_RUNTIME_ROOT", "LUMEN_REPO_URL", "LUMEN_DEMO_LLM_HOST",
        "HF_HOME", "HF_HUB_OFFLINE", "OLLAMA_MODELS",
    }
    missing = sorted(name for name in supported if f"`{name}`" not in reference)
    if missing:
        fail("configuration variables undocumented: " + ", ".join(missing))


def main() -> int:
    checks = (
        check_python,
        check_python_syntax,
        check_json,
        check_model_defaults,
        check_archived_code_is_inactive,
        check_markdown_links,
        check_documented_configuration,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
