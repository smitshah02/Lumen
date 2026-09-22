"""
Lumen package init.

Loads .env exactly once, here, because Python runs this before any `src.*`
submodule on every `python -m src.x` and every `from src.x import ...`.

This has to happen at package-import time, not in each entry point's main():
LUMEN_TRACING (src/obs/tracing.py), DATABASE_URL (src/storage/__init__.py),
LUMEN_LLM_MAIN/FAST (src/llm/local_client.py), LUMEN_DATA_PLANE
(src/config.py) and LUMEN_HNSW_EF_SEARCH are all read at *import*
time, so a load_dotenv() inside main() would arrive too late for all of them.

Before this existed, only an archived research utility called load_dotenv(), so
.env was inert: tracing was silently off, the MIMIC data dirs were unset, and
DATABASE_URL worked only because the hardcoded fallback in
src/storage/__init__.py happened to match the file.

override=False on purpose — a variable already exported in the shell beats
.env, so `LUMEN_HNSW_EF_SEARCH=200 python -m src.evals.retrieve_pool --out
pooled.json --limit 1` still does what it looks like it does.
"""

from pathlib import Path as _Path

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # python-dotenv is optional; the app must still start
    pass
else:
    # Explicit path rather than find_dotenv(): find_dotenv() walks the caller's
    # stack frames and raises AssertionError when there is no caller frame
    # (python -c, exec, some test runners).
    _ENV_PATH = _Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.is_file():
        _load_dotenv(_ENV_PATH, override=False)
