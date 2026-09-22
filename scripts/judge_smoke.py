"""Explicit live smoke probe for the local retrieval judge.

This is intentionally a script rather than a ``test_*.py`` module: it calls a
running Ollama service and must never execute during pytest discovery or module
import.

    python scripts/judge_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CASES = [
    ("abnormal potassium lab results", "Discharge labs: Potassium 6.1 mEq/L, critical high. Given kayexalate."),
    ("abnormal potassium lab results", "CBC: WBC 7.3 RBC 3.72 Hgb 11.2 Hct 33.8 Plt 210"),
    ("swollen legs fluid overload", "CERVICAL SPINE: vertebral body heights are preserved."),
]


def main() -> int:
    from src.evals.llm_judge import LLMJudge
    from src.llm.local_client import FAST_MODEL, build_call_fn

    judge = LLMJudge(
        model=FAST_MODEL,
        call_fn=build_call_fn(tier="fast", json_mode=True, max_tokens=200),
        max_workers=2,  # local single-GPU: high concurrency can thrash
    )
    for query, chunk in CASES:
        result = judge.judge(query, chunk)
        print(
            f"score={result.score}  relevant={result.is_relevant()}  "
            f"err={result.error}  reason={result.reason!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
