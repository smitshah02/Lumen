"""Control replay (data-foundation plan, decision R7).

Under the control profile, retrieval for the 15 frozen baseline questions must
return the same chunk ids in the same order as
reports/data_foundation/before_benchmark.json. That file is git-ignored
(MIMIC-derived), and the replay needs the research database and local models,
so this test skips when either is missing.

    python tests/test_control_replay.py     # run the replay directly
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "reports" / "data_foundation" / "before_benchmark.json"
SKIP = 77


def replay() -> int:
    sys.path.insert(0, str(ROOT))
    from src.storage import check_connection
    if not check_connection():
        return SKIP
    from src.agents.graph import patient_retrieval
    from src.retrieval.hybrid_retriever_v2 import detect_temporal_mode

    failed = 0
    for item in json.loads(BASELINE.read_text())["items"]:
        want = [s["chunk_id"] for s in item["result"]["sources"] if s["source_type"] == "note"]
        state = {"query": item["question"], "subject_id": item["subject_id"],
                 "temporal_mode": detect_temporal_mode(item["question"])}
        got = [e["chunk_id"] for e in patient_retrieval(state)["patient_evidence"]]
        if got != want:
            failed += 1
            print(f"{item['qid']}: want {want} got {got}")
    print(f"control replay: {15 - failed}/15 identical")
    return 1 if failed else 0


def test_control_replay_returns_the_baseline_chunks():
    import pytest
    if not BASELINE.exists():
        pytest.skip("baseline artifact not present (git-ignored)")
    env = {**os.environ, "LUMEN_DATA_PLANE": "research", "LUMEN_DATA_PROFILE": "control",
           "LUMEN_TRACING": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    run = subprocess.run([sys.executable, __file__], env=env, cwd=ROOT, capture_output=True, text=True)
    if run.returncode == SKIP:
        pytest.skip("research database not reachable")
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]


if __name__ == "__main__":
    sys.exit(replay())
