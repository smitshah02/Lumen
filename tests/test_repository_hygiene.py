import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dependency_free_repository_checks_pass():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/offline_checks.py")],
        cwd=ROOT,
        env={**os.environ, "LUMEN_OFFLINE_ALLOW_NONCANONICAL_PYTHON": "1"},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
