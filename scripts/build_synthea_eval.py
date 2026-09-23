#!/usr/bin/env python3
"""Build or verify deterministic Synthea final-evaluation gold artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evals.final_eval.synthea_cases import PROFILES, build, validate_artifacts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    result = (validate_artifacts(args.profile) if args.verify_only else build(args.profile))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
