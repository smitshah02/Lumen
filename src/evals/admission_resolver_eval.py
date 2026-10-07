"""Development accuracy check for the model-assisted admission resolver (plan E10).

Runs descriptive admission references with a known answer through the real
candidate loader, the local model and the code-checked acceptance rule.

    python -m src.evals.admission_resolver_eval [cases.json] [--out results.json]

The cases file is MIMIC-derived (subject and admission ids) and lives in the
git-ignored reports/data_foundation/. Each case is
{"subject_id", "question", "expected_hadm_id" (or null when no single admission
is right)}. These are development cases; held-out questions are not used here.

The number that matters most is `wrong_admission_applied`: a scope applied to
an admission other than the expected one. Abstaining is always safe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CASES = Path("reports/data_foundation/admission_resolver_dev.json")


def score(cases: list[dict], resolve) -> dict:
    """`resolve(subject_id, question)` -> AdmissionResolution. Returns counts and per-case outcomes."""
    rows = []
    for case in cases:
        res = resolve(case["subject_id"], case["question"])
        expected = case.get("expected_hadm_id")
        if res.status == "resolved":
            outcome = "correct" if res.hadm_id == expected else "wrong_admission_applied"
        else:
            outcome = "safe_abstention" if expected is None else "missed"
        rows.append({"subject_id": case["subject_id"], "expected_hadm_id": expected, "status": res.status,
                     "hadm_id": res.hadm_id, "outcome": outcome, "reason": res.reason})
    counts = {k: sum(1 for r in rows if r["outcome"] == k)
              for k in ("correct", "safe_abstention", "missed", "wrong_admission_applied")}
    answerable = sum(1 for c in cases if c.get("expected_hadm_id") is not None)
    return {"cases": len(rows), "answerable": answerable, **counts,
            "accuracy_on_answerable": round(counts["correct"] / answerable, 3) if answerable else None, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="?", default=str(DEFAULT_CASES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from src.agents.admission_scope import load_candidates, resolve_with_model
    from src.agents.graph import _ask_admission_model
    cases = json.loads(Path(args.cases).read_text())
    result = score(cases, lambda sid, q: resolve_with_model(q, load_candidates(sid), _ask_admission_model))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    for r in result["rows"]:
        print(f"  subject {r['subject_id']} expected {r['expected_hadm_id']} -> {r['status']} {r['hadm_id']} [{r['outcome']}]")
    return 1 if result["wrong_admission_applied"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
