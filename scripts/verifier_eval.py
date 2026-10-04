"""Verifier diagnostic: run the unchanged verifier on labelled and adversarial claims.

    ./scripts/lumen research verifier-eval

Builds a development diagnostic set (see src/evals/verifier_diagnostic.py):
  real       human-adjudicated SUPPORTED claims from the development cohort, and
             claims templated from SQL rows by the structured lookups
  synthetic  deterministic adversarial variants of those claims, labelled by rule

Each example is given to the production `verification` node on its own, with
its cited sources, exactly as synthesis would hand it over. Claims the code
checks cannot settle go to the local model, so Ollama must be running.

The dataset holds patient-derived claims and stays under
~/Lumen_local_results/verifier_eval. Only aggregate counts are printed. The
frozen holdout is never read.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import storage  # noqa: E402
from src.agents import citations  # noqa: E402
from src.evals import verifier_diagnostic as vd  # noqa: E402

OUT_DIR = Path.home() / "Lumen_local_results" / "verifier_eval"
DEV_ADJUDICATION = Path.home() / "Lumen_local_results" / "scorecard" / "adjudication-sample-v2.jsonl"
RETRIEVAL_BENCHMARK = Path.home() / "Lumen_local_results" / "retrieval_eval" / "benchmark.json"
SEED = 20261005
STRUCTURED_QUESTIONS = ("What was the most recent creatinine?",
                        "How did creatinine change over the patient's available record?")
ADMISSION_QUESTION = "How many hospital admissions does the patient have, and when was the most recent one?"
EVIDENCE_KEY = {"S": "patient_evidence", "G": "guideline_evidence", "P": "literature_evidence",
                "L": "lab_evidence", "A": "encounter_evidence"}


def human_bases() -> list[dict]:
    """Development claims a blinded human reviewer marked SUPPORTED."""
    if not DEV_ADJUDICATION.exists():
        return []
    rows = [json.loads(line) for line in DEV_ADJUDICATION.read_text().splitlines() if line.strip()]
    return [{"claim": r["claim"], "evidence": {l: s.get("excerpt") or "" for l, s in r["sources"].items()},
             "gold": vd.SUPPORTED, "category": "real_supported", "synthetic": False, "origin": "human_adjudicated"}
            for r in rows if r.get("human_verdict") == "SUPPORTED" and r.get("sources")
            and (r.get("hidden") or {}).get("cohort", "fixture") != "holdout"]


def structured_bases(graph_mod, subject_ids: list[int]) -> list[dict]:
    """Claims the structured lookups template from SQL rows: true by construction."""
    out = []
    for sid in subject_ids:
        for query in STRUCTURED_QUESTIONS:
            st = graph_mod.lab_lookup({"query": query, "subject_id": sid})
            evidence = {e["label"]: e["text"] for e in st.get("lab_evidence") or []}
            for c in st.get("citations") or []:
                out.append({"claim": c["claim"], "evidence": evidence, "gold": vd.SUPPORTED, "category": "real_supported",
                            "synthetic": False, "origin": "structured_sql",
                            "has_other_dates": len(set(re.findall(r"\d{4}-\d{2}-\d{2}", "".join(evidence.values())))) > 1})
                # an explicit directional claim, built from the same row values
                m = re.search(r"first value was ([\d.]+)(.*?) on (\S+) and the most recent was ([\d.]+)(.*?) on (\S+) \[(L\d+)\]", c["claim"])
                if m and float(m.group(1)) != float(m.group(4)):
                    word = "increased" if float(m.group(4)) > float(m.group(1)) else "decreased"
                    out.append({"claim": f"The value {word} from {m.group(1)}{m.group(2)} on {m.group(3)} to "
                                         f"{m.group(4)}{m.group(5)} on {m.group(6)} [{m.group(7)}].",
                                "evidence": evidence, "gold": vd.SUPPORTED, "category": "real_supported",
                                "synthetic": False, "origin": "structured_sql"})
        st = graph_mod.encounter_lookup({"query": ADMISSION_QUESTION, "subject_id": sid})
        evidence = {e["label"]: e["text"] for e in st.get("encounter_evidence") or []}
        many = len(re.findall(r"\d{4}-\d{2}-\d{2} ->", "".join(evidence.values()))) > 1
        for c in st.get("citations") or []:
            out.append({"claim": c["claim"], "evidence": evidence, "gold": vd.SUPPORTED, "category": "real_supported",
                        "synthetic": False, "origin": "structured_sql", "has_other_dates": many})
    return out


def to_state(example: dict) -> dict:
    """The state synthesis would hand to verification for this one claim."""
    state: dict = {"draft_answer": example["claim"]}
    evidence = []
    for label, body in example["evidence"].items():
        item = {"label": label, "text": body, "chunk_id": -1, "source_type": "note", "note_type": "note",
                "charttime": None, "score": 1.0}
        if label.startswith("P"):
            item["backend"] = "pubmed"
        state.setdefault(EVIDENCE_KEY.get(label[:1], "patient_evidence"), []).append(item)
        evidence.append(item)
    valid = citations.validate(example["claim"], evidence)["claims"]
    labels = [l for c in valid for l in c["valid_labels"]]
    state["citations"] = [{"claim": example["claim"], "label": labels[0] if labels else "", "labels": labels,
                           "chunk_id": -1, "verified": False, "verification_note": ""}]
    return state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--patients", type=int, default=6, help="development patients used for SQL-templated claims")
    p.add_argument("--structured-cap", type=int, default=18)
    p.add_argument("--per-category", type=int, default=8, help="synthetic variants kept per error type")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    if storage.DATA_PLANE != "research" or storage.engine.url.database == "lumen_holdout":
        print("refusing: the verifier diagnostic runs on the research plane and never on the holdout", file=sys.stderr)
        return 2
    import logging
    logging.disable(logging.WARNING)
    import src.agents.graph as graph_mod

    rng = random.Random(args.seed)
    subjects = sorted({q["subject_id"] for q in json.loads(RETRIEVAL_BENCHMARK.read_text())["questions"]})[:args.patients] \
        if RETRIEVAL_BENCHMARK.exists() else []
    human = human_bases()
    structured = structured_bases(graph_mod, subjects)
    rng.shuffle(structured)
    bases = human + structured[:args.structured_cap]

    pool: dict[str, list[dict]] = {}
    for base in human + structured:                         # variants may come from any true claim
        for v in vd.variants(base):
            pool.setdefault(v["category"], []).append(v)
    synthetic = []
    for category in sorted(pool):
        rng.shuffle(pool[category])
        seen, kept = set(), []
        for v in pool[category]:
            if v["claim"] not in seen and len(kept) < args.per_category:
                seen.add(v["claim"])
                kept.append(v)
        synthetic += kept
    examples = bases + synthetic

    rows = []
    for i, ex in enumerate(examples, 1):
        out = graph_mod.verification(to_state(ex))
        claim = out["citations"][0]
        trace = (out["verification"].get("claims") or [{}])[0]
        rows.append({"id": i, "gold": ex["gold"], "category": ex["category"], "synthetic": ex["synthetic"],
                     "origin": ex["origin"], "approved": bool(claim["verified"]), "method": trace.get("stage"),
                     "verifier_note": claim.get("verification_note"), "claim": ex["claim"]})

    real = [r for r in rows if not r["synthetic"]]
    origins = {}
    for r in real:
        o = "human_adjudicated" if r["origin"] == "human_adjudicated" else "structured_sql"
        a, n = origins.get(o, (0, 0))
        origins[o] = (a + r["approved"], n + 1)
    summary = {"name": "verifier diagnostic (development data + synthetic adversarial variants)", "seed": args.seed,
               "overall": vd.metrics(rows), "real": vd.metrics(real),
               "synthetic": vd.metrics([r for r in rows if r["synthetic"]]),
               "real_by_origin": {k: f"{a}/{n}" for k, (a, n) in origins.items()},
               "by_category": vd.by_category(rows),
               "decided_by": {m: sum(1 for r in rows if r["method"] == m) for m in sorted({str(r["method"]) for r in rows})}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.chmod(0o700)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dataset = OUT_DIR / f"diagnostic-{stamp}.jsonl"
    dataset.write_text("".join(json.dumps(r) + "\n" for r in rows))
    dataset.chmod(0o600)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    print(vd.render(summary))
    print(f"\ndecided by: {summary['decided_by']}")
    print(f"examples with claims (patient-derived, local only): {dataset}\naggregate: {OUT_DIR}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
