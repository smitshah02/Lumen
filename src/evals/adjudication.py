"""Independent human adjudication of claim support.

The scorecard's "claims supported" figure is Lumen's verifier judging Lumen's
answers. This module is the independent check: a person reads each claim next
to the source it cites and decides, WITHOUT seeing what the verifier said.

    build_sample()   a seeded, stratified sample of claims from scorecard runs
    blind_view()     exactly what the reviewer is shown — nothing else
    summarize()      human labels vs verifier labels, after labelling is done

Nothing here calls a model, and nothing assigns a human label.
"""

from __future__ import annotations

import random

VERDICTS = ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "UNCLEAR", "SKIP")
JUDGED = ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED")          # verdicts that count in the comparison
SOURCE_TYPES = {"S": "patient note", "L": "structured lab result", "A": "admissions record",
                "G": "guideline", "P": "published literature (PubMed)"}
DEFAULT_SEED = 20261004
STRUCTURED_CAP = 3      # [L#]/[A#] claims are templated from SQL; a few are enough


def source_kind(row: dict) -> str:
    labels = row.get("labels") or []
    return str(labels[0])[:1] if labels else "none"


def build_sample(rows: list[dict], target: int = 25, seed: int = DEFAULT_SEED,
                 structured_cap: int = STRUCTURED_CAP) -> list[dict]:
    """A reproducible sample spread over source type x verifier verdict, then
    over patients and question types. Structured-table claims are capped so
    model-written claims fill the sample. Order is shuffled so position says
    nothing about the verifier's opinion."""
    rng = random.Random(seed)
    unique: dict[tuple, dict] = {}
    for row in rows:
        if (row.get("claim") or "").strip():
            unique.setdefault((row.get("thread_id"), row.get("claim")), row)
    strata: dict[tuple, list[dict]] = {}
    for row in sorted(unique.values(), key=lambda r: (str(r.get("thread_id")), r.get("claim"))):
        strata.setdefault((source_kind(row), bool(row.get("verifier_verdict"))), []).append(row)
    for members in strata.values():
        rng.shuffle(members)
        # spread within a stratum: a new (patient, question type) before a repeat
        seen: set = set()
        members.sort(key=lambda r: ((r.get("subject_id"), r.get("case")) in seen
                                    or seen.add((r.get("subject_id"), r.get("case"))) or False))
    for kind in ("L", "A"):
        mine = [k for k in strata if k[0] == kind]
        keep = structured_cap
        for k in sorted(mine):
            strata[k] = strata[k][:keep]
            keep -= len(strata[k])
    picked: list[dict] = []
    keys = sorted(strata)
    while len(picked) < target and any(strata[k] for k in keys):
        for k in keys:                                   # round-robin: every stratum is represented first
            if strata[k] and len(picked) < target:
                picked.append(strata[k].pop(0))
    rng.shuffle(picked)
    return [{
        "id": i, "claim": row["claim"], "labels": row.get("labels") or [],
        "sources": {label: {"type": SOURCE_TYPES.get(str(label)[:1], "unknown"), "excerpt": excerpt}
                    for label, excerpt in (row.get("sources") or {}).items()},
        "human_verdict": None, "human_note": "",
        # Never shown while labelling.
        "hidden": {"verifier_verdict": bool(row.get("verifier_verdict")), "verifier_note": row.get("verifier_note"),
                   "case": row.get("case"), "subject_id": row.get("subject_id"), "thread_id": row.get("thread_id"),
                   "cohort": row.get("cohort")},
    } for i, row in enumerate(picked, 1)]


def blind_view(row: dict) -> dict:
    """The only fields a reviewer sees: the claim and what it cites."""
    return {"id": row["id"], "claim": row["claim"],
            "sources": [{"label": label, "type": src.get("type"), "excerpt": src.get("excerpt")}
                        for label, src in (row.get("sources") or {}).items()] or
                       [{"label": None, "type": "no citation", "excerpt": ""}]}


def strata_counts(sample: list[dict]) -> dict:
    out: dict[str, int] = {}
    for row in sample:
        key = f"{source_kind(row)}:{'supported' if row['hidden']['verifier_verdict'] else 'flagged'}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def summarize(rows: list[dict]) -> dict:
    """Human vs verifier. Disagreements are counted, never dropped."""
    labelled = [r for r in rows if r.get("human_verdict") in VERDICTS]
    judged = [r for r in labelled if r["human_verdict"] in JUDGED]
    human = {v: sum(1 for r in labelled if r["human_verdict"] == v) for v in VERDICTS}
    v_true = [r for r in judged if r["hidden"]["verifier_verdict"]]
    v_false = [r for r in judged if not r["hidden"]["verifier_verdict"]]
    exact = sum(1 for r in v_true if r["human_verdict"] == "SUPPORTED") + \
        sum(1 for r in v_false if r["human_verdict"] == "UNSUPPORTED")
    grouped = sum(1 for r in v_true if r["human_verdict"] != "UNSUPPORTED") + \
        sum(1 for r in v_false if r["human_verdict"] == "UNSUPPORTED")
    false_supported = [r for r in v_true if r["human_verdict"] == "UNSUPPORTED"]
    partial_as_supported = [r for r in v_true if r["human_verdict"] == "PARTIALLY_SUPPORTED"]
    false_flagged = [r for r in v_false if r["human_verdict"] == "SUPPORTED"]
    return {
        "sampled": len(rows), "labelled": len(labelled), "unlabelled": len(rows) - len(labelled),
        "judged": len(judged), "human": human,
        "human_supported_rate": (human["SUPPORTED"], len(judged)),
        "human_supported_or_partial_rate": (human["SUPPORTED"] + human["PARTIALLY_SUPPORTED"], len(judged)),
        "exact_agreement": (exact, len(judged)), "grouped_agreement": (grouped, len(judged)),
        "verifier_supported": len(v_true), "verifier_flagged": len(v_false),
        "false_supported": len(false_supported), "verifier_false_support_rate": (len(false_supported), len(v_true)),
        "verifier_supported_but_only_partial": len(partial_as_supported),
        "false_flagged": len(false_flagged),
        "disagreement_ids": sorted(r["id"] for r in false_supported + partial_as_supported + false_flagged
                                   + [r for r in v_false if r["human_verdict"] == "PARTIALLY_SUPPORTED"]),
    }


def render(summary: dict, title: str = "HUMAN ADJUDICATION vs LUMEN VERIFIER") -> str:
    def pct(pair):
        return f"{100 * pair[0] / pair[1]:.0f}% ({pair[0]}/{pair[1]})" if pair[1] else "n/a"

    h = summary["human"]
    rows = [
        ("Claims sampled", summary["sampled"]), ("Human-adjudicated", summary["labelled"]),
        ("Not yet labelled", summary["unlabelled"]),
        ("  Supported", h["SUPPORTED"]), ("  Partially supported", h["PARTIALLY_SUPPORTED"]),
        ("  Unsupported", h["UNSUPPORTED"]), ("  Unclear", h["UNCLEAR"]), ("  Skipped", h["SKIP"]),
        ("", ""),
        ("Human-supported claim rate", pct(summary["human_supported_rate"])),
        ("  counting partial as supported", pct(summary["human_supported_or_partial_rate"])),
        ("", ""),
        ("Verifier vs human (judged claims)", summary["judged"]),
        ("Exact agreement", pct(summary["exact_agreement"])),
        ("Agreement, partial grouped with supported", pct(summary["grouped_agreement"])),
        ("Verifier said supported", summary["verifier_supported"]),
        ("Verifier said flagged", summary["verifier_flagged"]),
        ("False-supported (verifier yes, human no)", summary["false_supported"]),
        ("VERIFIER FALSE-SUPPORT RATE", pct(summary["verifier_false_support_rate"])),
        ("Verifier supported, human only partial", summary["verifier_supported_but_only_partial"]),
        ("False-flagged (verifier no, human yes)", summary["false_flagged"]),
        ("Disagreement claim ids", summary["disagreement_ids"] or "none"),
    ]
    return "\n".join([title, ""] + [f"{name:44s} {value}" if name else "" for name, value in rows])
