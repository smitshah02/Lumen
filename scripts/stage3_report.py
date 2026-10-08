"""Stage 3 report: per-set results for control and candidate, and the fixed gate (plan G3).

Reads the raw outputs of scripts/stage3_compare.py and writes
reports/data_foundation/stage3_comparison.json and .md. Counts, ids and timings
only; no question, answer or note text.

The gate, fixed before any held-out question was run:
  regression  on each of the 42-question set, its 11 temporal questions and the
              75-case holdout, the candidate loses at most one correct answer
  targeted    on the targeted held-out set the candidate gains at least three
  safety      "wrong answer not sent to review" does not rise, by count or rate
  latency     the candidate's median rises by less than 25%, on every set
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports" / "data_foundation"
OUT = REPORTS / "stage3"
STRUCTURED_NODES = ("lab_lookup", "encounter_lookup", "structured_lookup")


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _pct(values, q):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    return round(values[min(len(values) - 1, round(q * (len(values) - 1)))], 1) if values else None


def _latency(values) -> dict:
    values = [v for v in values if isinstance(v, (int, float))]
    return {"n": len(values), "median_ms": round(statistics.median(values), 1) if values else None, "p95_ms": _pct(values, 0.95)}


def _paths(trails) -> dict:
    trails = list(trails)
    return {"structured_path": sum(1 for t in trails if any(n in t for n in STRUCTURED_NODES)),
            "retrieval_path": sum(1 for t in trails if "patient_retrieval" in t),
            "structured_lookup_node": sum(1 for t in trails if "structured_lookup" in t)}


# --- 42-question retrieval set and its 11 temporal questions ---------------------------------------

def retrieval_sets() -> dict | None:
    runs = {name: _load(OUT / f"retrieval_research_{name}.json") for name in ("control", "candidate")}
    if not all(runs.values()):
        return None
    out = {"retrieval_42": {}, "temporal_11": {}}
    for name, run in runs.items():
        rows = run["per_question"]
        temporal = [r for r in rows if r["temporal"]]
        mean = lambda key, rs: round(statistics.mean(r[key] for r in rs), 4)            # noqa: E731
        out["retrieval_42"][name] = {
            "database": run["database"], "profile": run["data_profile"], "build": run["build"], "labels": run["labels"],
            "questions": len(rows), "correct": int(sum(r["hit@5"] for r in rows)), "incorrect": int(sum(1 - r["hit@5"] for r in rows)),
            "correct_definition": "Hit@5: a relevant chunk in the top five (the benchmark's own metric)",
            "score": {k: mean(k, rows) for k in ("p@5", "r@5", "ndcg@5", "mrr", "hit@5", "hit@10", "hit@50")},
            "source_note_retrieved": int(sum(r["source_note_hit@5"] for r in rows)),
            "evidence_chunk_retrieved": int(sum(r["hit@5"] for r in rows)),
            "missed": [r["id"] for r in rows if not r["hit@5"]], "cross_patient_results": run["cross_patient_results"],
            "latency": _latency(r["ms"] for r in rows), "retrieval_path": len(rows), "structured_path": 0,
            **({"strict_min_40_tokens_correct": int(sum(r["strict"]["hit@5"] for r in rows))} if name == "candidate" else
               {"label_derivation_check": run["control_label_derivation_check"]})}
        out["temporal_11"][name] = {
            "questions": len(temporal), "correct": int(sum(r["target_hit@5"] for r in temporal)),
            "incorrect": int(sum(1 - r["target_hit@5"] for r in temporal)),
            "correct_definition": "target Hit@5: a chunk from the asked-for date in the top five (the benchmark's temporal metric)",
            "score": {k: mean(k, temporal) for k in ("target_hit@5", "target_rr", "ordered_correctly", "hit@5")},
            "missed": [r["id"] for r in temporal if not r["target_hit@5"]], "latency": _latency(r["ms"] for r in temporal)}
    return out


# --- 75-case holdout -------------------------------------------------------------------------------

def case_ok(r: dict) -> bool:
    """scripts/scorecard.py's own per-case verdict: no hard check failed and no violation."""
    return (not r.get("error") and not r.get("violations")
            and not any(r.get(k) is False for k in ("routing_ok", "temporal_ok", "structured_ok", "evidence_ok")))


LAB_TRUTH_VIOLATION = "structured answer disagrees with SQL truth"


def legacy_case_ok(r: dict) -> bool:
    """The same verdict with the original capped lab table as the truth for the
    creatinine cases. For comparison with the original frozen benchmark only."""
    if "legacy_capped_lab_ok" not in r:
        return case_ok(r)
    others = [v for v in (r.get("violations") or []) if v != LAB_TRUTH_VIOLATION]
    return not r.get("error") and not others and r.get("routing_ok") is not False and r["legacy_capped_lab_ok"]


def holdout_set() -> dict | None:
    out = {}
    for name in ("control", "candidate"):
        files = sorted((OUT / f"holdout75_holdout_{name}").glob("scorecard-*.json"))
        if not files:
            return None
        run = json.loads(files[-1].read_text())
        cases = run["cases"]
        ok = [r for r in cases if case_ok(r)]
        bad = [r for r in cases if not case_ok(r)]
        reviewed = lambda r: r.get("status") == "human_review_required"                # noqa: E731
        agg = run["aggregate"]
        out[name] = {
            "database": run["database"], "file": files[-1].name, "runs_found": len(files),
            "cases": len(cases), "correct": len(ok), "incorrect": len(bad), "errors": agg["errors"],
            "lab_truth": run.get("lab_truth"),
            "lab_cases": {"cases": sum(1 for r in cases if "legacy_capped_lab_ok" in r),
                          "correct_full_table_truth": sum(1 for r in cases if "legacy_capped_lab_ok" in r and case_ok(r)),
                          "correct_legacy_capped_truth": sum(1 for r in cases if "legacy_capped_lab_ok" in r and legacy_case_ok(r))},
            "legacy_capped_lab_score": sum(1 for r in cases if legacy_case_ok(r)),
            "correct_definition": "the scorecard's per-case verdict: no hard check failed and no violation",
            "failed_cases": [f"{r['case']}:{r['subject_id']}" for r in bad],
            "failed_checks": sorted({v if not v[:1].isdigit() else "unsupported claim(s) auto-approved"
                                     for r in bad for v in (r.get("violations") or [])}
                                    | {k for r in bad for k in ("routing_ok", "temporal_ok", "structured_ok", "evidence_ok") if r.get(k) is False}),
            "safety_contract": agg["safety_contract"], "routing_correct": agg["routing_correct"],
            "structured_truth": agg["structured_truth"], "temporal_correct": agg["temporal_correct"],
            "review_required": sum(1 for r in cases if reviewed(r)),
            "wrong_not_sent_to_review": sum(1 for r in bad if not reviewed(r)),
            "wrong_sent_to_review": sum(1 for r in bad if reviewed(r)),
            "unsafe_autoapproved_claims": agg["unsafe_autoapproved_claims"],
            "citation_validity": {"citation_resolution": agg["citation_resolution"], "source_exists": agg["source_exists"],
                                  "patient_isolation": agg["patient_isolation"],
                                  "integrity_ok_cases": sum(1 for r in cases if r.get("integrity_ok"))},
            "outcomes": agg["outcomes"],
            "latency": {"n": len(cases), "median_ms": agg["latency"]["total_p50_ms"], "p95_ms": agg["latency"]["total_p95_ms"],
                        "model_backed_median_ms": agg["latency"]["model_backed_p50_ms"],
                        "definition": "the scorecard's warm total per case"},
            "llm_calls": agg["latency"]["llm_calls"], **_paths(r.get("node_trail") or [] for r in cases),
            "admission_scope": _scope(OUT / f"holdout75_holdout_{name}" / "scope.json"),
        }
    return out


def _scope(path: Path) -> dict:
    """Admission-scope status per scored case, as stage3_compare read it from the checkpoints."""
    rows = (_load(path) or {}).values()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"status": counts, "applied": sum(1 for r in rows if r["applied"])}


# --- targeted sets, graded by hand ------------------------------------------------------------------

def _questions(name: str) -> dict | None:
    if name == "dev15":
        return {q["qid"]: q for q in json.loads((REPORTS / "before_benchmark.json").read_text())["items"]}
    import stage3_compare as s3
    path = s3.HELDOUT_QUESTIONS
    if path is None or not path.exists() or not (OUT / "ask_heldout_research_control.json").exists():
        return None                                  # never opened before the held-out run exists
    data = json.loads(path.read_text())
    return {q["qid"]: q for q in (data["items"] if isinstance(data, dict) else data)}


def _evidence_retrieved(question: dict, sources: list, conn) -> bool | None:
    """Is the chunk that holds the ground-truth text among the returned sources?
    Control: the chunk id recorded with the question. v2: a source of the same
    note whose stored offsets contain the place the ground-truth pattern matches."""
    from sqlalchemy import text
    prov = question.get("provenance") or {}
    notes = [s for s in sources if str(s.get("label", "")).startswith("S")]
    if not any(s.get("provenance") for s in notes):
        frozen = prov.get("evidence_chunk_id_in_current_index")
        return None if frozen is None else frozen in {s.get("chunk_id") for s in notes}
    if not prov.get("regex") or not prov.get("note_id"):
        return None
    body = conn.execute(text("SELECT COALESCE(text_original, text_deid) FROM clinical_notes WHERE note_id = :n"),
                        {"n": prov["note_id"]}).scalar() or ""
    spans = [m.span() for m in re.finditer(prov["regex"], body)]
    return any(s.get("note_id") == prov["note_id"] and s["provenance"]["start_offset"] <= a and b <= s["provenance"]["end_offset"]
               for s in notes if s.get("provenance") for a, b in spans)


def targeted_set(name: str) -> dict | None:
    questions = _questions(name)
    runs = {s: _load(OUT / f"ask_{name}_research_{s}.json") for s in ("control", "candidate")}
    sheet, key = _load(OUT / f"grades_{name}.json"), _load(OUT / f"grades_{name}.key.json")
    if not questions or not all(runs.values()) or not sheet or not key:
        return None
    if any(e["grade"] not in ("correct", "partial", "incorrect") for e in sheet["entries"]):
        return None
    grades = {(e["qid"], key[e["qid"]][e["arm"]].replace("research_", "")): e["grade"] for e in sheet["entries"]}
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url
    from src import storage
    engine = create_engine(make_url(storage.DATABASE_URL).set(database="lumen"),
                           connect_args={"options": "-c default_transaction_read_only=on"})
    out = {}
    with engine.connect() as conn:
        for system, run in runs.items():
            rows, per_question = run["rows"], []
            for row in rows:
                q, r = questions[row["qid"]], row["response"]
                sources, scope = r.get("sources") or [], r.get("admission_scope") or {}
                grade = grades[(row["qid"], system)]
                reviewed = r.get("status") == "human_review_required"
                gold_note = q.get("note_id") or (q.get("provenance") or {}).get("note_id")
                per_question.append({
                    "qid": row["qid"], "subject_id": row["subject_id"], "grade": grade, "status": r.get("status", r.get("error")),
                    "review_required": reviewed, "node_trail": r.get("node_trail") or [],
                    "latency_ms": r.get("latency_ms"), "llm_calls": (r.get("timings") or {}).get("llm_calls"),
                    "source_note_retrieved": None if gold_note is None else gold_note in {s.get("note_id") for s in sources},
                    "evidence_chunk_retrieved": _evidence_retrieved(q, sources, conn),
                    "citations": len(r.get("citations") or []),
                    "citations_verified": sum(1 for c in r.get("citations") or [] if c.get("verified")),
                    "scope_status": scope.get("status"), "scope_applied": bool(scope.get("applied")), "scope_hadm_id": scope.get("hadm_id"),
                    "scope_wrong": bool(scope.get("applied")) and q.get("hadm_id") is not None and scope.get("hadm_id") != q.get("hadm_id"),
                })
            count = lambda f: sum(1 for p in per_question if f(p))                    # noqa: E731
            statuses: dict[str, int] = {}
            for p in per_question:
                statuses[str(p["scope_status"])] = statuses.get(str(p["scope_status"]), 0) + 1
            out[system] = {
                "database": run["database"], "profile": run["profile"], "build": run["build"], "questions": len(per_question),
                "correct": count(lambda p: p["grade"] == "correct"), "partial": count(lambda p: p["grade"] == "partial"),
                "incorrect": count(lambda p: p["grade"] == "incorrect"),
                "correct_definition": "manual grade against the written ground truth, blind to the system",
                "source_note_retrieved": count(lambda p: p["source_note_retrieved"]),
                "evidence_chunk_retrieved": count(lambda p: p["evidence_chunk_retrieved"]),
                "review_required": count(lambda p: p["review_required"]),
                "incorrect_sent_to_review": count(lambda p: p["grade"] == "incorrect" and p["review_required"]),
                "incorrect_not_sent_to_review": count(lambda p: p["grade"] == "incorrect" and not p["review_required"]),
                "wrong_not_sent_to_review": count(lambda p: p["grade"] != "correct" and not p["review_required"]),
                "wrong_not_sent_to_review_definition": "incorrect or partial, and released without review (the baseline's definition)",
                "wrong_not_sent_to_review_ids": [p["qid"] for p in per_question if p["grade"] != "correct" and not p["review_required"]],
                "admission_scope": {"status": statuses, "applied": count(lambda p: p["scope_applied"]),
                                    "wrong_application": count(lambda p: p["scope_wrong"])},
                "citation_validity": {"claims": sum(p["citations"] for p in per_question),
                                      "verified": sum(p["citations_verified"] for p in per_question)},
                "latency": _latency(p["latency_ms"] for p in per_question),
                "llm_calls": sum(p["llm_calls"] or 0 for p in per_question), **_paths(p["node_trail"] for p in per_question),
                "per_question": [{k: p[k] for k in ("qid", "subject_id", "grade", "status", "review_required", "source_note_retrieved",
                                                    "evidence_chunk_retrieved", "scope_status", "scope_applied", "scope_wrong",
                                                    "latency_ms", "llm_calls")} | {"path": [n for n in p["node_trail"] if n in STRUCTURED_NODES + ("patient_retrieval",)]}
                                 for p in per_question],
            }
    engine.dispose()
    return out


# --- the gate ---------------------------------------------------------------------------------------

def gate(sets: dict) -> dict:
    conditions = []

    def add(name, group, passed, detail):
        conditions.append({"condition": name, "group": group, "result": "NOT RUN" if passed is None else "PASS" if passed else "FAIL",
                           "detail": detail})

    for key, label in (("retrieval_42", "42-question set"), ("temporal_11", "11 temporal questions"), ("holdout_75", "75-case holdout")):
        s = sets.get(key)
        if not s:
            add(f"{label}: candidate loses at most 1 correct", "regression", None, "set not run")
            continue
        lost = s["control"]["correct"] - s["candidate"]["correct"]
        add(f"{label}: candidate loses at most 1 correct", "regression", lost <= 1,
            f"control {s['control']['correct']}, candidate {s['candidate']['correct']}, lost {max(lost, 0)}")
    s = sets.get("targeted_heldout")
    if not s:
        add("targeted held-out: candidate gains at least 3 correct", "targeted", None, "set not run")
    else:
        gain = s["candidate"]["correct"] - s["control"]["correct"]
        add("targeted held-out: candidate gains at least 3 correct", "targeted", gain >= 3,
            f"control {s['control']['correct']}, candidate {s['candidate']['correct']}, gain {gain}")
    for key, label, size in (("holdout_75", "75-case holdout", "cases"), ("targeted_heldout", "targeted held-out", "questions")):
        s = sets.get(key)
        if not s:
            add(f"{label}: wrong answers not sent to review do not rise", "safety", None, "set not run")
            continue
        a, b = s["control"]["wrong_not_sent_to_review"], s["candidate"]["wrong_not_sent_to_review"]
        ra, rb = a / s["control"][size], b / s["candidate"][size]
        add(f"{label}: wrong answers not sent to review do not rise", "safety", b <= a and rb <= ra,
            f"control {a} of {s['control'][size]}, candidate {b} of {s['candidate'][size]}")
        if key == "holdout_75":
            a, b = s["control"]["unsafe_autoapproved_claims"], s["candidate"]["unsafe_autoapproved_claims"]
            add(f"{label}: unsupported claims auto-approved do not rise", "safety", b <= a, f"control {a}, candidate {b}")
    for key, label in (("retrieval_42", "42-question set (search)"), ("holdout_75", "75-case holdout (answer)"),
                       ("targeted_heldout", "targeted held-out (answer)")):
        s = sets.get(key)
        if not s:
            add(f"{label}: candidate median latency rises by less than 25%", "latency", None, "set not run")
            continue
        a, b = s["control"]["latency"]["median_ms"], s["candidate"]["latency"]["median_ms"]
        change = (b - a) / a * 100
        add(f"{label}: candidate median latency rises by less than 25%", "latency", change < 25,
            f"control {a:.0f} ms, candidate {b:.0f} ms, change {change:+.1f}%")
    results = {c["result"] for c in conditions}
    final = "INCOMPLETE" if "NOT RUN" in results else "FAIL" if "FAIL" in results else "PASS"
    return {"conditions": conditions, "final": final}


# --- output -----------------------------------------------------------------------------------------

def _row(label, control, candidate, fmt=str):
    delta = ""
    if isinstance(control, (int, float)) and isinstance(candidate, (int, float)):
        delta = f"{candidate - control:+.1f}" if isinstance(control, float) or isinstance(candidate, float) else f"{candidate - control:+d}"
    return f"| {label} | {fmt(control)} | {fmt(candidate)} | {delta} |"


def render(report: dict) -> str:
    lines = [f"# Stage 3 comparison ({report.get('run_id', 'stage3')}): control against the v2 candidate", "",
             f"Created {report['created']}. Local only; counts, ids and timings, no note text.", "",
             f"## Result: {report['gate']['final']}", "",
             "| Group | Condition | Result | Detail |", "|---|---|---|---|"]
    lines += [f"| {c['group']} | {c['condition']} | {c['result']} | {c['detail']} |" for c in report["gate"]["conditions"]]
    lines += ["", "## Systems", "", "| System | Database | Profile | Build |", "|---|---|---|---|"]
    lines += [f"| {n} | {s['database']} | {s['profile']} | {s['build'] or '-'} |" for n, s in report["systems"].items()]
    freeze = report.get("freeze") or {}
    lines += ["", f"Frozen configuration: `{freeze.get('path', 'not recorded')}` (identity {str(freeze.get('sha256'))[:12]}).",
              f"Holdout source tables identical to the frozen fingerprint: {report['holdout_state']['source_tables_identical']}; "
              f"tables changed: {report['holdout_state']['tables_changed'] or 'none'}."]
    titles = {"retrieval_42": "42-question retrieval set", "temporal_11": "11 temporal questions (subset of the 42)",
              "holdout_75": "75-case holdout", "targeted_dev15": "15 targeted development questions (descriptive only)",
              "targeted_heldout": "Targeted held-out questions"}
    for key, title in titles.items():
        s = report["sets"].get(key)
        lines += ["", f"## {title}", ""]
        if not s:
            lines.append("Not run.")
            continue
        c, v = s["control"], s["candidate"]
        lines += [f"Correct means: {c['correct_definition']}.", "", "| Metric | Control | Candidate | Change |", "|---|---|---|---|"]
        for label, field in (("Questions", "questions"), ("Cases", "cases"), ("Correct", "correct"), ("Partial", "partial"),
                             ("Incorrect", "incorrect"), ("Source note retrieved", "source_note_retrieved"),
                             ("Evidence chunk retrieved", "evidence_chunk_retrieved"), ("Review required", "review_required"),
                             ("Incorrect, sent to review", "incorrect_sent_to_review"), ("Wrong, sent to review", "wrong_sent_to_review"),
                             ("Incorrect, not sent to review", "incorrect_not_sent_to_review"),
                             ("Wrong, not sent to review", "wrong_not_sent_to_review"),
                             ("Unsupported claims auto-approved", "unsafe_autoapproved_claims"), ("LLM calls", "llm_calls"),
                             ("Structured path used", "structured_path"), ("Retrieval path used", "retrieval_path")):
            if field in c:
                lines.append(_row(label, c[field], v[field]))
        for metric, value in (c.get("score") or {}).items():
            lines.append(_row(metric, value, v["score"][metric]))
        lines.append(_row("Median latency (ms)", c["latency"]["median_ms"], v["latency"]["median_ms"]))
        lines.append(_row("p95 latency (ms)", c["latency"]["p95_ms"], v["latency"]["p95_ms"]))
        if "admission_scope" in c:
            lines.append(_row("Admission scope status", json.dumps(c["admission_scope"]["status"]), json.dumps(v["admission_scope"]["status"])))
            lines.append(_row("Admission scope applied", c["admission_scope"]["applied"], v["admission_scope"]["applied"]))
            if "wrong_application" in c["admission_scope"]:
                lines.append(_row("Admission scope applied to the wrong admission", c["admission_scope"]["wrong_application"],
                                  v["admission_scope"]["wrong_application"]))
        if "citation_validity" in c:
            lines.append(_row("Citation and evidence validity", json.dumps(c["citation_validity"]), json.dumps(v["citation_validity"])))
        if "legacy_capped_lab_score" in c:
            lines.append(_row("Legacy capped-lab score (history only, not in the gate)", c["legacy_capped_lab_score"], v["legacy_capped_lab_score"]))
            lines.append(_row("Lab cases correct, full-table truth", c["lab_cases"]["correct_full_table_truth"], v["lab_cases"]["correct_full_table_truth"]))
            lines.append(_row("Lab cases correct, legacy capped truth", c["lab_cases"]["correct_legacy_capped_truth"], v["lab_cases"]["correct_legacy_capped_truth"]))
            lines += ["", f"Lab truth: {json.dumps(c['lab_truth'])}. The complete lab table is the truth for both systems. This evaluator "
                          "correction was made before either system ran the 75 cases, because the pre-evaluation audit showed the "
                          "capped table was incomplete for all ten patients."]
        if "strict_min_40_tokens_correct" in v:
            lines += ["", f"Candidate labels: {v['labels']}. Under the strict rule the candidate has "
                          f"{v['strict_min_40_tokens_correct']} correct.",
                      f"Control label derivation check: {c['label_derivation_check']}."]
        for name, side in (("control", c), ("candidate", v)):
            for field in ("missed", "failed_cases", "wrong_not_sent_to_review_ids"):
                if side.get(field):
                    lines += ["", f"{name} {field.replace('_', ' ')}: {', '.join(map(str, side[field]))}"]
            if side.get("failed_checks"):
                lines += ["", f"{name} failed checks: {'; '.join(side['failed_checks'])}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    global OUT
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import stage3_compare as s3
    OUT = s3.OUT                                     # this run's outputs
    sets = {**(retrieval_sets() or {}), "holdout_75": holdout_set(), "targeted_dev15": targeted_set("dev15"),
            "targeted_heldout": targeted_set("heldout")}
    freeze_path = s3.FREEZE
    import hashlib
    report = {
        "run_id": s3.RUN_ID, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "systems": s3.systems(), "holdout_state": s3.holdout_state(),
        "freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest()}
                  if freeze_path.exists() else None,
        "sets": {k: v for k, v in sets.items() if v}, "sets_not_run": [k for k, v in sets.items() if not v],
    }
    report["gate"] = gate(report["sets"])
    for c in report["gate"]["conditions"]:
        print(f"{c['result']:8s} {c['group']:10s} {c['condition']}  ({c['detail']})")
    print(f"\nFINAL: {report['gate']['final']}   not run: {report['sets_not_run'] or 'none'}")
    if report["gate"]["final"] == "INCOMPLETE":      # a report is written once, when every set has been run
        print(f"not written: run {s3.RUN_ID!r} is incomplete")
        return 0
    s3.refuse_existing(s3.COMPARISON_JSON, s3.COMPARISON_MD)
    s3.COMPARISON_JSON.write_text(json.dumps(report, indent=1, default=str))
    s3.COMPARISON_MD.write_text(render(report))
    print(f"written: {s3.COMPARISON_JSON.name}, {s3.COMPARISON_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
