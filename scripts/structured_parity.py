"""Local parity check: structured API answers vs the rows they were read from.

For a sample of patients, asks the running API five structured questions and
compares each answer with a direct SQL query on `labevents` / `admissions`.

    ./scripts/lumen research parity                 # 5 random patients
    ./scripts/lumen research parity --patients 10 --seed 0.3
    ./scripts/lumen research parity --subjects-file ~/local/ids.json

Local only: refuses a non-loopback API and a non-research plane, and bypasses
any configured proxy. Prints PASS / FAIL / FELL_THROUGH per question class and
aggregate counts — never a subject id, a value, a date or an answer. Full
responses are written only when --out names a local file.

FELL_THROUGH means the API declined the deterministic path (for example a
non-numeric result at the end of the series) and answered from notes instead;
it is reported, not counted as a pass.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src import storage  # noqa: E402

ANALYTES = ("creatinine", "potassium", "hemoglobin")   # blood specimens, matched by exact label
TREND_QUESTIONS = {"trend": "How did {a} change over the patient's available record?",
                   "trend_short": "How did {a} change?", "trend_has": "How has {a} changed?",
                   "trend_updown": "Did {a} increase or decrease?"}
# Questions the structured lookups must decline (rows exist, a clause is not understood).
GUARD_QUESTIONS = (("encounter_lookup", "How many hospital admissions does the patient have, and what was "
                                        "the patient's favourite colour during each one?"),
                   ("lab_lookup", "How did creatinine change over time and what was the patient's "
                                  "favourite colour?"))
REFUSAL = "the available records do not contain enough information"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never through a proxy


def ask(base: str, subject_id: int, query: str) -> dict:
    req = urllib.request.Request(f"{base}/ask", method="POST",
                                 data=json.dumps({"subject_id": subject_id, "query": query}).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with _OPENER.open(req, timeout=900) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": type(e).__name__}


def sample(conn, n: int, seed: float | None) -> list[int]:
    if seed is not None:
        conn.execute(text("SELECT setseed(:s)"), {"s": seed})
    return list(conn.execute(text("""
        SELECT a.subject_id FROM admissions a
        WHERE EXISTS (SELECT 1 FROM note_chunks nc WHERE nc.subject_id = a.subject_id)
        GROUP BY a.subject_id HAVING COUNT(*) >= 2 ORDER BY random() LIMIT :n
    """), {"n": n}).scalars())


LAB_TRUTH_TABLES = ("labevents", "labevents_full")     # the ingested table (capped per patient) and the complete one


def truth(conn, sid: int, analytes, lab_table: str = "labevents") -> dict:
    if lab_table not in LAB_TRUTH_TABLES:
        raise ValueError(f"unknown lab truth table {lab_table!r}")
    labs = {}
    for a in analytes:
        rows = conn.execute(text(f"""
            SELECT l.charttime, l.valuenum FROM {lab_table} l JOIN d_labitems d ON d.itemid = l.itemid
            WHERE l.subject_id = :s AND lower(d.label) = :a AND lower(d.fluid) = 'blood'
              AND l.valuenum IS NOT NULL ORDER BY l.charttime
        """), {"s": sid, "a": a}).fetchall()
        labs[a] = [(str(t)[:10], float(v)) for t, v in rows]
    adm = conn.execute(text(
        "SELECT admittime FROM admissions WHERE subject_id = :s ORDER BY admittime"), {"s": sid}).fetchall()
    return {"labs": labs, "admits": [str(t)[:10] for (t,) in adm]}


def has_number(answer: str, value: float) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(f'{value:g}')}(?![\d]|\.\d)", answer) is not None


def checks(t: dict) -> list[tuple[str, str, str, callable]]:
    """(class, question, deterministic node, predicate over the answer text)."""
    admits = t["admits"]
    out = []
    for name, labs in t["labs"].items():
        if not labs:
            continue
        (d0, v0), (dn, vn) = labs[0], labs[-1]
        lo, hi = min(v for _, v in labs), max(v for _, v in labs)

        def trend_ok(a, labs=labs, d0=d0, dn=dn, nums=(v0, vn, lo, hi)):
            return ((len(labs) == 1 or f"measured {len(labs)} times" in a) and d0 in a and dn in a
                    and all(has_number(a, v) for v in nums))

        out += [(f"{name}.latest", f"What was the most recent {name}?", "lab_lookup",
                 lambda a, dn=dn, vn=vn: dn in a and has_number(a, vn)),
                (f"{name}.earliest", f"What was the earliest {name}?", "lab_lookup",
                 lambda a, d0=d0, v0=v0: d0 in a and has_number(a, v0))]
        out += [(f"{name}.{cls}", q.format(a=name), "lab_lookup", trend_ok) for cls, q in TREND_QUESTIONS.items()]
    if admits:
        out += [
            ("admission_count", "How many hospital admissions does the patient have?", "encounter_lookup",
             lambda a: re.search(rf"\b{len(admits)} recorded hospital admission", a) is not None),
            ("admission_latest", "When was the most recent admission?", "encounter_lookup",
             lambda a: f"began on {admits[-1]}" in a),
        ]
    return out


def refusal_guard(api: str, sids: list[int]) -> int:
    """Live probe: a refusal after a declined structured lookup must not be auto-approved."""
    tally = {"GUARDED": 0, "ANSWERED": 0, "NOT_DECLINED": 0, "FAIL": 0, "ERROR": 0}
    for alias, sid in enumerate(sids, 1):
        row = []
        for node, question in GUARD_QUESTIONS:
            r = ask(api, int(sid), question)
            trail = r.get("node_trail") or []
            declined = node in trail and "patient_retrieval" in trail
            refused = REFUSAL in (r.get("answer") or "").lower()
            held = r.get("status") == "human_review_required" or bool(r.get("needs_human_review"))
            verdict = ("ERROR" if "answer" not in r else "NOT_DECLINED" if not declined
                       else "ANSWERED" if not refused else "GUARDED" if held else "FAIL")
            tally[verdict] += 1
            row.append(f"{node}={verdict}(status={r.get('status')},review={r.get('review_status')},"
                       f"llm={(r.get('timings') or {}).get('llm_calls')})")
        print(f"P{alias}: " + "  ".join(row))
    print("\n" + "  ".join(f"{k}={v}" for k, v in tally.items()))
    print("GUARDED = refusal sent to human review (the guard fired). ANSWERED = the model answered "
          "from notes, guard not exercised. FAIL = refusal auto-approved.")
    return 1 if tally["FAIL"] or tally["ERROR"] or tally["NOT_DECLINED"] else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--patients", type=int, default=5)
    p.add_argument("--seed", type=float, help="setseed() value in [-1, 1] for a repeatable sample")
    p.add_argument("--subjects-file", help="local JSON list of subject ids to use instead of sampling")
    p.add_argument("--analytes", default=",".join(ANALYTES), help="comma-separated blood analyte labels")
    p.add_argument("--refusal-guard", action="store_true",
                   help="instead of parity, probe that a refusal after a declined lookup goes to review")
    p.add_argument("--out", help="local JSONL file for full responses (never printed)")
    args = p.parse_args(argv)

    if storage.DATA_PLANE != "research":
        print(f"refusing: parity runs on the research plane, not {storage.DATA_PLANE!r}", file=sys.stderr)
        return 2
    if urlparse(args.api).hostname not in ("127.0.0.1", "localhost", "::1"):
        print("refusing: the API must be on loopback", file=sys.stderr)
        return 2

    with storage.engine.connect() as conn:
        sids = (json.loads(Path(args.subjects_file).expanduser().read_text()) if args.subjects_file
                else sample(conn, args.patients, args.seed))
        if args.refusal_guard:
            return refusal_guard(args.api, sids)
        truths = [truth(conn, int(s), [a.strip().lower() for a in args.analytes.split(",") if a.strip()])
                  for s in sids]

    sink = Path(args.out).expanduser().open("w") if args.out else None
    tally: dict[str, dict[str, int]] = {}
    llm_calls, slowest = 0, 0.0
    for alias, (sid, t) in enumerate(zip(sids, truths), 1):
        row, seen = [], {}
        for cls, question, node, ok in checks(t):
            r = ask(args.api, int(sid), question)
            if sink:
                sink.write(json.dumps({"alias": f"P{alias}", "class": cls, "response": r}) + "\n")
            trail = r.get("node_trail") or []
            deterministic = node in trail and "patient_retrieval" not in trail
            verdict = ("ERROR" if "answer" not in r else "FELL_THROUGH" if not deterministic
                       else "PASS" if ok(r["answer"]) else "FAIL")
            if verdict in ("PASS", "FAIL"):
                llm_calls += (r.get("timings") or {}).get("llm_calls", 0)
                slowest = max(slowest, float(r.get("latency_ms") or 0))
            tally.setdefault(cls, {}).setdefault(verdict, 0)
            tally[cls][verdict] += 1
            seen[verdict] = seen.get(verdict, 0) + 1
            if verdict != "PASS":
                row.append(f"{cls}={verdict}")
        print(f"P{alias}: " + "  ".join(f"{k}={v}" for k, v in sorted(seen.items()))
              + ("   not passing: " + " ".join(row) if row else ""))
    if sink:
        sink.close()

    print()
    for cls, counts in tally.items():
        print(f"{cls:24s} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    total = {k: sum(c.get(k, 0) for c in tally.values()) for k in ("PASS", "FAIL", "FELL_THROUGH", "ERROR")}
    print(f"\npatients={len(sids)}  " + "  ".join(f"{k}={v}" for k, v in total.items())
          + f"  llm_calls_on_deterministic_answers={llm_calls}  slowest_deterministic_ms={slowest:.0f}")
    return 1 if total["FAIL"] or total["ERROR"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
