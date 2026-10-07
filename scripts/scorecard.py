"""Run the research scorecard against the local API.

    ./scripts/lumen research scorecard
    ./scripts/lumen research scorecard --only structured,hitl
    ./scripts/lumen research scorecard --stability-runs 3     # model-sensitive cases only, repeated
    ./scripts/lumen research scorecard --with-literature      # API started with LUMEN_LITERATURE_BACKEND=pubmed
    ./scripts/lumen research scorecard --compare A.json B.json [--adjudication a.jsonl b.jsonl]

Each case is asked through /ask, then checked against the database and the
run's checkpoint (see src/evals/scorecard.py). Prints one status line per case
and an aggregate table — never an answer, a claim or note text. Raw per-case
results (ids, trails, counts, timings) are written under --out-dir on this
machine, next to a claims file for human adjudication; the claims file
contains patient text and is not printed.

Local only: refuses a non-loopback API and a non-research plane, refuses when
the API and this process are on different databases, and refuses to run by
default while the API's PubMed backend is enabled.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy import text  # noqa: E402

import structured_parity as parity  # noqa: E402
from src import storage  # noqa: E402
from src.agents.verify import SOURCE_CHARS  # noqa: E402
from src.evals import adjudication, scorecard  # noqa: E402


def ready(api: str) -> dict:
    try:
        with parity._OPENER.open(f"{api}/ready", timeout=60) as r:
            return json.loads(r.read())
    except Exception as e:                       # /ready answers 503 with a body when a dependency is down
        try:
            return json.loads(getattr(e, "read", lambda: b"{}")() or b"{}")
        except ValueError:
            return {}


def load_subjects(path: str) -> list[int]:
    data = json.loads(Path(path).expanduser().read_text())
    return [int(s) for s in (data["subject_ids"] if isinstance(data, dict) else data)]


def build_cases(sids, truths, backend: str, only: set[str], profile: str) -> list[tuple[int, dict]]:
    cases = []
    for sid, truth in zip(sids, truths):
        for cls, query, node, ok in parity.checks(truth):
            if cls in scorecard.STRUCTURED_CLASSES:
                cases.append((sid, scorecard.structured_case(cls, query, node, ok)))
    cases += scorecard.rag_plan(sids, backend, profile)
    if only:
        cases = [(sid, c) for sid, c in cases if c["id"] in only or c["category"] in only]
    return cases


def database_rows(conn, sources: list[dict]) -> dict:
    """The chunks the response cites, as the database has them. Ids only."""
    rows = {}
    notes = [s["chunk_id"] for s in sources if str(s.get("label", "")).startswith("S")]
    guides = [s["chunk_id"] for s in sources if str(s.get("label", "")).startswith("G")]
    if notes:
        for cid, nid, sid in conn.execute(text(
                "SELECT chunk_id, note_id, subject_id FROM note_chunks WHERE chunk_id = ANY(:ids)"), {"ids": notes}):
            rows[("note", cid)] = {"note_id": nid, "subject_id": sid}
    if guides:
        for (cid,) in conn.execute(text(
                "SELECT chunk_id FROM guideline_chunks WHERE chunk_id = ANY(:ids)"), {"ids": guides}):
            rows[("guideline", cid)] = {}
    return rows


def claim_rows(result: dict, state: dict) -> list[dict]:
    """Claim -> cited source excerpts, for a human to judge. Contains patient text."""
    evidence = {e.get("label"): e for e in scorecard.state_evidence(state)}
    out = []
    for claim in state.get("citations") or []:
        if scorecard.is_system_sentence(claim.get("claim")):
            continue
        labels = claim.get("labels") or ([claim["label"]] if claim.get("label") else [])
        out.append({"case": result["case"], "subject_id": result["subject_id"], "thread_id": result["thread_id"],
                    "claim": claim.get("claim"), "labels": labels, "verifier_verdict": bool(claim.get("verified")),
                    "verifier_note": claim.get("verification_note"),
                    # the same window the verifier is given, so a reviewer never judges on less
                    "sources": {l: (evidence.get(l, {}).get("text") or "")[:SOURCE_CHARS] for l in labels}})
    return out


def cmd_compare(args) -> int:
    existing, holdout = (json.loads(Path(p).expanduser().read_text())["aggregate"] for p in args.compare)
    summaries = [None, None]
    for i, path in enumerate(args.adjudication or []):
        rows = [json.loads(line) for line in Path(path).expanduser().read_text().splitlines() if line.strip()]
        summary = adjudication.summarize(rows)
        summaries[i] = summary if summary["judged"] else None
    print(scorecard.compare(existing, holdout, *summaries))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--subjects-file", default="~/Lumen_local_results/p0/manual_subjects.json",
                   help="local JSON list of subject ids, or a holdout manifest")
    p.add_argument("--out-dir", default="~/Lumen_local_results/scorecard")
    p.add_argument("--profile", choices=["fixture", "holdout"], default="fixture")
    p.add_argument("--only", default="", help="comma-separated case ids or categories (for a quick run)")
    p.add_argument("--ask", metavar="CASE_ID", help="ask exactly one case; use with --patient")
    p.add_argument("--patient", type=int, metavar="N", help="with --ask: position in the subjects file, from 1")
    p.add_argument("--stability-runs", type=int, default=0, metavar="N",
                   help="repeat only the model-sensitive cases N times and report variability (3 is typical)")
    p.add_argument("--with-literature", action="store_true",
                   help="allow a run while the API's PubMed backend is enabled (makes outbound searches)")
    p.add_argument("--compare", nargs=2, metavar=("EXISTING", "HOLDOUT"), help="print a two-cohort table and exit")
    p.add_argument("--adjudication", nargs="+", help="with --compare: labelled sample files, existing then holdout")
    args = p.parse_args(argv)
    if args.compare:
        return cmd_compare(args)

    if storage.DATA_PLANE != "research":
        print(f"refusing: the scorecard runs on the research plane, not {storage.DATA_PLANE!r}", file=sys.stderr)
        return 2
    if urlparse(args.api).hostname not in ("127.0.0.1", "localhost", "::1"):
        print("refusing: the API must be on loopback", file=sys.stderr)
        return 2
    info = ready(args.api)
    backend, database = info.get("literature_backend"), info.get("database")
    if backend not in ("none", "pubmed"):
        print("refusing: could not read /ready (is the API running?)", file=sys.stderr)
        return 2
    if database != storage.engine.url.database:
        print(f"refusing: the API is on database {database!r} but this command is on "
              f"{storage.engine.url.database!r}. Start both the same way.", file=sys.stderr)
        return 2
    if backend == "pubmed" and not args.with_literature:
        print("refusing: the API has LUMEN_LITERATURE_BACKEND=pubmed, so literature questions would search PubMed.\n"
              "Restart the API without it, or pass --with-literature to allow those outbound searches.", file=sys.stderr)
        return 2

    sids = load_subjects(args.subjects_file)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    import logging
    logging.disable(logging.WARNING)
    from src.agents.graph import build_graph, close_pools      # checkpoints only; loads no model
    graph, _ = build_graph()

    results, claims = [], []
    with storage.engine.connect() as conn:
        present = set(conn.execute(text("SELECT DISTINCT subject_id FROM note_chunks WHERE subject_id = ANY(:ids)"),
                                   {"ids": sids}).scalars())
        missing = [f"P{i}" for i, s in enumerate(sids, 1) if s not in present]
        if missing:                              # never evaluate part of a cohort and present it as the whole
            print(f"refusing: {len(missing)} of {len(sids)} subjects in the subjects file are not loaded in "
                  f"{database}: {', '.join(missing)}. Load the whole cohort first.", file=sys.stderr)
            close_pools()
            return 2
        truths = [parity.truth(conn, sid, ["creatinine"]) for sid in sids]
        if args.ask:
            cases = [(sids[(args.patient or 1) - 1], scorecard.BY_ID[args.ask])]
        else:
            cases = build_cases(sids, truths, backend, only, args.profile)
        repeats = 1
        if args.stability_runs:
            cases = [(sid, c) for sid, c in cases if c["category"] in scorecard.MODEL_SENSITIVE]
            repeats = args.stability_runs
        print(f"{len(cases)} case(s) x {repeats} run(s), {len(sids)} patient(s), database: {database}, "
              f"literature backend: {backend}\n")
        for sid, case in cases:
            alias = f"P{sids.index(sid) + 1}"
            for run in range(1, repeats + 1):
                response = parity.ask(args.api, sid, case["query"])
                tag = f"{alias} {case['id']:30s}" + (f" run {run}" if repeats > 1 else "")
                if "answer" not in response:
                    results.append({"case": case["id"], "category": case["category"], "subject_id": sid,
                                    "error": response.get("error", "no answer"), "contract_ok": False,
                                    "violations": ["request failed"]})
                    print(f"{tag} ERROR {response.get('error')}")
                    continue
                state = graph.get_state({"configurable": {"thread_id": response["thread_id"]}}).values
                result = scorecard.evaluate(case, sid, response, state,
                                            database_rows(conn, response.get("sources") or []), backend)
                result["run"] = run
                results.append(result)
                claims += claim_rows(result, state)
                hard = [k[:-3] for k in ("routing_ok", "temporal_ok", "structured_ok", "evidence_ok") if result[k] is False]
                soft = "" if result["expected_ok"] is not False else f"  (usually {result['expect']}; this run: {result['outcome']})"
                verdict = "ok" if not hard and not result["violations"] else "CHECK: " + ", ".join(hard + result["violations"])
                print(f"{tag} {result['outcome']:13s} {verdict}{soft}")
    close_pools()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    kind = "stability" if args.stability_runs else "scorecard"
    agg = scorecard.aggregate(results)
    payload = {"created": stamp, "database": database, "profile": args.profile, "literature_backend": backend,
               "aggregate": agg, "cases": results}
    if args.stability_runs:
        payload["stability"] = scorecard.stability(results)
    (out_dir / f"{kind}-{stamp}.json").write_text(json.dumps(payload, indent=1, default=str))
    claims_path = out_dir / f"claims-{stamp}.jsonl"
    claims_path.write_text("".join(json.dumps(row) + "\n" for row in claims))
    claims_path.chmod(0o600)

    print("\n" + (scorecard.render_stability(payload["stability"]) if args.stability_runs else scorecard.render(agg)))
    print(f"\nraw results: {out_dir}/{kind}-{stamp}.json")
    print(f"claims for human adjudication ({len(claims)}, contains patient text): {claims_path}")
    return 1 if agg["violations"] or agg["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
