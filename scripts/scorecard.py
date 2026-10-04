"""Run the research scorecard against the local API.

    ./scripts/lumen research scorecard
    ./scripts/lumen research scorecard --only structured,hitl
    ./scripts/lumen research scorecard --with-literature      # API started with LUMEN_LITERATURE_BACKEND=pubmed

Each case is asked through /ask, then checked against the database and the
run's checkpoint (see src/evals/scorecard.py). Prints one status line per case
and an aggregate table — never an answer, a claim or note text. Raw per-case
results (ids, trails, counts, timings) and a claim adjudication sample are
written under --out-dir on this machine; the adjudication file contains
patient text for a human to review and is not printed.

Local only: refuses a non-loopback API and a non-research plane, and refuses
to run by default while the API's PubMed backend is enabled.
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
from src.evals import scorecard  # noqa: E402


def literature_backend(api: str) -> str:
    try:
        with parity._OPENER.open(f"{api}/ready", timeout=60) as r:
            return json.loads(r.read()).get("literature_backend", "none")
    except Exception as e:                       # /ready answers 503 with a body when a dependency is down
        body = getattr(e, "read", lambda: b"{}")()
        try:
            return json.loads(body or b"{}").get("literature_backend", "none")
        except ValueError:
            return "unknown"


def build_cases(sids: list[int], truths: list[dict], backend: str, only: set[str]) -> list[tuple[int, dict]]:
    cases = []
    for sid, truth in zip(sids, truths):
        for cls, query, node, ok in parity.checks(truth):
            if cls in scorecard.STRUCTURED_CLASSES:
                cases.append((sid, scorecard.structured_case(cls, query, node, ok)))
    rag = [c for c in scorecard.RAG_CASES if c.get("backend", backend) == backend]
    cases += [(sids[i % len(sids)], case) for i, case in enumerate(rag)]
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


def adjudication_rows(result: dict, response: dict, state: dict) -> list[dict]:
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
                    "sources": {l: (evidence.get(l, {}).get("text") or "")[:800] for l in labels},
                    "human_verdict": None})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--subjects-file", default="~/Lumen_local_results/p0/manual_subjects.json",
                   help="local JSON list of subject ids")
    p.add_argument("--out-dir", default="~/Lumen_local_results/scorecard")
    p.add_argument("--only", default="", help="comma-separated case ids or categories (for a quick run)")
    p.add_argument("--with-literature", action="store_true",
                   help="allow a run while the API's PubMed backend is enabled (makes outbound searches)")
    p.add_argument("--adjudication", type=int, default=15, help="claims to write for manual review")
    args = p.parse_args(argv)

    if storage.DATA_PLANE != "research":
        print(f"refusing: the scorecard runs on the research plane, not {storage.DATA_PLANE!r}", file=sys.stderr)
        return 2
    if urlparse(args.api).hostname not in ("127.0.0.1", "localhost", "::1"):
        print("refusing: the API must be on loopback", file=sys.stderr)
        return 2
    backend = literature_backend(args.api)
    if backend == "pubmed" and not args.with_literature:
        print("refusing: the API has LUMEN_LITERATURE_BACKEND=pubmed, so literature questions would search PubMed.\n"
              "Restart the API without it, or pass --with-literature to allow those outbound searches.", file=sys.stderr)
        return 2
    if backend not in ("none", "pubmed"):
        print("refusing: could not read literature_backend from /ready (is the API running?)", file=sys.stderr)
        return 2

    sids = [int(s) for s in json.loads(Path(args.subjects_file).expanduser().read_text())]
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    import logging
    logging.disable(logging.WARNING)
    from src.agents.graph import build_graph, close_pools      # checkpoints only; loads no model
    graph, _ = build_graph()

    results, adjudication = [], []
    with storage.engine.connect() as conn:
        truths = [parity.truth(conn, sid, ["creatinine"]) for sid in sids]
        cases = build_cases(sids, truths, backend, only)
        print(f"{len(cases)} case(s), {len(sids)} patient(s), literature backend: {backend}\n")
        for sid, case in cases:
            alias = f"P{sids.index(sid) + 1}"
            response = parity.ask(args.api, sid, case["query"])
            if "answer" not in response:
                results.append({"case": case["id"], "category": case["category"], "subject_id": sid,
                                "error": response.get("error", "no answer"), "violations": ["request failed"]})
                print(f"{alias} {case['id']:30s} ERROR {response.get('error')}")
                continue
            state = graph.get_state({"configurable": {"thread_id": response["thread_id"]}}).values
            result = scorecard.evaluate(case, sid, response, state, database_rows(conn, response.get("sources") or []),
                                        backend)
            results.append(result)
            adjudication += adjudication_rows(result, response, state)
            flags = [k[:-3] for k in ("routing_ok", "review_ok", "abstain_ok", "temporal_ok", "structured_ok", "evidence_ok")
                     if result[k] is False]
            print(f"{alias} {case['id']:30s} {result['status']:22s} "
                  f"{'ok' if not flags and not result['violations'] else 'CHECK: ' + ', '.join(flags + result['violations'])}")
    close_pools()

    agg = scorecard.aggregate(results)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"scorecard-{stamp}.json").write_text(json.dumps(
        {"created": stamp, "literature_backend": backend, "aggregate": agg, "cases": results}, indent=1, default=str))
    flagged = [r for r in adjudication if not r["verifier_verdict"]]
    passed = [r for r in adjudication if r["verifier_verdict"]]
    sample = (flagged + passed)[:args.adjudication] if len(flagged) >= args.adjudication else \
        flagged + passed[:args.adjudication - len(flagged)]
    with (out_dir / f"adjudication-{stamp}.jsonl").open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in sample)

    print("\n" + scorecard.render(agg))
    print(f"\nraw results: {out_dir}/scorecard-{stamp}.json")
    print(f"claims for manual adjudication ({len(sample)}, contains patient text): {out_dir}/adjudication-{stamp}.jsonl")
    return 1 if agg["violations"] or agg["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
