"""
Clinician Review CLI
====================
Resumes a graph paused at human_review, presents each flagged claim
alongside the source text it was cited against, and collects decisions.

    python -m src.agents.review_cli --thread <thread_id>
    python -m src.agents.review_cli --list

Durability check: start a run in one process, let it pause, exit, then
run this in a fresh process. State comes from Postgres, not memory.
"""

from __future__ import annotations

import logging
import argparse

from langgraph.types import Command

from src.agents.graph import build_graph

BAR = "=" * 72


def _prompt(flag: dict, n: int, total: int) -> dict:
    print("\n" + BAR)
    print(f"  FLAGGED CLAIM {n}/{total}   cited as [{flag['label'] or '--'}]")
    print(BAR)
    print(f"\n  CLAIM:\n    {flag['claim']}")
    print(f"\n  VERIFIER:\n    {flag['note']}")
    src = (flag.get("source_text") or "").strip()
    print(f"\n  CITED SOURCE:\n    {src if src else '(no source resolved)'}")
    print("\n  [a] approve   [s] strike   [e] escalate")

    while True:
        choice = input("  > ").strip().lower()
        if choice in ("a", "approve"):
            return {"action": "approve", "note": input("  note (optional): ").strip()}
        if choice in ("s", "strike"):
            return {"action": "strike", "note": input("  reason (optional): ").strip()}
        if choice in ("e", "escalate"):
            return {"action": "escalate", "note": input("  reason (optional): ").strip()}
        print("  please enter a, s, or e")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread", help="thread_id of the paused run")
    ap.add_argument("--list", action="store_true", help="show threads awaiting review")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    graph, _ = build_graph()

    if args.list:
        from src.storage import engine
        from sqlalchemy import text
        with engine.connect() as c:
            rows = c.execute(text(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            )).fetchall()
        print(f"\n  {len(rows)} thread(s):")
        for r in rows:
            st = graph.get_state({"configurable": {"thread_id": r[0]}})
            status = "AWAITING REVIEW" if st.next else "complete"
            print(f"    {r[0]:<24} {status}")
        return 0

    if not args.thread:
        ap.error("--thread is required (or use --list)")

    config = {"configurable": {"thread_id": args.thread}}
    snap = graph.get_state(config)

    if not snap.next:
        print(f"\n  thread {args.thread} is not paused — nothing to review.")
        print(f"  status: {snap.values.get('review_status')}")
        return 0

    interrupts = [i for t in snap.tasks for i in (t.interrupts or [])]
    if not interrupts:
        print(f"\n  thread {args.thread} is at {snap.next} but has no pending interrupt.")
        return 1

    payload = interrupts[0].value
    flagged = payload["flagged"]

    print("\n" + BAR)
    print(f"  QUERY:  {payload['query']}")
    print(BAR)
    print(f"\n  DRAFT ANSWER:\n    {payload['answer']}\n")
    print(f"  {len(flagged)} claim(s) require adjudication.")

    decisions = [_prompt(f, i, len(flagged)) for i, f in enumerate(flagged, 1)]

    # Same thread_id, same config — a different one cannot find the frozen state.
    out = graph.invoke(Command(resume=decisions), config=config)

    print("\n" + BAR)
    print(f"  STATUS: {out.get('review_status')}")
    print(BAR)
    print(f"\n{out.get('final_answer') or '(all claims struck)'}\n")
    for d in out.get("human_decisions", []):
        print(f"    #{d['index']}  {d['action']}  {d['note']}")
    print(f"\n  trail: {' -> '.join(out.get('node_trail', []))}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())