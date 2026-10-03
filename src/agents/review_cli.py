"""
Clinician Review CLI
====================
Presents the flagged claims of a run paused at human_review alongside the
source text each was cited against, collects decisions, and resumes the run
from its checkpoint through `src.agents.review` — the same service the API
uses. Presentation and input live here; nothing about resuming does.

    python -m src.agents.run_graph --query ... --subject ...   # asks, reviews inline, finishes
    python -m src.agents.review_cli --thread <thread_id>        # review a run paused earlier
    python -m src.agents.review_cli --list

Durability check: start a run in one process, let it pause, exit, then
run this in a fresh process. State comes from Postgres, not memory.
"""

from __future__ import annotations

import logging
import argparse

from src.agents import review
from src.agents.graph import build_graph
from src.obs import tracing

BAR = "=" * 72
_ACTIONS = {"a": "approve", "approve": "approve", "pass": "approve",
            "r": "reject", "reject": "reject",
            "s": "strike", "strike": "strike", "e": "escalate", "escalate": "escalate"}


def _prompt(flag: dict, n: int, total: int, ask=None) -> dict:
    ask = ask or input        # looked up at call time, so a caller can supply its own
    print("\n" + BAR)
    print(f"  FLAGGED CLAIM {n}/{total}   cited as [{flag['label'] or '--'}]")
    print(BAR)
    print(f"\n  CLAIM:\n    {flag['claim']}")
    print(f"\n  VERIFIER:\n    {flag['note']}")
    src = (flag.get("source_text") or "").strip()
    print(f"\n  CITED SOURCE:\n    {src if src else '(no source resolved)'}")
    print("\n  [a] approve this claim   [r] reject the whole draft"
          "   [s] strike this claim   [e] escalate")

    while True:
        action = _ACTIONS.get(ask("  > ").strip().lower())
        if action:
            return {"action": action, "note": ask("  note (optional): ").strip()}
        print("  please enter a, r, s, or e")


def collect(payload: dict, ask=None) -> list[dict]:
    """Show a paused draft and gather one decision per flagged claim. Rejecting
    rejects the whole draft, so no further claim is asked about."""
    flagged = payload["flagged"]
    print("\n" + BAR)
    print(f"  HUMAN REVIEW REQUIRED\n  QUERY:  {payload['query']}")
    print(BAR)
    print(f"\n  DRAFT ANSWER:\n    {payload['draft_answer']}\n")
    print(f"  {len(flagged)} claim(s) require adjudication.")

    decisions = []
    for i, flag in enumerate(flagged, 1):
        decision = _prompt(flag, i, len(flagged), ask)
        if decision["action"] == "reject":
            return [decision] * len(flagged)
        decisions.append(decision)
    return decisions


def review_interactively(graph, thread_id: str, ask=None) -> dict:
    """Prompt for the paused run's decisions and resume it from its checkpoint.
    Returns the final state. Raises review.ReviewNotFound / ReviewNotPending."""
    decisions = collect(review.pending(graph, thread_id), ask)
    print("\n  Resuming workflow...")
    return review.resume(graph, thread_id, decisions)


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
        tracing.flush()
        return 0

    if not args.thread:
        ap.error("--thread is required (or use --list)")

    try:
        out = review_interactively(graph, args.thread)
    except review.ReviewNotFound:
        print(f"\n  no checkpoint for thread {args.thread}.")
        return 1
    except review.ReviewNotPending as e:
        print(f"\n  thread {args.thread} is not paused — nothing to review.")
        print(f"  status: {e.review_status}")
        tracing.flush()
        return 0

    print("\n" + BAR)
    print(f"  STATUS: {out.get('review_status')}")
    print(BAR)
    print(f"\n{out.get('final_answer') or '(all claims struck)'}\n")
    for d in out.get("human_decisions", []):
        print(f"    #{d['index']}  {d['action']}  {d['note']}")
    print(f"\n  trail: {' -> '.join(out.get('node_trail', []))}\n")
    # The SDK batches; a short-lived CLI would exit before the background send.
    tracing.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
