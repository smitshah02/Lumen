"""
Graph smoke runner
==================
    python -m src.agents.run_graph
    python -m src.agents.run_graph --query "current medications" --subject 10000032
    python -m src.agents.run_graph --thread demo-1 --history
"""

from __future__ import annotations

import uuid
import logging
import argparse

from src.agents.graph import build_graph


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="most recent creatinine value")
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--thread", default=None)
    ap.add_argument("--history", action="store_true",
                    help="print checkpoint history for the thread instead of running")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    thread_id = args.thread or f"smoke-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    graph, _ = build_graph()

    if args.history:
        for i, snap in enumerate(graph.get_state_history(config)):
            print(f"  {i}: next={snap.next}  trail={snap.values.get('node_trail')}")
        return 0

    print("=" * 70)
    print(f"  LUMEN AGENT GRAPH — thread {thread_id}")
    print("=" * 70)

    out = graph.invoke(
        {"query": args.query, "subject_id": args.subject, "thread_id": thread_id},
        config=config,
    )

    print(f"\n  query        {out['query']}")
    print(f"  query_type   {out.get('query_type')}")
    print(f"  node_trail   {' -> '.join(out.get('node_trail', []))}")
    print(f"  review?      {out.get('needs_human_review')}")

    snap = graph.get_state(config)
    print(f"\n  checkpoint   id={snap.config['configurable']['checkpoint_id'][:12]}...  next={snap.next}")
    print(f"  (re-run with --thread {thread_id} --history to inspect)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())