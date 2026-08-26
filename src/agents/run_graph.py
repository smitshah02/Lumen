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
    ap.add_argument("--show-evidence", action="store_true",
                    help="dump the evidence block that synthesis actually received")
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

    if "__interrupt__" in out:
        payload = out["__interrupt__"][0].value
        print(f"\n  PAUSED for human review — {len(payload['flagged'])} flagged claim(s)")
        print(f"  state is checkpointed; this process can exit safely.\n")
        print(f"  resume with:")
        print(f"    python -m src.agents.review_cli --thread {thread_id}\n")
        return 0

    print(f"\n  query        {out['query']}")
    print(f"  query_type   {out.get('query_type')}   temporal={out.get('temporal_mode')}")
    print(f"  node_trail   {' -> '.join(out.get('node_trail', []))}")
    print(f"  evidence     {len(out.get('patient_evidence', []))} patient / "
          f"{len(out.get('guideline_evidence', []))} guideline")

    if args.show_evidence:
        for tag, key in (("PATIENT", "patient_evidence"), ("GUIDELINE", "guideline_evidence")):
            items = out.get(key, []) or []
            if not items:
                continue
            print(f"\n  === {tag} EVIDENCE ({len(items)}) ===")
            for e in items:
                print(f"  [{e['label']}] score={e['score']}  {e.get('note_type')}  {e.get('charttime') or ''}")
                print(f"      {e['text'][:280].strip()}...")

    print("\n" + "-" * 70)
    print(out.get("final_answer") or out.get("draft_answer", "(no answer)"))
    print("-" * 70)

    v = out.get("verification", {}) or {}
    cr = v.get("citation_report", {}) or {}
    print(f"\n  claims       {cr.get('n_claims', 0)}   cite_rate={cr.get('cite_rate', 0):.0%}")
    print(f"  bad labels   {cr.get('bad_labels', [])}")
    print(f"  verified     {v.get('checked', 0) - v.get('unsupported', 0)}/{v.get('checked', 0)}")
    print(f"  review?      {out.get('needs_human_review')}")

    for c in out.get("citations", []):
        mark = "OK " if c["verified"] else "!! "
        print(f"   {mark}[{c['label'] or '--'}] {c['claim'][:70]}")
        if not c["verified"]:
            print(f"        {c['verification_note'][:90]}")

    snap = graph.get_state(config)
    print(f"\n  checkpoint   id={snap.config['configurable']['checkpoint_id'][:12]}...  next={snap.next}")
    print(f"  (re-run with --thread {thread_id} --history to inspect)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())