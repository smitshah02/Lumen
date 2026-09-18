"""
Synthetic demo smoke tests (demo plane only)
============================================
Runs golden questions from src/demo_data/golden_qa.json through the PRODUCTION
retrieval and agent-graph code against the demo database. Prints ids, ranks and
verdicts only — never note text.

    LUMEN_DATA_PLANE=demo python scripts/demo_smoke_test.py safety
    LUMEN_DATA_PLANE=demo python scripts/demo_smoke_test.py retrieval [--ids demo_q01 ...]
    LUMEN_DATA_PLANE=demo python scripts/demo_smoke_test.py generation [--ids demo_q01 ...]

Retrieval PASS: at least `min_facts` expected facts appear in the top-5 evidence
texts the graph would hand to synthesis (PATIENT_TOP_K, temporal mode auto).
Generation PASS: non-empty answer, >= min_facts expected facts in the answer
(every token of the fact present, order-free — prose says "creatinine was 1.4
mg/dL"), >= 1 valid citation, no hallucinated citation labels.
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import logging
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GOLDEN = ROOT / "src" / "demo_data" / "golden_qa.json"
SYN = (90000000, 90999999)
DEFAULT_RETRIEVAL = ["demo_q01", "demo_q02", "demo_q03", "demo_q04", "demo_q05", "demo_q06"]
DEFAULT_GENERATION = ["demo_q01", "demo_q03", "demo_q06"]


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _found(facts, text) -> list[str]:
    t = _norm(text)
    return [f for f in facts if _norm(f) in t]


def _found_tokens(facts, text) -> list[str]:
    toks = set(_norm(text).replace("[", " ").replace("]", " ").replace(",", " ").rstrip(".").split())
    toks |= {t.rstrip(".,;:") for t in toks}
    return [f for f in facts if all(w in toks for w in _norm(f).split())]


def _golden(ids):
    items = {g["id"]: g for g in json.loads(GOLDEN.read_text())}
    return [items[i] for i in ids]


def cmd_safety() -> int:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    from src import storage
    out = {"plane": storage.DATA_PLANE, "database": storage.engine.url.database}
    with storage.engine.connect() as c:
        out["non_synthetic_rows"] = sum(
            c.execute(text(f"SELECT COUNT(*) FROM {t} WHERE subject_id NOT BETWEEN :a AND :b"), {"a": SYN[0], "b": SYN[1]}).scalar()
            for t in ("patients", "admissions", "diagnoses_icd", "labevents", "prescriptions", "clinical_notes", "note_chunks"))
        out["notes_with_text_original"] = c.execute(text("SELECT COUNT(*) FROM clinical_notes WHERE text_original IS NOT NULL")).scalar()
        out["notes_marked_synthetic"] = c.execute(text("SELECT COUNT(*) FROM clinical_notes WHERE phi_entities->>'synthetic' = 'true'")).scalar()
        out["notes_total"] = c.execute(text("SELECT COUNT(*) FROM clinical_notes")).scalar()
    # The research database must not have received any synthetic rows.
    research = create_engine(make_url(os.environ.get("DATABASE_URL", "")) if os.environ.get("DATABASE_URL") else
                             storage.engine.url.set(database=storage.RESEARCH_DB_NAME))
    with research.connect() as c:
        out["synthetic_rows_in_research_db"] = sum(
            c.execute(text(f"SELECT COUNT(*) FROM {t} WHERE subject_id BETWEEN :a AND :b"), {"a": SYN[0], "b": SYN[1]}).scalar()
            for t in ("patients", "clinical_notes", "note_chunks"))
    research.dispose()
    ok = (out["plane"] == "demo" and out["database"] != storage.RESEARCH_DB_NAME and out["non_synthetic_rows"] == 0
          and out["notes_with_text_original"] == 0 and out["notes_marked_synthetic"] == out["notes_total"]
          and out["synthetic_rows_in_research_db"] == 0)
    out["result"] = "PASS" if ok else "FAIL"
    print(json.dumps(out))
    return 0 if ok else 1


def cmd_retrieval(ids) -> int:
    from src.agents.graph import get_retrievers, PATIENT_TOP_K
    retriever, _ = get_retrievers()
    fails = 0
    print(f"{'id':<10} {'result':<6} {'rank':>4}  expected (found/needed)")
    for g in _golden(ids):
        res = retriever.search(query=g["query"], subject_id=g["subject_id"], temporal_filter="auto", top_k=PATIENT_TOP_K)
        texts = [(r.context_text or r.chunk_text) for r in res]      # what synthesis receives
        assert all(SYN[0] <= r.subject_id <= SYN[1] for r in res)
        found, rank = set(), None
        for i, t in enumerate(texts, 1):
            hit = _found(g["expected_facts"], t)
            if hit and rank is None:
                rank = i
            found.update(hit)
        ok = len(found) >= g["min_facts"]
        fails += not ok
        print(f"{g['id']:<10} {'PASS' if ok else 'FAIL':<6} {rank or '-':>4}  "
              f"{' | '.join(g['expected_facts'])} ({len(found)}/{g['min_facts']})")
    return 1 if fails else 0


def cmd_generation(ids) -> int:
    from src.agents.graph import build_graph, close_pools
    from src.agents import citations
    from src.llm.local_client import MAIN_MODEL, FAST_MODEL, HOST
    graph, _ = build_graph()
    fails = 0
    print(f"models: main={MAIN_MODEL} fast={FAST_MODEL} host={HOST}")
    print(f"{'id':<10} {'result':<6} {'cites':>5} {'bad':>3} {'verified':>8}  facts (found/needed)  review")
    for g in _golden(ids):
        cfg = {"configurable": {"thread_id": f"demo-smoke-{g['id']}-{uuid.uuid4().hex[:6]}"}}
        graph.invoke({"query": g["query"], "subject_id": g["subject_id"], "thread_id": cfg["configurable"]["thread_id"]}, config=cfg)
        st = graph.get_state(cfg).values          # also covers a run paused at human_review
        answer = st.get("final_answer") or st.get("draft_answer") or ""
        evidence = (st.get("patient_evidence") or []) + (st.get("guideline_evidence") or []) + (st.get("literature_evidence") or [])
        rep = citations.validate(answer, evidence)
        n_cites = sum(len(c["valid_labels"]) > 0 for c in rep["claims"])
        v = st.get("verification") or {}
        found = _found_tokens(g["expected_facts"], answer)
        egress = st.get("egress_log") or []
        ok = bool(answer.strip()) and len(found) >= g["min_facts"] and n_cites >= 1 and not rep["bad_labels"]
        fails += not ok
        print(f"{g['id']:<10} {'PASS' if ok else 'FAIL':<6} {n_cites:>5} {len(rep['bad_labels']):>3} "
              f"{v.get('checked', 0) - v.get('unsupported', 0):>3}/{v.get('checked', 0):<4}  "
              f"{len(found)}/{g['min_facts']} {found}  review={st.get('review_status')} external_calls={len(egress)}")
    close_pools()
    return 1 if fails else 0


def main() -> int:
    import src  # noqa: F401  (loads .env)
    if os.environ.get("LUMEN_DATA_PLANE", "research").strip().lower() != "demo":
        print("refusing: set LUMEN_DATA_PLANE=demo", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["safety", "retrieval", "generation"])
    ap.add_argument("--ids", nargs="+", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    if args.mode == "safety":
        return cmd_safety()
    if args.mode == "retrieval":
        return cmd_retrieval(args.ids or DEFAULT_RETRIEVAL)
    return cmd_generation(args.ids or DEFAULT_GENERATION)


if __name__ == "__main__":
    raise SystemExit(main())
