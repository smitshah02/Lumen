"""
Synthetic demo smoke tests (demo plane only)
============================================
Runs golden questions from src/demo_data/golden_qa.json through the PRODUCTION
retrieval and agent-graph code against the demo database. Prints ids, ranks and
verdicts only — never note text.

    LUMEN_DATA_PLANE=demo python scripts/demo_smoke_test.py safety
    LUMEN_DATA_PLANE=demo python scripts/demo_smoke_test.py retrieval [--ids demo_q01 ...]
    LUMEN_DATA_PLANE=demo python scripts/demo_smoke_test.py generation [--ids demo_q01 ...]

Safety PASS: the demo-plane assertions (plane, database, synthetic-only corpus,
no original note text) plus the plane-policy assertions, which need no database.
The research-plane assertion (no synthetic row reached the research database)
runs only when a research database actually exists: a demo-only deployment has
none, which is reported as "absent", not skipped in silence.

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


# Postgres SQLSTATE for "database ... does not exist". A demo-only deployment
# (RunPod) has no research database at all, which is the architecture working as
# intended — not a safety failure.
_UNDEFINED_DATABASE = "3D000"


def _research_url(storage):
    """Where the research database would live if this host had one.

    An explicit DATABASE_URL wins; otherwise it is the same server the demo
    plane is on, with the research database name.
    """
    from sqlalchemy.engine import make_url
    explicit = os.environ.get("DATABASE_URL")
    return make_url(explicit) if explicit else storage.engine.url.set(database=storage.RESEARCH_DB_NAME)


def _check_research_plane(storage) -> dict:
    """Assert no synthetic row reached the research database — when there is one.

    On a demo-only deployment the research database does not exist, so there is
    nothing that could have been contaminated and the guarantee holds trivially.
    That case is reported explicitly ("absent"), never silently skipped. A
    database that exists but cannot be read is a FAILURE: we found a research
    database and could not prove it is clean.
    """
    from sqlalchemy import create_engine, text
    url = _research_url(storage)
    res = {"database": url.database, "checked_tables": ["patients", "clinical_notes", "note_chunks"]}
    # create_engine inside the try as well: a malformed research URL must be
    # reported like any other unverifiable state, not crash the safety suite.
    engine = None
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 5})
        with engine.connect() as c:
            res["synthetic_rows_in_research_db"] = sum(
                c.execute(text(f"SELECT COUNT(*) FROM {t} WHERE subject_id BETWEEN :a AND :b"),
                          {"a": SYN[0], "b": SYN[1]}).scalar()
                for t in res["checked_tables"])
        res["state"] = "checked"
        res["result"] = "PASS" if res["synthetic_rows_in_research_db"] == 0 else "FAIL"
    except Exception as e:
        orig = getattr(e, "orig", e)
        code = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
        msg = str(orig).lower()
        # Match the DATABASE being absent, not a missing relation inside a
        # database that does exist — that one is unverifiable, not vacuous.
        absent = code == _UNDEFINED_DATABASE or ("database" in msg and "does not exist" in msg)
        if absent:
            res["state"] = "absent"
            res["result"] = "PASS"
            res["note"] = (f"no {url.database!r} database on this server — demo-only deployment; "
                           "the no-synthetic-rows-in-research guarantee holds trivially")
        else:
            res["state"] = "unreachable"
            res["result"] = "FAIL"
            res["error_type"] = type(orig).__name__
            res["note"] = "a research database may exist but could not be read; cannot prove it is clean"
    finally:
        if engine is not None:
            engine.dispose()
    return res


def cmd_safety() -> int:
    """Demo-plane safety assertions (always required) + research-plane
    assertions (only when a research database actually exists).

    Split deliberately: a demo-only cloud Pod must be able to prove its own
    isolation without a research database being present, and must still say so
    out loud rather than skipping the section.
    """
    from sqlalchemy import text
    from src import storage
    from src.obs import tracing

    # --- demo plane: always required ---------------------------------------
    demo = {"plane": storage.DATA_PLANE, "database": storage.engine.url.database,
            "research_db_name": storage.RESEARCH_DB_NAME}
    with storage.engine.connect() as c:
        demo["non_synthetic_rows"] = sum(
            c.execute(text(f"SELECT COUNT(*) FROM {t} WHERE subject_id NOT BETWEEN :a AND :b"), {"a": SYN[0], "b": SYN[1]}).scalar()
            for t in ("patients", "admissions", "diagnoses_icd", "labevents", "prescriptions", "clinical_notes", "note_chunks"))
        demo["notes_with_text_original"] = c.execute(text("SELECT COUNT(*) FROM clinical_notes WHERE text_original IS NOT NULL")).scalar()
        demo["notes_marked_synthetic"] = c.execute(text("SELECT COUNT(*) FROM clinical_notes WHERE phi_entities->>'synthetic' = 'true'")).scalar()
        demo["notes_total"] = c.execute(text("SELECT COUNT(*) FROM clinical_notes")).scalar()
        demo["subjects"] = c.execute(text("SELECT COUNT(*) FROM patients")).scalar()
    demo_ok = (demo["plane"] == "demo"
               and demo["database"] == storage.DEMO_DB_NAME
               and demo["database"] != storage.RESEARCH_DB_NAME
               and demo["non_synthetic_rows"] == 0
               and demo["notes_with_text_original"] == 0
               and demo["notes_total"] > 0
               and demo["notes_marked_synthetic"] == demo["notes_total"])
    demo["result"] = "PASS" if demo_ok else "FAIL"

    # --- plane policy: pure code, no database needed ------------------------
    tr = tracing.status()
    policy = {"tracing_enabled": bool(tr.get("enabled")), "tracing_host": tr.get("host"),
              "tracing_policy": tr.get("policy"), "tracing_state": tr.get("state"),
              "demo_db_isolated_from_research": storage.DEMO_DB_NAME != storage.RESEARCH_DB_NAME}
    # Traces carry note text, so a remote endpoint is only permissible on the
    # demo plane; src.obs.tracing enforces that in code, with no database. Only
    # "refused" means the policy was violated — "unavailable"/"misconfigured"
    # are operational states that do not leak anything.
    policy["tracing_policy_violation"] = tr.get("state") == "refused"
    policy["result"] = ("PASS" if policy["demo_db_isolated_from_research"]
                        and not policy["tracing_policy_violation"] else "FAIL")

    research = _check_research_plane(storage)

    out = {"plane": demo["plane"], "database": demo["database"],
           "demo": demo, "policy": policy, "research": research}
    ok = demo["result"] == "PASS" and policy["result"] == "PASS" and research["result"] == "PASS"
    out["result"] = "PASS" if ok else "FAIL"
    print(json.dumps(out, indent=1))
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
        # lab_evidence carries the [L#] sources the deterministic structured-lab
        # path cites. Omitting it made a correct deterministic answer look like
        # a hallucinated label and failed demo_q01 on a healthy deployment.
        evidence = ((st.get("patient_evidence") or []) + (st.get("guideline_evidence") or [])
                    + (st.get("literature_evidence") or []) + (st.get("lab_evidence") or []))
        rep = citations.validate(answer, evidence)
        n_cites = sum(len(c["valid_labels"]) > 0 for c in rep["claims"])
        v = st.get("verification") or {}
        found = _found_tokens(g["expected_facts"], answer)
        egress = st.get("egress_log") or []
        ok = bool(answer.strip()) and len(found) >= g["min_facts"] and n_cites >= 1 and not rep["bad_labels"]
        fails += not ok
        print(f"{g['id']:<10} {'PASS' if ok else 'FAIL':<6} {n_cites:>5} {len(rep['bad_labels']):>3} "
              f"{v.get('deterministic', 0) + v.get('checked', 0) - v.get('unsupported', 0):>3}"
              f"/{v.get('deterministic', 0) + v.get('checked', 0):<4}  "
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
