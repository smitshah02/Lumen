"""Stage 3 comparison (data-foundation plan, E15): control against the v2 candidate.

    python scripts/stage3_compare.py check                       # systems, manifests, frozen holdout fingerprint
    python scripts/stage3_compare.py retrieval                   # 42-question retrieval set (11 temporal), both systems
    python scripts/stage3_compare.py ask --set dev15             # targeted questions through /ask, both systems
    python scripts/stage3_compare.py holdout75                   # the 75-case scorecard, both systems
    python scripts/stage3_compare.py sheet --set dev15           # blinded grading sheet for a targeted set
    python scripts/stage3_compare.py report                      # stage3_comparison.json / .md and the gate

Nothing here decides what a right answer is. Each set keeps its own scorer:

    42 / 11   scripts/retrieval_eval.py labels and src/evals/retrieval_metrics.py
    75        scripts/scorecard.py and src/evals/scorecard.py, unchanged cases
    targeted  manual grading against the written ground truth, as in
              reports/data_foundation/before_benchmark.json, done blind to the system

Control and candidate are never compared by chunk_id: a v2 chunk_id is a handle
inside one table. Evidence is compared by note, section and offsets.

Each system runs in its own process, bound by environment to one database, one
profile and (for v2) one build, read from the evaluation manifests. Everything
written here stays in the git-ignored reports/data_foundation/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

REPORTS = ROOT / "reports" / "data_foundation"
OUT = REPORTS / "stage3"
HELDOUT_QUESTIONS = REPORTS / "heldout_questions.json"
PORT = 8765
HOLDOUT_FINGERPRINT_PREFIX = "72c7d06a"
HOLDOUT_BUILD = "e3a82619-bbea-440a-918f-7bcf4bacec58"
HOLDOUT_MANIFEST = Path.home() / "Lumen_local_results" / "holdout" / "manifest.json"
# One independent lab truth for both systems: the complete lab table, read from the holdout candidate
# database, which holds it for the same ten patients. The capped table is scored too, for history only.
LAB_TRUTH_TABLE, LAB_TRUTH_DATABASE = "labevents_full", "lumen_holdout_v2_eval"
WARMUP_QUERY = "What is the patient's social history?"       # asked once per system before timing; never scored
SOURCE_TABLES = ("patients", "admissions", "clinical_notes", "note_chunks", "chunk_search_labels", "labevents",
                 "d_labitems", "diagnoses_icd", "procedures_icd", "prescriptions", "note_index_state",
                 "ingestion_log", "ingestion_runs", "lumen_schema_version", "guideline_chunks")
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# --- which system runs where ---------------------------------------------------------------------

def systems() -> dict:
    """The four systems, from the frozen evaluation manifests."""
    research = json.loads((REPORTS / "eval_manifest_research_candidate.json").read_text())
    holdout = json.loads((REPORTS / "eval_manifest_holdout_candidate.json").read_text())
    control = json.loads((REPORTS / "eval_manifest_holdout_control.json").read_text())
    out = {
        "research_control": {"database": "lumen", "profile": "control", "build": None},
        "research_candidate": {"database": research["database"], "profile": research["data_profile"],
                               "build": research["v2_build"]["build_id"]},
        # The frozen holdout is never run against: a run writes checkpoint rows. Control runs on a
        # file-level copy of it, recorded in the control manifest with its fingerprint at creation.
        "holdout_control": {"database": (control.get("run_database") or {}).get("database"), "profile": control["data_profile"],
                            "build": None},
        "holdout_candidate": {"database": holdout["database"], "profile": holdout["data_profile"],
                              "build": holdout["v2_build"]["build_id"]},
    }
    expected = {"research_control": ("lumen", "control"), "research_candidate": ("lumen", "v2"),
                "holdout_control": ("lumen_holdout_control_eval", "control"), "holdout_candidate": ("lumen_holdout_v2_eval", "v2")}
    for name, (database, profile) in expected.items():
        if (out[name]["database"], out[name]["profile"]) != (database, profile):
            raise SystemExit(f"refusing: {name} is not {database}/{profile} in the manifests")
    if control["database"] != "lumen_holdout" or not (control.get("run_database") or {}).get("identical_to_frozen_holdout_at_creation"):
        raise SystemExit("refusing: the control copy is not recorded as identical to the frozen holdout")
    if out["holdout_candidate"]["build"] != HOLDOUT_BUILD or not out["research_candidate"]["build"]:
        raise SystemExit("refusing: candidate build ids do not match the manifests")
    return out


def env_for(system: dict) -> dict:
    """Environment that binds a child process to one system. The URL is never printed."""
    from sqlalchemy.engine import make_url
    from src import storage
    url = make_url(storage.DATABASE_URL).set(database=system["database"]).render_as_string(hide_password=False)
    env = {**os.environ, "DATABASE_URL": url, "LUMEN_DATA_PLANE": "research", "LUMEN_DATA_PROFILE": system["profile"],
           "LUMEN_TRACING": "0", "LUMEN_QUERY_EXPANSION": "0", "LUMEN_LITERATURE_BACKEND": "none",
           "LUMEN_API_PORT": str(PORT)}
    env.pop("LUMEN_CHUNK_BUILD", None)
    if system["build"]:
        env["LUMEN_CHUNK_BUILD"] = system["build"]
    return env


def fingerprint(database: str, tables=None) -> dict:
    """Row count and content hash per table, read-only. Counts and hashes only."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    from src import storage
    engine = create_engine(make_url(storage.DATABASE_URL).set(database=database),
                           connect_args={"options": "-c default_transaction_read_only=on"})
    out = {}
    with engine.connect() as c:
        names = [r[0] for r in c.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1"))]
        for name in names:
            if tables is not None and name not in tables:
                continue
            rows = c.execute(text(f'SELECT count(*) FROM "{name}"')).scalar()
            md5 = c.execute(text(f"SELECT md5(COALESCE(string_agg(h, '' ORDER BY h), '')) "
                                 f'FROM (SELECT md5(x::text) AS h FROM "{name}" x) s')).scalar()
            out[name] = {"rows": rows, "md5": md5}
    engine.dispose()
    return out


def fingerprint_id(tables: dict) -> str:
    return hashlib.sha256(json.dumps({t: (v["rows"], v["md5"]) for t, v in tables.items()}, sort_keys=True).encode()).hexdigest()


def holdout_state() -> dict:
    """The frozen holdout now, against the fingerprint recorded before any build:
    every table, and the source tables alone (a control run adds checkpoint rows)."""
    frozen = json.loads((REPORTS / "eval_manifest_holdout_control.json").read_text())["source_counts_before"]
    now = fingerprint("lumen_holdout")
    same = [t for t, v in frozen["tables"].items() if (now.get(t, {}).get("rows"), now.get(t, {}).get("md5")) == (v["rows"], v["md5"])]
    return {"fingerprint": fingerprint_id(now), "frozen_fingerprint": frozen["fingerprint"],
            "whole_database_identical": fingerprint_id(now) == frozen["fingerprint"],
            "source_tables_identical": all(t in same for t in SOURCE_TABLES if t in frozen["tables"]),
            "tables_changed": sorted(set(frozen["tables"]) - set(same)), "tables_added": sorted(set(now) - set(frozen["tables"]))}


def cmd_check(args) -> int:
    all_systems = systems()
    for name, s in all_systems.items():
        print(f"{name:20s} database={s['database']:22s} profile={s['profile']:8s} build={s['build'] or '-'}")
    state = holdout_state()
    print(f"lumen_holdout fingerprint {state['fingerprint'][:8]}… frozen {state['frozen_fingerprint'][:8]}… "
          f"identical={state['whole_database_identical']} source_tables_identical={state['source_tables_identical']} "
          f"changed={state['tables_changed']}")
    for name, s in all_systems.items():
        code = ("import json; from src import storage; from src.config import DATA_PROFILE, CHUNK_BUILD; "
                "from src.storage import readiness; "
                "print(json.dumps([storage.engine.url.database, DATA_PROFILE, CHUNK_BUILD, readiness.problems()]))")
        env = {**env_for(s), "PGOPTIONS": "-c default_transaction_read_only=on"}
        bound = json.loads(subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True).stdout.strip().splitlines()[-1])
        ok = bound[:3] == [s["database"], s["profile"], s["build"]] and not bound[3]
        print(f"{name:20s} bound={bound[:3]} problems={bound[3]} {'ok' if ok else 'MISMATCH'}")
        if not ok:
            return 2
    if args.require_frozen and not state["frozen_fingerprint"].startswith(HOLDOUT_FINGERPRINT_PREFIX):
        return 2
    return 0 if (state["whole_database_identical"] or not args.require_frozen) else 2


# --- 42-question retrieval set (11 temporal) ------------------------------------------------------

def offset_sections(note_text: str, chunks: list[tuple[int, int, int]]) -> dict[int, list[str]]:
    """retrieval_eval.chunk_sections for chunks whose character offsets are stored
    (v2): the same rule in the same unit, words. A chunk carries a section when
    half the chunk lies in the section, or half the section lies in the chunk.
    Nothing has to be searched for, so a chunk shorter than six words (a whole
    short section) is labelled too."""
    import bisect
    import retrieval_eval as bench
    starts = [m.start() for m in bench._WORD_RE.finditer(note_text)]
    spans = [(name, bisect.bisect_left(starts, start), bisect.bisect_left(starts, end))
             for name, start, end in bench.section_spans(note_text)]
    out = {}
    for chunk_id, char_start, char_end in chunks:
        first, last = bisect.bisect_left(starts, char_start), bisect.bisect_left(starts, char_end)
        found = []
        for name, start, end in spans:
            overlap = min(end, last) - max(start, first)
            if overlap > 0 and (2 * overlap >= last - first or 2 * overlap >= end - start):
                found.append((overlap, name))
        if found:
            out[chunk_id] = [name for _, name in sorted(found, reverse=True)]
    return out


def patient_labels(conn, subject_id: int, build: str | None, min_tokens: int) -> dict:
    """chunk_id -> (sections, note_id, charttime) for one patient, by the benchmark's
    own rule (retrieval_eval.build): the chunk lies in a standard section of the
    patient's own discharge summary, or the note is a chest radiology report.
    `build` selects the v2 chunks of that build instead of note_chunks."""
    from sqlalchemy import text
    import retrieval_eval as bench
    notes = conn.execute(text("""
        SELECT note_id, note_type, charttime, COALESCE(text_original, text_deid) FROM clinical_notes
        WHERE subject_id = :s ORDER BY charttime, note_id"""), {"s": subject_id}).fetchall()
    if build:
        chunks = conn.execute(text("""
            SELECT note_id, chunk_id, chunk_text, token_count, char_start, char_end FROM note_chunks_v2
            WHERE build_id = :b AND subject_id = :s ORDER BY note_id, section_ord, chunk_ord"""), {"b": build, "s": subject_id}).fetchall()
    else:
        chunks = conn.execute(text("""
            SELECT note_id, chunk_id, chunk_text, token_count FROM note_chunks
            WHERE subject_id = :s ORDER BY note_id, chunk_index"""), {"s": subject_id}).fetchall()
    by_note: dict[int, list] = {}
    offsets = {}
    for note_id, chunk_id, chunk_text, tokens, *span in chunks:
        by_note.setdefault(note_id, []).append((chunk_id, chunk_text, tokens or 0))
        if span:
            offsets[chunk_id] = tuple(span)
    labelled = {}
    for note_id, note_type, charttime, body in notes:
        mine = by_note.get(note_id, [])
        if note_type == "discharge":
            sections = (offset_sections(body or "", [(cid, *offsets[cid]) for cid, _, _ in mine]) if build
                        else bench.chunk_sections(body or "", [(cid, t) for cid, t, _ in mine]))
            for cid, _, tokens in mine:
                if cid in sections and tokens >= min_tokens:
                    labelled[cid] = (sections[cid], note_id, str(charttime))
        elif re.search(r"chest|cxr", (body or "")[:300], re.I):
            for cid, _, tokens in mine:
                if tokens >= min_tokens:
                    labelled[cid] = (["chest_radiology"], note_id, str(charttime))
    return labelled


def question_gold(question: dict, labelled: dict) -> dict:
    """Graded relevance for one benchmark question from labelled chunks, as retrieval_eval.build grades it."""
    import retrieval_eval as bench
    _, _, _, strong, weak, temporal = next(t for t in bench.TEMPLATES if t[0] == question["template"])
    gold = {cid: 2 for cid, (secs, _, _) in labelled.items() if set(secs) & set(strong)}
    gold.update({cid: 1 for cid, (secs, _, _) in labelled.items() if set(secs) & set(weak) and cid not in gold})
    if temporal and gold:
        times = sorted({labelled[cid][2] for cid in gold})
        target = times[-1] if temporal == "latest" else times[0]
        gold = {cid: (2 if labelled[cid][2] == target else 1) for cid in gold}
    return gold


def retrieval_child(out_path: str) -> int:
    """Runs bound to one system. Production search(), scored by the benchmark's metrics."""
    import logging
    logging.disable(logging.INFO)
    import retrieval_eval as bench
    import src.retrieval.hybrid_retriever_v2 as H
    from src import storage
    from src.config import CHUNK_BUILD, DATA_PROFILE
    from src.evals import retrieval_metrics as rm
    frozen = json.loads(bench.BENCHMARK.read_text())
    v2 = H.CHUNK_TABLE == H.V2_TABLE
    retriever = H.HybridRetriever()
    questions, rows, leaks, label_check = frozen["questions"], [], 0, {"questions": 0, "identical_to_frozen": 0}
    retriever.search(query=WARMUP_QUERY, subject_id=questions[0]["subject_id"], top_k=5)        # warm-up, not timed
    with storage.engine.connect() as conn:
        labels = {}
        for q in questions:
            sid = q["subject_id"]
            if sid not in labels:      # control labels are re-derived only to prove the derivation; v2 labels are scored
                labels[sid] = {"min40": patient_labels(conn, sid, CHUNK_BUILD if v2 else None, bench.MIN_TOKENS),
                               "any": patient_labels(conn, sid, CHUNK_BUILD, 0) if v2 else None}
    for q in questions:
        sid, frozen_gold = q["subject_id"], {int(k): v for k, v in q["relevance"].items()}
        derived = question_gold(q, labels[sid]["min40"])
        if v2:      # the frozen labels name note_chunks ids; the same rule over this build's chunks
            relevance, strict, gold_notes = question_gold(q, labels[sid]["any"]), derived, None
            gold_notes = {labels[sid]["any"][c][1] for c, g in relevance.items() if g == 2}
        else:
            relevance, strict = frozen_gold, None
            label_check["questions"] += 1
            label_check["identical_to_frozen"] += derived == frozen_gold
            gold_notes = {labels[sid]["min40"][c][1] for c, g in frozen_gold.items() if g == 2 and c in labels[sid]["min40"]}
        t0 = time.perf_counter()
        results = retriever.search(query=q["query"], subject_id=sid, temporal_filter="auto", top_k=bench.TOP_K)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        ranked = [r.chunk_id for r in results]
        leaks += sum(1 for r in results if r.subject_id != sid)
        row = {"id": q["id"], "kind": q["kind"], "temporal": q["temporal"], "subject_id": sid, "ms": ms,
               "gold_chunks": len(relevance), "gold_target_chunks": sum(1 for g in relevance.values() if g == 2),
               "ranked_notes": [r.note_id for r in results[:10]],
               "ranked_sources": [(r.provenance or {}).get("source_id") or f"note_chunks:{r.chunk_id}" for r in results[:10]],
               "source_note_hit@5": 1.0 if gold_notes & {r.note_id for r in results[:5]} else 0.0,
               **rm.score(ranked, relevance)}
        if q["temporal"]:
            row.update(rm.temporal_score(ranked, relevance))
        if strict is not None:
            row["strict"] = {"gold_chunks": len(strict), "hit@5": rm.score(ranked, strict)["hit@5"],
                             **({"target_hit@5": rm.temporal_score(ranked, strict)["target_hit@5"]} if q["temporal"] else {})}
        rows.append(row)
    Path(out_path).write_text(json.dumps({
        "database": storage.engine.url.database, "data_profile": DATA_PROFILE, "build": CHUNK_BUILD if v2 else None,
        "benchmark": {"name": frozen["name"], "seed": frozen["seed"], "questions": len(questions),
                      "sha256": hashlib.sha256(bench.BENCHMARK.read_bytes()).hexdigest()},
        "labels": ("frozen benchmark labels (note_chunks ids)" if not v2 else
                   "the benchmark's section rule applied to this build's chunks; no minimum token count, "
                   "because the v2 lexical arm can return short chunks; `strict` repeats the score with the >= 40 token rule"),
        "control_label_derivation_check": label_check if not v2 else None,
        "cross_patient_results": leaks, "per_question": rows}, indent=1))
    return 0


def cmd_retrieval(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_systems = systems()
    for name in ("research_control", "research_candidate"):
        out = OUT / f"retrieval_{name}.json"
        code = subprocess.run([sys.executable, __file__, "_retrieval", str(out)], cwd=ROOT,
                              env={**env_for(all_systems[name]), "PGOPTIONS": "-c default_transaction_read_only=on"}).returncode
        if code:
            return code
        data = json.loads(out.read_text())
        rows = data["per_question"]
        print(f"{name:20s} {data['database']}/{data['data_profile']} questions={len(rows)} hit@5={sum(r['hit@5'] for r in rows):.0f} "
              f"median_ms={statistics.median(r['ms'] for r in rows):.0f} leaks={data['cross_patient_results']} "
              f"label_check={data['control_label_derivation_check']}")
    return 0


# --- questions through /ask -----------------------------------------------------------------------

def _get(url: str, timeout: int = 30) -> dict:
    try:
        with _OPENER.open(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        try:
            return json.loads(getattr(e, "read", lambda: b"{}")() or b"{}")
        except ValueError:
            return {}


def _ask(subject_id: int, query: str) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/ask", method="POST",
                                 data=json.dumps({"subject_id": subject_id, "query": query}).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with _OPENER.open(req, timeout=900) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": type(e).__name__}


@contextmanager
def api(name: str, system: dict, warm_subject: int):
    """The API bound to one system, checked and warmed. Stopped on exit."""
    OUT.mkdir(parents=True, exist_ok=True)
    if _get(f"http://127.0.0.1:{PORT}/ready", 3):
        raise SystemExit(f"refusing: something is already listening on port {PORT}")
    log = open(OUT / f"api_{name}.log", "a")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.app:app", "--host", "127.0.0.1", "--port", str(PORT),
                             "--workers", "1"], cwd=ROOT, env=env_for(system), stdout=log, stderr=subprocess.STDOUT)
    try:
        info = {}
        for _ in range(120):
            info = _get(f"http://127.0.0.1:{PORT}/ready", 10)
            if info.get("status") == "ready" or proc.poll() is not None:
                break
            time.sleep(2)
        if (info.get("status"), info.get("database"), info.get("data_profile")) != ("ready", system["database"], system["profile"]):
            raise SystemExit(f"refusing: API for {name} is not ready on {system['database']}/{system['profile']}: "
                             f"{ {k: info.get(k) for k in ('status', 'database', 'data_profile')} }")
        warm = _ask(warm_subject, WARMUP_QUERY)                              # loads the retriever and the model; not scored
        if "answer" not in warm:
            raise SystemExit(f"refusing: warm-up request failed for {name}")
        builds = {(s.get("provenance") or {}).get("build_id") for s in warm.get("sources") or []
                  if str(s.get("label", "")).startswith("S")}
        if system["build"] and builds - {system["build"]}:
            raise SystemExit(f"refusing: {name} returned sources from another build")
        if not system["build"] and builds - {None}:
            raise SystemExit(f"refusing: {name} returned v2 sources under the control profile")
        yield info
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def load_questions(name: str, path: str | None) -> list[dict]:
    """A targeted set: {qid, subject_id, question, ground_truth, hadm_id?, note_id?, provenance?}."""
    if name == "dev15":
        return json.loads((REPORTS / "before_benchmark.json").read_text())["items"]
    data = json.loads(Path(path or HELDOUT_QUESTIONS).expanduser().read_text())
    return data["items"] if isinstance(data, dict) else data


def cmd_ask(args) -> int:
    questions = load_questions(args.set, args.questions)
    if args.set == "heldout":
        from src.retrieval.index_notes import read_subject_ids
        allowed = set(read_subject_ids("", str(REPORTS / "heldout_subject_ids.txt")))
        if {int(q["subject_id"]) for q in questions} != allowed:
            print("refusing: the held-out questions and heldout_subject_ids.txt name different patients", file=sys.stderr)
            return 2
    all_systems = systems()
    for name in ("research_control", "research_candidate"):
        out = OUT / f"ask_{args.set}_{name}.json"
        if out.exists():
            print(f"refusing: {out.name} exists; each question is asked once per system", file=sys.stderr)
            return 2
        rows = []
        with api(name, all_systems[name], warm_subject=10000032) as info:
            for q in questions:
                t0 = time.perf_counter()
                response = _ask(int(q["subject_id"]), q["question"])
                rows.append({"qid": q["qid"], "subject_id": int(q["subject_id"]), "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
                             "response": response})
                print(f"{name:20s} {q['qid']:12s} {response.get('status', response.get('error'))}")
        out.write_text(json.dumps({"set": args.set, "system": name, **all_systems[name], "ready": info,
                                   "created": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}, indent=1))
        out.chmod(0o600)
    return 0


def cmd_holdout75(args) -> int:
    subjects = json.loads(HOLDOUT_MANIFEST.read_text())["subject_ids"]
    all_systems = systems()
    for name in ("holdout_control", "holdout_candidate"):
        out_dir = OUT / f"holdout75_{name}"
        if out_dir.exists() and any(out_dir.glob("scorecard-*.json")):
            print(f"refusing: {out_dir.name} already holds a run; the 75 cases are run once per system", file=sys.stderr)
            return 2
        with api(name, all_systems[name], warm_subject=subjects[0]):
            code = subprocess.run([sys.executable, str(ROOT / "scripts" / "scorecard.py"), "--profile", "holdout",
                                   "--subjects-file", str(HOLDOUT_MANIFEST), "--out-dir", str(out_dir),
                                   "--api", f"http://127.0.0.1:{PORT}", "--lab-truth-table", LAB_TRUTH_TABLE,
                                   "--lab-truth-database", LAB_TRUTH_DATABASE], cwd=ROOT, env=env_for(all_systems[name])).returncode
        print(f"{name}: scorecard exit {code}")
        subprocess.run([sys.executable, __file__, "_scope", str(out_dir)], cwd=ROOT,
                       env={**env_for(all_systems[name]), "PGOPTIONS": "-c default_transaction_read_only=on"}, check=True)
    return 0


def scope_child(out_dir: str) -> int:
    """Admission-scope status of each scored case, read from the run's own checkpoint (read-only)."""
    import logging
    logging.disable(logging.WARNING)
    from src.agents.graph import build_graph, close_pools
    run = json.loads(sorted(Path(out_dir).glob("scorecard-*.json"))[-1].read_text())
    graph, _ = build_graph()
    rows = {}
    for case in run["cases"]:
        if case.get("thread_id"):
            state = graph.get_state({"configurable": {"thread_id": case["thread_id"]}}).values
            scope = state.get("admission_scope") or {}
            rows[case["thread_id"]] = {"status": scope.get("status", "not_enabled"), "hadm_id": scope.get("hadm_id"),
                                       "applied": bool(state.get("admission_scope_applied"))}
    close_pools()
    (Path(out_dir) / "scope.json").write_text(json.dumps(rows, indent=1))
    return 0


# --- blinded manual grading of a targeted set -----------------------------------------------------

def cmd_sheet(args) -> int:
    """One entry per (question, system), systems hidden as A/B in a per-question random
    order. The grader fills `grade` with correct / partial / incorrect."""
    sheet_path, key_path = OUT / f"grades_{args.set}.json", OUT / f"grades_{args.set}.key.json"
    if sheet_path.exists():
        print(f"refusing: {sheet_path.name} exists", file=sys.stderr)
        return 2
    questions = {q["qid"]: q for q in load_questions(args.set, args.questions)}
    runs = {name: {r["qid"]: r for r in json.loads((OUT / f"ask_{args.set}_{name}.json").read_text())["rows"]}
            for name in ("research_control", "research_candidate")}
    rng, sheet, key = random.Random(20261007), [], {}
    for qid, q in questions.items():
        order = ["research_control", "research_candidate"]
        rng.shuffle(order)
        key[qid] = dict(zip("AB", order))
        for arm, name in zip("AB", order):
            sheet.append({"qid": qid, "arm": arm, "question": q["question"], "ground_truth": q["ground_truth"],
                          "answer": re.sub(r"\[[A-Z]\d+\]", "[x]", (runs[name][qid]["response"].get("answer") or "")), "grade": None, "note": ""})
    sheet_path.write_text(json.dumps({"rubric": "correct = states the ground truth; partial = right core fact with a wrong or "
                                                "contradictory element; incorrect = anything else, including no answer",
                                      "entries": sheet}, indent=1))
    key_path.write_text(json.dumps(key, indent=1))
    for p in (sheet_path, key_path):
        p.chmod(0o600)
    print(f"{len(sheet)} entries to grade in {sheet_path.name}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--require-frozen", action="store_true", help="fail unless lumen_holdout matches its frozen fingerprint")
    sub.add_parser("retrieval")
    sub.add_parser("_retrieval").add_argument("out")
    sub.add_parser("_scope").add_argument("out_dir")
    for name in ("ask", "sheet"):
        s = sub.add_parser(name)
        s.add_argument("--set", choices=["dev15", "heldout"], required=True)
        s.add_argument("--questions", help="the held-out question file (never read for dev15)")
    sub.add_parser("holdout75")
    sub.add_parser("report")
    args = p.parse_args(argv)
    if args.command == "_retrieval":
        return retrieval_child(args.out)
    if args.command == "_scope":
        return scope_child(args.out_dir)
    if args.command == "report":
        import stage3_report
        return stage3_report.main()
    return {"check": cmd_check, "retrieval": cmd_retrieval, "ask": cmd_ask, "holdout75": cmd_holdout75,
            "sheet": cmd_sheet}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
