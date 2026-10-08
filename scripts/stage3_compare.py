"""Stage 3 comparison (data-foundation plan, E15): control against the v2 candidate.

    python scripts/stage3_compare.py check                       # systems, manifests, frozen holdout fingerprint
    python scripts/stage3_compare.py retrieval                   # 42-question retrieval set (11 temporal), both systems
    python scripts/stage3_compare.py ask --set dev15             # targeted questions through /ask, both systems
    python scripts/stage3_compare.py holdout75                   # the 75-case scorecard, both systems
    python scripts/stage3_compare.py sheet --set dev15           # blinded grading sheet for a targeted set
    python scripts/stage3_compare.py report                      # stage3_comparison.json / .md and the gate

A later run names itself and its own held-out files, and writes nowhere else:

    python scripts/stage3_compare.py --run-id stage3_run2 --heldout-questions Q.json --heldout-subjects IDS.txt preflight
    ... retrieval | ask --set dev15 | freeze | holdout75 | ask --set heldout | sheet --set heldout | report

Results are written once and never overwritten; held-out material is only run under that run's freeze.

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
# --- which evaluation run ---------------------------------------------------------------------------
# Every run has its own outputs and never writes into another's. The first run keeps the
# paths it was made with; a later run "stage3_runN" gets stage3/runN/ and stage3_runN_*.
RUN1 = "stage3"
RUN_ID = RUN1
OUT = REPORTS / "stage3"
FREEZE = REPORTS / "stage3_freeze.json"
COMPARISON_JSON, COMPARISON_MD = REPORTS / "stage3_comparison.json", REPORTS / "stage3_comparison.md"
HELDOUT_QUESTIONS: Path | None = REPORTS / "heldout_questions.json"
HELDOUT_SUBJECTS: Path | None = REPORTS / "heldout_subject_ids.txt"


def run_paths(run_id: str) -> dict:
    """Where one run keeps everything it writes."""
    if run_id != RUN1 and not re.fullmatch(r"stage3_run[0-9]+", run_id):
        raise SystemExit(f"refusing: run id must be {RUN1!r} or 'stage3_runN', not {run_id!r}")
    return {"out": REPORTS / "stage3" if run_id == RUN1 else REPORTS / "stage3" / run_id[len("stage3_"):],
            "freeze": REPORTS / f"{run_id}_freeze.json",
            "comparison_json": REPORTS / f"{run_id}_comparison.json", "comparison_md": REPORTS / f"{run_id}_comparison.md"}


def use_run(run_id: str = RUN1, heldout_questions: str | None = None, heldout_subjects: str | None = None) -> None:
    """Point the harness at one run. The first run is tied to the held-out files it was run with.
    Any other run must be given its own, explicitly, and may not reuse the first run's."""
    global RUN_ID, OUT, FREEZE, COMPARISON_JSON, COMPARISON_MD, HELDOUT_QUESTIONS, HELDOUT_SUBJECTS
    paths = run_paths(run_id)
    first = (REPORTS / "heldout_questions.json", REPORTS / "heldout_subject_ids.txt")
    given = tuple(Path(p).expanduser().resolve() if p else None for p in (heldout_questions, heldout_subjects))
    if run_id == RUN1:
        if any(g is not None and g != f.resolve() for g, f in zip(given, first)):
            raise SystemExit(f"refusing: run {RUN1!r} is tied to its own held-out files; use --run-id for another run")
        given = first
    elif any(g is not None and g in (f.resolve() for f in first) for g in given):
        raise SystemExit(f"refusing: run {run_id!r} may not reuse the held-out files of run {RUN1!r}")
    RUN_ID, OUT, FREEZE = run_id, paths["out"], paths["freeze"]
    COMPARISON_JSON, COMPARISON_MD = paths["comparison_json"], paths["comparison_md"]
    HELDOUT_QUESTIONS, HELDOUT_SUBJECTS = given


def heldout_files() -> tuple[Path, Path]:
    if HELDOUT_QUESTIONS is None or HELDOUT_SUBJECTS is None:
        raise SystemExit(f"refusing: run {RUN_ID!r} needs --heldout-questions and --heldout-subjects")
    return HELDOUT_QUESTIONS, HELDOUT_SUBJECTS


def refuse_existing(*paths: Path) -> None:
    """A result is written once. There is no override: a result that should not exist is removed by hand."""
    for path in paths:
        if path.exists():
            raise SystemExit(f"refusing: {path.relative_to(REPORTS)} already exists for run {RUN_ID!r}; results are never overwritten")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen() -> dict | None:
    return json.loads(FREEZE.read_text()) if FREEZE.exists() else None


def require_freeze(check_heldout: bool = False) -> dict:
    """Held-out material is only run under a written freeze, with the files that freeze names."""
    freeze = frozen()
    if not freeze:
        raise SystemExit(f"refusing: run {RUN_ID!r} has no freeze ({FREEZE.name}); write it before any held-out run")
    if check_heldout:
        questions, subjects = heldout_files()
        if (sha256(questions), sha256(subjects)) != (freeze["heldout"]["questions_sha256"], freeze["heldout"]["ids_sha256"]):
            raise SystemExit("refusing: the held-out files are not the ones the freeze recorded")
    return freeze

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
    """The four systems: as the run's freeze recorded them once it exists (so an
    earlier run keeps the build it was run on), otherwise from the evaluation manifests."""
    freeze = frozen()
    if freeze:
        return freeze["systems"]
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
    refuse_existing(FREEZE, *(OUT / f"retrieval_{name}.json" for name in ("research_control", "research_candidate")))
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
    data = json.loads(heldout_files()[0].read_text())
    return data["items"] if isinstance(data, dict) else data


def validate_heldout(questions_path: Path, subjects_path: Path, build_subjects) -> dict:
    """Shape only: the fields are there, ids are unique, and the patients are exactly
    those of the id file and all in the candidate build. No question or answer is returned."""
    from src.retrieval.index_notes import read_subject_ids
    data = json.loads(Path(questions_path).read_text())
    items = data["items"] if isinstance(data, dict) else data
    problems = []
    if not isinstance(items, list) or not items:
        raise SystemExit("refusing: the held-out question file holds no list of items")
    for field in ("qid", "subject_id", "question", "ground_truth"):
        missing = sum(1 for q in items if not isinstance(q, dict) or not str(q.get(field, "")).strip())
        if missing:
            problems.append(f"{missing} item(s) without {field}")
    qids = [q.get("qid") for q in items if isinstance(q, dict)]
    if len(set(qids)) != len(qids):
        problems.append("question ids are not unique")
    try:
        asked = {int(q["subject_id"]) for q in items}
    except (KeyError, TypeError, ValueError):
        asked = set()
        problems.append("a subject_id is not an integer")
    allowed = set(read_subject_ids("", str(subjects_path)))
    if asked != allowed:
        problems.append("the questions and the id file name different patients")
    if not allowed <= {int(s) for s in build_subjects}:
        problems.append("a held-out patient is not in the candidate build")
    if problems:
        raise SystemExit("refusing: " + "; ".join(problems))
    return {"questions": len(items), "patients": len(allowed)}


def cmd_ask(args) -> int:
    if args.set == "heldout":
        freeze = require_freeze(check_heldout=True)
        validate_heldout(*heldout_files(), freeze["builds"].get("research_candidate_patients")
                         or json.loads((REPORTS / "eval_manifest_research_candidate.json").read_text())["patient_ids"])
    refuse_existing(*(OUT / f"ask_{args.set}_{name}.json" for name in ("research_control", "research_candidate")))
    questions = load_questions(args.set, None)
    all_systems = systems()
    for name in ("research_control", "research_candidate"):
        out = OUT / f"ask_{args.set}_{name}.json"
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
    require_freeze()
    subjects = json.loads(HOLDOUT_MANIFEST.read_text())["subject_ids"]
    all_systems = systems()
    for name in ("holdout_control", "holdout_candidate"):           # checked for both before either is run
        refuse_existing(*(OUT / f"holdout75_{name}").glob("scorecard-*.json"))
    for name in ("holdout_control", "holdout_candidate"):
        out_dir = OUT / f"holdout75_{name}"
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
    refuse_existing(sheet_path, key_path)
    questions = {q["qid"]: q for q in load_questions(args.set, None)}
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


# --- sealing a finished run, and the checks before a new one ---------------------------------------

def artifact_files(run_id: str) -> list[Path]:
    """Every file a run owns: its output directory (not another run's inside it), its freeze and its report."""
    paths = run_paths(run_id)
    other_runs = [p for p in (REPORTS / "stage3").glob("run[0-9]*") if p.is_dir()] if run_id == RUN1 else []
    files = [f for f in sorted(paths["out"].rglob("*")) if f.is_file() and not any(o in f.parents for o in other_runs)]
    return files + [paths[k] for k in ("freeze", "comparison_json", "comparison_md") if paths[k].exists()]


def seal_path(run_id: str) -> Path:
    return REPORTS / f"{run_id}_artifact_hashes.json"


def cmd_seal(args) -> int:
    """Record the hash of every file of a finished run, so a later run can prove it left them alone."""
    refuse_existing(seal_path(RUN_ID))
    files = artifact_files(RUN_ID)
    if not COMPARISON_JSON.exists():
        raise SystemExit(f"refusing: run {RUN_ID!r} has no finished report to seal")
    seal_path(RUN_ID).write_text(json.dumps({"run_id": RUN_ID, "sealed": time.strftime("%Y-%m-%d %H:%M:%S"),
                                             "files": {str(f.relative_to(REPORTS)): sha256(f) for f in files}}, indent=1))
    seal_path(RUN_ID).chmod(0o400)
    print(f"sealed {len(files)} file(s) of run {RUN_ID!r} in {seal_path(RUN_ID).name}")
    return 0


def other_runs_intact() -> list[dict]:
    """Each other sealed run, checked file by file against its seal."""
    out = []
    for seal in sorted(REPORTS.glob("stage3*_artifact_hashes.json")):
        record = json.loads(seal.read_text())
        if record["run_id"] == RUN_ID:
            continue
        now = {str(f.relative_to(REPORTS)): sha256(f) for f in artifact_files(record["run_id"])}
        out.append({"run_id": record["run_id"], "files": len(record["files"]),
                    "changed": sorted(f for f, h in record["files"].items() if f in now and now[f] != h),
                    "missing": sorted(set(record["files"]) - set(now)), "added": sorted(set(now) - set(record["files"]))})
    return out


def completed_results() -> list[str]:
    """Held-out and final results this run already holds. Development outputs are not listed."""
    found = [*OUT.glob("ask_heldout_*.json"), *OUT.glob("grades_heldout*.json"), *OUT.glob("holdout75_*/scorecard-*.json")]
    return sorted(str(f.relative_to(REPORTS)) for f in [*found, *(p for p in (COMPARISON_JSON, COMPARISON_MD) if p.exists())])


def preflight(expect_build: str | None, live: bool = True) -> dict:
    """Everything that must hold before a run's held-out material is touched. Reads the id
    file and the manifests; of the question file only that it exists, and its hash."""
    from src.retrieval.index_notes import read_subject_ids
    questions, subjects = heldout_files()
    research = json.loads((REPORTS / "eval_manifest_research_candidate.json").read_text())
    ids = read_subject_ids("", str(subjects))
    build = systems()["research_candidate"]["build"]
    checks = {
        "run_id": RUN_ID, "paths": {"out": str(OUT.relative_to(REPORTS)), "freeze": FREEZE.name,
                                    "comparison": [COMPARISON_JSON.name, COMPARISON_MD.name]},
        "candidate_build": build, "build_is_expected": expect_build is None or build == expect_build,
        "heldout_subjects": len(ids), "heldout_subjects_in_manifest_build": len(set(ids) & set(research["patient_ids"])),
        "heldout_subjects_sha256": sha256(subjects),
        "heldout_questions_present": questions.exists(), "heldout_questions_sha256": sha256(questions) if questions.exists() else None,
        "freeze_exists": FREEZE.exists(), "completed_results": completed_results(), "other_runs": other_runs_intact(),
        "unsealed_earlier_run": RUN_ID != RUN1 and run_paths(RUN1)["comparison_json"].exists() and not seal_path(RUN1).exists(),
    }
    if live:
        from sqlalchemy import create_engine, text
        from sqlalchemy.engine import make_url
        from src import storage
        engine = create_engine(make_url(storage.DATABASE_URL).set(database=systems()["research_candidate"]["database"]),
                               connect_args={"options": "-c default_transaction_read_only=on"})
        with engine.connect() as c:
            checks["heldout_subjects_in_database_build"] = c.execute(text(
                "SELECT count(DISTINCT subject_id) FROM note_chunks_v2 WHERE build_id = :b AND subject_id = ANY(:i)"), {"b": build, "i": ids}).scalar()
            checks["build_status"] = c.execute(text("SELECT status FROM note_index_runs WHERE run_id = :b"), {"b": build}).scalar()
        engine.dispose()
    checks["ok"] = bool(
        checks["build_is_expected"] and checks["heldout_subjects"] and checks["heldout_questions_present"]
        and checks["heldout_subjects_in_manifest_build"] == checks["heldout_subjects"]
        and (not live or (checks["heldout_subjects_in_database_build"] == checks["heldout_subjects"] and checks["build_status"] == "completed"))
        and not checks["completed_results"] and not checks["unsealed_earlier_run"]
        and all(not (r["changed"] or r["missing"] or r["added"]) for r in checks["other_runs"]))
    return checks


def cmd_preflight(args) -> int:
    checks = preflight(args.expect_build)
    for key, value in checks.items():
        print(f"{key:38s} {value}")
    return 0 if checks["ok"] else 2


# --- the freeze ------------------------------------------------------------------------------------

GATE = {"regression_max_correct_lost": 1, "regression_sets": ["retrieval_42", "temporal_11", "holdout_75"],
        "targeted_min_correct_gained": 3, "safety": "wrong answers not sent to review do not rise, by count or rate; "
                                                    "unsupported auto-approved claims do not rise",
        "latency_max_median_rise_percent": 25, "latency_sets": ["retrieval_42", "holdout_75", "targeted_heldout"],
        "final": "PASS only if every condition passes; a set that was not run is never a pass"}
FROZEN_FILES = ("scripts/stage3_compare.py", "scripts/stage3_report.py", "scripts/scorecard.py", "scripts/structured_parity.py",
                "scripts/retrieval_eval.py", "src/evals/scorecard.py", "src/evals/retrieval_metrics.py", "src/agents/admission_scope.py",
                "src/agents/classify.py", "src/agents/graph.py", "src/agents/prompts.py", "src/retrieval/hybrid_retriever_v2.py",
                "src/generation/structured_lookup.py", "src/generation/lab_query.py", "src/storage/readiness.py", "src/config.py")


def build_freeze(expect_build: str | None = None, live: bool = True) -> dict:
    """What a run is evaluated with, written down before any held-out material is run."""
    from src.agents import admission_scope as adm, prompts
    from src.config import PROFILES
    from src.llm import local_client
    from src.storage import readiness as rd
    checks = preflight(expect_build, live)
    if not checks["ok"]:
        raise SystemExit("refusing: the preflight does not pass; run `preflight` to see why")
    retrieval = {name: OUT / f"retrieval_{name}.json" for name in ("research_control", "research_candidate")}
    if not all(p.exists() for p in retrieval.values()):
        raise SystemExit("refusing: run the 42-question development set for this run before freezing it")
    questions, subjects = heldout_files()
    research = json.loads((REPORTS / "eval_manifest_research_candidate.json").read_text())
    shape = validate_heldout(questions, subjects, research["patient_ids"])
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout      # noqa: E731
    untracked = [f for f in git("ls-files", "--others", "--exclude-standard").split() if f]
    dirty = git("diff", "HEAD") + "".join(f"{f}:{sha256(ROOT / f)}\n" for f in sorted(untracked))
    record = {
        "name": "Stage 3 evaluation freeze", "run_id": RUN_ID, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "rule": "After this file is written no behaviour or evaluator change is made in response to held-out results.",
        "code": {"branch": git("branch", "--show-current").strip(), "commit": git("rev-parse", "HEAD").strip(),
                 "working_tree_clean": not dirty, "dirty_tree_sha256": hashlib.sha256(dirty.encode()).hexdigest() if dirty else None,
                 "file_sha256": {f: sha256(ROOT / f) for f in FROZEN_FILES}},
        "systems": systems(),
        "builds": {"research_candidate": checks["candidate_build"], "holdout_candidate": HOLDOUT_BUILD,
                   "research_candidate_patients": research["patient_ids"], "chunk_configuration": research["v2_build"]["configuration"]},
        "models": {"main": local_client.MAIN_MODEL, "fast": local_client.FAST_MODEL,
                   "embedding": research["v2_build"]["configuration"].get("embedding"),
                   "reranker": research["retrieval_configuration"].get("reranker"),
                   "query_encoder": research["retrieval_configuration"].get("query_encoder")},
        "retrieval_configuration": research["retrieval_configuration"],
        "query_expansion": "off (LUMEN_QUERY_EXPANSION=0 in every system)",
        "e10_admission_resolver": {
            "frozen": True, "max_candidates": adm.MAX_CANDIDATES, "lines_shown": adm.LINES_SHOWN,
            "generic_terms_sha256": hashlib.sha256(" ".join(sorted(adm._GENERIC)).encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompts.ADMISSION_SYSTEM.encode()).hexdigest(),
            "acceptance": "candidate id; quote in that admission's evidence; quote shares a distinctive term with the phrase; quote in no other candidate"},
        "structured_readiness": {
            "tables": list(rd.STRUCTURED_TABLES), "lab_table_by_profile": {name: p["lab_table"] for name, p in PROFILES.items()},
            "check": "every lab and structured read: one statement for load status, recorded counts, non-empty tables and the lab "
                     "index; a second only to name missing tables; nothing cached",
            "admission_scoped_lab": "a resolved admission constrains the lab SQL by hadm_id; modes latest, earliest, trend, lowest, highest"},
        "evaluators": {
            "retrieval_42_and_temporal_11": {
                "correct": "42-set: Hit@5; temporal subset: target Hit@5", "control_labels": "the frozen benchmark labels, unchanged",
                "candidate_labels": "the benchmark's section rule applied to the build's chunks by their stored source offsets; no 40-token minimum",
                "why_no_minimum": "the 40-token minimum described what the control index could return; it was a property of that "
                                  "representation, not of the ground truth",
                "results_used_by_the_gate": {name: sha256(p) for name, p in retrieval.items()}},
            "holdout_75": {
                "scorer": "scripts/scorecard.py --profile holdout; cases unchanged", "correct": "no hard check failed and no violation",
                "primary_lab_truth": {"table": LAB_TRUTH_TABLE, "database": LAB_TRUTH_DATABASE, "applies_to": "both systems, the 20 creatinine cases"},
                "legacy_capped_lab_score": "the same verdict with the capped labevents table as the lab truth; history only, never in PASS/FAIL",
                "holdout_manifest_sha256": sha256(HOLDOUT_MANIFEST) if HOLDOUT_MANIFEST.exists() else None},
            "targeted": {"scorer": "manual grade against the written ground truth, blind to the system (A/B shuffled per question, citation labels masked)",
                         "judge": "Claude", "rubric": "correct = states the ground truth; partial = right core fact with a wrong or contradictory "
                                                      "element; incorrect = anything else, including no answer",
                         "wrong_not_sent_to_review": "incorrect or partial, and released without human review"}},
        "heldout": {"questions_file": str(questions.relative_to(ROOT)) if questions.is_relative_to(ROOT) else str(questions),
                    "questions_sha256": sha256(questions),
                    "ids_file": str(subjects.relative_to(ROOT)) if subjects.is_relative_to(ROOT) else str(subjects),
                    "ids_sha256": sha256(subjects), **shape, "results_generated_before_freeze": checks["completed_results"]},
        "gate": GATE,
        "manifests": {f: sha256(REPORTS / f) for f in ("eval_manifest_research_candidate.json", "eval_manifest_holdout_control.json",
                                                         "eval_manifest_holdout_candidate.json")},
        "development_outputs_before_freeze": {str(f.relative_to(REPORTS)): sha256(f) for f in sorted(OUT.glob("*.json"))},
        "other_runs_intact": checks["other_runs"],
    }
    if live:
        try:
            tags = json.loads(_OPENER.open("http://127.0.0.1:11434/api/tags", timeout=10).read())["models"]
            record["models"]["ollama_digests"] = {m["name"]: m["digest"] for m in tags}
        except Exception as e:
            raise SystemExit(f"refusing: cannot read the local model digests ({type(e).__name__}); start Ollama first")
        state = holdout_state()
        if not state["source_tables_identical"]:
            raise SystemExit("refusing: lumen_holdout no longer matches its frozen fingerprint")
        frozen_sources = fingerprint_id(fingerprint("lumen_holdout", SOURCE_TABLES))
        record["databases"] = {"lumen_holdout": {"role": "frozen original; never run against", "fingerprint": state["fingerprint"]},
                               **{db: {"source_tables_match_frozen_holdout": fingerprint_id(fingerprint(db, SOURCE_TABLES)) == frozen_sources}
                                  for db in ("lumen_holdout_control_eval", "lumen_holdout_v2_eval")}}
    return record


def cmd_freeze(args) -> int:
    refuse_existing(FREEZE)
    record = build_freeze(args.expect_build)
    FREEZE.write_text(json.dumps(record, indent=1, default=str))
    FREEZE.chmod(0o400)
    print(f"freeze written: {FREEZE.name} sha256 {sha256(FREEZE)[:16]} build {record['builds']['research_candidate']} "
          f"questions {record['heldout']['questions']} patients {record['heldout']['patients']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", default=RUN1, help=f"{RUN1!r} (the first run, default) or 'stage3_runN'")
    p.add_argument("--heldout-questions", help="this run's targeted held-out questions; required for any run but the first")
    p.add_argument("--heldout-subjects", help="this run's held-out patient ids, one per line")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--require-frozen", action="store_true", help="fail unless lumen_holdout matches its frozen fingerprint")
    for name in ("preflight", "freeze"):
        sub.add_parser(name).add_argument("--expect-build", help="fail unless the research candidate is this build")
    sub.add_parser("seal")
    sub.add_parser("retrieval")
    sub.add_parser("_retrieval").add_argument("out")
    sub.add_parser("_scope").add_argument("out_dir")
    for name in ("ask", "sheet"):
        sub.add_parser(name).add_argument("--set", choices=["dev15", "heldout"], required=True)
    sub.add_parser("holdout75")
    sub.add_parser("report")
    args = p.parse_args(argv)
    if args.command == "_retrieval":
        return retrieval_child(args.out)
    if args.command == "_scope":
        return scope_child(args.out_dir)
    use_run(args.run_id, args.heldout_questions, args.heldout_subjects)
    if args.command == "report":
        import stage3_report
        return stage3_report.main()
    return {"check": cmd_check, "preflight": cmd_preflight, "freeze": cmd_freeze, "seal": cmd_seal, "retrieval": cmd_retrieval,
            "ask": cmd_ask, "holdout75": cmd_holdout75, "sheet": cmd_sheet}[args.command](args)


if __name__ == "__main__":
    import stage3_compare                       # one module object, so the report sees the run chosen here
    raise SystemExit(stage3_compare.main())
