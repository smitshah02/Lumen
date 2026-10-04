"""Unseen-patient holdout: select, ingest and evaluate patients outside the indexed cohort.

The working research database is never written to. The holdout lives in its
own database (`lumen_holdout`) on the same local Postgres server, built with
the unchanged ingestion, de-identification, chunking, embedding and indexing
code. Nothing here tunes anything.

    ./scripts/lumen research holdout select              # freeze a seeded manifest (once)
    ./scripts/lumen research holdout create-db
    ./scripts/lumen research holdout ingest [--limit 3]  # slow: scans the raw CSVs
    ./scripts/lumen research holdout status
    ./scripts/lumen research holdout start               # the API, on the holdout database
    ./scripts/lumen research holdout scorecard           # the scorecard, on the holdout database

Selection uses the SAME eligibility rules as the original cohort
(src.storage.ingest.select_cohort defaults), removes every patient already in
the research database, and draws a seeded random sample. It looks at data
availability only and runs before any answer is generated. The manifest is
written once and never overwritten.

Subject ids and all artifacts stay under ~/Lumen_local_results/holdout.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HOLDOUT_DB = "lumen_holdout"
OUT_DIR = Path.home() / "Lumen_local_results" / "holdout"
MANIFEST = OUT_DIR / "manifest.json"
ACTIVE = "LUMEN_HOLDOUT_ACTIVE"          # set in child processes that are bound to the holdout database
SEED = 20261004


def _use_local_raw_dirs() -> None:
    """Point the ingestion at the raw MIMIC files that are actually on disk.
    A configured MIMIC_IV_DIR / MIMIC_IV_NOTE_DIR that does not exist is
    replaced, for this process only, by the repository default."""
    try:
        from dotenv import dotenv_values
        configured = dotenv_values(ROOT / ".env")
    except Exception:
        configured = {}
    for var, default in (("MIMIC_IV_DIR", ROOT / "data" / "mimiciv"), ("MIMIC_IV_NOTE_DIR", ROOT / "data" / "mimic-iv-note")):
        value = os.environ.get(var) or configured.get(var)
        if value and not (ROOT / value).expanduser().exists() and default.exists():
            print(f"note: {var} is configured to a directory that does not exist; using {default.relative_to(ROOT)}")
            os.environ[var] = str(default)


def _research():
    """The research engine. Refuses when this process is already bound to the holdout."""
    from sqlalchemy import text
    from src import storage
    if storage.DATA_PLANE != "research":
        sys.exit(f"refusing: holdout commands run on the research plane, not {storage.DATA_PLANE!r}")
    if storage.engine.url.database == HOLDOUT_DB:
        sys.exit("refusing: DATABASE_URL already points at the holdout database")
    return storage, text


def _holdout_env() -> dict:
    """Environment for a child process whose database is the holdout. The URL is
    passed in the environment only; it is never printed."""
    from sqlalchemy.engine import make_url
    storage, _ = _research()
    url = make_url(storage.DATABASE_URL).set(database=HOLDOUT_DB).render_as_string(hide_password=False)
    return {**os.environ, "DATABASE_URL": url, ACTIVE: "1", "LUMEN_DATA_PLANE": "research"}


def _in_holdout():
    """The holdout engine, for code running in a child started with _holdout_env()."""
    from sqlalchemy import text
    from src import storage
    if os.environ.get(ACTIVE) != "1" or storage.engine.url.database != HOLDOUT_DB:
        sys.exit("refusing: this step must run bound to the holdout database")
    return storage, text


def _manifest() -> dict:
    if not MANIFEST.exists():
        sys.exit(f"no manifest at {MANIFEST}; run `holdout select` first")
    return json.loads(MANIFEST.read_text())


def _run(args: list[str], env: dict) -> None:
    subprocess.run(args, cwd=ROOT, env=env, check=True)


# ---------------------------------------------------------------- select ----
def select(n: int, seed: int) -> int:
    if MANIFEST.exists():
        sys.exit(f"refusing: {MANIFEST} already exists. The holdout is frozen once selected.")
    storage, text = _research()
    from src.storage import ingest
    with storage.engine.connect() as c:
        current = {int(s) for s in c.execute(text("SELECT subject_id FROM patients")).scalars()}
    eligible = ingest.select_cohort(limit=10 ** 9)       # the original cohort's rules, unchanged
    unseen = sorted(eligible - current)
    if len(unseen) < n:
        sys.exit(f"only {len(unseen)} eligible unseen patients exist locally; cannot draw {n}")
    chosen = sorted(random.Random(seed).sample(unseen, n))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.chmod(0o700)
    MANIFEST.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed, "n": n,
        "eligibility": "src.storage.ingest.select_cohort defaults: adult, >=3 admissions, a discharge "
                       "summary, a guideline-relevant diagnosis (identical to the indexed cohort)",
        "selection": "random.Random(seed).sample(sorted(eligible - current_cohort), n); no answer was "
                     "generated for any of these patients before this file was written",
        "eligible_in_raw_mimic": len(eligible), "current_cohort": len(current),
        "current_cohort_within_eligible": len(eligible & current), "unseen_pool": len(unseen),
        "overlap_with_current_cohort": len(set(chosen) & current), "subject_ids": chosen}, indent=1))
    MANIFEST.chmod(0o600)
    print(f"eligible in raw MIMIC: {len(eligible)} | current cohort: {len(current)} | unseen pool: {len(unseen)}")
    print(f"selected {n} with seed {seed}; overlap with current cohort: {len(set(chosen) & current)}")
    print(f"manifest frozen at {MANIFEST}")
    return 0


# ------------------------------------------------------------- create-db ----
def create_db() -> int:
    storage, text = _research()
    with storage.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as c:
        if c.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": HOLDOUT_DB}).scalar():
            print(f"database {HOLDOUT_DB} already exists")
        else:
            c.execute(text(f'CREATE DATABASE "{HOLDOUT_DB}"'))
            print(f"created database {HOLDOUT_DB} (the research database was not touched)")
    env = _holdout_env()
    for module, extra in (("src.storage.schema", []), ("src.storage.schema", ["--upgrade"]),
                          ("src.storage.checkpoints", []), ("src.storage.load_d_labitems", [])):
        _run([sys.executable, "-m", module, *extra], env)
    return 0


# ---------------------------------------------------------------- ingest ----
def ingest(limit: int | None) -> int:
    if os.environ.get(ACTIVE) != "1":                    # re-run this step bound to the holdout database
        args = [sys.executable, __file__, "ingest"] + (["--limit", str(limit)] if limit else [])
        _run(args, _holdout_env())
        return 0
    storage, text = _in_holdout()
    from src.storage import ingest as ing
    manifest = _manifest()
    wanted = manifest["subject_ids"][:limit] if limit else manifest["subject_ids"]
    with storage.engine.connect() as c:
        have = {int(s) for s in c.execute(text("SELECT DISTINCT subject_id FROM clinical_notes")).scalars()}
    todo = set(wanted) - have
    print(f"holdout ingest: {len(wanted)} requested, {len(have & set(wanted))} already loaded, {len(todo)} to load")
    t0 = time.time()
    if todo:
        run_id = str(uuid.uuid4())
        ing._start_ingestion_run(run_id, {"operation": "holdout", "seed": manifest["seed"], "patients": len(todo),
                                          "max_labs_per_patient": 200, "skip_deid": False})
        try:
            ing.load_patients(todo)
            ing.load_admissions(todo)
            ing.load_diagnoses(todo)
            ing.load_labevents(todo, max_per_patient=200)
            ing.load_prescriptions(todo)
            ing.load_procedures(todo)
            ing.load_clinical_notes(todo, run_deid=True)
        except BaseException as exc:
            ing._finish_ingestion_run(run_id, "failed", type(exc).__name__)
            raise
        ing._finish_ingestion_run(run_id, "completed")
    t1 = time.time()
    _run([sys.executable, "-m", "src.retrieval.index_notes", "--cooldown", "0"], dict(os.environ))
    print(f"ingest {t1 - t0:.0f}s, index {time.time() - t1:.0f}s")
    return status()


# ---------------------------------------------------------------- status ----
def status() -> int:
    if os.environ.get(ACTIVE) != "1":
        storage, text = _research()
        with storage.engine.connect() as c:
            current = {int(s) for s in c.execute(text("SELECT subject_id FROM patients")).scalars()}
        ids = set(_manifest()["subject_ids"])
        print(f"manifest: {len(ids)} patients, seed {_manifest()['seed']} | research cohort: {len(current)} | "
              f"overlap: {len(ids & current)}")
        _run([sys.executable, __file__, "status"], _holdout_env())
        return 1 if ids & current else 0
    storage, text = _in_holdout()
    ids = _manifest()["subject_ids"]
    with storage.engine.connect() as c:
        def one(sql):
            return c.execute(text(sql), {"ids": ids}).scalar()
        rows = {
            "patients": one("SELECT count(*) FROM patients"),
            "patients outside the manifest": one("SELECT count(*) FROM patients WHERE NOT (subject_id = ANY(:ids))"),
            "admissions": one("SELECT count(*) FROM admissions"), "labevents": one("SELECT count(*) FROM labevents"),
            "notes": one("SELECT count(*) FROM clinical_notes"),
            "notes de-identified": one("SELECT count(text_deid) FROM clinical_notes"),
            "chunks": one("SELECT count(*) FROM note_chunks"),
            "chunks with embedding": one("SELECT count(embedding) FROM note_chunks"),
            "chunks of patients outside the manifest":
                one("SELECT count(*) FROM note_chunks WHERE NOT (subject_id = ANY(:ids))"),
            "patients with notes": one("SELECT count(DISTINCT subject_id) FROM clinical_notes"),
            "patients with labs": one("SELECT count(DISTINCT subject_id) FROM labevents"),
        }
    print(f"database {storage.engine.url.database}: " + " | ".join(f"{k} {v}" for k, v in rows.items()))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--seed", type=int, default=SEED)
    sub.add_parser("create-db")
    i = sub.add_parser("ingest")
    i.add_argument("--limit", type=int, default=None, help="load only the first N manifest patients (pilot)")
    sub.add_parser("status")
    sub.add_parser("start", help="start the API bound to the holdout database")
    sub.add_parser("scorecard", help="run the scorecard against the holdout database (extra flags are passed on)")
    args, rest = p.parse_known_args(argv)
    if rest and args.command != "scorecard":
        p.error(f"unrecognized arguments: {' '.join(rest)}")
    _use_local_raw_dirs()

    if args.command == "select":
        return select(args.n, args.seed)
    if args.command == "create-db":
        return create_db()
    if args.command == "ingest":
        return ingest(args.limit)
    if args.command == "status":
        return status()
    if args.command == "start":
        os.execvpe(str(ROOT / "scripts" / "lumen"), [str(ROOT / "scripts" / "lumen"), "research", "start"], _holdout_env())
    if args.command == "scorecard":
        _manifest()
        os.execvpe(sys.executable, [sys.executable, str(ROOT / "scripts" / "scorecard.py"), "--profile", "holdout",
                                    "--subjects-file", str(MANIFEST), "--out-dir", str(OUT_DIR / "scorecard"),
                                    *rest], _holdout_env())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
