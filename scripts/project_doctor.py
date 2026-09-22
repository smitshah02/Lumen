"""Read-only operational readiness report for Lumen.

This command starts nothing, downloads nothing, creates no schema, and writes
no artifacts. It distinguishes configuration/dependency, schema, data, model,
and service failures so an operator knows which explicit lifecycle target is
needed next.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # noqa: F401,E402  (loads .env without overriding exported values)
from src import config  # noqa: E402


@dataclass
class Check:
    group: str
    name: str
    status: str
    detail: str


def add(checks: list[Check], group: str, name: str, ok: bool, detail: str,
        *, advisory: bool = False) -> None:
    checks.append(Check(group, name, "PASS" if ok else ("WARN" if advisory else "FAIL"), detail))


def check_runtime(checks: list[Check]) -> None:
    actual = sys.version_info[:2]
    add(checks, "runtime", "python", actual == (3, 12),
        f"{platform.python_version()} (required 3.12)")
    packages = {
        "torch": "torch", "transformers": "transformers", "sqlalchemy": "sqlalchemy",
        "psycopg": "psycopg", "langgraph": "langgraph", "fastapi": "fastapi",
        "pydantic": "pydantic", "mcp": "mcp", "spacy": "spacy",
    }
    missing = []
    for label, module in packages.items():
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append(f"{label}({type(exc).__name__})")
    add(checks, "runtime", "python_dependencies", not missing,
        "all required imports available" if not missing else "missing: " + ", ".join(missing))
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        detail = f"torch={torch.__version__} cuda={torch.version.cuda or 'none'} available={cuda}"
        add(checks, "runtime", "accelerator", cuda, detail, advisory=True)
    except Exception as exc:
        add(checks, "runtime", "accelerator", False, type(exc).__name__, advisory=True)


def check_configuration(checks: list[Check]) -> None:
    add(checks, "configuration", "data_plane", True, config.DATA_PLANE)
    try:
        from src.llm import local_client
        add(checks, "configuration", "model_endpoint_policy", True,
            f"accepted host={local_client.runtime_config()['host']}")
    except Exception as exc:
        add(checks, "configuration", "model_endpoint_policy", False,
            f"{type(exc).__name__}: {exc}")
    if config.DATA_PLANE == "research":
        required = ("LUMEN_PG_VOLUME", "LUMEN_PG_PASSWORD")
        missing = [name for name in required if not os.environ.get(name)]
        add(checks, "configuration", "research_compose_selection", not missing,
            "explicit volume and credential selected" if not missing else "missing: " + ", ".join(missing))
    else:
        add(checks, "configuration", "research_compose_selection", True,
            "not applicable to demo plane")
    add(checks, "configuration", "model_registry", config.MODELS_CONFIG.get("schema_version") == 1,
        str(config.MODELS_CONFIG_PATH))


def check_database(checks: list[Check]) -> None:
    try:
        from sqlalchemy import text
        from src import storage
        from src.storage.schema import SCHEMA_VERSION
        from src.retrieval.index_provenance import configuration_hash
        with storage.engine.connect() as conn:
            database = conn.execute(text("SELECT current_database()" )).scalar()
            expected_database = (storage.DEMO_DB_NAME if storage.DATA_PLANE == "demo"
                                 else storage.RESEARCH_DB_NAME)
            extension = conn.execute(text(
                "SELECT extversion FROM pg_extension WHERE extname='vector'"
            )).scalar()
            wanted_tables = {
                "patients", "admissions", "labevents", "clinical_notes", "note_chunks",
                "d_labitems", "guideline_chunks", "lumen_schema_version", "ingestion_log",
                "ingestion_runs", "note_index_runs", "note_index_state",
            }
            tables = set(conn.execute(text(
                "SELECT tablename FROM pg_tables WHERE schemaname=current_schema()"
            )).scalars())
            missing_tables = sorted(wanted_tables - tables)
            wanted_indexes = {"idx_chunks_subject", "idx_chunks_fts", "idx_chunks_embedding",
                              "idx_chunks_note_position_unique"}
            indexes = set(conn.execute(text(
                "SELECT indexname FROM pg_indexes WHERE schemaname=current_schema()"
            )).scalars())
            missing_indexes = sorted(wanted_indexes - indexes)
            version = (conn.execute(text(
                "SELECT COALESCE(MAX(version),0) FROM lumen_schema_version"
            )).scalar() if "lumen_schema_version" in tables else 0)
            counts = {}
            for table in ("clinical_notes", "note_chunks", "labevents", "d_labitems"):
                counts[table] = (conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                                 if table in tables else None)
            eligible_notes = indexed_notes = None
            ingestion_status = None
            if not missing_tables:
                eligible_notes = conn.execute(text("""
                    SELECT COUNT(*) FROM clinical_notes
                    WHERE COALESCE(text_deid, text_original) IS NOT NULL
                      AND COALESCE(text_deid, text_original) != ''
                """)).scalar()
                indexed_notes = conn.execute(text("""
                    SELECT COUNT(*) FROM note_index_state nis
                    WHERE nis.status='completed' AND nis.config_hash=:config_hash
                      AND nis.chunk_count=(
                          SELECT COUNT(*) FROM note_chunks nc WHERE nc.note_id=nis.note_id
                      )
                """), {"config_hash": configuration_hash()}).scalar()
                if storage.DATA_PLANE == "research":
                    ingestion_status = conn.execute(text("""
                        SELECT status FROM ingestion_runs ORDER BY started_at DESC LIMIT 1
                    """)).scalar()
                    if ingestion_status is None:
                        legacy_completed = conn.execute(text("""
                            SELECT COUNT(*) FROM (
                                SELECT DISTINCT ON (table_name) table_name, status
                                FROM ingestion_log
                                WHERE table_name IN ('patients', 'admissions', 'clinical_notes')
                                ORDER BY table_name, id DESC
                            ) latest WHERE status='completed'
                        """)).scalar()
                        if legacy_completed == 3:
                            ingestion_status = "legacy_completed"
        add(checks, "database", "connectivity", True, f"connected database={database}")
        add(checks, "database", "plane_database", database == expected_database,
            f"plane={storage.DATA_PLANE} actual={database} expected={expected_database}")
        add(checks, "database", "pgvector", extension == config.PGVECTOR_VERSION,
            f"installed={extension or 'missing'} expected={config.PGVECTOR_VERSION}")
        add(checks, "schema", "version", version == SCHEMA_VERSION,
            f"installed={version} expected={SCHEMA_VERSION}")
        add(checks, "schema", "tables", not missing_tables,
            "complete" if not missing_tables else "missing: " + ", ".join(missing_tables))
        add(checks, "schema", "indexes", not missing_indexes,
            "complete" if not missing_indexes else "missing: " + ", ".join(missing_indexes))
        empty = [name for name, count in counts.items() if count in (None, 0)]
        add(checks, "data", "runtime_corpus", not empty,
            " ".join(f"{name}={count}" for name, count in counts.items()) if counts else "unavailable")
        ingestion_ok = (storage.DATA_PLANE == "demo" or
                        ingestion_status in ("completed", "legacy_completed"))
        add(checks, "data", "ingestion_complete", ingestion_ok,
            "synthetic loader verified separately" if storage.DATA_PLANE == "demo" else
            f"latest={ingestion_status or 'untracked'}")
        index_ok = bool(eligible_notes) and indexed_notes == eligible_notes
        add(checks, "data", "index_provenance", index_ok,
            f"current={indexed_notes} eligible={eligible_notes}")
    except Exception as exc:
        add(checks, "database", "connectivity", False,
            f"{type(exc).__name__}: database unavailable (credentials are not shown)")
        for group, name in (("database", "plane_database"), ("database", "pgvector"), ("schema", "version"),
                            ("schema", "tables"), ("schema", "indexes"),
                            ("data", "runtime_corpus"), ("data", "ingestion_complete"),
                            ("data", "index_provenance")):
            add(checks, group, name, False, "not checked because database is unavailable")


def check_models(checks: list[Check]) -> None:
    try:
        from scripts import fetch_models
        for name, spec in fetch_models.selected_models("runtime").items():
            errors = fetch_models.verify_local_model(name, spec, config.MODELS_DIR / name)
            add(checks, "models", name, not errors,
                "provenance and hashes verified" if not errors else "; ".join(errors))
    except Exception as exc:
        add(checks, "models", "retrieval_weights", False, f"{type(exc).__name__}: {exc}")
    try:
        import requests
        from src.llm import local_client
        response = requests.get(f"{local_client.HOST}/api/tags", timeout=3)
        response.raise_for_status()
        tags = {item.get("name", "").removesuffix(":latest")
                for item in response.json().get("models", [])}
        for tier, model in (("main", local_client.MAIN_MODEL), ("fast", local_client.FAST_MODEL)):
            normalized = model.removesuffix(":latest")
            add(checks, "models", f"ollama_{tier}", normalized in tags,
                f"{model} {'installed' if normalized in tags else 'missing'}")
    except Exception as exc:
        add(checks, "models", "ollama_service", False, f"{type(exc).__name__}: unreachable")


def run() -> list[Check]:
    checks: list[Check] = []
    check_runtime(checks)
    check_configuration(checks)
    check_database(checks)
    check_models(checks)
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    checks = run()
    if args.json:
        print(json.dumps({"ready": not any(c.status == "FAIL" for c in checks),
                          "checks": [asdict(c) for c in checks]}, indent=2))
    else:
        for group in dict.fromkeys(c.group for c in checks):
            print(f"\n[{group}]")
            for check in (c for c in checks if c.group == group):
                print(f"  {check.status:<4} {check.name:<28} {check.detail}")
        failures = [c for c in checks if c.status == "FAIL"]
        print(f"\n{'READY' if not failures else 'NOT READY'} — {len(failures)} failure(s)")
    return 1 if any(c.status == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
