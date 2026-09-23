"""Validate and load a frozen Synthea CSV cohort into the canonical schema.

This is deliberately a thin source adapter. It creates no notes or indexes and
refuses every plane/database except the isolated Synthea target.

    LUMEN_DATA_PLANE=synthea python -m src.storage.ingest_synthea --profile dev
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
SUBJECT_BASE = 80_000_000
HADM_BASE = 800_000_000
ITEM_BASE = 980_000
LABEVENT_BASE = 8_000_000_000
PHARMACY_BASE = 8_100_000_000
NOTE_BASE = 850_000_000
MAPPING_VERSION = "synthea-sorted-v1"
NOTE_RENDERER_VERSION = "synthea-encounter-summary-v1"
OBSERVATION_NAMESPACE = "synthea-v4-csv-observation"

REQUIRED_COLUMNS = {
    "patients.csv": {"Id", "BIRTHDATE", "DEATHDATE", "GENDER"},
    "encounters.csv": {
        "Id", "PATIENT", "START", "STOP", "ENCOUNTERCLASS", "CODE",
        "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION",
    },
    "observations.csv": {
        "DATE", "PATIENT", "ENCOUNTER", "CATEGORY", "CODE",
        "DESCRIPTION", "VALUE", "UNITS", "TYPE",
    },
    "conditions.csv": {"START", "STOP", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "DESCRIPTION"},
    "medications.csv": {
        "START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION",
        "REASONDESCRIPTION",
    },
    "procedures.csv": {
        "START", "STOP", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE",
        "DESCRIPTION", "REASONDESCRIPTION",
    },
}
OBSERVATION_DUPLICATE_FIELDS = (
    "DATE", "PATIENT", "ENCOUNTER", "CATEGORY", "CODE",
    "DESCRIPTION", "VALUE", "UNITS", "TYPE",
)


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    profile: str
    manifest: dict
    manifest_sha256: str
    rows: dict[str, list[dict]]


@dataclass(frozen=True)
class Mappings:
    patients: dict[str, int]
    encounters: dict[str, int]
    items: dict[str, int]
    item_metadata: dict[str, tuple[str, str | None]]
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class PreparedData:
    patients: list[dict]
    admissions: list[dict]
    labitems: list[dict]
    labevents: list[dict]
    diagnoses: list[dict]
    prescriptions: list[dict]
    procedures: list[dict]
    warnings: dict[str, int]

    def counts(self) -> dict[str, int]:
        return {
            "patients": len(self.patients),
            "admissions": len(self.admissions),
            "d_labitems": len(self.labitems),
            "labevents": len(self.labevents),
            "diagnoses_icd": len(self.diagnoses),
            "prescriptions": len(self.prescriptions),
            "procedures_icd": len(self.procedures),
        }


@dataclass(frozen=True)
class PreparedNotes:
    rows: list[dict]
    section_counts: dict[str, int]
    length_stats: dict[str, int | float]
    corpus_sha256: str


def _note_corpus_hash(rows: Iterable[tuple[int, str]]) -> tuple[str, int]:
    """Hash the canonical note payload without materializing a second corpus."""
    digest = hashlib.sha256()
    digest.update(b"[")
    count = 0
    for note_id, note_text in rows:
        if count:
            digest.update(b",")
        digest.update(json.dumps(
            {"note_id": note_id, "text": note_text},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"))
        count += 1
    digest.update(b"]")
    return digest.hexdigest(), count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = []
        for source_row, row in enumerate(reader, 1):
            rows.append({**row, "_source_row": source_row})
    return columns, rows


def validate_source(root: Path, expected_profile: str) -> SourceDataset:
    root = root.resolve()
    manifest_path = root / "generation_manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Synthea generation manifest {manifest_path}: {exc}") from exc
    if manifest.get("synthetic") is not True:
        raise RuntimeError("source manifest is not marked synthetic")
    if manifest.get("profile") != expected_profile:
        raise RuntimeError(
            f"source profile {manifest.get('profile')!r} does not match {expected_profile!r}"
        )
    synthea = manifest.get("synthea") or {}
    if synthea.get("project") != "synthetichealth/synthea" or not synthea.get("version"):
        raise RuntimeError("manifest is not an identified official Synthea source")
    recorded_files = manifest.get("csv_files")
    if not isinstance(recorded_files, dict):
        raise RuntimeError("manifest has no csv_files mapping")

    loaded = {}
    for name, required in REQUIRED_COLUMNS.items():
        path = root / "csv" / name
        recorded = recorded_files.get(name)
        if not path.is_file() or not isinstance(recorded, dict):
            raise RuntimeError(f"required source file is missing from disk/manifest: {name}")
        actual_sha = sha256_file(path)
        if actual_sha != recorded.get("sha256"):
            raise RuntimeError(f"source hash mismatch: {name}")
        columns, rows = _read_csv(path)
        if columns != recorded.get("columns"):
            raise RuntimeError(f"source columns do not match manifest: {name}")
        missing = sorted(required - set(columns))
        if missing:
            raise RuntimeError(f"{name} missing required columns: {', '.join(missing)}")
        if len(rows) != recorded.get("rows"):
            raise RuntimeError(
                f"source row-count mismatch: {name} actual={len(rows)} recorded={recorded.get('rows')}"
            )
        loaded[name] = rows
    if len(loaded["patients.csv"]) != manifest.get("actual_patient_count"):
        raise RuntimeError("patients.csv count does not match actual_patient_count")
    return SourceDataset(
        root=root,
        profile=expected_profile,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        rows=loaded,
    )


def _unique_sorted(rows: list[dict], field: str, source: str) -> list[str]:
    values = [row[field] for row in rows]
    if any(not value for value in values):
        raise RuntimeError(f"{source} contains a blank {field}")
    if len(values) != len(set(values)):
        raise RuntimeError(f"{source} contains duplicate {field} values")
    return sorted(values)


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _write_csv(path: Path, columns: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_mappings(source: SourceDataset) -> Mappings:
    patients_sorted = _unique_sorted(source.rows["patients.csv"], "Id", "patients.csv")
    encounters_sorted = _unique_sorted(source.rows["encounters.csv"], "Id", "encounters.csv")
    patient_map = {value: SUBJECT_BASE + i for i, value in enumerate(patients_sorted, 1)}
    encounter_map = {value: HADM_BASE + i for i, value in enumerate(encounters_sorted, 1)}
    if max(patient_map.values(), default=0) > 2_147_483_647:
        raise RuntimeError("subject_id mapping exceeds PostgreSQL INTEGER")
    if max(encounter_map.values(), default=0) > 2_147_483_647:
        raise RuntimeError("hadm_id mapping exceeds PostgreSQL INTEGER")

    concepts = defaultdict(
    lambda: {"labels": Counter(), "categories": Counter()}
    )

    for row in source.rows["observations.csv"]:
        code = row["CODE"]
        if not code:
            raise RuntimeError("observations.csv contains a blank CODE")

        label = _normalize_text(row["DESCRIPTION"])
        category = row["CATEGORY"].strip()

        concepts[code]["labels"][label] += 1
        concepts[code]["categories"][category] += 1

    item_map = {
        code: ITEM_BASE + i
        for i, code in enumerate(sorted(concepts), 1)
    }

    metadata = {}
    for code in sorted(concepts):
        labels = concepts[code]["labels"]
        categories = concepts[code]["categories"]

        label = min(
            labels,
            key=lambda value: (-labels[value], value.casefold(), value),
        )
        category = min(
            categories,
            key=lambda value: (-categories[value], value.casefold(), value),
        ) or None

        metadata[code] = (label, category)

    derived = source.root / "derived"
    staging = Path(tempfile.mkdtemp(prefix=".derived-", dir=source.root))
    try:
        _write_csv(
            staging / "patient_ids.csv", ["source_patient_id", "subject_id"],
            ({"source_patient_id": source_id, "subject_id": patient_map[source_id]}
             for source_id in patients_sorted),
        )
        encounter_by_id = {row["Id"]: row for row in source.rows["encounters.csv"]}
        _write_csv(
            staging / "encounter_ids.csv",
            ["source_encounter_id", "hadm_id", "source_patient_id", "subject_id"],
            ({
                "source_encounter_id": source_id,
                "hadm_id": encounter_map[source_id],
                "source_patient_id": encounter_by_id[source_id]["PATIENT"],
                "subject_id": patient_map[encounter_by_id[source_id]["PATIENT"]],
            } for source_id in encounters_sorted),
        )
        _write_csv(
            staging / "observation_items.csv",
            ["source_namespace", "source_code", "itemid", "label", "category"],
            ({
                "source_namespace": OBSERVATION_NAMESPACE,
                "source_code": code,
                "itemid": item_map[code],
                "label": metadata[code][0],
                "category": metadata[code][1] or "",
            } for code in sorted(item_map)),
        )
        _write_csv(
            staging / "observation_events.csv",
            ["source_file", "source_row", "labevent_id", "source_patient_id",
             "source_encounter_id", "source_code"],
            ({
                "source_file": "observations.csv",
                "source_row": row["_source_row"],
                "labevent_id": LABEVENT_BASE + row["_source_row"],
                "source_patient_id": row["PATIENT"],
                "source_encounter_id": row["ENCOUNTER"],
                "source_code": row["CODE"],
            } for row in source.rows["observations.csv"]),
        )
        _write_csv(
            staging / "medication_events.csv",
            ["source_file", "source_row", "pharmacy_id", "source_patient_id",
             "source_encounter_id", "source_code"],
            ({
                "source_file": "medications.csv",
                "source_row": row["_source_row"],
                "pharmacy_id": PHARMACY_BASE + row["_source_row"],
                "source_patient_id": row["PATIENT"],
                "source_encounter_id": row["ENCOUNTER"],
                "source_code": row["CODE"],
            } for row in source.rows["medications.csv"]),
        )
        mapping_files = sorted(staging.glob("*.csv"))
        manifest = {
            "schema_version": 1,
            "mapping_algorithm": MAPPING_VERSION,
            "source_manifest_sha256": source.manifest_sha256,
            "source_profile": source.profile,
            "row_counts": {
                "patients": len(patient_map),
                "encounters": len(encounter_map),
                "observation_items": len(item_map),
                "observation_events": len(source.rows["observations.csv"]),
                "medication_events": len(source.rows["medications.csv"]),
            },
            "files": {
                path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in mapping_files
            },
        }
        (staging / "mapping_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if derived.exists():
            shutil.rmtree(derived)
        staging.replace(derived)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest_path = derived / "mapping_manifest.json"
    return Mappings(
        patients=patient_map,
        encounters=encounter_map,
        items=item_map,
        item_metadata=metadata,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
    )


def _utc_naive(value: str) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _date(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


def _age_on(birth: date, anchor: date) -> int:
    return anchor.year - birth.year - ((anchor.month, anchor.day) < (birth.month, birth.day))


def _require_ref(mapping: dict[str, int], value: str, source: str, row: int) -> int:
    try:
        return mapping[value]
    except KeyError:
        raise RuntimeError(f"{source} row {row} has unknown reference {value!r}") from None


def prepare_data(source: SourceDataset, mappings: Mappings) -> PreparedData:
    end_date = datetime.strptime(
        source.manifest["generation_parameters"]["end_date"], "%Y%m%d"
    ).date()
    patients = []
    for row in source.rows["patients.csv"]:
        birth, death = _date(row["BIRTHDATE"]), _date(row["DEATHDATE"])
        anchor = death or end_date
        patients.append({
            "subject_id": mappings.patients[row["Id"]],
            "gender": row["GENDER"] or None,
            "anchor_age": _age_on(birth, anchor),
            "anchor_year": anchor.year,
            "anchor_year_group": f"{anchor.year // 10 * 10} - {anchor.year // 10 * 10 + 9}",
            "dod": death,
        })

    admissions = []
    for row in source.rows["encounters.csv"]:
        start, stop = _utc_naive(row["START"]), _utc_naive(row["STOP"])
        if start is None or stop is None or start > stop:
            raise RuntimeError(f"encounters.csv row {row['_source_row']} has invalid START/STOP")
        admissions.append({
            "hadm_id": mappings.encounters[row["Id"]],
            "subject_id": _require_ref(
                mappings.patients, row["PATIENT"], "encounters.csv", row["_source_row"]
            ),
            "admittime": start,
            "dischtime": stop,
            "admission_type": row["ENCOUNTERCLASS"],
        })

    labitems = [
        {"itemid": mappings.items[code], "label": metadata[0],
         "fluid": None, "category": metadata[1]}
        for code, metadata in sorted(mappings.item_metadata.items())
    ]
    labevents = []
    duplicate_observations = 0
    seen_observations = set()
    for row in source.rows["observations.csv"]:
        source_values = tuple(row.get(column, "") for column in OBSERVATION_DUPLICATE_FIELDS)
        duplicate_observations += source_values in seen_observations
        seen_observations.add(source_values)
        value_num = None
        if row["TYPE"] == "numeric":
            try:
                value_num = float(row["VALUE"])
            except ValueError:
                raise RuntimeError(
                    f"observations.csv row {row['_source_row']} is numeric but VALUE is not"
                ) from None
            if not math.isfinite(value_num):
                raise RuntimeError(f"observations.csv row {row['_source_row']} is not finite")
        hadm_id = None
        if row["ENCOUNTER"]:
            hadm_id = _require_ref(
                mappings.encounters, row["ENCOUNTER"], "observations.csv", row["_source_row"]
            )
        labevents.append({
            "labevent_id": LABEVENT_BASE + row["_source_row"],
            "subject_id": _require_ref(
                mappings.patients, row["PATIENT"], "observations.csv", row["_source_row"]
            ),
            "hadm_id": hadm_id,
            "itemid": mappings.items[row["CODE"]],
            "charttime": _utc_naive(row["DATE"]),
            "value": row["VALUE"],
            "valuenum": value_num,
            "valueuom": row["UNITS"] or None,
        })

    eligible_conditions = [row for row in source.rows["conditions.csv"] if row["SYSTEM"] == "ICD10"]
    by_encounter: dict[str, list[dict]] = defaultdict(list)
    for row in eligible_conditions:
        by_encounter[row["ENCOUNTER"]].append(row)
    diagnoses = []
    for encounter_id in sorted(by_encounter):
        rows = sorted(
            by_encounter[encounter_id],
            key=lambda row: (row["START"], row["CODE"], row["DESCRIPTION"], row["_source_row"]),
        )
        for seq_num, row in enumerate(rows, 1):
            if len(row["CODE"]) > 10:
                raise RuntimeError(f"ICD10 code does not fit schema: {row['CODE']!r}")
            diagnoses.append({
                "subject_id": _require_ref(
                    mappings.patients, row["PATIENT"], "conditions.csv", row["_source_row"]
                ),
                "hadm_id": _require_ref(
                    mappings.encounters, row["ENCOUNTER"], "conditions.csv", row["_source_row"]
                ),
                "seq_num": seq_num,
                "icd_code": row["CODE"],
                "icd_version": 10,
            })

    prescriptions = []
    invalid_medication_stops = 0
    for row in source.rows["medications.csv"]:
        start, stop = _utc_naive(row["START"]), _utc_naive(row["STOP"])
        if stop is not None and (start is None or stop < start):
            stop = None
            invalid_medication_stops += 1
        prescriptions.append({
            "subject_id": _require_ref(
                mappings.patients, row["PATIENT"], "medications.csv", row["_source_row"]
            ),
            "hadm_id": _require_ref(
                mappings.encounters, row["ENCOUNTER"], "medications.csv", row["_source_row"]
            ),
            "pharmacy_id": PHARMACY_BASE + row["_source_row"],
            "starttime": start,
            "stoptime": stop,
            "drug_type": "synthea",
            "drug": row["DESCRIPTION"],
        })

    non_icd_procedures = sum(
        row["SYSTEM"] not in {"ICD9", "ICD10"} for row in source.rows["procedures.csv"]
    )
    procedures = []
    for row in source.rows["procedures.csv"]:
        if row["SYSTEM"] not in {"ICD9", "ICD10"}:
            continue
        if len(row["CODE"]) > 10:
            raise RuntimeError(f"procedure ICD code does not fit schema: {row['CODE']!r}")
        procedures.append({
            "subject_id": _require_ref(
                mappings.patients, row["PATIENT"], "procedures.csv", row["_source_row"]
            ),
            "hadm_id": _require_ref(
                mappings.encounters, row["ENCOUNTER"], "procedures.csv", row["_source_row"]
            ),
            "seq_num": 0,
            "chartdate": _utc_naive(row["START"]).date(),
            "icd_code": row["CODE"],
            "icd_version": 10 if row["SYSTEM"] == "ICD10" else 9,
        })
    # Sequence is only meaningful within one encounter.
    grouped_procedures: dict[int, list[dict]] = defaultdict(list)
    for row in procedures:
        grouped_procedures[row["hadm_id"]].append(row)
    for rows in grouped_procedures.values():
        rows.sort(key=lambda row: (row["chartdate"], row["icd_code"]))
        for seq_num, row in enumerate(rows, 1):
            row["seq_num"] = seq_num

    warnings = {
        "unscoped_observations": sum(row["hadm_id"] is None for row in labevents),
        "exact_duplicate_observations_preserved": duplicate_observations,
        "excluded_non_icd_conditions": len(source.rows["conditions.csv"]) - len(diagnoses),
        "excluded_non_icd_procedures": non_icd_procedures,
        "invalid_medication_stops_nulled": invalid_medication_stops,
    }
    return PreparedData(
        patients, admissions, labitems, labevents, diagnoses,
        prescriptions, procedures, warnings,
    )


def _fact_line(row: dict, fields: tuple[tuple[str, str], ...]) -> str:
    parts = [f"{label}={row[field]}" for label, field in fields if row.get(field)]
    return "- " + " | ".join(parts)


def _group_linked_facts(
    source: SourceDataset, mappings: Mappings, filename: str, timestamp_field: str,
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in source.rows[filename]:
        encounter_id = row["ENCOUNTER"]
        if not encounter_id:
            continue
        if encounter_id not in mappings.encounters:
            raise RuntimeError(
                f"{filename} row {row['_source_row']} has unknown encounter {encounter_id!r}"
            )
        grouped[encounter_id].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (
            row[timestamp_field], row["CODE"], row["DESCRIPTION"], row["_source_row"],
        ))
    return grouped


def prepare_notes(source: SourceDataset, mappings: Mappings) -> PreparedNotes:
    conditions = _group_linked_facts(source, mappings, "conditions.csv", "START")
    procedures = _group_linked_facts(source, mappings, "procedures.csv", "START")
    medications = _group_linked_facts(source, mappings, "medications.csv", "START")
    observations = _group_linked_facts(source, mappings, "observations.csv", "DATE")
    encounter_rows = {row["Id"]: row for row in source.rows["encounters.csv"]}

    section_counts = {
        "encounters": 0,
        "conditions": 0,
        "procedures": 0,
        "medications": 0,
        "observations": 0,
    }
    notes = []
    for position, encounter_id in enumerate(sorted(mappings.encounters), 1):
        encounter = encounter_rows[encounter_id]
        lines = ["Encounter Summary", "", "Encounter:"]
        for label, field in (
            ("Start", "START"),
            ("Stop", "STOP"),
            ("Type", "ENCOUNTERCLASS"),
            ("Code", "CODE"),
            ("Description", "DESCRIPTION"),
            ("Reason Code", "REASONCODE"),
            ("Reason Description", "REASONDESCRIPTION"),
        ):
            if encounter[field]:
                lines.append(f"{label}: {encounter[field]}")

        sections = (
            ("Conditions", conditions.get(encounter_id, []), (
                ("Start", "START"), ("Stop", "STOP"), ("System", "SYSTEM"),
                ("Code", "CODE"), ("Description", "DESCRIPTION"),
            )),
            ("Procedures", procedures.get(encounter_id, []), (
                ("Start", "START"), ("Stop", "STOP"), ("System", "SYSTEM"),
                ("Code", "CODE"), ("Description", "DESCRIPTION"),
                ("Reason", "REASONDESCRIPTION"),
            )),
            ("Medications", medications.get(encounter_id, []), (
                ("Start", "START"), ("Stop", "STOP"), ("Code", "CODE"),
                ("Description", "DESCRIPTION"), ("Reason", "REASONDESCRIPTION"),
            )),
            ("Observations", observations.get(encounter_id, []), (
                ("Date", "DATE"), ("Category", "CATEGORY"), ("Code", "CODE"),
                ("Description", "DESCRIPTION"), ("Value", "VALUE"),
                ("Units", "UNITS"), ("Type", "TYPE"),
            )),
        )
        for heading, facts, fields in sections:
            if not facts:
                continue
            section_counts[heading.lower()] += 1
            lines.extend(("", f"{heading}:"))
            lines.extend(_fact_line(row, fields) for row in facts)
        section_counts["encounters"] += 1

        text = "\n".join(lines) + "\n"
        notes.append({
            "note_id": NOTE_BASE + position,
            "subject_id": mappings.patients[encounter["PATIENT"]],
            "hadm_id": mappings.encounters[encounter_id],
            "note_type": "encounter_summary",
            "charttime": _utc_naive(encounter["STOP"] or encounter["START"]),
            "text": text,
            "encounter_uuid": encounter_id,
        })

    corpus_sha256, _ = _note_corpus_hash(
        (row["note_id"], row["text"]) for row in notes
    )
    lengths = [len(row["text"]) for row in notes]
    length_stats = {
        "min": min(lengths, default=0),
        "median": statistics.median(lengths) if lengths else 0,
        "max": max(lengths, default=0),
    }
    return PreparedNotes(notes, section_counts, length_stats, corpus_sha256)


def validate_target_database(
    *, plane: str, actual: str | None, expected: str,
    research: str | None, demo: str,
) -> None:
    if plane != "synthea":
        raise RuntimeError(f"Synthea ingestion refuses LUMEN_DATA_PLANE={plane!r}")
    if not actual or actual != expected:
        raise RuntimeError(f"Synthea target is {actual!r}, expected {expected!r}")
    if actual in {research, demo}:
        raise RuntimeError(f"Synthea ingestion refuses protected database {actual!r}")
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", actual):
        raise RuntimeError(f"unsafe Synthea database name {actual!r}")


def _ensure_database(storage) -> None:
    from sqlalchemy import create_engine, text

    url = storage.engine.url
    server = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with server.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname=:name"),
                {"name": storage.SYNTHEA_DB_NAME},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{storage.SYNTHEA_DB_NAME}"'))
    finally:
        server.dispose()


def _assert_database_contains_only_synthea(storage, *, allow_notes: bool = False) -> None:
    from sqlalchemy import text

    checks = {
        "patients": "subject_id NOT BETWEEN 80000001 AND 80999999",
        "admissions": "subject_id NOT BETWEEN 80000001 AND 80999999 OR hadm_id NOT BETWEEN 800000001 AND 809999999",
        "diagnoses_icd": "subject_id NOT BETWEEN 80000001 AND 80999999 OR hadm_id NOT BETWEEN 800000001 AND 809999999",
        "labevents": "subject_id NOT BETWEEN 80000001 AND 80999999 OR labevent_id NOT BETWEEN 8000000001 AND 8999999999",
        "d_labitems": "itemid NOT BETWEEN 980001 AND 989999",
        "prescriptions": "subject_id NOT BETWEEN 80000001 AND 80999999 OR hadm_id NOT BETWEEN 800000001 AND 809999999 OR pharmacy_id IS NULL OR pharmacy_id NOT BETWEEN 8100000001 AND 8199999999",
        "procedures_icd": "subject_id NOT BETWEEN 80000001 AND 80999999 OR hadm_id NOT BETWEEN 800000001 AND 809999999",
    }
    with storage.engine.connect() as conn:
        actual = conn.execute(text("SELECT current_database()" )).scalar()
        validate_target_database(
            plane=storage.DATA_PLANE, actual=actual, expected=storage.SYNTHEA_DB_NAME,
            research=storage.RESEARCH_DB_NAME, demo=storage.DEMO_DB_NAME,
        )
        foreign = 0
        for table, predicate in checks.items():
            if conn.execute(text("SELECT to_regclass(:table)"), {"table": table}).scalar():
                foreign += conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE {predicate}")).scalar()
        if conn.execute(text("SELECT to_regclass('clinical_notes')")).scalar():
            if allow_notes:
                foreign += conn.execute(text("""
                    SELECT COUNT(*) FROM clinical_notes
                    WHERE subject_id NOT BETWEEN 80000001 AND 80999999
                       OR note_id NOT BETWEEN 850000001 AND 859999999
                       OR note_type <> 'encounter_summary'
                       OR COALESCE(phi_entities->>'source', '') <> 'synthea'
                """)).scalar()
            else:
                foreign += conn.execute(text("SELECT COUNT(*) FROM clinical_notes")).scalar()
        if conn.execute(text("SELECT to_regclass('note_chunks')")).scalar():
            foreign += conn.execute(text("SELECT COUNT(*) FROM note_chunks")).scalar()
        if foreign:
            raise RuntimeError(
                f"Synthea target contains {foreign} non-Synthea or later-phase rows; refusing"
            )


def _chunks(rows: list[dict], size: int = 1000):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def _insert_many(conn, statement: str, rows: list[dict]) -> None:
    from sqlalchemy import text

    for batch in _chunks(rows):
        conn.execute(text(statement), batch)


def _replace_canonical(storage, prepared: PreparedData) -> None:
    from sqlalchemy import text

    with storage.engine.begin() as conn:
        for table in (
            "diagnoses_icd", "prescriptions", "procedures_icd", "labevents",
            "d_labitems", "admissions", "patients",
        ):
            conn.execute(text(f"DELETE FROM {table}"))
        _insert_many(conn, """
            INSERT INTO patients
                (subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod)
            VALUES
                (:subject_id, :gender, :anchor_age, :anchor_year, :anchor_year_group, :dod)
        """, prepared.patients)
        _insert_many(conn, """
            INSERT INTO admissions
                (hadm_id, subject_id, admittime, dischtime, admission_type)
            VALUES
                (:hadm_id, :subject_id, :admittime, :dischtime, :admission_type)
        """, prepared.admissions)
        _insert_many(conn, """
            INSERT INTO d_labitems (itemid, label, fluid, category)
            VALUES (:itemid, :label, :fluid, :category)
        """, prepared.labitems)
        _insert_many(conn, """
            INSERT INTO labevents
                (labevent_id, subject_id, hadm_id, itemid, charttime,
                 value, valuenum, valueuom)
            VALUES
                (:labevent_id, :subject_id, :hadm_id, :itemid, :charttime,
                 :value, :valuenum, :valueuom)
        """, prepared.labevents)
        _insert_many(conn, """
            INSERT INTO diagnoses_icd
                (subject_id, hadm_id, seq_num, icd_code, icd_version)
            VALUES
                (:subject_id, :hadm_id, :seq_num, :icd_code, :icd_version)
        """, prepared.diagnoses)
        _insert_many(conn, """
            INSERT INTO prescriptions
                (subject_id, hadm_id, pharmacy_id, starttime, stoptime, drug_type, drug)
            VALUES
                (:subject_id, :hadm_id, :pharmacy_id, :starttime, :stoptime, :drug_type, :drug)
        """, prepared.prescriptions)
        _insert_many(conn, """
            INSERT INTO procedures_icd
                (subject_id, hadm_id, seq_num, chartdate, icd_code, icd_version)
            VALUES
                (:subject_id, :hadm_id, :seq_num, :chartdate, :icd_code, :icd_version)
        """, prepared.procedures)


def _replace_notes(storage, source: SourceDataset, mappings: Mappings, notes: PreparedNotes) -> None:
    from sqlalchemy import text

    marker_base = {
        "synthetic": True,
        "source": "synthea",
        "source_generation_manifest_sha256": source.manifest_sha256,
        "mapping_manifest_sha256": mappings.manifest_sha256,
        "note_renderer_version": NOTE_RENDERER_VERSION,
    }
    rows = [{
        **row,
        "marker": json.dumps(
            {**marker_base, "encounter_uuid": row["encounter_uuid"]}, sort_keys=True,
        ),
    } for row in notes.rows]
    with storage.engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM clinical_notes
            WHERE note_id BETWEEN 850000001 AND 859999999
        """))
        _insert_many(conn, """
            INSERT INTO clinical_notes
                (note_id, subject_id, hadm_id, note_type, charttime,
                 text_original, text_deid, phi_entities)
            VALUES
                (:note_id, :subject_id, :hadm_id, :note_type, :charttime,
                 NULL, :text, CAST(:marker AS jsonb))
        """, rows)


def clear_synthea_note_corpus(storage) -> None:
    """Explicitly clear only Synthea note/index rows before a profile switch."""
    from sqlalchemy import text

    actual = storage.engine.url.database
    validate_target_database(
        plane=storage.DATA_PLANE, actual=actual, expected=storage.SYNTHEA_DB_NAME,
        research=storage.RESEARCH_DB_NAME, demo=storage.DEMO_DB_NAME,
    )
    with storage.engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM note_index_state
            WHERE note_id BETWEEN 850000001 AND 859999999
        """))
        conn.execute(text("""
            DELETE FROM note_chunks
            WHERE note_id BETWEEN 850000001 AND 859999999
        """))
        conn.execute(text("""
            DELETE FROM clinical_notes
            WHERE note_id BETWEEN 850000001 AND 859999999
              AND COALESCE(phi_entities->>'source', '') = 'synthea'
        """))


def _record_run(storage, run_id: str, status: str, configuration: dict, error: str | None = None) -> None:
    from sqlalchemy import text

    payload = json.dumps(configuration, sort_keys=True)
    with storage.engine.begin() as conn:
        if status == "running":
            conn.execute(text("""
                INSERT INTO ingestion_runs (run_id, status, configuration)
                VALUES (:run_id, 'running', CAST(:configuration AS jsonb))
            """), {"run_id": run_id, "configuration": payload})
        else:
            conn.execute(text("""
                UPDATE ingestion_runs
                SET status=:status, completed_at=NOW(), error_message=:error,
                    configuration=CAST(:configuration AS jsonb)
                WHERE run_id=:run_id
            """), {
                "run_id": run_id, "status": status, "error": error,
                "configuration": payload,
            })


def ingest(source: SourceDataset, mappings: Mappings, prepared: PreparedData) -> dict:
    import src  # noqa: F401 - load .env without overriding explicit plane selection
    from sqlalchemy import text
    from src import storage
    from src.storage.schema import create_schema

    actual = storage.engine.url.database
    validate_target_database(
        plane=storage.DATA_PLANE, actual=actual, expected=storage.SYNTHEA_DB_NAME,
        research=storage.RESEARCH_DB_NAME, demo=storage.DEMO_DB_NAME,
    )
    _ensure_database(storage)
    _assert_database_contains_only_synthea(storage)
    create_schema()
    _assert_database_contains_only_synthea(storage)

    run_id = str(uuid.uuid4())
    configuration = {
        "data_plane": "synthea",
        "profile": source.profile,
        "source_manifest_sha256": source.manifest_sha256,
        "mapping_manifest_sha256": mappings.manifest_sha256,
        "mapping_algorithm": MAPPING_VERSION,
        "row_counts": prepared.counts(),
        "warnings": prepared.warnings,
    }
    _record_run(storage, run_id, "running", configuration)
    try:
        _replace_canonical(storage, prepared)
        with storage.engine.connect() as conn:
            actual_counts = {
                table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                for table in prepared.counts()
            }
        if actual_counts != prepared.counts():
            raise RuntimeError(
                f"post-load count mismatch: expected={prepared.counts()} actual={actual_counts}"
            )
        configuration["actual_row_counts"] = actual_counts
        _record_run(storage, run_id, "completed", configuration)
    except BaseException as exc:
        _record_run(storage, run_id, "failed", configuration, type(exc).__name__)
        raise
    return {
        "run_id": run_id,
        "database": actual,
        "data_plane": storage.DATA_PLANE,
        "source_manifest_sha256": source.manifest_sha256,
        "mapping_manifest_sha256": mappings.manifest_sha256,
        "counts": actual_counts,
        "warnings": prepared.warnings,
    }


def ingest_notes(source: SourceDataset, mappings: Mappings, notes: PreparedNotes) -> dict:
    import src  # noqa: F401 - load .env before storage resolves the selected plane
    from sqlalchemy import text
    from src import storage

    actual = storage.engine.url.database
    validate_target_database(
        plane=storage.DATA_PLANE, actual=actual, expected=storage.SYNTHEA_DB_NAME,
        research=storage.RESEARCH_DB_NAME, demo=storage.DEMO_DB_NAME,
    )
    _assert_database_contains_only_synthea(storage, allow_notes=True)
    expected_admissions = {
        (mappings.encounters[row["Id"]], mappings.patients[row["PATIENT"]])
        for row in source.rows["encounters.csv"]
    }
    with storage.engine.connect() as conn:
        actual_admissions = {
            tuple(row) for row in conn.execute(text("SELECT hadm_id, subject_id FROM admissions"))
        }
    if actual_admissions != expected_admissions:
        raise RuntimeError(
            "lumen_synthea admissions do not match this source manifest; refusing note writes"
        )

    run_id = str(uuid.uuid4())
    configuration = {
        "data_plane": "synthea",
        "operation": "clinical_notes",
        "profile": source.profile,
        "source_manifest_sha256": source.manifest_sha256,
        "mapping_manifest_sha256": mappings.manifest_sha256,
        "note_renderer_version": NOTE_RENDERER_VERSION,
        "note_count": len(notes.rows),
        "section_counts": notes.section_counts,
        "length_stats": notes.length_stats,
        "corpus_sha256": notes.corpus_sha256,
    }
    _record_run(storage, run_id, "running", configuration)
    try:
        _replace_notes(storage, source, mappings, notes)
        with storage.engine.connect() as conn:
            stored = conn.execution_options(stream_results=True).execute(text("""
                    SELECT note_id, text_deid FROM clinical_notes
                    WHERE note_id BETWEEN 850000001 AND 859999999
                    ORDER BY note_id
                """))
            stored_sha256, stored_count = _note_corpus_hash(
                (row.note_id, row.text_deid) for row in stored
            )
        if stored_count != len(notes.rows) or stored_sha256 != notes.corpus_sha256:
            raise RuntimeError(
                "post-load note count/hash mismatch: "
                f"expected={len(notes.rows)}/{notes.corpus_sha256} "
                f"actual={stored_count}/{stored_sha256}"
            )
        configuration["actual_note_count"] = stored_count
        _record_run(storage, run_id, "completed", configuration)
    except BaseException as exc:
        _record_run(storage, run_id, "failed", configuration, type(exc).__name__)
        raise
    return {
        "run_id": run_id,
        "database": actual,
        "data_plane": storage.DATA_PLANE,
        "notes_created": len(notes.rows),
        "encounters_without_notes": len(mappings.encounters) - len(notes.rows),
        "section_counts": notes.section_counts,
        "length_stats": notes.length_stats,
        "corpus_sha256": notes.corpus_sha256,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--clinical-notes-only", action="store_true",
        help="render and replace deterministic encounter_summary notes only",
    )
    parser.add_argument(
        "--with-clinical-notes", action="store_true",
        help="load canonical rows, then render deterministic encounter_summary notes",
    )
    parser.add_argument(
        "--replace-existing-profile", action="store_true",
        help="explicitly replace the existing Synthea DEV/EVAL note corpus and index state",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    synthea_root = Path(
        os.environ.get("LUMEN_SYNTHEA_DIR", ROOT / "data" / "synthea")
    ).expanduser()
    dataset_dir = args.dataset_dir or synthea_root / args.profile
    source = validate_source(dataset_dir, args.profile)
    mappings = build_mappings(source)
    if args.clinical_notes_only and args.with_clinical_notes:
        raise SystemExit("choose only one of --clinical-notes-only or --with-clinical-notes")
    if args.clinical_notes_only:
        notes = prepare_notes(source, mappings)
        print(json.dumps(ingest_notes(source, mappings, notes), indent=2, sort_keys=True))
        return 0
    if args.replace_existing_profile:
        import src  # noqa: F401
        from src import storage
        clear_synthea_note_corpus(storage)
    prepared = prepare_data(source, mappings)
    result = ingest(source, mappings, prepared)
    if args.with_clinical_notes:
        result["clinical_notes"] = ingest_notes(
            source, mappings, prepare_notes(source, mappings)
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
