from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.config import get_data_plane
from src.storage.ingest_synthea import (
    HADM_BASE,
    ITEM_BASE,
    NOTE_BASE,
    SUBJECT_BASE,
    _replace_notes,
    build_mappings,
    prepare_data,
    prepare_notes,
    validate_source,
    validate_target_database,
)


FIXTURE_ROWS = {
    "patients.csv": [
        {"Id": "patient-b", "BIRTHDATE": "1980-06-01", "DEATHDATE": "", "GENDER": "F"},
        {"Id": "patient-a", "BIRTHDATE": "1970-01-01", "DEATHDATE": "2020-01-02", "GENDER": "M"},
    ],
    "encounters.csv": [
        {
            "Id": "encounter-b", "PATIENT": "patient-b",
            "START": "2024-02-01T10:00:00Z", "STOP": "2024-02-01T11:00:00Z",
            "ENCOUNTERCLASS": "ambulatory", "CODE": "enc-b", "DESCRIPTION": "Visit B",
            "REASONCODE": "reason-b", "REASONDESCRIPTION": "Reason B",
        },
        {
            "Id": "encounter-a", "PATIENT": "patient-a",
            "START": "2024-01-01T10:00:00Z", "STOP": "2024-01-01T12:00:00Z",
            "ENCOUNTERCLASS": "inpatient", "CODE": "enc-a", "DESCRIPTION": "Visit A",
            "REASONCODE": "", "REASONDESCRIPTION": "",
        },
    ],
    "observations.csv": [
        {
            "DATE": "2024-01-01T10:30:00Z", "PATIENT": "patient-a",
            "ENCOUNTER": "encounter-a", "CATEGORY": "laboratory", "CODE": "code-z",
            "DESCRIPTION": "  Test   Z  ", "VALUE": "12.5", "UNITS": "mg/dL", "TYPE": "numeric",
        },
        {
            "DATE": "2024-02-01T10:30:00Z", "PATIENT": "patient-b", "ENCOUNTER": "",
            "CATEGORY": "survey", "CODE": "code-a", "DESCRIPTION": "Text A",
            "VALUE": "123", "UNITS": "", "TYPE": "text",
        },
        {
            "DATE": "2024-02-01T10:30:00Z", "PATIENT": "patient-b", "ENCOUNTER": "",
            "CATEGORY": "survey", "CODE": "code-a", "DESCRIPTION": "Text A",
            "VALUE": "123", "UNITS": "", "TYPE": "text",
        },
    ],
    "conditions.csv": [
        {
            "START": "2024-01-01", "STOP": "", "PATIENT": "patient-a",
            "ENCOUNTER": "encounter-a", "SYSTEM": "ICD10", "CODE": "E11.9",
            "DESCRIPTION": "Diabetes",
        },
        {
            "START": "2024-02-01", "STOP": "", "PATIENT": "patient-b",
            "ENCOUNTER": "encounter-b", "SYSTEM": "SNOMED-CT", "CODE": "44054006",
            "DESCRIPTION": "Diabetes",
        },
    ],
    "medications.csv": [
        {
            "START": "2024-01-01T10:00:00Z", "STOP": "2024-01-02T10:00:00Z",
            "PATIENT": "patient-a", "ENCOUNTER": "encounter-a", "CODE": "med-1",
            "DESCRIPTION": "Medicine One", "REASONDESCRIPTION": "Reason for medicine one",
        },
        {
            "START": "2024-02-02T10:00:00Z", "STOP": "2024-02-01T10:00:00Z",
            "PATIENT": "patient-b", "ENCOUNTER": "encounter-b", "CODE": "med-2",
            "DESCRIPTION": "Medicine Two", "REASONDESCRIPTION": "",
        },
    ],
    "procedures.csv": [
        {
            "START": "2024-01-01T10:00:00Z", "STOP": "2024-01-01T11:00:00Z",
            "PATIENT": "patient-a", "ENCOUNTER": "encounter-a", "SYSTEM": "SNOMED-CT",
            "CODE": "proc-1", "DESCRIPTION": "Procedure One", "REASONDESCRIPTION": "Reason one",
        },
        {
            "START": "2024-02-01T10:00:00Z", "STOP": "2024-02-01T11:00:00Z",
            "PATIENT": "patient-b", "ENCOUNTER": "encounter-b", "SYSTEM": "CDT",
            "CODE": "proc-2", "DESCRIPTION": "Procedure Two", "REASONDESCRIPTION": "",
        },
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> Path:
    csv_dir = root / "csv"
    csv_dir.mkdir(parents=True)
    files = {}
    for name, rows in FIXTURE_ROWS.items():
        path = csv_dir / name
        columns = list(rows[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        files[name] = {"columns": columns, "rows": len(rows), "sha256": _sha256(path)}
    manifest = {
        "schema_version": 1,
        "synthetic": True,
        "profile": "dev",
        "actual_patient_count": 2,
        "generation_parameters": {"end_date": "20250101"},
        "synthea": {"project": "synthetichealth/synthea", "version": "v4.0.0"},
        "csv_files": files,
    }
    (root / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def prepared(tmp_path):
    source = validate_source(_write_fixture(tmp_path / "dev"), "dev")
    mappings = build_mappings(source)
    return source, mappings, prepare_data(source, mappings)


def test_deterministic_sorted_ids_and_mapping_artifacts(prepared):
    source, mappings, _ = prepared
    assert mappings.patients == {"patient-a": SUBJECT_BASE + 1, "patient-b": SUBJECT_BASE + 2}
    assert mappings.encounters == {"encounter-a": HADM_BASE + 1, "encounter-b": HADM_BASE + 2}
    assert mappings.items == {"code-a": ITEM_BASE + 1, "code-z": ITEM_BASE + 2}

    derived = source.root / "derived"
    before = {path.name: path.read_bytes() for path in sorted(derived.iterdir())}
    repeated = build_mappings(source)
    after = {path.name: path.read_bytes() for path in sorted(derived.iterdir())}
    assert before == after
    assert repeated.manifest_sha256 == mappings.manifest_sha256
    assert set(after) == {
        "mapping_manifest.json", "patient_ids.csv", "encounter_ids.csv",
        "observation_items.csv", "observation_events.csv", "medication_events.csv",
    }


def test_manifest_validation_fails_closed_on_hash_and_columns(tmp_path):
    root = _write_fixture(tmp_path / "hash")
    with (root / "csv" / "patients.csv").open("a", encoding="utf-8") as handle:
        handle.write("corrupt\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_source(root, "dev")

    root = _write_fixture(tmp_path / "columns")
    manifest_path = root / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["csv_files"]["patients.csv"]["columns"] = ["Id"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="columns do not match manifest"):
        validate_source(root, "dev")


def test_synthea_plane_and_database_guards():
    assert get_data_plane("synthea") == "synthea"
    with pytest.raises(RuntimeError, match="refuses LUMEN_DATA_PLANE"):
        validate_target_database(
            plane="research", actual="lumen_synthea", expected="lumen_synthea",
            research="lumen", demo="lumen_demo",
        )
    for protected in ("lumen", "lumen_demo"):
        with pytest.raises(RuntimeError):
            validate_target_database(
                plane="synthea", actual=protected, expected=protected,
                research="lumen", demo="lumen_demo",
            )


def test_patients_and_encounters_resolve_deterministically(prepared):
    _, mappings, data = prepared
    assert {row["subject_id"] for row in data.patients} == set(mappings.patients.values())
    assert data.patients[0]["anchor_year"] == 2025
    assert data.patients[1]["anchor_year"] == 2020
    assert {row["hadm_id"] for row in data.admissions} == set(mappings.encounters.values())
    assert {row["subject_id"] for row in data.admissions} == set(mappings.patients.values())
    assert {row["admission_type"] for row in data.admissions} == {"ambulatory", "inpatient"}


def test_observation_numeric_text_unscoped_and_duplicates(prepared):
    _, _, data = prepared
    numeric = next(row for row in data.labevents if row["value"] == "12.5")
    text_rows = [row for row in data.labevents if row["value"] == "123"]
    assert numeric["valuenum"] == 12.5
    assert all(row["valuenum"] is None for row in text_rows)
    assert all(row["hadm_id"] is None for row in text_rows)
    assert len(text_rows) == 2
    assert data.warnings["unscoped_observations"] == 2
    assert data.warnings["exact_duplicate_observations_preserved"] == 1


def test_only_icd10_conditions_and_no_non_icd_procedures(prepared):
    _, _, data = prepared
    assert [(row["icd_code"], row["icd_version"]) for row in data.diagnoses] == [("E11.9", 10)]
    assert data.warnings["excluded_non_icd_conditions"] == 1
    assert data.procedures == []
    assert data.warnings["excluded_non_icd_procedures"] == 2


def test_medications_resolve_and_invalid_stop_is_nulled(prepared):
    _, mappings, data = prepared
    assert len(data.prescriptions) == 2
    assert {row["subject_id"] for row in data.prescriptions} == set(mappings.patients.values())
    assert {row["hadm_id"] for row in data.prescriptions} == set(mappings.encounters.values())
    assert data.prescriptions[0]["stoptime"] is not None
    assert data.prescriptions[1]["stoptime"] is None
    assert data.warnings["invalid_medication_stops_nulled"] == 1


def test_repeated_preparation_has_identical_canonical_rows(prepared):
    source, mappings, first = prepared
    second = prepare_data(source, mappings)
    assert second == first
    assert second.counts() == first.counts()


def test_deterministic_notes_use_mapped_identity_and_direct_facts(prepared):
    source, mappings, _ = prepared
    first = prepare_notes(source, mappings)
    second = prepare_notes(source, mappings)
    assert first == second
    assert first.corpus_sha256 == second.corpus_sha256
    assert [row["note_id"] for row in first.rows] == [NOTE_BASE + 1, NOTE_BASE + 2]

    by_hadm = {row["hadm_id"]: row for row in first.rows}
    note_a = by_hadm[mappings.encounters["encounter-a"]]
    note_b = by_hadm[mappings.encounters["encounter-b"]]
    assert note_a["subject_id"] == mappings.patients["patient-a"]
    assert note_b["subject_id"] == mappings.patients["patient-b"]
    assert "System=ICD10" in note_a["text"]
    assert "System=SNOMED-CT" in note_b["text"]
    assert "System=SNOMED-CT | Code=proc-1" in note_a["text"]
    assert "System=CDT | Code=proc-2" in note_b["text"]
    assert "Description=Medicine One" in note_a["text"]
    assert "Description=Test   Z" not in note_a["text"]
    assert "Description=  Test   Z  " in note_a["text"]
    assert "Text A" not in note_a["text"] + note_b["text"]
    assert "RxNorm" not in note_a["text"] + note_b["text"]
    assert "Dose=" not in note_a["text"] + note_b["text"]
    assert first.section_counts == {
        "encounters": 2, "conditions": 2, "procedures": 2,
        "medications": 2, "observations": 1,
    }


def test_replacing_notes_twice_does_not_duplicate(prepared):
    source, mappings, _ = prepared
    notes = prepare_notes(source, mappings)

    class FakeConnection:
        def __init__(self):
            self.rows = [{"note_id": -1}]

        def execute(self, statement, parameters=None):
            sql = str(statement)
            if "DELETE FROM clinical_notes" in sql:
                self.rows.clear()
            elif "INSERT INTO clinical_notes" in sql:
                self.rows.extend(parameters)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class FakeEngine:
        def __init__(self):
            self.connection = FakeConnection()

        def begin(self):
            return self.connection

    class FakeStorage:
        engine = FakeEngine()

    _replace_notes(FakeStorage, source, mappings, notes)
    first_rows = list(FakeStorage.engine.connection.rows)
    _replace_notes(FakeStorage, source, mappings, notes)
    second_rows = FakeStorage.engine.connection.rows
    assert len(first_rows) == len(second_rows) == 2
    assert [row["note_id"] for row in first_rows] == [row["note_id"] for row in second_rows]
    assert len({row["note_id"] for row in second_rows}) == 2
