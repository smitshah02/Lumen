"""Focused, model-free tests for deterministic Synthea gold integration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.evals.final_eval import doctor, providers
from src.evals.final_eval import synthea_cases as S


def _fixture():
    patient = "patient-a"
    encounters = ["enc-a", "enc-b", "enc-c"]
    observations = [
        {"DATE": f"202{i}-01-0{i+1}T00:00:00Z", "PATIENT": patient,
         "ENCOUNTER": encounter, "CATEGORY": "laboratory", "CODE": "x-1",
         "DESCRIPTION": "Example analyte", "VALUE": str(i + 1.0),
         "UNITS": "mg/dL", "TYPE": "numeric", "_source_row": i + 1}
        for i, encounter in enumerate(encounters)
    ]
    source = SimpleNamespace(
        profile="dev",
        rows={
            "observations.csv": observations,
            "conditions.csv": [{"START": "2020-01-01", "STOP": "", "PATIENT": patient,
                "ENCOUNTER": "enc-a", "SYSTEM": "SNOMED-CT", "CODE": "unique-c",
                "DESCRIPTION": "Unique condition", "_source_row": 1}],
            "procedures.csv": [], "medications.csv": [],
        },
    )
    mappings = SimpleNamespace(
        patients={"patient-a": 80000001, "patient-b": 80000002},
        encounters={e: 800000001 + i for i, e in enumerate(encounters)},
    )
    notes = SimpleNamespace(rows=[
        {"encounter_uuid": e, "note_id": 850000001 + i,
         "subject_id": 80000001, "hadm_id": 800000001 + i,
         "text": "Encounter Summary\nDescription=Example analyte\n"}
        for i, e in enumerate(encounters)
    ])
    return source, mappings, notes


def test_temporal_gold_uses_source_timestamps_and_is_deterministic():
    source, mappings, notes = _fixture()
    first = S._temporal_candidates(source, mappings, notes)
    second = S._temporal_candidates(source, mappings, notes)
    assert first == second
    temporal = first["temporal"]
    assert any(c["temporal"] == "earliest" and "2020-01-01" in c["expected_answer"]
               for c in temporal)
    assert any(c["temporal"] == "latest" and "2022-01-03" in c["expected_answer"]
               for c in temporal)
    trend = first["longitudinal"][0]
    assert [p["source_row"] for p in trend["evidence_provenance"]] == [1, 2, 3]
    assert len(trend["evidence_note_ids"]) == 3


def test_isolation_and_abstention_targets_lack_the_witness_fact():
    source, mappings, notes = _fixture()
    cases = S._negative_candidates(source, mappings, notes)
    generated = cases["patient_isolation"] + cases["abstention"]
    assert generated
    for case in generated:
        assert case["subject_id"] == mappings.patients["patient-b"]
        assert case["unsupported"] is True
        assert case["evidence_note_ids"] == []
        assert case["must_not_contain"]
        assert any(
            witness in case["must_not_contain"]
            for witness in ("Unique condition", "Example analyte", "x-1")
            )

def test_selection_has_stable_ids_and_hash():
    source, mappings, notes = _fixture()
    pools = S._temporal_candidates(source, mappings, notes)
    a = S._select("dev", pools)
    b = S._select("dev", pools)
    assert a == b
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert S._canonical_hash(a) == S._canonical_hash(b)


def test_dev_artifacts_cannot_be_opened_as_eval(tmp_path):
    out = tmp_path / "derived" / "evaluation"
    out.mkdir(parents=True)
    cases = [{"id": "synthea_dev_x_001", "gold_profile": "dev"}]
    gold = out / "golden_qa.json"
    gold.write_text(json.dumps(cases))
    identity = {"golden_qa_sha256": S.sha256_file(gold), "case_count": 1,
                "case_ids": ["synthea_dev_x_001"]}
    (out / "evaluation_manifest.json").write_text(json.dumps({
        "provider": "synthea", "profile": "dev", "identity": identity,
        "golden_set_fingerprint": S._canonical_hash(identity),
    }))
    with pytest.raises(RuntimeError, match="provider/profile mismatch"):
        S.validate_artifacts("eval", tmp_path)


def test_synthea_preflight_rejects_wrong_plane_database_and_profile(monkeypatch):
    class FakeProvider:
        name = "synthea"
        profile = "dev"
        data_plane = "synthea"
        expected_database = "lumen_synthea"

        def fingerprint(self):
            return {"profile": "dev", "source_manifest_sha256": "s",
                    "mapping_manifest_sha256": "m", "note_corpus_sha256": "n",
                    "index_configuration_hash": "i"}

    monkeypatch.setenv("LUMEN_DATA_PLANE", "research")
    checks = doctor.check_data_plane(case_provider=FakeProvider(), probe=lambda: {
        "connected": True, "database": "lumen", "counts": {"note_chunks": 1, "labevents": 1},
        "corpus_identity": {"profile": "eval", "source_manifest_sha256": "wrong"},
    })
    by_name = {check.name: check for check in checks}
    assert by_name["data_plane_matches_provider"].status == doctor.FAIL
    assert by_name["database_matches_provider"].status == doctor.FAIL
    assert by_name["synthea_gold_matches_corpus"].status == doctor.FAIL


def test_synthea_provider_refuses_unknown_profile():
    with pytest.raises(ValueError):
        providers.get_provider("synthea", "research")
