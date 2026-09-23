"""Deterministic Synthea golden-case construction and artifact validation.

Ground truth comes only from the frozen CSV source and deterministic mappings.
No retrieval result, database row, model, or generated answer participates in
case selection or expected-answer construction.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.retrieval.index_provenance import configuration_hash, index_configuration
from src.storage.ingest_synthea import (
    MAPPING_VERSION, NOTE_RENDERER_VERSION, SUBJECT_BASE, build_mappings, prepare_notes,
    sha256_file, validate_source,
)

ROOT = Path(__file__).resolve().parents[3]
GENERATOR_VERSION = "synthea-golden-v1"
PROFILES = ("dev", "eval")
TARGETS = {"dev": 28, "eval": 100}
QUOTAS = {
    "dev": {
        "condition": 4, "procedure": 4, "medication": 3,
        "observation": 4, "temporal": 4, "longitudinal": 2,
        "recent_encounter": 2, "structured_narrative": 2,
        "patient_isolation": 2, "abstention": 2, "long_note": 1,
    },
    "eval": {
        "condition": 10, "procedure": 10, "medication": 10,
        "observation": 15, "temporal": 15, "longitudinal": 10,
        "recent_encounter": 5, "structured_narrative": 5,
        "patient_isolation": 5, "abstention": 10, "long_note": 5,
    },
}


def artifact_dir(profile: str, root: Path | None = None) -> Path:
    _profile(profile)
    base = root or ROOT / "data" / "synthea" / profile
    return Path(base) / "derived" / "evaluation"


def artifact_paths(profile: str, root: Path | None = None) -> tuple[Path, Path]:
    out = artifact_dir(profile, root)
    return out / "golden_qa.json", out / "evaluation_manifest.json"


def _profile(profile: str) -> None:
    if profile not in PROFILES:
        raise ValueError(f"unknown Synthea profile {profile!r}; expected dev or eval")


def _canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _norm(value: str) -> str:
    return " ".join((value or "").lower().split())


def _date(value: str) -> str:
    return (value or "")[:10]


def _note_maps(notes) -> tuple[dict[str, dict], dict[int, dict]]:
    by_encounter = {row["encounter_uuid"]: row for row in notes.rows}
    by_id = {row["note_id"]: row for row in notes.rows}
    return by_encounter, by_id


def _provenance(filename: str, row: dict) -> dict:
    timestamp = row.get("DATE") or row.get("START") or row.get("STOP")
    return {
        "source_file": filename,
        "source_row": row.get("_source_row"),
        "source_timestamp": timestamp or None,
        "source_patient_id": row.get("PATIENT") or None,
        "source_encounter_id": row.get("ENCOUNTER") or row.get("Id") or None,
    }


def _base(source, mappings, note_by_encounter, row, *, category: str, question: str,
          facts: list[str], answer: str, source_file: str, temporal: str = "all",
          answer_type: str = "exact_fact", unsupported: bool = False,
          must_not: list[str] | None = None, provenance: list[dict] | None = None) -> dict:
    patient_uuid = row["PATIENT"]
    encounter_uuid = row.get("ENCOUNTER") or row.get("Id")
    note = note_by_encounter.get(encounter_uuid)
    return {
        "subject_id": mappings.patients[patient_uuid],
        "query": question,
        "category": category,
        "answer_type": answer_type,
        "difficulty": "hard" if category in {"longitudinal", "long_note"} else "medium",
        "temporal": temporal,
        "expected_facts": facts,
        "min_facts": 0 if unsupported else max(1, len(facts)),
        "expected_answer": answer,
        "unsupported": unsupported,
        "must_not_contain": must_not or [],
        "evidence_hadm_ids": [mappings.encounters[encounter_uuid]] if note else [],
        "evidence_note_ids": [note["note_id"]] if note else [],
        "evidence_note_types": ["encounter_summary"] if note else [],
        "evidence_provenance": provenance or [_provenance(source_file, row)],
        "gold_profile": source.profile,
        "_patient_uuid": patient_uuid,
        "_stable": (source_file, row.get("_source_row", 0), question),
    }


def _linked(rows: list[dict], note_by_encounter: dict) -> list[dict]:
    return [r for r in rows if r.get("PATIENT") and r.get("ENCOUNTER") in note_by_encounter]


def _fact(value: str, code: str = "") -> str:
    return f"{value} ({code})" if code else value


def _simple_candidates(source, mappings, notes) -> dict[str, list[dict]]:
    by_encounter, _ = _note_maps(notes)
    out: dict[str, list[dict]] = defaultdict(list)

    conditions = _linked(source.rows["conditions.csv"], by_encounter)
    for system in ("ICD10", "SNOMED-CT"):
        for row in conditions:
            if row["SYSTEM"] != system or not row["DESCRIPTION"]:
                continue
            fact = _fact(row["DESCRIPTION"], row["CODE"])
            out["condition"].append(_base(
                source, mappings, by_encounter, row, category="condition",
                question=f"What condition was documented during this encounter ({row['CODE']})?",
                facts=[fact], answer=f"The record documents {fact} in {system}.",
                source_file="conditions.csv", answer_type="condition_history"))
            out["condition"][-1]["_variant"] = system

    procedures = _linked(source.rows["procedures.csv"], by_encounter)
    for system in ("SNOMED-CT", "CDT"):
        for row in procedures:
            if row["SYSTEM"] != system or not row["DESCRIPTION"]:
                continue
            fact = _fact(row["DESCRIPTION"], row["CODE"])
            out["procedure"].append(_base(
                source, mappings, by_encounter, row, category="procedure",
                question=f"Which procedure was recorded during this encounter ({row['CODE']})?",
                facts=[fact], answer=f"The documented procedure was {fact} ({system}).",
                source_file="procedures.csv", answer_type="procedure_fact"))
            out["procedure"][-1]["_variant"] = system

    for row in _linked(source.rows["medications.csv"], by_encounter):
        if row["DESCRIPTION"]:
            fact = _fact(row["DESCRIPTION"], row["CODE"])
            out["medication"].append(_base(
                source, mappings, by_encounter, row, category="medication",
                question=f"What medication was recorded during this encounter ({row['CODE']})?",
                facts=[fact], answer=f"The recorded medication was {fact}.",
                source_file="medications.csv", answer_type="medication_history"))

    observations = _linked(source.rows["observations.csv"], by_encounter)
    for kind in ("numeric", "text"):
        for row in observations:
            if row["TYPE"] != kind or not row["DESCRIPTION"] or not row["VALUE"]:
                continue
            value = f"{row['VALUE']} {row['UNITS']}".strip()
            fact = f"{row['DESCRIPTION']} {value}"
            out["observation"].append(_base(
                source, mappings, by_encounter, row, category="observation",
                question=f"What value was recorded for {row['DESCRIPTION']} during the encounter?",
                facts=[fact], answer=f"{row['DESCRIPTION']} was recorded as {value}.",
                source_file="observations.csv", answer_type="observation_value"))
            out["observation"][-1]["_variant"] = kind

    # Narrative complements: these facts are deliberately excluded from the
    # ICD-only canonical tables but are valid in encounter_summary notes.
    complement = ([r for r in conditions if r["SYSTEM"] != "ICD10"]
                  + [r for r in procedures if r["SYSTEM"] in {"SNOMED-CT", "CDT"}])
    for row in complement:
        filename = "conditions.csv" if "conditions" in source.rows and row in conditions else "procedures.csv"
        fact = _fact(row["DESCRIPTION"], row["CODE"])
        out["structured_narrative"].append(_base(
            source, mappings, by_encounter, row, category="structured_narrative",
            question=f"What non-ICD narrative fact with code {row['CODE']} appears in the encounter note?",
            facts=[fact], answer=f"The encounter note documents {fact} ({row['SYSTEM']}).",
            source_file=filename, answer_type="narrative_complement"))
        out["structured_narrative"][-1]["_variant"] = row["SYSTEM"]
    return out


def _temporal_candidates(source, mappings, notes) -> dict[str, list[dict]]:
    by_encounter, _ = _note_maps(notes)
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in _linked(source.rows["observations.csv"], by_encounter):
        if row["TYPE"] == "numeric" and row["VALUE"] and row["DESCRIPTION"]:
            groups[(row["PATIENT"], row["DESCRIPTION"], row["UNITS"])].append(row)
    out: dict[str, list[dict]] = defaultdict(list)
    for key, rows in sorted(groups.items()):
        by_hadm = {}
        for row in rows:
            old = by_hadm.get(row["ENCOUNTER"])
            if old is None or (row["DATE"], row["_source_row"]) > (old["DATE"], old["_source_row"]):
                by_hadm[row["ENCOUNTER"]] = row
        ordered = sorted(by_hadm.values(), key=lambda r: (r["DATE"], r["ENCOUNTER"], r["_source_row"]))
        if len(ordered) < 3:
            continue
        for label, row in (("earliest", ordered[0]), ("latest", ordered[-1])):
            value = f"{row['VALUE']} {row['UNITS']}".strip()
            fact = f"{row['DESCRIPTION']} {value} on {_date(row['DATE'])}"
            out["temporal"].append(_base(
                source, mappings, by_encounter, row, category="temporal",
                question=f"What was the {label} recorded {row['DESCRIPTION']} value?",
                facts=[fact], answer=f"The {label} recorded value was {value} on {_date(row['DATE'])}.",
                source_file="observations.csv", temporal=label, answer_type=f"{label}_value"))
            out["temporal"][-1]["_variant"] = label

        sample = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
        facts = [f"{r['DESCRIPTION']} {r['VALUE']} {r['UNITS']} on {_date(r['DATE'])}".strip()
                 for r in sample]
        first = sample[0]
        case = _base(
            source, mappings, by_encounter, first, category="longitudinal",
            question=f"How did {first['DESCRIPTION']} change across the available encounters?",
            facts=facts, answer="; ".join(facts) + ".", source_file="observations.csv",
            temporal="trend", answer_type="trend",
            provenance=[_provenance("observations.csv", r) for r in sample])
        case["evidence_hadm_ids"] = [mappings.encounters[r["ENCOUNTER"]] for r in sample]
        case["evidence_note_ids"] = [by_encounter[r["ENCOUNTER"]]["note_id"] for r in sample]
        out["longitudinal"].append(case)
    return out


def _recent_candidates(source, mappings, notes) -> list[dict]:
    by_encounter, _ = _note_maps(notes)
    grouped = defaultdict(list)
    for row in source.rows["encounters.csv"]:
        if row["Id"] in by_encounter:
            grouped[row["PATIENT"]].append(row)
    out = []
    for rows in grouped.values():
        row = max(rows, key=lambda r: (r["STOP"], r["START"], r["Id"]))
        fact = f"{row['DESCRIPTION']} on {_date(row['STOP'] or row['START'])}"
        out.append(_base(
            source, mappings, by_encounter, row, category="recent_encounter",
            question="What was the patient's most recent documented encounter?",
            facts=[fact], answer=f"The most recent encounter was {fact}.",
            source_file="encounters.csv", temporal="latest", answer_type="recent_encounter"))
    return out


def _negative_candidates(source, mappings, notes) -> dict[str, list[dict]]:
    by_encounter, _ = _note_maps(notes)
    all_rows = []
    for filename in ("conditions.csv", "procedures.csv", "medications.csv", "observations.csv"):
        all_rows += [(filename, r) for r in _linked(source.rows[filename], by_encounter)
                     if r.get("DESCRIPTION") and r.get("CODE")]
    owners = defaultdict(set)
    for _, row in all_rows:
        owners[(_norm(row["DESCRIPTION"]), row["CODE"])].add(row["PATIENT"])
    unique = [(f, r) for f, r in all_rows
              if len(owners[(_norm(r["DESCRIPTION"]), r["CODE"])]) == 1]
    patients = sorted(mappings.patients)
    out = {"patient_isolation": [], "abstention": []}
    for index, (filename, owner) in enumerate(sorted(unique, key=lambda x: (
            x[1]["PATIENT"], x[0], x[1]["CODE"], x[1]["_source_row"]))):
        target = next((p for p in patients if p != owner["PATIENT"]), None)
        if target is None:
            continue
        witness = dict(owner)
        witness["PATIENT"] = target
        # No encounter is evidence for an absent fact.
        witness["ENCOUNTER"] = ""
        category = "patient_isolation" if index % 2 == 0 else "abstention"
        question = (f"Was {owner['DESCRIPTION']} ({owner['CODE']}) documented for this patient?"
                    if category == "patient_isolation" else
                    f"What recorded evidence shows {owner['DESCRIPTION']} ({owner['CODE']}) for this patient?")
        case = _base(
            source, mappings, by_encounter, witness, category=category,
            question=question, facts=[],
            answer="The available records do not contain enough information.",
            source_file=filename, unsupported=True,
            must_not=[owner["DESCRIPTION"], owner["CODE"]],
            provenance=[{**_provenance(filename, owner), "role": "other_patient_witness"}])
        case["evidence_hadm_ids"] = []
        case["evidence_note_ids"] = []
        case["evidence_note_types"] = []
        out[category].append(case)
    return out


def _long_note_candidates(source, mappings, notes) -> list[dict]:
    by_encounter, _ = _note_maps(notes)
    observations = defaultdict(list)
    for row in _linked(source.rows["observations.csv"], by_encounter):
        observations[row["ENCOUNTER"]].append(row)
    out = []
    for note in sorted(notes.rows, key=lambda n: (-len(n["text"]), n["note_id"])):
        text = note["text"]
        for row in sorted(observations[note["encounter_uuid"]], key=lambda r: r["_source_row"], reverse=True):
            marker = f"Description={row['DESCRIPTION']}"
            position = text.rfind(marker)
            if position < int(len(text) * 0.60) or not row["VALUE"]:
                continue
            value = f"{row['VALUE']} {row['UNITS']}".strip()
            case = _base(
                source, mappings, by_encounter, row, category="long_note",
                question=f"What value is documented for {row['DESCRIPTION']} in the detailed encounter record?",
                facts=[f"{row['DESCRIPTION']} {value}"],
                answer=f"{row['DESCRIPTION']} was recorded as {value}.",
                source_file="observations.csv", answer_type="deep_note_fact")
            case["deep_note_char_offset"] = position
            case["note_text_chars"] = len(text)
            out.append(case)
            break
    return out


def _select(profile: str, pools: dict[str, list[dict]]) -> list[dict]:
    usage = Counter()
    cap = 4 if profile == "dev" else 2
    selected = []
    for category, quota in QUOTAS[profile].items():
        candidates = sorted(pools.get(category, []), key=lambda c: c["_stable"])
        chosen = []
        variants = Counter()
        while len(chosen) < quota:
            eligible = [c for c in candidates if c not in chosen
                        and usage[c["subject_id"]] < cap]
            if not eligible:
                break
            candidate = min(eligible, key=lambda c: (
                variants[c.get("_variant", "default")], c["_stable"]))
            chosen.append(candidate)
            usage[candidate["subject_id"]] += 1
            variants[candidate.get("_variant", "default")] += 1
        if len(chosen) < quota:
            for candidate in candidates:
                if candidate in chosen:
                    continue
                chosen.append(candidate)
                usage[candidate["subject_id"]] += 1
                if len(chosen) == quota:
                    break
        selected.extend(chosen)

    counters = Counter()
    clean = []
    for case in selected:
        category = case["category"]
        counters[category] += 1
        item = {k: v for k, v in case.items() if not k.startswith("_")}
        item["id"] = f"synthea_{profile}_{category}_{counters[category]:03d}"
        clean.append(item)
    return clean


def build(profile: str, dataset_root: Path | None = None) -> dict:
    _profile(profile)
    root = Path(dataset_root or ROOT / "data" / "synthea" / profile)
    source = validate_source(root, profile)
    mappings = build_mappings(source)
    notes = prepare_notes(source, mappings)

    pools = _simple_candidates(source, mappings, notes)
    temporal = _temporal_candidates(source, mappings, notes)
    pools["temporal"].extend(temporal["temporal"])
    pools["longitudinal"].extend(temporal["longitudinal"])
    pools["recent_encounter"].extend(_recent_candidates(source, mappings, notes))
    for name, values in _negative_candidates(source, mappings, notes).items():
        pools[name].extend(values)
    pools["long_note"].extend(_long_note_candidates(source, mappings, notes))

    cases = _select(profile, pools)
    minimum = 25 if profile == "dev" else 80
    if len(cases) < minimum:
        raise RuntimeError(
            f"only {len(cases)} eligible cases for {profile}; target is {TARGETS[profile]}"
        )
    golden_path, manifest_path = artifact_paths(profile, root)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_text = json.dumps(cases, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    golden_path.write_text(golden_text, encoding="utf-8")
    golden_sha = hashlib.sha256(golden_text.encode("utf-8")).hexdigest()

    generation_manifest = source.manifest
    source_dataset_fp = generation_manifest.get("dataset_fingerprint") or _canonical_hash({
        k: generation_manifest.get(k) for k in (
            "profile", "synthea", "population_seed", "clinician_seed",
            "requested_population", "actual_patient_count", "generation_parameters",
            "config_sha256", "csv_files",
        )
    })
    category_counts = dict(sorted(Counter(c["category"] for c in cases).items()))
    identity = {
        "generator_version": GENERATOR_VERSION,
        "profile": profile,
        "source_manifest_sha256": source.manifest_sha256,
        "source_dataset_fingerprint": source_dataset_fp,
        "mapping_manifest_sha256": mappings.manifest_sha256,
        "mapping_algorithm": MAPPING_VERSION,
        "note_renderer_version": NOTE_RENDERER_VERSION,
        "note_corpus_sha256": notes.corpus_sha256,
        "index_configuration": index_configuration(),
        "index_configuration_hash": configuration_hash(),
        "golden_qa_sha256": golden_sha,
        "case_count": len(cases),
        "category_counts": category_counts,
        "unique_evaluation_patients": len({c["subject_id"] for c in cases}),
        "case_ids": [c["id"] for c in cases],
    }
    manifest = {
        "schema_version": 1,
        "synthetic": True,
        "provider": "synthea",
        "profile": profile,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "golden_file": golden_path.name,
        "identity": identity,
        "golden_set_fingerprint": _canonical_hash(identity),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_artifacts(profile: str, dataset_root: Path | None = None) -> dict:
    _profile(profile)
    root = Path(dataset_root or ROOT / "data" / "synthea" / profile)
    golden_path, manifest_path = artifact_paths(profile, root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = json.loads(golden_path.read_text(encoding="utf-8"))
    if manifest.get("provider") != "synthea" or manifest.get("profile") != profile:
        raise RuntimeError("Synthea evaluation manifest provider/profile mismatch")
    identity = manifest.get("identity") or {}
    if manifest.get("golden_set_fingerprint") != _canonical_hash(identity):
        raise RuntimeError("Synthea golden-set fingerprint mismatch")
    if identity.get("golden_qa_sha256") != sha256_file(golden_path):
        raise RuntimeError("Synthea golden file hash mismatch")
    if identity.get("case_count") != len(cases):
        raise RuntimeError("Synthea golden case-count mismatch")
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)) or ids != identity.get("case_ids"):
        raise RuntimeError("Synthea golden case IDs are duplicate or unstable")
    source_manifest = root / "generation_manifest.json"
    mapping_manifest = root / "derived" / "mapping_manifest.json"
    if identity.get("source_manifest_sha256") != sha256_file(source_manifest):
        raise RuntimeError("Synthea source/gold fingerprint mismatch")
    if identity.get("mapping_manifest_sha256") != sha256_file(mapping_manifest):
        raise RuntimeError("Synthea mapping/gold fingerprint mismatch")
    if any(case.get("gold_profile") != profile for case in cases):
        raise RuntimeError("Synthea golden cases mix DEV/EVAL profiles")
    source = validate_source(root, profile)
    rows_by_number = {
        filename: {row["_source_row"]: row for row in rows}
        for filename, rows in source.rows.items()
    }
    for case in cases:
        provenance = case.get("evidence_provenance") or []
        if not provenance:
            raise RuntimeError(f"{case.get('id')}: missing source provenance")
        for item in provenance:
            filename, row_number = item.get("source_file"), item.get("source_row")
            if row_number not in rows_by_number.get(filename, {}):
                raise RuntimeError(
                    f"{case.get('id')}: missing {filename} source row {row_number}"
                )
        if not case.get("unsupported") and not case.get("evidence_note_ids"):
            raise RuntimeError(f"{case.get('id')}: positive case has no expected note")

    # Negative cases are valid only while every forbidden fact is absent from
    # the target patient's source rows. This is recomputed, not trusted from
    # the generated JSON.
    patient_ids = sorted(row["Id"] for row in source.rows["patients.csv"])
    facts_by_patient = defaultdict(set)
    for filename in ("conditions.csv", "procedures.csv", "medications.csv", "observations.csv"):
        for row in source.rows[filename]:
            facts_by_patient[row["PATIENT"]].update({_norm(row.get("DESCRIPTION", "")),
                                                       _norm(row.get("CODE", ""))})
    for case in cases:
        if not case.get("unsupported"):
            continue
        position = int(case["subject_id"]) - SUBJECT_BASE - 1
        if position < 0 or position >= len(patient_ids):
            raise RuntimeError(f"{case.get('id')}: invalid target subject mapping")
        target_facts = facts_by_patient[patient_ids[position]]
        leaks = [fact for fact in case.get("must_not_contain", []) if _norm(fact) in target_facts]
        if leaks:
            raise RuntimeError(f"{case.get('id')}: negative target contains {leaks}")
    return {**identity, "golden_set_fingerprint": manifest["golden_set_fingerprint"],"path": str(golden_path), 
            "sha256": sha256_file(golden_path),
            "manifest_path": str(manifest_path), "manifest_sha256_matches": True,
            "dataset_version": GENERATOR_VERSION, "n_cases": len(cases),
            "provider": "synthea", "provider_data_plane": "synthea"}
