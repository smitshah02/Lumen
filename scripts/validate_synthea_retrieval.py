#!/usr/bin/env python3
"""Deterministic Phase 6 retrieval validation for the frozen Synthea DEV corpus.

This is a thin harness over the canonical Lumen retrieval functions. It does
not implement retrieval, fusion, embeddings, reranking, or temporal ranking.
Cases are selected deterministically from source CSV rows and checked against
their generated encounter notes with no LLM judge.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEV = ROOT / "data" / "synthea" / "dev"
TOP_K = 5
LONG_NOTE_ID = 850_003_022
MODES = ("lexical", "vector", "hybrid", "hybrid_rerank")


def _rows(name: str) -> list[dict]:
    with (DEV / "csv" / name).open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row, _source_row=i) for i, row in enumerate(csv.DictReader(handle), 1)]


def _mapping(name: str, key: str, value: str) -> dict[str, int]:
    with (DEV / "derived" / name).open(newline="", encoding="utf-8") as handle:
        return {row[key]: int(row[value]) for row in csv.DictReader(handle)}


def _norm(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _condition_fact(row: dict) -> str:
    return f"System={row['SYSTEM']} | Code={row['CODE']} | Description={row['DESCRIPTION']}"


def _procedure_fact(row: dict) -> str:
    return f"System={row['SYSTEM']} | Code={row['CODE']} | Description={row['DESCRIPTION']}"


def _medication_fact(row: dict) -> str:
    return f"Code={row['CODE']} | Description={row['DESCRIPTION']}"


def _observation_fact(row: dict, *, description_only: bool = False) -> str:
    if description_only:
        return f"Description={row['DESCRIPTION']}"
    return (
        f"Code={row['CODE']} | Description={row['DESCRIPTION']} | "
        f"Value={row['VALUE']}"
    )


def _query_for(row: dict, category: str) -> str:
    if category == "observation":
        return " ".join(x for x in (row["DESCRIPTION"], row["VALUE"], row["UNITS"]) if x)
    return " ".join(x for x in (row["DESCRIPTION"], row.get("CODE", "")) if x)


def _load_source() -> dict:
    encounters = _rows("encounters.csv")
    encounter_ids = sorted(row["Id"] for row in encounters)
    return {
        "patients": _rows("patients.csv"),
        "encounters": encounters,
        "encounter_by_id": {row["Id"]: row for row in encounters},
        "conditions": _rows("conditions.csv"),
        "procedures": _rows("procedures.csv"),
        "medications": _rows("medications.csv"),
        "observations": _rows("observations.csv"),
        "patient_ids": _mapping("patient_ids.csv", "source_patient_id", "subject_id"),
        "encounter_ids": _mapping("encounter_ids.csv", "source_encounter_id", "hadm_id"),
        "note_ids": {encounter_id: 850_000_000 + i
                     for i, encounter_id in enumerate(encounter_ids, 1)},
    }


def _chunk_matches(conn, note_id: int, fact: str, min_index: int = 0) -> list[dict]:
    from sqlalchemy import text

    rows = conn.execute(text("""
        SELECT chunk_id, chunk_index, chunk_text, token_count
        FROM note_chunks
        WHERE note_id=:note_id AND token_count >= 40 AND chunk_index >= :min_index
        ORDER BY chunk_index, chunk_id
    """), {"note_id": note_id, "min_index": min_index}).mappings().all()
    needle = _norm(fact)
    return [dict(row) for row in rows if needle in _norm(row["chunk_text"])]


def _case_from_row(
    source: dict,
    row: dict,
    *,
    case_id: str,
    category: str,
    query: str,
    expected_fact: str,
    source_file: str,
    temporal_requirement: str | None = None,
) -> dict:
    encounter_id = row["ENCOUNTER"]
    return {
        "case_id": case_id,
        "subject_id": source["patient_ids"][row["PATIENT"]],
        "query": query,
        "expected_note_id": source["note_ids"][encounter_id],
        "expected_note_ids": [source["note_ids"][encounter_id]],
        "expected_hadm_id": source["encounter_ids"][encounter_id],
        "expected_hadm_ids": [source["encounter_ids"][encounter_id]],
        "expected_fact": expected_fact,
        "category": category,
        "temporal_requirement": temporal_requirement,
        "source_file": source_file,
        "source_row": row["_source_row"],
    }


def _ranked_candidates(rows: list[dict]) -> list[dict]:
    frequency = Counter(_norm(row["DESCRIPTION"]) for row in rows)
    return sorted(rows, key=lambda row: (
        frequency[_norm(row["DESCRIPTION"])],
        -len(row["DESCRIPTION"]),
        row["DESCRIPTION"],
        row.get("ENCOUNTER", ""),
        row["_source_row"],
    ))


def _pick_source_case(
    conn,
    source: dict,
    rows: list[dict],
    *,
    case_id: str,
    category: str,
    source_file: str,
    fact_fn,
    used_notes: set[int],
) -> dict:
    for row in _ranked_candidates(rows):
        encounter_id = row.get("ENCOUNTER", "")
        if not encounter_id or encounter_id not in source["note_ids"]:
            continue
        note_id = source["note_ids"][encounter_id]
        fact = fact_fn(row)
        if note_id in used_notes or not _chunk_matches(conn, note_id, fact):
            continue
        used_notes.add(note_id)
        return _case_from_row(
            source, row, case_id=case_id, category=category,
            query=_query_for(row, category), expected_fact=fact,
            source_file=source_file,
        )
    raise RuntimeError(f"could not derive searchable case {case_id}")


def _temporal_group(source: dict, conn) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in source["observations"]:
        if row["ENCOUNTER"] in source["note_ids"] and row["DESCRIPTION"]:
            groups[(row["PATIENT"], row["DESCRIPTION"])].append(row)

    candidates = []
    for (patient_id, description), rows in groups.items():
        by_encounter: dict[str, dict] = {}
        for row in rows:
            current = by_encounter.get(row["ENCOUNTER"])
            if current is None or row["DATE"] > current["DATE"]:
                by_encounter[row["ENCOUNTER"]] = row
        ordered = sorted(by_encounter.values(), key=lambda row: (
            row["DATE"], row["ENCOUNTER"], row["_source_row"],
        ))
        if len(ordered) < 3:
            continue
        stop_times = [source["encounter_by_id"][row["ENCOUNTER"]]["STOP"] for row in ordered]
        if stop_times != sorted(stop_times):
            continue
        if not all(_chunk_matches(
            conn, source["note_ids"][row["ENCOUNTER"]],
            _observation_fact(row, description_only=True),
        ) for row in (ordered[0], ordered[-1])):
            continue
        candidates.append((-len(ordered), patient_id, description, ordered))
    if not candidates:
        raise RuntimeError("no searchable longitudinal observation group found")
    return sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[0][3]


def _build_cases(conn) -> list[dict]:
    source = _load_source()
    used_notes: set[int] = set()
    cases = []

    cases.append(_pick_source_case(
        conn, source,
        [row for row in source["conditions"] if row["SYSTEM"] == "ICD10"],
        case_id="condition_icd10", category="condition", source_file="conditions.csv",
        fact_fn=_condition_fact, used_notes=used_notes,
    ))
    cases.append(_pick_source_case(
        conn, source,
        [row for row in source["conditions"] if row["SYSTEM"] == "SNOMED-CT"],
        case_id="condition_snomed", category="condition", source_file="conditions.csv",
        fact_fn=_condition_fact, used_notes=used_notes,
    ))
    for system, suffix in (("SNOMED-CT", "snomed"), ("CDT", "cdt")):
        cases.append(_pick_source_case(
            conn, source,
            [row for row in source["procedures"] if row["SYSTEM"] == system],
            case_id=f"procedure_{suffix}", category="procedure",
            source_file="procedures.csv", fact_fn=_procedure_fact, used_notes=used_notes,
        ))
    for i in range(1, 3):
        cases.append(_pick_source_case(
            conn, source, source["medications"], case_id=f"medication_{i}",
            category="medication", source_file="medications.csv",
            fact_fn=_medication_fact, used_notes=used_notes,
        ))
    observation_rows = [
        row for row in source["observations"]
        if row["VALUE"] and row["UNITS"] and row["ENCOUNTER"]
    ]
    for i in range(1, 3):
        cases.append(_pick_source_case(
            conn, source, observation_rows, case_id=f"observation_{i}",
            category="observation", source_file="observations.csv",
            fact_fn=_observation_fact, used_notes=used_notes,
        ))

    temporal = _temporal_group(source, conn)
    first, last = temporal[0], temporal[-1]
    description = first["DESCRIPTION"]
    earliest = _case_from_row(
        source, first, case_id="temporal_earliest", category="temporal",
        query=f"earliest {description} value", expected_fact=_observation_fact(first),
        source_file="observations.csv", temporal_requirement="earliest",
    )
    latest = _case_from_row(
        source, last, case_id="temporal_latest", category="temporal",
        query=f"latest {description} value", expected_fact=_observation_fact(last),
        source_file="observations.csv", temporal_requirement="latest",
    )
    trend = _case_from_row(
        source, first, case_id="temporal_trend", category="temporal",
        query=f"{description} trend over time",
        expected_fact=_observation_fact(first, description_only=True),
        source_file="observations.csv", temporal_requirement="trend",
    )
    trend["expected_note_ids"] = [
        source["note_ids"][row["ENCOUNTER"]] for row in temporal
    ]
    trend["expected_hadm_ids"] = [
        source["encounter_ids"][row["ENCOUNTER"]] for row in temporal
    ]
    cases.extend((earliest, latest, trend))

    long_encounter = next(
        encounter_id for encounter_id, note_id in source["note_ids"].items()
        if note_id == LONG_NOTE_ID
    )
    long_rows = sorted(
        [row for row in source["observations"] if row["ENCOUNTER"] == long_encounter],
        key=lambda row: (row["DATE"], row["CODE"], row["DESCRIPTION"], row["_source_row"]),
        reverse=True,
    )
    long_case = None
    for row in long_rows:
        fact = _observation_fact(row)
        matches = _chunk_matches(conn, LONG_NOTE_ID, fact, min_index=10)
        if matches:
            long_case = _case_from_row(
                source, row, case_id="long_note_deep_fact", category="long_note",
                query=_query_for(row, "observation"), expected_fact=fact,
                source_file="observations.csv",
            )
            long_case["expected_min_chunk_index"] = 10
            long_case["source_chunk_indices"] = [match["chunk_index"] for match in matches]
            break
    if long_case is None:
        raise RuntimeError("no source observation found beyond chunk 10 of the long note")
    cases.append(long_case)

    observation_description_counts = Counter(
        _norm(row["DESCRIPTION"]) for row in observation_rows
    )
    owner_row = next(
        row for row in _ranked_candidates(observation_rows)
        if row["PATIENT"] != first["PATIENT"]
        and observation_description_counts[_norm(row["DESCRIPTION"])] == 1
    )
    target_patient = sorted(
        patient_id for patient_id in source["patient_ids"] if patient_id != owner_row["PATIENT"]
    )[0]
    cases.append({
        "case_id": "patient_isolation_distractor",
        "subject_id": source["patient_ids"][target_patient],
        "query": _query_for(owner_row, "observation"),
        "expected_note_id": None,
        "expected_note_ids": [],
        "expected_hadm_id": None,
        "expected_hadm_ids": [],
        "expected_fact": _observation_fact(owner_row),
        "category": "patient_isolation",
        "temporal_requirement": None,
        "source_file": "observations.csv",
        "source_row": owner_row["_source_row"],
        "forbidden_subject_id": source["patient_ids"][owner_row["PATIENT"]],
    })

    if len(cases) != 13:
        raise RuntimeError(f"expected 13 cases, built {len(cases)}")
    return cases


def _readiness(storage) -> dict:
    from sqlalchemy import text

    actual = storage.engine.url.database
    if storage.DATA_PLANE != "synthea":
        raise RuntimeError(f"refusing LUMEN_DATA_PLANE={storage.DATA_PLANE!r}")
    if actual != storage.SYNTHEA_DB_NAME or actual in {
        storage.RESEARCH_DB_NAME, storage.DEMO_DB_NAME,
    }:
        raise RuntimeError(f"refusing non-isolated database {actual!r}")
    with storage.engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
              (SELECT COUNT(*) FROM clinical_notes) AS notes,
              (SELECT COUNT(*) FROM note_chunks) AS chunks,
              (SELECT COUNT(DISTINCT note_id) FROM note_chunks) AS indexed_notes,
              (SELECT COUNT(*) FROM note_chunks WHERE text_search IS NULL) AS missing_lexical,
              (SELECT COUNT(*) FROM note_chunks WHERE embedding IS NULL) AS missing_embeddings
        """)).mappings().one()
    result = {"plane": storage.DATA_PLANE, "database": actual, **dict(row)}
    if result != {
        "plane": "synthea", "database": storage.SYNTHEA_DB_NAME,
        "notes": 3912, "chunks": 11126, "indexed_notes": 3912,
        "missing_lexical": 0, "missing_embeddings": 0,
    }:
        raise RuntimeError(f"Synthea index is not ready: {result}")
    return result


def _to_result(row: dict, score_field: str, source_name: str):
    from src.retrieval.hybrid_retriever_v2 import RetrievalResult

    score = float(row.get(score_field, 0.0))
    return RetrievalResult(
        chunk_id=row["chunk_id"], note_id=row["note_id"],
        subject_id=row["subject_id"], hadm_id=row["hadm_id"],
        note_type=row["note_type"], chunk_index=row["chunk_index"],
        chunk_text=row["chunk_text"], token_count=row["token_count"],
        charttime=str(row["charttime"]) if row.get("charttime") else None,
        bm25_score=score if score_field == "bm25_score" else 0.0,
        vector_score=score if score_field == "vector_score" else 0.0,
        final_score=score, sources=[source_name],
    )


def _run_mode(mode: str, case: dict, embedder=None, reranker=None, cache=None) -> list:
    from src.evals.retrieval_configs import RERANK_FALLBACK_THRESHOLD
    from src.retrieval.hybrid_retriever_v2 import (
        QUERY_EXPANSION,
        apply_temporal_filter,
        bm25_search,
        deduplicate_by_note,
        detect_temporal_mode,
        expand_context,
        expand_query,
        reciprocal_rank_fusion,
        strip_temporal_intent,
        temporal_candidate_limit,
        vector_search,
    )

    query = case["query"]
    subject_id = case["subject_id"]
    cache = cache if cache is not None else {}
    temporal_mode = detect_temporal_mode(query)
    retrieval_query = strip_temporal_intent(query, temporal_mode)
    candidate_top_n = temporal_candidate_limit(60, temporal_mode, subject_id)
    expansions = expand_query(retrieval_query)[1] if QUERY_EXPANSION else []

    bm25_rows = []
    vector_rows = []
    if mode in ("lexical", "hybrid", "hybrid_rerank"):
        key = ("bm25", subject_id, query)
        if key not in cache:
            cache[key] = bm25_search(
                query=retrieval_query, expansions=expansions, subject_id=subject_id,
                note_type="encounter_summary", top_n=candidate_top_n, min_tokens=40,
            )
        bm25_rows = cache[key]
    if mode in ("vector", "hybrid", "hybrid_rerank"):
        key = ("vector", subject_id, query)
        if key not in cache:
            cache[key] = vector_search(
                query_embedding=embedder.embed_query(retrieval_query), subject_id=subject_id,
                note_type="encounter_summary", top_n=candidate_top_n, min_tokens=40,
            )
        vector_rows = cache[key]

    if mode == "lexical":
        results = [_to_result(row, "bm25_score", "bm25") for row in bm25_rows]
        return apply_temporal_filter(
            results, mode=temporal_mode, score_attr="final_score"
        )[:TOP_K]
    if mode == "vector":
        results = [_to_result(row, "vector_score", "vector") for row in vector_rows]
        return apply_temporal_filter(
            results, mode=temporal_mode, score_attr="final_score"
        )[:TOP_K]

    merged = reciprocal_rank_fusion(bm25_rows, vector_rows)
    merged = deduplicate_by_note(merged, max_per_note=2)
    merged = apply_temporal_filter(merged, mode=temporal_mode)
    candidates = expand_context(merged[:40], window=1, max_context_tokens=600)

    if mode == "hybrid_rerank" and candidates:
        reranked = reranker.rerank(
            retrieval_query, candidates, top_k=len(candidates)
        )
        if max((result.rerank_score for result in reranked), default=0.0) < RERANK_FALLBACK_THRESHOLD:
            for result in candidates:
                result.final_score = result.rrf_score
            ordered = sorted(candidates, key=lambda result: result.rrf_score, reverse=True)
        else:
            ordered = reranked
    else:
        for result in candidates:
            result.final_score = result.rrf_score
        ordered = candidates
    return apply_temporal_filter(
        ordered, mode=temporal_mode, score_attr="final_score"
    )[:TOP_K]


def _result_text(result) -> str:
    return result.context_text or result.chunk_text


def _relevant(case: dict, result) -> bool:
    return (
        result.subject_id == case["subject_id"]
        and result.note_id in case["expected_note_ids"]
        and result.hadm_id in case["expected_hadm_ids"]
        and _norm(case["expected_fact"]) in _norm(_result_text(result))
    )


def _parse_time(value) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value).replace("Z", ""))


def _score_case(case: dict, results: list) -> dict:
    cross_patient = [result.chunk_id for result in results if result.subject_id != case["subject_id"]]
    if case["category"] == "patient_isolation":
        leaked_fact = [
            result.chunk_id for result in results
            if _norm(case["expected_fact"]) in _norm(_result_text(result))
        ]
        return {
            "case_id": case["case_id"], "category": case["category"],
            "cross_patient_violations": len(cross_patient),
            "forbidden_fact_matches": len(leaked_fact),
            "pass": not cross_patient and not leaked_fact,
        }

    ranks = [i for i, result in enumerate(results, 1) if _relevant(case, result)]
    rank = min(ranks, default=None)
    scored = {
        "case_id": case["case_id"], "category": case["category"],
        "rank": rank, "hit@1": rank == 1, "hit@5": rank is not None,
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "cross_patient_violations": len(cross_patient),
    }
    requirement = case.get("temporal_requirement")
    if requirement in ("latest", "earliest"):
        scored["temporal_pass"] = rank == 1
    elif requirement == "trend":
        relevant_results = [result for result in results if _relevant(case, result)]
        times = [_parse_time(result.charttime) for result in relevant_results]
        times = [value for value in times if value is not None]
        scored["temporal_pass"] = len(times) >= 2 and times == sorted(times)
        scored["temporal_timepoints"] = len(set(times))
    if case["category"] == "long_note":
        deep = [
            result.chunk_index for result in results
            if _relevant(case, result)
            and result.chunk_index >= case["expected_min_chunk_index"]
        ]
        scored["deep_chunk_indices"] = deep
        scored["long_note_pass"] = bool(deep)
    return scored


def _aggregate(mode: str, scores: list[dict]) -> dict:
    retrieval = [score for score in scores if score["category"] != "patient_isolation"]
    isolation = [score for score in scores if score["category"] == "patient_isolation"]
    temporal = [score for score in retrieval if "temporal_pass" in score]
    long_note = next(score for score in retrieval if score["category"] == "long_note")
    return {
        "mode": mode,
        "cases_scored": len(retrieval),
        "hit@1": sum(score["hit@1"] for score in retrieval) / len(retrieval),
        "hit@5": sum(score["hit@5"] for score in retrieval) / len(retrieval),
        "mrr": sum(score["reciprocal_rank"] for score in retrieval) / len(retrieval),
        "cross_patient_violations": sum(score["cross_patient_violations"] for score in scores),
        "patient_isolation_pass": all(score["pass"] for score in isolation),
        "temporal_passes": sum(score["temporal_pass"] for score in temporal),
        "temporal_cases": len(temporal),
        "long_note_pass": long_note.get("long_note_pass", False),
        "cases": scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--check-ready", action="store_true")
    parser.add_argument("--print-cases", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if os.environ.get("LUMEN_DATA_PLANE", "research").strip().lower() != "synthea":
        raise SystemExit("refusing: set LUMEN_DATA_PLANE=synthea")

    import src  # noqa: F401 - load repository .env after explicit environment
    from src import storage

    readiness = _readiness(storage)
    if args.check_ready and not args.print_cases and args.mode == "all" and args.output is None:
        print(json.dumps({"readiness": readiness}, indent=2, sort_keys=True))
        return 0

    with storage.engine.connect() as conn:
        cases = _build_cases(conn)
    if args.print_cases:
        print(json.dumps({"readiness": readiness, "cases": cases}, indent=2, sort_keys=True))
        return 0

    selected_modes = MODES if args.mode == "all" else (args.mode,)
    embedder = None
    reranker = None
    if any(mode != "lexical" for mode in selected_modes):
        from src.retrieval.embeddings import MedCPTEmbedder
        embedder = MedCPTEmbedder()
    if "hybrid_rerank" in selected_modes:
        from src.retrieval.hybrid_retriever_v2 import BGEReranker
        reranker = BGEReranker()

    summaries = []
    candidate_cache = {}
    for mode in selected_modes:
        scores = [
            _score_case(
                case,
                _run_mode(mode, case, embedder, reranker, candidate_cache),
            )
            for case in cases
        ]
        summaries.append(_aggregate(mode, scores))

    report = {
        "phase": 6,
        "dataset": "synthea-dev",
        "top_k": TOP_K,
        "readiness": readiness,
        "case_count": len(cases),
        "cases": cases,
        "modes": summaries,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    isolation_failed = any(
        summary["cross_patient_violations"] or not summary["patient_isolation_pass"]
        for summary in summaries
    )
    return 1 if isolation_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
