"""Pure regression tests for the production temporal retrieval logic.

These cases were promoted from the former ``src.retrieval.temporal_fix``
manual harness. They use synthetic shifted dates and require no database,
models, network, or patient data.
"""

from __future__ import annotations

import pytest

from src.retrieval.hybrid_retriever_v2 import (
    RetrievalResult,
    apply_temporal_filter,
    detect_temporal_mode,
)


def _result(chunk_id: int, subject_id: int, charttime, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        note_id=chunk_id,
        subject_id=subject_id,
        hadm_id=None,
        note_type="discharge",
        chunk_index=0,
        chunk_text="",
        token_count=0,
        charttime=charttime,
        rrf_score=score,
    )


def test_recent_window_uses_the_patients_own_latest_record():
    records = [
        _result(1, 100, "2150-12-01 09:00:00", 0.90),
        _result(2, 100, "2150-01-01 09:00:00", 0.85),
    ]
    assert [r.chunk_id for r in apply_temporal_filter(records, mode="recent", recency_days=365)] == [1, 2]

    records = [
        _result(1, 100, "2150-12-01 09:00:00", 0.90),
        _result(2, 100, "2150-01-01 09:00:00", 0.85),
    ]
    assert [r.chunk_id for r in apply_temporal_filter(records, mode="recent", recency_days=180)] == [1]


def test_per_patient_anchors_are_independent_of_date_shift():
    records = [
        _result(10, 100, "2150-12-01", 0.50),
        _result(11, 100, "2150-11-01", 0.50),
        _result(20, 200, "2185-06-01", 0.50),
        _result(21, 200, "2185-05-01", 0.50),
    ]
    scores = {r.chunk_id: r.rrf_score for r in apply_temporal_filter(records, mode="latest")}
    assert scores[10] > scores[11]
    assert scores[20] > scores[21]
    assert scores[11] == pytest.approx(scores[21], abs=0.01)
    assert scores[10] > 0.50


def test_trend_mode_is_chronological_and_places_undated_records_last():
    records = [
        _result(31, 100, "2150-09-15", 0.7),
        _result(32, 100, "2150-03-15", 0.7),
        _result(33, 100, "2150-12-20", 0.7),
        _result(34, 100, None, 0.7),
    ]
    assert [r.chunk_id for r in apply_temporal_filter(records, mode="trend")] == [32, 31, 33, 34]


def test_recent_mode_keeps_undated_records():
    records = [
        _result(31, 100, "2150-09-15", 0.7),
        _result(33, 100, "2150-12-20", 0.7),
        _result(34, 100, None, 0.7),
        _result(35, 100, None, 0.9),
    ]
    kept = {r.chunk_id for r in apply_temporal_filter(records, mode="recent", recency_days=30)}
    assert {34, 35} <= kept


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("most recent HbA1c", "latest"),
        ("current medications", "latest"),
        ("trend in HbA1c over the last 12 months", "trend"),
        ("creatinine progression", "trend"),
        ("potassium in the last 7 days", "recent"),
        ("lab results this admission", "recent"),
        ("any historical mention of penicillin reaction", "all"),
        ("abnormal potassium lab results", "all"),
    ],
)
def test_temporal_intent_detection(query, expected):
    assert detect_temporal_mode(query) == expected
