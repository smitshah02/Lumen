"""Ranking metrics for the development retrieval benchmark, checked by hand."""
import importlib.util
import math
from pathlib import Path

import pytest

from src.evals import retrieval_metrics as rm

ROOT = Path(__file__).resolve().parents[1]
REL = {1: 2, 2: 1, 3: 2, 9: 0}                       # three relevant documents, one explicitly irrelevant
RANKED = [7, 1, 8, 2, 6, 3]


def test_precision_recall_hit_at_k():
    assert rm.precision_at_k(RANKED, REL, 5) == pytest.approx(2 / 5)      # docs 1 and 2 in the top 5
    assert rm.recall_at_k(RANKED, REL, 5) == pytest.approx(2 / 3)
    assert rm.recall_at_k(RANKED, REL, 10) == 1.0
    assert rm.hit_at_k(RANKED, REL, 5) == 1.0 and rm.hit_at_k([7, 8], REL, 5) == 0.0
    assert rm.precision_at_k([1], REL, 5) == pytest.approx(1 / 5)         # short lists are not rewarded
    assert rm.recall_at_k(RANKED, {}, 5) == 0.0 and rm.precision_at_k([], REL, 5) == 0.0
    assert rm.precision_at_k([9, 9, 9], REL, 5) == 0.0                    # grade 0 is not relevant


def test_reciprocal_rank():
    assert rm.reciprocal_rank(RANKED, REL) == pytest.approx(1 / 2)
    assert rm.reciprocal_rank([7, 8], REL) == 0.0
    assert rm.reciprocal_rank(RANKED, REL, min_grade=2) == pytest.approx(1 / 2)
    assert rm.reciprocal_rank([2, 7, 1], REL, min_grade=2) == pytest.approx(1 / 3)


def test_ndcg_uses_grades_and_the_ideal_ordering():
    gains = [0, 3, 0, 1, 0]                                                # 2**grade - 1 for the top five
    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(gains, 1))
    ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert rm.ndcg_at_k(RANKED, REL, 5) == pytest.approx(dcg / ideal)
    assert rm.ndcg_at_k([1, 3, 2], REL, 5) == pytest.approx(1.0)           # perfect order
    assert rm.ndcg_at_k([2, 1, 3], REL, 5) < 1.0                           # a weaker document first costs something
    assert rm.ndcg_at_k([7, 8], REL, 5) == 0.0 and rm.ndcg_at_k(RANKED, {}, 5) == 0.0


def test_macro_average_counts_each_question_once():
    a, b = rm.score([1, 3, 2], REL), rm.score([7, 8], REL)
    avg = rm.macro([a, b])
    assert avg["hit@5"] == 0.5 and avg["mrr"] == 0.5 and avg["ndcg@5"] == 0.5
    assert set(avg) == set(rm.METRICS) and rm.macro([])["p@5"] is None


def test_lift_reports_absolute_and_relative_change():
    lift = rm.lift({"ndcg@5": 0.5, "p@5": 0.4, "r@5": 0.0, "mrr": 0.8}, {"ndcg@5": 0.6, "p@5": 0.3, "r@5": 0.1, "mrr": 0.8})
    assert lift["ndcg@5"] == {"base": 0.5, "improved": 0.6, "absolute": 0.1, "relative_pct": 20.0}
    assert lift["p@5"]["absolute"] == -0.1 and lift["p@5"]["relative_pct"] == -25.0     # a drop is reported as a drop
    assert lift["r@5"]["relative_pct"] is None and lift["mrr"]["absolute"] == 0.0


def test_temporal_score_separates_the_target_from_other_dates():
    rel = {10: 2, 11: 2, 20: 1, 21: 1}                                    # 2 = the asked-for note, 1 = another date
    assert rm.temporal_score([10, 20, 5], rel) == {"target_hit@5": 1.0, "target_rr": 1.0, "ordered_correctly": 1.0}
    assert rm.temporal_score([20, 21, 10], rel) == {"target_hit@5": 1.0, "target_rr": pytest.approx(1 / 3),
                                                    "ordered_correctly": 0.0}
    assert rm.temporal_score([20, 5, 6, 7, 8, 10], rel)["target_hit@5"] == 0.0
    assert rm.temporal_score([5, 6], rel) == {"target_hit@5": 0.0, "target_rr": 0.0, "ordered_correctly": 0.0}
    assert rm.temporal_score([5, 11], rel)["ordered_correctly"] == 1.0     # no distractor returned
    assert rm.temporal_macro([rm.temporal_score([10], rel), rm.temporal_score([20], rel)])["ordered_correctly"] == 0.5


# --- benchmark construction: section labels come from note structure ---------------
@pytest.fixture
def builder():
    spec = importlib.util.spec_from_file_location("retrieval_eval_cli", ROOT / "scripts" / "retrieval_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NOTE = """Name: X
Allergies:
Penicillin causes a rash and shellfish causes swelling of the lips.

Brief Hospital Course:
The patient was admitted for evaluation of chest pain and was treated with
aspirin, then monitored on telemetry for two days without further events.

Discharge Medications:
1. Aspirin 81 mg daily
2. Metoprolol tartrate 25 mg twice daily
3. Atorvastatin 40 mg nightly

Discharge Diagnosis:
Unstable angina, hypertension and hyperlipidemia were the final diagnoses.
"""


def test_chunks_are_assigned_to_the_section_they_come_from(builder):
    chunks = [(1, "Penicillin causes a rash and shellfish causes swelling of the lips."),
              (2, "The patient was admitted for evaluation of chest pain and was treated with aspirin, then monitored"),
              (3, "1. Aspirin 81 mg daily 2. Metoprolol tartrate 25 mg twice daily 3. Atorvastatin 40 mg nightly"),
              (4, "Unstable angina, hypertension and hyperlipidemia were the final diagnoses."),
              (5, "Text that does not occur anywhere in this particular clinical note at all.")]
    sections = builder.chunk_sections(NOTE, chunks)
    assert sections == {1: ["allergies"], 2: ["course"], 3: ["dc_meds"], 4: ["dc_dx"]}     # the unlocated chunk is not guessed
    assert [name for name, _, _ in builder.section_spans(NOTE)] == ["allergies", "course", "dc_meds", "dc_dx"]


def test_a_short_section_merged_into_a_longer_chunk_is_still_labelled(builder):
    # The chunker merges a short section into its neighbour. The chunk's midpoint is in the
    # medication list, but the whole diagnosis section is inside it, so it carries both.
    merged = ("1. Aspirin 81 mg daily 2. Metoprolol tartrate 25 mg twice daily 3. Atorvastatin 40 mg nightly "
              "Discharge Diagnosis: Unstable angina, hypertension and hyperlipidemia were the final diagnoses.")
    assert builder.chunk_sections(NOTE, [(7, merged)]) == {7: ["dc_meds", "dc_dx"]}


def test_benchmark_is_development_only_and_labels_precede_retrieval(builder):
    source = (ROOT / "scripts" / "retrieval_eval.py").read_text()
    assert "lumen_holdout" in source and "frozen holdout is not used" in source
    assert "already exists. The benchmark is frozen once built" in source
    build = source[source.index("def build("):source.index("@contextmanager")]
    assert "HybridRetriever" not in build and "bm25_search" not in build and "retriever.search(" not in build   # no retriever in labelling
    ids = [t[0] for t in builder.TEMPLATES]
    assert len(ids) == len(set(ids)) and sum(1 for t in builder.TEMPLATES if t[5]) >= 5
    assert {t[5] for t in builder.TEMPLATES} == {None, "latest", "earliest"}


def test_stage_isolation_restores_the_production_retriever(builder):
    module = type("M", (), {"bm25_search": "lexical-fn", "vector_search": "vector-fn"})
    retriever = type("R", (), {"reranker": "bge", "rerank_candidates": 40})()
    for config, lexical_on, vector_on, rerank_on in (("lexical", True, False, False), ("vector", False, True, False),
                                                     ("hybrid", True, True, False), ("hybrid_bge", True, True, True)):
        with builder._arm(module, retriever, config):
            assert (module.bm25_search == "lexical-fn") is lexical_on
            assert (module.vector_search == "vector-fn") is vector_on
            assert (retriever.reranker == "bge") is rerank_on
        assert (module.bm25_search, module.vector_search, retriever.reranker) == ("lexical-fn", "vector-fn", "bge")
        assert retriever.rerank_candidates == 40
