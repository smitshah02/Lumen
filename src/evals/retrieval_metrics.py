"""Ranking metrics for the development retrieval benchmark.

Standard definitions, implemented directly. `relevance` maps a document id to
a grade: 2 = contains the required evidence, 1 = useful but not the target,
0 or absent = irrelevant. Precision, recall, hit and MRR treat any grade > 0 as
relevant; nDCG uses the grades.

All averages are macro-averages over questions: each question counts once.
"""

from __future__ import annotations

import math
import statistics

METRICS = ("p@5", "r@5", "ndcg@5", "mrr", "hit@5", "r@10", "hit@10", "ndcg@10", "r@20", "hit@20", "r@50", "hit@50")


def _relevant(relevance: dict) -> set:
    return {doc for doc, grade in relevance.items() if grade > 0}


def precision_at_k(ranked: list, relevance: dict, k: int) -> float:
    """Relevant documents in the top k, divided by k (not by how many came back)."""
    rel = _relevant(relevance)
    return sum(1 for doc in ranked[:k] if doc in rel) / k


def recall_at_k(ranked: list, relevance: dict, k: int) -> float:
    rel = _relevant(relevance)
    return sum(1 for doc in ranked[:k] if doc in rel) / len(rel) if rel else 0.0


def hit_at_k(ranked: list, relevance: dict, k: int) -> float:
    rel = _relevant(relevance)
    return 1.0 if any(doc in rel for doc in ranked[:k]) else 0.0


def reciprocal_rank(ranked: list, relevance: dict, min_grade: int = 1) -> float:
    """1 / rank of the first document graded at least `min_grade`; 0 if none was returned."""
    for rank, doc in enumerate(ranked, 1):
        if relevance.get(doc, 0) >= min_grade:
            return 1.0 / rank
    return 0.0


def dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(grades, 1))


def ndcg_at_k(ranked: list, relevance: dict, k: int) -> float:
    """DCG of the ranking over the DCG of the best possible ordering of all graded documents."""
    ideal = dcg(sorted((g for g in relevance.values() if g > 0), reverse=True)[:k])
    return dcg([relevance.get(doc, 0) for doc in ranked[:k]]) / ideal if ideal else 0.0


def score(ranked: list, relevance: dict) -> dict:
    """Every metric for one question."""
    return {"p@5": precision_at_k(ranked, relevance, 5), "r@5": recall_at_k(ranked, relevance, 5),
            "ndcg@5": ndcg_at_k(ranked, relevance, 5), "mrr": reciprocal_rank(ranked, relevance),
            "hit@5": hit_at_k(ranked, relevance, 5), "r@10": recall_at_k(ranked, relevance, 10),
            "hit@10": hit_at_k(ranked, relevance, 10), "ndcg@10": ndcg_at_k(ranked, relevance, 10),
            "r@20": recall_at_k(ranked, relevance, 20), "hit@20": hit_at_k(ranked, relevance, 20),
            "r@50": recall_at_k(ranked, relevance, 50), "hit@50": hit_at_k(ranked, relevance, 50)}


def macro(per_question: list[dict]) -> dict:
    return {m: round(statistics.mean(q[m] for q in per_question), 4) if per_question else None for m in METRICS}


def lift(base: dict, improved: dict, metrics=("ndcg@5", "p@5", "r@5", "mrr")) -> dict:
    """Absolute and relative change from `base` to `improved`. No significance is claimed."""
    out = {}
    for m in metrics:
        delta = improved[m] - base[m]
        out[m] = {"base": base[m], "improved": improved[m], "absolute": round(delta, 4),
                  "relative_pct": round(100 * delta / base[m], 1) if base[m] else None}
    return out


def temporal_score(ranked: list, relevance: dict) -> dict:
    """For a latest/earliest question: grade 2 marks the note the question asks
    for, grade 1 the same kind of evidence from another date (a distractor).

    target_hit@5      the target is in the top 5
    target_rr         1 / rank of the first target chunk
    ordered_correctly the target is ranked above every distractor that was returned
    """
    first_target = next((i for i, d in enumerate(ranked, 1) if relevance.get(d, 0) == 2), None)
    first_distractor = next((i for i, d in enumerate(ranked, 1) if relevance.get(d, 0) == 1), None)
    return {"target_hit@5": 1.0 if first_target and first_target <= 5 else 0.0,
            "target_rr": 1.0 / first_target if first_target else 0.0,
            "ordered_correctly": 1.0 if first_target and (first_distractor is None or first_target < first_distractor) else 0.0}


def temporal_macro(per_question: list[dict]) -> dict:
    keys = ("target_hit@5", "target_rr", "ordered_correctly")
    return {k: round(statistics.mean(q[k] for q in per_question), 4) if per_question else None for k in keys}
