"""
Phase 7 — failure taxonomy
==========================
One line per (case, failure tag). A case may carry several tags: an answer can
be both temporally wrong and missing a citation, and collapsing that to a
single "worst" label loses the second problem.

Tags are derived from results, not declared in advance — the vocabulary below
is whatever src/evals/final_eval/deterministic.py:_failure_tags actually
emitted, plus the judge-side tags added here.

No raw clinical text is written. A representative excerpt of a generated claim
is truncated, and the evidence behind it is referenced by label only.
"""

from __future__ import annotations

from collections import Counter

EXCERPT_CHARS = 140

# Judge-side tags. A low score is a finding, not a verdict — the judge is a
# second opinion, never ground truth, so these are kept separate from the
# deterministic tags and are labelled as such in the report.
JUDGE_TAGS = {
    "groundedness": "judge_low_groundedness",
    "factual_correctness": "judge_low_factual_correctness",
    "completeness": "judge_low_completeness",
    "answer_relevance": "judge_low_relevance",
    "temporal_correctness": "judge_low_temporal",
    "abstention_quality": "judge_low_abstention",
}
JUDGE_LOW_THRESHOLD = 2          # scores of 0-2 are "material problem" on the 0-4 rubric


def _evidence_for(det: dict, tag: str) -> dict:
    """What a reviewer needs to find this failure again, without note text."""
    c, f = det["citations"], det["facts"]
    if tag in ("incorrect_value", "unconfirmed_fact", "incomplete_answer"):
        return {"facts": [{"fact": r["fact"], "outcome": r["outcome"],
                           "detail": r["detail"][:EXCERPT_CHARS]}
                          for r in f["results"] if r["outcome"] != "strict_match"]}
    if tag == "missing_citation":
        return {"uncited_claims": [x[:EXCERPT_CHARS] for x in c["uncited_factual_claims"]]}
    if tag == "invalid_citation_label":
        return {"labels": c["hallucinated_labels_pre_strip"],
                "labels_available": c["labels_available"]}
    if tag == "cross_patient_leakage":
        return {"leaked": det["isolation"]["leaked"]}
    if tag == "unresolved_evidence_provenance":
        return {"labels": det["isolation"]["unresolved_labels"]}
    if tag == "incorrect_temporal_interpretation":
        return {"mode": det["temporal"]["mode"], "detail": det["temporal"]["detail"][:EXCERPT_CHARS]}
    if tag == "missed_refusal":
        return {k: det["abstention"][k] for k in
                ("refusal_detected", "must_not_contain_violations", "fabricated_citations")}
    if tag == "missed_ambiguity":
        return {"markers": det["ambiguity"].get("markers"),
                "hard_refusal": det["ambiguity"].get("hard_refusal")}
    if tag == "forbidden_content":
        return {"violations": det["must_not_contain"]["violations"]}
    if tag == "review_routing_miss":
        return {"review_worthy_reasons": det["review_worthy"]["reasons"],
                "routing": det["routing_observed"]}
    if tag in ("synthesis_failure", "empty_answer", "invalid_response_schema", "evaluator_error"):
        return {"status": det["execution"]["status"], "error_type": det["execution"]["error_type"],
                "graph_errors": det["execution"]["graph_errors"][:3]}
    return {}


def build(run_dir, progress=None) -> dict:
    """Write failures.jsonl and return the tag counts."""
    say = progress or (lambda *_a, **_k: None)
    dets = {d["query_id"]: d for d in run_dir.read_jsonl("deterministic")}
    judges = {j["query_id"]: j for j in run_dir.read_jsonl("judge")}
    run_dir.file("failures").unlink(missing_ok=True)

    counts, per_case = Counter(), Counter()
    for qid in sorted(dets):
        det = dets[qid]
        for tag in det["failure_tags"]:
            run_dir.append("failures", {
                "query_id": qid, "source": "deterministic", "tag": tag,
                "category": det["category"], "answer_type": det["answer_type"],
                "difficulty": det["difficulty"], "evidence": _evidence_for(det, tag)})
            counts[tag] += 1
            per_case[qid] += 1

        j = judges.get(qid)
        if not j:
            continue
        if j["status"] not in ("ok", "partial"):
            run_dir.append("failures", {
                "query_id": qid, "source": "judge", "tag": "judge_failure",
                "category": det["category"], "answer_type": det["answer_type"],
                "difficulty": det["difficulty"],
                "evidence": {"status": j["status"], "error": (j.get("error") or "")[:200]}})
            counts["judge_failure"] += 1
            per_case[qid] += 1
        for dim, tag in JUDGE_TAGS.items():
            d = (j.get("dimensions") or {}).get(dim) or {}
            if d.get("applicable") and isinstance(d.get("score"), int) \
                    and d["score"] <= JUDGE_LOW_THRESHOLD:
                run_dir.append("failures", {
                    "query_id": qid, "source": "judge", "tag": tag,
                    "category": det["category"], "answer_type": det["answer_type"],
                    "difficulty": det["difficulty"],
                    "evidence": {"score": d["score"], "reason": (d.get("reason") or "")[:EXCERPT_CHARS],
                                 "criterion": d.get("criterion")}})
                counts[tag] += 1
                per_case[qid] += 1
        say(f"  {qid:<10} tags={det['failure_tags']}")
    return {"by_tag": dict(counts.most_common()),
            "cases_with_failures": len(per_case),
            "total_findings": sum(counts.values())}
