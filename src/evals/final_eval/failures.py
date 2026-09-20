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

# The complete vocabulary, with what each tag means. Kept here so the taxonomy
# can be read (and asserted in tests) in one place instead of being inferred
# from the two emitting functions. The names are the ones the framework has
# always emitted: renaming them would orphan the existing smoke artifacts for
# no measurement gain.
TAXONOMY = {
    # --- execution -------------------------------------------------------
    "evaluator_error": "the case raised inside the evaluator and produced no answer",
    "execution_error": "the graph recorded an error but still produced an answer",
    "synthesis_failure": "synthesis failed; the API would return 5xx",
    "empty_answer": "the run completed but the answer is empty",
    "invalid_response_schema": "the collected row is missing required response fields",
    # --- content ---------------------------------------------------------
    "incorrect_value": "a deterministic factual contradiction against a gold value",
    "incomplete_answer": "matched facts below the case's min_facts (a quality failure, "
                         "NOT review-worthy: see deterministic._review_worthy)",
    "unconfirmed_fact": "strict_unmatched fact — deterministic matching could not settle "
                        "it; semantic paraphrase is the judge's to resolve",
    "forbidden_content": "a must_not_contain term appears in the answer",
    # --- citations -------------------------------------------------------
    "missing_citation": "a factual claim carries no valid citation label",
    "invalid_citation_label": "a hallucinated label the synthesis repair stripped before "
                              "the stored answer (pre-strip; the honest measure)",
    "invalid_visible_citation": "a bad label still visible in the stored answer (post-strip)",
    # --- scope / isolation ----------------------------------------------
    "cross_patient_leakage": "cited or retrieved evidence belongs to another subject",
    "admission_scope_violation": "a cited chunk falls outside the gold admissions",
    "unresolved_evidence_provenance": "a label could not be resolved, so it cannot be "
                                      "cleared of leakage",
    # --- temporal / behaviour -------------------------------------------
    "incorrect_temporal_interpretation": "wrong latest value, order or window",
    "missed_refusal": "an unsupported case was answered as fact, or abstained badly",
    "missed_ambiguity": "an ambiguous case was answered with false certainty",
    # --- routing ---------------------------------------------------------
    "review_routing_miss": "auto-approved despite an independently detected finding",
    # --- judge (second opinion, never ground truth) ----------------------
    "judge_error": "the independent judge backend was unavailable for this case",
    "judge_parse_error": "the judge returned output that could not be parsed",
    "judge_inconsistent": "the judge contradicted its own criterion assessments twice; "
                          "scores preserved, case not counted as passed",
    "judge_skipped": "no answer existed to judge",
}

# judge status -> tag. A judge failure is always attributable to its cause.
JUDGE_STATUS_TAGS = {
    "backend_error": "judge_error",
    "parse_error": "judge_parse_error",
    "judge_inconsistent": "judge_inconsistent",
    "skipped_no_answer": "judge_skipped",
}

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
    if tag == "invalid_visible_citation":
        return {"labels": c["hallucinated_labels_post_strip"],
                "labels_available": c["labels_available"]}
    if tag == "admission_scope_violation":
        return {"expected_hadm_ids": det["admission_scope"].get("expected_hadm_ids"),
                "out_of_scope": det["admission_scope"].get("out_of_scope")}
    if tag in ("synthesis_failure", "empty_answer", "invalid_response_schema",
               "evaluator_error", "execution_error"):
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
            tag = JUDGE_STATUS_TAGS.get(j["status"], "judge_error")
            run_dir.append("failures", {
                "query_id": qid, "source": "judge", "tag": tag,
                "category": det["category"], "answer_type": det["answer_type"],
                "difficulty": det["difficulty"],
                "evidence": {"status": j["status"], "error": (j.get("error") or "")[:200],
                             "consistency_violations": j.get("consistency_violations") or [],
                             "repair_attempted": bool(j.get("repair_attempted"))}})
            counts[tag] += 1
            per_case[qid] += 1
        for dim, tag in JUDGE_TAGS.items():
            if j["status"] not in ("ok", "partial"):
                break          # a disowned verdict contributes no dimension findings
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
