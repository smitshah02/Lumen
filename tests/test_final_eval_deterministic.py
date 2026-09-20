"""Deterministic answer-level checks.

Each test pins one rule that decides whether a case passes, using synthetic
response rows — no DB, no models, no network.
"""

import pytest

from src.evals.final_eval import cases as case_mod
from src.evals.final_eval import deterministic as D


@pytest.fixture(scope="module")
def gold():
    return {c.query_id: c for c in case_mod.load_cases()}


def row(answer, *, qid="demo_q01", labels=("S1",), bad_labels=(), subject=90000001,
        status="completed", hadm=91000013, needs_review=False, review_status="auto_approved"):
    """A synthetic collected response in the shape collect.py writes."""
    return {
        "query_id": qid, "subject_id": subject, "status": status, "answer": answer,
        "citations": [{"label": l, "labels": [l], "chunk_id": 100 + i, "claim": answer,
                       "verified": True} for i, l in enumerate(labels)],
        "sources": [{"label": l, "chunk_id": 100 + i, "subject_id": subject,
                     "hadm_id": hadm, "resolution": "note_chunks"}
                    for i, l in enumerate(labels)],
        "node_trail": ["triage", "patient_retrieval", "synthesis", "verification"],
        "citation_report": {"bad_labels": list(bad_labels)},
        "verification_summary": {"synthesis_failed": False},
        "needs_human_review": needs_review, "review_status": review_status,
        "escalation_reason": "none", "deterministic_lab_path": False,
        "graph_errors": [], "timings": {"total_ms": 100.0},
    }


# --- fact matching ---------------------------------------------------------
def test_structured_fact_matches_with_numeric_normalisation(gold):
    r = D.evaluate_case(gold["demo_q01"], row("The creatinine was 1.40 mg/dL [S1]."))
    assert r["facts"]["results"][0]["outcome"] == D.FACT_MATCH
    assert r["facts"]["matched"] == 1


def test_wrong_value_for_the_same_measurand_is_a_contradiction(gold):
    r = D.evaluate_case(gold["demo_q01"], row("The creatinine was 2.6 mg/dL [S1]."))
    res = r["facts"]["results"][0]
    assert res["outcome"] == D.FACT_CONTRADICTION
    assert "2.6" in res["detail"]
    assert r["case_pass"] is False
    assert "incorrect_value" in r["failure_tags"]


def test_absent_value_without_a_rival_is_unmatched_not_a_contradiction(gold):
    """Silence is not a wrong answer. It must not be scored as one."""
    r = D.evaluate_case(gold["demo_q01"], row("The creatinine result is in the chart [S1]."))
    assert r["facts"]["results"][0]["outcome"] == D.FACT_UNMATCHED
    assert r["facts"]["n_contradictions"] == 0


def test_text_only_fact_can_never_be_a_contradiction(gold):
    """A paraphrase must not be reported as a factual error — the judge, not the
    matcher, decides semantic equivalence."""
    r = D.evaluate_case(gold["demo_q04"],
                        row("The patient has systolic heart failure, EF 30% [S1].", qid="demo_q04"))
    res = r["facts"]["results"][0]
    assert res["kind"] == "text_only"
    assert res["outcome"] == D.FACT_UNMATCHED
    assert r["facts"]["n_contradictions"] == 0


def test_unrelated_number_elsewhere_cannot_manufacture_a_contradiction(gold):
    """The rival value must be in a sentence about the same measurand."""
    a = ("The creatinine was 1.4 mg/dL [S1]. Separately the potassium was 3.9 mEq/L "
         "and the dose was 2.6 mg [S1].")
    r = D.evaluate_case(gold["demo_q01"], row(a))
    assert r["facts"]["results"][0]["outcome"] == D.FACT_MATCH


# --- temporal --------------------------------------------------------------
def test_trend_values_in_chronological_order_pass(gold):
    a = "Creatinine rose to 1.8 mg/dL [S1], then 2.1 mg/dL [S2], recovering to 1.4 mg/dL [S3]."
    r = D.evaluate_case(gold["demo_q02"], row(a, qid="demo_q02", labels=("S1", "S2", "S3")))
    assert r["temporal"] == pytest.approx(r["temporal"])  # structural
    assert r["temporal"]["applicable"] is True
    assert r["temporal"]["pass"] is True


def test_trend_values_out_of_order_fail(gold):
    a = "Creatinine was 1.4 mg/dL [S1], then 1.8 mg/dL [S2], then 2.1 mg/dL [S3]."
    r = D.evaluate_case(gold["demo_q02"], row(a, qid="demo_q02", labels=("S1", "S2", "S3")))
    assert r["temporal"]["pass"] is False
    assert "incorrect_temporal_interpretation" in r["failure_tags"]
    assert r["case_pass"] is False


def test_temporal_not_applicable_without_a_structured_anchor(gold):
    r = D.evaluate_case(gold["demo_q21"],
                        row("The available records do not contain enough information to answer this.",
                            qid="demo_q21", labels=()))
    assert r["temporal"]["applicable"] is False
    assert r["temporal"]["pass"] is None


# --- abstention vs ambiguity ----------------------------------------------
def test_correct_abstention_on_unsupported_case(gold):
    r = D.evaluate_case(gold["demo_q21"],
                        row("The available records do not contain enough information to answer this.",
                            qid="demo_q21", labels=(), subject=90000017))
    assert r["abstention"]["applicable"] is True and r["abstention"]["pass"] is True
    assert r["ambiguity"]["applicable"] is False
    assert r["case_pass"] is True


def test_fabricated_value_on_unsupported_case_fails_abstention(gold):
    r = D.evaluate_case(gold["demo_q21"],
                        row("The most recent hemoglobin A1c was 6.5% [S1].",
                            qid="demo_q21", subject=90000017))
    assert r["abstention"]["pass"] is False
    assert r["must_not_contain"]["violations"]          # "hemoglobin a1c" is forbidden here
    assert "missed_refusal" in r["failure_tags"]
    assert r["case_pass"] is False


def test_ambiguity_passes_without_the_hard_refusal_sentence(gold):
    """An ambiguous case is handled correctly by explaining the conflict; it does
    not have to emit the refusal sentence."""
    a = "Apixaban is currently on hold pending gastroenterology review [S1]."
    r = D.evaluate_case(gold["demo_q28"], row(a, qid="demo_q28", subject=90000023))
    assert r["ambiguity"]["applicable"] is True
    assert r["ambiguity"]["pass"] is True
    assert r["ambiguity"]["hard_refusal"] is False
    assert r["abstention"]["applicable"] is False


def test_false_certainty_on_ambiguous_case_fails(gold):
    r = D.evaluate_case(gold["demo_q35"],
                        row("Yes, the patient is allergic to penicillin [S1].",
                            qid="demo_q35", subject=90000031))
    assert r["ambiguity"]["pass"] is False
    assert "missed_ambiguity" in r["failure_tags"]


# --- citations -------------------------------------------------------------
def test_hallucinated_label_is_measured_pre_strip(gold):
    """The runtime strips bad labels out of the stored answer, so re-validating
    it always looks clean. The honest count comes from the pre-strip record."""
    r = D.evaluate_case(gold["demo_q01"],
                        row("The creatinine was 1.4 mg/dL [S1].", bad_labels=["S7"]))
    assert r["citations"]["hallucinated_labels_pre_strip"] == ["S7"]
    assert r["citations"]["hallucinated_labels_post_strip"] == []
    assert "invalid_citation_label" in r["failure_tags"]
    assert r["review_worthy"]["is_review_worthy"] is True
    # Citation problems are reported separately and do NOT sink the case pass.
    assert r["case_pass"] is True


def test_uncited_factual_claim_is_flagged(gold):
    a = "The creatinine was 1.4 mg/dL [S1]. Renal function is worsening."
    r = D.evaluate_case(gold["demo_q01"], row(a))
    assert r["citations"]["n_factual_claims"] == 2
    assert r["citations"]["n_cited_factual_claims"] == 1
    assert "missing_citation" in r["failure_tags"]


def test_refusal_sentence_is_not_counted_as_an_uncited_claim(gold):
    r = D.evaluate_case(gold["demo_q21"],
                        row("The available records do not contain enough information to answer this.",
                            qid="demo_q21", labels=(), subject=90000017))
    assert r["citations"]["uncited_factual_claims"] == []
    assert "missing_citation" not in r["failure_tags"]


# --- isolation -------------------------------------------------------------
def test_cross_patient_evidence_is_a_hard_failure(gold):
    r = row("The creatinine was 1.4 mg/dL [S1].")
    r["sources"][0]["subject_id"] = 90000099
    out = D.evaluate_case(gold["demo_q01"], r)
    assert out["isolation"]["pass"] is False
    assert out["case_pass"] is False
    assert "cross_patient_leakage" in out["failure_tags"]
    assert out["review_worthy"]["is_review_worthy"] is True


def test_unresolved_provenance_is_reported_not_assumed_clean(gold):
    r = row("The creatinine was 1.4 mg/dL [S1].")
    r["sources"][0].update({"subject_id": None, "resolution": "unresolved"})
    out = D.evaluate_case(gold["demo_q01"], r)
    assert out["isolation"]["unresolved_labels"] == ["S1"]
    assert "unresolved_evidence_provenance" in out["failure_tags"]


# --- case pass composition -------------------------------------------------
def test_min_facts_cannot_compensate_for_a_contradiction():
    """A met fact threshold must not launder a clearly wrong extra statement.

    Uses a synthetic case (no gold case has min_facts below its fact count with
    distinct structured measurands) so the composition rule is pinned directly.
    """
    from src.evals.final_eval.cases import EvalCase, GoldFact
    case = EvalCase(
        query_id="synthetic_q", subject_id=90000001, query="discharge medications?",
        category="medications", answer_type="medication_history", difficulty="medium",
        temporal="all",
        expected_facts=(GoldFact.build("torsemide 20 mg"), GoldFact.build("apixaban 5 mg")),
        min_facts=1, expected_answer="Torsemide 20 mg and apixaban 5 mg.",
        unsupported=False, must_not_contain=(), evidence_hadm_ids=(), evidence_note_types=())
    r = D.evaluate_case(case, row("Torsemide 20 mg daily and apixaban 75 mg twice daily [S1]."))
    assert r["facts"]["matched"] == 1
    assert r["facts"]["threshold_met"] is True          # 1 >= min_facts
    assert r["facts"]["n_contradictions"] == 1          # apixaban dose is wrong
    assert r["case_pass"] is False
    assert r["case_pass_components"]["no_factual_contradiction"] is False


def test_omitted_trend_value_is_incompleteness_not_a_contradiction(gold):
    """demo_q02's gold lists three creatinine values. An answer that reports a
    correct but shorter slice of the same timeline has omitted a time point; it
    has not stated a wrong number, and must not be labelled a factual error."""
    a = ("Creatinine was 1.3 mg/dL [S1]. It rose to 1.8 mg/dL [S1], then fell to 1.5 mg/dL "
         "[S1]. It was 1.7 mg/dL [S2] and increased to 2.1 mg/dL [S2].")
    r = D.evaluate_case(gold["demo_q02"], row(a, qid="demo_q02", labels=("S1", "S2")))
    assert r["facts"]["n_contradictions"] == 0
    assert r["facts"]["matched"] == 2                   # 1.8 and 2.1 present
    assert r["facts"]["n_strict_unmatched"] == 1        # 1.4 omitted
    # The omission is still caught — by the temporal check, with the right label.
    assert r["temporal"]["pass"] is False
    assert "incorrect_value" not in r["failure_tags"]
    assert r["review_worthy"]["reasons"] == ["temporal_violation"]


def test_evaluator_error_never_passes_and_stays_visible(gold):
    r = D.evaluate_case(gold["demo_q01"],
                        {**row("", status="evaluator_error"), "error_type": "TimeoutError"})
    assert r["case_pass"] is False
    assert r["execution"]["evaluator_error"] is True
    assert "evaluator_error" in r["failure_tags"]


# --- review worthiness (non-circular) --------------------------------------
def test_review_worthiness_ignores_the_runtime_verifier(gold):
    """A clean answer the runtime escalated is NOT review-worthy by this
    definition — otherwise review_recall would measure the verifier against
    itself."""
    r = D.evaluate_case(gold["demo_q01"],
                        row("The creatinine was 1.4 mg/dL [S1].",
                            needs_review=True, review_status="pending"))
    assert r["review_worthy"]["is_review_worthy"] is False
    assert r["routing_observed"]["needs_human_review"] is True
    assert "review_routing_miss" not in r["failure_tags"]


def test_review_routing_miss_when_a_real_finding_was_auto_approved(gold):
    r = D.evaluate_case(gold["demo_q01"],
                        row("The creatinine was 2.6 mg/dL [S1].",
                            needs_review=False, review_status="auto_approved"))
    # demo_q01 is a "latest" case, so a wrong value is both a factual
    # contradiction and a temporal violation. Both are legitimate grounds.
    assert "factual_contradiction" in r["review_worthy"]["reasons"]
    assert "temporal_violation" in r["review_worthy"]["reasons"]
    assert "review_routing_miss" in r["failure_tags"]


def test_incompleteness_is_not_treated_as_review_worthy(gold):
    """An answer that is correct as far as it goes is a quality problem, not a
    safety one. Counting it as review-worthy would hold review routing to a
    signal the grounding verifier does not have."""
    r = D.evaluate_case(gold["demo_q19"],
                        row("The dose was intensified from 5 mg/kg to 10 mg/kg [S1].",
                            qid="demo_q19", subject=90000015, hadm=91000153))
    assert r["facts"]["threshold_met"] is False
    assert "incomplete_answer" in r["failure_tags"]
    assert r["review_worthy"]["is_review_worthy"] is False
    assert "review_routing_miss" not in r["failure_tags"]


# --- admission scope -------------------------------------------------------
def test_cited_evidence_outside_the_gold_admissions_is_reported(gold):
    case = gold["demo_q01"]
    row = {"query_id": "demo_q01", "status": "completed",
           "answer": "The creatinine was 1.4 mg/dL [S1].",
           "citations": [{"label": "S1", "labels": ["S1"], "claim": "c", "verified": True}],
           "sources": [{"label": "S1", "chunk_id": 1, "subject_id": 90000001,
                        "hadm_id": 99999999, "resolution": "note_chunks"}],
           "node_trail": [], "citation_report": {}, "graph_errors": []}
    out = D.check_admission_scope(case, row)
    assert out["applicable"] is True
    assert out["pass"] is False
    assert out["out_of_scope"] == [{"label": "S1", "hadm_id": 99999999}]
    assert out["expected_hadm_ids"] == sorted(int(h) for h in case.evidence_hadm_ids)


def test_uncited_out_of_scope_evidence_is_not_a_violation(gold):
    """The check is about what the answer CITED, not what retrieval returned."""
    case = gold["demo_q01"]
    row = {"query_id": "demo_q01", "status": "completed", "answer": "x [S1].",
           "citations": [{"label": "S1", "labels": ["S1"], "claim": "c", "verified": True}],
           "sources": [{"label": "S1", "chunk_id": 1, "subject_id": 90000001,
                        "hadm_id": int(case.evidence_hadm_ids[0]),
                        "resolution": "note_chunks"},
                       {"label": "S2", "chunk_id": 2, "subject_id": 90000001,
                        "hadm_id": 99999999, "resolution": "note_chunks"}],
           "node_trail": [], "citation_report": {}, "graph_errors": []}
    out = D.check_admission_scope(case, row)
    assert out["pass"] is True and out["out_of_scope"] == []


def test_an_unknown_admission_is_reported_not_assumed_in_scope(gold):
    case = gold["demo_q01"]
    row = {"query_id": "demo_q01", "status": "completed", "answer": "x [S1].",
           "citations": [{"label": "S1", "labels": ["S1"], "claim": "c", "verified": True}],
           "sources": [{"label": "S1", "chunk_id": 1, "subject_id": 90000001,
                        "hadm_id": None, "resolution": "unresolved"}],
           "node_trail": [], "citation_report": {}, "graph_errors": []}
    out = D.check_admission_scope(case, row)
    assert out["unknown_admission_labels"] == ["S1"]


def test_admission_scope_is_not_applicable_without_gold_admissions(gold):
    case = next((c for c in gold.values() if not c.evidence_hadm_ids), None)
    if case is None:
        pytest.skip("every gold case lists admissions")
    out = D.check_admission_scope(case, {"citations": [], "sources": []})
    assert out["applicable"] is False and out["pass"] is None
