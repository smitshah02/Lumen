"""Verifier diagnostic: the adversarial variants are false by construction, and the metrics add up."""
import pytest

from src.evals import verifier_diagnostic as vd

EVIDENCE = {"L1": "Creatinine — 3 numeric value(s) in time order, 2150-01-02 to 2150-03-04. First: 1.1 mg/dL on 2150-01-02. "
                  "Most recent: 1.6 mg/dL on 2150-03-04."}
LATEST = {"claim": "The most recent creatinine was 1.6 mg/dL on 2150-03-04 [L1].", "evidence": EVIDENCE,
          "origin": "structured_sql", "has_other_dates": True}
ENDS = {"claim": "The first value was 1.1 mg/dL on 2150-01-02 and the most recent was 1.6 mg/dL on 2150-03-04 [L1].",
        "evidence": EVIDENCE, "origin": "structured_sql"}
TREND = {"claim": "The value increased from 1.1 mg/dL on 2150-01-02 to 1.6 mg/dL on 2150-03-04 [L1].",
         "evidence": EVIDENCE, "origin": "structured_sql"}
DOSE = {"claim": "The patient takes metoprolol 25 mg twice daily [S1].",
        "evidence": {"S1": "Discharge medications: metoprolol 25 mg twice daily."}, "origin": "human_adjudicated"}


def test_each_transformation_makes_a_claim_the_source_cannot_support():
    v = vd.numeric_mismatch(LATEST)
    assert v["category"] == "numeric_mismatch" and "1.6" not in v["claim"] and "[L1]" in v["claim"]
    changed = v["claim"].split(" was ")[1].split(" ")[0]
    assert changed not in EVIDENCE["L1"]                                    # not true by coincidence
    assert vd.numeric_mismatch(DOSE)["category"] == "medication_dose_mismatch"
    d = vd.date_mismatch(LATEST)
    assert d["claim"] == "The most recent creatinine was 1.6 mg/dL on 2153-03-04 [L1]." and d["category"] == "temporal_mismatch"
    assert vd.direction_mismatch(TREND)["claim"].startswith("The value decreased from 1.1")
    assert vd.direction_mismatch(LATEST) is None                            # nothing to reverse
    assert vd.temporal_swap(ENDS)["claim"] == ("The first value was 1.6 mg/dL on 2150-03-04 and the most recent was "
                                               "1.1 mg/dL on 2150-01-02 [L1].")
    assert vd.temporal_swap(LATEST)["claim"].startswith("The earliest creatinine was 1.6")
    assert vd.temporal_swap({**LATEST, "has_other_dates": False}) is None   # one date: earliest == most recent
    assert vd.uncited(LATEST)["claim"] == "The most recent creatinine was 1.6 mg/dL on 2150-03-04."
    assert "[L9]" in vd.invalid_citation(LATEST)["claim"]
    added = vd.added_clause(LATEST)
    assert vd.ADDED_CLAUSE in added["claim"] and added["claim"].endswith("[L1].")
    assert vd.added_clause({**LATEST, "evidence": {"L1": "started dialysis"}}) is None   # would not be unsupported


def test_every_variant_is_labelled_unsupported_and_synthetic():
    made = vd.variants(LATEST) + vd.variants(ENDS) + vd.variants(TREND) + vd.variants(DOSE)
    assert len(made) >= 15 and all(v["gold"] == vd.UNSUPPORTED and v["synthetic"] for v in made)
    assert all("synthetic variant" in v["origin"] for v in made)
    assert {v["category"] for v in made} == {"numeric_mismatch", "medication_dose_mismatch", "temporal_mismatch",
                                             "direction_mismatch", "temporal_swap", "uncited_claim",
                                             "unsupported_added_clause"}
    assert not any(v["claim"] in (LATEST["claim"], ENDS["claim"], TREND["claim"], DOSE["claim"]) for v in made)


def _rows(tp, fn, fp, tn):
    return ([{"gold": vd.UNSUPPORTED, "approved": False, "category": "numeric_mismatch", "method": "fast_model"}] * tp
            + [{"gold": vd.UNSUPPORTED, "approved": True, "category": "temporal_swap", "method": "deterministic"}] * fn
            + [{"gold": vd.SUPPORTED, "approved": False, "category": "real_supported", "method": "fast_model"}] * fp
            + [{"gold": vd.SUPPORTED, "approved": True, "category": "real_supported", "method": "deterministic"}] * tn)


def test_metrics_follow_their_definitions():
    m = vd.metrics(_rows(tp=6, fn=2, fp=1, tn=9))
    assert m["n"] == 18 and m["gold_supported"] == 10 and m["gold_unsupported"] == 8
    assert m["confusion"] == {"unsupported_flagged": 6, "unsupported_approved": 2, "supported_flagged": 1, "supported_approved": 9}
    assert m["unsupported_recall"] == 0.75 and m["unsupported_precision"] == round(6 / 7, 4)
    assert m["unsupported_f1"] == round(2 * (6 / 7) * 0.75 / ((6 / 7) + 0.75), 4)
    assert m["supported_precision"] == round(9 / 11, 4) and m["false_support_rate"] == round(2 / 11, 4)
    assert m["false_support_rate"] + m["supported_precision"] == pytest.approx(1.0, abs=1e-3)
    empty = vd.metrics([])
    assert empty["unsupported_recall"] is None and empty["false_support_rate"] is None


def test_failures_are_broken_down_by_error_type_and_never_hidden():
    rows = _rows(tp=6, fn=2, fp=1, tn=9)
    cats = vd.by_category(rows)
    assert cats == {"numeric_mismatch": {"n": 6, "caught": 6, "caught_by_code": 0, "caught_by_model": 6, "missed": 0, "recall": 1.0},
                    "temporal_swap": {"n": 2, "caught": 0, "caught_by_code": 0, "caught_by_model": 0, "missed": 2, "recall": 0.0}}
    summary = {"overall": vd.metrics(rows), "real": vd.metrics(rows[8:]), "synthetic": vd.metrics(rows[:8]),
               "real_by_origin": {"human_adjudicated": "4/5"}, "by_category": cats}
    text = vd.render(summary)
    assert "synthetic adversarial 8" in text and "False-support rate" in text and "temporal_swap" in text


# ---- deterministic guards added after the baseline diagnostic (synthetic evidence only) ----
from src.agents.verify import deterministic_verdict, temporal_contradiction, unanchored_clause  # noqa: E402

LAB = ("Creatinine — 3 numeric value(s) in time order, 2150-01-04 to 2150-03-09. First: 1.2 mg/dL on 2150-01-04. "
       "Most recent: 0.8 mg/dL on 2150-03-09. Lowest: 0.8 mg/dL on 2150-03-09. Highest: 1.4 mg/dL on 2150-02-01. "
       "Last 3 value(s): 2150-01-04 1.2 mg/dL; 2150-02-01 1.4 mg/dL; 2150-03-09 0.8 mg/dL. Source: labevents table.")
ADMISSIONS = ("3 admission(s), admitted -> discharged, oldest first: 2150-01-02 -> 2150-01-09; "
              "2150-04-11 -> 2150-04-15; 2150-08-20 -> 2150-08-27. Source: admissions table.")


@pytest.mark.parametrize(("claim", "source", "labels", "expected"), [
    ("The first value was 1.2 mg/dL on 2150-01-04 [L1].", LAB, ["L1"], "supported"),                 # correct first
    ("The first value was 0.8 mg/dL on 2150-03-09 [L1].", LAB, ["L1"], "unsupported"),               # incorrect first
    ("The earliest creatinine was 0.8 mg/dL [L1].", LAB, ["L1"], "unsupported"),
    ("The latest creatinine was 0.8 mg/dL [L1].", LAB, ["L1"], "supported"),                         # correct latest
    ("The most recent creatinine was 1.2 mg/dL [L1].", LAB, ["L1"], "unsupported"),                  # incorrect latest
    ("The last creatinine was 1.2 mg/dL [L1].", LAB, ["L1"], "unsupported"),
    ("The first value was 0.8 mg/dL on 2150-03-09 and the most recent was 1.2 mg/dL on 2150-01-04 [L1].",
     LAB, ["L1"], "unsupported"),                                                                    # swapped
    ("The most recent admission began on 2150-08-20; that stay's discharge date was 2150-08-27 [A1].",
     ADMISSIONS, ["A1"], "supported"),
    ("The earliest admission began on 2150-08-20; that stay's discharge date was 2150-08-27 [A1].",
     ADMISSIONS, ["A1"], "unsupported"),                                                             # an earlier row exists
    ("The most recent admission began on 2150-01-02; that stay's discharge date was 2150-01-09 [A1].",
     ADMISSIONS, ["A1"], "unsupported"),                                                             # a later row exists
])
def test_first_and_most_recent_are_checked_against_what_the_evidence_says(claim, source, labels, expected):
    assert deterministic_verdict(claim, source, labels)[0] == expected


def test_temporal_attribution_is_left_alone_when_the_evidence_cannot_decide_it():
    note = "Creatinine 1.2 on 2150-01-04, later 0.8 on 2150-03-09. Follow-up booked for 2150-05-01."
    # a free-text note: its dates are not one series and nothing labels a role
    assert temporal_contradiction("The most recent creatinine was 0.8 on 2150-03-09 [S1].", note, ["S1"]) is None
    assert temporal_contradiction("Creatinine was 1.4 mg/dL on 2150-02-01 [L1].", LAB, ["L1"]) is None      # no role word
    assert temporal_contradiction("Over the last 3 values creatinine fell [L1].", LAB, ["L1"]) is None      # counting, not a role
    assert temporal_contradiction("The most recent value is discussed below [L1].", LAB, ["L1"]) is None    # nothing attributed


def test_an_assertion_with_no_anchor_is_not_approved_on_another_clauses_numbers():
    compound = "Creatinine improved to 0.8 mg/dL on 2150-03-09 and the patient was discharged on dialysis [L1]."
    assert "discharged on dialysis" in unanchored_clause(compound)
    assert deterministic_verdict(compound, LAB, ["L1"])[0] == "unresolved"            # goes to the model, never auto-approved
    # every clause carries its own anchor, or the leftover is a fragment: still decided in code
    both = "The lowest value was 0.8 mg/dL on 2150-03-09 and the highest was 1.4 mg/dL on 2150-02-01 [L1]."
    assert unanchored_clause(both) is None and deterministic_verdict(both, LAB, ["L1"])[0] == "supported"
    assert unanchored_clause("Creatinine was 0.8 mg/dL on 2150-03-09 and stable [L1].") is None
    assert unanchored_clause("Creatinine was 0.8 mg/dL on 2150-03-09 [L1].") is None
