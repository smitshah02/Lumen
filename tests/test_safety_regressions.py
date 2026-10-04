"""Regressions for two failures the evaluation found. Values are synthetic:
the real cases involve patient text, which never enters the repository.

1. Holdout scorecard, temporal question: the answer was the fixed decline plus
   two uncited sentences. Verification called it a "pure refusal" because it
   contained the decline and cited nothing, marked all three sentences
   verified, and released it without review.

2. Blinded human adjudication: "…increased from 6.0 on admission to 5.8 at
   discharge [S2]" was auto-supported because both numbers occur in the cited
   note. The claim contradicts itself, and number matching cannot see that.

The tests assert the safety contract, not any particular model wording.
"""
import importlib.util
from pathlib import Path

import pytest

from src.agents import citations, verify
from tests.test_routing import REFUSAL, _cite, _ev, graph_mod, spy  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
NOTE = "Imaging: none this admission. Labs on admission: K 5.1. At discharge K was 4.2."


def _verify(graph_mod, claims, draft, **state):
    return graph_mod.verification({"patient_evidence": [_ev(text=NOTE)], "draft_answer": draft,
                                   "citations": [_cite(c, labels) for c, labels in claims], **state})


# --- 1. a decline does not excuse the sentences written next to it ----------------
@pytest.mark.parametrize("extra", [
    ["No chest imaging was performed during the most recent admission.", "The last study predates this stay."],
    ["The most recent potassium was 4.2."],
])
def test_uncited_sentences_next_to_a_decline_are_not_auto_approved(graph_mod, spy, extra):
    claims = [(s, []) for s in extra] + [(REFUSAL, [])]
    out = _verify(graph_mod, claims, " ".join(extra + [REFUSAL]))
    assert out["needs_human_review"] is True and out["review_status"] == "pending"
    assert out["verification"]["refusal"] is False                        # not a pure decline
    assert out["verification"]["unsupported"] >= len(extra)
    assert all(c["verified"] is False for c in out["citations"][:len(extra)])
    assert graph_mod.route_after_verification(out) == "human_review"
    assert spy == []                                                      # decided by code: nothing is cited


@pytest.mark.parametrize("decline", [
    REFUSAL, "The available records do not contain enough information to answer this question.",
    "The available records do not contain enough information.",
])
def test_the_fixed_decline_alone_is_still_safe_and_auto_approved(graph_mod, spy, decline):
    out = _verify(graph_mod, [(decline, [])], decline)
    assert out["needs_human_review"] is False and out["review_status"] == "auto_approved"
    assert out["verification"]["refusal"] is True and out["citations"][0]["verified"] is True and spy == []


def test_the_no_evidence_decline_is_still_safe(graph_mod, spy):
    out = graph_mod.verification({"draft_answer": REFUSAL, "citations": []})
    assert out["review_status"] == "auto_approved" and out["needs_human_review"] is False


def test_a_cited_and_supported_answer_is_unaffected(graph_mod, spy):
    claim = "Potassium was 5.1 on admission [S1]."
    out = _verify(graph_mod, [(claim, ["S1"])], claim)
    assert out["review_status"] == "auto_approved" and out["citations"][0]["verified"] is True and spy == []


@pytest.mark.parametrize("sentence,boilerplate", [
    (REFUSAL, True),
    ("The available records do not contain enough information to answer this question.", True),
    ("The available records do not contain enough information, but potassium was 4.2 at discharge.", False),
    ("The available records do not contain enough information; however the last admission was in March.", False),
    ("The available records do not contain enough information to say whether the 3 prior studies showed change.", False),
    ("No imaging is documented. The available records do not contain enough information to answer this.", False),
    ("Potassium was 4.2 at discharge.", False), ("", False),
])
def test_only_the_decline_itself_is_boilerplate(sentence, boilerplate):
    assert citations.is_refusal_claim(sentence) is boilerplate


def test_refusal_guard_for_structured_rows_still_fires_on_a_pure_decline(graph_mod, spy):
    out = _verify(graph_mod, [(REFUSAL, [])], REFUSAL, structured_rows=9)
    assert out["needs_human_review"] is True and out["review_status"] == "pending"


# --- 2. a claim that contradicts its own numbers ---------------------------------
@pytest.mark.parametrize("claim", [
    "The patient's potassium level increased from 5.1 on admission to 4.2 at discharge [S1].",
    "Potassium rose from 5.1 to 4.2 [S1].",
    "Potassium decreased from 4.2 to 5.1 over the stay [S1].",
    "The dose was reduced from 20 mg to 40 mg [S1].",
])
def test_a_self_contradictory_direction_is_unsupported_by_code(graph_mod, spy, claim):
    verdict, note = verify.deterministic_verdict(claim, NOTE + " Dose 20 mg, later 40 mg.", ["S1"])
    assert verdict == "unsupported" and "contradicts itself" in note
    out = graph_mod.verification({"patient_evidence": [_ev(text=NOTE + " Dose 20 mg, later 40 mg.")],
                                  "draft_answer": claim, "citations": [_cite(claim, ["S1"])]})
    assert out["citations"][0]["verified"] is False and out["needs_human_review"] is True
    assert out["verification"]["claims"][0]["stage"] == "deterministic" and spy == []


@pytest.mark.parametrize("claim,expected", [
    ("Potassium decreased from 5.1 on admission to 4.2 at discharge [S1].", "supported"),   # consistent
    ("Potassium fell from 5.1 to 4.2 [S1].", "supported"),
    ("Potassium was 5.1 on admission and 4.2 at discharge [S1].", "supported"),           # no direction asserted
    # both directions: the direction check cannot decide it. "Potassium rose" has no anchor of
    # its own, so since the unanchored-clause rule the model decides; it is never refuted in code.
    ("Potassium rose and then fell from 5.1 to 4.2 [S1].", "unresolved"),
    ("Potassium increased from 5.1 to 6.3 [S1].", "unresolved"),                          # consistent, anchor absent
    ("Potassium increased during the stay [S1].", "unresolved"),                          # no numbers at all
])
def test_consistent_or_undecidable_claims_keep_their_previous_verdict(claim, expected):
    assert verify.deterministic_verdict(claim, NOTE, ["S1"])[0] == expected


def test_dates_are_not_read_as_a_direction():
    assert verify.direction_contradiction("Creatinine rose from 2150-03-09 to 2150-03-01 [S1].") is None
    assert verify.direction_contradiction("It increased from 6.0 on admission to 5.8 at discharge.") is not None


# --- 3. a holdout cohort cannot be silently partial -------------------------------
@pytest.fixture
def holdout():
    spec = importlib.util.spec_from_file_location("holdout_cli", ROOT / "scripts" / "holdout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_frozen_subject_is_loaded_or_reported_with_a_reason(holdout):
    ids = list(range(101, 111))
    full = {t: set(ids) for t in holdout.TABLES}
    assert all(r["loaded"] and r["reason"] == "" for r in holdout.completeness(ids, full))
    # what a 3-patient pilot followed by a 7-patient load left behind
    partial = {t: set(ids[3:]) for t in holdout.TABLES}
    rows = holdout.completeness(ids, partial)
    assert [r["alias"] for r in rows if not r["loaded"]] == ["H1", "H2", "H3"]
    assert all("not loaded" in r["reason"] for r in rows[:3]) and sum(r["loaded"] for r in rows) == 7
    # a subject with a patient row but no notes is incomplete, not loaded
    no_notes = {**full, "clinical_notes": set(ids[1:]), "note_chunks": set(ids[1:])}
    assert holdout.completeness(ids, no_notes)[0] == {
        "alias": "H1", "loaded": False, "missing_tables": ["clinical_notes", "note_chunks"],
        "reason": "incomplete: missing from clinical_notes, note_chunks"}
    # labs are not guaranteed by eligibility: absent labs are reported but do not block
    no_labs = {**full, "labevents": set(ids[1:])}
    assert holdout.completeness(ids, no_labs)[0]["loaded"] is True
    assert "no rows in labevents" in holdout.completeness(ids, no_labs)[0]["reason"]


def test_holdout_ingest_has_no_incremental_mode(holdout):
    source = (ROOT / "scripts" / "holdout.py").read_text()
    ingest = source[source.index("def ingest("):source.index("# ---------------------------------------------------------------- status ----")]
    assert "already loaded" not in ingest and "todo" not in ingest          # the whole requested set, every time
    assert "return status(expected=len(wanted))" in ingest                  # and the result is checked
    runner = (ROOT / "scripts" / "scorecard.py").read_text()
    assert "are not loaded in" in runner and "Load the whole cohort first" in runner
