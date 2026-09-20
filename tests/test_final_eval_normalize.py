"""Normalization primitives: numbers, units, dates, phrases, sentences.

Pure — no DB, no models, no network. These rules decide whether a clinical fact
counts as matched, so each one is pinned in both directions: what it must
accept AND what it must never accept.
"""

from decimal import Decimal

import pytest

from src.evals.final_eval import normalize as N


# --- numeric equivalence ---------------------------------------------------
@pytest.mark.parametrize("gold, answer, expected", [
    ("1.4", "creatinine was 1.4 mg/dL", True),
    ("1.4", "creatinine was 1.40 mg/dL", True),     # trailing zero is the same value
    ("1.40", "creatinine was 1.4 mg/dL", True),
    ("1.4", "creatinine was 21.4 mg/dL", False),    # not a substring match
    ("1.4", "creatinine was 1.42 mg/dL", False),
    ("1.4", "creatinine was 1.5 mg/dL", False),
])
def test_quantity_value_equivalence(gold, answer, expected):
    q = N.Quantity(Decimal(gold), "mg/dl")
    assert N.contains_quantity(q, answer) is expected


@pytest.mark.parametrize("unit_in_answer, expected", [
    ("mg/dL", True), ("mg/dl", True), ("MG/DL", True), ("mg/L", False), ("mg", False),
])
def test_unit_canonicalisation(unit_in_answer, expected):
    q = N.Quantity(Decimal("1.4"), "mg/dl")
    assert N.contains_quantity(q, f"creatinine 1.4 {unit_in_answer}") is expected


def test_dose_cannot_be_matched_by_a_larger_dose():
    """40 mg must never be satisfied by 140 mg — the classic wrong-dose trap."""
    q = N.Quantity(Decimal("40"), "mg")
    assert N.contains_quantity(q, "furosemide 140 mg daily") is False
    assert N.contains_quantity(q, "furosemide 40 mg daily") is True


def test_unit_bearing_number_is_not_a_bare_number():
    """'5 mg' is a quantity; an unrelated bare 5 must not stand in for it."""
    assert N.parse_bare_numbers("lisinopril 5 mg daily") == []
    q = N.Quantity(Decimal("5"), "mg/kg")
    assert N.contains_quantity(q, "infliximab 5 mg daily") is False


def test_combination_product_strength():
    qs = N.parse_quantities("budesonide-formoterol 200-6 mcg")
    assert qs == [N.Quantity(Decimal("6"), "mcg")]
    assert N.parse_bare_numbers("budesonide-formoterol 200-6 mcg") == [Decimal("200")]


def test_liters_and_percent_units():
    assert N.parse_quantities("4 liters")[0] == N.Quantity(Decimal("4"), "l")
    assert N.parse_quantities("6 L of oxygen")[0] == N.Quantity(Decimal("6"), "l")
    assert N.parse_quantities("A1c 7.1%")[0] == N.Quantity(Decimal("7.1"), "%")
    assert N.parse_quantities("ANC 0.2 K/uL")[0] == N.Quantity(Decimal("0.2"), "k/ul")


# --- dates -----------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("on 2023-07-10", ["2023-07-10"]),
    ("on July 10, 2023", ["2023-07-10"]),
    ("on 10 July 2023", ["2023-07-10"]),
    ("on March 18, 2024 and 2023-07-10", ["2024-03-18", "2023-07-10"]),
    ("no dates here", []),
    ("2023-13-45", []),                    # impossible date is discarded, not clamped
])
def test_date_parsing(text, expected):
    assert N.parse_dates(text) == expected


def test_date_digits_are_not_read_as_numbers():
    """'2024-03-18' must not contribute 2024, 3 and 18 as numeric anchors."""
    assert N.parse_bare_numbers("measured on 2024-03-18") == []


# --- citations and text ----------------------------------------------------
def test_citation_marker_contributes_no_numeric_anchor():
    assert N.parse_bare_numbers("the value was normal [S1]") == []
    assert N.contains_number(Decimal("1"), "the value was normal [S1]") is False


@pytest.mark.parametrize("phrase, text, expected", [
    ("sacubitril-valsartan", "started sacubitril/valsartan", True),
    ("heart failure with reduced ejection fraction",
     "He has Heart Failure with Reduced Ejection Fraction.", True),
    ("edema", "pulmonary edema noted", True),
    ("edema", "edematous tissue", False),              # whole-word only
    ("pneumonia", "no pneumonia", True),               # containment, not polarity
])
def test_contains_phrase(phrase, text, expected):
    assert N.contains_phrase(phrase, text) is expected


def test_sentence_split_does_not_break_decimals():
    s = N.split_sentences("Creatinine was 1.4 mg/dL [S1]. It later rose to 2.1 mg/dL [S2].")
    assert len(s) == 2
    assert "1.4 mg/dL" in s[0] and "2.1 mg/dL" in s[1]


@pytest.mark.parametrize("fact, expected", [
    ("creatinine 1.4 mg/dL", ["creatinine"]),
    ("total bilirubin 2.4 mg/dL", ["bilirubin"]),
    ("mean gradient of 52 mmHg", ["gradient"]),
    ("4 liters", []),                                  # unit word is not the measurand
    ("2023-07-10", []),
])
def test_content_tokens(fact, expected):
    assert N.content_tokens(fact) == expected


def test_percent_unit_is_parsed_not_dropped():
    """Regression: a word-boundary guard after '%' can never match, which would
    silently reduce 'hemoglobin A1c 7.1%' to a bare 7.1 with no unit. Six gold
    facts are expressed in %, so this check is load-bearing."""
    q = N.Quantity(Decimal("7.1"), "%")
    assert N.contains_quantity(q, "the A1c was 7.1%") is True
    assert N.contains_quantity(q, "the A1c was 7.1") is False    # a unit is required
    assert N.contains_quantity(N.Quantity(Decimal("13.2"), "%"), "improved from 13.2% to 6.8%")


def test_every_gold_structured_fact_parses_an_anchor():
    """No gold fact may be silently unparseable: a fact whose anchors do not
    parse can never be matched, and would depress the score for a reason that
    has nothing to do with the system under test."""
    from src.evals.final_eval.cases import load_cases
    weak = [f.text for c in load_cases() for f in c.expected_facts
            if f.kind == "structured" and not (f.quantities or f.dates or f.bare_numbers)]
    assert weak == []
