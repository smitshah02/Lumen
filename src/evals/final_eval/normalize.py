"""
Normalization primitives for deterministic answer-level checks
==============================================================
Pure stdlib. No DB, no torch, no network, no project imports — so every rule
here is unit-testable on its own and the test suite runs in milliseconds.

Deliberate difference from src/agents/verify.py
-----------------------------------------------
The runtime verifier matches anchors VERBATIM against the source it cites,
one-sided, and can only conclude "supported" or "unresolved". That strictness
is correct for grounding: a number the note does not literally contain must not
be waved through.

This module does a different job — comparing an answer against a GOLD fact —
so it normalizes numerically instead: "1.4 mg/dL" and "1.40 mg/dL" are the same
lab value and must match. Units are canonicalized ("mg/dL" == "mg/dl"), values
compared as Decimals, and dates parsed to ISO before comparison.

What it deliberately does NOT do is guess. There is no stemming, no synonym
table, no fuzzy/edit-distance matching and no embedding similarity. A fact
either normalizes to an exact match or it is reported unmatched, and a
semantically equivalent paraphrase is left for the independent judge to
resolve. Loosening this until the numbers improve is how a benchmark stops
meaning anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------
# Longest-first so "mg/kg" is never shortened to "mg" and "mmol/L" never to "L".
# Mirrors the unit vocabulary of src/agents/verify.py so the evaluator and the
# runtime verifier agree on what counts as a quantity.
_UNIT_ALTS = (
    r"mcg/kg|mg/kg|mg/dL|g/dL|mEq/L|mmol/L|ng/mL|pg/mL|uIU/mL|IU/L|K/uL|"
    r"mmHg|liters?|units?|hours?|days?|weeks?|months?|years?|"
    r"mcg|mg|mL|kg|cm|L|g|%"
)

# Aliases fold to one canonical spelling. Everything is lowercased first, so
# only genuine spelling variants need an entry.
_UNIT_ALIASES = {
    "liter": "l", "liters": "l",
    "unit": "unit", "units": "unit",
    "hour": "hour", "hours": "hour",
    "day": "day", "days": "day",
    "week": "week", "weeks": "week",
    "month": "month", "months": "month",
    "year": "year", "years": "year",
    "ug": "mcg", "µg": "mcg",
}


def canon_unit(unit: str) -> str:
    """Canonical lowercase form of a unit. 'mg/dL' -> 'mg/dl', 'liters' -> 'l'."""
    u = (unit or "").strip().lower()
    return _UNIT_ALIASES.get(u, u)


# --------------------------------------------------------------------------
# Quantities, numbers, dates
# --------------------------------------------------------------------------
# The trailing guard is a lookahead, NOT \b: a word boundary after a symbol
# unit like "%" can never match (both sides non-word), so "7.1%" would be
# silently read as a bare 7.1 with no unit. Every HbA1c fact in the gold set
# is expressed in %, so that would have quietly disabled six of them.
_QTY_RE = re.compile(rf"(?<![\d.])(\d+(?:\.\d+)?)\s*({_UNIT_ALTS})(?![a-z0-9])", re.I)
_NUM_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d])")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_CITE_RE = re.compile(r"\[([SLGP]\d+)\]")

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}
_MONTH_ALT = "|".join(_MONTHS)
# "July 10, 2023" / "Jul 10 2023" is not supported in abbreviated form on
# purpose: the demo corpus renders full month names or ISO, and accepting
# ambiguous abbreviations would widen the matcher without evidence for it.
_MDY_RE = re.compile(rf"\b({_MONTH_ALT})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I)
_DMY_RE = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_ALT})\s+(\d{{4}})\b", re.I)


@dataclass(frozen=True)
class Quantity:
    """A number that carries the unit giving it meaning."""
    value: Decimal
    unit: str          # canonical

    def __str__(self) -> str:
        return f"{_plain(self.value)} {self.unit}"


def _plain(d: Decimal) -> str:
    """Decimal without exponent or trailing-zero noise: 1.40 -> '1.4'."""
    n = d.normalize()
    return f"{n:f}"


def _dec(s: str):
    try:
        return Decimal(s)
    except (InvalidOperation, TypeError, ValueError):
        return None


def strip_citations(text: str) -> str:
    """Remove [S1]-style markers. Without this, '[S1]' contributes a phantom
    numeric anchor of 1 — the same trap src/agents/verify.py guards against."""
    return _CITE_RE.sub(" ", text or "")


def norm_text(text: str) -> str:
    """Lowercase, citation-free, whitespace-collapsed, hyphens and slashes
    softened to spaces so 'sacubitril-valsartan' and 'sacubitril/valsartan'
    compare equal. Punctuation that can carry meaning inside a number (. and %)
    is preserved."""
    t = strip_citations(text or "").lower()
    t = re.sub(r"[‐-―\-/]", " ", t)
    t = re.sub(r"[^a-z0-9.%\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_quantities(text: str) -> list[Quantity]:
    """Every value+unit pair in the text, in order of appearance."""
    out = []
    for m in _QTY_RE.finditer(strip_citations(text)):
        v = _dec(m.group(1))
        if v is not None:
            out.append(Quantity(v, canon_unit(m.group(2))))
    return out


def parse_bare_numbers(text: str) -> list[Decimal]:
    """Numbers that carry no unit. A quantified number is excluded: '5 mg' is a
    quantity, not a bare 5, so an unrelated '5' cannot stand in for it."""
    src = strip_citations(text)
    quantified = set()
    for m in _QTY_RE.finditer(src):
        quantified.add((m.start(1), m.end(1)))
    out = []
    for m in _NUM_RE.finditer(_ISO_DATE_RE.sub(" ", src)):
        if (m.start(), m.end()) in quantified:
            continue
        v = _dec(m.group(0))
        if v is not None:
            out.append(v)
    return out


def parse_dates(text: str) -> list[str]:
    """Every date, normalized to ISO YYYY-MM-DD. Accepts ISO, 'March 18, 2024'
    and '18 March 2024'. An impossible date (month 13, day 32) is discarded
    rather than silently clamped."""
    src = strip_citations(text)
    out: list[str] = []

    def _add(y: int, mo: int, d: int):
        """ISO string, or None for an impossible date (discarded, not clamped)."""
        return f"{y:04d}-{mo:02d}-{d:02d}" if (1 <= mo <= 12 and 1 <= d <= 31) else None

    found: list = []                       # (position, iso) so order follows the text

    for m in _ISO_DATE_RE.finditer(src):
        iso = _add(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if iso:
            found.append((m.start(), iso))
    for m in _MDY_RE.finditer(src):
        iso = _add(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
        if iso:
            found.append((m.start(), iso))
    for m in _DMY_RE.finditer(src):
        iso = _add(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
        if iso:
            found.append((m.start(), iso))

    for _, iso in sorted(found):
        if iso not in out:
            out.append(iso)
    return out


# --------------------------------------------------------------------------
# Containment
# --------------------------------------------------------------------------
def contains_quantity(q: Quantity, text: str) -> bool:
    """True when the text states the same value in the same unit. Numeric, so
    1.4 mg/dL == 1.40 mg/dL; unit-aware, so 40 mg never matches 140 mg and
    5 mg/kg never matches a bare 5 mg."""
    return any(x.unit == q.unit and x.value == q.value for x in parse_quantities(text))


def contains_number(value: Decimal, text: str) -> bool:
    """True when the number appears, quantified or not. Whole-number matching:
    1.4 is not found inside 21.4 or 1.42."""
    if any(x.value == value for x in parse_quantities(text)):
        return True
    return any(v == value for v in parse_bare_numbers(text))


def contains_date(iso: str, text: str) -> bool:
    return iso in parse_dates(text)


def contains_phrase(phrase: str, text: str) -> bool:
    """Normalized substring containment, on whole-word boundaries so 'edema'
    does not match inside a longer token."""
    p, t = norm_text(phrase), norm_text(text)
    if not p:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", t) is not None


# --------------------------------------------------------------------------
# Sentences and content words
# --------------------------------------------------------------------------
# Same shape as src/agents/citations._SENT_RE: split only on .!? followed by
# whitespace AND a capital/opening bracket, so "1.4 mg/dL" is never split and
# clinical abbreviations survive.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for block in (text or "").split("\n"):
        parts.extend(s.strip() for s in _SENT_RE.split(block.strip()) if s.strip())
    return parts


# Words that carry no identifying power in a clinical fact. Kept small and
# explicit: this list decides which token must appear for a structured fact to
# be considered "about" the right thing, so it is reviewed, not generated.
_STOPWORDS = {
    "the", "and", "was", "were", "from", "with", "for", "not", "had", "has",
    "been", "that", "this", "its", "his", "her", "their", "after", "before",
    "during", "into", "over", "under", "per", "dose", "mean", "total", "of",
    "on", "in", "at", "to", "a", "an", "is", "are", "by", "up", "out", "off",
}


def content_tokens(text: str) -> list[str]:
    """Identifying words in a fact: alphabetic, 3+ characters, not stopwords.

    Quantities are removed first, so a unit word ('liters', 'mmHg') is never
    mistaken for the thing being measured.

    'creatinine 1.4 mg/dL'      -> ['creatinine']
    'total bilirubin 2.4 mg/dL' -> ['bilirubin']
    'mean gradient of 52 mmHg'  -> ['gradient']
    '4 liters'                  -> []           (nothing identifies it)
    """
    stripped = _QTY_RE.sub(" ", strip_citations(text or ""))
    return [t for t in re.findall(r"[a-z]{3,}", norm_text(stripped)) if t not in _STOPWORDS]
