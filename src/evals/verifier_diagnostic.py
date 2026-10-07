"""Verifier diagnostic: does the verifier flag claims that are known to be wrong?

A baseline MEASUREMENT of the verifier as it is. Nothing here changes it.

The set has two parts, always reported separately:

  real       claims with an independent label: SUPPORTED by a human reviewer
             (blinded adjudication, development cohort), or true by construction
             because they were templated from SQL rows.
  synthetic  adversarial variants made from those claims by a fixed rule that
             makes them false: change a number, a date or a dose, reverse a
             direction, swap first and most recent, drop or break the
             citation, or append an unsupported clause. Their label follows
             from the rule, and each variant is checked against the source so
             the changed value is not present there by coincidence.

Synthetic diagnostics show which KINDS of error the verifier catches. They are
not clinical validation and do not replace human adjudication.

The safety-relevant class is UNSUPPORTED:
  unsupported recall     flagged / all truly unsupported
  unsupported precision  truly unsupported / all flagged
  false-support rate     truly unsupported among approved / all approved
"""

from __future__ import annotations

import re

from src.agents.citations import CITE_RE

SUPPORTED, UNSUPPORTED = "SUPPORTED", "UNSUPPORTED"
_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_NUM_RE = re.compile(r"(?<![\w.-])(\d+(?:\.\d+)?)(?![\w-]|\.\d)")
_DOSE_UNIT_RE = re.compile(r"\s*(mg|mcg|units?|mL|g)\b(?!/)", re.I)     # "mg", not "mg/dL"
_SWAPS = [("increased", "decreased"), ("rose", "fell"), ("higher", "lower")]
ADDED_CLAUSE = "and required transfer to intensive care for emergency dialysis"


def _sources(base: dict) -> str:
    return "\n".join(base.get("evidence", {}).values())


def _variant(base: dict, claim: str, category: str) -> dict:
    return {"claim": claim, "evidence": base["evidence"], "gold": UNSUPPORTED, "category": category,
            "synthetic": True, "origin": f"synthetic variant of a {base['origin']} claim"}


def numeric_mismatch(base: dict) -> dict | None:
    """Change the first number to one that does not occur in the source."""
    bare = _DATE_RE.sub(lambda m: " " * len(m.group()), CITE_RE.sub(lambda m: " " * len(m.group()), base["claim"]))
    src = _sources(base)
    for m in _NUM_RE.finditer(bare):
        value = m.group(1)
        decimals = len(value.split(".")[1]) if "." in value else 0
        for step in (1.7, 2.3, 3.9, 7.1, 11.3):
            new = f"{float(value) + step:.{decimals}f}" if decimals else str(int(value) + int(step * 4) + 1)
            if re.search(rf"(?<![\d.]){re.escape(new)}(?![\d])", src) or new in base["claim"]:
                continue
            claim = base["claim"][:m.start(1)] + new + base["claim"][m.end(1):]
            dose = _DOSE_UNIT_RE.match(bare[m.end(1):])
            return _variant(base, claim, "medication_dose_mismatch" if dose else "numeric_mismatch")
    return None


def date_mismatch(base: dict) -> dict | None:
    """Move the first date by three years, to a date the source does not contain."""
    m = _DATE_RE.search(base["claim"])
    if not m:
        return None
    new = f"{int(m.group(1)) + 3}-{m.group(2)}-{m.group(3)}"
    if new in _sources(base) or new in base["claim"]:
        return None
    return _variant(base, base["claim"][:m.start()] + new + base["claim"][m.end():], "temporal_mismatch")


def direction_mismatch(base: dict) -> dict | None:
    """Reverse the stated direction of a 'from A to B' claim whose direction was true."""
    if not re.search(r"\bfrom\b.*\bto\b", base["claim"]):
        return None
    for a, b in _SWAPS:
        for old, new in ((a, b), (b, a)):
            if re.search(rf"\b{old}\b", base["claim"]) and not re.search(rf"\b{new}\b", base["claim"]):
                return _variant(base, re.sub(rf"\b{old}\b", new, base["claim"], count=1), "direction_mismatch")
    return None


def temporal_swap(base: dict) -> dict | None:
    """Swap 'first' and 'most recent' (or 'most recent' and 'earliest') when they name different facts."""
    claim = base["claim"]
    if "first value was" in claim and "the most recent was" in claim:
        m = re.search(r"first value was (.+?) and the most recent was (.+?)(?= \[)", claim)
        if m and m.group(1) != m.group(2):
            return _variant(base, claim.replace(m.group(1), "\0").replace(m.group(2), m.group(1)).replace("\0", m.group(2)),
                            "temporal_swap")
    if base.get("has_other_dates") and re.match(r"The most recent\b.*\b(?:was|began)\b", claim) and "earliest" not in claim:
        return _variant(base, re.sub(r"\bmost recent\b", "earliest", claim, count=1), "temporal_swap")
    return None


def uncited(base: dict) -> dict | None:
    claim = " ".join(CITE_RE.sub("", base["claim"]).split()).replace(" .", ".")
    return _variant(base, claim, "uncited_claim") if claim != base["claim"] else None


def invalid_citation(base: dict) -> dict | None:
    """Cite a label that is not among the sources."""
    labels = CITE_RE.findall(base["claim"])
    if not labels:
        return None
    wrong = f"{labels[0][0]}9"
    if wrong in base["evidence"]:
        return None
    return _variant(base, CITE_RE.sub(f"[{wrong}]", base["claim"]), "uncited_claim")


def added_clause(base: dict) -> dict | None:
    """Append a factual clause the source says nothing about, inside the cited sentence."""
    src = _sources(base).lower()
    if "dialysis" in src or "intensive care" in src or not CITE_RE.search(base["claim"]):
        return None
    m = CITE_RE.search(base["claim"])
    return _variant(base, f"{base['claim'][:m.start()].rstrip()} {ADDED_CLAUSE} {base['claim'][m.start():]}",
                    "unsupported_added_clause")


TRANSFORMS = (numeric_mismatch, date_mismatch, direction_mismatch, temporal_swap, uncited, invalid_citation, added_clause)


def variants(base: dict) -> list[dict]:
    """Every adversarial variant that can be made from one supported claim."""
    out = [t(base) for t in TRANSFORMS]
    return [v for v in out if v is not None and v["claim"] != base["claim"]]


# -------------------------------------------------------------------- metrics ----
def _ratio(n: int, d: int):
    return round(n / d, 4) if d else None


def metrics(rows: list[dict]) -> dict:
    """`rows` need `gold` and `approved` (the verifier marked the claim supported)."""
    tp = sum(1 for r in rows if r["gold"] == UNSUPPORTED and not r["approved"])      # caught
    fn = sum(1 for r in rows if r["gold"] == UNSUPPORTED and r["approved"])          # false support
    fp = sum(1 for r in rows if r["gold"] == SUPPORTED and not r["approved"])        # false flag
    tn = sum(1 for r in rows if r["gold"] == SUPPORTED and r["approved"])
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else (0.0 if tp + fp + fn else None)
    return {"n": len(rows), "gold_supported": tn + fp, "gold_unsupported": tp + fn,
            "confusion": {"unsupported_flagged": tp, "unsupported_approved": fn,
                          "supported_flagged": fp, "supported_approved": tn},
            "unsupported_recall": recall, "unsupported_precision": precision, "unsupported_f1": f1,
            "supported_precision": _ratio(tn, tn + fn), "false_support_rate": _ratio(fn, tn + fn),
            "supported_recall": _ratio(tn, tn + fp)}


def by_category(rows: list[dict]) -> dict:
    """For each kind of synthetic error: how many were made and how many the verifier caught."""
    out: dict[str, dict] = {}
    for r in rows:
        if r["gold"] != UNSUPPORTED:
            continue
        c = out.setdefault(r["category"], {"n": 0, "caught": 0, "caught_by_code": 0, "caught_by_model": 0})
        c["n"] += 1
        if not r["approved"]:
            c["caught"] += 1
            c["caught_by_model" if r.get("method") == "fast_model" else "caught_by_code"] += 1
    for c in out.values():
        c["missed"] = c["n"] - c["caught"]
        c["recall"] = _ratio(c["caught"], c["n"])
    return dict(sorted(out.items()))


def render(summary: dict) -> str:
    def f(v):
        return "n/a" if v is None else f"{v:.3f}"
    o, cm = summary["overall"], summary["overall"]["confusion"]
    lines = ["LUMEN VERIFIER DIAGNOSTIC (baseline measurement; development data + synthetic diagnostics)", "",
             f"Examples                    {o['n']}   (real {summary['real']['n']}, synthetic adversarial {summary['synthetic']['n']})",
             f"Gold supported / unsupported {o['gold_supported']} / {o['gold_unsupported']}", "",
             f"Unsupported recall          {f(o['unsupported_recall'])}",
             f"Unsupported precision       {f(o['unsupported_precision'])}",
             f"Unsupported F1              {f(o['unsupported_f1'])}",
             f"Verifier-supported precision {f(o['supported_precision'])}",
             f"False-support rate          {f(o['false_support_rate'])}", "",
             "Confusion matrix            verifier flagged   verifier approved",
             f"  gold unsupported          {cm['unsupported_flagged']:>10d}   {cm['unsupported_approved']:>15d}",
             f"  gold supported            {cm['supported_flagged']:>10d}   {cm['supported_approved']:>15d}", "",
             f"Real supported claims approved: {summary['real']['confusion']['supported_approved']}/{summary['real']['n']}"
             f"   (human-adjudicated {summary['real_by_origin'].get('human_adjudicated', 'n/a')}, "
             f"SQL-templated {summary['real_by_origin'].get('structured_sql', 'n/a')})", "",
             "Synthetic error type          n   caught   by code   by model   missed"]
    for name, c in summary["by_category"].items():
        lines.append(f"  {name:26s} {c['n']:>3d} {c['caught']:>8d} {c['caught_by_code']:>9d} {c['caught_by_model']:>10d} {c['missed']:>8d}")
    return "\n".join(lines)
