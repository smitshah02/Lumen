"""
Claim verification — deterministic first, one batched model call for the rest
============================================================================
Verification used to be the single largest source of serial latency: one MAIN
generation *per citation*, run in a loop. A five-sentence answer cost five
sequential 30B calls after synthesis had already finished.

Two changes, neither of which weakens grounding:

1. DETERMINISTIC PASS. A clinical claim's load-bearing content is its numbers
   and dates — "creatinine 1.4 mg/dL on 2024-03-18". Those are checkable in
   code against the cited source text, exactly and without judgement. If every
   numeric and date anchor in a claim appears verbatim in the text it cites,
   the claim is supported and no model is needed to say so.

   The pass is deliberately one-sided. It can only conclude "supported" or
   "unresolved" — never "unsupported". A missing number might be a legitimate
   derivation ("rose by 0.4"), and silently failing such a claim would route
   patients to human review for the wrong reason. Anything it cannot settle is
   handed to the model, which is what used to see every claim anyway.

   It also refuses to auto-support:
     - claims with no numeric or date anchor at all (pure prose assertions)
     - claims citing [G#]/[P#] — a guideline or paper is a general statement,
       and "does this apply to this patient" is a judgement, not a string match
     - claims with no citation at all

   Anchors are matched as QUANTITIES (value + unit) wherever a unit is present,
   so "lisinopril 5 mg" is checked as "5 mg" and cannot be waved through by an
   unrelated "5" elsewhere in the note. A claim is checked against the union of
   every source it cites, because "creatinine rose from 1.3 to 1.8 [S1][S2]" is
   supported by both chunks and by neither alone.

2. BATCHED FALLBACK. Whatever is left goes out in ONE call carrying each
   distinct source once, instead of N calls each re-sending the system prompt
   and a 3000-character source. Same verdict vocabulary, same strict procedure.

   Claims are numbered 1..n within the batch and mapped back by the caller;
   numbering them by their position in the whole answer made the model answer
   about claims that were not in the batch. Verdicts are salvaged from a
   truncated response rather than discarding the whole reply.

The caller's routing rule is untouched: unsupported > 0 still means human
review. Nothing here can turn "unresolved" into "supported".
"""

from __future__ import annotations

import re
import json
import logging

from src.agents.citations import CITE_RE

logger = logging.getLogger(__name__)

SOURCE_CHARS = 3000          # per source, as before
MAX_BATCH = 8                # claims per model call; an answer is <= 6 sentences
REASON_CHARS = 120           # cap the model's free text so the JSON cannot run long

# A date the notes actually render (ISO) — matched before numbers so the parts
# of "2024-03-18" are not mistaken for three separate numeric anchors.
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# A quantity: a number with the unit that gives it meaning. Checking "5 mg"
# rather than a bare "5" is what stops a claim asserting the wrong dose from
# being auto-supported because some unrelated "5" appears in the note.
_UNITS = (r"mg/dL|g/dL|mEq/L|mmol/L|ng/mL|pg/mL|uIU/mL|IU/L|K/uL|mcg/kg|mg/kg|units?|"
          r"mg|mcg|mL|L|kg|g|%|cm|mmHg|hours?|days?|weeks?|months?|years?|liters?")
_QTY_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*({_UNITS})\b", re.I)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# Bare integers with no unit attached: ordinals, counts, list numbering. A claim
# whose only number is one of these is handed to the model rather than matched
# on a digit that means nothing on its own. Numbers that DO carry a unit are
# never treated as trivial — that is the whole point of _QTY_RE.
_TRIVIAL = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}


def anchors(claim: str) -> tuple[set[str], set[str], set[str]]:
    """(dates, quantities, bare numbers) a claim asserts.

    Citation markers are stripped first — `[S1]` would otherwise contribute a
    phantom numeric anchor of "1". Quantities are normalised to "<value> <unit>"
    lowercase so "40mg" and "40 mg" are the same anchor.
    """
    text = CITE_RE.sub(" ", claim or "")
    dates = set(_DATE_RE.findall(text))
    rest = _DATE_RE.sub(" ", text)
    qty = {f"{m.group(1)} {m.group(2).lower()}" for m in _QTY_RE.finditer(rest)}
    quantified = {m.group(1) for m in _QTY_RE.finditer(rest)}
    nums = {n for n in _NUM_RE.findall(rest) if n not in _TRIVIAL and n not in quantified}
    return dates, qty, nums


def _number_in(value: str, source: str) -> bool:
    """Whole-number match: 1.4 must not be found inside 21.4 or 1.42."""
    return re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", source) is not None


def _quantity_in(qty: str, source: str) -> bool:
    """`40 mg` matches "40 mg", "40mg" or "40  MG", never "140 mg"."""
    value, unit = qty.split(" ", 1)
    pat = rf"(?<![\d.]){re.escape(value)}\s*{re.escape(unit)}(?![a-z])"
    return re.search(pat, source, re.I) is not None


def deterministic_verdict(claim: str, source_text: str, labels) -> tuple[str, str]:
    """Returns (verdict, note). verdict is "supported" or "unresolved" only.

    `labels` is every citation the claim carries; `source_text` must be the
    concatenation of those sources. A claim citing two chunks ("creatinine rose
    from 1.3 to 1.8 [S1][S2]") is supported by their union, and checking it
    against only the first one manufactured an escalation that was never real.
    """
    labels = [labels] if isinstance(labels, str) else list(labels or [])
    if not labels:
        return "unresolved", "no citation to check against"
    if any(l.startswith(("G", "P")) for l in labels):
        return "unresolved", "general-source claim needs judgement"
    dates, qty, nums = anchors(claim)
    if not dates and not qty and not nums:
        return "unresolved", "no numeric or date anchor to check"
    src = source_text or ""
    missing = ([d for d in dates if d not in src]
               + [q for q in qty if not _quantity_in(q, src)]
               + [n for n in nums if not _number_in(n, src)])
    if missing:
        return "unresolved", "anchor not found verbatim in source"
    n = len(dates) + len(qty) + len(nums)
    return "supported", f"deterministic: {n} anchor(s) matched verbatim in {'+'.join(labels)}"


# ---------------------------------------------------------------------------
# Batched model fallback
# ---------------------------------------------------------------------------
def build_batch_prompt(items: list[dict]) -> tuple[str, dict[int, int]]:
    """items: [{"i": global index, "claim": str, "labels": [str], "source": str}].

    Returns (prompt, batch_no -> global index).

    Claims are numbered 1..n *within the batch*. They used to be labelled with
    their global index, so a batch of the 2nd and 4th claims asked the model
    about "claim 1" and "claim 3" while telling it there were 2 — and the model
    answered 0 and 1. Every verdict was then either dropped as out of range or,
    worse, applied to the wrong claim. Contiguous local numbering removes the
    ambiguity entirely; the caller maps back.

    Each distinct source is printed once and referenced by its label, so a
    five-claim answer over two chunks sends two chunks, not five copies.
    """
    sources: dict[str, str] = {}
    for it in items:
        for label, text in zip(it["labels"], it.get("sources") or [it.get("source", "")]):
            sources.setdefault(label, (text or "")[:SOURCE_CHARS])

    parts = ["SOURCES:"]
    for label, text in sources.items():
        parts.append(f"[{label}]\n{text}\n")
    parts.append("CLAIMS:")
    mapping: dict[int, int] = {}
    for n, it in enumerate(items, 1):
        mapping[n] = it["i"]
        cited = ", ".join(f"[{l}]" for l in it["labels"]) or "(no citation)"
        parts.append(f"{n}. (cites {cited}) {it['claim']}")
    n_items = len(items)
    parts.append(f"\nReturn exactly {n_items} result{'' if n_items == 1 else 's'}, "
                 f"one for each claim number 1 to {n_items}.")
    return "\n".join(parts), mapping


_OBJ_RE = re.compile(r"\{[^{}]*\}")
_VALID_VERDICTS = {"supported", "partial", "unsupported"}


def _rows(raw: str) -> list[dict]:
    """Every JSON object in the response, whole or truncated.

    A clean parse is tried first. If the model ran past its token budget the
    array is cut mid-object, json.loads fails, and the old code returned {} —
    turning a whole answer's worth of supported claims into unsupported ones.
    Scanning for complete `{...}` objects salvages every verdict that did
    arrive; the incomplete tail is simply absent and fails safe as before.
    """
    try:
        body = json.loads(raw)
        rows = body.get("results") if isinstance(body, dict) else body
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    except Exception:
        pass
    out = []
    for m in _OBJ_RE.finditer(raw or ""):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    if out:
        logger.warning("batch verdicts salvaged from unparseable response (%d object(s))", len(out))
    return out


def parse_batch(raw: str, mapping: dict[int, int]) -> dict[int, tuple[str, str]]:
    """Map GLOBAL claim index -> (verdict, reason).

    `mapping` is batch_no -> global index from build_batch_prompt. A verdict for
    a number outside the batch is dropped. Anything the model omits stays
    absent, and the caller treats an absent verdict as unsupported — the same
    failure direction the per-claim loop had.
    """
    rows = _rows(raw)
    out: dict[int, tuple[str, str]] = {}
    unindexed = []
    for row in rows:
        verdict = str(row.get("verdict", "")).strip().lower()
        if verdict not in _VALID_VERDICTS:
            continue
        reason = str(row.get("reason", ""))[:200]
        n = row.get("i", row.get("index", row.get("claim", row.get("n"))))
        try:
            n = int(n)
        except (TypeError, ValueError):
            unindexed.append((verdict, reason))
            continue
        if n in mapping:
            out[mapping[n]] = (verdict, reason)

    # A model that returned the right number of verdicts but no usable indices
    # is reporting them in order. Accept that only on an exact count match, and
    # only when nothing was indexed — a partial mix could mis-attribute.
    if not out and unindexed and len(unindexed) == len(mapping):
        logger.warning("batch verdicts had no indices; assigning %d in order", len(unindexed))
        for n, (verdict, reason) in enumerate(unindexed, 1):
            out[mapping[n]] = (verdict, reason)
    return out
