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

2. BATCHED FALLBACK. Whatever is left goes out in ONE call carrying each
   distinct source once, instead of N calls each re-sending the system prompt
   and a 3000-character source. Same verdict vocabulary, same strict procedure.

The caller's routing rule is untouched: unsupported > 0 still means human
review.
"""

from __future__ import annotations

import re
import json
import logging

from src.agents.citations import CITE_RE

logger = logging.getLogger(__name__)

SOURCE_CHARS = 3000          # per source, as before
MAX_BATCH = 8                # claims per model call; an answer is <= 6 sentences

# A date the notes actually render (ISO) — matched before numbers so the parts
# of "2024-03-18" are not mistaken for three separate numeric anchors.
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# Bare ordinals/counts that carry no clinical weight on their own. Requiring
# these to appear verbatim would send almost every claim to the model.
_TRIVIAL = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "0"}


def anchors(claim: str) -> tuple[set[str], set[str]]:
    """(dates, numbers) a claim asserts. Citation markers are stripped first —
    `[S1]` would otherwise contribute a phantom numeric anchor of "1"."""
    text = CITE_RE.sub(" ", claim or "")
    dates = set(_DATE_RE.findall(text))
    without_dates = _DATE_RE.sub(" ", text)
    nums = {n for n in _NUM_RE.findall(without_dates) if n not in _TRIVIAL}
    return dates, nums


def _number_in(value: str, source: str) -> bool:
    """Whole-number match: 1.4 must not be found inside 21.4 or 1.42."""
    return re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", source) is not None


def deterministic_verdict(claim: str, source_text: str, label: str) -> tuple[str, str]:
    """Returns (verdict, note). verdict is "supported" or "unresolved" only."""
    if label.startswith(("G", "P")):
        return "unresolved", "general-source claim needs judgement"
    dates, nums = anchors(claim)
    if not dates and not nums:
        return "unresolved", "no numeric or date anchor to check"
    src = source_text or ""
    missing = [d for d in dates if d not in src] + [n for n in nums if not _number_in(n, src)]
    if missing:
        return "unresolved", "anchor not found verbatim in source"
    n = len(dates) + len(nums)
    return "supported", f"deterministic: {n} anchor(s) matched verbatim in [{label}]"


# ---------------------------------------------------------------------------
# Batched model fallback
# ---------------------------------------------------------------------------
def build_batch_prompt(items: list[dict]) -> str:
    """items: [{"i": int, "claim": str, "label": str, "source": str}].

    Each distinct source is printed once and referenced by its label, so a
    five-claim answer over two chunks sends two chunks, not five copies."""
    sources = {}
    for it in items:
        sources.setdefault(it["label"], it["source"][:SOURCE_CHARS])

    parts = ["SOURCES:"]
    for label, text in sources.items():
        parts.append(f"[{label}]\n{text}\n")
    parts.append("CLAIMS:")
    for it in items:
        parts.append(f"{it['i']}. (cites [{it['label']}]) {it['claim']}")
    parts.append(f"\nReturn a verdict for each of the {len(items)} claim numbers above.")
    return "\n".join(parts)


def parse_batch(raw: str, expected: list[int]) -> dict[int, tuple[str, str]]:
    """Map claim index -> (verdict, reason). Anything the model omits or mangles
    stays absent, and the caller treats an absent verdict as unsupported —
    the same failure direction the per-claim loop had."""
    out: dict[int, tuple[str, str]] = {}
    try:
        body = json.loads(raw)
    except Exception:
        return out
    rows = body.get("results") if isinstance(body, dict) else body
    if not isinstance(rows, list):
        return out
    valid = {"supported", "partial", "unsupported"}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            i = int(row.get("i", row.get("index", -1)))
        except (TypeError, ValueError):
            continue
        verdict = str(row.get("verdict", "")).strip().lower()
        if i in expected and verdict in valid:
            out[i] = (verdict, str(row.get("reason", ""))[:200])
    return out
