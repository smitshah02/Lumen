"""
Citation Extraction and Validation
==================================
The 8B model WILL emit labels that were never in the evidence block —
[S7] when only [S1]-[S5] exist. Prompt rules do not reliably prevent
this, so it is enforced in code and measured, not assumed away.

Splits an answer into claims, attaches cited labels, and flags:
  - hallucinated labels (not in the evidence set)
  - uncited factual sentences
"""

from __future__ import annotations

import re

CITE_RE = re.compile(r"\[([SLGP]\d+)\]")   # P = published literature
# Sentence split that tolerates clinical abbreviations (mg., q.d., Dr.)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

_REFUSAL = "the available records do not contain enough information"


def extract_labels(text: str) -> list[str]:
    return CITE_RE.findall(text or "")


def split_claims(answer: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(answer.strip()) if s.strip()]


# A fragment made of nothing but citation markers (plus stray whitespace or a
# trailing period). Models emit these as a line of their own after the sentence
# they belong to — "The patient is taking furosemide.\n[S5]" — which the
# sentence splitter then reads as two claims: one uncited and one meaningless.
_CITE_ONLY_RE = re.compile(r"^(?:\s*\[[SLGP]\d+\]\s*)+\.?\s*$")
_PARA_RE = re.compile(r"(\n[ \t]*\n)")
_TRAILING_PUNCT_RE = re.compile(r"([.!?]+)\s*$")


def normalize_orphan_citations(answer: str) -> str:
    """Attach a citation-only fragment to the sentence IMMEDIATELY before it.

    Purely a formatting repair, applied before any claim is verified. It moves
    a marker the model already wrote onto the claim it already wrote it for.
    It never invents a marker, never searches the rest of the answer for one to
    rescue an uncited claim, and never reasons about whether a source supports
    anything.

    Deliberately narrow:
      * the fragment must consist only of citation markers
      * it attaches to the directly preceding claim, nothing further back
      * it never crosses a blank line, so an orphan opening a new paragraph
        stays where it is
      * with no preceding claim in the paragraph it is left alone
    A substantive sentence with no adjacent marker stays uncited, and an
    uncited claim is still an unsupported claim.
    """
    if not answer or "[" not in answer:
        return answer
    parts = _PARA_RE.split(answer)
    return "".join(p if _PARA_RE.fullmatch(p) else _normalize_paragraph(p) for p in parts)


def _attach(prev: str, fragment: str) -> str:
    """Move the fragment's markers onto `prev`, inside its trailing punctuation."""
    # Scope the duplicate check to the sentence being attached to. `prev` may be
    # a whole line of several sentences, and an [S1] belonging to an earlier one
    # must not suppress the [S1] the model wrote for this one.
    tail_claim = (split_claims(prev) or [prev])[-1]
    already = set(CITE_RE.findall(tail_claim))
    add = [f"[{m}]" for m in CITE_RE.findall(fragment) if m not in already]
    if not add:
        return prev
    m = _TRAILING_PUNCT_RE.search(prev)
    tail, base = (m.group(1), prev[:m.start()]) if m else ("", prev)
    return f"{base.rstrip()} {' '.join(add)}{tail}"


def _normalize_paragraph(text: str) -> str:
    if not text.strip():
        return text
    lead = text[:len(text) - len(text.lstrip())]

    # Pass 1, by line. A marker on a line of its own belongs to the line above.
    # This has to run before the sentence splitter, which only breaks after
    # .!? — so an orphan line was being absorbed into the sentence that FOLLOWS
    # it, crediting the next claim with a citation written for the previous one.
    lines, changed = [], False
    for line in text.splitlines():
        if lines and line.strip() and _CITE_ONLY_RE.match(line):
            prev_i = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), None)
            if prev_i is not None:
                lines[prev_i] = _attach(lines[prev_i], line)
                changed = True
                continue
        lines.append(line)
    body = "\n".join(lines)

    # Pass 2, by claim. Catches a trailing orphan on the same line, e.g.
    # "... heart failure. [S1] [S3]".
    claims = split_claims(body)
    merged: list[str] = []
    for claim in claims:
        if merged and _CITE_ONLY_RE.match(claim):
            merged[-1] = _attach(merged[-1], claim)
            changed = True
            continue
        merged.append(claim)
    if not changed:
        return text
    return lead + (" ".join(merged) if len(merged) != len(claims) else body)


def validate(answer: str, evidence: list[dict]) -> dict:
    """
    Returns:
      claims          list of {claim, labels, valid_labels, bad_labels, uncited}
      bad_labels      hallucinated labels across the whole answer
      is_refusal      the model correctly declined
      cite_rate       fraction of claims carrying >= 1 valid label
    """
    valid = {e["label"] for e in evidence}
    by_label = {e["label"]: e for e in evidence}

    is_refusal = _REFUSAL in (answer or "").lower()
    claims, all_bad = [], set()

    for sentence in split_claims(answer or ""):
        labels = extract_labels(sentence)
        good = [l for l in labels if l in valid]
        bad = [l for l in labels if l not in valid]
        all_bad.update(bad)
        claims.append({
            "claim": sentence,
            "labels": labels,
            "valid_labels": good,
            "bad_labels": bad,
            "uncited": len(labels) == 0,
            "source_text": by_label[good[0]]["text"] if good else None,
        })

    cited = sum(1 for c in claims if c["valid_labels"])
    return {
        "claims": claims,
        "bad_labels": sorted(all_bad),
        "is_refusal": is_refusal,
        "n_claims": len(claims),
        "cite_rate": (cited / len(claims)) if claims else 0.0,
    }


def strip_bad_labels(answer: str, evidence: list[dict]) -> str:
    """Remove hallucinated markers so they never reach a reader."""
    valid = {e["label"] for e in evidence}
    return CITE_RE.sub(lambda m: m.group(0) if m.group(1) in valid else "", answer or "")

def rebuild_answer(claims: list[dict], keep: set[int]) -> str:
    """Reassemble an answer from the claims that survived review."""
    kept = [c["claim"].strip() for i, c in enumerate(claims) if i in keep]
    return " ".join(kept).strip()