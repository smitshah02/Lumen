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

CITE_RE = re.compile(r"\[([SLG]\d+)\]")
# Sentence split that tolerates clinical abbreviations (mg., q.d., Dr.)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

_REFUSAL = "the available records do not contain enough information"


def extract_labels(text: str) -> list[str]:
    return CITE_RE.findall(text or "")


def split_claims(answer: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(answer.strip()) if s.strip()]


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