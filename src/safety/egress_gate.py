"""
PHI Egress Gate
===============
Every outbound call to an external tool passes through here first.

The rule this enforces: patient-derived text may not leave the machine.
Abstracted clinical concepts may. "hyperkalemia management in CKD stage 4"
crosses; a note excerpt does not, and neither does any identifier.

Rules run in order and the FIRST match decides. Each rule is named so the
eval can report which one caught what — a gate whose blocks you cannot
attribute is a gate you cannot tune.

Blocked payloads are never logged. Only a SHA-256 of them is, which is
enough to prove two attempts were identical without retaining the text.
"""

from __future__ import annotations

import re
import hashlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Minimum length of a verbatim span that counts as leaked note text.
# Lower catches more but false-blocks ordinary clinical phrasing; 40 chars
# is roughly a full clause. Tuned by eval_egress.py, not by guesswork.
VERBATIM_MIN = 40
SHINGLE = 24          # sliding window for near-verbatim detection

# De-identification placeholders. Their presence PROVES the string came
# from a note — no human types "[PERSON]" into a literature query.
DEID_MARKERS = re.compile(r"\[(?:PERSON|DATE|LOCATION|PHONE|ID|AGE|EMAIL)\]|_{3,}")

# Structured identifiers that must never cross.
IDENTIFIERS = [
    (re.compile(r"\bsubject[_ ]?id\b", re.I), "subject_id"),
    (re.compile(r"\bhadm[_ ]?id\b", re.I), "hadm_id"),
    (re.compile(r"\bchunk[_ ]?id\b", re.I), "chunk_id"),
    (re.compile(r"\bnote[_ ]?id\b", re.I), "note_id"),
    (re.compile(r"\b\d{8}\b"), "8-digit id"),
]

# MIMIC dates are shifted ~100 years forward. A 21xx date is therefore a
# tell that the string came from the corpus rather than a person.
SHIFTED_DATE = re.compile(r"\b2[01]\d{2}-\d{2}-\d{2}\b")

# Clinical note structure markers.
NOTE_STRUCTURE = re.compile(
    r"(?:\bDISCHARGE LABS\b|\bINDICATION:|\bTECHNIQUE:|\bCOMPARISON:"
    r"|\bEXAMINATION:|\bIMPRESSION:|\bFINDINGS:|\bDischarge Medications\b"
    r"|\bDischarge Disposition\b|\bLevel of Consciousness\b|\bActivity Status\b)"
)

# Lab-result shorthand: "Creat-0.4", "K-5.4*", "Hgb-11.6*"
LAB_SHORTHAND = re.compile(r"\b[A-Z][A-Za-z]{1,8}-\d+\.?\d*\*?(?:\s|$)")

MAX_QUERY_CHARS = 300     # a concept query is short; a note excerpt is not


@dataclass
class EgressDecision:
    allowed: bool
    rule: str | None
    tool: str
    payload_sha256: str
    detail: str = ""

    def as_record(self) -> dict:
        return {"tool": self.tool, "allowed": self.allowed, "rule": self.rule,
                "payload_sha256": self.payload_sha256, "detail": self.detail}


@dataclass
class EgressGate:
    """Holds the patient text seen this session, to detect verbatim leakage."""
    corpus: set[str] = field(default_factory=set)      # shingles of seen text
    log: list[dict] = field(default_factory=list)

    def load_evidence(self, evidence: list[dict]) -> None:
        """Register retrieved patient text as protected. Call after every
        patient_retrieval so the gate knows what must not cross."""
        for e in evidence:
            self._add(e.get("text", ""))

    def _add(self, text: str) -> None:
        norm = _normalize(text)
        for i in range(0, max(len(norm) - SHINGLE + 1, 0)):
            self.corpus.add(norm[i:i + SHINGLE])

    # -- rules, in priority order ------------------------------------------

    def _check(self, s: str) -> tuple[str | None, str]:
        if len(s) > MAX_QUERY_CHARS:
            return "oversized_payload", f"{len(s)} chars > {MAX_QUERY_CHARS}"

        if DEID_MARKERS.search(s):
            return "deid_marker", "contains a de-identification placeholder"

        for pat, name in IDENTIFIERS:
            if pat.search(s):
                return "identifier", f"contains {name}"

        if SHIFTED_DATE.search(s):
            return "shifted_date", "contains a MIMIC date-shifted timestamp"

        if NOTE_STRUCTURE.search(s):
            return "note_structure", "contains clinical note section markers"

        if LAB_SHORTHAND.search(s):
            return "lab_shorthand", "contains raw lab-result shorthand"

        # Verbatim / near-verbatim overlap with retrieved patient text.
        norm = _normalize(s)
        if len(norm) >= SHINGLE and self.corpus:
            hits = sum(1 for i in range(len(norm) - SHINGLE + 1)
                       if norm[i:i + SHINGLE] in self.corpus)
            # Consecutive matching shingles of this many => a verbatim span
            if hits >= (VERBATIM_MIN - SHINGLE + 1):
                return "verbatim_overlap", f"{hits} shingles match retrieved patient text"

        return None, ""

    def check(self, tool: str, args: dict) -> EgressDecision:
        """Evaluate one outbound tool call. Never mutates args."""
        blob = " ".join(str(v) for v in args.values() if v is not None)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()

        rule, detail = self._check(blob)
        decision = EgressDecision(
            allowed=rule is None, rule=rule, tool=tool,
            payload_sha256=digest, detail=detail,
        )
        self.log.append(decision.as_record())

        if rule:
            # Log the hash and the rule. NEVER the payload.
            logger.warning(f"[egress] BLOCKED {tool} rule={rule} sha={digest[:12]} ({detail})")
        else:
            logger.info(f"[egress] allowed {tool} sha={digest[:12]}")
        return decision


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation. Makes shingle
    matching robust to the reformatting an LLM applies when it paraphrases."""
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower())


def call_external(gate: EgressGate, tool_name: str, args: dict, fn):
    """Wrap any external tool. Blocked calls never reach fn()."""
    decision = gate.check(tool_name, args)
    if not decision.allowed:
        return {"blocked": True, "rule": decision.rule, "detail": decision.detail,
                "guidance": "Rephrase as a general clinical concept with no "
                            "patient text, identifiers, or dates."}
    return fn(**args)