"""
Phase 4 — independent offline LLM-as-a-Judge  (prompt version aj2)
=================================================================
A second, independent opinion on answer quality, scored per dimension.

Why aj2 exists
--------------
aj1 asked for six holistic scores directly. On the 3-case smoke it returned
completeness=4 for demo_q19 with the reason "includes all required facts",
while the answer omitted the required `insurance interruption` fact entirely.
A model asked for a summary judgement produces a summary judgement: it never
had to look at the required facts one at a time, so nothing in the output could
disagree with itself.

aj2 makes that structurally impossible to repeat:

  1. every required criterion is supplied with an id and must come back with a
     status (supported / partially_supported / missing / contradicted /
     not_applicable), the evidence labels behind it and a reason;
  2. the holistic dimensions are assigned ONLY after that, and the prompt says
     so explicitly;
  3. `check_consistency` then verifies the dimensions against the judge's OWN
     criterion assessments. completeness=4 alongside a `missing` criterion is
     not a low-quality verdict — it is an incoherent one, and it is refused.

An incoherent verdict is returned to the model once, with the specific
contradictions named, and re-judged. If the repair is still incoherent the row
is recorded as `judge_inconsistent`: the scores are preserved exactly as the
judge produced them, the case is NOT counted as passed, and no score is
clamped, rewritten or zeroed. A judge that cannot reason consistently is a
judge failure, not a low grade for the system under test.

Independence
------------
The judge NEVER sees, and this module never assembles into a prompt:
  * the runtime verifier's verdicts, reasons or per-claim notes
  * `verified` flags on citations
  * needs_human_review / review_status / escalation_reason
  * any deterministic check result
  * any prior judge output or aggregate score

`build_prompt` builds from an explicit allowlist, and `forbidden_keys_present`
exists so a test can assert the rendered prompt is clean rather than trusting
that it is. The patient identifier is deliberately NOT supplied: it is not
needed to judge an answer against its evidence.

Failure handling
----------------
A judge that errors, returns unparseable output, or contradicts itself twice is
recorded with a status of `backend_error`, `parse_error` or `judge_inconsistent`
and is counted as a judge failure. It is never converted into a 0 or a pass.
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from src.evals.final_eval.judge_backend import JudgeUnavailable

logger = logging.getLogger(__name__)

# Bumped from aj1: the schema, the prompt and the consistency contract all
# changed, so a cached aj1 verdict must never be served for an aj2 request.
# JudgeCache.key includes this string for exactly that reason.
JUDGE_PROMPT_VERSION = "aj2"

DEFAULT_CACHE_PATH = os.environ.get(
    "LUMEN_JUDGE_CACHE", str(Path(__file__).resolve().parents[3] / ".cache" / "final_eval_judge.json"))

DIMENSIONS = ("factual_correctness", "groundedness", "completeness",
              "answer_relevance", "temporal_correctness", "abstention_quality")

EVIDENCE_CHARS = 2400        # per source, mirrors the runtime verifier's budget

# --- criterion vocabulary --------------------------------------------------
CRITERION_STATUSES = ("supported", "partially_supported", "missing",
                      "contradicted", "not_applicable")
# Statuses that mean the criterion was NOT fully met. Any of these forbids a
# perfect completeness score — that is the whole point of aj2.
UNSATISFIED = ("partially_supported", "missing", "contradicted")

# Criterion kinds. `fact` comes from the gold expected_facts; the others are
# derived from the gold case's own flags, never invented per run.
KIND_FACT, KIND_TEMPORAL, KIND_ABSTENTION, KIND_AMBIGUITY = (
    "fact", "temporal", "abstention", "ambiguity")

TOP_SCORE = 4


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Criterion:
    """One thing the judge must individually rule on before scoring anything.

    `id` is what the verdict is matched back on, so a model that paraphrases
    the criterion text cannot silently drop one.
    """
    id: str
    text: str
    kind: str


def criteria_for(case) -> list[Criterion]:
    """The required criteria for a case, derived from GOLD only.

    Deliberately excludes `must_not_contain`: a prohibition inverts the meaning
    of every status word ("supported" would mean the answer did the forbidden
    thing), and mixing the two vocabularies is exactly the kind of ambiguity
    that produces an incoherent verdict. Prohibited content is stated in the
    prompt as MUST NOT ASSERT and is checked deterministically, where it is a
    hard gate rather than a graded opinion.
    """
    out: list[Criterion] = []
    for i, f in enumerate(case.expected_facts, 1):
        out.append(Criterion(id=f"c{i}", text=f.text, kind=KIND_FACT))
    if case.temporal_applicable:
        detail = ("the single most recent value, identified as the latest"
                  if case.temporal == "latest" else
                  "every time point in the sequence, in chronological order")
        out.append(Criterion(
            id=f"t{len(out) + 1}",
            text=f"The answer must give {detail}, judged within this patient's own "
                 f"record rather than against today's date.",
            kind=KIND_TEMPORAL))
    if case.expects_abstention:
        out.append(Criterion(
            id=f"a{len(out) + 1}",
            text="The record does not contain the answer. The response must say so "
                 "and must not state a value or cite evidence for one.",
            kind=KIND_ABSTENTION))
    if case.expects_ambiguity:
        out.append(Criterion(
            id=f"a{len(out) + 1}",
            text="The record is conflicting or unresolved on this point. The response "
                 "must state that uncertainty rather than asserting one side as fact. "
                 "An outright refusal also satisfies this; the exact refusal wording "
                 "is not required.",
            kind=KIND_AMBIGUITY))
    return out


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class CriterionAssessment(BaseModel):
    criterion_id: str
    criterion: str = ""
    status: Literal["supported", "partially_supported", "missing",
                    "contradicted", "not_applicable"]
    evidence_labels: list[str] = Field(default_factory=list)
    reason: str = ""


class Dimension(BaseModel):
    score: Optional[int] = Field(None, ge=0, le=4)
    applicable: bool = True
    reason: str = ""
    criterion: Optional[str] = None
    confidence: Optional[str] = None


class JudgeVerdict(BaseModel):
    criterion_assessments: list[CriterionAssessment] = Field(default_factory=list)
    unsupported_content: list[str] = Field(default_factory=list)
    factual_correctness: Dimension
    groundedness: Dimension
    completeness: Dimension
    answer_relevance: Dimension
    temporal_correctness: Dimension
    abstention_quality: Dimension


def _dim_schema() -> dict:
    return {"type": "object", "properties": {
        "score": {"type": ["integer", "null"], "minimum": 0, "maximum": 4},
        "applicable": {"type": "boolean"},
        "reason": {"type": "string"},
        "criterion": {"type": ["string", "null"]},
        "confidence": {"type": ["string", "null"], "enum": ["low", "medium", "high", None]},
    }, "required": ["score", "applicable", "reason"]}


# Written out flat rather than via model_json_schema(): Ollama's structured
# output does not resolve $ref/$defs, and a schema it cannot parse is silently
# ignored, which would drop the constraint without telling anyone.
OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "criterion_assessments": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "criterion_id": {"type": "string"},
                "criterion": {"type": "string"},
                "status": {"type": "string", "enum": list(CRITERION_STATUSES)},
                "evidence_labels": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["criterion_id", "status", "reason"]}},
        "unsupported_content": {"type": "array", "items": {"type": "string"}},
        **{d: _dim_schema() for d in DIMENSIONS},
    },
    "required": ["criterion_assessments", "unsupported_content", *DIMENSIONS],
}


SYSTEM_PROMPT = """You are an independent clinical answer evaluator. You grade a generated answer against a reference answer and the evidence that was cited to produce it.

You are NOT the system that produced the answer. You have no information about how it was produced, whether it was checked, or whether anyone flagged it. Judge only what you are shown.

Work in three steps, in this order. Do not skip step 1.

STEP 1 — ASSESS EVERY CRITERION INDIVIDUALLY.
You are given a numbered list of REQUIRED CRITERIA. For EACH one, emit an object in "criterion_assessments" with the criterion's exact id and one status:
  supported            the answer states this, and the cited evidence backs it
  partially_supported  the answer gestures at it but is incomplete, vague or imprecise
  missing              the answer does not state this at all
  contradicted         the answer states something incompatible with it
  not_applicable       this criterion genuinely does not apply to this question
Quote in "reason" the words from the answer that decided the status, or write "not present in the answer". Put the labels of the evidence that supports it in "evidence_labels" (empty list if none).
Assess EVERY criterion you are given. A criterion you do not mention is treated as an evaluation failure, not as a pass.

STEP 2 — LIST UNSUPPORTED CONTENT.
In "unsupported_content", list any material factual statement the answer makes that the CITED EVIDENCE does not establish. Clinically plausible content that is not in the evidence belongs in this list. Empty list if there is none.

STEP 3 — ONLY NOW ASSIGN THE DIMENSION SCORES, 0-4:
  4  fully correct / fully grounded / fully complete; nothing to qualify
  3  correct with a minor omission or imprecision that does not mislead
  2  partially correct; a material element is missing, imprecise or overstated
  1  mostly incorrect or largely ungrounded; at most an incidental element is right
  0  incorrect, fabricated, or contradicted by the cited evidence

  factual_correctness  Do the answer's assertions match the REFERENCE ANSWER and the criteria? A correct value stated in different words or units is still correct. Minor non-misleading imprecision is 3, not 2.
  groundedness         Is every assertion established by the CITED EVIDENCE? Judge strictly against the evidence shown, not against your own medical knowledge. Distinguish supported, reasonably inferred, unsupported and contradicted.
  completeness         Coverage of the REQUIRED CRITERIA. This score MUST follow your own step-1 assessments; fluent phrasing is not coverage.
  answer_relevance     Does it answer the question asked, without drifting? A concise answer is not penalised; an answer padded with unrequested material is.
  temporal_correctness Latest value, chronological order, event-to-admission relationship and the requested window, judged inside this patient's record. Never assume today's date.
  abstention_quality   Where the record does not support an answer, or is conflicting, does the answer say so honestly instead of inventing certainty?

THESE SCORES MUST AGREE WITH YOUR STEP-1 ASSESSMENTS:
- completeness = 4 is only valid when NO criterion is missing, partially_supported or contradicted.
- factual_correctness = 4 is only valid when NO criterion is contradicted.
- groundedness = 4 is only valid when "unsupported_content" is empty.
- temporal_correctness = 4 is only valid when no temporal criterion is missing or contradicted.
- abstention_quality = 4 is only valid when no abstention or ambiguity criterion is missing, partially_supported or contradicted.
A verdict that breaks one of these rules will be rejected and sent back to you.

Rules:
- The APPLICABLE DIMENSIONS list tells you which dimensions to score. For any dimension not on that list, return {"score": null, "applicable": false, "reason": "not applicable"}.
- "criterion" on a dimension should name the specific criterion the score keys to, when one applies.
- Every "reason" must be at most 200 characters.
- Return ONLY the JSON object. No prose, no markdown."""


REPAIR_INSTRUCTION = """Your previous verdict was internally inconsistent: the dimension scores contradict your own criterion assessments.

{violations}

Re-read your criterion assessments and the consistency rules, then return a corrected JSON object in the same schema. Correct whichever side is wrong — if a criterion really is met, fix the assessment; if it is not, fix the score. Do not change an assessment you believe is right merely to raise a score. Return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Prompt assembly (allowlist)
# ---------------------------------------------------------------------------
# State that would leak the system's own assessment into the judge. These are
# snake_case field names and sentinel values, which cannot occur in ordinary
# prose — so a hit is always real leakage, never an innocent English word like
# "judge" or "flagged" in the instructions.
FORBIDDEN_IDENTIFIERS = (
    "needs_human_review", "review_status", "escalation_reason", "verification_note",
    "verification_trace", "citation_report", "deterministic_verdict", "fast_verdict",
    "case_pass", "review_worthy", "is_review_worthy", "auto_approved",
    "human_review_required", "n_flagged_claims", "deterministic_lab_path",
    "sent_to_fast_verifier", "synthesis_failed", "subject_id",
    "failure_tags", "judge_prompt_version", "consistency_violations",
)
# Bare words that ARE legitimate English but would be leakage in key position
# ("verified: true"). Checked only where they look like a field, not prose.
FORBIDDEN_FIELDLIKE = ("verified", "unsupported", "flagged", "escalated")


def applicable_dimensions(case) -> list:
    """Decided from the GOLD case, never by the judge, so applicability counts
    are deterministic and identical across runs."""
    dims = ["factual_correctness", "groundedness", "completeness", "answer_relevance"]
    if case.temporal_applicable:
        dims.append("temporal_correctness")
    if case.expects_abstention or case.expects_ambiguity:
        dims.append("abstention_quality")
    return dims


def cited_evidence(row: dict, evidence: list) -> list:
    """Only the evidence the answer actually cites, in label order.

    Falls back to all retrieved evidence when the answer cites nothing, so an
    uncited answer is still judged for groundedness against what was available
    rather than against an empty block.
    """
    by_label = {e.get("label"): (e.get("text") or "") for e in (evidence or [])}
    cited = []
    for c in row.get("citations") or []:
        for l in ([c.get("label")] if c.get("label") else []) + list(c.get("labels") or []):
            if l and l not in cited:
                cited.append(l)
    labels = cited or list(by_label)
    return [{"label": l, "text": by_label.get(l, "")[:EVIDENCE_CHARS]}
            for l in labels if l in by_label]


def build_prompt(case, row: dict, evidence: list) -> tuple:
    """(system, user, applicable dimensions, criteria). Explicit allowlist."""
    dims = applicable_dimensions(case)
    crits = criteria_for(case)
    ev = cited_evidence(row, evidence)

    if case.expects_abstention:
        expectation = ("The record does NOT contain the answer. A correct response declines "
                       "and does not state a value or cite evidence for one.")
    elif case.expects_ambiguity:
        expectation = ("The record is conflicting or unresolved on this point. A correct "
                       "response states the uncertainty rather than asserting certainty. It "
                       "does not have to refuse outright.")
    else:
        expectation = "The record supports an answer. A correct response gives it."

    parts = [
        f"QUESTION:\n{case.query}",
        f"\nANSWERABILITY EXPECTATION:\n{expectation}",
        f"\nREFERENCE ANSWER:\n{case.expected_answer}",
        "\nREQUIRED CRITERIA (assess every one of these, by id):\n" + (
            "\n".join(f"  {c.id} [{c.kind}] {c.text}" for c in crits)
            or "  (none specified)"),
    ]
    if case.must_not_contain:
        parts.append("\nMUST NOT ASSERT:\n" + "\n".join(f"  - {t}" for t in case.must_not_contain))
    parts.append("\nCITED EVIDENCE:\n" + ("\n\n".join(f"[{e['label']}]\n{e['text']}" for e in ev)
                                          if ev else "  (the answer cites no evidence)"))
    parts.append(f"\nGENERATED ANSWER:\n{row.get('answer') or '(empty)'}")
    parts.append("\nAPPLICABLE DIMENSIONS:\n  " + ", ".join(dims))
    parts.append("\nAssess every criterion first, then return the JSON object.")
    return SYSTEM_PROMPT, "\n".join(parts), dims, crits


_EVIDENCE_HEADER = "\nCITED EVIDENCE:\n"


def prompt_without_evidence(user: str) -> str:
    """The prompt minus the cited-evidence block.

    The evidence is the frozen system's own rendered output and is passed
    VERBATIM, because groundedness has to be judged against exactly what the
    model saw. The deterministic lab path renders "Source: labevents table,
    subject <id>." inside that text (src/agents/graph.py:lab_lookup), so a
    subject id can appear there. Redacting it would change the thing being
    judged. No subject id is supplied as a field of its own, and everything
    outside the evidence block is held to the full independence check.
    """
    head, sep, tail = (user or "").partition(_EVIDENCE_HEADER)
    if not sep:
        return user or ""
    _, _, after = tail.partition("\nGENERATED ANSWER:\n")
    return head + ("\nGENERATED ANSWER:\n" + after if after else "")


def forbidden_keys_present(text: str) -> list:
    """Which forbidden state identifiers appear in a rendered prompt. Used by
    tests to prove independence rather than assume it."""
    low = (text or "").lower()
    hits = [k for k in FORBIDDEN_IDENTIFIERS if k in low]
    for w in FORBIDDEN_FIELDLIKE:
        if re.search(rf'(?<![a-z_])"?{w}"?\s*[:=]', low):
            hits.append(w)
    return hits


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I | re.M)


def extract_json(raw: str) -> Optional[dict]:
    """First balanced JSON object in the response.

    Same defence as src/evals/llm_judge.py: strip fences, try a clean parse,
    then scan for the first balanced object so a model that prefixes prose or
    runs past its token budget does not cost the whole verdict.
    """
    if not raw:
        return None
    text = _FENCE_RE.sub("", raw).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    start = None
    return None


def _norm_assessments(obj: dict) -> list:
    """Criterion assessments, keeping only well-formed rows with a known status.

    A row whose status is not in the vocabulary is DROPPED rather than coerced:
    the criterion then reads as unassessed, which is a consistency violation and
    triggers the repair. Guessing what an unknown status meant would defeat the
    check this whole module exists for.
    """
    out = []
    for a in (obj.get("criterion_assessments") or []):
        if not isinstance(a, dict):
            continue
        status = str(a.get("status") or "").strip().lower()
        cid = str(a.get("criterion_id") or a.get("id") or "").strip()
        if not cid or status not in CRITERION_STATUSES:
            continue
        labels = a.get("evidence_labels") or []
        out.append({
            "criterion_id": cid,
            "criterion": str(a.get("criterion") or "")[:200],
            "status": status,
            "evidence_labels": [str(l)[:12] for l in labels if isinstance(l, (str, int))][:12],
            "reason": str(a.get("reason") or "")[:200],
        })
    return out


def parse_verdict(raw: str) -> tuple:
    """(JudgeVerdict | None, error). Never raises, never invents a score."""
    obj = extract_json(raw)
    if obj is None:
        return None, "no JSON object in response"
    normalized = {
        "criterion_assessments": _norm_assessments(obj),
        "unsupported_content": [str(u)[:200] for u in (obj.get("unsupported_content") or [])
                                if isinstance(u, (str, int, float))][:20],
    }
    for d in DIMENSIONS:
        v = obj.get(d)
        if not isinstance(v, dict):
            normalized[d] = {"score": None, "applicable": False,
                             "reason": "dimension missing from judge response"}
            continue
        score = v.get("score")
        if isinstance(score, str):
            score = int(score) if score.strip().isdigit() else None
        if isinstance(score, bool) or not isinstance(score, int):
            score = None
        if score is not None and not 0 <= score <= 4:
            score = None
        normalized[d] = {
            "score": score,
            "applicable": bool(v.get("applicable", score is not None)),
            "reason": str(v.get("reason", ""))[:200],
            "criterion": (str(v["criterion"])[:120] if v.get("criterion") else None),
            "confidence": (str(v["confidence"]).lower()[:10] if v.get("confidence") else None),
        }
    try:
        return JudgeVerdict(**normalized), None
    except ValidationError as e:
        return None, f"schema validation failed: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------
def _score(dims: dict, name: str):
    d = dims.get(name) or {}
    s = d.get("score")
    return s if isinstance(s, int) and not isinstance(s, bool) else None


def check_consistency(assessments: list, dims: dict, criteria: list,
                      unsupported_content: list, applicable: list) -> list:
    """Where the verdict contradicts itself. Empty list means coherent.

    This never changes a score. It only reports, so the caller can ask the
    model to resolve the contradiction itself and, failing that, record the
    verdict as a judge failure with its numbers intact.
    """
    by_id = {a["criterion_id"]: a for a in assessments}
    kind_of = {c.id: c.kind for c in criteria}
    v: list = []

    # Every supplied criterion must have been assessed. This is what forces the
    # model to look at each one; without it, omission is free.
    unassessed = [c.id for c in criteria if c.id not in by_id]
    if unassessed:
        v.append({"rule": "criterion_not_assessed",
                  "detail": f"no assessment returned for criterion(s): {', '.join(unassessed)}"})

    considered = [a for a in assessments
                  if a["criterion_id"] in kind_of and a["status"] != "not_applicable"]
    unmet = [a for a in considered if a["status"] in UNSATISFIED]
    contradicted = [a for a in considered if a["status"] == "contradicted"]

    def _names(rows):
        return ", ".join(f"{a['criterion_id']}={a['status']}" for a in rows)

    if "completeness" in applicable and _score(dims, "completeness") == TOP_SCORE and unmet:
        v.append({"rule": "completeness_inflated",
                  "detail": f"completeness=4 but criteria are not fully met: {_names(unmet)}"})

    if "factual_correctness" in applicable and \
            _score(dims, "factual_correctness") == TOP_SCORE and contradicted:
        v.append({"rule": "factual_correctness_inflated",
                  "detail": f"factual_correctness=4 but criteria are contradicted: "
                            f"{_names(contradicted)}"})

    if "groundedness" in applicable and _score(dims, "groundedness") == TOP_SCORE \
            and unsupported_content:
        v.append({"rule": "groundedness_inflated",
                  "detail": "groundedness=4 but the verdict lists content unsupported by the "
                            f"cited evidence: {unsupported_content[0][:120]!r}"
                            + (f" (+{len(unsupported_content) - 1} more)"
                               if len(unsupported_content) > 1 else "")})

    temporal_bad = [a for a in considered
                    if kind_of.get(a["criterion_id"]) == KIND_TEMPORAL
                    and a["status"] in ("missing", "contradicted")]
    if "temporal_correctness" in applicable and \
            _score(dims, "temporal_correctness") == TOP_SCORE and temporal_bad:
        v.append({"rule": "temporal_correctness_inflated",
                  "detail": f"temporal_correctness=4 but a temporal criterion is not met: "
                            f"{_names(temporal_bad)}"})

    abst_bad = [a for a in considered
                if kind_of.get(a["criterion_id"]) in (KIND_ABSTENTION, KIND_AMBIGUITY)
                and a["status"] in UNSATISFIED]
    if "abstention_quality" in applicable and \
            _score(dims, "abstention_quality") == TOP_SCORE and abst_bad:
        v.append({"rule": "abstention_quality_inflated",
                  "detail": f"abstention_quality=4 but the abstention/ambiguity criterion is "
                            f"not met: {_names(abst_bad)}"})
    return v


def unknown_evidence_labels(assessments: list, evidence: list) -> list:
    """Labels the judge cited that were not shown to it.

    Recorded as a finding on the row, not as an inconsistency: a stray label is
    a judge-quality signal, but it does not make the verdict self-contradictory
    and should not consume the single repair attempt.
    """
    known = {e["label"] for e in evidence}
    seen: list = []
    for a in assessments:
        for l in a.get("evidence_labels") or []:
            if l not in known and l not in seen:
                seen.append(l)
    return seen


def _repair_user(user: str, violations: list) -> str:
    bullets = "\n".join(f"  - {v['detail']}" for v in violations)
    return user + "\n\n" + REPAIR_INSTRUCTION.format(violations=bullets)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class JudgeCache:
    """SHA-256 disk cache. A key covers everything that can change a verdict, so
    a hit is always a verdict for exactly this input."""

    def __init__(self, path: str | None = DEFAULT_CACHE_PATH):
        self.path = Path(path) if path else None
        self._data = {}
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                logger.warning("judge cache at %s unreadable; starting empty", self.path)

    @staticmethod
    def key(*, prompt_version: str, model: str, query_id: str, query: str, answer: str,
            evidence: list, criteria: dict) -> str:
        payload = "\x00".join([
            prompt_version, model, query_id, query, answer or "",
            "|".join(f"{e['label']}:{hashlib.sha256((e['text'] or '').encode()).hexdigest()}"
                     for e in evidence),
            json.dumps(criteria, sort_keys=True, ensure_ascii=False),
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, k: str):
        return self._data.get(k)

    def put(self, k: str, value: dict) -> None:
        self._data[k] = value
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False))
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------
def criteria_of(case) -> dict:
    """The gold content that steers the verdict — part of the cache key, so a
    gold edit can never be served a stale judgement."""
    return {"expected_answer": case.expected_answer,
            "expected_facts": [f.text for f in case.expected_facts],
            "must_not_contain": list(case.must_not_contain),
            "unsupported": case.unsupported, "answer_type": case.answer_type,
            "temporal": case.temporal,
            "criteria": [{"id": c.id, "kind": c.kind, "text": c.text}
                         for c in criteria_for(case)]}


def _gold_gated(dumped: dict, dims: list) -> dict:
    """Gold decides applicability, not the judge: a score for a dimension the
    case does not exercise is discarded rather than averaged in."""
    for d in DIMENSIONS:
        if d not in dims:
            dumped[d] = {"score": None, "applicable": False,
                         "reason": "not applicable for this case",
                         "criterion": None, "confidence": None}
        elif dumped[d]["score"] is None:
            dumped[d]["applicable"] = False
            dumped[d]["reason"] = dumped[d]["reason"] or "judge returned no score"
    return dumped


def judge_case(backend, cache: JudgeCache, case, row: dict, evidence: list) -> dict:
    """One case. Always returns a row; failures are recorded, never scored."""
    system, user, dims, crits = build_prompt(case, row, evidence)
    ev = cited_evidence(row, evidence)
    key = JudgeCache.key(prompt_version=JUDGE_PROMPT_VERSION, model=backend.model,
                         query_id=case.query_id, query=case.query,
                         answer=row.get("answer") or "", evidence=ev,
                         criteria=criteria_of(case))
    base = {"query_id": case.query_id, "judge_model": backend.model,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "applicable_dimensions": dims, "cache_key": key,
            "n_evidence": len(ev),
            "n_criteria": len(crits),
            "criteria": [{"id": c.id, "kind": c.kind, "text": c.text} for c in crits]}

    hit = cache.get(key)
    if hit is not None:
        return {**base, **hit, "cached": True}

    if row.get("status") == "evaluator_error":
        out = {"status": "skipped_no_answer", "error": "case failed during collection",
               "dimensions": _null_dims("case produced no answer to judge"),
               "criterion_assessments": [], "unsupported_content": [],
               "consistency_violations": [], "repair_attempted": False}
        return {**base, **out, "cached": False}

    attempt, verdict, err, violations = 1, None, None, []
    try:
        raw = backend.complete(system, user, OLLAMA_SCHEMA)
    except JudgeUnavailable as e:
        # Explicit, uncached: the judge did not run. Never a score, never the
        # runtime model standing in for it.
        return {**base, "status": "backend_error", "error": str(e)[:300],
                "dimensions": _null_dims("judge backend unavailable"),
                "criterion_assessments": [], "unsupported_content": [],
                "consistency_violations": [], "repair_attempted": False, "cached": False}

    verdict, err = parse_verdict(raw)
    if verdict is not None:
        d = verdict.model_dump()
        violations = check_consistency(d["criterion_assessments"], d, crits,
                                       d["unsupported_content"], dims)
        if violations:
            # One repair attempt, naming the exact contradictions. The model
            # resolves them itself; nothing here rewrites a score.
            attempt = 2
            try:
                raw2 = backend.complete(system, _repair_user(user, violations), OLLAMA_SCHEMA)
            except JudgeUnavailable as e:
                return {**base, "status": "backend_error",
                        "error": f"repair attempt failed: {str(e)[:260]}",
                        "dimensions": _null_dims("judge backend unavailable during repair"),
                        "criterion_assessments": d["criterion_assessments"],
                        "unsupported_content": d["unsupported_content"],
                        "consistency_violations": violations,
                        "repair_attempted": True, "cached": False}
            v2, err2 = parse_verdict(raw2)
            if v2 is not None:
                verdict, err = v2, None
                d = verdict.model_dump()
                violations = check_consistency(d["criterion_assessments"], d, crits,
                                               d["unsupported_content"], dims)
            else:
                err = f"repair response unparseable: {err2}"
                verdict = None

    if verdict is None:
        out = {"status": "parse_error", "error": err,
               "dimensions": _null_dims(f"unparseable judge response: {err}"),
               "criterion_assessments": [], "unsupported_content": [],
               "consistency_violations": violations, "repair_attempted": attempt > 1}
    else:
        d = verdict.model_dump()
        assessments = d["criterion_assessments"]
        dumped = _gold_gated({k: d[k] for k in DIMENSIONS}, dims)
        missing_scores = [x for x in dims if dumped[x]["score"] is None]
        if violations:
            # Scores are preserved EXACTLY as returned. Not clamped, not zeroed,
            # not counted as a pass — this is a judge failure, and the numbers
            # are kept so a human can see what it actually claimed.
            status, error = "judge_inconsistent", "; ".join(v["rule"] for v in violations)
        elif missing_scores:
            status, error = "partial", f"no score for {missing_scores}"
        else:
            status, error = "ok", None
        out = {"status": status, "error": error, "dimensions": dumped,
               "criterion_assessments": assessments,
               "unsupported_content": d["unsupported_content"],
               "consistency_violations": violations,
               "repair_attempted": attempt > 1,
               "unknown_evidence_labels": unknown_evidence_labels(assessments, ev),
               "criterion_status_counts": criterion_status_counts(assessments)}
    cache.put(key, out)
    return {**base, **out, "cached": False}


def criterion_status_counts(assessments: list) -> dict:
    return {s: sum(1 for a in assessments if a["status"] == s) for s in CRITERION_STATUSES}


def _null_dims(reason: str) -> dict:
    return {d: {"score": None, "applicable": False, "reason": reason,
                "criterion": None, "confidence": None} for d in DIMENSIONS}


def run(run_dir, case_index: dict, backend, cache: JudgeCache | None = None,
        resume: bool = False, progress=None) -> dict:
    """Judge every collected response into judge.jsonl, in deterministic order."""
    say = progress or (lambda *_a, **_k: None)
    from src.evals.final_eval.collect import load_evidence_cache

    cache = cache or JudgeCache()
    ev_by_qid = load_evidence_cache(run_dir)
    rows = sorted(run_dir.read_jsonl("responses"), key=lambda r: r["query_id"])
    done = run_dir.completed_ids("judge") if resume else set()
    if not resume:
        run_dir.file("judge").unlink(missing_ok=True)

    stats = {"judged": 0, "cached": 0, "failures": 0, "inconsistent": 0,
             "repaired": 0, "skipped_existing": len(done)}
    for row in rows:
        qid = row["query_id"]
        if qid in done:
            continue
        case = case_index.get(qid)
        if case is None:
            continue
        out = judge_case(backend, cache, case, row, ev_by_qid.get(qid) or [])
        run_dir.append("judge", out)
        stats["judged"] += 1
        stats["cached"] += bool(out.get("cached"))
        stats["inconsistent"] += out["status"] == "judge_inconsistent"
        stats["repaired"] += bool(out.get("repair_attempted"))
        if out["status"] not in ("ok", "partial"):
            stats["failures"] += 1
        scores = {d: out["dimensions"][d]["score"] for d in out["applicable_dimensions"]}
        counts = out.get("criterion_status_counts") or {}
        say(f"  {qid:<10} {out['status']:<18} {scores} "
            f"crit={ {k: v for k, v in counts.items() if v} }")
    return stats
