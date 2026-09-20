"""
Phase 4 — independent offline LLM-as-a-Judge
============================================
A second, independent opinion on answer quality, scored per dimension.

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
A judge that errors or returns unparseable output is recorded with a status of
`backend_error` or `parse_error` and null scores. It is counted as a judge
failure and is never converted into a 0 or into a pass.
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from src.evals.final_eval.judge_backend import JudgeUnavailable

logger = logging.getLogger(__name__)

JUDGE_PROMPT_VERSION = "aj1"
DEFAULT_CACHE_PATH = os.environ.get(
    "LUMEN_JUDGE_CACHE", str(Path(__file__).resolve().parents[3] / ".cache" / "final_eval_judge.json"))

DIMENSIONS = ("factual_correctness", "groundedness", "completeness",
              "answer_relevance", "temporal_correctness", "abstention_quality")

EVIDENCE_CHARS = 2400        # per source, mirrors the runtime verifier's budget


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class Dimension(BaseModel):
    score: Optional[int] = Field(None, ge=0, le=4)
    applicable: bool = True
    reason: str = ""
    criterion: Optional[str] = None
    confidence: Optional[str] = None


class JudgeVerdict(BaseModel):
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
OLLAMA_SCHEMA = {"type": "object",
                 "properties": {d: _dim_schema() for d in DIMENSIONS},
                 "required": list(DIMENSIONS)}


SYSTEM_PROMPT = """You are an independent clinical answer evaluator. You grade a generated answer against a reference and its cited evidence.

You are NOT the system that produced the answer. You have no information about how it was produced, whether it was checked, or whether anyone flagged it. Judge only what you are shown.

Score each dimension 0-4:
  4  fully correct / fully grounded / fully complete; nothing to qualify
  3  correct with a minor omission or imprecision that does not mislead
  2  partially correct; a material element is missing, imprecise or overstated
  1  mostly incorrect or largely ungrounded; at most an incidental element is right
  0  incorrect, fabricated, or contradicted by the cited evidence

Dimensions:
  factual_correctness  Do the answer's assertions match the REFERENCE ANSWER and REQUIRED FACTS? A correct value stated in different words or units is still correct.
  groundedness         Is every assertion supported by the CITED EVIDENCE shown below? Content that is plausible but absent from the evidence scores 2 or lower, no matter how clinically reasonable it is.
  completeness         Does the answer cover the required facts the question asks for?
  answer_relevance     Does it answer the question that was asked, without drifting?
  temporal_correctness Where the question concerns the latest value or a trend, is the time ordering and recency right?
  abstention_quality   Where the record does not support an answer or is conflicting, does the answer say so honestly instead of inventing certainty?

Rules:
- The APPLICABLE DIMENSIONS list tells you which dimensions to score. For any dimension not on that list, return {"score": null, "applicable": false, "reason": "not applicable"}.
- "criterion" should name the specific required fact or expectation your score keys to, when one applies.
- "reason" must be at most 200 characters and must cite what in the answer or evidence drove the score.
- Return ONLY a JSON object with exactly the six dimension keys. No prose, no markdown."""


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
)
# Bare words that ARE legitimate English but would be leakage in key position
# ("verified: true"). Checked only where they look like a field, not prose.
FORBIDDEN_FIELDLIKE = ("verified", "unsupported", "flagged", "escalated")
_FIELDLIKE_RE = None


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
    """(system, user, applicable dimensions). Built from an explicit allowlist."""
    dims = applicable_dimensions(case)
    ev = cited_evidence(row, evidence)
    facts = [f.text for f in case.expected_facts]

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
        ("\nREQUIRED FACTS:\n" + "\n".join(f"  - {f}" for f in facts)) if facts
        else "\nREQUIRED FACTS:\n  (none specified)",
    ]
    if case.must_not_contain:
        parts.append("\nMUST NOT ASSERT:\n" + "\n".join(f"  - {t}" for t in case.must_not_contain))
    if case.temporal_applicable:
        parts.append(f"\nTEMPORAL EXPECTATION:\n  the question asks for the '{case.temporal}' "
                     f"value(s), judged within this patient's own timeline")
    parts.append("\nCITED EVIDENCE:\n" + ("\n\n".join(f"[{e['label']}]\n{e['text']}" for e in ev)
                                          if ev else "  (the answer cites no evidence)"))
    parts.append(f"\nGENERATED ANSWER:\n{row.get('answer') or '(empty)'}")
    parts.append("\nAPPLICABLE DIMENSIONS:\n  " + ", ".join(dims))
    parts.append("\nReturn the JSON object now.")
    return SYSTEM_PROMPT, "\n".join(parts), dims


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


def parse_verdict(raw: str) -> tuple:
    """(JudgeVerdict | None, error). Never raises, never invents a score."""
    obj = extract_json(raw)
    if obj is None:
        return None, "no JSON object in response"
    normalized = {}
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
            "temporal": case.temporal}


def judge_case(backend, cache: JudgeCache, case, row: dict, evidence: list) -> dict:
    """One case. Always returns a row; failures are recorded, never scored."""
    system, user, dims = build_prompt(case, row, evidence)
    ev = cited_evidence(row, evidence)
    key = JudgeCache.key(prompt_version=JUDGE_PROMPT_VERSION, model=backend.model,
                         query_id=case.query_id, query=case.query,
                         answer=row.get("answer") or "", evidence=ev,
                         criteria=criteria_of(case))
    base = {"query_id": case.query_id, "judge_model": backend.model,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "applicable_dimensions": dims, "cache_key": key,
            "n_evidence": len(ev)}

    hit = cache.get(key)
    if hit is not None:
        return {**base, **hit, "cached": True}

    if row.get("status") == "evaluator_error":
        out = {"status": "skipped_no_answer", "error": "case failed during collection",
               "dimensions": _null_dims("case produced no answer to judge")}
        return {**base, **out, "cached": False}

    try:
        raw = backend.complete(system, user, OLLAMA_SCHEMA)
    except JudgeUnavailable as e:
        # Explicit, uncached: the judge did not run. Never a score, never the
        # runtime model standing in for it.
        return {**base, "status": "backend_error", "error": str(e)[:300],
                "dimensions": _null_dims("judge backend unavailable"), "cached": False}

    verdict, err = parse_verdict(raw)
    if verdict is None:
        out = {"status": "parse_error", "error": err,
               "dimensions": _null_dims(f"unparseable judge response: {err}")}
    else:
        dumped = verdict.model_dump()
        # Gold decides applicability, not the judge: a score for a dimension the
        # case does not exercise is discarded rather than averaged in.
        for d in DIMENSIONS:
            if d not in dims:
                dumped[d] = {"score": None, "applicable": False,
                             "reason": "not applicable for this case",
                             "criterion": None, "confidence": None}
            elif dumped[d]["score"] is None:
                dumped[d]["applicable"] = False
                dumped[d]["reason"] = dumped[d]["reason"] or "judge returned no score"
        missing = [d for d in dims if dumped[d]["score"] is None]
        out = {"status": "ok" if not missing else "partial",
               "error": None if not missing else f"no score for {missing}",
               "dimensions": dumped}
    cache.put(key, out)
    return {**base, **out, "cached": False}


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

    stats = {"judged": 0, "cached": 0, "failures": 0, "skipped_existing": len(done)}
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
        if out["status"] not in ("ok", "partial"):
            stats["failures"] += 1
        scores = {d: out["dimensions"][d]["score"] for d in out["applicable_dimensions"]}
        say(f"  {qid:<10} {out['status']:<14} {scores}")
    return stats
