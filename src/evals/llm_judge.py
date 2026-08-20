"""
LLM-as-Judge for Clinical Retrieval Relevance
=============================================
Grades how well a retrieved chunk answers a query, on a 0-3 scale, using a GROQ
model (default Llama-3.3-70B). Built to replace the keyword-presence judge,
whose circularity inflated and saturated the earlier scores.

Robustness features:
  - Deterministic grading (temperature 0) so re-runs are stable.
  - Bulletproof JSON parsing: strips code fences, extracts the first balanced
    object, repairs trailing commas, coerces score types — never raises.
  - Disk cache keyed by (prompt_version, model, query, chunk_text): the same
    (query, chunk) pair is judged ONCE, ever, across configs and across runs.
  - Bounded-concurrency batch judging (ThreadPoolExecutor) with in-batch dedup.
  - Retries with exponential backoff + jitter; rate-limit aware.
  - Graceful degradation: a chunk that can't be judged defaults to 0 (NOT
    credited) and is flagged with an error, so failures are visible, not silent.
  - Pooling: judge the union of all configs' results per query once, giving a
    shared relevant-set (standard TREC-style methodology) for honest recall.

Usage:
    from src.evals.llm_judge import LLMJudge, build_pooled_relevance

    judge = LLMJudge(model="llama-3.3-70b-versatile")   # reads GROQ_API_KEY
    relevant_ids, grades, per_config = build_pooled_relevance(
        query_text, configs_results, judge, threshold=2
    )

Requires: pip install groq   (and env var GROQ_API_KEY)
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import random
import hashlib
import logging
import threading
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are an expert clinical information-retrieval judge. Given a SEARCH QUERY and a retrieved TEXT CHUNK from a clinical note, rate how well the chunk's INFORMATION answers the query, on a 0-3 scale.

Judge information relevance, NOT keyword overlap. A chunk that merely contains a word from the query in an unrelated context is not relevant. Examples: a cervical-spine MRI is irrelevant to "swollen legs"; a generic CBC panel that does not report the asked-about value is irrelevant to "abnormal potassium"; a liver MRI is irrelevant to "kidneys shutting down".

Scale:
3 = Directly answers the query: contains the specific finding, value, medication, or fact asked for.
2 = Relevant: same clinical topic and useful context, though not the exact answer.
1 = Marginal: tangentially related; shares a term but not the intent.
0 = Irrelevant: different topic, or only incidental keyword overlap.

The text is de-identified; placeholders like [PERSON], [DATE], [LOCATION], or ___ are normal and must not lower the score.

Respond with ONLY a JSON object: {"score": <integer 0-3>, "reason": "<one short sentence>"}. Output no text outside the JSON."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class JudgeResult:
    query: str
    chunk_id: object
    score: int
    reason: str
    cached: bool = False
    error: Optional[str] = None

    def is_relevant(self, threshold: int = 2) -> bool:
        return self.score >= threshold


# ---------------------------------------------------------------------------
# Robust JSON handling (module-level so it's unit-testable on its own)
# ---------------------------------------------------------------------------
def extract_json(text: Optional[str]) -> Optional[dict]:
    """Pull a JSON object out of an LLM response, tolerating fences/prose."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                frag = t[start:i + 1]
                for candidate in (frag, re.sub(r",\s*([}\]])", r"\1", frag)):
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        continue
                return None
    return None


def coerce_score(raw: Optional[dict]) -> tuple[Optional[int], str]:
    """Validate/clamp the score field; return (score|None, reason)."""
    if not isinstance(raw, dict):
        return None, ""
    val = raw.get("score", raw.get("relevance", raw.get("rating")))
    score: Optional[int] = None
    try:
        score = int(round(float(val)))
    except (TypeError, ValueError):
        if isinstance(val, str):
            m = re.search(r"[0-3]", val)
            score = int(m.group()) if m else None
    reason = str(raw.get("reason", raw.get("explanation", "")))[:300]
    if score is None:
        return None, reason
    return max(0, min(3, score)), reason


def _with_retries(fn: Callable, max_retries: int, base_delay: float, max_rate_limit_retries: int = 30):
    """Run fn() with exponential backoff + jitter; rate-limit aware.

    429s don't spend the same limited `max_retries` budget as genuine errors:
    the real wait time is already enforced by the caller's shared throttle
    (LLMJudge._note_rate_limit sets the resume time from Groq's own
    Retry-After header), so here we just keep retrying — bounded by
    `max_rate_limit_retries` — instead of giving up and silently scoring the
    chunk 0 after a handful of attempts, which would bias results toward
    "irrelevant" purely because of rate limiting, not judgment.
    """
    last_exc = None
    attempt = 0
    rate_limit_attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:  # broad: GROQ/openai SDK exception types vary by version
            last_exc = e
            msg = str(e).lower()
            status = getattr(e, "status_code", None)
            is_rate = status == 429 or "rate" in msg or "429" in msg or "quota" in msg or "overloaded" in msg
            if is_rate:
                rate_limit_attempt += 1
                if rate_limit_attempt > max_rate_limit_retries:
                    break
                # the shared throttle already knows the real Retry-After wait;
                # this is just a small jitter so threads don't all wake in lockstep.
                logger.debug(f"judge call rate-limited (retry {rate_limit_attempt}/{max_rate_limit_retries})")
                time.sleep(random.uniform(0, base_delay))
                continue
            if attempt >= max_retries:
                break
            delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
            logger.debug(f"judge call failed (attempt {attempt + 1}): {e}; retrying in {delay:.1f}s")
            time.sleep(delay)
            attempt += 1
    raise last_exc


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------
class LLMJudge:
    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 200,
        timeout: float = 30.0,
        max_retries: int = 4,
        base_delay: float = 1.0,
        max_workers: int = 8,
        n_votes: int = 1,
        json_mode: bool = True,
        max_chunk_chars: int = 4000,
        cache_path: Optional[str] = ".cache/llm_judge.json",
        prompt_version: str = PROMPT_VERSION,
        call_fn: Optional[Callable[[list], str]] = None,
        requests_per_second: float = 0.5,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_workers = max_workers
        self.n_votes = max(1, n_votes)
        self.json_mode = json_mode
        self.max_chunk_chars = max_chunk_chars
        self.cache_path = cache_path
        self.prompt_version = prompt_version
        self._call_fn = call_fn          # injectable (tests / alternate providers)
        self._client = None
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = self._load_cache()

        # Global request pacing: spreads calls out across ALL worker threads so
        # a `max_workers`-wide burst doesn't slam the Groq rate limit. Only
        # applies to real API calls (not the injected call_fn used in tests),
        # and only paces *when* a request starts — retries/backoff on 429s are
        # unaffected, this just makes them rarer.
        self.requests_per_second = requests_per_second
        self._min_interval = (1.0 / requests_per_second) if requests_per_second and requests_per_second > 0 else 0.0
        self._rate_lock = threading.Lock()
        self._next_call_at = 0.0

        if self._call_fn is None:
            logger.warning("No call_fn provided — build via make_ollama_judge() before judging.")

    # ---- prompt + key ----
    def _messages(self, query: str, chunk_text: str) -> list[dict]:
        user = f"QUERY: {query}\n\nCHUNK:\n{chunk_text}\n\nJSON:"
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]

    def _cache_key(self, query: str, chunk_text: str) -> str:
        payload = f"{self.prompt_version}\x00{self.model}\x00{query}\x00{(chunk_text or '')[:self.max_chunk_chars]}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ---- pacing (shared across all threads) ----
    def _throttle(self):
        if self._min_interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = self._next_call_at - now
            self._next_call_at = max(now, self._next_call_at) + self._min_interval
        if wait > 0:
            time.sleep(wait)

    def _note_rate_limit(self, exc: Exception):
        """On a 429, read Groq's real Retry-After and push the SHARED cooldown
        out so every thread's next _throttle() blocks until then — not just
        the thread that got limited. Without this, N worker threads each back
        off independently and keep re-firing into a window that's still
        exhausted, turning one rate-limit into a storm of them."""
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status != 429:
            return
        retry_after = None
        headers = getattr(response, "headers", None) or {}
        raw_ms = headers.get("retry-after-ms")
        if raw_ms is not None:
            try:
                retry_after = float(raw_ms) / 1000
            except (TypeError, ValueError):
                retry_after = None
        if retry_after is None:
            raw_s = headers.get("retry-after")
            try:
                retry_after = float(raw_s)
            except (TypeError, ValueError):
                retry_after = None
        if retry_after is None:
            retry_after = self.base_delay * 8  # no header given: conservative guess
        retry_after = min(retry_after, 120.0)  # sanity cap
        with self._rate_lock:
            resume_at = time.monotonic() + retry_after
            if resume_at > self._next_call_at:
                self._next_call_at = resume_at
                logger.info(f"rate limited (429): pausing all judge threads for {retry_after:.1f}s")

    # ---- LLM call (isolated + injectable) ----
    def _call(self, messages: list[dict]) -> str:
        if self._call_fn is None:
            raise RuntimeError(
                "LLMJudge has no call_fn. Build it with "
                "src.evals.ollama_backend.make_ollama_judge() to use local Ollama."
            )
        return self._call_fn(messages)

    # ---- single uncached judgement (never raises) ----
    def _judge_uncached(self, query: str, chunk_text: str) -> JudgeResult:
        text = (chunk_text or "")[: self.max_chunk_chars]
        messages = self._messages(query, text)
        votes: list[tuple[int, str]] = []
        err: Optional[str] = None
        for _ in range(self.n_votes):
            try:
                raw = _with_retries(lambda: self._call(messages), self.max_retries, self.base_delay)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:300]
                continue
            score, reason = coerce_score(extract_json(raw))
            if score is not None:
                votes.append((score, reason))
            else:
                err = "unparseable response"
        if not votes:
            # Conservative: unjudgeable -> irrelevant, but flagged so it's visible.
            return JudgeResult(query, None, 0, "JUDGE_ERROR", error=err or "no valid votes")
        scores = sorted(v[0] for v in votes)
        final = scores[len(scores) // 2]  # median vote
        reason = next((r for s, r in votes if s == final and r), votes[0][1])
        return JudgeResult(query, None, final, reason)

    # ---- cache helpers ----
    def _load_cache(self) -> dict:
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"cache load failed ({e}); starting empty")
        return {}

    def _flush_cache(self):
        if not self.cache_path:
            return
        try:
            d = os.path.dirname(self.cache_path)
            if d:
                os.makedirs(d, exist_ok=True)
            with self._lock:
                data = dict(self._cache)
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.cache_path)  # atomic
        except Exception as e:
            logger.warning(f"cache flush failed: {e}")

    @staticmethod
    def _from_cache(entry: dict, query: str) -> JudgeResult:
        return JudgeResult(query, None, int(entry.get("score", 0)),
                           str(entry.get("reason", "")), cached=True,
                           error=entry.get("error"))

    # ---- public API ----
    def judge(self, query: str, chunk_text: str, chunk_id=None) -> JudgeResult:
        return self.judge_batch([(query, chunk_text, chunk_id)])[0]

    def judge_batch(self, items: list[tuple]) -> list[JudgeResult]:
        """items: list of (query, chunk_text, chunk_id). Cached + concurrent + deduped."""
        unique: dict[str, tuple] = {}
        for query, text, _cid in items:
            unique.setdefault(self._cache_key(query, text), (query, text))

        key_results: dict[str, JudgeResult] = {}
        to_compute: list[tuple[str, str, str]] = []
        with self._lock:
            for key, (query, text) in unique.items():
                if key in self._cache:
                    key_results[key] = self._from_cache(self._cache[key], query)
                else:
                    to_compute.append((key, query, text))

        if to_compute:
            done = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = {ex.submit(self._judge_uncached, q, t): k for (k, q, t) in to_compute}
                for fut in as_completed(futs):
                    key = futs[fut]
                    res = fut.result()  # _judge_uncached never raises
                    key_results[key] = res
                    with self._lock:
                        self._cache[key] = {"score": res.score, "reason": res.reason, "error": res.error}
                    done += 1
                    if done % 25 == 0:
                        logger.info(f"judged {done}/{len(to_compute)} new pairs")
            self._flush_cache()

        out = []
        for query, text, cid in items:
            base = key_results[self._cache_key(query, text)]
            out.append(replace(base, chunk_id=cid, query=query))
        return out

    def judge_pool(self, query: str, chunks: list[tuple]) -> dict:
        """chunks: list of (chunk_id, chunk_text). Returns {chunk_id: JudgeResult}."""
        items = [(query, text, cid) for cid, text in chunks]
        return {r.chunk_id: r for r in self.judge_batch(items)}


# ---------------------------------------------------------------------------
# Pooling + graded metrics for honest config comparison
# ---------------------------------------------------------------------------
def build_pooled_relevance(query: str, configs_results: dict, judge: LLMJudge, threshold: int = 2):
    """
    Pool every config's retrieved chunks, judge the UNION once, and score each
    config against that shared relevant-set.

    configs_results: dict[config_name -> ranked list of objects with
                     .chunk_id and .chunk_text] (best-first).
    Returns (relevant_ids, grades, per_config) where per_config maps
    config_name -> {"binary": [...], "graded": [...]} aligned to its rank order.
    """
    pool: dict = {}
    for results in configs_results.values():
        for r in results:
            pool.setdefault(r.chunk_id, r.chunk_text)

    judged = judge.judge_pool(query, list(pool.items()))
    grades = {cid: jr.score for cid, jr in judged.items()}
    relevant_ids = {cid for cid, g in grades.items() if g >= threshold}

    per_config = {}
    for name, results in configs_results.items():
        per_config[name] = {
            "binary": [1 if r.chunk_id in relevant_ids else 0 for r in results],
            "graded": [grades.get(r.chunk_id, 0) for r in results],
        }
    return relevant_ids, grades, per_config


def ndcg_at_k_graded(graded_ranked: list[int], pool_grades: list[int], k: int = 10) -> float:
    """Graded nDCG@k. IDCG uses the best possible ordering of ALL pooled grades."""
    def dcg(gains):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))
    idcg = dcg(sorted(pool_grades, reverse=True))
    return (dcg(graded_ranked) / idcg) if idcg > 0 else 0.0


def recall_at_k_pooled(binary_ranked: list[int], n_relevant_pooled: int, k: int = 10) -> float:
    """Recall@k against the pooled relevant-set denominator."""
    if n_relevant_pooled <= 0:
        return 0.0
    return min(sum(binary_ranked[:k]) / n_relevant_pooled, 1.0)


# ===========================================================================
# Self-test with a MOCK LLM (no network) — exercises the robustness scaffolding
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    import tempfile

    print("=" * 70)
    print("  LLM-as-Judge — robustness self-test (mock LLM, no network)")
    print("=" * 70)

    # 1) JSON extraction torture test
    print("\n[1] extract_json / coerce_score on messy outputs:")
    cases = [
        '{"score": 3, "reason": "exact value present"}',                 # clean
        '```json\n{"score": 2, "reason": "same topic"}\n```',            # fenced
        'Sure! Here is my rating:\n{"score":1,"reason":"marginal"}\nThanks',  # prose-wrapped
        '{"score": 2, "reason": "trailing comma",}',                     # trailing comma
        '{"score": "5", "reason": "out of range string"}',              # str + clamp
        '{"rating": 0}',                                                 # alt field, no reason
        'totally not json',                                             # garbage
        '',                                                              # empty
    ]
    for c in cases:
        s, reason = coerce_score(extract_json(c))
        print(f"    score={str(s):>4}  reason={reason!r:<28}  <- {c[:42]!r}")

    # 2) Mock judge: deterministic scenarios by chunk content
    def mock_llm(messages):
        chunk = messages[-1]["content"]
        if "SPINE" in chunk.upper():
            return 'Here you go: {"score": 0, "reason": "spine MRI, unrelated to query"}'
        if "POTASSIUM 6.1" in chunk.upper():
            return '```json\n{"score": 3, "reason": "reports abnormal potassium value"}\n```'
        if "CBC" in chunk.upper():
            return '{"score": 0, "reason": "generic CBC, no potassium",}'   # trailing comma
        if "EDEMA" in chunk.upper():
            return '{"score": 2, "reason": "same topic, fluid overload"}'
        return 'broken output, no json here'   # forces conservative default

    tmp = tempfile.mktemp(suffix=".json")
    judge = LLMJudge(call_fn=mock_llm, cache_path=tmp, max_workers=4)

    print("\n[2] judge_batch with a duplicate + a broken response:")
    items = [
        ("abnormal potassium lab results", "Labs show POTASSIUM 6.1 critical high", "c1"),
        ("abnormal potassium lab results", "CBC: WBC 7.3 RBC 3.72 Hgb 11", "c2"),
        ("swollen legs", "CERVICAL SPINE: vertebral body heights preserved", "c3"),
        ("swollen legs", "lower extremity EDEMA, started furosemide", "c4"),
        ("swollen legs", "lower extremity EDEMA, started furosemide", "c4dup"),  # dup text
        ("unknown query", "some chunk that triggers a broken LLM reply", "c5"),
    ]
    for r in judge.judge_batch(items):
        flag = f" ERROR={r.error}" if r.error else ""
        print(f"    {r.chunk_id:<6} score={r.score}  rel@2={r.is_relevant()}  {r.reason!r}{flag}")

    print("\n[3] cache hit on re-judge (should all be cached=True, no new calls):")
    judge2 = LLMJudge(call_fn=mock_llm, cache_path=tmp, max_workers=4)
    again = judge2.judge_batch(items[:4])
    print("    cached flags:", [r.cached for r in again])

    # 3) Pooling across two configs
    print("\n[4] build_pooled_relevance across 2 configs:")
    @dataclass
    class R:
        chunk_id: str
        chunk_text: str
    configs = {
        "bm25_only": [R("c2", "CBC: WBC 7.3 RBC 3.72 Hgb 11"),
                      R("c1", "Labs show POTASSIUM 6.1 critical high")],
        "hybrid":    [R("c1", "Labs show POTASSIUM 6.1 critical high"),
                      R("c4", "lower extremity EDEMA, started furosemide")],
    }
    rel_ids, grades, per_cfg = build_pooled_relevance(
        "abnormal potassium lab results", configs, judge2, threshold=2)
    print(f"    pooled grades: {grades}")
    print(f"    relevant_ids (>=2): {rel_ids}")
    for name, d in per_cfg.items():
        ndcg = ndcg_at_k_graded(d["graded"], list(grades.values()), k=5)
        rec = recall_at_k_pooled(d["binary"], len(rel_ids), k=5)
        print(f"    {name:<10} binary={d['binary']} graded={d['graded']} "
              f"nDCG@5={ndcg:.3f} recall@5={rec:.2f}")

    os.path.exists(tmp) and os.remove(tmp)
    print("\nAll robustness checks ran.")
