"""
Lumen Answer Generation  (grounded RAG, local Ollama)
=====================================================
Turns a question into a grounded answer over the clinical notes: retrieve with
the hybrid retriever, assemble the top chunks into a cited context block, and
have a LOCAL Ollama model (default Qwen2.5-14B) write an answer that is allowed
to use ONLY that context. No clinical text leaves the machine — same DUA-safe
setup as the judge.

Design choices that matter for a clinical RAG:
  * Grounded-only. The system prompt forbids outside knowledge about the
    patient and requires a [S#] citation on every claim. If the context
    doesn't answer the question, the model must return a fixed sentence
    instead of guessing.
  * Patient-scoped. `subject_id` / `hadm_id` pass straight through to
    HybridRetriever.search(), so a patient question never mixes patients.
  * Injectable retriever. Pass an existing HybridRetriever, or let this build
    one lazily — so you control when the torch models load (they share RAM
    with Ollama's 14B on a 16 GB machine).

Requires a running Ollama server with the model pulled:
    ollama serve
    ollama pull qwen2.5:14b

Usage (library):
    from src.generation.answer_generator import AnswerGenerator
    gen = AnswerGenerator()
    out = gen.answer("What were the patient's potassium levels?", subject_id=10014354)
    print(out.answer)
    for s in out.sources_used:
        print(s["tag"], s["note_type"], s["charttime"], s["chunk_id"])

Usage (CLI):
    python -m src.generation.answer_generator "abnormal potassium labs" --subject 10014354 --top-k 6
"""
from __future__ import annotations

import re
import time
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional

import requests

from src.retrieval.hybrid_retriever_v2 import HybridRetriever, RetrievalResult
from src.generation.lab_query import LabResolver

logger = logging.getLogger(__name__)

DEFAULT_GEN_MODEL = "qwen2.5:14b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# The sentinel the model must return when the context is insufficient. We check
# for it to flag ungrounded / no-answer cases downstream.
NO_ANSWER = "The available records do not contain enough information to answer this."

SYSTEM_PROMPT = f"""You are a careful clinical information assistant. You answer \
questions using ONLY the numbered note excerpts provided in the context.

Rules:
1. Use only the provided excerpts. Do NOT use outside medical knowledge to state \
any fact about this patient. You may use general knowledge only to interpret \
what the excerpts say, never to add facts not in them.
2. Cite every clinical claim with its source tag(s) in square brackets, e.g. \
"Potassium was 6.1 mEq/L [S2]." A claim may cite more than one, e.g. [S1][S3].
3. Preserve exact values, units, and dates as written; do not round or invent. \
When a value has no date written next to it in the text, use the charttime shown \
in that excerpt's [S#] header as its date (e.g. "Potassium 3.5 mEq/L on <charttime> \
[S2]"). Treat a value's date as unknown ONLY when the text has no inline date AND \
the header charttime is empty.
4. When the question asks to "list", for "all"/"each", or how something \
changed/trended over time, report EVERY matching value found across ALL excerpts — \
each with its date and source tag, in chronological order. Do not stop after the \
first match. If genuinely only one value is present, say so explicitly rather than \
implying it is the only one that ever existed.
5. If excerpts conflict or come from different times for a SINGLE current value, \
say so and prefer the most recent (using the header charttime). This does not \
override rule 4: a "list all"/"trend" question still gets every value.
6. If the excerpts do not contain enough information to answer, reply with \
EXACTLY this sentence and nothing else: "{NO_ANSWER}"
7. Be concise and factual. Describe what the records show; do not give treatment \
advice or recommendations beyond what is documented.
8. Sources tagged [L#] are structured lab results and are already dated;\
treat them as authoritative for numeric values and cite them exactly like note sources.\
When a question asks for lab values or trends, report the [L#] series in full.
"""


@dataclass
class GeneratedAnswer:
    question: str
    answer: str
    sources_used: list = field(default_factory=list)   # sources the model cited
    retrieved: list = field(default_factory=list)       # all RetrievalResult passed in
    grounded: bool = False                              # cited >=1 source & not the sentinel
    model: str = DEFAULT_GEN_MODEL
    elapsed_s: float = 0.0
    error: Optional[str] = None


def _format_charttime(ct: Optional[str]) -> str:
    return ct if ct else "time unknown"


def build_context_block(results: list[RetrievalResult], max_chars: int = 8000) -> tuple[str, list[dict]]:
    """
    Render retrieved chunks as tagged, headed excerpts the model can cite.

    Returns (context_text, source_index) where source_index[i] describes [S{i+1}].
    Highest-scored chunks come first; we stop adding once max_chars is hit so the
    prompt stays within the model's context window.
    """
    lines: list[str] = []
    source_index: list[dict] = []
    used = 0

    for i, r in enumerate(results):
        tag = f"S{len(source_index) + 1}"
        body = (r.context_text or r.chunk_text or "").strip()
        if not body:
            continue
        header = (f"[{tag}] note_type={r.note_type} | charttime={_format_charttime(r.charttime)} "
                  f"| note_id={r.note_id} | chunk_id={r.chunk_id}")
        block = f"{header}\n{body}"

        # Budget: keep whole excerpts; stop before overflowing the window.
        if used + len(block) > max_chars and source_index:
            break
        lines.append(block)
        used += len(block) + 2
        source_index.append({
            "tag": tag,
            "chunk_id": r.chunk_id,
            "note_id": r.note_id,
            "subject_id": r.subject_id,
            "hadm_id": r.hadm_id,
            "note_type": r.note_type,
            "charttime": r.charttime,
            "final_score": round(float(r.final_score), 4),
        })

    return "\n\n".join(lines), source_index


def _cited_tags(answer: str) -> set[str]:
    """Pull [S3] / [L1] style citations out of the model's answer."""
    return set(re.findall(r"[\[(]([SL]\d+)[\])]", answer))


class AnswerGenerator:
    def __init__(
        self,
        model: str = DEFAULT_GEN_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        retriever: Optional[HybridRetriever] = None,
        use_reranker: bool = True,
        use_labs: bool = True,
        top_k: int = 6,
        temperature: float = 0.1,     # low: factual, near-deterministic
        num_ctx: int = 8192,          # room for ~6 context chunks + prompt
        num_predict: int = 600,       # answer length cap
        max_context_chars: int = 8000,
        timeout: float = 300.0,
        keep_alive: str = "30m",
    ):
        self.model = model
        self.host = host.rstrip("/")
        self._retriever = retriever
        self.use_reranker = use_reranker
        self.top_k = top_k
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.max_context_chars = max_context_chars
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.use_labs = use_labs
        self._labs = None


    @property
    def retriever(self) -> HybridRetriever:
        # Lazy: don't load MedCPT + reranker until the first query.
        if self._retriever is None:
            logger.info("Building HybridRetriever (loading retrieval models)...")
            self._retriever = HybridRetriever(use_reranker=self.use_reranker)
        return self._retriever
    
    @property
    def labs(self) -> LabResolver:
        if self._labs is None:
            self._labs = LabResolver()   # loads d_labitems once
        return self._labs

    def _ollama_chat(self, messages: list[dict]) -> str:
        resp = requests.post(
            f"{self.host}/api/chat",
            timeout=self.timeout,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def answer(
        self,
        question: str,
        subject_id: Optional[int] = None,
        hadm_id: Optional[int] = None,
        note_type: Optional[str] = None,
        temporal_filter: str = "auto",
    ) -> GeneratedAnswer:
        t0 = time.time()

        # --- structured labs path (quantitative questions) ---
        lab_ctx, lab_sources = "", []
        if self.use_labs and subject_id is not None:
            lab_ctx, lab_sources, _matched = self.labs.labs_for_question(
                question, subject_id, hadm_id=hadm_id)

        # --- notes retrieval (narrative context; also powers hybrid answers) ---
        results = self.retriever.search(
            query=question, subject_id=subject_id, hadm_id=hadm_id,
            note_type=note_type, temporal_filter=temporal_filter, top_k=self.top_k,
        )
        note_ctx, note_sources = build_context_block(results, self.max_context_chars)

        if not lab_ctx and not note_ctx:
            return GeneratedAnswer(
                question=question, answer=NO_ANSWER, retrieved=results, grounded=False,
                model=self.model, elapsed_s=round(time.time() - t0, 2),
            )

        # assemble context: labs first (precise, dated), then note excerpts
        sections = []
        if lab_ctx:
            sections.append("STRUCTURED LAB RESULTS (already dated; authoritative for numeric values):\n\n" + lab_ctx)
        if note_ctx:
            sections.append("NOTE EXCERPTS:\n\n" + note_ctx)
        context = "\n\n".join(sections)
        source_index = lab_sources + note_sources

        user_msg = (
            f"CONTEXT:\n\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"Answer using only the context above, citing sources as [L#] (labs) or [S#] (notes)."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            answer_text = self._ollama_chat(messages)
        except Exception as e:
            return GeneratedAnswer(
                question=question, answer="", retrieved=results, grounded=False,
                model=self.model, elapsed_s=round(time.time() - t0, 2),
                error=f"{type(e).__name__}: {e}"[:300],
            )

        cited = _cited_tags(answer_text)
        sources_used = [s for s in source_index if s["tag"] in cited]
        is_sentinel = answer_text.strip().rstrip(".") == NO_ANSWER.rstrip(".")
        grounded = bool(cited) and not is_sentinel

        return GeneratedAnswer(
            question=question, answer=answer_text, sources_used=sources_used,
            retrieved=results, grounded=grounded, model=self.model,
            elapsed_s=round(time.time() - t0, 2),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Lumen grounded RAG answer generation (local Ollama)")
    ap.add_argument("question", type=str, help="The question to answer")
    ap.add_argument("--subject", type=int, default=None, help="subject_id to scope retrieval to one patient")
    ap.add_argument("--hadm", type=int, default=None, help="hadm_id to scope to one admission")
    ap.add_argument("--note-type", type=str, default=None, help="Restrict to a note type")
    ap.add_argument("--temporal", type=str, default="auto", help="Temporal filter mode (default auto)")
    ap.add_argument("--top-k", type=int, default=6, help="Chunks to retrieve for context (default 6)")
    ap.add_argument("--model", type=str, default=DEFAULT_GEN_MODEL, help="Ollama model tag")
    ap.add_argument("--json", action="store_true", help="Print full result as JSON")
    args = ap.parse_args()

    gen = AnswerGenerator(model=args.model, top_k=args.top_k)
    out = gen.answer(
        args.question, subject_id=args.subject, hadm_id=args.hadm,
        note_type=args.note_type, temporal_filter=args.temporal,
    )

    if args.json:
        print(json.dumps({
            "question": out.question, "answer": out.answer, "grounded": out.grounded,
            "sources_used": out.sources_used, "model": out.model,
            "elapsed_s": out.elapsed_s, "error": out.error,
        }, indent=2))
    else:
        print("\n" + "=" * 80)
        print(f"Q: {out.question}")
        print("=" * 80)
        print(out.answer)
        print("-" * 80)
        print(f"grounded={out.grounded}  sources={[s['tag'] for s in out.sources_used]}  "
              f"model={out.model}  {out.elapsed_s}s")
        if out.error:
            print(f"ERROR: {out.error}")
