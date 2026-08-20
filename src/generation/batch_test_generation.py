"""
Batch generation test — fire many questions at one patient
===========================================================
Runs a list of questions against a single subject_id through the RAG generator
and prints, for each: the answer, which sources it cited, the grounded flag, and
the raw retrieval arm counts (bm25 / vec / merged) pulled from the retriever's
own log line — so you can spot patterns fast:

  * grounded=False + bm25=0  -> the topic isn't in the indexed notes (e.g. vitals
    that live in structured tables). Refusal is correct here.
  * grounded=True citing only [S1] on a "trend"/"list all" question -> it likely
    answered from one chunk; bump --top-k.
  * refusal on an unanswerable question -> the safety property working.

The retriever is built ONCE and reused, so the torch models load a single time
for the whole sweep (not per question). Ollama keeps the model warm via
keep_alive, so later questions are faster than the first.

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    ollama serve            # (separate terminal, if not already running)

    # default question set:
    python -m src.generation.batch_test_generation --subject 10882916 --top-k 8

    # your own questions, one per line in a file:
    python -m src.generation.batch_test_generation --subject 10882916 --file questions.txt

    # pick model / export:
    python -m src.generation.batch_test_generation --subject 10882916 \
        --model qwen2.5:7b --top-k 8 --export sweep_10882916.json
"""
from __future__ import annotations

import re
import json
import logging
import argparse
from collections import Counter

from src.generation.answer_generator import AnswerGenerator, DEFAULT_GEN_MODEL

# A mix from easy -> demanding, ending with deliberately unanswerable stress
# tests. The last three SHOULD come back as the refusal sentence (grounded=False).
DEFAULT_QUESTIONS = [
    "Summarize this patient's hospital course.",
    "Why was the patient admitted, and what was done during the stay?",
    "What were the main diagnoses and how were they treated?",
    "What medications was the patient given, and what were they for?",
    "List all the patient's potassium values with their dates.",
    "How did the patient's creatinine or kidney function change over the admission?",
    "What imaging or procedures did the patient have, and what did they show?",
    "Did the patient have any signs of infection during the stay?",
    # --- unanswerable-from-notes stress tests: correct answer is the refusal ---
    "What is the patient's family history of cancer?",
    "What did the patient eat for breakfast on the third day?",
]

# Matches the retriever's INFO line: "... N results (bm25=60, vec=60, merged=90, +4 expanded)"
_COUNTS_RE = re.compile(r"bm25=(\d+).*?vec=(\d+).*?merged=(\d+)")


class RetrievalCountCapture(logging.Handler):
    """Grabs the retriever's most recent 'Search ... results (bm25=...)' line."""
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.last_msg = None

    def emit(self, record):
        msg = record.getMessage()
        if "results (bm25=" in msg:
            self.last_msg = msg

    def take(self) -> dict:
        """Return the last captured counts and clear it."""
        counts = {"bm25": None, "vec": None, "merged": None}
        if self.last_msg:
            m = _COUNTS_RE.search(self.last_msg)
            if m:
                counts = {"bm25": int(m.group(1)), "vec": int(m.group(2)), "merged": int(m.group(3))}
        self.last_msg = None
        return counts


def _returned_source_mix(retrieved) -> str:
    """How the returned top-k broke down by retrieval arm (from each chunk's sources)."""
    c = Counter()
    for r in retrieved:
        srcs = tuple(sorted(getattr(r, "sources", []) or []))
        if len(srcs) > 1:
            c["both"] += 1
        elif srcs:
            c[srcs[0]] += 1
    return " ".join(f"{k}={v}" for k, v in sorted(c.items())) or "-"


def run_batch(subject_id: int, questions: list[str], model: str, top_k: int,
              temporal: str = "auto", export_path: str = None, truncate: int = None):
    # Attach the count-capturing handler to the retriever's logger.
    ret_logger = logging.getLogger("src.retrieval.hybrid_retriever_v2")
    ret_logger.setLevel(logging.INFO)
    capture = RetrievalCountCapture()
    ret_logger.addHandler(capture)

    gen = AnswerGenerator(model=model, top_k=top_k, use_reranker=False)

    print("=" * 90)
    print(f"  BATCH GENERATION TEST — subject_id={subject_id}   model={model}   top_k={top_k}")
    print("=" * 90)

    rows = []
    for i, q in enumerate(questions, 1):
        out = gen.answer(q, subject_id=subject_id, temporal_filter=temporal)
        counts = capture.take()
        tags = [s["tag"] for s in out.sources_used]

        ans = out.answer
        if truncate and len(ans) > truncate:
            ans = ans[:truncate].rstrip() + " …"

        print("\n" + "-" * 90)
        print(f"[{i}/{len(questions)}]  grounded={out.grounded}  "
              f"bm25={counts['bm25']} vec={counts['vec']} merged={counts['merged']}  "
              f"returned_mix=({_returned_source_mix(out.retrieved)})  "
              f"sources={tags}  {out.elapsed_s}s")
        print(f"Q: {q}")
        print(f"A: {ans}")
        if out.error:
            print(f"!! ERROR: {out.error}")

        rows.append({
            "question": q,
            "answer": out.answer,
            "grounded": out.grounded,
            "sources_used": tags,
            "retrieval_counts": counts,
            "n_retrieved": len(out.retrieved),
            "elapsed_s": out.elapsed_s,
            "error": out.error,
        })

    # ---- summary ----
    grounded_n = sum(1 for r in rows if r["grounded"])
    refused_n = sum(1 for r in rows if not r["grounded"] and not r["error"])
    errored_n = sum(1 for r in rows if r["error"])
    zero_bm25 = [r["question"] for r in rows if r["retrieval_counts"]["bm25"] == 0]
    avg_t = sum(r["elapsed_s"] for r in rows) / len(rows) if rows else 0.0

    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print(f"  questions: {len(rows)}   grounded: {grounded_n}   refused: {refused_n}   errored: {errored_n}")
    print(f"  avg time/query: {avg_t:.1f}s")
    if zero_bm25:
        print(f"  bm25=0 (topic likely not in indexed notes) on {len(zero_bm25)} question(s):")
        for q in zero_bm25:
            print(f"      - {q}")

    ret_logger.removeHandler(capture)

    if export_path:
        with open(export_path, "w") as f:
            json.dump({"subject_id": subject_id, "model": model, "top_k": top_k, "results": rows}, f, indent=2)
        print(f"\n  exported -> {export_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Batch RAG generation test against one patient")
    ap.add_argument("--subject", type=int, required=True, help="subject_id to test")
    ap.add_argument("--file", type=str, default=None, help="Text file of questions, one per line")
    ap.add_argument("--model", type=str, default=DEFAULT_GEN_MODEL, help="Ollama generation model")
    ap.add_argument("--top-k", type=int, default=8, help="Chunks retrieved per question (default 8)")
    ap.add_argument("--temporal", type=str, default="auto", help="Temporal filter mode")
    ap.add_argument("--truncate", type=int, default=None, help="Truncate printed answers to N chars")
    ap.add_argument("--export", type=str, default=None, help="Export full results to JSON")
    args = ap.parse_args()

    if args.file:
        with open(args.file) as f:
            questions = [ln.strip() for ln in f if ln.strip()]
    else:
        questions = DEFAULT_QUESTIONS

    run_batch(args.subject, questions, model=args.model, top_k=args.top_k,
              temporal=args.temporal, export_path=args.export, truncate=args.truncate)
