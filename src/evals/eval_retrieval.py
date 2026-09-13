"""
RETIRED — do not use this harness.
==================================
This module used to run the golden set and print a P@5 / R@5 / MRR / nDCG@5
summary table. Those numbers were wrong in three independent ways, and they
contradicted the numbers in README.md, which come from the two-phase pipeline
below. It has been retired rather than repaired.

What was wrong with it
----------------------
1. It graded with a KEYWORD judge (`judge_relevance`): a chunk counted as
   relevant if it contained any keyword from the golden query's
   `relevance_criteria`. BM25 retrieves on those same keywords, so the judge
   scored highest exactly what the retriever was built to find. README.md
   already says this judge was abandoned for being circular and saturating.

2. Its nDCG@k built the ideal DCG from the RETRIEVED results, not from the
   relevance pool. So finding one relevant chunk at rank 1 scored nDCG = 1.000,
   identical to finding five out of five. Measured on the real run, 102 of 140
   config x query cells were exactly 1.0 — the metric could not separate the
   configurations it existed to compare.

3. Its recall@k divided by `min_relevant`, a hand-written constant (2 or 3) in
   golden_dataset.py, rather than by the size of the judged relevance pool.

On top of that, the head-to-head "wins" table broke ties with `>`, so all 23
tied queries were silently awarded to whichever config came first in the dict —
which is where "BM25 Only wins 22/28" came from.

What to use instead
-------------------
Two phases, deliberately separate processes so the reranker and the judge model
never share memory:

    # phase 1 — retrieval only (needs Postgres, not Ollama)
    python -m src.evals.retrieve_pool --out pooled.json

    # phase 2 — pooled LLM judging + scoring (needs Ollama, not Postgres)
    python -m src.evals.judge_and_score --in pooled.json --export results.json

That pipeline pools every config's results per query, judges the union once
with a local model, and scores each config against the shared relevant set:
graded nDCG with the ideal built from the full pool, recall against the pooled
denominator, and a `_meta` block recording model / threshold / top_k. It is the
pipeline that produced the table in README.md.

The retrieval configurations themselves were the sound half of this file and
now live in `src/evals/retrieval_configs.py`.

See AUDIT_NOTES.md for the full measurements behind each point above.
"""

from __future__ import annotations

import sys

_MESSAGE = __doc__


def main() -> int:
    sys.stderr.write(_MESSAGE)
    sys.stderr.write(
        "\nRefusing to run: this harness produces numbers that contradict "
        "README.md.\nUse the two-phase pipeline shown above.\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
