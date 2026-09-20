"""Lumen final answer-level evaluation framework.

Deterministic answer-level evaluation, an independent offline LLM judge, and
versioned run artifacts for the FROZEN Lumen architecture. Nothing in this
package may import from the online path in a way that changes its behaviour:
the graph, retriever, API and prompts are read, executed and recorded, never
modified.
"""

# 1.1.0 — aj2 independent judge (per-criterion assessment + consistency
# validation + one repair attempt), calibration, preflight doctor, read-only
# comparison and the API contract cross-check. The measured system is
# unchanged; this version identifies the EVALUATOR that produced a run.
EVALUATOR_VERSION = "final_eval/1.1.0"
