"""Lumen final answer-level evaluation framework.

Deterministic answer-level evaluation, an independent offline LLM judge, and
versioned run artifacts for the FROZEN Lumen architecture. Nothing in this
package may import from the online path in a way that changes its behaviour:
the graph, retriever, API and prompts are read, executed and recorded, never
modified.
"""

EVALUATOR_VERSION = "final_eval/1.0.0"
