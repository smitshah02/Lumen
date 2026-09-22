# Archive

Everything below this directory is retained for scientific or implementation
provenance and is outside the supported runtime import path.

| Directory | Retired material | Current replacement |
|---|---|---|
| `legacy_generation/` | Direct `AnswerGenerator` and batch harness | `src/agents/graph.py` + `src/llm/local_client.py` |
| `legacy_retrieval_harnesses/` | Manual retriever/temporal scripts | `HybridRetriever` + tests in `tests/` |
| `reranker_training/` | Offline reranker-data/training utilities | Runtime BGE reranker loaded directly by retrieval |
| `longitudinal_hf/` | One-off longitudinal Hugging Face study | No production replacement; historical research only |
| `historical/` | Point-in-time audit notes | Current README and `docs/` |

Do not import archive modules from active code, deploy them, or interpret them
as maintained command interfaces. Historical files may mention paths, models,
or architecture that no longer apply.
