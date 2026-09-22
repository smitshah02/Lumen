# Legacy retrieval harnesses

These scripts are preserved for historical provenance but are not authoritative
runtime or test entry points.

- `test_retriever_v3.py` manually replayed individual retrieval stages for an
  interactive comparison. The production implementation is
  `src.retrieval.hybrid_retriever_v2.HybridRetriever.search`.
- `temporal_fix.py` documented the original per-patient temporal correction and
  contrasted it with the pre-fix implementation. Its production assertions now
  live in `tests/test_temporal_retrieval.py`; real-data temporal measurement
  remains in `src/evals/eval_temporal.py`.

The archived scripts retain their former imports and narrative output. They are
not supported commands and must not be used as alternative implementations.
