# Retrieval benchmark artifacts (v1)

This directory is the immutable, aggregate-only evidence bundle for the
retrieval configuration published in the project README. It contains no saved
pooled retrieval payloads or note text.

- `results.json` and `results_baseline.json` are the historical aggregate
  comparisons.
- `interview_baseline/` records the frozen five-configuration baseline. Its
  `run_manifest.json` contains the full environment, model revisions, commands,
  and provenance (Git SHA `40eae7c3615e601edf5643291c527c9b12477f66`).
- `interview_tuning/` records the registered dev/held-out tuning protocol,
  sweeps, frozen selection, and one-time held-out verification.

The selected production parameters are authoritative in
`interview_tuning/selected_config.json`: BM25/vector weights `1.5/0.75`, RRF
`k=60`, overlap bonus `0.5`, 40 rerank candidates, and query expansion off.
Runtime defaults mirror that frozen selection. Historical JSON fields may
still mention the paths used when the runs were captured; those files are not
rewritten so their provenance remains intact.

The active benchmark framework stays in `src/evals/` because the published
comparison and tuned configuration depend on its pooled-relevance protocol; it
is not a second serving implementation.
