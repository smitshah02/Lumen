# Archived reranker training utilities

These files preserve the earlier CSV-to-training-example pipeline and its
external Groq quality-audit helper. They are research provenance, not part of
Lumen's serving or evaluation runtime.

The quality audit can transmit training text to an external service and must
not be used with restricted MIMIC text under the PhysioNet DUA. Its optional
`openai` dependency is intentionally absent from the production requirements.

The ignored, restricted training artifacts remain at
`src/reranker/reranker_training_data/`. They were deliberately not moved,
opened, or modified during archival. The archived scripts may retain paths from
their original location and are not supported commands.
