# Architecture

## Canonical request path

```text
HTTP client
  → FastAPI (`src/api/app.py`)
  → LangGraph (`src/agents/graph.py`)
      → deterministic classification / structured lab lookup when applicable
      → HybridRetriever (`src/retrieval/hybrid_retriever_v2.py`)
          → query expansion (optional)
          → Postgres full-text + MedCPT vector retrieval
          → reciprocal-rank fusion and temporal handling
          → adjacent-chunk context and BGE cross-encoder reranking
      → role-based local Ollama (`src/llm/local_client.py`)
      → citation normalization and claim verification
      → auto-approval, refusal, or checkpointed human review
  → typed API response
```

FastAPI is intentionally thin. It owns HTTP validation, request identity,
loopback enforcement for research, dependency errors, and response schemas. It
does not implement a second retrieval or generation path. The CLI runner and
final evaluator call the same graph.

The LLM client exposes roles rather than model names. Simple synthesis,
classification fallback, concept extraction, and verification use the FAST
tier; complex longitudinal synthesis uses MAIN. Defaults come from
`configs/models.json` and deployments may override the tags.

Every factual answer is normalized into claims and citations. Deterministic
checks resolve claims where possible; unresolved claims go to the local FAST
verifier. Unsupported output is refused or checkpointed for review rather than
silently presented as verified.

## Data and state

Postgres 16 + pgvector stores structured MIMIC fields, de-identified notes,
chunks, embeddings, ingestion state, index provenance, and versioned schema
state. LangGraph checkpoint tables are created by an explicit idempotent setup
command. Normal graph construction validates them without mutating schema.

Indexing records the chunker configuration, embedding model revision, vector
dimension, expected note count, completed note count, and run state. A unique
`(note_id, chunk_index)` constraint prevents duplicate positions. Per-note
replacement is transactional.

## Data planes

`LUMEN_DATA_PLANE=demo` selects a separate `lumen_demo` database populated from
the checked-in generated fixtures. These identifiers are confined to the
synthetic range. Remote HTTPS tracing is permitted only on this plane.

`LUMEN_DATA_PLANE=research` selects the configured research database. HTTP is
loopback-only. Model and judge endpoints must be loopback or
`host.docker.internal` unless the conspicuous
`LUMEN_RESEARCH_ALLOW_REMOTE_MODELS=1` override is set. Tracing remains local
even when that model override is enabled.

## Optional components

- The MCP server is an adapter over the same database, HybridRetriever, and
  `LabResolver`; it does not define plane policy or carry a second demo corpus.
- Langfuse tracing is disabled by default and degrades to a no-op. Research
  traces may only target a local endpoint.
- Guideline retrieval is separate supporting evidence and is never represented
  as patient-specific history.
- Archived generation, retrieval harnesses, and training studies are not on
  the import path and are not supported runtime alternatives.

## Evaluation boundaries

Smoke/preflight, retrieval quality, safety, answer quality, and performance are
separate measurements. The final evaluator consumes cases through a provider
protocol; only the frozen synthetic demo provider is registered. This keeps the
engine reusable without embedding or enabling real-patient cases.
