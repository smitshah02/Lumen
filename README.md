# Lumen

Lumen is a local clinical retrieval-augmented generation system for
patient-scoped questions over de-identified MIMIC-IV notes. Its authoritative
runtime is:

```text
FastAPI → LangGraph → HybridRetriever → role-based local Ollama
        → citation normalization and verification → review/refusal → API response
```

The repository supports two isolated data planes:

- `demo` contains only generated synthetic patients and is the only plane
  permitted on RunPod or other remote demo infrastructure.
- `research` contains real MIMIC-derived data and is intended for controlled
  local or institutional execution. The API is loopback-only and remote model
  endpoints are rejected by default.

Lumen is a research system, not a clinical decision-support product.

## Quick start: synthetic demo

Prerequisites are CPython 3.12, Docker Compose, sufficient disk/RAM for the
configured local models, and the values from `.env.example`.

```bash
cp .env.example .env
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
./scripts/lumen demo setup
curl http://127.0.0.1:8000/ready
./scripts/lumen demo test
```

`demo setup` is explicit because it downloads model weights, initializes the
synthetic database, and builds the retrieval index. Routine startup does not
ingest or reindex:

```bash
./scripts/lumen demo start
./scripts/lumen demo stop
```

## Research plane

Research setup is deliberately separate. Choose an existing or newly created
external Postgres volume, configure local model endpoints and MIMIC paths, then
run each state-changing step explicitly:

```bash
./scripts/lumen research doctor
./scripts/lumen research db
./scripts/lumen research schema
./scripts/lumen research ingest --confirm-research-ingest
./scripts/lumen research index
./scripts/lumen research start
```

Never put MIMIC data, derived patient text, credentials, model weights, or
evaluation evidence containing note text in Git. See the
[research runbook](docs/runbooks/research.md) before using this plane.

## Evaluation

The evaluation surfaces have distinct jobs:

- `demo_smoke_test.py`: live safety and retrieval preflight.
- `src/evals/`: reproducible retrieval, temporal, and egress evaluation.
- `scripts/final_eval.py`: canonical answer-level evaluation with immutable
  run artifacts and an independent local judge.
- `scripts/performance_eval.py`: deployment, latency, GPU, call-budget, and
  tracing evidence.

Common commands:

```bash
./scripts/lumen demo eval --subset smoke --skip-judge
./scripts/lumen demo performance
./scripts/lumen research test
./scripts/lumen research eval --out /path/to/pooled.json
```

Published retrieval evidence and the configuration it supports are preserved
under [`artifacts/benchmarks/retrieval-v1/`](artifacts/benchmarks/retrieval-v1/).
The 28-query pooled benchmark reports P@5 `0.843`, recall@5 `0.276`, MRR
`0.884`, and nDCG@5 `0.687` for Hybrid RRF + BGE reranking. Recall uses the
union of judged pools as its denominator, so it is intended for comparison
across configurations rather than as corpus-wide recall.

## Repository map

| Path | Purpose |
|---|---|
| [`src/api/`](src/api/) | Thin FastAPI boundary and readiness checks |
| [`src/agents/`](src/agents/) | Canonical LangGraph routing, synthesis, verification, and review |
| [`src/retrieval/`](src/retrieval/) | MedCPT/BM25 hybrid retrieval, BGE reranking, and indexing |
| [`src/storage/`](src/storage/) | Versioned schema, ingestion, and checkpoint setup |
| [`src/generation/lab_query.py`](src/generation/lab_query.py) | Shared deterministic structured-lab service |
| [`src/evals/`](src/evals/) | Retrieval, safety, temporal, and final-evaluation engines |
| [`src/mcp_server/`](src/mcp_server/) | Optional MCP adapter over shared services |
| [`scripts/lumen`](scripts/lumen) | Thin supported lifecycle facade |
| [`archive/`](archive/) | Retired implementations retained only for provenance |

Further reading:

- [Architecture](docs/architecture.md)
- [Demo runbook](docs/runbooks/demo.md)
- [Research runbook](docs/runbooks/research.md)
- [Configuration reference](docs/configuration.md)
- [Operations](docs/operations.md)
- [Runtime compatibility](docs/runtime-compatibility.md)
- [Database lifecycle](docs/database-lifecycle.md)
- [Infrastructure pins](docs/infrastructure-pins.md)
