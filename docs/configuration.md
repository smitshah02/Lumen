# Configuration reference

Lumen reads `.env` through `src/__init__.py`; already exported environment
values win. Configuration is evaluated at import time, so restart Python
processes after changing it. Never commit populated environment files.

## Data plane and database

| Variable | Default | Meaning |
|---|---|---|
| `LUMEN_DATA_PLANE` | `research` | Exactly `demo` or `research`. |
| `DATABASE_URL` | `postgresql://postgres:lumen@localhost:5434/lumen` | Research SQLAlchemy URL. Set explicitly for real use. |
| `LUMEN_DEMO_DATABASE_URL` | derived from `DATABASE_URL` with database `lumen_demo` | Full demo URL; must not resolve to the research database. |
| `LUMEN_DEMO_DB_NAME` | `lumen_demo` | Database name used when deriving the demo URL. |
| `LUMEN_DB_CONNECT_TIMEOUT` | `10` | Postgres connection timeout in seconds. |
| `LUMEN_PG_PASSWORD` | required by research Compose | Container password; must match `DATABASE_URL`. |
| `LUMEN_PG_PORT` | `5434` | Research Compose host port. |
| `LUMEN_PG_VOLUME` | required by research Compose | Exact external volume name. No implicit volume is selected. |
| `LUMEN_DEMO_PG_PASSWORD` | `lumen-demo-local-only` in local demo Compose; generated on RunPod | Synthetic database password. |
| `LUMEN_CHECKPOINT_AUTO_SETUP` | `0` | Defensive checkpoint-schema creation during graph build. Normal setup is explicit. |

The research Compose variables are required even if the application can derive
older development defaults. This is intentional: choosing a real database and
volume must be explicit.

## Local models

Canonical tags and pinned Hugging Face revisions live in
`configs/models.json`.

| Variable | Default | Meaning |
|---|---|---|
| `LUMEN_LLM_HOST` | `http://localhost:11434` | Ollama base URL. Research permits only local hosts unless explicitly overridden. |
| `LUMEN_LLM_MAIN` | registry `runtime_main` | Complex/longitudinal synthesis tag. |
| `LUMEN_LLM_FAST` | registry `runtime_fast` | Triage fallback, concepts, simple synthesis, and verification tag. |
| `LUMEN_LLM_KEEPALIVE` | `10m` | Ollama model residency duration. |
| `LUMEN_LLM_WARMUP` | `0` | `1` loads both tiers in a background startup thread. |
| `LUMEN_DETERMINISTIC_LABS` | `1` | Answer supported latest-lab questions directly from `labevents`. |
| `LUMEN_RESEARCH_ALLOW_REMOTE_MODELS` | `0` | Conspicuous override for remote research LLM/judge endpoints. Does not relax tracing policy. |
| `LUMEN_JUDGE_MODEL` | registry `independent_judge` | Final/retrieval evaluation judge; must differ from runtime tiers. |
| `LUMEN_JUDGE_HOST` | `LUMEN_LLM_HOST` | Independent judge Ollama endpoint, subject to the same research policy. |
| `LUMEN_JUDGE_NUM_CTX` | `12288` | Final-judge context window. |
| `LUMEN_JUDGE_MAX_TOKENS` | `1600` | Final-judge output budget. |
| `LUMEN_JUDGE_CACHE` | `.cache/final_eval_judge.json` | Local verdict cache; may contain evaluation-derived content. |

`LUMEN_DEMO_LLM_HOST` is a Compose substitution used to set
`LUMEN_LLM_HOST` inside the demo API container. It defaults to
`http://ollama:11434`; `http://host.docker.internal:11434` selects a host
Ollama daemon.

## Paths

| Variable | Default | Meaning |
|---|---|---|
| `LUMEN_DATA_DIR` | `<repo>/data` | Base data directory. |
| `MIMIC_IV_DIR` | `<data>/mimiciv` | MIMIC-IV core directory. |
| `MIMIC_IV_NOTE_DIR` | `<data>/mimic-iv-note` | MIMIC-IV-Note directory. |
| `LUMEN_GUIDELINES_DIR` | `<data>/guidelines` | Local guideline documents. |
| `LUMEN_MODELS_DIR` | `<repo>/models` | Retrieval-model directories and provenance manifests. |
| `LUMEN_EVAL_RESULTS_ROOT` | `<repo>/results/final_eval` | Final-evaluation run root. Completed runs are immutable. |

`HF_HOME`, `HF_HUB_OFFLINE`, and `OLLAMA_MODELS` are standard Hugging Face and
Ollama variables used by containers/RunPod to keep caches and weights on the
intended filesystem. The API image sets Hugging Face offline mode; model setup
temporarily sets it to `0` for an explicit download.

## Retrieval

| Variable | Default | Meaning |
|---|---|---|
| `LUMEN_RRF_BM25_WEIGHT` | `1.5` | Full-text contribution to reciprocal-rank fusion. |
| `LUMEN_RRF_VECTOR_WEIGHT` | `0.75` | MedCPT vector contribution to fusion. |
| `LUMEN_QUERY_EXPANSION` | `0` | Enable deterministic clinical query expansion. |
| `LUMEN_HNSW_EF_SEARCH` | `1000` | pgvector HNSW search effort, clamped to 1–1000. |

These are published tuned defaults. Changes alter index/query provenance and
should be evaluated rather than silently made per deployment.

## API, logs, and tracing

| Variable | Default | Meaning |
|---|---|---|
| `LUMEN_API_PORT` | `8000` | Port used by the research facade. |
| `LUMEN_API_BIND` | `127.0.0.1` | Demo Compose published address and RunPod bind address. |
| `LUMEN_API_ALLOW_NONLOCAL` | `0` | RunPod-only explicit override for an unauthenticated non-loopback bind. |
| `LUMEN_LOG_LEVEL` | `INFO` | Structured application log level. |
| `LUMEN_TRACING` | `0` | Enable Langfuse tracing when set to exactly `1`. |
| `LANGFUSE_BASE_URL` | SDK default if unset | Preferred Langfuse endpoint. |
| `LANGFUSE_HOST` | unset | Deprecated endpoint alias used only when `LANGFUSE_BASE_URL` is absent. |
| `LANGFUSE_PUBLIC_KEY` | unset | Langfuse credential; never reported by readiness. |
| `LANGFUSE_SECRET_KEY` | unset | Langfuse credential; never reported by readiness. |

Research tracing accepts local endpoints only. Demo tracing accepts local
endpoints or remote HTTPS. Traces may contain retrieved text and prompts.

## Facade and RunPod deployment

| Variable | Default | Meaning |
|---|---|---|
| `LUMEN_PYTHON` | `.venv/bin/python`, then `PYTHON`/`python3` | Interpreter selected by `scripts/lumen`. |
| `LUMEN_ROOT` | `/workspace/lumen` on RunPod | Persistent source/log/result root used by performance tooling. |
| `LUMEN_RUNTIME_ROOT` | `/root/lumen-runtime` on RunPod | Pod-local venv/model/log/PID root. |
| `LUMEN_REPO_URL` | unset | Optional Git source for bootstrap; normal sync uses an explicit allowlist. |

`OLLAMA_HOST` is set internally by RunPod service commands to address the local
daemon. `OLLAMA_VERSION` is passed to the official installer from the pinned
bootstrap constant. They are operational implementation details, not Lumen
application configuration.

## Valid combinations

- `research`: `DATABASE_URL`, `LUMEN_PG_PASSWORD`, and `LUMEN_PG_VOLUME` must
  identify the intended database; LLM and judge endpoints are local by default;
  API and tracing are loopback/local.
- `demo` local Docker: the API always receives plane `demo`, a `lumen_demo`
  URL, and explicit model tags. It cannot select the research volume.
- `demo` RunPod: bootstrap writes a private Pod-local environment file and
  refuses research paths, database names, or synchronized restricted roots.
