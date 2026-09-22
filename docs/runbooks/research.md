# Real-MIMIC research runbook

The research plane is for controlled local or institutional environments with
authorized MIMIC-IV access. It must not be deployed by the RunPod demo scripts.
The commands below are explicit because ingestion and indexing are expensive,
privacy-sensitive state changes.

## Prerequisites

- Authorized local MIMIC-IV and MIMIC-IV-Note files.
- CPython 3.12 and dependencies from `requirements.txt`.
- Local Ollama with the MAIN, FAST, and (for final judging) independent judge
  tags in `configs/models.json`.
- Pinned retrieval weights installed with `scripts/fetch_models.py`.
- An approved external Postgres Docker volume. Read
  [database-lifecycle.md](../database-lifecycle.md) before selecting one.

## Configure

Copy `.env.example` to `.env`, change the password, and set:

- `LUMEN_DATA_PLANE=research`
- matching `DATABASE_URL` and `LUMEN_PG_PASSWORD`
- the exact `LUMEN_PG_VOLUME`
- local `LUMEN_LLM_HOST` and `LUMEN_JUDGE_HOST`
- `MIMIC_IV_DIR`, `MIMIC_IV_NOTE_DIR`, `LUMEN_MODELS_DIR`, and optional
  `LUMEN_GUIDELINES_DIR`

The doctor never prints credentials:

```bash
./scripts/lumen research doctor
```

## Initialize and ingest

For a new database, create/select the external volume, start Postgres, then
apply the versioned schema and explicit checkpoint schema:

```bash
docker volume create lumen_pgdata       # new installs only
./scripts/lumen research db
./scripts/lumen research schema
```

Load authorized source files only after confirming the target database:

```bash
./scripts/lumen research ingest --confirm-research-ingest
./scripts/lumen research index
```

Ingestion records running/completed/failed state. Indexing transactionally
replaces each note's chunks and records completion plus configuration
provenance. Neither doctor nor normal API startup performs ingestion.

## Start and operate

Start Ollama separately, confirm all configured tags are installed, then:

```bash
./scripts/lumen research doctor
./scripts/lumen research start
curl http://127.0.0.1:8000/ready
```

The research API binds to `127.0.0.1` through the facade and rejects
non-loopback clients. Remote LLM and judge endpoints fail closed. Do not enable
the remote-model override unless the endpoint and network are explicitly
approved for the data involved. Research tracing is always restricted to a
local Langfuse endpoint because spans can contain note text.

## Test and evaluate

```bash
./scripts/lumen research test
./scripts/lumen research eval --out /approved/local/path/pooled.json
```

The checked-in final answer evaluation has only a synthetic demo provider and
will refuse the research plane. A future research case provider must remain
local, explicitly registered, and must never commit patient examples or
evidence artifacts.

Review all generated output before moving or committing it. Retrieval pools,
judge caches, tracing exports, logs, and evidence caches may contain derived
clinical text even when aggregate scorecards do not.
