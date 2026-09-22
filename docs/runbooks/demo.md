# Synthetic demo runbook

The demo plane contains generated records only. It is the sole supported plane
for Docker demo deployment and RunPod.

## Local Docker

1. Install CPython 3.12 dependencies and copy `.env.example` to `.env`.
2. Keep `LUMEN_LLM_MAIN` and `LUMEN_LLM_FAST` synchronized with
   `configs/models.json` (the example already tracks them).
3. Initialize once:

   ```bash
   ./scripts/lumen demo setup
   ```

   This starts isolated demo Postgres/Ollama services, downloads pinned
   retrieval weights, pulls both Ollama tiers, loads generated fixtures,
   creates checkpoint tables, indexes notes, and starts the API.

4. Check and test:

   ```bash
   ./scripts/lumen demo doctor
   curl http://127.0.0.1:8000/health
   curl http://127.0.0.1:8000/ready
   ./scripts/lumen demo test
   ```

5. Day-to-day lifecycle:

   ```bash
   ./scripts/lumen demo stop
   ./scripts/lumen demo start
   ```

Stopping preserves named volumes. Do not add `-v` unless you intentionally want
to destroy synthetic demo state.

## Evaluation

```bash
./scripts/lumen demo eval --subset smoke --skip-judge
./scripts/lumen demo performance
./scripts/lumen demo tracing
```

The full final evaluation needs the independent judge model from
`configs/models.json`. `--skip-judge` is deterministic-only and reports that
fact in the run manifest.

## RunPod

RunPod deployment remains an explicit two-step operational path; it is not run
by the facade:

```bash
scripts/sync_to_pod.sh root@POD_HOST SSH_PORT /path/to/key
ssh -p SSH_PORT root@POD_HOST \
  'LUMEN_DATA_PLANE=demo bash /workspace/lumen/repo/scripts/bootstrap_pod.sh'
```

The sync script sends only an allowlisted source set. It excludes `data/`,
`models/`, `backups/`, `.env*`, results, API-key files, caches, and reranker
training data. Bootstrap refuses research-era paths or a database named
`lumen`, builds only `lumen_demo`, and uses generated synthetic fixtures.

Service control on the Pod:

```bash
bash /workspace/lumen/repo/scripts/start_cloud_demo.sh status
bash /workspace/lumen/repo/scripts/start_cloud_demo.sh stop
bash /workspace/lumen/repo/scripts/start_cloud_demo.sh all
```

API and Ollama shutdown uses validated PID ownership; no broad process matching
is used. Access the unauthenticated API through an SSH tunnel. Non-loopback API
binding requires the explicit `LUMEN_API_ALLOW_NONLOCAL=1` override.

The bootstrap performs network downloads and package installation. Static CI
and local regression tests do not deploy or contact RunPod.
