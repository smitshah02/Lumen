# Operations

`scripts/lumen` is the supported runtime lifecycle entry point; `make help`
provides development/setup shortcuts. Commands are intentionally
separate so setup never implies research ingestion, a deployment, or a model
benchmark.

Typical clean local setup:

```bash
cp .env.example .env                 # then change credentials
docker volume create lumen_pgdata    # once; or select an existing volume
make setup
make compose-check
make db-up
make schema
make models
```

The equivalent day-to-day commands are `scripts/lumen research doctor`,
`scripts/lumen research db`, `scripts/lumen research schema`, and
`scripts/lumen research start`. Research ingestion requires the conspicuous
`scripts/lumen research ingest --confirm-research-ingest` form.

Research ingestion remains an explicit expert action:

```bash
.venv/bin/python -m src.storage.ingest
.venv/bin/python -m src.retrieval.index_notes
```

It is deliberately not a dependency of `setup`, `validate`, or `doctor`.

For the isolated synthetic demo, use `make demo-init`. For routine checks use
`make validate`; it renders configuration, verifies generated demo fixtures,
and runs tests without starting services, downloading weights, or ingesting
data. `make smoke-demo` requires an already initialized live demo stack.

Schema upgrades are additive and explicit (`make schema`). Model downloads use
the `runtime` profile only; historical evaluation weights require an explicit
`--profile legacy-eval` invocation.
