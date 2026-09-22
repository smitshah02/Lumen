# Database lifecycle

Lumen's research database uses an external Docker volume. Compose never creates
or deletes that volume, and `docker compose down -v` cannot remove it.

## Clean setup

1. Copy `.env.example` to `.env` and change the local password in both
   `LUMEN_PG_PASSWORD` and `DATABASE_URL` to the same value.
2. Create the named external volume once:

   ```bash
   docker volume create lumen_pgdata
   ```

3. Keep `LUMEN_PG_VOLUME=lumen_pgdata`, then run the normal setup facade.

## Existing database

Do not rename, copy, or delete a volume during repository setup. Resolve the
existing volume name with `docker volume ls`, set that exact name in
`LUMEN_PG_VOLUME`, and confirm it before running Compose. There is intentionally
no hash-like fallback in the repository: selecting an existing database is an
operator decision.

`LUMEN_PG_PASSWORD` supplies the container credential and must match the
password embedded in `DATABASE_URL`. Changing it does not rewrite the password
inside an already initialized Postgres volume; update Postgres deliberately
before changing an existing installation's connection string.

Schema creation and upgrades are separate from container lifecycle:

```bash
python -m src.storage.schema --upgrade
python -m src.storage.schema --version
```

Upgrades are idempotent and versioned. They never ingest, truncate, or delete
clinical data.
