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

## Backup and restore (research database)

The research database holds MIMIC-derived text. Backups stay on this machine,
in the Git-ignored `backups/` directory, readable by the owner only.

Logical backup (portable, about 3 GB for the 5,000-patient cohort):

```bash
set -euo pipefail
mkdir -p backups && chmod 700 backups
TS=$(date +%Y%m%d_%H%M%S)
docker exec lumen-pg pg_dump -U postgres -d lumen -Fc > backups/lumen_$TS.dump
chmod 600 backups/lumen_$TS.dump
# Read every data block, not just the table of contents (a truncated dump
# still has a complete TOC). Exit status 0 means the archive is whole.
docker exec -i lumen-pg pg_restore -f /dev/null < backups/lumen_$TS.dump
shasum -a 256 backups/lumen_$TS.dump > backups/lumen_$TS.dump.sha256
```

Restore from a dump into a scratch database (slow: the HNSW index over every
chunk embedding is rebuilt):

```bash
docker cp backups/lumen_<ts>.dump lumen-pg:/tmp/restore.dump   # -j cannot read from stdin
docker exec lumen-pg createdb -U postgres lumen_restore
docker exec lumen-pg pg_restore -U postgres -d lumen_restore -j 4 /tmp/restore.dump
docker exec lumen-pg rm /tmp/restore.dump
```

Physical copy of a volume (byte-identical, keeps every index; stop the
database first):

```bash
docker stop lumen-pg
docker volume create <new_volume>
docker run --rm --user root --entrypoint sh -v <old_volume>:/from:ro -v <new_volume>:/to \
  pgvector/pgvector:0.8.6-pg16-bookworm -c 'cp -a /from/. /to/'
```

### Legacy volume kept as rollback

The cohort was originally created in an anonymous Docker volume by an unpinned
`pgvector/pgvector:pg16` container. It was copied into the named volume
`lumen_mimic_pgdata`, which the pinned Compose service now uses
(`LUMEN_PG_VOLUME=lumen_mimic_pgdata`). The original volume is unmodified and
is held by a stopped container, `lumen-pg-legacy`, created with plain
`docker create` on the original image and bound to loopback. To return to it:

```bash
docker stop lumen-pg && docker start lumen-pg-legacy     # same port, 127.0.0.1:5434
```

and back again with `docker stop lumen-pg-legacy && docker start lumen-pg`.
Never run both: they publish the same port.

Two things to know:

- The holder container must not carry Compose labels. A container created by
  this project's Compose file is treated as the `db` service even after a
  `docker rename`, and the next `docker compose up` replaces it.
- An anonymous volume is deleted by `docker volume prune` once no container
  references it. `lumen-pg-legacy` is what keeps the original volume safe;
  remove it only when the copy has been trusted for long enough.

The legacy database predates the versioned schema, so `/ready` reports it as
not ready.

## Adopting an index that predates provenance tracking

A database indexed before `note_index_state` existed has no record of which
chunker or model revision produced its chunks. After `schema --upgrade`:

```bash
python scripts/fetch_models.py --adopt-local      # manifests for weights already on disk
./scripts/lumen research adopt --check
./scripts/lumen research adopt --apply --confirm-legacy-adoption
```

Run adoption before `research index`. Until a legacy index is adopted, the
indexer refuses to run without `--reindex`, because every note would otherwise
look unindexed and be deleted and re-embedded. An adopted index stops counting
as indexed if the pinned embedding model in `configs/models.json` changes.

Adoption writes provenance rows only. It records the index under its own
`legacy-adopted` configuration hash rather than the current one, because the
current chunker configuration cannot be shown to have produced it. Readiness
accepts that hash and reports `index_provenance: legacy_adopted`; the indexer
treats adopted notes as indexed, so nothing is re-embedded unless `--reindex`
is passed explicitly.

## MIMIC text is local-only

All MIMIC text is DUA-restricted and stays on this machine, in every column and
every table: `clinical_notes.text_original` (the note as PhysioNet distributes it,
already de-identified), `text_deid` (that text after Presidio), `note_chunks` and
`note_chunks_v2`. No column is "safe to share". The research plane serves stdio
and loopback clients only, backups stay in the Git-ignored `backups/` directory,
and benchmark artifacts stay in the Git-ignored `reports/data_foundation/`.

The earlier rule "never read `text_original`" is retired: the control index is
built from `text_deid`, and the v2 index is built from the note as written. The
demo plane is unaffected. It holds synthetic notes only and no `text_original`.

## Clinical_notes readiness index

The research `lumen` database includes an additive partial index, `idx_notes_eligible`, on `clinical_notes(note_id)` using the same eligibility predicate as the readiness probe. It was added during Stage 4 pre-switch hardening to avoid a full-table scan of `clinical_notes` during `/ready`; it does not change note contents, retrieval semantics, chunking, embeddings, or the control retrieval path. This index is present in the research `lumen` database but is not required in the demo or holdout databases. The definition is included in `SCHEMA_SQL` for newly initialized databases, but the existing `--upgrade` path does not create it automatically for an already-populated database; on an existing large database it should be created manually with `CREATE INDEX CONCURRENTLY` using the definition in `src/storage/schema.py`. To roll back this additive schema change without affecting data or the v2 build, run `DROP INDEX CONCURRENTLY IF EXISTS idx_notes_eligible;`.
