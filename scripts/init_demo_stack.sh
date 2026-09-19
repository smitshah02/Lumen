#!/usr/bin/env bash
# One-time, idempotent initialization of the SYNTHETIC demo stack
# (docker-compose.demo.yml). Reuses the normal Lumen paths:
#   model weights -> scripts/fetch_models.py      (pinned HF revisions, /models volume)
#   LLM           -> ollama pull qwen3:8b          (skipped if already present)
#   data          -> scripts/load_synthetic_demo.py (refuses any plane but demo)
#   index         -> python -m src.retrieval.index_notes (only unindexed notes)
# Safe to re-run: every step skips or replaces synthetic rows only.
set -euo pipefail

cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f docker-compose.demo.yml)
MODEL="qwen3:8b"

if [ "${LUMEN_DATA_PLANE:-demo}" != "demo" ]; then
  echo "refusing: LUMEN_DATA_PLANE=${LUMEN_DATA_PLANE}; the demo stack only runs the demo plane" >&2
  exit 2
fi

echo "==> db + ollama"
"${COMPOSE[@]}" up -d --wait db ollama

echo "==> confirm the api container resolves to the demo database"
"${COMPOSE[@]}" run --rm --no-deps api python -c "
from src import storage
assert storage.DATA_PLANE == 'demo', storage.DATA_PLANE
assert storage.engine.url.database == 'lumen_demo', storage.engine.url.database
print('plane=demo database=lumen_demo')"

echo "==> retrieval model weights (MedCPT, BGE)"
"${COMPOSE[@]}" run --rm --no-deps -e HF_HUB_OFFLINE=0 api python scripts/fetch_models.py

echo "==> ${MODEL} in the ollama service"
if "${COMPOSE[@]}" exec -T ollama ollama list | awk 'NR>1 {print $1}' | grep -qx "${MODEL}"; then
  echo "${MODEL}: present, skipping"
else
  "${COMPOSE[@]}" exec -T ollama ollama pull "${MODEL}"
fi

echo "==> synthetic records"
"${COMPOSE[@]}" run --rm --no-deps api python scripts/load_synthetic_demo.py

echo "==> note chunks + MedCPT embeddings"
"${COMPOSE[@]}" run --rm --no-deps api python -m src.retrieval.index_notes

echo "==> api"
"${COMPOSE[@]}" up -d api
echo "done. check: curl http://127.0.0.1:8000/ready"
