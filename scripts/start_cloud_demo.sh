#!/usr/bin/env bash
# Start / stop the SYNTHETIC demo services on the RunPod Pod (demo plane only).
# Requires a completed scripts/bootstrap_pod.sh (which calls this script).
#
# Runtime state is Pod-local (/root/lumen-runtime); the geesefs Global Volume
# (/workspace/lumen) only holds code, log snapshots and results.
#
#   bash scripts/start_cloud_demo.sh            # all: postgres, ollama, api + health checks
#   bash scripts/start_cloud_demo.sh status
#   bash scripts/start_cloud_demo.sh stop       # stops processes; keeps all data and caches
set -euo pipefail

PERSIST_ROOT=${LUMEN_ROOT:-/workspace/lumen}
REPO_ROOT=$PERSIST_ROOT/repo
RUNTIME_ROOT=${LUMEN_RUNTIME_ROOT:-/root/lumen-runtime}
VENV_ROOT=$RUNTIME_ROOT/venv
RUN_LOGS=$RUNTIME_ROOT/logs
ENV_FILE=$RUNTIME_ROOT/lumen.env
PGBIN=/usr/lib/postgresql/16/bin
API=http://127.0.0.1:8000

die() { echo "ERROR: $*" >&2; exit 1; }
[ -f "$ENV_FILE" ] || die "no $ENV_FILE — run scripts/bootstrap_pod.sh first"
set -a; . "$ENV_FILE"; set +a
[ "${LUMEN_DATA_PLANE:-}" = "demo" ] || die "refusing: LUMEN_DATA_PLANE is not demo"
case "$LUMEN_DEMO_DATABASE_URL" in */lumen_demo) ;; *) die "refusing: database is not lumen_demo" ;; esac
case "${OLLAMA_MODELS:-}" in /workspace/*|"") die "refusing: OLLAMA_MODELS must be Pod-local, got '${OLLAMA_MODELS:-}'" ;; esac
BIND=${LUMEN_API_BIND:-127.0.0.1}
[ "$BIND" = "127.0.0.1" ] || [ "${LUMEN_API_ALLOW_NONLOCAL:-0}" = "1" ] \
  || die "refusing to bind the unauthenticated API to $BIND (set LUMEN_API_ALLOW_NONLOCAL=1 to override)"
mkdir -p "$RUN_LOGS"

start_postgres() {
  if "$PGBIN/pg_isready" -q -h 127.0.0.1 -p 5432; then echo "postgres: running"; return; fi
  pg_ctlcluster 16 main start
  echo "postgres: started (/var/lib/postgresql/16/main)"
}

start_ollama() {
  if curl -sf -m 2 http://127.0.0.1:11434/api/version >/dev/null; then echo "ollama: running"; return; fi
  OLLAMA_MODELS=$OLLAMA_MODELS OLLAMA_HOST=127.0.0.1:11434 nohup ollama serve >> "$RUN_LOGS/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do curl -sf -m 2 http://127.0.0.1:11434/api/version >/dev/null && { echo "ollama: started"; return; }; sleep 1; done
  die "ollama did not start (see $RUN_LOGS/ollama.log)"
}

start_api() {
  if ! curl -sf -m 2 "$API/health" >/dev/null; then
    (cd "$REPO_ROOT" && nohup "$VENV_ROOT/bin/uvicorn" src.api.app:app --host "$BIND" --port 8000 --workers 1 \
        >> "$RUN_LOGS/api.log" 2>&1 & echo $! > "$RUNTIME_ROOT/api.pid")
    for _ in $(seq 1 60); do curl -sf -m 2 "$API/health" >/dev/null && break; sleep 1; done
  fi
  echo "health: $(curl -s -m 5 -w ' HTTP %{http_code}' "$API/health")"
  for _ in $(seq 1 30); do
    code=$(curl -s -m 10 -o "$RUN_LOGS/ready.json" -w '%{http_code}' "$API/ready" || true)
    [ "$code" = "200" ] && break; sleep 2
  done
  echo "ready:  $(cat "$RUN_LOGS/ready.json") HTTP $code"
  [ "$code" = "200" ] || die "API is not ready"
}

stop_all() {
  pkill -f "uvicorn src.api.app:app" && echo "api: stopped" || echo "api: not running"
  pkill -f "ollama serve" && echo "ollama: stopped" || echo "ollama: not running"
  if "$PGBIN/pg_isready" -q -h 127.0.0.1 -p 5432; then
    pg_ctlcluster 16 main stop -m fast && echo "postgres: stopped"
  else echo "postgres: not running"; fi
  echo "venv, models and qwen3:8b remain under $RUNTIME_ROOT; code/logs/results under $PERSIST_ROOT"
}

status() {
  "$PGBIN/pg_isready" -h 127.0.0.1 -p 5432 || true
  curl -s -m 2 http://127.0.0.1:11434/api/version && echo || echo "ollama: down"
  OLLAMA_HOST=127.0.0.1:11434 ollama ps 2>/dev/null || true
  curl -s -m 10 "$API/ready" && echo || echo "api: down"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
}

case "${1:-all}" in
  postgres) start_postgres ;;
  ollama)   start_ollama ;;
  api)      start_api ;;
  all)      start_postgres; start_ollama; start_api ;;
  stop)     stop_all ;;
  status)   status ;;
  *) die "usage: $0 [all|postgres|ollama|api|status|stop]" ;;
esac
