#!/usr/bin/env bash
# Start / stop the SYNTHETIC demo services on the RunPod Pod (demo plane only).
# Requires a completed scripts/bootstrap_pod.sh (which calls this script).
#
#   bash scripts/start_cloud_demo.sh            # all: postgres, ollama, api + health checks
#   bash scripts/start_cloud_demo.sh status
#   bash scripts/start_cloud_demo.sh stop       # stops processes; keeps all data and caches
set -euo pipefail

ROOT=${LUMEN_ROOT:-/workspace/lumen}
REPO=$ROOT/repo
PGBIN=/usr/lib/postgresql/16/bin
PGDATA=$ROOT/postgres
API=http://127.0.0.1:8000

die() { echo "ERROR: $*" >&2; exit 1; }
[ -f "$ROOT/lumen.env" ] || die "no $ROOT/lumen.env — run scripts/bootstrap_pod.sh first"
set -a; . "$ROOT/lumen.env"; set +a
[ "${LUMEN_DATA_PLANE:-}" = "demo" ] || die "refusing: LUMEN_DATA_PLANE is not demo"
case "$LUMEN_DEMO_DATABASE_URL" in */lumen_demo) ;; *) die "refusing: database is not lumen_demo" ;; esac
BIND=${LUMEN_API_BIND:-127.0.0.1}
[ "$BIND" = "127.0.0.1" ] || [ "${LUMEN_API_ALLOW_NONLOCAL:-0}" = "1" ] \
  || die "refusing to bind the unauthenticated API to $BIND (set LUMEN_API_ALLOW_NONLOCAL=1 to override)"

start_postgres() {
  if "$PGBIN/pg_isready" -q -h 127.0.0.1 -p 5432; then echo "postgres: running"; return; fi
  su postgres -c "$PGBIN/pg_ctl -D $PGDATA -l $ROOT/logs/postgres.log -w start" >/dev/null
  echo "postgres: started"
}

start_ollama() {
  if curl -sf -m 2 http://127.0.0.1:11434/api/version >/dev/null; then echo "ollama: running"; return; fi
  OLLAMA_MODELS=$ROOT/ollama OLLAMA_HOST=127.0.0.1:11434 nohup ollama serve >> "$ROOT/logs/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do curl -sf -m 2 http://127.0.0.1:11434/api/version >/dev/null && { echo "ollama: started"; return; }; sleep 1; done
  die "ollama did not start (see $ROOT/logs/ollama.log)"
}

start_api() {
  if ! curl -sf -m 2 "$API/health" >/dev/null; then
    (cd "$REPO" && nohup "$ROOT/venv/bin/uvicorn" src.api.app:app --host "$BIND" --port 8000 --workers 1 \
        >> "$ROOT/logs/api.log" 2>&1 & echo $! > "$ROOT/api.pid")
    for _ in $(seq 1 60); do curl -sf -m 2 "$API/health" >/dev/null && break; sleep 1; done
  fi
  echo "health: $(curl -s -m 5 -w ' HTTP %{http_code}' "$API/health")"
  for _ in $(seq 1 30); do
    code=$(curl -s -m 10 -o "$ROOT/logs/ready.json" -w '%{http_code}' "$API/ready" || true)
    [ "$code" = "200" ] && break; sleep 2
  done
  echo "ready:  $(cat "$ROOT/logs/ready.json") HTTP $code"
  [ "$code" = "200" ] || die "API is not ready"
}

stop_all() {
  pkill -f "uvicorn src.api.app:app" && echo "api: stopped" || echo "api: not running"
  pkill -f "ollama serve" && echo "ollama: stopped" || echo "ollama: not running"
  if "$PGBIN/pg_isready" -q -h 127.0.0.1 -p 5432; then
    su postgres -c "$PGBIN/pg_ctl -D $PGDATA -m fast -w stop" >/dev/null && echo "postgres: stopped"
  else echo "postgres: not running"; fi
  echo "data, models, qwen3:8b and the venv remain under $ROOT"
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
