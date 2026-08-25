#!/usr/bin/env bash
# Lumen pod bootstrap — RunPod burst day only. NOT used in local dev.
# Re-run after EVERY pod start: only /workspace survives restarts.
set -euo pipefail

PGVER=16
PGBIN=/usr/lib/postgresql/$PGVER/bin
PGDATA=/workspace/pgdata

echo "==> system packages"
apt-get update -qq
apt-get install -y -qq curl ca-certificates gnupg lsb-release build-essential git tmux

echo "==> postgres $PGVER (must match your local major version)"
if ! command -v $PGBIN/initdb >/dev/null 2>&1; then
  install -d /usr/share/postgresql-common/pgdg
  curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
    https://www.postgresql.org/media/keys/ACCC4CF8.asc
  echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] \
https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
    > /etc/apt/sources.list.d/pgdg.list
  apt-get update -qq
  apt-get install -y -qq postgresql-$PGVER postgresql-server-dev-$PGVER
fi

echo "==> pgvector"
if [ ! -f /usr/lib/postgresql/$PGVER/lib/vector.so ]; then
  rm -rf /tmp/pgvector
  git clone -q --branch v0.8.0 https://github.com/pgvector/pgvector.git /tmp/pgvector
  make -C /tmp/pgvector -s && make -C /tmp/pgvector -s install
fi

echo "==> postgres data dir on the network volume"
if [ ! -d "$PGDATA/base" ]; then
  mkdir -p "$PGDATA"; chown postgres:postgres "$PGDATA"
  su postgres -c "$PGBIN/initdb -D $PGDATA -E UTF8"
  echo "listen_addresses = 'localhost'" >> "$PGDATA/postgresql.conf"
  echo "shared_buffers = 2GB"           >> "$PGDATA/postgresql.conf"
fi
chown -R postgres:postgres "$PGDATA"

echo "==> start postgres"
su postgres -c "$PGBIN/pg_ctl -D $PGDATA -l /workspace/pg.log -w start" || true

echo "==> ollama (models cached on the volume)"
export OLLAMA_MODELS=/workspace/ollama
mkdir -p "$OLLAMA_MODELS"
command -v ollama >/dev/null 2>&1 || curl -fsSL https://ollama.com/install.sh | sh
pkill -f "ollama serve" || true
OLLAMA_MODELS=/workspace/ollama OLLAMA_NUM_PARALLEL=2 \
  nohup ollama serve > /workspace/ollama.log 2>&1 &
sleep 5

echo "==> make Path.home()/'Lumen' resolve to the volume (no code changes needed)"
ln -sfn /workspace/Lumen /root/Lumen

echo
echo "postgres:  $($PGBIN/pg_isready -h localhost && echo up)"
echo "ollama:    $(curl -s localhost:11434/api/tags >/dev/null && echo up)"
echo "gpu:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"