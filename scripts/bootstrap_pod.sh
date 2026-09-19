#!/usr/bin/env bash
# Lumen SYNTHETIC demo — RunPod GPU Pod bootstrap (demo plane ONLY).
#
# Prepares a fresh Pod and brings up Postgres+pgvector (lumen_demo only),
# Ollama (qwen3:8b on the GPU), the Python env (MedCPT/BGE on CUDA) and the
# FastAPI service bound to 127.0.0.1:8000 (reach it through an SSH tunnel).
#
# Re-run after EVERY Pod start: only /workspace survives restarts, so system
# packages are reinstalled, while the venv, model weights, qwen3:8b, the
# database and the index persist and are skipped when already present.
#
# Code arrives one of two ways (never the research data):
#   * pushed from the laptop:  scripts/sync_to_pod.sh <user@host> <port>   (default)
#   * LUMEN_REPO_URL=<git url> — sparse, blob-filtered clone of requirements.txt, src/, scripts/
#
#   bash /workspace/lumen/repo/scripts/bootstrap_pod.sh
set -euo pipefail

ROOT=${LUMEN_ROOT:-/workspace/lumen}
REPO=$ROOT/repo
PGVER=16
PGBIN=/usr/lib/postgresql/$PGVER/bin
PGDATA=$ROOT/postgres
MODEL=qwen3:8b

say() { echo; echo "==> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# --- 0. data-plane guard ---------------------------------------------------------
[ "${LUMEN_DATA_PLANE:-demo}" = "demo" ] || die "refusing: LUMEN_DATA_PLANE=${LUMEN_DATA_PLANE}. This Pod serves the synthetic demo plane only."
export LUMEN_DATA_PLANE=demo

# --- 1. platform, GPU, disk -------------------------------------------------------
say "platform"
[ "$(uname -s)/$(uname -m)" = "Linux/x86_64" ] || die "expected Linux/x86_64, got $(uname -s)/$(uname -m)"
[ "$(id -u)" = "0" ] || die "run as root (RunPod default)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: the Pod has no NVIDIA GPU access"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || die "nvidia-smi failed"
mkdir -p "$ROOT"/{logs,results,models,ollama}
need_gb=$([ -x "$ROOT/venv/bin/python" ] && echo 5 || echo "${LUMEN_MIN_FREE_GB:-30}")
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge "$need_gb" ] || die "only ${free_gb}G free under $ROOT; need >= ${need_gb}G"
echo "free under $ROOT: ${free_gb}G"

# --- 2. code (allowlisted; never data) ------------------------------------------------
say "code"
if [ -n "${LUMEN_REPO_URL:-}" ]; then
  apt-get update -qq && apt-get install -y -qq git >/dev/null
  if [ ! -d "$REPO/.git" ]; then
    git clone -q --filter=blob:none --no-checkout "$LUMEN_REPO_URL" "$REPO"
    git -C "$REPO" sparse-checkout set --no-cone '/requirements.txt' '/src/' '!/src/reranker/' '/scripts/'
    git -C "$REPO" checkout -q
  else
    git -C "$REPO" pull -q --ff-only
  fi
fi
[ -f "$REPO/src/api/app.py" ] || die "no code at $REPO — run scripts/sync_to_pod.sh from the laptop, or set LUMEN_REPO_URL"
for p in data backups models .env API_keys.docx API_keys.pages results src/reranker/reranker_training_data; do
  [ ! -e "$REPO/$p" ] || die "refusing: $REPO/$p must never be on the Pod"
done
if find "$REPO" \( -name 'pooled*.json' -o -name 'tune_*.json' -o -name '*.dump' -o -name '*.csv.gz' -o -iname '*mimic*.csv*' \) | grep -q .; then
  die "refusing: research artifacts found under $REPO"
fi
# Volumes reused from the old research "burst day" bootstrap may hold MIMIC data.
for p in /workspace/pgdata /workspace/Lumen /root/Lumen; do
  [ ! -e "$p" ] || die "refusing: $p exists (research-era path). Use a fresh volume for the synthetic demo."
done
cat "$REPO/REVISION" 2>/dev/null || true

# --- 3. system packages (reinstalled after every restart) ------------------------
say "system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates gnupg lsb-release build-essential git openssl zstd >/dev/null

say "postgres $PGVER"
if [ ! -x "$PGBIN/initdb" ]; then
  install -d /usr/share/postgresql-common/pgdg
  curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc
  echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
    > /etc/apt/sources.list.d/pgdg.list
  apt-get update -qq
  apt-get install -y -qq "postgresql-$PGVER" "postgresql-server-dev-$PGVER" >/dev/null
fi

say "pgvector"
if [ ! -f "/usr/lib/postgresql/$PGVER/lib/vector.so" ]; then
  rm -rf /tmp/pgvector
  git clone -q --branch v0.8.0 https://github.com/pgvector/pgvector.git /tmp/pgvector
  make -C /tmp/pgvector -s && make -C /tmp/pgvector -s install
fi

say "ollama binary"
command -v ollama >/dev/null 2>&1 || curl -fsSL https://ollama.com/install.sh | sh

# --- 4. python env (persists on the volume) -----------------------------------------
say "python env"
export UV_INSTALL_DIR=$ROOT/uv/bin UV_PYTHON_INSTALL_DIR=$ROOT/uv/python UV_CACHE_DIR=$ROOT/uv/cache
UV=$ROOT/uv/bin/uv
[ -x "$UV" ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/uv/bin" INSTALLER_NO_MODIFY_PATH=1 sh
[ -x "$ROOT/venv/bin/python" ] || "$UV" venv -q --python 3.14 "$ROOT/venv"
PY=$ROOT/venv/bin/python
req_hash=$(sha256sum "$REPO/requirements.txt" | cut -c1-16)
if [ "$(cat "$ROOT/venv/.req_hash" 2>/dev/null)" != "$req_hash" ]; then
  # torch 2.11's default PyPI build targets CUDA 13 (driver >= 580); older drivers get cu128.
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  idx=$([ "$drv" -ge 580 ] && echo cu130 || echo cu128)
  echo "driver $drv -> torch index $idx"
  "$UV" pip install -q --python "$PY" --index-url "https://download.pytorch.org/whl/$idx" torch==2.11.0
  "$UV" pip install -q --python "$PY" -r "$REPO/requirements.txt"
  echo "$req_hash" > "$ROOT/venv/.req_hash"
fi
"$PY" -c "import torch; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok, torch.cuda.get_device_name(0) if ok else ''); raise SystemExit(0 if ok else 1)" \
  || die "PyTorch cannot see the GPU"

# --- 5. runtime environment (secret stays in a 0600 file, never printed) -----------
say "environment"
if [ ! -f "$ROOT/lumen.env" ]; then
  pw=${LUMEN_DEMO_PG_PASSWORD:-$(openssl rand -hex 24)}
  umask 077
  cat > "$ROOT/lumen.env" <<EOF
LUMEN_DATA_PLANE=demo
LUMEN_DEMO_DATABASE_URL=postgresql://lumen_demo:${pw}@127.0.0.1:5432/lumen_demo
LUMEN_LLM_HOST=http://127.0.0.1:11434
LUMEN_LLM_MAIN=${MODEL}
LUMEN_LLM_FAST=${MODEL}
LUMEN_MODELS_DIR=${ROOT}/models
HF_HOME=${ROOT}/models/.hf-cache
LUMEN_TRACING=0
LUMEN_API_BIND=127.0.0.1
EOF
  umask 022
fi
set -a; . "$ROOT/lumen.env"; set +a
[ "$LUMEN_DATA_PLANE" = "demo" ] || die "lumen.env is not the demo plane"

# --- 6. postgres data dir + lumen_demo (and nothing else) ----------------------------
say "database"
if [ ! -d "$PGDATA/base" ]; then
  mkdir -p "$PGDATA"; chown postgres:postgres "$PGDATA"
  # peer auth on the local socket (bootstrap admin), passwords on TCP (the app)
  su postgres -c "$PGBIN/initdb -D $PGDATA -E UTF8 --locale=C --auth-local=peer --auth-host=scram-sha-256" >/dev/null
  echo "listen_addresses = 'localhost'" >> "$PGDATA/postgresql.conf"
  echo "shared_buffers = 1GB"           >> "$PGDATA/postgresql.conf"
fi
chown -R postgres:postgres "$PGDATA"
bash "$REPO/scripts/start_cloud_demo.sh" postgres
if su postgres -c "psql -Atc \"SELECT 1 FROM pg_database WHERE datname = 'lumen'\"" | grep -q 1; then
  die "a database named 'lumen' exists on this Pod; the research database must never be here"
fi
pw=$(printf '%s' "$LUMEN_DEMO_DATABASE_URL" | sed -E 's#^postgresql://[^:]+:([^@]+)@.*#\1#')
su postgres -c "psql -q -v ON_ERROR_STOP=1" <<SQL
DO \$\$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'lumen_demo') THEN CREATE ROLE lumen_demo LOGIN; END IF;
END \$\$;
ALTER ROLE lumen_demo PASSWORD '${pw}';
SQL
unset pw
su postgres -c "psql -Atc \"SELECT 1 FROM pg_database WHERE datname = 'lumen_demo'\"" | grep -q 1 \
  || su postgres -c "createdb -O lumen_demo lumen_demo"
su postgres -c "psql -q -d lumen_demo -c 'CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;'"

# --- 7. ollama + qwen3:8b -------------------------------------------------------------
say "ollama + ${MODEL}"
bash "$REPO/scripts/start_cloud_demo.sh" ollama
if OLLAMA_HOST=127.0.0.1:11434 ollama list | awk 'NR>1 {print $1}' | grep -qx "$MODEL"; then
  echo "$MODEL: present"
else
  OLLAMA_HOST=127.0.0.1:11434 ollama pull "$MODEL"
fi

# --- 8. retrieval model weights (pinned revisions, skipped when present) -----------
say "MedCPT + BGE weights"
(cd "$REPO" && HF_HUB_OFFLINE=0 "$PY" scripts/fetch_models.py)

# --- 9. synthetic data + index (skipped when already complete) ---------------------
say "synthetic corpus"
state=$(cd "$REPO" && "$PY" - <<'PY'
import json
from sqlalchemy import text
from src import storage
m = json.load(open("src/demo_data/manifest.json"))
try:
    with storage.engine.connect() as c:
        notes = c.execute(text("SELECT COUNT(*) FROM clinical_notes WHERE phi_entities->>'version' = :v"), {"v": m["version"]}).scalar()
        chunks, emb = c.execute(text("SELECT COUNT(*), COUNT(embedding) FROM note_chunks")).one()
        unindexed = c.execute(text("SELECT COUNT(*) FROM clinical_notes cn WHERE NOT EXISTS (SELECT 1 FROM note_chunks nc WHERE nc.note_id = cn.note_id)")).scalar()
    ok = notes == m["counts"]["synthetic_notes"] and chunks > 0 and emb == chunks and unindexed == 0
    print("complete" if ok else f"incomplete notes={notes} chunks={chunks} embedded={emb} unindexed={unindexed}")
except Exception as e:
    print(f"empty ({type(e).__name__})")
PY
)
echo "$state"
if [ "$state" != "complete" ]; then
  (cd "$REPO" && "$PY" scripts/load_synthetic_demo.py && "$PY" -m src.retrieval.index_notes)
fi

# --- 10. API + health checks ----------------------------------------------------------
say "api"
bash "$REPO/scripts/start_cloud_demo.sh" api
echo
echo "Bootstrap complete. From the laptop: ssh -N -L 8000:127.0.0.1:8000 <pod-ssh>  then  curl http://127.0.0.1:8000/ready"
