#!/usr/bin/env bash
# Lumen SYNTHETIC demo — RunPod GPU Pod bootstrap (demo plane ONLY).
#
# Filesystem layout (RunPod /workspace is a geesefs FUSE Global Volume: writable,
# but no chmod/chown/exec bits/atomic renames — so nothing executable lives there):
#
#   /workspace/lumen/                 persistent, ordinary files only
#       repo/      source (pushed by scripts/sync_to_pod.sh)
#       logs/      bootstrap log + snapshots of service logs
#       results/   cloud_run/ evaluation artifacts
#   /root/lumen-runtime/              Pod-local (POSIX), rebuilt after Pod recreation
#       venv/      python3 -m venv --system-site-packages (reuses the template's CUDA torch)
#       models/    MedCPT + BGE weights (LUMEN_MODELS_DIR)
#       ollama/    qwen3:8b (OLLAMA_MODELS)
#       cache/     HF_HOME
#       logs/      live service logs;  lumen.env (0600, DB password)
#   /var/lib/postgresql/16/main       the package's default cluster (lumen_demo only)
#
# Idempotent: re-running skips everything already in place. After a Pod is
# recreated, the runtime tree and database are rebuilt from synthetic sources.
#
#   LUMEN_DATA_PLANE=demo bash /workspace/lumen/repo/scripts/bootstrap_pod.sh
set -euo pipefail

PERSIST_ROOT=/workspace/lumen
REPO_ROOT=$PERSIST_ROOT/repo
LOG_ROOT=$PERSIST_ROOT/logs
RESULTS_ROOT=$PERSIST_ROOT/results
RUNTIME_ROOT=/root/lumen-runtime
VENV_ROOT=$RUNTIME_ROOT/venv
MODELS_ROOT=$RUNTIME_ROOT/models
HF_HOME=$RUNTIME_ROOT/cache/huggingface
OLLAMA_MODELS=$RUNTIME_ROOT/ollama
ENV_FILE=$RUNTIME_ROOT/lumen.env
PGVER=16
PG_CLUSTER="$PGVER main"
MODEL=qwen3:8b
MIN_FREE_GB=15

say() { echo; echo "==> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# --- 0. data-plane guard ---------------------------------------------------------
[ "${LUMEN_DATA_PLANE:-demo}" = "demo" ] || die "refusing: LUMEN_DATA_PLANE=${LUMEN_DATA_PLANE}. This Pod serves the synthetic demo plane only."
export LUMEN_DATA_PLANE=demo

# --- 1. platform, GPU, directories, disk ------------------------------------------
say "platform"
[ "$(uname -s)/$(uname -m)" = "Linux/x86_64" ] || die "expected Linux/x86_64, got $(uname -s)/$(uname -m)"
[ "$(id -u)" = "0" ] || die "run as root (RunPod default)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: the Pod has no NVIDIA GPU access"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || die "nvidia-smi failed"

mkdir -p "$LOG_ROOT" "$RESULTS_ROOT"                       # persistent volume: no chmod/chown here
mkdir -p "$MODELS_ROOT" "$OLLAMA_MODELS" "$RUNTIME_ROOT/cache" "$HF_HOME" "$RUNTIME_ROOT/logs"
# Leftovers of the earlier layout that put runtime state on the Global Volume.
for stale in uv venv models ollama postgres cache lumen.env api.pid; do
  if [ -e "$PERSIST_ROOT/$stale" ]; then rm -rf "${PERSIST_ROOT:?}/$stale"; echo "removed stale $PERSIST_ROOT/$stale"; fi
done

heavy_pending=0
[ -f "$VENV_ROOT/.req_hash" ] || heavy_pending=1
[ -f "$MODELS_ROOT/bge-reranker/model.safetensors" ] || heavy_pending=1
[ -d "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/qwen3" ] || heavy_pending=1
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "free on /: ${free_gb}G (heavy installs pending: $heavy_pending)"
if [ "$heavy_pending" = "1" ] && [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  die "only ${free_gb}G free on /; need >= ${MIN_FREE_GB}G for packages, model weights and qwen3:8b"
fi

# --- 2. code (allowlisted; never data) ------------------------------------------------
say "code"
if [ -n "${LUMEN_REPO_URL:-}" ]; then
  apt-get update -qq && apt-get install -y -qq git >/dev/null
  if [ ! -d "$REPO_ROOT/.git" ]; then
    git clone -q --filter=blob:none --no-checkout "$LUMEN_REPO_URL" "$REPO_ROOT"
    git -C "$REPO_ROOT" sparse-checkout set --no-cone '/requirements.txt' '/src/' '!/src/reranker/' '/scripts/'
    git -C "$REPO_ROOT" checkout -q
  else
    git -C "$REPO_ROOT" pull -q --ff-only
  fi
fi
[ -f "$REPO_ROOT/src/api/app.py" ] || die "no code at $REPO_ROOT — run scripts/sync_to_pod.sh from the laptop, or set LUMEN_REPO_URL"
for p in data backups models .env API_keys.docx API_keys.pages results src/reranker/reranker_training_data; do
  [ ! -e "$REPO_ROOT/$p" ] || die "refusing: $REPO_ROOT/$p must never be on the Pod"
done
if find "$REPO_ROOT" \( -name 'pooled*.json' -o -name 'tune_*.json' -o -name '*.dump' -o -name '*.csv.gz' -o -iname '*mimic*.csv*' \) | grep -q .; then
  die "refusing: research artifacts found under $REPO_ROOT"
fi
# Volumes reused from the old research "burst day" bootstrap may hold MIMIC data.
for p in /workspace/pgdata /workspace/Lumen /root/Lumen; do
  [ ! -e "$p" ] || die "refusing: $p exists (research-era path). Use a fresh volume for the synthetic demo."
done
cat "$REPO_ROOT/REVISION" 2>/dev/null || true

# --- 3. system packages (Pod-local) ---------------------------------------------------
say "system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates gnupg lsb-release build-essential git openssl zstd >/dev/null

say "postgres $PGVER"
if [ ! -x "/usr/lib/postgresql/$PGVER/bin/postgres" ]; then
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
# No systemd in the container: the installer's "systemd is not running" warning is
# expected; start_cloud_demo.sh runs `ollama serve` itself.
command -v ollama >/dev/null 2>&1 || curl -fsSL https://ollama.com/install.sh | sh

# --- 4. python: the template's CUDA torch + a venv on the Pod-local disk -------------
say "python"
command -v python3 >/dev/null || die "python3 not found (use the RunPod PyTorch template)"
python3 --version
python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" \
  || die "Python >= 3.11 required by requirements.txt (numpy/pandas pins); choose a newer PyTorch template"
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" \
  || die "the template's PyTorch cannot see the GPU"

if [ ! -x "$VENV_ROOT/bin/python" ]; then
  if ! python3 -m venv --system-site-packages "$VENV_ROOT" 2>/dev/null; then
    pyver=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    apt-get install -y -qq "python${pyver}-venv" >/dev/null || apt-get install -y -qq python3-venv >/dev/null
    rm -rf "$VENV_ROOT"
    python3 -m venv --system-site-packages "$VENV_ROOT"
  fi
fi
PY=$VENV_ROOT/bin/python

req_hash=$(sha256sum "$REPO_ROOT/requirements.txt" | cut -c1-16)
if [ "$(cat "$VENV_ROOT/.req_hash" 2>/dev/null)" != "$req_hash" ]; then
  # Keep the template's CUDA torch: drop torch* pins from a /tmp copy of requirements.txt
  # and constrain torch to the installed build so nothing can replace it.
  grep -v -E '^(torch|torchvision|torchaudio)([=<>~! ;]|$)' "$REPO_ROOT/requirements.txt" > /tmp/lumen-requirements-notorch.txt
  echo "torch==$("$PY" -c 'import torch; print(torch.__version__)')" > /tmp/lumen-constraints.txt
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r /tmp/lumen-requirements-notorch.txt -c /tmp/lumen-constraints.txt
  echo "$req_hash" > "$VENV_ROOT/.req_hash"
fi
"$PY" -m pip check || echo "WARNING: pip check reported conflicts (see above; the template's own packages may be listed)"
"$PY" -c "import torch, transformers; ok=torch.cuda.is_available(); print('venv torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok, torch.cuda.get_device_name(0) if ok else '', '| transformers', transformers.__version__); raise SystemExit(0 if ok else 1)" \
  || die "PyTorch in the venv cannot see the GPU"

# --- 5. runtime environment (Pod-local 0600 file; secret never printed) ---------------
say "environment"
if [ ! -f "$ENV_FILE" ]; then
  pw=${LUMEN_DEMO_PG_PASSWORD:-$(openssl rand -hex 24)}
  umask 077
  cat > "$ENV_FILE" <<EOF
LUMEN_DATA_PLANE=demo
LUMEN_DEMO_DATABASE_URL=postgresql://lumen_demo:${pw}@127.0.0.1:5432/lumen_demo
LUMEN_LLM_HOST=http://127.0.0.1:11434
LUMEN_LLM_MAIN=${MODEL}
LUMEN_LLM_FAST=${MODEL}
LUMEN_MODELS_DIR=${MODELS_ROOT}
HF_HOME=${HF_HOME}
OLLAMA_MODELS=${OLLAMA_MODELS}
LUMEN_TRACING=0
LUMEN_API_BIND=127.0.0.1
LUMEN_ROOT=${PERSIST_ROOT}
LUMEN_RUNTIME_ROOT=${RUNTIME_ROOT}
EOF
  umask 022
fi
set -a; . "$ENV_FILE"; set +a
[ "$LUMEN_DATA_PLANE" = "demo" ] || die "$ENV_FILE is not the demo plane"

# --- 6. postgres: package default cluster on the Pod-local disk, lumen_demo only ----
say "database"
pg_lsclusters -h | awk '{print $1" "$2}' | grep -qx "$PG_CLUSTER" \
  || pg_createcluster $PG_CLUSTER -- --auth-local=peer --auth-host=scram-sha-256 >/dev/null
bash "$REPO_ROOT/scripts/start_cloud_demo.sh" postgres
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

# --- 7. ollama + qwen3:8b (stored under OLLAMA_MODELS, Pod-local) -------------------
say "ollama + ${MODEL}"
bash "$REPO_ROOT/scripts/start_cloud_demo.sh" ollama
if OLLAMA_HOST=127.0.0.1:11434 ollama list | awk 'NR>1 {print $1}' | grep -qx "$MODEL"; then
  echo "$MODEL: present"
else
  OLLAMA_HOST=127.0.0.1:11434 ollama pull "$MODEL"
fi

# --- 8. retrieval model weights (pinned revisions, Pod-local) ----------------------
say "MedCPT + BGE weights -> $MODELS_ROOT"
(cd "$REPO_ROOT" && HF_HUB_OFFLINE=0 "$PY" scripts/fetch_models.py)

# --- 9. synthetic data + index (skipped when already complete) ---------------------
say "synthetic corpus"
state=$(cd "$REPO_ROOT" && "$PY" - <<'PY'
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
  (cd "$REPO_ROOT" && "$PY" scripts/load_synthetic_demo.py && "$PY" -m src.retrieval.index_notes)
fi

# --- 10. API + health checks ----------------------------------------------------------
say "api"
bash "$REPO_ROOT/scripts/start_cloud_demo.sh" api

# Snapshot Pod-local service logs onto the persistent volume (plain copies, no perms).
cp "$RUNTIME_ROOT"/logs/*.log "$RUNTIME_ROOT"/logs/*.json "$LOG_ROOT"/ 2>/dev/null || true
echo
df -h /
echo
echo "Bootstrap complete. From the laptop: ssh -N -L 8000:127.0.0.1:8000 <pod-ssh>  then  curl http://127.0.0.1:8000/ready"
