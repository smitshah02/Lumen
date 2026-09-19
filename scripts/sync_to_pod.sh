#!/usr/bin/env bash
# Push ONLY the allowlisted application code to the RunPod Pod — the same set the
# Docker image uses. Never sends data/, models/, backups, .env, API key files,
# results/, caches or the MIMIC-derived reranker training data.
#
#   scripts/sync_to_pod.sh root@<pod-ip> <ssh-port> [path/to/ssh-key]
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=${1:?usage: sync_to_pod.sh user@host port [ssh-key]}
PORT=${2:?usage: sync_to_pod.sh user@host port [ssh-key]}
KEY=${3:-}
DEST=/workspace/lumen/repo
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new)
[ -n "$KEY" ] && SSH+=(-i "$KEY")

FILES=(requirements.txt src
       scripts/bootstrap_pod.sh scripts/start_cloud_demo.sh scripts/cloud_eval.py
       scripts/load_synthetic_demo.py scripts/fetch_models.py scripts/demo_smoke_test.py)

rev="$(git rev-parse HEAD) dirty=$([ -n "$(git status --porcelain -- "${FILES[@]}")" ] && echo yes || echo no)"
"${SSH[@]}" "$TARGET" "mkdir -p $DEST"
rsync -azR --delete --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' --exclude 'src/reranker/' \
  -e "${SSH[*]}" "${FILES[@]}" "$TARGET:$DEST/"
printf '%s\n' "$rev" | "${SSH[@]}" "$TARGET" "cat > $DEST/REVISION"
echo "synced to $TARGET:$DEST ($rev)"
