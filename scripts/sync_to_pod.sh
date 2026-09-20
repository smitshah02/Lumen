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

# Source provenance from the LOCAL repo (.git is never synced). Written to a temp
# file, streamed to the Pod as .deployment_source.json, then deleted locally.
# Contains only: commit SHA, branch, dirty flag, UTC timestamp — no paths, users, hosts or keys.
prov=$(mktemp)
trap 'rm -f "$prov"' EXIT
python3 - "$(git rev-parse HEAD)" "$(git branch --show-current)" "$([ -n "$(git status --porcelain)" ] && echo 1)" > "$prov" <<'PY'
import json, sys
from datetime import datetime, timezone
sha, branch, porcelain = sys.argv[1:4]
print(json.dumps({"git_sha": sha, "branch": branch or None, "dirty_worktree": bool(porcelain),
                  "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "provenance_source": "sync_metadata"}, indent=2))
PY
"${SSH[@]}" "$TARGET" "mkdir -p $DEST"
# /workspace on RunPod can be an object-backed FUSE mount (geesefs): no chown/chmod,
# no temp-file rename. So no -a (it implies -pgoD): copy content + file mtimes only,
# never owner/group/perms or directory times, and write files in place.
rsync -rtzR --omit-dir-times --no-perms --no-owner --no-group --inplace --delete --no-times \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' --exclude 'src/reranker/' \
  -e "${SSH[*]}" "${FILES[@]}" "$TARGET:$DEST/"
"${SSH[@]}" "$TARGET" "cat > $DEST/.deployment_source.json" < "$prov"
echo "synced to $TARGET:$DEST"
cat "$prov"
