#!/bin/bash
# Download RadixArk/Qwen3.8-27B-DSpark @ pinned revision (TP=1 harness).
set -euo pipefail
DEST="${DSPARK_DRAFT_DIR:-/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark}"
REPO="${DSPARK_DRAFT_REPO:-RadixArk/Qwen3.8-27B-DSpark}"
REV="${DSPARK_DRAFT_REV:-b9a5dbdf03bc999c6c73c426b19c2d9041cea393}"
LOG="${DSPARK_FETCH_LOG:-/workspace/logs/dspark-draft-fetch.log}"
DONE="${DSPARK_FETCH_DONE:-/workspace/logs/dspark-draft-fetch.done}"
mkdir -p "$(dirname "$DEST")" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "$(date -Is) fetch start repo=$REPO rev=$REV dest=$DEST"
if [ -f "$DEST/config.json" ]; then
  echo "$(date -Is) already present: $DEST"
  echo "$DEST" >"$DONE"
  exit 0
fi
export PATH=/workspace/.venv-uq/bin:${PATH:-}
export DSPARK_DRAFT_DIR="$DEST" DSPARK_DRAFT_REPO="$REPO" DSPARK_DRAFT_REV="$REV"
export DSPARK_FETCH_DONE="$DONE"
python - <<'PY'
from huggingface_hub import snapshot_download
import os

dest = os.environ["DSPARK_DRAFT_DIR"]
repo = os.environ["DSPARK_DRAFT_REPO"]
rev = os.environ["DSPARK_DRAFT_REV"]
path = snapshot_download(
    repo_id=repo,
    revision=rev,
    local_dir=dest,
    local_dir_use_symlinks=False,
)
print("downloaded", path, "rev", rev, flush=True)
with open(os.environ["DSPARK_FETCH_DONE"], "w") as f:
    f.write(path + "\n" + rev + "\n")
PY
echo "$(date -Is) fetch done"
