#!/bin/bash
# Download z-lab/Qwen3.8-27B-DFlash2 into the local HF models tree.
# New-file-only helper. Safe to re-run (snapshot_download resumes).
set -euo pipefail
DEST="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
REPO="${DFLASH_DRAFT_REPO:-z-lab/Qwen3.8-27B-DFlash2}"
REV="${DFLASH_DRAFT_REV:-50307d4c4cde6860d4eee73e2547cd786fe8e8a4}"
LOG="${DFLASH_FETCH_LOG:-/workspace/logs/dflash-draft-fetch.log}"
DONE="${DFLASH_FETCH_DONE:-/workspace/logs/dflash-draft-fetch.done}"
mkdir -p "$(dirname "$DEST")" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "$(date -Is) fetch start repo=$REPO rev=$REV dest=$DEST"
if [ -f "$DEST/config.json" ]; then
  echo "$(date -Is) already present: $DEST"
  echo "$DEST" >"$DONE"
  exit 0
fi
export DFLASH_DRAFT_DIR="$DEST" DFLASH_DRAFT_REPO="$REPO" DFLASH_DRAFT_REV="$REV" DFLASH_FETCH_DONE="$DONE"
python - <<'PY'
from huggingface_hub import snapshot_download
import os
dest = os.environ["DFLASH_DRAFT_DIR"]
repo = os.environ["DFLASH_DRAFT_REPO"]
rev = os.environ["DFLASH_DRAFT_REV"]
path = snapshot_download(
    repo_id=repo, revision=rev, local_dir=dest, local_dir_use_symlinks=False
)
print("downloaded", path, "rev", rev, flush=True)
open(os.environ["DFLASH_FETCH_DONE"], "w").write(path + "\n" + rev + "\n")
PY
echo "$(date -Is) fetch done"
