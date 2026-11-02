#!/bin/bash
# E4 smoke: UQ + ROCM_AITER_FA_UQ_DENSE → expect draft block ~784, LCM 3136.
#
#   HIP_VISIBLE_DEVICES=0 PORT=8452 MODE=dspark_uq_dense \
#     bash benchmarks/ultraquant_mtp/run_uq_dense_geometry_smoke.sh
trap '' HUP
mkdir -p /workspace/logs
MODE="${MODE:?need MODE=dspark_uq_dense|dflash_uq_dense}"
exec >>"/workspace/logs/uq-dense-geometry-${MODE}.log" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DFLASH_DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
DSPARK_DRAFT="${DSPARK_DRAFT_DIR:-/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP device}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ULTRAQUANT_FUSED_MTP=1
export VLLM_ULTRAQUANT_PDV_VERIFY=1
export VLLM_ULTRAQUANT_PAIR_WAVES=2

PORT="${PORT:?need PORT}"
UTIL="${UTIL:-0.70}"
MAX_LEN="${MAX_LEN:-8192}"
OUT="$ROOT/out/p0_kv_geometry/${MODE}"
mkdir -p "$OUT"
log() { echo "$(date -Is) $*"; }

case "$MODE" in
  dspark_uq_dense)
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark_dense.json
    DRAFT="$DSPARK_DRAFT"
    EXPECT_LCM=3136
    ;;
  dflash_uq_dense)
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38_uq/configs/uq_dflash2_dense.json
    DRAFT="$DFLASH_DRAFT"
    EXPECT_LCM=3136
    ;;
  *) log "FAIL bad MODE=$MODE"; exit 1 ;;
esac

SPEC=$(DRAFT="$DRAFT" MAX_LEN="$MAX_LEN" SPEC_FILE="$SPEC_FILE" python - <<'PY'
import json, os
cfg = json.load(open(os.environ["SPEC_FILE"]))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = int(os.environ["MAX_LEN"])
print(json.dumps(cfg))
PY
)

SERVER_PID=
cleanup() {
  [ -n "${SERVER_PID:-}" ] || return 0
  local pid="$SERVER_PID" c tree
  SERVER_PID=
  tree="$pid"
  for c in $(pgrep -P "$pid" 2>/dev/null || true); do
    tree="$tree $c $(pgrep -P "$c" 2>/dev/null || true)"
  done
  # shellcheck disable=SC2086
  kill $tree 2>/dev/null || true
  sleep 10
  # shellcheck disable=SC2086
  kill -9 $tree 2>/dev/null || true
}
trap cleanup EXIT

log "UQ dense geometry smoke MODE=$MODE hip=$HIP_VISIBLE_DEVICES port=$PORT expect_lcm=$EXPECT_LCM"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend ULTRAQUANT_SPEC_COMPAT --kv-cache-dtype ultraquant_4bit \
  --block-size 64 \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$UTIL" \
  --enable-prefix-caching --no-enable-log-requests \
  --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  --speculative-config "$SPEC" \
  >"$OUT/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "FAIL server exited"
    tail -n 100 "$OUT/server.log" || true
    exit 1
  fi
  sleep 5
done
if [ "$ready" != 1 ]; then
  log "FAIL not ready"
  tail -n 100 "$OUT/server.log" || true
  exit 1
fi

# Extract LCM
python - <<PY
import json, re, sys
from pathlib import Path
log = Path("$OUT/server.log").read_text(errors="replace")
m = re.search(r"lcm block sizes[^\d]*(\d+)", log, re.I)
lcm = int(m.group(1)) if m else None
# also try kv_cache_config style
if lcm is None:
    m = re.search(r"kv_lcm_block_sizes[\"']?\s*[:=]\s*(\d+)", log)
    lcm = int(m.group(1)) if m else None
draft_blk = None
for pat in [r"draft[^\n]{0,40}block[^\n]{0,20}(\d+)", r"block_size[= ]+(\d+)"]:
    mm = re.findall(pat, log, re.I)
    if mm:
        draft_blk = int(mm[-1])
geo = {"mode": "$MODE", "kv_lcm_block_sizes": lcm, "expect_lcm": $EXPECT_LCM,
       "draft_block_guess": draft_blk, "pass": lcm == $EXPECT_LCM}
Path("$OUT/geometry.json").write_text(json.dumps(geo, indent=2) + "\n")
print(json.dumps(geo))
sys.exit(0 if geo["pass"] else 2)
PY
rc=$?
if [ "$rc" = 0 ]; then
  log "PASS $MODE lcm=$EXPECT_LCM"
else
  log "FAIL $MODE geometry (see $OUT/geometry.json)"
  grep -E "lcm|block size|KV cache|Error|ValueError" "$OUT/server.log" | tail -n 40 || true
fi
cleanup
exit "$rc"
