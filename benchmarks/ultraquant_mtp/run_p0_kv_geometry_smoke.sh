#!/bin/bash
# P0-1 geometry smoke (MI355X): serve briefly, assert hybrid KV LCM / group sizes.
# Does NOT touch shipping kernels. Opt-in only.
#
#   HIP_VISIBLE_DEVICES=3 PORT=8440 MODE=dspark_fp8_bs64_kv8 \
#     bash benchmarks/ultraquant_mtp/run_p0_kv_geometry_smoke.sh
#
# MODE:
#   dflash_fp8_bs64      — FP8 DFlash + --block-size 64 (draft kv auto)
#   dflash_fp8_bs64_kv8  — same + draft kv_cache_dtype=fp8
#   dspark_fp8_bs64      — FP8 DSpark + --block-size 64 (draft kv auto)
#   dspark_fp8_bs64_kv8  — same + draft kv_cache_dtype=fp8  (primary P0-1 proof)
trap '' HUP
mkdir -p /workspace/logs
MODE="${MODE:?need MODE}"
exec >>"/workspace/logs/p0-kv-geometry-${MODE}.log" 2>&1
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

PORT="${PORT:?need PORT}"
UTIL="${UTIL:-0.90}"
MAX_LEN="${MAX_LEN:-8192}"
OUT="$ROOT/out/p0_kv_geometry/${MODE}"
mkdir -p "$OUT"
log() { echo "$(date -Is) $*"; }

case "$MODE" in
  dflash_fp8_bs64)
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2.json
    DRAFT="$DFLASH_DRAFT"
    BLOCK=64
    EXPECT_LCM=1664
    ;;
  dflash_fp8_bs64_kv8)
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2_kv8.json
    DRAFT="$DFLASH_DRAFT"
    BLOCK=64
    EXPECT_LCM=1664
    ;;
  dspark_fp8_bs64)
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38/configs/fp8_dspark.json
    DRAFT="$DSPARK_DRAFT"
    BLOCK=64
    EXPECT_LCM=1664
    ;;
  dspark_fp8_bs64_kv8)
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38/configs/fp8_dspark_kv8.json
    DRAFT="$DSPARK_DRAFT"
    BLOCK=64
    EXPECT_LCM=1664
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

log "P0 geometry smoke MODE=$MODE hip=$HIP_VISIBLE_DEVICES port=$PORT block=$BLOCK expect_lcm=$EXPECT_LCM"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8 \
  --block-size "$BLOCK" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$UTIL" \
  --enable-prefix-caching --no-enable-log-requests \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  --speculative-config "$SPEC" \
  >"$OUT/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "FAIL $MODE server exited"
    tail -n 80 "$OUT/server.log" || true
    exit 1
  fi
  sleep 5
done
if [ "$ready" != 1 ]; then
  log "FAIL $MODE not ready"
  tail -n 80 "$OUT/server.log" || true
  exit 1
fi
log "READY $MODE"

# Extract geometry lines
OUT_DIR="$OUT" EXPECT_LCM="$EXPECT_LCM" MODE="$MODE" python - <<'PY'
import json, os, pathlib, re
log = pathlib.Path(os.environ["OUT_DIR"], "server.log").read_text(errors="replace")
expect = int(os.environ["EXPECT_LCM"])
mode = os.environ["MODE"]
groups = re.findall(r"kv cache group sizes\s*(\[[^\]]+(?:\s*\[[^\]]+)*)", log)
# Prefer a full capture of the sizes line
m = re.search(r"kv cache group sizes\s*(.+)$", log, re.M)
group_blob = m.group(1).strip() if m else (groups[-1] if groups else None)
lcms = re.findall(r"kv lcm block sizes\s*(\d+)", log)
attn_bs = re.findall(r"Setting attention block size to (\d+)", log)
pad = "padding layers" in log.lower()
out = {
    "mode": mode,
    "expect_lcm": expect,
    "attention_block_size": int(attn_bs[-1]) if attn_bs else None,
    "kv_cache_group_sizes": group_blob,
    "kv_lcm_block_sizes": int(lcms[-1]) if lcms else None,
    "mentions_padding_layers": pad,
}
path = pathlib.Path(os.environ["OUT_DIR"], "geometry.json")
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
ok = out["kv_lcm_block_sizes"] == expect
blob = out["kv_cache_group_sizes"] or ""
has_bad = ("[32]" in blob) or ("3296" in blob)
if ok and not has_bad:
    print("PASS geometry")
    raise SystemExit(0)
print(f"FAIL geometry (want lcm={expect}, no 32/3296 groups)")
raise SystemExit(2)
PY
rc=$?
cleanup
SERVER_PID=
if [ "$rc" = 0 ]; then
  log "PASS $MODE → $OUT/geometry.json"
else
  log "FAIL $MODE → see $OUT/server.log $OUT/geometry.json"
fi
exit "$rc"
