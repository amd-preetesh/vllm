#!/bin/bash
# E5: accept-vs-length probe for DSpark/DFlash UQ (and FP8 peers).
# Writes out/uq_accept_probe/<tag>/ — does not overwrite AgentX cells.
#
#   HIP_VISIBLE_DEVICES=0 PORT=8453 TAG=dspark_uq \
#     bash benchmarks/ultraquant_mtp/run_uq_accept_probe.sh
#
# Flip / restart: TAG=dflash_uq RUN=1|2 writes out/.../dflash_uq_r1|r2
# FORCE=1 re-runs even if probe.txt exists.
# Optional: VLLM_DFLASH_DEBUG_NULL_BLOCK=1 VLLM_DFLASH_NULL_BLOCK_ZERO=1
trap '' HUP
mkdir -p /workspace/logs
TAG="${TAG:?need TAG=dspark_uq|dflash_uq|dspark_fp8|dflash_fp8}"
RUN="${RUN:-}"
OUT_TAG="$TAG"
if [ -n "$RUN" ]; then
  OUT_TAG="${TAG}_r${RUN}"
fi
exec >>"/workspace/logs/uq-accept-probe-${OUT_TAG}.log" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DFLASH_DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
DSPARK_DRAFT="${DSPARK_DRAFT_DIR:-/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark}"
PROBE=$ROOT/benchmarks/ultraquant_mtp/docs/probe_accept_vs_length.py
CTX="${PROBE_CTX:-131072,196608,229376}"
GEN="${PROBE_GEN:-256,1024,2048}"
FORCE="${FORCE:-0}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP device}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_USE_V2_MODEL_RUNNER=1

PORT="${PORT:?need PORT}"
OUT=$ROOT/out/uq_accept_probe/$OUT_TAG
mkdir -p "$OUT"
log() { echo "$(date -Is) $*"; }

ATTN=ROCM_AITER_UNIFIED_ATTN
KV=fp8
BLOCK=
UQ=0
case "$TAG" in
  dspark_uq)
    SPEC=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark.json
    DRAFT=$DSPARK_DRAFT
    ATTN=ULTRAQUANT_SPEC_COMPAT
    KV=ultraquant_4bit
    BLOCK=64
    UQ=1
    ;;
  dflash_uq)
    SPEC=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38_uq/configs/uq_dflash2.json
    DRAFT=$DFLASH_DRAFT
    ATTN=ULTRAQUANT_SPEC_COMPAT
    KV=ultraquant_4bit
    BLOCK=64
    UQ=1
    ;;
  dspark_fp8)
    SPEC=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38/configs/fp8_dspark.json
    DRAFT=$DSPARK_DRAFT
    ;;
  dflash_fp8)
    SPEC=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2.json
    DRAFT=$DFLASH_DRAFT
    ;;
  *) log "FAIL TAG=$TAG"; exit 1 ;;
esac

if [ "$UQ" = 1 ]; then
  export VLLM_ULTRAQUANT_FUSED_MTP=1 VLLM_ULTRAQUANT_PDV_VERIFY=1 VLLM_ULTRAQUANT_PAIR_WAVES=2
  export VLLM_ULTRAQUANT_SHARED_KV=0 VLLM_ULTRAQUANT_V3=0 VLLM_ULTRAQUANT_ROWPACK_VERIFY=0 VLLM_ULTRAQUANT_PAIR_VERIFY=0
  unset VLLM_ULTRAQUANT_PDV_EXPERIMENT
fi

SPEC_JSON=$(SPEC="$SPEC" DRAFT="$DRAFT" python - <<'PY'
import json, os
cfg = json.load(open(os.environ["SPEC"]))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = 262144
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

if [ "$FORCE" != 1 ] && [ -f "$OUT/probe.txt" ]; then
  log "SKIP already have $OUT/probe.txt"
  exit 0
fi

log "PROBE serve $OUT_TAG hip=$HIP_VISIBLE_DEVICES port=$PORT null_dbg=${VLLM_DFLASH_DEBUG_NULL_BLOCK:-} null_zero=${VLLM_DFLASH_NULL_BLOCK_ZERO:-}"
serve_args=(
  serve "$MODEL" --port "$PORT" --tensor-parallel-size 1
  --attention-backend "$ATTN" --kv-cache-dtype "$KV"
  --max-model-len 262144 --gpu-memory-utilization 0.90
  --enable-prefix-caching --no-enable-log-requests
  --max-num-seqs 1 --max-num-batched-tokens 8192
  --language-model-only --reasoning-parser qwen3 --trust-remote-code
  --speculative-config "$SPEC_JSON"
)
[ -n "$BLOCK" ] && serve_args+=(--block-size "$BLOCK")
vllm "${serve_args[@]}" >"$OUT/server.log" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B" && break
  kill -0 "$SERVER_PID" 2>/dev/null || { log "FAIL server"; tail -n 60 "$OUT/server.log"; exit 1; }
  sleep 5
done

log "PROBE run ctx=$CTX gen=$GEN"
python "$PROBE" --port "$PORT" --model "$MODEL" \
  --ctx "$CTX" --gen "$GEN" \
  >"$OUT/probe.txt" 2>&1 || log "WARN probe rc=$?"
curl -s "http://127.0.0.1:$PORT/metrics" \
  | grep -E 'spec_decode|prefix_cache' >"$OUT/metrics_end.txt" || true
# Prefer jsonl copy when probe printed JSON lines
grep -E '^\{' "$OUT/probe.txt" >"$OUT/probe.jsonl" 2>/dev/null || true
log "PROBE done $OUT_TAG"
cleanup
