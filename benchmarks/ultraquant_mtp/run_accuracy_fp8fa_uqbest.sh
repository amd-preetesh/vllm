#!/bin/bash
# GSM8K + GPQA-Diamond accuracy for one of the Pareto arms:
#   ARM=dspark_fp8_fa | dflash_fp8_fa | dspark_uq_opt | dflash_uq_swa
#       | dspark_uq_opt_adaptive | dspark_uq_opt_adaptive_v2 | dspark_uq_opt_adaptive_v3
#       | dspark_fp8_adaptive
#
# Matches AgentX serve pairing (attn / draft / KV). Seed=7, conc=32.
# Skips a bench if summary.json already exists.
trap '' HUP
set -uo pipefail

ARM="${ARM:?need ARM=dspark_fp8_fa|dflash_fp8_fa|dspark_uq_opt|dflash_uq_swa|dspark_uq_opt_adaptive|dspark_uq_opt_adaptive_v2|dspark_uq_opt_adaptive_v3|dspark_fp8_adaptive}"
ROOT=/workspace/vllm
EVAL_HARNESS=/workspace/eval-harness
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
GSM_CONFIG=$EVAL_HARNESS/configs/benchmarks/gsm8k_ffdsl.json
GPQA_CONFIG=$EVAL_HARNESS/configs/benchmarks/gpqa_diamond_qwen38_card.json
PORT="${PORT:-8850}"
CONC="${CONC:-32}"
# Cap util to what's actually free on the visible device (vLLM checks free, not total).
UTIL_CAP="${UTIL_CAP:-0.90}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP device}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_USE_V2_MODEL_RUNNER=1

case "$HIP_VISIBLE_DEVICES" in
  [0-7]) ;;
  *) echo "$(date -Is) FAIL bad HIP=$HIP_VISIBLE_DEVICES"; exit 1 ;;
esac

UTIL=$(
  CAP="$UTIL_CAP" /workspace/.venv-uq/bin/python - <<'PY'
import os
import torch
cap = float(os.environ["CAP"])
free, total = torch.cuda.mem_get_info(0)
# leave ~8 GiB headroom
free_frac = max(0.0, (free - 8 * (1024**3)) / total)
print(f"{min(cap, free_frac):.4f}")
PY
)
# Refuse to start if util would be too low to load the model.
if python3 -c 'import sys; sys.exit(0 if float("'"$UTIL"'") >= 0.55 else 1)'; then
  :
else
  echo "$(date -Is) FAIL arm=$ARM hip=$HIP_VISIBLE_DEVICES util=$UTIL too low (need emptier GPU)"
  exit 2
fi

case "$ARM" in
  dspark_fp8_fa)
    OUT=$ROOT/out/dspark_qwen38_fp8_fair_fa
    DRAFT=/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/fp8_fair/configs/fp8_dspark_fair_fa.json
    TGT_ATTN=ROCM_AITER_FA
    KV=fp8
    RETENTION=49920
    MNBT=2048
    BLOCK=64
    ;;
  dspark_fp8_adaptive)
    # Fair FP8 FA + enable_adaptive_verification (post #53929).
    OUT=$ROOT/out/dspark_qwen38_fp8_fair_fa_adaptive
    DRAFT=/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/fp8_fair/configs/fp8_dspark_fair_fa_adaptive.json
    TGT_ATTN=ROCM_AITER_FA
    KV=fp8
    RETENTION=49920
    MNBT=2048
    BLOCK=64
    ;;
  dflash_fp8_fa)
    OUT=$ROOT/out/dflash_qwen38_fp8_fair_fa
    DRAFT=/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/fp8_fair/configs/fp8_dflash2_fair_fa.json
    TGT_ATTN=ROCM_AITER_FA
    KV=fp8
    RETENTION=49920
    MNBT=2048
    BLOCK=64
    ;;
  dspark_uq_opt)
    OUT=$ROOT/out/dspark_qwen38_uq_opt
    DRAFT=/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark_dense.json
    TGT_ATTN=ULTRAQUANT_SPEC_COMPAT
    KV=ultraquant_4bit
    RETENTION=50176
    MNBT=2048
    BLOCK=64
    ;;
  dflash_uq_swa)
    OUT=$ROOT/out/dflash_qwen38_uq_dflash_swa
    DRAFT=/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38_uq/configs/uq_dflash2_swa.json
    TGT_ATTN=ULTRAQUANT_SPEC_COMPAT
    KV=ultraquant_4bit
    RETENTION=50176
    MNBT=2048
    BLOCK=64
    ;;
  dspark_uq_opt_adaptive)
    OUT=$ROOT/out/dspark_qwen38_uq_opt_adaptive
    DRAFT=/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark_dense_adaptive.json
    TGT_ATTN=ULTRAQUANT_SPEC_COMPAT
    KV=ultraquant_4bit
    RETENTION=50176
    MNBT=2048
    BLOCK=64
    ;;
  dspark_uq_opt_adaptive_v2)
    # Fixed adaptive accept path (floor + ones init); same serve config as above.
    OUT=$ROOT/out/dspark_qwen38_uq_opt_adaptive_v2
    DRAFT=/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark_dense_adaptive.json
    TGT_ATTN=ULTRAQUANT_SPEC_COMPAT
    KV=ultraquant_4bit
    RETENTION=50176
    MNBT=2048
    BLOCK=64
    ;;
  dspark_uq_opt_adaptive_v3)
    # Post PR #53929 / 3462da7 port (honest varlen capture + hash). Smoke 72% accept.
    OUT=$ROOT/out/dspark_qwen38_uq_opt_adaptive_v3
    DRAFT=/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark_dense_adaptive.json
    TGT_ATTN=ULTRAQUANT_SPEC_COMPAT
    KV=ultraquant_4bit
    RETENTION=50176
    MNBT=2048
    BLOCK=64
    ;;
  *) echo "$(date -Is) FAIL bad ARM=$ARM"; exit 1 ;;
esac

ACC=$OUT/accuracy
mkdir -p "$ACC" /workspace/logs
LOG=/workspace/logs/accuracy-${ARM}.log
exec >>"$LOG" 2>&1
cd "$ROOT"
log() { echo "$(date -Is) $*"; }

make_spec() {
  local max_len="$1"
  DRAFT="$DRAFT" MAX_LEN="$max_len" SPEC_FILE="$SPEC_FILE" python - <<'PY'
import json, os
cfg = json.load(open(os.environ["SPEC_FILE"]))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = int(os.environ["MAX_LEN"])
print(json.dumps(cfg))
PY
}

SERVER_PID=
cleanup_server() {
  [ -n "${SERVER_PID:-}" ] || return 0
  local pid="$SERVER_PID" c tree
  SERVER_PID=
  tree="$pid"
  for c in $(pgrep -P "$pid" 2>/dev/null || true); do
    tree="$tree $c $(pgrep -P "$c" 2>/dev/null || true)"
  done
  # shellcheck disable=SC2086
  kill $tree 2>/dev/null || true
  sleep 12
  # shellcheck disable=SC2086
  kill -9 $tree 2>/dev/null || true
}
trap cleanup_server EXIT

start_server() {
  local out="$1" max_len="$2"
  cleanup_server
  mkdir -p "$out"
  local spec
  spec=$(make_spec "$max_len")
  log "SERVE arm=$ARM hip=$HIP_VISIBLE_DEVICES port=$PORT max_len=$max_len util=$UTIL attn=$TGT_ATTN kv=$KV ret=$RETENTION"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend "$TGT_ATTN" --kv-cache-dtype "$KV" \
    --block-size "$BLOCK" \
    --max-model-len "$max_len" --gpu-memory-utilization "$UTIL" \
    --enable-prefix-caching --no-enable-log-requests \
    --max-num-seqs "$CONC" --max-num-batched-tokens "$MNBT" \
    --prefix-cache-retention-interval "$RETENTION" \
    --language-model-only --reasoning-parser qwen3 --trust-remote-code \
    --speculative-config "$spec" \
    >"$out/server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0 i
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL server exited"
      tail -n 60 "$out/server.log" || true
      cleanup_server
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL server not ready"
    tail -n 60 "$out/server.log" || true
    cleanup_server
    return 1
  fi
  log "READY arm=$ARM max_len=$max_len"
}

run_gsm() {
  local out=$ACC/gsm8k_seed7
  if [ -f "$out/summary.json" ]; then
    log "SKIP gsm8k (summary exists)"
    return 0
  fi
  mkdir -p "$out"
  start_server "$out" 4096 || return 1
  log "EVAL gsm8k seed=7"
  python "$EVAL_HARNESS/drivers/gsm8k_native.py" \
    --port "$PORT" --model "$MODEL" --max_model_len 4096 \
    --bench_config "$GSM_CONFIG" --out_dir "$out" --seed 7 \
    --concurrency_override "$CONC" >"$out/eval.log" 2>&1 || log "WARN gsm8k"
  cleanup_server
  if [ -f "$out/summary.json" ]; then log "END gsm8k"; else log "FAIL gsm8k"; return 1; fi
}

run_gpqa() {
  local out=$ACC/gpqa_seed7
  if [ -f "$out/summary.json" ]; then
    log "SKIP gpqa (summary exists)"
    return 0
  fi
  mkdir -p "$out"
  start_server "$out" 262144 || return 1
  log "EVAL gpqa seed=7"
  python "$EVAL_HARNESS/drivers/gpqa_native.py" \
    --port "$PORT" --model "$MODEL" --max_model_len 262144 \
    --bench_config "$GPQA_CONFIG" --out_dir "$out" --seed 7 \
    --concurrency_override "$CONC" >"$out/eval.log" 2>&1 || log "WARN gpqa"
  cleanup_server
  if [ -f "$out/summary.json" ]; then log "END gpqa"; else log "FAIL gpqa"; return 1; fi
}

log "accuracy start arm=$ARM pid=$$ hip=$HIP_VISIBLE_DEVICES port=$PORT"
if [ ! -f "$DRAFT/config.json" ]; then
  log "FAIL missing draft $DRAFT"
  exit 1
fi
run_gsm || true
run_gpqa || true
cleanup_server
log "accuracy done arm=$ARM"
