#!/bin/bash
# Fair FP8 KV8 AgentX: same scheduler knobs as UQ opt (ret≈50k + mnbt2048 +
# block64). Matches the four-way FP8 pairing:
#   target ROCM_AITER_FA + fp8 KV, draft TRITON_ATTN + fp8 KV.
# Outs: out/{dspark,dflash}_qwen38_fp8_fair_fa/
# Legacy UNIFIED+FA/SWA outs stay in *_fp8_fair/ (do not overwrite).
#
#   CFG=dspark|dflash HIP_VISIBLE_DEVICES=6 PORT=8470 CONCS="8 16 20 24 30 32" \
#     bash benchmarks/ultraquant_mtp/fp8_fair/run_fp8_fair_agentx.sh
trap '' HUP
mkdir -p /workspace/logs
CFG="${CFG:?need CFG=dspark|dflash}"
exec >>"/workspace/logs/fp8-fair-fa-${CFG}.log" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp/fp8_fair
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DFLASH_DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
DSPARK_DRAFT="${DSPARK_DRAFT_DIR:-/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SA_DIR/.hf-cache}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export VLLM_USE_V2_MODEL_RUNNER=1
# No UQ PDV knobs — pure FP8 fair baseline.
unset VLLM_ULTRAQUANT_FUSED_MTP VLLM_ULTRAQUANT_PDV_VERIFY VLLM_ULTRAQUANT_PDV_EXPERIMENT \
  VLLM_ULTRAQUANT_MIXED_VERIFY VLLM_ULTRAQUANT_CONT_PREFILL 2>/dev/null || true

case "$HIP_VISIBLE_DEVICES" in
  [0-7]) ;;
  *) echo "$(date -Is) FAIL HIP=$HIP_VISIBLE_DEVICES"; exit 1 ;;
esac

PORT="${PORT:?need PORT}"
SEED="${SEED:-42}"
DURATION="${DURATION:-300}"
CONCS=(${CONCS:-8 16 20 24 30 32})
MAX_LEN="${MAX_LEN:-262144}"
UTIL="${UTIL:-0.90}"
BLOCK=64
MNBT=2048
# FP8 DSpark/DFlash LCM is 3328 (not UQ dense 3136). 50176=16×3136 is invalid
# here; 49920=15×3328 is the ~50k checkpoint that passes the multiple check.
RETENTION="${RETENTION:-49920}"

EXP="${EXP:-fair_fa}"
case "$CFG" in
  dspark)
    DRAFT="$DSPARK_DRAFT"
    case "$EXP" in
      fair_fa)
        SPEC_FILE=$HERE/configs/fp8_dspark_fair_fa.json
        OUT_BASE=$ROOT/out/dspark_qwen38_fp8_fair_fa
        ;;
      adaptive)
        # Adaptive verification on the fair FP8 FA stack (post #53929 port).
        SPEC_FILE=$HERE/configs/fp8_dspark_fair_fa_adaptive.json
        OUT_BASE=$ROOT/out/dspark_qwen38_fp8_fair_fa_adaptive
        ;;
      *) echo "$(date -Is) FAIL bad EXP=$EXP for dspark"; exit 1 ;;
    esac
    ;;
  dflash)
    if [ "$EXP" != fair_fa ]; then
      echo "$(date -Is) FAIL EXP=$EXP not supported for dflash (adaptive is dspark-only)"
      exit 1
    fi
    DRAFT="$DFLASH_DRAFT"
    SPEC_FILE=$HERE/configs/fp8_dflash2_fair_fa.json
    OUT_BASE=$ROOT/out/dflash_qwen38_fp8_fair_fa
    ;;
  *) echo "$(date -Is) FAIL bad CFG=$CFG"; exit 1 ;;
esac

cd "$ROOT"
log() { echo "$(date -Is) $*"; }

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

make_spec() {
  DRAFT="$DRAFT" MAX_LEN="$MAX_LEN" CFG="$SPEC_FILE" python - <<'PY'
import json, os
cfg = json.load(open(os.environ["CFG"]))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = int(os.environ["MAX_LEN"])
print(json.dumps(cfg))
PY
}

fit_util_for_serve() {
  local want="$1"
  WANT_UTIL="$want" python - <<'PY'
import os, time, torch
want = float(os.environ["WANT_UTIL"])
time.sleep(3)
for _ in range(8):
    free, total = torch.cuda.mem_get_info(0)
    util = min(want, max(0.40, (free / total) - 0.03))
    if free >= util * total and free / (1024**3) >= 100:
        print(f"{util:.4f}")
        raise SystemExit(0)
    time.sleep(8)
raise SystemExit(1)
PY
}

run_cell() {
  local conc="$1"
  local out="$OUT_BASE/sa/seed${SEED}_c${conc}"
  if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "SKIP $CFG fair c$conc"
    return 0
  fi
  if [ ! -f "$DRAFT/config.json" ]; then
    log "FAIL missing draft $DRAFT"
    return 1
  fi
  cleanup_server
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
  fi
  mkdir -p "$out"
  local cell_util
  cell_util=$(fit_util_for_serve "$UTIL" 2>"$out/fit_util.log") || {
    log "FAIL $CFG c$conc fit_util"
    return 1
  }
  local spec
  spec=$(make_spec)
  # Four-way match: target FA+fp8, draft TRITON+fp8 (not FA draft — Mixed FP8).
  log "SA START $CFG fair_fa c$conc hip=$HIP_VISIBLE_DEVICES port=$PORT mnbt=$MNBT ret=$RETENTION util=$cell_util attn=FA draft=TRITON"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend ROCM_AITER_FA --kv-cache-dtype fp8 \
    --block-size "$BLOCK" \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization "$cell_util" \
    --enable-prefix-caching --no-enable-log-requests \
    --max-num-seqs "$conc" --max-num-batched-tokens "$MNBT" \
    --prefix-cache-retention-interval "$RETENTION" \
    --language-model-only --reasoning-parser qwen3 --trust-remote-code \
    --speculative-config "$spec" >"$out/server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0 i
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL $CFG c$conc server exited"
      tail -n 80 "$out/server.log" || true
      cleanup_server
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL $CFG c$conc not ready"
    tail -n 80 "$out/server.log" || true
    cleanup_server
    return 1
  fi
  log "READY $CFG fair c$conc"

  local extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')
  (cd "$INFERENCEX_DIR" && "$AIPERF" profile --scenario inferencex-agentx-mvp \
    --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
    --model "$MODEL" --concurrency "$conc" --benchmark-duration "$DURATION" \
    --unsafe-override \
    --stats-interval 30 --random-seed "$SEED" --failed-request-threshold 0.10 \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-grace-period 300 \
    --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
    --max-context-length "$MAX_LEN" \
    --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir "$out/aiperf_artifacts" \
    --public-dataset semianalysis_cc_traces_weka_062126_256k \
    "${extra[@]}") >"$out/benchmark.log" 2>&1 &
  local bench=$!
  wait "$bench" || log "WARN aiperf rc=$?"
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions|generation_tokens|gpu_cache)' \
    >"$out/metrics_end.txt" || true
  {
    echo "cfg=$CFG fair_fa ret=$RETENTION mnbt=$MNBT block=$BLOCK"
    echo "spec=$SPEC_FILE attn=ROCM_AITER_FA kv=fp8 draft=TRITON_ATTN draft_kv=fp8"
  } >"$out/cell_meta.txt"
  cleanup_server
  if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "END $CFG fair c$conc"
  else
    log "FAIL $CFG fair c$conc no profile"
    return 1
  fi
}

log "fp8 $EXP start cfg=$CFG hip=$HIP_VISIBLE_DEVICES port=$PORT concs=${CONCS[*]} out=$OUT_BASE"
for conc in "${CONCS[@]}"; do
  run_cell "$conc" || true
done
cleanup_server
log "fp8 $EXP done cfg=$CFG"
