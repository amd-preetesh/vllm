#!/bin/bash
# Stream 2: AgentX concurrency sweep for DFlash2/DSpark × FP8(KV8)/UQ.
# Concs: 8 16 20 24 30 (c32 already measured).
#
# ROLE=a  → dflash_fp8
# ROLE=b  → dspark_fp8
# ROLE=c  → dflash_uq then dspark_uq
# ROLE=p01 → dflash_fp8_p01 + dspark_fp8_p01 (block-size 64, separate outs)
#
# Usage (inside preshah_vllm_ultraquant):
#   HIP_VISIBLE_DEVICES=1 PORT=8410 ROLE=a \
#     bash benchmarks/ultraquant_mtp/run_dflash_dspark_agentx_conc_sweep.sh
trap '' HUP
mkdir -p /workspace/logs
ROLE="${ROLE:?need ROLE=a|b|c|p01|p01_dflash|p01_dspark}"
exec >>"/workspace/logs/dflash-dspark-agentx-conc-${ROLE}.log" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DFLASH_DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
DSPARK_DRAFT="${DSPARK_DRAFT_DIR:-/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP device}"
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

case "$HIP_VISIBLE_DEVICES" in
  [0-7]) ;;
  *) echo "$(date -Is) FAIL refusing HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"; exit 1 ;;
esac

PORT="${PORT:?need PORT}"
SEED="${SEED:-42}"
DURATION="${DURATION:-300}"
CONCS=(${CONCS:-8 16 20 24 30})
MAX_LEN=262144
MNBT="${MNBT:-8192}"
UTIL_FP8="${UTIL_FP8:-0.90}"
UTIL_UQ="${UTIL_UQ:-0.90}"

cd "$ROOT"
log() { echo "$(date -Is) $*"; }

case "$ROLE" in
  a) CONFIGS=(dflash_fp8) ;;
  b) CONFIGS=(dspark_fp8) ;;
  c) CONFIGS=(dflash_uq dspark_uq) ;;
  # P0-1 follow-up matrix: same models/methods, FP8 + --block-size 64 → separate outs
  p01) CONFIGS=(dflash_fp8_p01 dspark_fp8_p01) ;;
  p01_dflash) CONFIGS=(dflash_fp8_p01) ;;
  p01_dspark) CONFIGS=(dspark_fp8_p01) ;;
  *) log "FAIL unknown ROLE=$ROLE"; exit 1 ;;
esac

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
  sleep 15
  # shellcheck disable=SC2086
  kill -9 $tree 2>/dev/null || true
}
trap cleanup_server EXIT

sample_idle() {
  local dest="$1" bench="$2"
  : >"$dest"
  while kill -0 "$bench" 2>/dev/null; do
    {
      echo "---- $(date -Is)"
      curl -s "http://127.0.0.1:$PORT/metrics" | awk '
        /^vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|num_preemptions_total|prefix_cache_hits_total|prefix_cache_queries_total|spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total|prompt_tokens_total|generation_tokens_total)\{/ { print }
      '
    } >>"$dest"
    sleep 10
  done
}

make_spec() {
  local cfg="$1" draft="$2"
  DRAFT="$draft" MAX_LEN="$MAX_LEN" CFG="$cfg" python - <<'PY'
import json, os
cfg = json.load(open(os.environ["CFG"]))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = int(os.environ["MAX_LEN"])
print(json.dumps(cfg))
PY
}

config_paths() {
  local name="$1"
  case "$name" in
    dflash_fp8)
      OUT_BASE=$ROOT/out/dflash_qwen38_fp8
      SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2.json
      DRAFT="$DFLASH_DRAFT"
      ATTN=ROCM_AITER_UNIFIED_ATTN
      KV=fp8
      BLOCK=
      UTIL="$UTIL_FP8"
      UQ_ENV=0
      ;;
    dspark_fp8)
      OUT_BASE=$ROOT/out/dspark_qwen38_fp8
      SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38/configs/fp8_dspark.json
      DRAFT="$DSPARK_DRAFT"
      ATTN=ROCM_AITER_UNIFIED_ATTN
      KV=fp8
      BLOCK=
      UTIL="$UTIL_FP8"
      UQ_ENV=0
      ;;
    dflash_uq)
      OUT_BASE=$ROOT/out/dflash_qwen38_uq
      SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38_uq/configs/uq_dflash2.json
      DRAFT="$DFLASH_DRAFT"
      ATTN=ULTRAQUANT_SPEC_COMPAT
      KV=ultraquant_4bit
      BLOCK=64
      UTIL="$UTIL_UQ"
      UQ_ENV=1
      ;;
    dspark_uq)
      OUT_BASE=$ROOT/out/dspark_qwen38_uq
      SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark.json
      DRAFT="$DSPARK_DRAFT"
      ATTN=ULTRAQUANT_SPEC_COMPAT
      KV=ultraquant_4bit
      BLOCK=64
      UTIL="$UTIL_UQ"
      UQ_ENV=1
      ;;
    dflash_fp8_p01)
      # P0-1: block 64 → hybrid KV LCM 1664 (smoke-verified)
      OUT_BASE=$ROOT/out/dflash_qwen38_fp8_p01
      SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2.json
      DRAFT="$DFLASH_DRAFT"
      ATTN=ROCM_AITER_UNIFIED_ATTN
      KV=fp8
      BLOCK=64
      UTIL="$UTIL_FP8"
      UQ_ENV=0
      ;;
    dspark_fp8_p01)
      # P0-1 partial: block 64 → LCM 3328 (draft kv8 Mixed FP8 still blocked)
      OUT_BASE=$ROOT/out/dspark_qwen38_fp8_p01
      SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38/configs/fp8_dspark.json
      DRAFT="$DSPARK_DRAFT"
      ATTN=ROCM_AITER_UNIFIED_ATTN
      KV=fp8
      BLOCK=64
      UTIL="$UTIL_FP8"
      UQ_ENV=0
      ;;
    *) log "FAIL bad config $name"; return 1 ;;
  esac
}

set_uq_env() {
  if [ "$1" = 1 ]; then
    export VLLM_ULTRAQUANT_FUSED_MTP=1
    export VLLM_ULTRAQUANT_SHARED_KV=0
    export VLLM_ULTRAQUANT_V3=0
    export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
    export VLLM_ULTRAQUANT_PAIR_WAVES=2
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
    unset VLLM_ULTRAQUANT_PDV_EXPERIMENT
  else
    unset VLLM_ULTRAQUANT_FUSED_MTP VLLM_ULTRAQUANT_SHARED_KV VLLM_ULTRAQUANT_V3
    unset VLLM_ULTRAQUANT_ROWPACK_VERIFY VLLM_ULTRAQUANT_PAIR_WAVES
    unset VLLM_ULTRAQUANT_PAIR_VERIFY VLLM_ULTRAQUANT_PDV_VERIFY
    unset VLLM_ULTRAQUANT_PDV_EXPERIMENT
  fi
}

run_cell() {
  local cfg_name="$1" conc="$2"
  config_paths "$cfg_name" || return 1
  local out="$OUT_BASE/sa/seed${SEED}_c${conc}"
  if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "SKIP $cfg_name c$conc"
    return 0
  fi
  if [ ! -f "$DRAFT/config.json" ]; then
    log "FAIL $cfg_name missing draft $DRAFT"
    return 1
  fi
  cleanup_server
  mkdir -p "$out"
  set_uq_env "$UQ_ENV"
  local spec
  spec=$(make_spec "$SPEC_FILE" "$DRAFT")
  log "SA START $cfg_name c$conc hip=$HIP_VISIBLE_DEVICES port=$PORT attn=$ATTN kv=$KV util=$UTIL"
  local serve_args=(
    serve "$MODEL" --port "$PORT" --tensor-parallel-size 1
    --attention-backend "$ATTN" --kv-cache-dtype "$KV"
    --max-model-len "$MAX_LEN" --gpu-memory-utilization "$UTIL"
    --enable-prefix-caching --no-enable-log-requests
    --max-num-seqs "$conc" --max-num-batched-tokens "$MNBT"
    --language-model-only --reasoning-parser qwen3 --trust-remote-code
    --speculative-config "$spec"
  )
  if [ -n "$BLOCK" ]; then
    serve_args+=(--block-size "$BLOCK")
  fi
  vllm "${serve_args[@]}" >"$out/server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0 i
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL $cfg_name c$conc server exited"
      tail -n 60 "$out/server.log" || true
      cleanup_server
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL $cfg_name c$conc server not ready"
    tail -n 60 "$out/server.log" || true
    cleanup_server
    return 1
  fi
  log "READY $cfg_name c$conc"

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
  until grep -q "Phase profiling (profiling) started" "$out/benchmark.log" 2>/dev/null; do
    if ! kill -0 "$bench" 2>/dev/null; then
      log "FAIL $cfg_name c$conc aiperf exited before measuring"
      tail -n 40 "$out/benchmark.log" || true
      cleanup_server
      return 1
    fi
    sleep 5
  done
  log "SA MEASURING $cfg_name c$conc"
  sample_idle "$out/idle_samples.txt" "$bench" &
  local sampler=$!
  wait "$bench" || log "WARN $cfg_name c$conc aiperf rc=$?"
  wait "$sampler" 2>/dev/null || true
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions|num_requests_running|num_requests_waiting|gpu_cache_usage)' \
    >"$out/metrics_end.txt" || true
  {
    printf '%s\n' "config=$cfg_name conc=$conc seed=$SEED duration=$DURATION attn=$ATTN kv=$KV util=$UTIL"
    [ -n "$BLOCK" ] && printf '%s\n' "block_size=$BLOCK"
  } >"$out/cell_meta.txt"
  cleanup_server
  if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "END $cfg_name c$conc"
  else
    log "FAIL $cfg_name c$conc no profile export"
    return 1
  fi
}

log "agentx conc sweep start pid=$$ role=$ROLE hip=$HIP_VISIBLE_DEVICES port=$PORT concs=${CONCS[*]} configs=${CONFIGS[*]}"
for cfg in "${CONFIGS[@]}"; do
  for conc in "${CONCS[@]}"; do
    run_cell "$cfg" "$conc" || true
  done
done

# Retry missing cells once.
log "agentx conc sweep retry pass role=$ROLE"
for cfg in "${CONFIGS[@]}"; do
  for conc in "${CONCS[@]}"; do
    config_paths "$cfg" || continue
    out="$OUT_BASE/sa/seed${SEED}_c${conc}"
    if [ ! -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
      log "RETRY $cfg c$conc"
      rm -rf "$out"
      run_cell "$cfg" "$conc" || true
    fi
  done
done
cleanup_server
log "agentx conc sweep done role=$ROLE"
