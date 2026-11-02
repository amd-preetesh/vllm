#!/bin/bash
# Qwen3.8-27B, TP=1, concurrency 32, MTP 5.
# Three cells: UltraQuant fused MTP, UltraQuant with metadata rebuilt each
# draft step, then FP8 AITER unified attention.
# HIP 0. One GPU. No torch trace; this run is the serving comparison.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/kernel-profile-tp1-fused.log 2>&1
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SA_DIR/.hf-cache}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0

PORT="${PROFILE_PORT:-8057}"
RUN_S="${PROFILE_RUN_S:-300}"
CONC="${PROFILE_CONC:-32}"
OUT_ROOT="$ROOT/out/kernel_profile_tp1_fused3"
mkdir -p "$OUT_ROOT"
cd "$ROOT"

log() { echo "$(date -Is) $*"; }

SERVER_PID=
cleanup_server() {
  [ -n "${SERVER_PID:-}" ] || return 0
  local pid="$SERVER_PID" c
  SERVER_PID=
  local tree="$pid"
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

run_aiperf() {
  local out_dir="$1" seconds="$2"
  shift 2
  (cd "$INFERENCEX_DIR" && "$AIPERF" profile --scenario inferencex-agentx-mvp \
    --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
    --model "$MODEL" --concurrency "$CONC" --benchmark-duration "$seconds" \
    --unsafe-override \
    --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-grace-period 300 \
    --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
    --max-context-length 262144 \
    --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir "$out_dir" \
    --public-dataset semianalysis_cc_traces_weka_062126_256k \
    "$@")
}

run_cell() {
  cleanup_server
  local name="$1" kv="$2" fused="$3"
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"
  export VLLM_ULTRAQUANT_FUSED_MTP="$fused"
  local extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')

  log "START ${name} hip=${HIP_VISIBLE_DEVICES} tp=1 conc=${CONC} kv=${kv} mtp=5 fused=${fused}"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$kv" \
    --max-model-len 262144 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching --no-enable-log-requests \
    --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
    --language-model-only --reasoning-parser qwen3 --trust-remote-code \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
    > "$out/server.log" 2>&1 &
  SERVER_PID=$!

  local ready=0 i
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL ${name} server exited"
      cleanup_server
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL ${name} server not ready"
    cleanup_server
    return 1
  fi
  log "READY ${name}"

  run_aiperf "$out/aiperf_artifacts" "$RUN_S" "${extra[@]}" > "$out/benchmark.log" 2>&1 &
  local bench=$!
  until grep -q "Phase profiling (profiling) started" "$out/benchmark.log" 2>/dev/null; do
    if ! kill -0 "$bench" 2>/dev/null; then
      log "FAIL ${name} aiperf exited before measuring"
      cleanup_server
      return 1
    fi
    sleep 5
  done
  log "MEASURING ${name}"
  wait "$bench" || log "WARN ${name} aiperf rc=$?"
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' > "$out/metrics_end.txt" || true
  cleanup_server
  log "END ${name}"
}

log "fused compare start pid=$$ ppid=$PPID hip=${HIP_VISIBLE_DEVICES} tp=1 conc=${CONC} mtp=5 run=${RUN_S}s"
run_cell qwen3-8-27b_ultraquant_4bit_mtp5_tp1_fused ultraquant_4bit 1 || log "CELL fused stopped early"
run_cell qwen3-8-27b_ultraquant_4bit_mtp5_tp1_rebuild ultraquant_4bit 0 || log "CELL rebuild stopped early"
run_cell qwen3-8-27b_fp8_mtp5_tp1 fp8 0 || log "CELL fp8 stopped early"
log "fused compare complete"
