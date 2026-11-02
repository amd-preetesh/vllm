#!/bin/bash
# One AgentX cell for the concurrency Pareto.
# Prod FlyDSL is pair verify off. Fused FlyDSL is not launched here.
# Required: PARETO_NAME PARETO_KV PARETO_PAIR PARETO_CONC HIP_VISIBLE_DEVICES PROFILE_PORT
trap '' HUP
mkdir -p /workspace/logs
NAME="${PARETO_NAME:?}"
exec >> "/workspace/logs/kernel-profile-tp1-pareto-${NAME}.log" 2>&1
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SA_DIR/.hf-cache}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export VLLM_ULTRAQUANT_FUSED_MTP=1
export VLLM_ULTRAQUANT_SHARED_KV=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_PAIR_WAVES=2
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_VERIFY="${PARETO_PAIR:?}"

KV="${PARETO_KV:?}"
CONC="${PARETO_CONC:?}"
PORT="${PROFILE_PORT:?}"
RUN_S="${PROFILE_RUN_S:-300}"
OUT_ROOT="$ROOT/out/kernel_profile_tp1_pareto_fill"
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

out="$OUT_ROOT/$NAME"
if [ -f "$out/metrics_end.txt" ] && [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
  log "SKIP ${NAME} already complete"
  log "pareto cell complete"
  exit 0
fi
mkdir -p "$out"
extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')

log "pareto cell start pid=$$ ppid=$PPID hip=${HIP_VISIBLE_DEVICES} tp=1 conc=${CONC} kv=${KV} mtp=5 pair=${PARETO_PAIR} port=${PORT}"
log "START ${NAME} hip=${HIP_VISIBLE_DEVICES} tp=1 conc=${CONC} kv=${KV} mtp=5 pair=${PARETO_PAIR}"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$KV" \
  --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --no-enable-log-requests \
  --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  > "$out/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "FAIL ${NAME} server exited"
    cleanup_server
    exit 1
  fi
  sleep 5
done
if [ "$ready" != 1 ]; then
  log "FAIL ${NAME} server not ready"
  cleanup_server
  exit 1
fi
log "READY ${NAME}"

run_aiperf "$out/aiperf_artifacts" "$RUN_S" "${extra[@]}" > "$out/benchmark.log" 2>&1 &
bench=$!
until grep -q "Phase profiling (profiling) started" "$out/benchmark.log" 2>/dev/null; do
  if ! kill -0 "$bench" 2>/dev/null; then
    log "FAIL ${NAME} aiperf exited before measuring"
    cleanup_server
    exit 1
  fi
  sleep 5
done
log "MEASURING ${NAME}"
wait "$bench" || log "WARN ${NAME} aiperf rc=$?"
curl -s "http://127.0.0.1:$PORT/metrics" \
  | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' > "$out/metrics_end.txt" || true
cleanup_server
log "END ${NAME}"
log "pareto cell complete"
