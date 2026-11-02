#!/bin/bash
# Resumable Qwen3.8-27B MTP serving matrix. HIP 0 only.
# Ready cells run immediately. Gated cells wait for logs/weekend-gates/<name>.
trap '' HUP
mkdir -p /workspace/logs /workspace/logs/weekend-gates
exec >> "${WEEKEND_LOG:-/workspace/logs/kernel-profile-tp1-weekend.log}" 2>&1
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
MANIFEST="${WEEKEND_MANIFEST:-$ROOT/benchmarks/ultraquant_mtp/weekend_manifest.tsv}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SA_DIR/.hf-cache}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export VLLM_ULTRAQUANT_FUSED_MTP=1
export VLLM_ULTRAQUANT_SHARED_KV=0
export VLLM_ULTRAQUANT_V3=0

PORT="${PROFILE_PORT:-8062}"
OUT_ROOT="${WEEKEND_OUT:-$ROOT/out/kernel_profile_tp1_weekend}"
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
  local out_dir="$1" seconds="$2" conc="$3"
  shift 3
  (cd "$INFERENCEX_DIR" && "$AIPERF" profile --scenario inferencex-agentx-mvp \
    --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
    --model "$MODEL" --concurrency "$conc" --benchmark-duration "$seconds" \
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
  local id="$1" mtp="$2" conc="$3" seconds="$4" pair="$5" waves="$6" aux="$7" tok="$8" parts="$9"
  local out="$OUT_ROOT/$id"
  if [ -f "$out/metrics_end.txt" ] && [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "SKIP ${id} already complete"
    return 0
  fi
  cleanup_server
  mkdir -p "$out"
  export VLLM_ULTRAQUANT_PAIR_VERIFY="$pair"
  export VLLM_ULTRAQUANT_PAIR_WAVES="$waves"
  export VLLM_ULTRAQUANT_ROWPACK_VERIFY="${WEEKEND_ROWPACK:-0}"
  export VLLM_ULTRAQUANT_KV_AUX="$aux"
  export VLLM_ULTRAQUANT_TOK_PER_PART="$tok"
  if [ "$parts" = "0" ]; then
    unset VLLM_ULTRAQUANT_MAX_PARTITIONS
  else
    export VLLM_ULTRAQUANT_MAX_PARTITIONS="$parts"
  fi
  local extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')

  log "START ${id} hip=${HIP_VISIBLE_DEVICES} tp=1 conc=${conc} mtp=${mtp} pair=${pair} waves=${waves} rowpack=${WEEKEND_ROWPACK:-0} aux=${aux} tok=${tok} parts=${parts}"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype ultraquant_4bit \
    --max-model-len 262144 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching --no-enable-log-requests \
    --max-num-seqs "$conc" --max-num-batched-tokens 8192 \
    --language-model-only --reasoning-parser qwen3 --trust-remote-code \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${mtp}}" \
    > "$out/server.log" 2>&1 &
  SERVER_PID=$!

  local ready=0 i
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL ${id} server exited"
      cleanup_server
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL ${id} server not ready"
    cleanup_server
    return 1
  fi
  log "READY ${id}"

  run_aiperf "$out/aiperf_artifacts" "$seconds" "$conc" "${extra[@]}" > "$out/benchmark.log" 2>&1 &
  local bench=$!
  until grep -q "Phase profiling (profiling) started" "$out/benchmark.log" 2>/dev/null; do
    if ! kill -0 "$bench" 2>/dev/null; then
      log "FAIL ${id} aiperf exited before measuring"
      cleanup_server
      return 1
    fi
    sleep 5
  done
  log "MEASURING ${id}"
  wait "$bench" || log "WARN ${id} aiperf rc=$?"
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' > "$out/metrics_end.txt" || true
  cleanup_server
  log "END ${id}"
}

log "weekend compare start pid=$$ ppid=$PPID hip=${HIP_VISIBLE_DEVICES} port=${PORT} manifest=${MANIFEST}"
for pass in 1 2; do
  log "weekend pass ${pass}"
  while IFS=$'\t' read -r id mtp conc seconds pair waves aux tok parts gate; do
    case "$id" in
      ""|\#*) continue ;;
    esac
    if [ "$gate" != "-" ] && [ ! -f "/workspace/logs/weekend-gates/$gate" ]; then
      log "SKIP ${id} waiting for gate ${gate}"
      continue
    fi
    run_cell "$id" "$mtp" "$conc" "$seconds" "$pair" "$waves" "$aux" "$tok" "$parts" \
      || log "CELL ${id} stopped early"
  done < "$MANIFEST"
done
log "weekend compare complete"
