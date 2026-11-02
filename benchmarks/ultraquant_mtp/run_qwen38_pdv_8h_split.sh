#!/bin/bash
# Eight-hour PDV vs FP8 split. One GPU, TP 1.
# ROLE=a: seeds 42-45, two paired repeats, a third seed-42 pair, then accuracy.
# ROLE=b: seeds 46-49, two paired repeats, then one densely sampled seed-42 pair.
trap '' HUP
mkdir -p /workspace/logs
exec >> "/workspace/logs/pdv-8h-${ROLE:?need ROLE}.log" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
EVAL_HARNESS=/workspace/eval-harness
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
GSM_CONFIG="$EVAL_HARNESS/configs/benchmarks/gsm8k_ffdsl.json"
GPQA_CONFIG="$EVAL_HARNESS/configs/benchmarks/gpqa_diamond_qwen38_card.json"

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
export VLLM_ULTRAQUANT_FUSED_MTP=1
export VLLM_ULTRAQUANT_SHARED_KV=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_WAVES=2

case "$HIP_VISIBLE_DEVICES" in
  [0-7]) ;;
  *) echo "$(date -Is) FAIL refusing HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"; exit 1 ;;
esac

PORT="${PDV_8H_PORT:?need port}"
CONC=32
OUT="$ROOT/out/pdv_8h_split"
ACC="$ROOT/out/pdv_8h_accuracy"
mkdir -p "$OUT" "$ACC"
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
  sleep 15
  # shellcheck disable=SC2086
  kill -9 $tree 2>/dev/null || true
}
trap cleanup_server EXIT

sample_idle() {
  local dest="$1" bench="$2"
  : > "$dest"
  while kill -0 "$bench" 2>/dev/null; do
    {
      echo "---- $(date -Is)"
      curl -s "http://127.0.0.1:$PORT/metrics" | awk '
        /^vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|prefix_cache_hits_total|prefix_cache_queries_total|spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total|prompt_tokens_total|generation_tokens_total)\{/ { print }
      '
    } >> "$dest"
    sleep 10
  done
}

run_cell() {
  local name="$1" kv="$2" mode="$3" seed="$4" interval="${5:-30}"
  local out="$OUT/$name"
  if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ] && [ -f "$out/idle_samples.txt" ]; then
    log "SKIP $name"
    return 0
  fi
  cleanup_server
  mkdir -p "$out"
  if [ "$mode" = "pdv" ]; then
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
  else
    export VLLM_ULTRAQUANT_PDV_VERIFY=0
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
  fi
  log "SA START $name hip=$HIP_VISIBLE_DEVICES kv=$kv mode=$mode seed=$seed"
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
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL $name server exited"
      cleanup_server
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL $name server not ready"
    cleanup_server
    return 1
  fi
  if [ "$mode" = "pdv" ] && ! grep -q "PDV FlyDSL verify is enabled" "$out/server.log"; then
    log "FAIL $name PDV line missing"
    cleanup_server
    return 1
  fi
  local extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')
  (cd "$INFERENCEX_DIR" && "$AIPERF" profile --scenario inferencex-agentx-mvp \
    --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
    --model "$MODEL" --concurrency "$CONC" --benchmark-duration 300 \
    --unsafe-override \
    --stats-interval "$interval" --random-seed "$seed" --failed-request-threshold 0.10 \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-grace-period 300 \
    --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
    --max-context-length 262144 \
    --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir "$out/aiperf_artifacts" \
    --public-dataset semianalysis_cc_traces_weka_062126_256k \
    "${extra[@]}") > "$out/benchmark.log" 2>&1 &
  local bench=$!
  until grep -q "Phase profiling (profiling) started" "$out/benchmark.log" 2>/dev/null; do
    if ! kill -0 "$bench" 2>/dev/null; then
      log "FAIL $name aiperf exited before measuring"
      cleanup_server
      return 1
    fi
    sleep 5
  done
  log "SA MEASURING $name"
  sample_idle "$out/idle_samples.txt" "$bench" &
  local sampler=$!
  wait "$bench" || log "WARN $name aiperf rc=$?"
  wait "$sampler" 2>/dev/null || true
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions|num_requests_running|num_requests_waiting|gpu_cache_usage)' \
    > "$out/metrics_end.txt" || true
  cleanup_server
  log "SA END $name"
}

run_pair() {
  local seed="$1" rep="$2" interval="${3:-30}"
  if [ $((rep % 2)) -eq 1 ]; then
    run_cell "fp8_seed${seed}_r${rep}" fp8 fp8 "$seed" "$interval" || true
    run_cell "pdv_seed${seed}_r${rep}" ultraquant_4bit pdv "$seed" "$interval" || true
  else
    run_cell "pdv_seed${seed}_r${rep}" ultraquant_4bit pdv "$seed" "$interval" || true
    run_cell "fp8_seed${seed}_r${rep}" fp8 fp8 "$seed" "$interval" || true
  fi
}

serve_accuracy() {
  local kv="$1" mode="$2" max_len="$3"
  cleanup_server
  if [ "$mode" = "pdv" ]; then
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
  else
    export VLLM_ULTRAQUANT_PDV_VERIFY=0
  fi
  export VLLM_ULTRAQUANT_PAIR_VERIFY=0
  local tag="acc_${mode}_len${max_len}"
  mkdir -p "$ACC/$tag"
  log "ACC SERVE mode=$mode kv=$kv max_len=$max_len"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$kv" \
    --max-model-len "$max_len" --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --no-enable-log-requests \
    --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
    --language-model-only --reasoning-parser qwen3 --trust-remote-code \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
    > "$ACC/$tag/server.log" 2>&1 &
  SERVER_PID=$!
  local i
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
      log "ACC READY mode=$mode max_len=$max_len"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL acc server mode=$mode"
      cleanup_server
      return 1
    fi
    sleep 5
  done
  log "FAIL acc server not ready mode=$mode"
  cleanup_server
  return 1
}

run_gsm() {
  local mode="$1" seed="$2"
  local out="$ACC/gsm8k_${mode}_seed${seed}"
  if [ -f "$out/summary.json" ]; then
    log "SKIP gsm8k $mode seed=$seed"
    return 0
  fi
  mkdir -p "$out"
  log "EVAL gsm8k $mode seed=$seed"
  python "$EVAL_HARNESS/drivers/gsm8k_native.py" \
    --port "$PORT" --model "$MODEL" --max_model_len 4096 \
    --bench_config "$GSM_CONFIG" --out_dir "$out" --seed "$seed" \
    --concurrency_override "$CONC" > "$out/eval.log" 2>&1 || log "WARN gsm8k $mode seed=$seed"
  if [ -f "$out/summary.json" ]; then log "END gsm8k $mode seed=$seed"; else log "FAIL gsm8k $mode seed=$seed"; fi
}

run_gpqa() {
  local mode="$1" seed="$2"
  local out="$ACC/gpqa_${mode}_seed${seed}"
  if [ -f "$out/summary.json" ]; then
    log "SKIP gpqa $mode seed=$seed"
    return 0
  fi
  mkdir -p "$out"
  log "EVAL gpqa $mode seed=$seed"
  python "$EVAL_HARNESS/drivers/gpqa_native.py" \
    --port "$PORT" --model "$MODEL" --max_model_len 262144 \
    --bench_config "$GPQA_CONFIG" --out_dir "$out" --seed "$seed" \
    --concurrency_override "$CONC" > "$out/eval.log" 2>&1 || log "WARN gpqa $mode seed=$seed"
  if [ -f "$out/summary.json" ]; then log "END gpqa $mode seed=$seed"; else log "FAIL gpqa $mode seed=$seed"; fi
}

log "pdv 8h start pid=$$ ppid=$PPID role=$ROLE hip=$HIP_VISIBLE_DEVICES port=$PORT"
if [ "$ROLE" = "a" ]; then
  for seed in 42 43 44 45; do
    run_pair "$seed" 1
    run_pair "$seed" 2
  done
  run_pair 42 3
  if serve_accuracy fp8 fp8 4096; then
    for seed in 7 11 13; do run_gsm fp8 "$seed"; done
  fi
  cleanup_server
  if serve_accuracy ultraquant_4bit pdv 4096; then
    for seed in 7 11 13; do run_gsm pdv "$seed"; done
  fi
  cleanup_server
  if serve_accuracy fp8 fp8 262144; then
    run_gpqa fp8 7
  fi
  cleanup_server
  if serve_accuracy ultraquant_4bit pdv 262144; then
    run_gpqa pdv 7
  fi
  cleanup_server
elif [ "$ROLE" = "b" ]; then
  for seed in 46 47 48 49; do
    run_pair "$seed" 1
    run_pair "$seed" 2
  done
  run_pair 42 4 10
else
  log "FAIL unknown ROLE=$ROLE"
  exit 1
fi
log "pdv 8h complete role=$ROLE"
touch "/workspace/logs/pdv-8h-${ROLE}.ok"
