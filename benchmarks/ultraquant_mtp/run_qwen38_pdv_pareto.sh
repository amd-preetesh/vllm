#!/bin/bash
# PDV vs FP8 AgentX concurrency sweep for a Pareto plot.
# ROLE=a: concurrencies 8 and 16. ROLE=b: 36 and 40.
# Concurrency 32 is already measured; skip it unless FORCE_C32=1.
# Shipping PDV kernel (VLLM_ULTRAQUANT_PDV_EXPERIMENT unset).
trap '' HUP
mkdir -p /workspace/logs
exec >> "/workspace/logs/pdv-pareto-${ROLE:?need ROLE}.log" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B

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
export VLLM_ULTRAQUANT_PDV_VERIFY=1
unset VLLM_ULTRAQUANT_PDV_EXPERIMENT

case "$HIP_VISIBLE_DEVICES" in
  [0-7]) ;;
  *) echo "$(date -Is) FAIL refusing HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"; exit 1 ;;
esac

PORT="${PDV_PARETO_PORT:?need port}"
SEED="${PDV_PARETO_SEED:-42}"
DURATION="${PDV_PARETO_DURATION:-300}"
OUT="$ROOT/out/pdv_pareto"
mkdir -p "$OUT"
cd "$ROOT"
log() { echo "$(date -Is) $*"; }

if [ "$ROLE" = "a" ]; then
  CONCS=(8 16)
elif [ "$ROLE" = "b" ]; then
  CONCS=(36 40)
else
  log "FAIL unknown ROLE=$ROLE"
  exit 1
fi
# Optional override, e.g. PDV_PARETO_CONCS="8" to remeasure only c=8.
if [ -n "${PDV_PARETO_CONCS:-}" ]; then
  # shellcheck disable=SC2206
  CONCS=($PDV_PARETO_CONCS)
fi
if [ "${FORCE_C32:-0}" = "1" ]; then
  CONCS+=(32)
fi

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
  local name="$1" kv="$2" mode="$3" conc="$4"
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
  # High-concurrency UltraQuant needs headroom for continuation dequant /
  # concat workspace on top of the KV pool. 0.90 packs the pool so tightly
  # that the first long prefill OOMs allocating ~1 GiB of shared workspace.
  local mem_util=0.90
  if [ "$mode" = "pdv" ] && [ "$conc" -ge 36 ]; then
    mem_util=0.85
  fi
  log "SA START $name hip=$HIP_VISIBLE_DEVICES kv=$kv mode=$mode conc=$conc seed=$SEED mem_util=$mem_util"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$kv" \
    --max-model-len 262144 --gpu-memory-utilization "$mem_util" \
    --enable-prefix-caching --no-enable-log-requests \
    --max-num-seqs "$conc" --max-num-batched-tokens 8192 \
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
    --model "$MODEL" --concurrency "$conc" --benchmark-duration "$DURATION" \
    --unsafe-override \
    --stats-interval 30 --random-seed "$SEED" --failed-request-threshold 0.10 \
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
  printf '%s\n' "mode=$mode conc=$conc seed=$SEED duration=$DURATION" > "$out/cell_meta.txt"
  cleanup_server
  log "SA END $name"
}

log "pdv pareto start pid=$$ ppid=$PPID role=$ROLE hip=$HIP_VISIBLE_DEVICES port=$PORT concs=${CONCS[*]} seed=$SEED"
for conc in "${CONCS[@]}"; do
  # Alternate order by concurrency so neither mode always runs first cold.
  if [ $((conc % 8)) -eq 0 ]; then
    run_cell "fp8_c${conc}_seed${SEED}" fp8 fp8 "$conc" || true
    run_cell "pdv_c${conc}_seed${SEED}" ultraquant_4bit pdv "$conc" || true
  else
    run_cell "pdv_c${conc}_seed${SEED}" ultraquant_4bit pdv "$conc" || true
    run_cell "fp8_c${conc}_seed${SEED}" fp8 fp8 "$conc" || true
  fi
done
# Retry any cell that never produced a profile export (e.g. PDV OOM at c=36).
log "pdv pareto retry pass role=$ROLE"
for conc in "${CONCS[@]}"; do
  for mode_kv in "fp8:fp8" "pdv:ultraquant_4bit"; do
    mode="${mode_kv%%:*}"
    kv="${mode_kv##*:}"
    name="${mode}_c${conc}_seed${SEED}"
    if [ ! -f "$OUT/$name/aiperf_artifacts/profile_export_aiperf.json" ]; then
      log "RETRY $name"
      rm -rf "$OUT/$name"
      run_cell "$name" "$kv" "$mode" "$conc" || true
    fi
  done
done
cleanup_server
log "pdv pareto done role=$ROLE"
