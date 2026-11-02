#!/bin/bash
# One AgentX cell: MODE=fp8|pdv|prod|hoist_k|qblock_waves|pdv2|pdv2_fp4q|pdv2_fp4q  CONC=8|16|32|36|40
# prod = UltraQuant production FlyDSL (PDV off). Same scenario as Pareto.
trap '' HUP
mkdir -p /workspace/logs
MODE="${MODE:?need MODE}"
CONC="${CONC:?need CONC}"
case "$MODE" in
  fp8|pdv|prod|hoist_k|qblock_waves|pdv2|pdv2_fp4q|pdv2_fp4q) ;;
  *) echo "$(date -Is) FAIL bad MODE=$MODE"; exit 1 ;;
esac
case "$CONC" in
  8|16|32|36|40) ;;
  *) echo "$(date -Is) FAIL bad CONC=$CONC"; exit 1 ;;
esac
exec >> "/workspace/logs/pdv-matrix-agentx-${MODE}-c${CONC}.log" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
SEED="${PDV_PARETO_SEED:-42}"
DURATION="${PDV_PARETO_DURATION:-300}"
PORT="${PDV_AGENTX_PORT:?need PDV_AGENTX_PORT}"
OUT="$ROOT/out/pdv_pareto/${MODE}_c${CONC}_seed${SEED}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?}"
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
export VLLM_ULTRAQUANT_PAIR_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_WAVES=2
unset VLLM_ULTRAQUANT_PDV_EXPERIMENT

case "$MODE" in
  fp8)
    KV=fp8
    export VLLM_ULTRAQUANT_PDV_VERIFY=0
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    ;;
  pdv)
    KV=ultraquant_4bit
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    ;;
  prod)
    KV=ultraquant_4bit
    export VLLM_ULTRAQUANT_PDV_VERIFY=0
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
    ;;
  hoist_k|qblock_waves|pdv2|pdv2_fp4q)
    KV=ultraquant_4bit
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    export VLLM_ULTRAQUANT_PDV_EXPERIMENT="$MODE"
    ;;
esac

# Reuse hoist_k experiment SA at c=32 if present
if [ "$MODE" = hoist_k ] && [ "$CONC" = 32 ]; then
  alt="$ROOT/out/pdv_exp_hoist_k/sa/seed42_c32/aiperf_artifacts/profile_export_aiperf.json"
  if [ -f "$alt" ] && [ ! -f "$OUT/aiperf_artifacts/profile_export_aiperf.json" ]; then
    mkdir -p "$OUT/aiperf_artifacts"
    cp -a "$ROOT/out/pdv_exp_hoist_k/sa/seed42_c32/aiperf_artifacts/." "$OUT/aiperf_artifacts/"
    : > "$OUT/idle_samples.txt"
    echo "reused pdv_exp_hoist_k sa c32" > "$OUT/cell_meta.txt"
    echo "$(date -Is) SKIP hoist_k c32 reused experiment SA"
    exit 0
  fi
fi

mkdir -p "$OUT"
cd "$ROOT"
log() { echo "$(date -Is) $*"; }

if [ -f "$OUT/aiperf_artifacts/profile_export_aiperf.json" ] && [ -f "$OUT/idle_samples.txt" ]; then
  log "SKIP $MODE c=$CONC already complete"
  exit 0
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

mem_util=0.90
if [ "$MODE" != fp8 ] && [ "$CONC" -ge 36 ]; then
  mem_util=0.85
fi

cleanup_server
log "SA START mode=$MODE conc=$CONC hip=$HIP_VISIBLE_DEVICES port=$PORT kv=$KV mem_util=$mem_util"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$KV" \
  --max-model-len 262144 --gpu-memory-utilization "$mem_util" \
  --enable-prefix-caching --no-enable-log-requests \
  --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
ready=0
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "FAIL server exited"
    exit 1
  fi
  sleep 5
done
if [ "$ready" != 1 ]; then
  log "FAIL server not ready"
  exit 1
fi
case "$MODE" in
  fp8) ;;
  prod)
    if grep -q "PDV FlyDSL verify is enabled" "$OUT/server.log"; then
      log "FAIL prod mode unexpectedly enabled PDV"
      exit 1
    fi
    ;;
  *)
    if ! grep -q "PDV FlyDSL verify is enabled" "$OUT/server.log"; then
      log "FAIL PDV line missing"
      exit 1
    fi
    ;;
esac

extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')
(cd "$INFERENCEX_DIR" && "$AIPERF" profile --scenario inferencex-agentx-mvp \
  --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model "$MODEL" --concurrency "$CONC" --benchmark-duration "$DURATION" \
  --unsafe-override \
  --stats-interval 30 --random-seed "$SEED" --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-grace-period 300 \
  --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
  --max-context-length 262144 \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --output-artifact-dir "$OUT/aiperf_artifacts" \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  "${extra[@]}") > "$OUT/benchmark.log" 2>&1 &
bench=$!
until grep -q "Phase profiling (profiling) started" "$OUT/benchmark.log" 2>/dev/null; do
  if ! kill -0 "$bench" 2>/dev/null; then
    log "FAIL aiperf exited before measuring"
    exit 1
  fi
  sleep 5
done
log "SA MEASURING mode=$MODE conc=$CONC"
sample_idle "$OUT/idle_samples.txt" "$bench" &
sampler=$!
wait "$bench" || log "WARN aiperf rc=$?"
wait "$sampler" 2>/dev/null || true
curl -s "http://127.0.0.1:$PORT/metrics" \
  | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions|num_requests_running|num_requests_waiting|gpu_cache_usage)' \
  > "$OUT/metrics_end.txt" || true
printf '%s\n' "mode=$MODE conc=$CONC seed=$SEED duration=$DURATION" > "$OUT/cell_meta.txt"
cleanup_server
if [ -f "$OUT/aiperf_artifacts/profile_export_aiperf.json" ]; then
  log "SA END mode=$MODE conc=$CONC ok"
  exit 0
fi
log "FAIL mode=$MODE conc=$CONC missing profile export"
exit 1
