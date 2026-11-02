#!/bin/bash
# Resume Qwen3.5-397B stream 2 after the long-context restart.
# GSM8K seeds 7, 11, and 13 are already in out/qwen35_397b_stream2.
# Waits until both GPUs actually have 270 GiB free before each server start.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/qwen35-397b-stream2-resume.log 2>&1
set -uo pipefail

ROOT=/workspace/vllm
EVAL_HARNESS=/workspace/eval-harness
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.5-397B-A17B-FP8
GPQA_CONFIG="$EVAL_HARNESS/configs/benchmarks/gpqa_diamond_qwen38_card.json"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need two HIP devices}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SA_DIR/.hf-cache}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export VLLM_ULTRAQUANT_PDV_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_VERIFY=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_SHARED_KV=0

case "${HIP_VISIBLE_DEVICES:-}" in
  [0-7],[0-7]) ;;
  *)
    echo "$(date -Is) FAIL refusing HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-empty}"
    exit 1
    ;;
esac

PORT="${STREAM2_PORT:-8077}"
CONC=8
TP=2
SEEDS="7"
OUT="$ROOT/out/qwen35_397b_stream2"
mkdir -p "$OUT"
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
  sleep 20
  # shellcheck disable=SC2086
  kill -9 $tree 2>/dev/null || true
}
trap cleanup_server EXIT

cell_done() {
  [ -f "$1/summary.json" ] && [ -f "$1/metrics_end.txt" ]
}

wait_for_free_gpu() {
  local i
  for i in $(seq 1 60); do
    if python3 - <<'PY'
import torch
ok = True
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"gpu{i} free_gib={free/2**30:.1f} total_gib={total/2**30:.1f}", flush=True)
    if free < 270 * (1024**3):
        ok = False
raise SystemExit(0 if ok else 1)
PY
    then
      log "GPUs have 270 GiB free"
      return 0
    fi
    sleep 10
  done
  log "FAIL GPUs stayed below 270 GiB free"
  return 1
}

start_server() {
  local tag="$1" max_len="$2" spec="$3" prefix="$4"
  cleanup_server
  wait_for_free_gpu || return 1
  local out="$OUT/$tag"
  mkdir -p "$out"
  local args=(
    vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "$TP"
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8
    --max-model-len "$max_len" --gpu-memory-utilization 0.90
    --no-enable-log-requests
    --max-num-seqs "$CONC" --max-num-batched-tokens 8192
    --language-model-only --reasoning-parser qwen3 --trust-remote-code
  )
  if [ "$prefix" = "1" ]; then
    args+=(--enable-prefix-caching)
  else
    args+=(--no-enable-prefix-caching)
  fi
  if [ -n "$spec" ]; then
    args+=(--speculative-config "$spec")
  fi
  log "SERVE tag=${tag} max_len=${max_len} spec=${spec:-none} prefix=${prefix} hip=${HIP_VISIBLE_DEVICES} tp=${TP} conc=${CONC}"
  "${args[@]}" > "$out/server.log" 2>&1 &
  SERVER_PID=$!
  local i
  for i in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.5-397B"; then
      log "READY tag=${tag}"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL tag=${tag} server exited; see ${out}/server.log"
      cleanup_server
      return 1
    fi
    sleep 10
  done
  log "FAIL tag=${tag} server not ready"
  cleanup_server
  return 1
}

run_gpqa() {
  local seed="$1"
  local out="$OUT/gpqa_fp8kv_seed${seed}"
  if cell_done "$out"; then
    log "SKIP gpqa seed=${seed}"
    return 0
  fi
  mkdir -p "$out"
  log "EVAL gpqa seed=${seed}"
  python "$EVAL_HARNESS/drivers/gpqa_native.py" \
    --port "$PORT" --model "$MODEL" --max_model_len 262144 \
    --bench_config "$GPQA_CONFIG" \
    --out_dir "$out" --seed "$seed" \
    --concurrency_override "$CONC" \
    > "$out/eval.log" 2>&1 || log "WARN gpqa seed=${seed} driver rc=$?"
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' > "$out/metrics_end.txt" || true
  if [ -f "$out/summary.json" ]; then
    log "END gpqa seed=${seed}"
  else
    log "FAIL gpqa seed=${seed} missing summary.json"
    return 1
  fi
}

run_sa() {
  local out="$OUT/sa_fp8kv_c8"
  if [ -f "$out/metrics_end.txt" ] && [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "SKIP SA already complete"
    return 0
  fi
  mkdir -p "$out"
  local extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')
  log "SA START"
  (cd "$INFERENCEX_DIR" && "$AIPERF" profile --scenario inferencex-agentx-mvp \
    --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
    --model "$MODEL" --concurrency "$CONC" --benchmark-duration 300 \
    --unsafe-override \
    --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 \
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
      log "FAIL SA aiperf exited before measuring"
      return 1
    fi
    sleep 5
  done
  log "SA MEASURING"
  wait "$bench" || log "WARN SA aiperf rc=$?"
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' > "$out/metrics_end.txt" || true
  log "SA END"
}

log "397b stream2 resume start pid=$$ ppid=$PPID hip=${HIP_VISIBLE_DEVICES} port=${PORT} tp=${TP} conc=${CONC}"

if ! start_server baseline_gpqa 262144 "" 0; then
  log "397b stream2 resume stopped baseline gpqa server failed"
  exit 1
fi
for seed in $SEEDS; do
  run_gpqa "$seed" || true
done
cleanup_server

if ! start_server baseline_sa 262144 "" 1; then
  log "397b stream2 resume stopped baseline sa server failed"
  exit 1
fi
run_sa || log "SA stopped early"
cleanup_server

if start_server mtp_probe 4096 '{"method":"mtp","num_speculative_tokens":1}' 0; then
  log "MTP probe READY"
  curl -sf "http://127.0.0.1:$PORT/v1/models" > "$OUT/mtp_probe/models.json" || true
  cleanup_server
else
  log "MTP probe failed"
fi

log "EAGLE probe: no Qwen3.5-397B EAGLE or DFlash checkpoint under /huggingface/pretrained_models. Skipping the serve."
log "UltraQuant for this model is not a serving job. Full attention is GQA 16 and the decode kernel is GQA 6."
log "397b stream2 resume complete"
touch /workspace/logs/qwen35-397b-stream2-resume.ok
