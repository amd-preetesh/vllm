#!/bin/bash
# Qwen3.5-397B-A17B-FP8, TP 2, fp8 KV, AgentX seed 42.
# vLLM equivalent of the SGLang recipe's native MTP chain:
# 3 draft steps on the single MTP layer, verify window of 4 tokens.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/qwen35-397b-mtp3-sa.log 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.5-397B-A17B-FP8

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

PORT="${MTP3_PORT:-8079}"
CONC=8
TP=2
SPEC='{"method":"mtp","num_speculative_tokens":3}'
OUT="$ROOT/out/qwen35_397b_mtp3_sa"
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

log "397b mtp3 sa start pid=$$ ppid=$PPID hip=${HIP_VISIBLE_DEVICES} port=${PORT} spec=${SPEC}"
wait_for_free_gpu || exit 1
mkdir -p "$OUT/sa"
log "SERVE mtp3 max_len=262144 prefix=1 hip=${HIP_VISIBLE_DEVICES} tp=${TP} conc=${CONC}"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "$TP" \
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8 \
  --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --no-enable-log-requests \
  --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  --speculative-config "$SPEC" \
  > "$OUT/sa/server.log" 2>&1 &
SERVER_PID=$!
ready=0
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.5-397B"; then
    log "READY mtp3"
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "FAIL mtp3 server exited; see ${OUT}/sa/server.log"
    exit 1
  fi
  sleep 10
done
if [ "$ready" != "1" ]; then
  log "FAIL mtp3 server not ready"
  exit 1
fi

extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')
log "SA START seed=42"
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
  --output-artifact-dir "$OUT/sa/aiperf_artifacts" \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  "${extra[@]}") > "$OUT/sa/benchmark.log" 2>&1 &
bench=$!
until grep -q "Phase profiling (profiling) started" "$OUT/sa/benchmark.log" 2>/dev/null; do
  if ! kill -0 "$bench" 2>/dev/null; then
    log "FAIL SA aiperf exited before measuring"
    exit 1
  fi
  sleep 5
done
log "SA MEASURING"
wait "$bench" || log "WARN SA aiperf rc=$?"
curl -s "http://127.0.0.1:$PORT/metrics" \
  | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' > "$OUT/sa/metrics_end.txt" || true
log "SA END"
log "397b mtp3 sa complete"
touch /workspace/logs/qwen35-397b-mtp3-sa.ok
