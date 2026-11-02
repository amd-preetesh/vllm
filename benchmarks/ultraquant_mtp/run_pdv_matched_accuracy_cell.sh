#!/bin/bash
# Matched seed-7 GSM8K + GPQA for one kernel mode.
# MODE=fp8|prod|pdv|hoist_k
# Fresh out root: vllm/out/pdv_matched_accuracy_seed7/{mode}/
# Same knobs for all modes: MTP 5, conc 32, thinking=model_default,
# no prefix cache, GSM max_len=4096, GPQA max_len=262144.
trap '' HUP
mkdir -p /workspace/logs
MODE="${MODE:?need MODE=fp8|prod|pdv|hoist_k}"
case "$MODE" in
  fp8|prod|pdv|hoist_k) ;;
  *) echo "$(date -Is) FAIL bad MODE=$MODE"; exit 1 ;;
esac
exec >> "/workspace/logs/pdv-matched-acc-${MODE}.log" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
EVAL_HARNESS=/workspace/eval-harness
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
GSM_CONFIG="$EVAL_HARNESS/configs/benchmarks/gsm8k_ffdsl.json"
GPQA_CONFIG="$EVAL_HARNESS/configs/benchmarks/gpqa_diamond_qwen38_card.json"
SEED=7
CONC=32
THINKING=model_default
PORT="${PDV_ACC_PORT:?need PDV_ACC_PORT}"
OUT="$ROOT/out/pdv_matched_accuracy_seed7/$MODE"
mkdir -p "$OUT"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP device}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/SemiAnalysis/.hf-cache}"
export VLLM_ULTRAQUANT_FUSED_MTP=1
export VLLM_ULTRAQUANT_SHARED_KV=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_WAVES=2
export VLLM_ULTRAQUANT_PAIR_VERIFY=0
unset VLLM_ULTRAQUANT_PDV_EXPERIMENT

case "$MODE" in
  fp8)
    KV=fp8
    export VLLM_ULTRAQUANT_PDV_VERIFY=0
    ;;
  prod)
    KV=ultraquant_4bit
    export VLLM_ULTRAQUANT_PDV_VERIFY=0
    ;;
  pdv)
    KV=ultraquant_4bit
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
    ;;
  hoist_k)
    KV=ultraquant_4bit
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
    export VLLM_ULTRAQUANT_PDV_EXPERIMENT=hoist_k
    ;;
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

start_server() {
  local max_len="$1" tag="$2"
  cleanup_server
  local slog="$OUT/${tag}_server.log"
  log "SERVE mode=$MODE kv=$KV max_len=$max_len hip=$HIP_VISIBLE_DEVICES port=$PORT"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$KV" \
    --max-model-len "$max_len" --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --no-enable-log-requests \
    --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
    --language-model-only --reasoning-parser qwen3 --trust-remote-code \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
    >"$slog" 2>&1 &
  SERVER_PID=$!
  local ready=0 i
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL server exited"; cleanup_server; return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL server not ready"; cleanup_server; return 1
  fi
  if [ "$MODE" = "pdv" ] || [ "$MODE" = "hoist_k" ]; then
    if ! grep -q "PDV FlyDSL verify is enabled" "$slog"; then
      log "FAIL PDV verify line missing"; cleanup_server; return 1
    fi
  fi
  if [ "$MODE" = "hoist_k" ]; then
    if ! grep -q "Experiment=hoist_k" "$slog"; then
      log "FAIL hoist_k experiment tag missing"; cleanup_server; return 1
    fi
  fi
  if [ "$MODE" = "prod" ] || [ "$MODE" = "fp8" ]; then
    if grep -q "PDV FlyDSL verify is enabled" "$slog"; then
      log "FAIL unexpected PDV verify for mode=$MODE"; cleanup_server; return 1
    fi
  fi
  log "READY mode=$MODE max_len=$max_len"
}

run_gsm() {
  local out="$OUT/gsm8k_seed${SEED}"
  if [ -f "$out/summary.json" ]; then
    log "SKIP gsm8k (already present — delete to reforce)"
    return 0
  fi
  mkdir -p "$out"
  start_server 4096 gsm8k || return 1
  log "EVAL gsm8k mode=$MODE seed=$SEED thinking=$THINKING"
  python "$EVAL_HARNESS/drivers/gsm8k_native.py" \
    --port "$PORT" --model "$MODEL" --max_model_len 4096 \
    --bench_config "$GSM_CONFIG" --out_dir "$out" --seed "$SEED" \
    --enable_thinking "$THINKING" \
    --concurrency_override "$CONC" >"$out/eval.log" 2>&1 \
    || log "WARN gsm8k rc=$?"
  cleanup_server
  if [ -f "$out/summary.json" ]; then
    python -c "import json; d=json.load(open('$out/summary.json')); print('GSM8K', d.get('accuracy'), d.get('correct'), '/', d.get('total'))" \
      | sed "s/^/$(date -Is) /"
    log "END gsm8k"
  else
    log "FAIL gsm8k"
    return 1
  fi
}

run_gpqa() {
  local out="$OUT/gpqa_seed${SEED}"
  if [ -f "$out/summary.json" ]; then
    log "SKIP gpqa (already present — delete to reforce)"
    return 0
  fi
  mkdir -p "$out"
  start_server 262144 gpqa || return 1
  log "EVAL gpqa mode=$MODE seed=$SEED thinking=$THINKING"
  python "$EVAL_HARNESS/drivers/gpqa_native.py" \
    --port "$PORT" --model "$MODEL" --max_model_len 262144 \
    --bench_config "$GPQA_CONFIG" --out_dir "$out" --seed "$SEED" \
    --enable_thinking "$THINKING" \
    --concurrency_override "$CONC" >"$out/eval.log" 2>&1 \
    || log "WARN gpqa rc=$?"
  cleanup_server
  if [ -f "$out/summary.json" ]; then
    python -c "import json; d=json.load(open('$out/summary.json')); print('GPQA', d.get('accuracy'), d.get('correct'), '/', d.get('total'))" \
      | sed "s/^/$(date -Is) /"
    log "END gpqa"
  else
    log "FAIL gpqa"
    return 1
  fi
}

# Provenance stamp
cat >"$OUT/cohort_meta.json" <<EOF
{
  "mode": "$MODE",
  "seed": $SEED,
  "thinking": "$THINKING",
  "conc": $CONC,
  "mtp": 5,
  "prefix_caching": false,
  "gsm_max_model_len": 4096,
  "gpqa_max_model_len": 262144,
  "kv_cache_dtype": "$KV",
  "pdv_verify": "${VLLM_ULTRAQUANT_PDV_VERIFY}",
  "pdv_experiment": "${VLLM_ULTRAQUANT_PDV_EXPERIMENT:-}",
  "started_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

log "matched-acc start mode=$MODE hip=$HIP_VISIBLE_DEVICES port=$PORT"
run_gsm || true
run_gpqa || true
log "matched-acc DONE mode=$MODE"
