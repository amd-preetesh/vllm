#!/bin/bash
# Qwen3.8-27B + DFlash2 on UltraQuant + DFlash FlyDSL (Phase 3).
# Target: ULTRAQUANT_DFLASH + ultraquant_4bit (dflash_flydsl experiment)
# Draft:  ROCM_AITER_FA + auto (non-causal), K=7
# Runner: VLLM_USE_V2_MODEL_RUNNER=1
# New file only — does not edit shipping MTP/PDV/DFlash sources.
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/dflash-qwen38-flydsl.log
exec >>"$LOG" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
EVAL_HARNESS=/workspace/eval-harness
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
# HF tree is often RO in-container; default draft under /workspace.
DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
HERE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38
SPEC_JSON=$HERE/configs/uq_dflash2.json
GSM_CONFIG=$EVAL_HARNESS/configs/benchmarks/gsm8k_ffdsl.json
GPQA_CONFIG=$EVAL_HARNESS/configs/benchmarks/gpqa_diamond_qwen38_card.json

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
# P2: shipping PDV on UltraQuant (leave experiment unset = shipping kernel).
export VLLM_ULTRAQUANT_FUSED_MTP=1
export VLLM_ULTRAQUANT_SHARED_KV=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_WAVES=2
export VLLM_ULTRAQUANT_PAIR_VERIFY=0
export VLLM_ULTRAQUANT_PDV_VERIFY=1
export VLLM_ULTRAQUANT_PDV_EXPERIMENT=dflash

case "$HIP_VISIBLE_DEVICES" in
  [0-7]) ;;
  *) echo "$(date -Is) FAIL refusing HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"; exit 1 ;;
esac

PORT="${DFLASH_PORT:-${PDV_EXP_PORT:-8371}}"
CONC=32
OUT=$ROOT/out/dflash_qwen38_flydsl
ACC=$OUT/accuracy
SA=$OUT/sa
mkdir -p "$ACC" "$SA"
cd "$ROOT"
log() { echo "$(date -Is) $*"; }

make_spec() {
  local max_len="$1"
  DRAFT="$DRAFT" MAX_LEN="$max_len" python - <<'PY'
import json, os
cfg = json.load(open("/workspace/vllm/benchmarks/ultraquant_mtp/dflash_qwen38/configs/uq_dflash2.json"))
cfg["model"] = os.environ["DRAFT"]
# Draft max_model_len must be <= target max_model_len.
cfg["max_model_len"] = int(os.environ["MAX_LEN"])
print(json.dumps(cfg))
PY
}

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

ensure_draft() {
  if [ -f "$DRAFT/config.json" ]; then
    log "draft present $DRAFT"
    return 0
  fi
  log "fetching draft → $DRAFT"
  DFLASH_DRAFT_DIR="$DRAFT" bash "$HERE/fetch_draft.sh" || {
    log "FAIL draft fetch"
    return 1
  }
}

start_server() {
  local out="$1" max_len="$2"
  cleanup_server
  mkdir -p "$out"
  local spec
  spec=$(make_spec "$max_len")
  log "SERVE dflash2_flydsl max_len=$max_len hip=$HIP_VISIBLE_DEVICES port=$PORT"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend ULTRAQUANT_DFLASH --kv-cache-dtype ultraquant_4bit \
    --max-model-len "$max_len" --gpu-memory-utilization "${DFLASH_GPU_UTIL:-0.90}" \
    --enable-prefix-caching --no-enable-log-requests \
    --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
    --language-model-only --reasoning-parser qwen3 --trust-remote-code \
    --speculative-config "$spec" \
    >"$out/server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0 i
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL server exited"
      tail -n 80 "$out/server.log" || true
      cleanup_server
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL server not ready"
    tail -n 80 "$out/server.log" || true
    cleanup_server
    return 1
  fi
  # Prefer DFlash markers; accept either method string or draft class.
  if ! grep -Eiq 'dflash|DFlash2' "$out/server.log"; then
    log "WARN no DFlash string in server.log yet; continuing"
  fi
  log "READY dflash2_flydsl"
}

run_smoke() {
  local marker=$OUT/smoke_ok.txt
  if [ -f "$marker" ]; then
    log "SKIP smoke"
    return 0
  fi
  mkdir -p "$OUT/smoke"
  start_server "$OUT/smoke" 8192 || return 1
  log "SMOKE chat"
  python "$HERE/bench_dflash_smoke.py" --port "$PORT" --model "$MODEL" \
    >"$OUT/smoke/smoke.log" 2>&1 || {
    log "FAIL smoke"
    return 1
  }
  echo ok >"$marker"
  log "SMOKE ok"
  cleanup_server
}

run_gsm() {
  local out=$ACC/gsm8k_seed7
  if [ -f "$out/summary.json" ]; then
    log "SKIP gsm8k"
    return 0
  fi
  mkdir -p "$out"
  start_server "$out" 4096 || return 1
  log "EVAL gsm8k seed=7"
  python "$EVAL_HARNESS/drivers/gsm8k_native.py" \
    --port "$PORT" --model "$MODEL" --max_model_len 4096 \
    --bench_config "$GSM_CONFIG" --out_dir "$out" --seed 7 \
    --concurrency_override "$CONC" >"$out/eval.log" 2>&1 || log "WARN gsm8k"
  cleanup_server
  if [ -f "$out/summary.json" ]; then log "END gsm8k"; else log "FAIL gsm8k"; fi
}

run_gpqa() {
  local out=$ACC/gpqa_seed7
  if [ -f "$out/summary.json" ]; then
    log "SKIP gpqa"
    return 0
  fi
  mkdir -p "$out"
  start_server "$out" 262144 || return 1
  log "EVAL gpqa seed=7"
  python "$EVAL_HARNESS/drivers/gpqa_native.py" \
    --port "$PORT" --model "$MODEL" --max_model_len 262144 \
    --bench_config "$GPQA_CONFIG" --out_dir "$out" --seed 7 \
    --concurrency_override "$CONC" >"$out/eval.log" 2>&1 || log "WARN gpqa"
  cleanup_server
  if [ -f "$out/summary.json" ]; then log "END gpqa"; else log "FAIL gpqa"; fi
}

run_sa() {
  local out=$SA/seed42_c32
  if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "SKIP sa"
    return 0
  fi
  mkdir -p "$out"
  start_server "$out" 262144 || return 1
  log "SA START seed=42 conc=$CONC"
  local extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')
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
    "${extra[@]}") >"$out/benchmark.log" 2>&1 || log "WARN sa rc=$?"
  cleanup_server
  if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "END sa"
  else
    log "FAIL sa"
  fi
}

log "dflash2_flydsl start pid=$$ hip=$HIP_VISIBLE_DEVICES port=$PORT"
ensure_draft || exit 1
run_smoke || exit 1
if [ "${DFLASH_SMOKE_ONLY:-0}" = 1 ]; then
  cleanup_server
  log "dflash2_flydsl smoke-only done"
  exit 0
fi
run_gsm || true
run_gpqa || true
run_sa || true
cleanup_server
log "dflash2_flydsl done"
