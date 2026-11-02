#!/bin/bash
# AgentX debug launcher for Qwen3.8-27B + DFlash2 FP8 (Phase 1).
# Same server defaults as run_qwen38_dflash2_fp8.sh unless toggled via env.
#
# Env toggles (defaults match the failing AgentX run):
#   DFLASH_MAX_LEN=262144
#   DFLASH_CONC=32
#   DFLASH_MNBT=8192                 # max-num-batched-tokens
#   DFLASH_GPU_UTIL=0.90             # --gpu-memory-utilization
#   DFLASH_EAGER=0                   # 1 → --enforce-eager
#   DFLASH_NO_PREFIX_CACHE=0         # 1 → --no-enable-prefix-caching
#   DFLASH_SPEC=1                    # 0 → no speculative-config (target-only)
#   DFLASH_SERIALIZE=0               # 1 → AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1
#   DFLASH_MODE=sa                   # sa | serve  (serve = keep server up for probe)
#   DFLASH_WARMUP_GRACE=300
#   DFLASH_BENCH_DURATION=300
#   DFLASH_PORT / PDV_EXP_PORT
#
# New file only — does not edit shipping MTP/PDV/DFlash sources.
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/dflash-qwen38-fp8-agentx-debug.log
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
DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
HERE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38
SPEC_JSON=$HERE/configs/fp8_dflash2.json

DFLASH_MAX_LEN="${DFLASH_MAX_LEN:-262144}"
DFLASH_CONC="${DFLASH_CONC:-32}"
DFLASH_MNBT="${DFLASH_MNBT:-8192}"
DFLASH_GPU_UTIL="${DFLASH_GPU_UTIL:-0.90}"
DFLASH_EAGER="${DFLASH_EAGER:-0}"
DFLASH_NO_PREFIX_CACHE="${DFLASH_NO_PREFIX_CACHE:-0}"
DFLASH_SPEC="${DFLASH_SPEC:-1}"
DFLASH_SERIALIZE="${DFLASH_SERIALIZE:-0}"
DFLASH_MODE="${DFLASH_MODE:-sa}"
DFLASH_WARMUP_GRACE="${DFLASH_WARMUP_GRACE:-300}"
DFLASH_BENCH_DURATION="${DFLASH_BENCH_DURATION:-300}"

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

if [ "$DFLASH_SERIALIZE" = 1 ]; then
  export AMD_SERIALIZE_KERNEL=3
  export HIP_LAUNCH_BLOCKING=1
fi

case "$HIP_VISIBLE_DEVICES" in
  [0-7]) ;;
  *) echo "$(date -Is) FAIL refusing HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"; exit 1 ;;
esac

PORT="${DFLASH_PORT:-${PDV_EXP_PORT:-8361}}"
OUT=$ROOT/out/dflash_qwen38_fp8_agentx_debug
TAG="ml${DFLASH_MAX_LEN}_c${DFLASH_CONC}_mnbt${DFLASH_MNBT}_u${DFLASH_GPU_UTIL}"
[ "$DFLASH_EAGER" = 1 ] && TAG="${TAG}_eager"
[ "$DFLASH_NO_PREFIX_CACHE" = 1 ] && TAG="${TAG}_noprefix"
[ "$DFLASH_SPEC" = 0 ] && TAG="${TAG}_nospec"
[ "$DFLASH_SERIALIZE" = 1 ] && TAG="${TAG}_serialize"
SA=$OUT/$TAG
mkdir -p "$SA"
cd "$ROOT"
log() { echo "$(date -Is) $*"; }

make_spec() {
  local max_len="$1"
  DRAFT="$DRAFT" MAX_LEN="$max_len" python - <<'PY'
import json, os
cfg = json.load(open("/workspace/vllm/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2.json"))
cfg["model"] = os.environ["DRAFT"]
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
  local serve_args=(
    "$MODEL" --port "$PORT" --tensor-parallel-size 1
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8
    --max-model-len "$max_len" --gpu-memory-utilization "$DFLASH_GPU_UTIL"
    --no-enable-log-requests
    --max-num-seqs "$DFLASH_CONC" --max-num-batched-tokens "$DFLASH_MNBT"
    --language-model-only --reasoning-parser qwen3 --trust-remote-code
  )
  if [ "$DFLASH_NO_PREFIX_CACHE" = 1 ]; then
    serve_args+=(--no-enable-prefix-caching)
  else
    serve_args+=(--enable-prefix-caching)
  fi
  if [ "$DFLASH_EAGER" = 1 ]; then
    serve_args+=(--enforce-eager)
  fi
  if [ "$DFLASH_SPEC" = 1 ]; then
    local spec
    spec=$(make_spec "$max_len")
    serve_args+=(--speculative-config "$spec")
  fi
  log "SERVE debug tag=$TAG max_len=$max_len conc=$DFLASH_CONC mnbt=$DFLASH_MNBT eager=$DFLASH_EAGER noprefix=$DFLASH_NO_PREFIX_CACHE spec=$DFLASH_SPEC serialize=$DFLASH_SERIALIZE hip=$HIP_VISIBLE_DEVICES port=$PORT"
  vllm serve "${serve_args[@]}" >"$out/server.log" 2>&1 &
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
  if [ "$DFLASH_SPEC" = 1 ] && ! grep -Eiq 'dflash|DFlash2' "$out/server.log"; then
    log "WARN no DFlash string in server.log yet; continuing"
  fi
  log "READY debug"
}

run_sa() {
  local out=$SA
  if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "SKIP sa (artifact exists at $out)"
    return 0
  fi
  mkdir -p "$out"
  start_server "$out" "$DFLASH_MAX_LEN" || return 1
  log "SA START seed=42 conc=$DFLASH_CONC duration=$DFLASH_BENCH_DURATION"
  local extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')
  (cd "$INFERENCEX_DIR" && "$AIPERF" profile --scenario inferencex-agentx-mvp \
    --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
    --model "$MODEL" --concurrency "$DFLASH_CONC" --benchmark-duration "$DFLASH_BENCH_DURATION" \
    --unsafe-override \
    --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-grace-period "$DFLASH_WARMUP_GRACE" \
    --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
    --max-context-length "$DFLASH_MAX_LEN" \
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

run_serve_only() {
  mkdir -p "$SA"
  start_server "$SA" "$DFLASH_MAX_LEN" || return 1
  log "SERVE_ONLY ready — probe with probe_long_prompt_acceptance.py --port $PORT"
  # Hold until killed; EXIT trap cleans up.
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    sleep 30
  done
  log "SERVE_ONLY server exited"
}

log "dflash2_fp8 agentx-debug start pid=$$ mode=$DFLASH_MODE tag=$TAG hip=$HIP_VISIBLE_DEVICES port=$PORT"
ensure_draft || exit 1
case "$DFLASH_MODE" in
  sa) run_sa || true ;;
  serve) run_serve_only || true ;;
  *) log "FAIL unknown DFLASH_MODE=$DFLASH_MODE (want sa|serve)"; exit 1 ;;
esac
cleanup_server
log "dflash2_fp8 agentx-debug done tag=$TAG"
