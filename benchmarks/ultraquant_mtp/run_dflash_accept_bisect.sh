#!/bin/bash
# P0-2: DFlash accept-vs-length on windowed draft vs no-window copy.
# Requires idle HIP. Writes out/dflash_accept_bisect/{window,nowin}/
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/dflash-accept-bisect.log
exec >>"$LOG" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DRAFT_WIN="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
DRAFT_NOWIN="${DFLASH_DRAFT_NOWIN:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2-nowin}"
SPEC_BASE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2.json
PROBE=$ROOT/benchmarks/ultraquant_mtp/docs/probe_accept_vs_length.py
OUT=$ROOT/out/dflash_accept_bisect
BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-64}"
CTX="${PROBE_CTX:-1024,32768,131072}"
GEN="${PROBE_GEN:-512,1536,2560,4096}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP device}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_USE_V2_MODEL_RUNNER=1

PORT="${PORT:?need PORT}"
MAX_LEN=262144
log() { echo "$(date -Is) $*"; }
mkdir -p "$OUT"

if [ ! -f "$DRAFT_NOWIN/config.json" ]; then
  log "FAIL missing nowin draft at $DRAFT_NOWIN"
  exit 1
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
  sleep 12
  # shellcheck disable=SC2086
  kill -9 $tree 2>/dev/null || true
}
trap cleanup_server EXIT

serve_variant() {
  local name="$1" draft="$2"
  local cell=$OUT/$name
  mkdir -p "$cell"
  if [ -f "$cell/accept_vs_length.jsonl" ]; then
    log "SKIP probe $name"
    return 0
  fi
  cleanup_server
  local spec
  spec=$(DRAFT="$draft" MAX_LEN="$MAX_LEN" SPEC_JSON="$SPEC_BASE" python - <<'PY'
import json, os
cfg = json.load(open(os.environ["SPEC_JSON"]))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = int(os.environ["MAX_LEN"])
print(json.dumps(cfg))
PY
)
  log "SERVE $name draft=$draft block=$BLOCK_SIZE port=$PORT"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8 \
    --block-size "$BLOCK_SIZE" \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.90 \
    --enable-prefix-caching --no-enable-log-requests \
    --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --language-model-only --reasoning-parser qwen3 --trust-remote-code \
    --speculative-config "$spec" \
    >"$cell/server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0 i
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then ready=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL $name server"; tail -n 40 "$cell/server.log"; return 1
    fi
    sleep 5
  done
  [ "$ready" = 1 ] || { log "FAIL $name not ready"; return 1; }
  log "PROBE $name ctx=$CTX gen=$GEN"
  python "$PROBE" --port "$PORT" --model "$MODEL" --ctx "$CTX" --gen "$GEN" \
    | tee "$cell/accept_vs_length.jsonl"
  cleanup_server
  SERVER_PID=
  log "END probe $name"
}

log "accept bisect start hip=$HIP_VISIBLE_DEVICES port=$PORT"
serve_variant window "$DRAFT_WIN" || true
serve_variant nowin "$DRAFT_NOWIN" || true
log "accept bisect done → $OUT"
