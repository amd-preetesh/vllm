#!/bin/bash
# Narrow long+prefix repro for DFlash2.
# Separates cold prefill vs deliberate prefix hit; finds fail length.
#
# Env:
#   HIP_VISIBLE_DEVICES (required)
#   DFLASH_PORT (default 8363)
#   DFLASH_LENGTHS (default 49152,65536,81920,98304)
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/dflash-prefix-repro.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp/dflash_qwen38
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
HIP="${HIP_VISIBLE_DEVICES:?need HIP}"
PORT="${DFLASH_PORT:-8363}"
LENGTHS="${DFLASH_LENGTHS:-49152,65536,81920,98304}"

log() { echo "$(date -Is) $*"; }

kill_port() {
  local pids
  pids=$(ps aux | awk -v p="--port ${PORT}" '$0 ~ p && $0 !~ /awk/ {print $2}')
  [ -n "$pids" ] && kill $pids 2>/dev/null || true
  sleep 5
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
  pkill -f "run_qwen38_dflash2_fp8_agentx_debug.sh" 2>/dev/null || true
  sleep 2
}

wait_ready() {
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" | grep -q "Qwen3.8-27B"; then
      return 0
    fi
    sleep 5
  done
  return 1
}

start_serve() {
  local tag="$1"
  shift
  kill_port
  (
    export HIP_VISIBLE_DEVICES="$HIP"
    export CUDA_VISIBLE_DEVICES="$HIP"
    unset ROCR_VISIBLE_DEVICES
    export DFLASH_PORT="$PORT"
    export DFLASH_CONC=1
    export DFLASH_EAGER=1
    export DFLASH_MAX_LEN=262144
    export DFLASH_MNBT=8192
    export DFLASH_GPU_UTIL=0.90
    export DFLASH_MODE=serve
    export DFLASH_SPEC=1
    export DFLASH_SERIALIZE=0
    export DFLASH_NO_PREFIX_CACHE=0
    for kv in "$@"; do export "$kv"; done
    bash "$HERE/run_qwen38_dflash2_fp8_agentx_debug.sh"
  ) &
  if ! wait_ready; then
    log "FAIL $tag not ready"
    kill_port
    return 1
  fi
  log "READY $tag"
}

run_probe() {
  local name="$1" lengths="$2" extra="$3"
  local plog=/workspace/logs/dflash-prefix-repro-${name}.log
  log "PROBE $name lengths=$lengths extra=$extra"
  PATH=/workspace/.venv-uq/bin:$PATH \
  PYTHONPATH=/workspace/aiter:/workspace/vllm \
  PYTHONNOUSERSITE=1 \
  python "$HERE/probe_long_prompt_acceptance.py" \
    --port "$PORT" --model "$MODEL" \
    --lengths "$lengths" --max-tokens 64 \
    $extra >"$plog" 2>&1 || log "WARN probe $name rc=$?"
  log "PROBE_DONE $name"
  cat "$plog" || true
}

log "prefix-repro start hip=$HIP port=$PORT lengths=$LENGTHS"

# A: prefix ON, cold only (no repeat) — does first long request alone fault?
start_serve "prefix_cold" DFLASH_NO_PREFIX_CACHE=0 || exit 1
run_probe "prefix_cold" "$LENGTHS" ""
kill_port
sleep 8

# B: prefix ON, each length twice — second should hit prefix cache
start_serve "prefix_hit" DFLASH_NO_PREFIX_CACHE=0 || exit 1
run_probe "prefix_hit" "$LENGTHS" "--repeat-for-prefix-hit"
kill_port
sleep 8

# C: control — noprefix, with repeat (should stay healthy)
start_serve "noprefix_ctrl" DFLASH_NO_PREFIX_CACHE=1 || exit 1
run_probe "noprefix_ctrl" "$LENGTHS" "--repeat-for-prefix-hit"
kill_port

log "prefix-repro ALL DONE"
