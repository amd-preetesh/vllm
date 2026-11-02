#!/bin/bash
# Sequential toggle bisect at 64k–131k (diagnosis plan step 2).
# Runs inside preshah_vllm_ultraquant. One config at a time on HIP_VISIBLE_DEVICES.
#
# Logs: /workspace/logs/dflash-toggle-bisect.log
# Per-config probe: /workspace/logs/dflash-toggle-<tag>.log
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/dflash-toggle-bisect.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp/dflash_qwen38
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
PORT_BASE="${DFLASH_PORT_BASE:-8361}"
HIP="${HIP_VISIBLE_DEVICES:?need HIP_VISIBLE_DEVICES}"
LENGTHS="${DFLASH_BISECT_LENGTHS:-65536,98304,131072}"

log() { echo "$(date -Is) $*"; }

kill_port() {
  local port="$1"
  local pids
  pids=$(ps aux | awk -v p="--port ${port}" '$0 ~ p && $0 !~ /awk/ {print $2}')
  if [ -n "$pids" ]; then
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 5
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi
  # also kill leftover debug wrappers for this port
  pkill -f "run_qwen38_dflash2_fp8_agentx_debug.sh" 2>/dev/null || true
  sleep 2
}

wait_ready() {
  local port="$1" deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" | grep -q "Qwen3.8-27B"; then
      return 0
    fi
    sleep 5
  done
  return 1
}

run_one() {
  local name="$1"
  shift
  # remaining args are KEY=VAL exports
  local port=$PORT_BASE
  local tag_env=()
  local kv
  for kv in "$@"; do
    tag_env+=("$kv")
  done

  log "===== CONFIG $name ====="
  kill_port "$port"

  (
    export HIP_VISIBLE_DEVICES="$HIP"
    export CUDA_VISIBLE_DEVICES="$HIP"
    unset ROCR_VISIBLE_DEVICES
    export DFLASH_PORT="$port"
    export DFLASH_CONC=1
    export DFLASH_EAGER=1
    export DFLASH_MAX_LEN=262144
    export DFLASH_GPU_UTIL=0.90
    export DFLASH_MODE=serve
    # defaults then overrides
    export DFLASH_SPEC=1
    export DFLASH_NO_PREFIX_CACHE=0
    export DFLASH_MNBT=8192
    export DFLASH_SERIALIZE=0
    for kv in "${tag_env[@]}"; do
      export "$kv"
    done
    bash "$HERE/run_qwen38_dflash2_fp8_agentx_debug.sh"
  ) &
  local serve_wrap=$!

  if ! wait_ready "$port"; then
    log "FAIL $name server not ready"
    kill_port "$port"
    wait "$serve_wrap" 2>/dev/null || true
    return 1
  fi
  log "READY $name — probing $LENGTHS"

  local probe_log=/workspace/logs/dflash-toggle-${name}.log
  PATH=/workspace/.venv-uq/bin:$PATH \
  PYTHONPATH=/workspace/aiter:/workspace/vllm \
  PYTHONNOUSERSITE=1 \
  python "$HERE/probe_long_prompt_acceptance.py" \
    --port "$port" \
    --model "$MODEL" \
    --lengths "$LENGTHS" \
    --max-tokens 256 \
    --repeat-for-prefix-hit \
    >"$probe_log" 2>&1 || log "WARN probe rc=$?"

  log "PROBE_DONE $name"
  cat "$probe_log" || true

  kill_port "$port"
  wait "$serve_wrap" 2>/dev/null || true
  sleep 10
  log "===== END $name ====="
}

log "toggle-bisect start hip=$HIP lengths=$LENGTHS"
# Order: isolate draft vs target first, then prefix/chunking, then serialize for stack.
run_one "spec0"            DFLASH_SPEC=0
run_one "noprefix"         DFLASH_NO_PREFIX_CACHE=1
run_one "mnbt32k"          DFLASH_MNBT=32768
run_one "serialize"        DFLASH_SERIALIZE=1
run_one "noprefix_mnbt32k" DFLASH_NO_PREFIX_CACHE=1 DFLASH_MNBT=32768
log "toggle-bisect ALL DONE"
