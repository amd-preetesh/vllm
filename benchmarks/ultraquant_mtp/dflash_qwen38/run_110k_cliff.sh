#!/bin/bash
# Fine cliff between 98k OK and 110k FAIL under DFlash+prefix.
# One length per fresh serve to avoid cumulative pollution.
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/dflash-110k-cliff.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp/dflash_qwen38
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
HIP="${HIP_VISIBLE_DEVICES:?need HIP}"
PORT="${DFLASH_PORT:-8364}"
OUT_ROOT=/workspace/vllm/out/dflash_qwen38_fp8_agentx_debug
LENGTHS="${DFLASH_LENGTHS:-100000,104000,106000,108000,109000,110000}"

export PATH=/workspace/.venv-uq/bin:$PATH
export PYTHONPATH=/workspace/aiter:/workspace/vllm
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export VLLM_USE_V2_MODEL_RUNNER=1
export CUDA_VISIBLE_DEVICES="$HIP"
unset ROCR_VISIBLE_DEVICES
unset AMD_SERIALIZE_KERNEL HIP_LAUNCH_BLOCKING || true

log() { echo "$(date -Is) $*"; }

kill_port() {
  local pids
  pids=$(ps aux | awk -v p="--port ${PORT}" '$0 ~ p && $0 !~ /awk/ {print $2}')
  [ -n "$pids" ] && kill $pids 2>/dev/null || true
  sleep 8
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
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

make_spec() {
  DRAFT="$DRAFT" MAX_LEN=262144 python - <<'PY'
import json, os
cfg = json.load(open("/workspace/vllm/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2.json"))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = int(os.environ["MAX_LEN"])
print(json.dumps(cfg))
PY
}

start_serve() {
  local name="$1"
  kill_port
  local out=$OUT_ROOT/cliff_${name}
  mkdir -p "$out"
  local spec
  spec=$(make_spec)
  log "SERVE $name"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8 \
    --max-model-len 262144 --gpu-memory-utilization 0.90 \
    --no-enable-log-requests --max-num-seqs 1 \
    --max-num-batched-tokens 8192 \
    --language-model-only --reasoning-parser qwen3 --trust-remote-code \
    --enforce-eager \
    --speculative-config "$spec" \
    --enable-prefix-caching \
    >"$out/server.log" 2>&1 &
  echo $! >"$out/server.pid"
  if ! wait_ready; then
    log "FAIL $name not ready"
    tail -n 50 "$out/server.log" || true
    kill_port
    return 1
  fi
  log "READY $name"
  echo "$out" > /tmp/dflash_cliff_out.txt
}

run_one() {
  local len="$1"
  local name="len_${len}"
  start_serve "$name" || return 1
  local plog=/workspace/logs/dflash-110k-cliff-${len}.log
  log "PROBE $len"
  python "$HERE/probe_long_prompt_acceptance.py" \
    --port "$PORT" --model "$MODEL" \
    --lengths "$len" --max-tokens 32 \
    >"$plog" 2>&1 || log "WARN probe $len rc=$?"
  cat "$plog" || true
  local out
  out=$(cat /tmp/dflash_cliff_out.txt 2>/dev/null || true)
  if [ -n "$out" ] && grep -q "Memory access fault" "$out/server.log" 2>/dev/null; then
    log "FAULT $len memfault"
    grep -n "Memory access fault" "$out/server.log" | tail -2
  elif grep -q "text='OK'" "$plog" 2>/dev/null; then
    log "PASS $len"
  else
    log "FAIL $len see $plog"
  fi
  kill_port
  sleep 8
}

log "110k-cliff start hip=$HIP port=$PORT lengths=$LENGTHS"
IFS=',' read -ra LS <<< "$LENGTHS"
for len in "${LS[@]}"; do
  run_one "$len"
done
log "110k-cliff ALL DONE"
