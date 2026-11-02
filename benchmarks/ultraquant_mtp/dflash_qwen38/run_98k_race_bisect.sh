#!/bin/bash
# Follow-up after 98k-bisect: serialize made 98304 PASS → suspect async race.
# Matrix: reconfirm cliff, serialize longer, disable async-scheduling, SPEC=0.
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/dflash-98k-race-bisect.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp/dflash_qwen38
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
HIP="${HIP_VISIBLE_DEVICES:?need HIP}"
PORT="${DFLASH_PORT:-8364}"
OUT_ROOT=/workspace/vllm/out/dflash_qwen38_fp8_agentx_debug

export PATH=/workspace/.venv-uq/bin:$PATH
export PYTHONPATH=/workspace/aiter:/workspace/vllm
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export VLLM_USE_V2_MODEL_RUNNER=1
export CUDA_VISIBLE_DEVICES="$HIP"
unset ROCR_VISIBLE_DEVICES

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
  local extra="${1:-}"
  [ -z "$extra" ] && extra="{}"
  DRAFT="$DRAFT" MAX_LEN=262144 EXTRA="$extra" python - <<'PY'
import json, os
cfg = json.load(open("/workspace/vllm/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2.json"))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = int(os.environ["MAX_LEN"])
extra = os.environ.get("EXTRA") or "{}"
if extra.strip() and extra.strip() != "{}":
    cfg.update(json.loads(extra))
print(json.dumps(cfg))
PY
}

start_serve() {
  local name="$1" mnbt="$2" noprefix="$3" serialize="$4" spec_extra="$5" no_async="$6" no_spec="$7"
  kill_port
  local out=$OUT_ROOT/race_${name}
  mkdir -p "$out"
  local args=(
    "$MODEL" --port "$PORT" --tensor-parallel-size 1
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8
    --max-model-len 262144 --gpu-memory-utilization 0.90
    --no-enable-log-requests --max-num-seqs 1
    --max-num-batched-tokens "$mnbt"
    --language-model-only --reasoning-parser qwen3 --trust-remote-code
    --enforce-eager
  )
  if [ "$no_spec" = 1 ]; then
    log "SERVE $name SPEC=0"
  else
    local spec
    spec=$(make_spec "$spec_extra")
    args+=(--speculative-config "$spec")
  fi
  if [ "$noprefix" = 1 ]; then
    args+=(--no-enable-prefix-caching)
  else
    args+=(--enable-prefix-caching)
  fi
  if [ "$no_async" = 1 ]; then
    args+=(--no-async-scheduling)
  fi
  if [ "$serialize" = 1 ]; then
    export AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1
  else
    unset AMD_SERIALIZE_KERNEL HIP_LAUNCH_BLOCKING || true
  fi
  log "SERVE $name mnbt=$mnbt noprefix=$noprefix serialize=$serialize no_async=$no_async no_spec=$no_spec extra=$spec_extra"
  vllm serve "${args[@]}" >"$out/server.log" 2>&1 &
  echo $! >"$out/server.pid"
  if ! wait_ready; then
    log "FAIL $name not ready"
    tail -n 80 "$out/server.log" || true
    kill_port
    return 1
  fi
  if grep -q "could be identified as the draft model's" "$out/server.log"; then
    log "WARN $name draft-KV warning STILL present"
  else
    log "OK $name no draft-KV warning"
  fi
  grep -E "async.scheduling|Async scheduling|Disabling fine-grained|kv cache group sizes|GPU KV cache size|mamba cache mode" "$out/server.log" \
    | sed "s/^/$(date -Is) NOTE $name /" || true
  log "READY $name"
  echo "$out" > /tmp/dflash_race_out.txt
}

run_probe() {
  local name="$1" lengths="$2"
  local plog=/workspace/logs/dflash-98k-race-${name}.log
  log "PROBE $name lengths=$lengths"
  python "$HERE/probe_long_prompt_acceptance.py" \
    --port "$PORT" --model "$MODEL" \
    --lengths "$lengths" --max-tokens 32 \
    >"$plog" 2>&1 || log "WARN probe $name rc=$?"
  log "PROBE_DONE $name"
  cat "$plog" || true
  local out
  out=$(cat /tmp/dflash_race_out.txt 2>/dev/null || true)
  if [ -n "$out" ] && grep -q "Memory access fault" "$out/server.log" 2>/dev/null; then
    log "FAULT $name memfault seen"
    grep -n "Memory access fault" "$out/server.log" | tail -3
  fi
}

log "98k-race-bisect start hip=$HIP port=$PORT"

# 1) Reconfirm cliff without serialize (single length near cliff)
start_serve "reconfirm_96k" 8192 0 0 "{}" 0 0 || exit 1
run_probe "reconfirm_96k" "96000"
kill_port; sleep 10

# 2) Serialize should pass 96k and stretch to 131k
start_serve "serialize_long" 8192 0 1 "{}" 0 0 || exit 1
run_probe "serialize_long" "96000,98304,110000,131072"
kill_port; sleep 10

# 3) Disable async scheduling (no kernel serialize) — if green, race is async
start_serve "no_async" 8192 0 0 "{}" 1 0 || exit 1
run_probe "no_async" "96000,98304,110000,131072"
kill_port; sleep 10

# 4) SPEC=0 + prefix (target/Mamba only) — isolate draft vs align
start_serve "spec0_prefix" 8192 0 0 "{}" 0 1 || exit 1
run_probe "spec0_prefix" "96000,98304,110000,131072"
kill_port

log "98k-race-bisect ALL DONE"
