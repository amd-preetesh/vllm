#!/bin/bash
# Matched, fixed-length decode-heavy FP8 vs PDV serving profile.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/pdv-decode-heavy.log 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
OUT=$ROOT/out/pdv_decode_heavy
PORT="${PDV_DECODE_PORT:-8085}"
CONC=32
INPUT_LEN=32768
OUTPUT_LEN=1024
PROMPTS=96

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_ULTRAQUANT_FUSED_MTP=1
export VLLM_ULTRAQUANT_SHARED_KV=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_WAVES=2

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
  sleep 15
  # shellcheck disable=SC2086
  kill -9 $tree 2>/dev/null || true
}
trap cleanup_server EXIT

wait_for_free_memory() {
  local i free
  for i in $(seq 1 60); do
    free=$("$VENV/bin/python" - <<'PY' 2>/dev/null || echo 0
import torch
free, _ = torch.cuda.mem_get_info()
print(free // (1024**3))
PY
)
    log "FREE memory=${free}GiB"
    if [ "${free:-0}" -ge 270 ]; then
      return 0
    fi
    sleep 10
  done
  log "FAIL GPU memory did not return to 270GiB"
  return 1
}

sample_runtime() {
  local dest="$1" bench_pid="$2"
  : > "$dest"
  while kill -0 "$bench_pid" 2>/dev/null; do
    {
      echo "---- $(date -Is)"
      curl -s "http://127.0.0.1:$PORT/metrics" | grep -E \
        '^vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total|prompt_tokens_total|generation_tokens_total|num_preemptions_total)' \
        || true
      if [ -n "${PHYSICAL_GPU:-}" ]; then
        rocm-smi --showuse --showmemuse 2>/dev/null | grep "GPU\\[$PHYSICAL_GPU\\]" || true
      fi
    } >> "$dest"
    sleep 2
  done
}

run_mode() {
  local mode="$1" kv="$2" pdv="$3"
  local dir="$OUT/$mode"
  if [ -f "$dir/result.json" ]; then
    log "SKIP $mode already complete"
    return
  fi
  cleanup_server
  wait_for_free_memory || return 1
  mkdir -p "$dir"
  export VLLM_ULTRAQUANT_PDV_VERIFY="$pdv"
  log "SERVE $mode kv=$kv pdv=$pdv hip=$HIP_VISIBLE_DEVICES"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$kv" \
    --max-model-len 65536 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching --no-enable-log-requests \
    --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
    --language-model-only --reasoning-parser qwen3 --trust-remote-code \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
    > "$dir/server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0 i
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL $mode server exited"
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL $mode server not ready"
    return 1
  fi
  if [ "$pdv" = 1 ] && ! grep -q "PDV FlyDSL verify is enabled" "$dir/server.log"; then
    log "FAIL $mode missing PDV enable line"
    return 1
  fi
  log "BENCH $mode input=$INPUT_LEN output=$OUTPUT_LEN prompts=$PROMPTS conc=$CONC"
  vllm bench serve \
    --backend vllm --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0.0 --ignore-eos --num-prompts "$PROMPTS" \
    --max-concurrency "$CONC" --request-rate inf --seed 42 \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --metric-percentiles "50,90,99" \
    --save-result --result-dir "$dir" --result-filename result.json \
    > "$dir/benchmark.log" 2>&1 &
  local bench=$!
  sample_runtime "$dir/runtime_samples.txt" "$bench" &
  local sampler=$!
  wait "$bench"
  local rc=$?
  wait "$sampler" 2>/dev/null || true
  curl -s "http://127.0.0.1:$PORT/metrics" > "$dir/metrics_end.txt" || true
  log "END $mode rc=$rc"
  cleanup_server
  return "$rc"
}

log "pdv decode-heavy start pid=$$ ppid=$PPID hip=$HIP_VISIBLE_DEVICES physical=${PHYSICAL_GPU:-unknown}"
run_mode fp8 fp8 0 || log "WARN fp8 stopped"
run_mode pdv ultraquant_4bit 1 || log "WARN pdv stopped"
log "pdv decode-heavy complete"
touch /workspace/logs/pdv-decode-heavy.ok
