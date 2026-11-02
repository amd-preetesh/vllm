#!/bin/bash
# Decode-focused fixed workload: ISL 1024 · OSL 8192 · 96 prompts · ignore-eos · seed 42
# MODE=fp8|pdv|prod|hoist_k|qblock_waves  CONC=8|16|32|36|40
# prod = UltraQuant production FlyDSL (PDV off, pair off) — baseline UQ verify
trap '' HUP
mkdir -p /workspace/logs
MODE="${MODE:?need MODE=fp8|pdv|prod|hoist_k|qblock_waves}"
CONC="${CONC:-32}"
case "$MODE" in
  fp8|pdv|prod|hoist_k|qblock_waves) ;;
  *) echo "$(date -Is) FAIL bad MODE=$MODE"; exit 1 ;;
esac
case "$CONC" in
  8|16|32|36|40) ;;
  *) echo "$(date -Is) FAIL bad CONC=$CONC"; exit 1 ;;
esac
exec >> "/workspace/logs/pdv-decode-i1k-o8k-${MODE}-c${CONC}.log" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
OUT=$ROOT/out/pdv_decode_i1k_o8k/${MODE}_c${CONC}
PORT="${PDV_DECODE_PORT:?need PDV_DECODE_PORT}"
INPUT_LEN=1024
OUTPUT_LEN=8192
PROMPTS=96
MAX_LEN=16384

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
unset VLLM_ULTRAQUANT_PDV_EXPERIMENT

case "$MODE" in
  fp8)
    KV=fp8
    export VLLM_ULTRAQUANT_PDV_VERIFY=0
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    ;;
  pdv)
    KV=ultraquant_4bit
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    ;;
  prod)
    # Production FlyDSL: one token per wave, not PDV fused verify.
    KV=ultraquant_4bit
    export VLLM_ULTRAQUANT_PDV_VERIFY=0
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
    ;;
  hoist_k|qblock_waves)
    KV=ultraquant_4bit
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    export VLLM_ULTRAQUANT_PDV_EXPERIMENT="$MODE"
    ;;
esac

mkdir -p "$OUT"
cd "$ROOT"
log() { echo "$(date -Is) $*"; }

# Legacy c=32 path from earlier runs
if [ -f "$ROOT/out/pdv_decode_i1k_o8k/$MODE/result.json" ] && [ ! -f "$OUT/result.json" ] && [ "$CONC" = 32 ]; then
  ln -sfn "$ROOT/out/pdv_decode_i1k_o8k/$MODE/result.json" "$OUT/result.json" || \
    cp "$ROOT/out/pdv_decode_i1k_o8k/$MODE/result.json" "$OUT/result.json"
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
    if [ "${free:-0}" -ge 250 ]; then
      return 0
    fi
    sleep 10
  done
  log "FAIL GPU memory did not return to 250GiB"
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
    } >> "$dest"
    sleep 2
  done
}

if [ -f "$OUT/result.json" ]; then
  log "SKIP $MODE c=$CONC already complete"
  exit 0
fi

cleanup_server
wait_for_free_memory || exit 1
log "SERVE mode=$MODE conc=$CONC kv=$KV hip=$HIP_VISIBLE_DEVICES port=$PORT exp=${VLLM_ULTRAQUANT_PDV_EXPERIMENT:-shipping}"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$KV" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --no-enable-log-requests \
  --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
ready=0
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "FAIL server exited"
    exit 1
  fi
  sleep 5
done
if [ "$ready" != 1 ]; then
  log "FAIL server not ready"
  exit 1
fi
case "$MODE" in
  fp8) ;;
  prod)
    if grep -q "PDV FlyDSL verify is enabled" "$OUT/server.log"; then
      log "FAIL prod mode unexpectedly enabled PDV"
      exit 1
    fi
    ;;
  *)
    if ! grep -q "PDV FlyDSL verify is enabled" "$OUT/server.log"; then
      log "FAIL missing PDV enable line"
      exit 1
    fi
    ;;
esac
log "BENCH mode=$MODE conc=$CONC input=$INPUT_LEN output=$OUTPUT_LEN prompts=$PROMPTS"
vllm bench serve \
  --backend vllm --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" --dataset-name random \
  --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --random-range-ratio 0.0 --ignore-eos --num-prompts "$PROMPTS" \
  --max-concurrency "$CONC" --request-rate inf --seed 42 \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --metric-percentiles "50,90,99" \
  --save-result --result-dir "$OUT" --result-filename result.json \
  > "$OUT/benchmark.log" 2>&1 &
bench=$!
sample_runtime "$OUT/runtime_samples.txt" "$bench" &
sampler=$!
wait "$bench"
rc=$?
wait "$sampler" 2>/dev/null || true
curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics_end.txt" || true
printf '%s\n' "mode=$MODE conc=$CONC isl=$INPUT_LEN osl=$OUTPUT_LEN prompts=$PROMPTS seed=42" > "$OUT/cell_meta.txt"
log "END mode=$MODE conc=$CONC rc=$rc"
cleanup_server
exit "$rc"
