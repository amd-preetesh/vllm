#!/bin/bash
# P0-1: profile one DSpark UQ decode-heavy window (rocprofv3 kernel stats).
# No production kernel edits. Compares against MTP UQ microbench when present.
#
#   HIP_VISIBLE_DEVICES=N PORT=8470 \
#     bash benchmarks/ultraquant_mtp/run_uq_decode_rocprof_p01.sh
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-p01-decode-rocprof.log
exec >>"$LOG" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp
OUT=$ROOT/out/uq_p01_decode_rocprof
VENV=/workspace/.venv-uq
AITER_SRC=/workspace/aiter
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DRAFT=/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark
SPEC=$HERE/dspark_qwen38_uq/configs/uq_dspark.json
PORT="${PORT:-8470}"
UTIL="${UTIL:-0.70}"
CTX_TOKENS="${CTX_TOKENS:-131072}"
GEN_TOKENS="${GEN_TOKENS:-256}"

export PATH="$VENV/bin:/opt/rocm/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP device}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ULTRAQUANT_FUSED_MTP=1
export VLLM_ULTRAQUANT_SHARED_KV=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_PDV_VERIFY=1
unset VLLM_ULTRAQUANT_MIXED_VERIFY
unset VLLM_DFLASH_DRAFT_ATTN_EXPERIMENT
unset VLLM_ULTRAQUANT_PDV_EXPERIMENT

log() { echo "$(date -Is) $*"; }
mkdir -p "$OUT"
cd "$ROOT"

SERVER_PID=
cleanup() {
  [ -n "${SERVER_PID:-}" ] || return 0
  local pid="$SERVER_PID" c tree
  SERVER_PID=
  tree="$pid"
  for c in $(pgrep -P "$pid" 2>/dev/null || true); do
    tree="$tree $c $(pgrep -P "$c" 2>/dev/null || true)"
  done
  # shellcheck disable=SC2086
  kill $tree 2>/dev/null || true
  sleep 10
  # shellcheck disable=SC2086
  kill -9 $tree 2>/dev/null || true
}
trap cleanup EXIT

log "P0-1 start hip=$HIP_VISIBLE_DEVICES port=$PORT ctx=$CTX_TOKENS gen=$GEN_TOKENS"

SPEC_JSON=$(SPEC="$SPEC" DRAFT="$DRAFT" python - <<'PY'
import json, os
cfg = json.load(open(os.environ["SPEC"]))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = 262144
print(json.dumps(cfg))
PY
)

log "launch DSpark UQ server (ret50k-like: retention 50176, block 64)"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend ULTRAQUANT_SPEC_COMPAT --kv-cache-dtype ultraquant_4bit \
  --block-size 64 --max-model-len 262144 --gpu-memory-utilization "$UTIL" \
  --enable-prefix-caching --no-enable-log-requests \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  --prefix-cache-retention-interval 50176 \
  --speculative-config "$SPEC_JSON" \
  >"$OUT/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "FAIL server exited"
    tail -n 80 "$OUT/server.log" || true
    exit 1
  fi
  sleep 5
done
[ "$ready" = 1 ] || { log "FAIL not ready"; exit 1; }
log "server ready"

# Warm long context (prefill), then profile a decode-only completion.
python - <<PY
import json, urllib.request
port = int("$PORT")
model = "$MODEL"
ctx = int("$CTX_TOKENS")
filler = ("The quick brown fox jumps over the lazy dog. " * 200)
# Approximate token count via chars/4; enough for long ctx warmup.
prompt = (filler * ((ctx // 200) + 1))[: ctx * 4]
body = dict(model=model, prompt=prompt, max_tokens=8, temperature=0.0, ignore_eos=True)
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
print("warmup start chars", len(prompt))
urllib.request.urlopen(req, timeout=7200).read()
print("warmup done")
open("$OUT/warmup_done", "w").write("ok\n")
PY
log "warmup done"

log "rocprofv3 around decode-only gen=$GEN_TOKENS"
timeout -k 30 900 rocprofv3 \
  --kernel-trace --stats --truncate-kernels \
  -d "$OUT" -o p01_dspark_uq_decode -f csv json \
  -- python - <<PY
import json, urllib.request
port = int("$PORT")
model = "$MODEL"
gen = int("$GEN_TOKENS")
# Short prompt; prefix cache should hit the warmed trajectory when possible.
# Even on miss, gen tokens are decode-dominated after the first chunk.
prompt = "Continue the previous analysis with concrete next steps.\n" * 32
body = dict(model=model, prompt=prompt, max_tokens=gen, temperature=0.0, ignore_eos=True)
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
out = json.loads(urllib.request.urlopen(req, timeout=7200).read())
print("decode done usage", out.get("usage"))
open("$OUT/decode_usage.json", "w").write(json.dumps(out.get("usage"), indent=2))
PY
rc=$?
log "rocprof rc=$rc"

if [ -f "$OUT/p01_dspark_uq_decode_kernel_stats.csv" ]; then
  head -n 40 "$OUT/p01_dspark_uq_decode_kernel_stats.csv" >"$OUT/top_kernels.txt"
  log "top kernels → $OUT/top_kernels.txt"
  # Highlight PDV / FA / gemm rows
  grep -Ei 'ultraquant|pdv|flash|gemm|mamba|conv|aiter' \
    "$OUT/p01_dspark_uq_decode_kernel_stats.csv" \
    | head -n 60 >"$OUT/attn_related_kernels.txt" || true
fi

curl -s "http://127.0.0.1:$PORT/metrics" \
  | grep -E '^vllm:(spec_decode|prefix_cache|generation_tokens|prompt_tokens)' \
  >"$OUT/metrics_end.txt" || true

cleanup
SERVER_PID=
log "P0-1 done → $OUT"
touch "$OUT/DONE"
