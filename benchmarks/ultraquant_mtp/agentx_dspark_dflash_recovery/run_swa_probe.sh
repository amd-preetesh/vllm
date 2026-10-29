#!/bin/bash
# Phase 2b: serve FP8 DFlash with SWA win_table and run accept-vs-length probe.
#   HIP_VISIBLE_DEVICES=3 PORT=8730 bash .../run_swa_probe.sh
trap '' HUP
set -uo pipefail

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DRAFT=/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2
PORT="${PORT:?need PORT}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need HIP}"
OUT=$ROOT/out/agentx_dspark_dflash_recovery/swa_probe
mkdir -p "$OUT" /workspace/logs/agentx_dspark_dflash_recovery
LOG=/workspace/logs/agentx_dspark_dflash_recovery/swa_probe.log
exec >>"$LOG" 2>&1
log() { echo "$(date -Is) probe $*"; }

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_DFLASH_DRAFT_ATTN_EXPERIMENT=win_table
export VLLM_DFLASH_DEBUG_NULL_BLOCK=1

SPEC=$(python - <<PY
import json
print(json.dumps({
  "method": "dflash",
  "model": "$DRAFT",
  "num_speculative_tokens": 7,
  "attention_backend": "ROCM_AITER_FA_DFLASH_SWA",
  "kv_cache_dtype": "fp8",
  "max_model_len": 131072,
}))
PY
)

SERVER_PID=
cleanup() {
  [ -n "${SERVER_PID:-}" ] || return 0
  kill "$SERVER_PID" 2>/dev/null || true
  sleep 5
  kill -9 "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

if command -v fuser >/dev/null 2>&1; then fuser -k "${PORT}/tcp" 2>/dev/null || true; fi

log "START serve hip=$HIP_VISIBLE_DEVICES port=$PORT"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend ROCM_AITER_FA --kv-cache-dtype fp8 \
  --block-size 64 --max-model-len 131072 --gpu-memory-utilization 0.45 \
  --enable-prefix-caching --no-enable-log-requests \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  --speculative-config "$SPEC" >"$OUT/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then log "FAIL server"; tail -n 80 "$OUT/server.log"; exit 1; fi
  sleep 5
done
[ "$ready" = 1 ] || { log "FAIL not ready"; exit 1; }
log "READY — running probe"

python3 "$ROOT/benchmarks/ultraquant_mtp/docs/probe_accept_vs_length.py" \
  --port "$PORT" --model "$MODEL" \
  --ctx 1024,32768,131072 --gen 512,1536,2560,4096 \
  | tee "$OUT/probe_accept_vs_length.txt"

# Gate: at ctx=131072 gen=4096, accept_len should stay >= 2.5 if SWA fix works
python3 - <<'PY' | tee "$OUT/probe_gate.txt"
from pathlib import Path
import re
text = Path("/workspace/vllm/out/agentx_dspark_dflash_recovery/swa_probe/probe_accept_vs_length.txt").read_text()
# lines look like: ctx=131072 gen=4096 accept_len=X.XX ...
rows = []
for line in text.splitlines():
    m = re.search(r"ctx=(\d+).*gen=(\d+).*accept_len=([0-9.]+)", line)
    if m:
        rows.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
long = [r for r in rows if r[0] >= 32768 and r[1] >= 2560]
ok = all(r[2] >= 2.5 for r in long) if long else False
print({"long_rows": long, "pass": ok})
print("PASS" if ok else "FAIL")
PY

log "DONE"
cleanup
