#!/bin/bash
# A/B: does the DFlash+prefix memfault need a real prefix-cache hit?
#   DFLASH_AB_MODE=shared   -> prompts share a long "word word ..." prefix
#   DFLASH_AB_MODE=distinct -> each prompt starts with a unique token (no hit)
# Extra env for the server (e.g. VLLM_DFLASH_PREFIX_HIT_SAFE=1) is passed through.
trap '' HUP
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp/dflash_qwen38
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
HIP="${HIP_VISIBLE_DEVICES:?need HIP}"
PORT="${DFLASH_PORT:-8365}"
MODE="${DFLASH_AB_MODE:-shared}"
TAG="${DFLASH_AB_TAG:-$MODE}"
LENGTHS="${DFLASH_LENGTHS:-20000,24000,30000,36000}"
MNBT="${DFLASH_MNBT:-8192}"
OUT=/workspace/vllm/out/dflash_qwen38_fp8_agentx_debug/ab_${TAG}
LOG=/workspace/logs/dflash-ab-${TAG}.log
mkdir -p /workspace/logs "$OUT"
exec >>"$LOG" 2>&1

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

spec=$(DRAFT="$DRAFT" python - <<'PY'
import json, os
cfg = json.load(open("/workspace/vllm/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2.json"))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = 262144
print(json.dumps(cfg))
PY
)

log "AB start tag=$TAG mode=$MODE hip=$HIP port=$PORT lengths=$LENGTHS mnbt=$MNBT"
env | grep -E '^VLLM_DFLASH|^VLLM_MAMBA|^AMD_|^HIP_LAUNCH' || true
kill_port
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8 \
  --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --no-enable-log-requests --max-num-seqs 1 \
  --max-num-batched-tokens "$MNBT" \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  --enforce-eager \
  --speculative-config "$spec" \
  --enable-prefix-caching \
  >"$OUT/server.log" 2>&1 &

deadline=$((SECONDS + 900))
until curl -sf "http://127.0.0.1:${PORT}/v1/models" | grep -q "Qwen3.8-27B"; do
  if (( SECONDS > deadline )); then log "FAIL not ready"; tail -50 "$OUT/server.log"; kill_port; exit 1; fi
  sleep 5
done
log "READY"

extra=()
[ "$MODE" = "distinct" ] && extra=(--distinct-prefix "uniq")
python "$HERE/probe_long_prompt_acceptance.py" --port "$PORT" --model "$MODEL" \
  --lengths "$LENGTHS" --max-tokens 32 "${extra[@]}" || log "WARN probe rc=$?"

if grep -q "Memory access fault" "$OUT/server.log"; then
  log "FAULT $TAG"
  grep -n "Memory access fault" "$OUT/server.log" | tail -2
else
  log "NOFAULT $TAG"
fi
grep -E "Prefix cache hit rate" "$OUT/server.log" | tail -3 | cut -c1-300
grep -E "DFLASH_PREFIX_HIT|mamba hit audit" "$OUT/server.log" | tail -20 | cut -c1-400
kill_port
log "AB done $TAG"
