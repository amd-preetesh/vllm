#!/bin/bash
# P0-1 proof: DFlash FP8 AgentX c32 with --block-size 64 (LCM=1664 verified).
# Writes to out/dflash_qwen38_fp8_p01/ (does not overwrite baseline matrix).
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/dflash-fp8-p01-agentx.log
exec >>"$LOG" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
SPEC_JSON="${DFLASH_SPEC_JSON:-$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2.json}"
BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-64}"
CONC="${DFLASH_CONC:-32}"
SEED="${DFLASH_SEED:-42}"
DURATION="${DFLASH_DURATION:-300}"
OUT_NAME="${DFLASH_OUT_NAME:-dflash_qwen38_fp8_p01}"
OUT=$ROOT/out/$OUT_NAME
SA=$OUT/sa
mkdir -p "$SA"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP device}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SA_DIR/.hf-cache}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export VLLM_USE_V2_MODEL_RUNNER=1

PORT="${DFLASH_PORT:-${PORT:-8440}}"
MAX_LEN=262144
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

out=$SA/seed${SEED}_c${CONC}
if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
  log "SKIP already have $out"
  exit 0
fi
mkdir -p "$out"

SPEC=$(DRAFT="$DRAFT" MAX_LEN="$MAX_LEN" SPEC_JSON="$SPEC_JSON" python - <<'PY'
import json, os
cfg = json.load(open(os.environ["SPEC_JSON"]))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = int(os.environ["MAX_LEN"])
print(json.dumps(cfg))
PY
)

log "SERVE dflash_p01 conc=$CONC block=$BLOCK_SIZE spec=$SPEC_JSON hip=$HIP_VISIBLE_DEVICES port=$PORT"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8 \
  --block-size "$BLOCK_SIZE" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --no-enable-log-requests \
  --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  --speculative-config "$SPEC" \
  >"$out/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "FAIL server exited"; tail -n 60 "$out/server.log"; exit 1
  fi
  sleep 5
done
[ "$ready" = 1 ] || { log "FAIL not ready"; exit 1; }

python - <<PY
import re, pathlib, json
log = pathlib.Path("$out/server.log").read_text(errors="replace")
m = re.search(r"kv lcm block sizes\s*(\d+)", log)
lcm = int(m.group(1)) if m else None
g = re.search(r"kv cache group sizes\s*(.+)$", log, re.M)
blob = g.group(1).strip() if g else ""
pathlib.Path("$out/geometry.json").write_text(json.dumps({"lcm": lcm, "groups": blob}, indent=2)+"\n")
print("geometry", {"lcm": lcm, "groups": blob})
if lcm != 1664:
    raise SystemExit(f"FAIL unexpected lcm={lcm}")
print("PASS geometry lcm=1664")
PY

log "SA START seed=$SEED conc=$CONC"
extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')
(cd "$INFERENCEX_DIR" && "$AIPERF" profile --scenario inferencex-agentx-mvp \
  --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model "$MODEL" --concurrency "$CONC" --benchmark-duration "$DURATION" \
  --unsafe-override \
  --stats-interval 30 --random-seed "$SEED" --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-grace-period 300 \
  --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
  --max-context-length "$MAX_LEN" \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --output-artifact-dir "$out/aiperf_artifacts" \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  "${extra[@]}") >"$out/benchmark.log" 2>&1 || log "WARN aiperf rc=$?"

curl -s "http://127.0.0.1:$PORT/metrics" \
  | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions|num_requests_running|num_requests_waiting|gpu_cache_usage)' \
  >"$out/metrics_end.txt" || true
printf '%s\n' "p01=1 block=$BLOCK_SIZE spec=$SPEC_JSON conc=$CONC seed=$SEED" >"$out/cell_meta.txt"
cleanup_server
if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
  log "END $out"
else
  log "FAIL no profile export"
  exit 1
fi
