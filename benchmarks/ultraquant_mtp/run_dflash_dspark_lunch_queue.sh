#!/bin/bash
# Lunch auto-queue: DFlash2 fair-speed then DSpark FP8 (smoke→GSM→GPQA→AgentX).
# Hard constraint: vLLM --tensor-parallel-size 1 on one HIP device.
trap '' HUP
set -uo pipefail
mkdir -p /workspace/logs
LOG=/workspace/logs/dflash-dspark-lunch-queue.log
STATE=/workspace/logs/dflash-dspark-lunch-queue.state
exec >>"$LOG" 2>&1

HIP="${HIP_VISIBLE_DEVICES:?need HIP}"
export HIP_VISIBLE_DEVICES="$HIP" CUDA_VISIBLE_DEVICES="$HIP"
unset ROCR_VISIBLE_DEVICES
export PATH=/workspace/.venv-uq/bin:$PATH
export PYTHONPATH=/workspace/aiter:/workspace/vllm
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export VLLM_USE_V2_MODEL_RUNNER=1

ROOT=/workspace/vllm
DFLASH=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38
DSPARK=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38
UTIL="${GPU_UTIL:-0.90}"
PORT_BASE="${LUNCH_PORT_BASE:-8380}"

log() { echo "$(date -Is) $*"; }
mark() { echo "$1" >"$STATE"; log "STATE $1"; }

log "LUNCH QUEUE start hip=$HIP util=$UTIL tp=1 port_base=$PORT_BASE pid=$$"
mark "start"

# ---- 0) DSpark draft fetch (can overlap conceptually; do first if missing) ----
mark "dspark_fetch"
bash "$DSPARK/fetch_draft.sh" || log "WARN dspark fetch rc=$?"
[ -f /workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark/config.json ] \
  && log "DSpark draft OK" || log "FAIL DSpark draft missing"

# ---- 1) DFlash short-prompt fair-speed (TP=1) ----
mark "dflash_short_tput"
export DFLASH_PORT=$PORT_BASE DFLASH_GPU_UTIL="$UTIL"
bash "$DFLASH/run_short_prompt_throughput.sh" || log "WARN dflash short tput rc=$?"
[ -f $ROOT/out/dflash_qwen38_fp8_short_tput/summary.json ] \
  && log "DFlash short tput OK" || log "FAIL dflash short tput"
sleep 20

# ---- 2) DSpark FP8 full Part1 (TP=1 inside script) ----
mark "dspark_fp8"
if [ ! -f /workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark/config.json ]; then
  log "DSpark draft still missing — retry fetch"
  bash "$DSPARK/fetch_draft.sh" || log "WARN dspark fetch retry rc=$?"
fi
export DSPARK_PORT=$((PORT_BASE + 1)) HIP_VISIBLE_DEVICES="$HIP"
bash "$DSPARK/run_qwen38_dspark_fp8.sh" || log "WARN dspark fp8 rc=$?"

mark "done"
log "LUNCH QUEUE ALL DONE hip=$HIP tp=1"
# summarize
python - <<'PY'
import json
from pathlib import Path
p=Path("/workspace/vllm/out/dflash_qwen38_fp8_short_tput/summary.json")
if p.exists():
    print("DFLASH_SHORT", json.dumps(json.load(open(p))["rows"]))
for name in ["gsm8k_seed7","gpqa_seed7"]:
    s=Path(f"/workspace/vllm/out/dspark_qwen38_fp8/accuracy/{name}/summary.json")
    if s.exists():
        j=json.load(open(s)); print("DSPARK", name, j.get("accuracy"))
sa=Path("/workspace/vllm/out/dspark_qwen38_fp8/sa/seed42_c32/aiperf_artifacts/profile_export_aiperf.json")
if sa.exists():
    j=json.load(open(sa))
    print("DSPARK_AGENTX tok/s", j.get("output_token_throughput",{}).get("avg"))
PY
