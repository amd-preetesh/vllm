#!/bin/bash
# GSM8K + GPQA for DSpark UQ opt + adaptive v3 (PR #53929 port) when a GPU is free.
# Waits for AgentX v3 DONE flag when present so accuracy does not race the ladder.
trap '' HUP
set -uo pipefail

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp
# shellcheck source=/dev/null
source "$HERE/gpu_claim.sh"

LOG=/workspace/logs/accuracy-dspark-opt-adaptive-v3-queue.log
NEED_GIB="${NEED_GIB:-250}"
PORT="${PORT:-8861}"
UTIL_CAP="${UTIL_CAP:-0.70}"
ARM=dspark_uq_opt_adaptive_v3
STEM=dspark_qwen38_uq_opt_adaptive_v3
AGENTX_DONE=/workspace/logs/uq-dspark-opt-adaptive-v3-agentx.DONE
WAIT_AGENTX="${WAIT_AGENTX:-1}"

mkdir -p /workspace/logs
exec >>"$LOG" 2>&1
log() { echo "$(date -Is) $*"; }

arm_done() {
  [ -f "$ROOT/out/$STEM/accuracy/gsm8k_seed7/summary.json" ] \
    && [ -f "$ROOT/out/$STEM/accuracy/gpqa_seed7/summary.json" ]
}

log "accuracy adaptive_v3 queue start need_gib=$NEED_GIB port=$PORT wait_agentx=$WAIT_AGENTX"
for i in $(seq 1 1440); do
  if arm_done; then
    log "DONE $ARM already has gsm+gpqa"
    date -Is >/workspace/logs/accuracy-dspark-opt-adaptive-v3.DONE
    exit 0
  fi
  if [ "$WAIT_AGENTX" = 1 ] && [ ! -f "$AGENTX_DONE" ]; then
    if [ $((i % 10)) -eq 1 ]; then
      log "waiting for AgentX v3 DONE ($AGENTX_DONE) try=$i"
    fi
    sleep 30
    continue
  fi
  if claim_gpu "$NEED_GIB"; then
    log "LAUNCH $ARM hip=$HIP_VISIBLE_DEVICES port=$PORT util_cap=$UTIL_CAP"
    ARM="$ARM" PORT="$PORT" UTIL_CAP="$UTIL_CAP" bash "$HERE/run_accuracy_fp8fa_uqbest.sh" || log "WARN rc=$?"
    release_gpu
    python3 /workspace/logs/refresh-dspark-adaptive-live.py 2>/dev/null || true
    if arm_done; then
      log "DONE $ARM"
      date -Is >/workspace/logs/accuracy-dspark-opt-adaptive-v3.DONE
      exit 0
    fi
    log "incomplete — retry"
  fi
  if [ $((i % 10)) -eq 1 ]; then
    log "waiting for >=${NEED_GIB}GiB (try $i)"
  fi
  sleep 30
done
log "FAIL timed out"
exit 1
