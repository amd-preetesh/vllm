#!/bin/bash
# P0-3 only: DFlash UQ + SWA window-table draft backend, AgentX c32.
# flock claim + pre-serve util fit; retries if engine init still fails.
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-dflash-swa-waiter.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
# shellcheck source=/dev/null
source "$HERE/gpu_claim.sh"

PORT="${PORT:-8462}"
NEED_GIB="${NEED_GIB:-260}"
UTIL="${UTIL:-0.70}"
CONCS="${CONCS:-32}"

log() { echo "$(date -Is) $*"; }

OUT=/workspace/vllm/out/dflash_qwen38_uq_dflash_swa/sa/seed42_c32
if [ -f "$OUT/aiperf_artifacts/profile_export_aiperf.json" ]; then
  log "SKIP already have dflash_swa c32"
  exit 0
fi
# Clear failed partial cell so the runner does not confuse SKIP logic
if [ -d "$OUT" ] && [ ! -f "$OUT/aiperf_artifacts/profile_export_aiperf.json" ]; then
  rm -rf "$OUT"
  log "cleared failed partial $OUT"
fi

log "dflash_swa waiter start need_gib=$NEED_GIB util=$UTIL port=$PORT concs=$CONCS"
for i in $(seq 1 1440); do
  if claim_gpu "$NEED_GIB"; then
    log "GPU $HIP_VISIBLE_DEVICES claimed (flock held) — launching dflash_swa AgentX"
    EXP=dflash_swa CFG=dflash_uq PORT="$PORT" CONCS="$CONCS" UTIL="$UTIL" \
      bash "$HERE/run_uq_spec_ab_agentx.sh" || log "WARN dflash_swa rc=$?"
    if [ -f "$OUT/aiperf_artifacts/profile_export_aiperf.json" ]; then
      release_gpu
      log "dflash_swa DONE"
      exit 0
    fi
    log "FAIL no profile; will retry on next free GPU"
    release_gpu
  fi
  if [ $((i % 10)) -eq 1 ]; then
    log "still waiting for >=${NEED_GIB}GiB (try $i)"
  fi
  sleep 30
done
log "FAIL timed out"
exit 1
