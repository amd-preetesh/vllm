#!/bin/bash
# Wait for a free GPU, then cold-cache adaptive smoke after PR #53929 port.
trap '' HUP
set -uo pipefail
ROOT=/workspace/vllm
CLAIM=$ROOT/benchmarks/ultraquant_mtp/gpu_claim.sh
SMOKE=$ROOT/benchmarks/ultraquant_mtp/debug_adaptive_pr53929_smoke.sh
LOG=/workspace/logs/adaptive-pr53929-smoke-waiter.log
NEED_GIB="${NEED_GIB:-170}"
PORT="${PORT:-8893}"
mkdir -p /workspace/logs
exec >>"$LOG" 2>&1
source "$CLAIM"
log() { echo "$(date -Is) $*"; }

log "waiter start need_gib=$NEED_GIB port=$PORT pid=$$"
while true; do
  if claim_gpu "$NEED_GIB"; then
    hip="$HIP_VISIBLE_DEVICES"
    log "CLAIMED hip=$hip — launching smoke"
    # Keep claim file for this waiter while smoke runs.
    echo "$(date -Is) pid=$$ hip=$hip need=$NEED_GIB role=adaptive_pr53929_smoke" \
      >"/workspace/logs/gpu_claims/hip${hip}.claim"
    if [ -n "${_CLAIM_FD:-}" ]; then
      flock -u "$_CLAIM_FD" 2>/dev/null || true
    fi
    export HIP_VISIBLE_DEVICES="$hip" CUDA_VISIBLE_DEVICES="$hip" PORT
    unset ROCR_VISIBLE_DEVICES
    set +e
    bash "$SMOKE"
    rc=$?
    set -e
    rm -f "/workspace/logs/gpu_claims/hip${hip}.claim" 2>/dev/null || true
    release_gpu 2>/dev/null || true
    log "smoke finished rc=$rc"
    if [ "$rc" -eq 0 ]; then
      echo "$(date -Is) PASS" >/workspace/logs/adaptive-pr53929-smoke.DONE
    else
      echo "$(date -Is) FAIL rc=$rc" >/workspace/logs/adaptive-pr53929-smoke.FAIL
    fi
    exit "$rc"
  fi
  log "waiting for >=${NEED_GIB}GiB"
  sleep 45
done
