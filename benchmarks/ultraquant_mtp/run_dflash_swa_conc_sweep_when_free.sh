#!/bin/bash
# DFlash UQ SWA AgentX concurrency fill: c=8,16,20,24,30,32.
# Skips cells that already have aiperf artifacts (c32 is done).
# Claims one free GPU via flock; survives HUP/SSH disconnect.
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-dflash-swa-conc-sweep.log
STATE=/workspace/logs/uq-dflash-swa-conc-sweep.state
DONEFLAG=/workspace/logs/uq-dflash-swa-conc-sweep.DONE
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
ROOT=/workspace/vllm
# shellcheck source=/dev/null
source "$HERE/gpu_claim.sh"

PORT="${PORT:-8463}"
NEED_GIB="${NEED_GIB:-200}"
UTIL="${UTIL:-0.70}"
CONCS="${CONCS:-8 16 20 24 30 32}"
POLL_SEC="${POLL_SEC:-45}"

log() { echo "$(date -Is) $*"; }
mark() { echo "$1" >"$STATE"; log "STATE $1"; }

all_done() {
  local c out
  for c in $CONCS; do
    out=$ROOT/out/dflash_qwen38_uq_dflash_swa/sa/seed42_c${c}
    if [ ! -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
      return 1
    fi
  done
  return 0
}

pending_list() {
  local c out
  for c in $CONCS; do
    out=$ROOT/out/dflash_qwen38_uq_dflash_swa/sa/seed42_c${c}
    if [ ! -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
      echo -n "$c "
    fi
  done
}

log "dflash_swa conc sweep start need=$NEED_GIB util=$UTIL port=$PORT concs=$CONCS pid=$$"
mark "waiting"
rm -f "$DONEFLAG"

if all_done; then
  mark "already_done"
  date -Is >"$DONEFLAG"
  log "ALL SWA concs already present — exit"
  exit 0
fi

for i in $(seq 1 2880); do  # ~36h at 45s
  if all_done; then
    break
  fi
  pend=$(pending_list)
  mark "waiting:pending=$pend"
  if claim_gpu "$NEED_GIB"; then
    hip="$HIP_VISIBLE_DEVICES"
    log "GPU $hip claimed — launching SWA AgentX concs=[$pend]"
    mark "running:hip=$hip:pending=$pend"
    # Drop global flock so other waiters can claim *other* GPUs; keep claim file.
    if [ -n "${_CLAIM_FD:-}" ]; then
      flock -u "$_CLAIM_FD" 2>/dev/null || true
    fi
    echo "$(date -Is) pid=$$ hip=$hip role=dflash_swa_sweep" >"/workspace/logs/gpu_claims/hip${hip}.claim"
    # Runner skips completed cells internally.
    HIP_VISIBLE_DEVICES="$hip" CUDA_VISIBLE_DEVICES="$hip" \
      EXP=dflash_swa CFG=dflash_uq PORT="$PORT" CONCS="$CONCS" UTIL="$UTIL" \
      bash "$HERE/run_uq_spec_ab_agentx.sh" || log "WARN dflash_swa sweep rc=$?"
    rm -f "/workspace/logs/gpu_claims/hip${hip}.claim" 2>/dev/null || true
    unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    release_gpu 2>/dev/null || true
    mark "post_run"
    # Refresh Pareto canvas from host path if mounted
    if [ -f /workspace/logs/refresh-dflash-dspark-agentx-pareto.py ] \
      || [ -f /home/preshah/workspace/logs/refresh-dflash-dspark-agentx-pareto.py ]; then
      python3 /home/preshah/workspace/logs/refresh-dflash-dspark-agentx-pareto.py \
        2>/dev/null \
        || python3 /workspace/logs/refresh-dflash-dspark-agentx-pareto.py \
        2>/dev/null \
        || true
    fi
    if all_done; then
      break
    fi
    log "still missing: $(pending_list) — will re-claim"
  else
    log "no free GPU yet (need ${NEED_GIB}GiB); sleep $POLL_SEC"
  fi
  sleep "$POLL_SEC"
done

if all_done; then
  mark "done"
  date -Is >"$DONEFLAG"
  log "dflash_swa conc sweep ALL DONE"
  python3 /home/preshah/workspace/logs/refresh-dflash-dspark-agentx-pareto.py 2>/dev/null || true
  exit 0
fi

mark "timeout"
log "FAIL timed out with pending=$(pending_list)"
exit 1
