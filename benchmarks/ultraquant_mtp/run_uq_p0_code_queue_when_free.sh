#!/bin/bash
# After opt (or immediately if opt already done): run P0 code A/Bs.
# Order matches research ranking:
#   1) P0-3 DFlash SWA window table — AgentX c32 (skip if dedicated waiter owns it)
#   2) P0-2 mixed-batch PDV — AgentX c32 on DSpark + DFlash (ret50k base)
#   3) P0-1 decode-step rocprof (DSpark UQ)
#
# Uses flock claim + pre-serve util fit. Does not kill other users' jobs.
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-p0-code-queue.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
# shellcheck source=/dev/null
source "$HERE/gpu_claim.sh"

PORT="${PORT:-8462}"
NEED_GIB="${NEED_GIB:-260}"
UTIL="${UTIL:-0.70}"
CONCS_C32="${CONCS_C32:-32}"
WAIT_OPT="${WAIT_OPT:-1}"

log() { echo "$(date -Is) $*"; }

wait_for_opt_if_needed() {
  [ "$WAIT_OPT" = 1 ] || return 0
  local opt_log=/workspace/logs/uq-opt-agentx-waiter.log
  log "waiting for opt waiter to finish (WAIT_OPT=1)"
  for i in $(seq 1 1440); do
    if ! pgrep -f run_uq_opt_agentx_when_free.sh >/dev/null 2>&1; then
      if grep -q "opt AgentX waiter done" "$opt_log" 2>/dev/null \
        || grep -q "FAIL timed out" "$opt_log" 2>/dev/null \
        || [ ! -f "$opt_log" ]; then
        log "opt waiter idle — starting P0 code queue"
        return 0
      fi
    fi
    if [ $((i % 10)) -eq 1 ]; then
      log "still waiting on opt (try $i)"
    fi
    sleep 30
  done
  log "WARN opt wait timed out; continuing P0 queue anyway"
}

run_with_gpu() {
  local label="$1"
  shift
  for i in $(seq 1 1440); do
    if claim_gpu "$NEED_GIB"; then
      log "$label: GPU $HIP_VISIBLE_DEVICES claimed (flock held)"
      "$@"
      local rc=$?
      release_gpu
      return $rc
    fi
    if [ $((i % 10)) -eq 1 ]; then
      log "$label: waiting for >=${NEED_GIB}GiB (try $i)"
    fi
    sleep 30
  done
  log "FAIL $label timed out waiting for GPU"
  return 1
}

log "P0 code queue start port=$PORT util=$UTIL need_gib=$NEED_GIB"
wait_for_opt_if_needed

# --- P0-3: DFlash SWA (skip if dedicated waiter is running or profile exists) ---
SWA_OUT=/workspace/vllm/out/dflash_qwen38_uq_dflash_swa/sa/seed42_c32
if [ -f "$SWA_OUT/aiperf_artifacts/profile_export_aiperf.json" ]; then
  log "SKIP P0-3 dflash_swa — profile already present"
elif pgrep -f run_dflash_swa_when_free.sh >/dev/null 2>&1; then
  log "SKIP P0-3 dflash_swa — dedicated waiter owns it"
else
  run_with_gpu "P0-3 dflash_swa" bash -c '
    EXP=dflash_swa CFG=dflash_uq PORT='"$PORT"' CONCS="'"$CONCS_C32"'" UTIL='"$UTIL"' \
      bash "'"$HERE"'/run_uq_spec_ab_agentx.sh"
  ' || log "WARN P0-3 rc=$?"
fi

# --- P0-2: mixed-batch PDV ---
run_with_gpu "P0-2 mixed_pdv dspark" bash -c '
  EXP=mixed_pdv CFG=dspark_uq PORT='"$PORT"' CONCS="'"$CONCS_C32"'" UTIL='"$UTIL"' \
    bash "'"$HERE"'/run_uq_spec_ab_agentx.sh"
' || log "WARN P0-2 dspark rc=$?"

run_with_gpu "P0-2 mixed_pdv dflash" bash -c '
  EXP=mixed_pdv CFG=dflash_uq PORT='"$PORT"' CONCS="'"$CONCS_C32"'" UTIL='"$UTIL"' \
    bash "'"$HERE"'/run_uq_spec_ab_agentx.sh"
' || log "WARN P0-2 dflash rc=$?"

# --- P0-1: decode rocprof ---
run_with_gpu "P0-1 rocprof" bash -c '
  PORT='"$PORT"' UTIL='"$UTIL"' \
    bash "'"$HERE"'/run_uq_decode_rocprof_p01.sh"
' || log "WARN P0-1 rc=$?"

log "P0 code queue done"
touch /workspace/logs/uq-p0-code-queue.DONE
