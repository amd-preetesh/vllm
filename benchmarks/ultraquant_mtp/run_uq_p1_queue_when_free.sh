#!/bin/bash
# After P0 code queue: run medium-priority research items.
#   E6  PDV qlen-8 @ 200k bench
#   E7  K=6 AgentX c32 (DSpark + DFlash, ret50k)
#   P1-1 continuation-prefill v2 AgentX c32 (DSpark + DFlash, ret50k)
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-p1-queue.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
# shellcheck source=/dev/null
source "$HERE/gpu_claim.sh"

PORT="${PORT:-8464}"
NEED_GIB="${NEED_GIB:-260}"
# E6 is a synthetic kernel bench — lower free-VRAM bar.
NEED_GIB_E6="${NEED_GIB_E6:-40}"
UTIL="${UTIL:-0.70}"
CONCS_C32="${CONCS_C32:-32}"
WAIT_P0="${WAIT_P0:-1}"

log() { echo "$(date -Is) $*"; }

wait_for_p0_if_needed() {
  [ "$WAIT_P0" = 1 ] || return 0
  local done=/workspace/logs/uq-p0-code-queue.DONE
  local p0_log=/workspace/logs/uq-p0-code-queue.log
  log "waiting for P0 queue (WAIT_P0=1)"
  for i in $(seq 1 2880); do  # up to 24h @ 30s
    if [ -f "$done" ]; then
      log "P0 DONE file present — starting P1 queue"
      return 0
    fi
    if ! pgrep -f run_uq_p0_code_queue_when_free.sh >/dev/null 2>&1 \
      && ! pgrep -f run_uq_opt_agentx_when_free.sh >/dev/null 2>&1 \
      && ! pgrep -f run_dflash_swa_when_free.sh >/dev/null 2>&1; then
      if grep -q "P0 code queue done" "$p0_log" 2>/dev/null \
        || [ ! -f "$p0_log" ]; then
        log "P0/opt/swa idle — starting P1 queue"
        return 0
      fi
    fi
    if [ $((i % 10)) -eq 1 ]; then
      log "still waiting on P0/opt/swa (try $i)"
    fi
    sleep 30
  done
  log "WARN P0 wait timed out; continuing P1 queue anyway"
}

run_with_gpu() {
  local label="$1"
  local need="$2"
  shift 2
  for i in $(seq 1 1440); do
    if claim_gpu "$need"; then
      log "$label: GPU $HIP_VISIBLE_DEVICES claimed need=${need}GiB (flock held)"
      "$@"
      local rc=$?
      release_gpu
      return $rc
    fi
    if [ $((i % 10)) -eq 1 ]; then
      log "$label: waiting for >=${need}GiB (try $i)"
    fi
    sleep 30
  done
  log "FAIL $label timed out waiting for GPU"
  return 1
}

log "P1 queue start port=$PORT util=$UTIL need_gib=$NEED_GIB e6_need=$NEED_GIB_E6"
wait_for_p0_if_needed

# --- E6: PDV qlen-8 long-ctx bench ---
E6_OUT=/workspace/vllm/out/uq_e6_pdv_qlen8/summary.json
if [ -f "$E6_OUT" ]; then
  log "SKIP E6 — $E6_OUT exists"
else
  run_with_gpu "E6 pdv_qlen8" "$NEED_GIB_E6" bash -c '
    HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
      bash "'"$HERE"'/run_uq_e6_pdv_qlen8.sh"
  ' || log "WARN E6 rc=$?"
fi

# --- E7: K=6 AgentX c32 ---
run_with_gpu "E7 k6 dspark" "$NEED_GIB" bash -c '
  EXP=k6 CFG=dspark_uq PORT='"$PORT"' CONCS="'"$CONCS_C32"'" UTIL='"$UTIL"' \
    bash "'"$HERE"'/run_uq_spec_ab_agentx.sh"
' || log "WARN E7 dspark rc=$?"

run_with_gpu "E7 k6 dflash" "$NEED_GIB" bash -c '
  EXP=k6 CFG=dflash_uq PORT='"$PORT"' CONCS="'"$CONCS_C32"'" UTIL='"$UTIL"' \
    bash "'"$HERE"'/run_uq_spec_ab_agentx.sh"
' || log "WARN E7 dflash rc=$?"

# --- P1-1: continuation-prefill v2 AgentX c32 ---
run_with_gpu "P1-1 cont_v2 dspark" "$NEED_GIB" bash -c '
  EXP=cont_v2 CFG=dspark_uq PORT='"$PORT"' CONCS="'"$CONCS_C32"'" UTIL='"$UTIL"' \
    bash "'"$HERE"'/run_uq_spec_ab_agentx.sh"
' || log "WARN P1-1 dspark rc=$?"

run_with_gpu "P1-1 cont_v2 dflash" "$NEED_GIB" bash -c '
  EXP=cont_v2 CFG=dflash_uq PORT='"$PORT"' CONCS="'"$CONCS_C32"'" UTIL='"$UTIL"' \
    bash "'"$HERE"'/run_uq_spec_ab_agentx.sh"
' || log "WARN P1-1 dflash rc=$?"

log "P1 queue done"
touch /workspace/logs/uq-p1-queue.DONE
