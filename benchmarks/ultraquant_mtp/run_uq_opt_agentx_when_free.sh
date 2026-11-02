#!/bin/bash
# Wait for a GPU with enough free VRAM, then run the stacked UQ "opt" AgentX
# matrix for DSpark and DFlash2:
#   dense draft (ROCM_AITER_FA_UQ_DENSE) + ret50k + mnbt2048 + block64
#
# Uses flock claim + pre-serve util fit (via run_uq_spec_ab) so confirm→serve
# VRAM races no longer produce 0-cell matrices. Retries until at least one
# profile lands (or timeout).
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-opt-agentx-waiter.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
# shellcheck source=/dev/null
source "$HERE/gpu_claim.sh"

PORT="${PORT:-8460}"
# UTIL=0.70 needs ~201 GiB; require more headroom at claim so fit_util can hold 0.70.
NEED_GIB="${NEED_GIB:-260}"
UTIL="${UTIL:-0.70}"
CONCS="${CONCS:-8 16 20 24 30 32}"

log() { echo "$(date -Is) $*"; }

count_profiles() {
  local n=0 d
  for d in /workspace/vllm/out/dspark_qwen38_uq_opt/sa/seed42_c* \
           /workspace/vllm/out/dflash_qwen38_uq_opt/sa/seed42_c*; do
    [ -f "$d/aiperf_artifacts/profile_export_aiperf.json" ] && n=$((n + 1))
  done
  echo "$n"
}

# Drop empty partial cells so the runner re-attempts them.
clear_empty_opt_cells() {
  local d
  for d in /workspace/vllm/out/dspark_qwen38_uq_opt/sa/seed42_c* \
           /workspace/vllm/out/dflash_qwen38_uq_opt/sa/seed42_c*; do
    [ -d "$d" ] || continue
    if [ ! -f "$d/aiperf_artifacts/profile_export_aiperf.json" ]; then
      rm -rf "$d"
      log "cleared empty $d"
    fi
  done
}

log "opt AgentX waiter start need_gib=$NEED_GIB util=$UTIL port=$PORT concs=$CONCS"
clear_empty_opt_cells

for i in $(seq 1 1440); do  # up to 12h @ 30s
  n=$(count_profiles)
  if [ "$n" -ge 12 ]; then
    log "opt AgentX waiter done ($n/12 profiles)"
    exit 0
  fi

  if claim_gpu "$NEED_GIB"; then
    log "GPU $HIP_VISIBLE_DEVICES claimed (flock held) — launching opt AgentX matrix (have $n/12)"
    EXP=opt CFG=dspark_uq PORT="$PORT" CONCS="$CONCS" UTIL="$UTIL" \
      bash "$HERE/run_uq_spec_ab_agentx.sh" || log "WARN dspark opt rc=$?"
    EXP=opt CFG=dflash_uq PORT="$PORT" CONCS="$CONCS" UTIL="$UTIL" \
      bash "$HERE/run_uq_spec_ab_agentx.sh" || log "WARN dflash opt rc=$?"
    release_gpu
    clear_empty_opt_cells
    n=$(count_profiles)
    log "opt pass finished profiles=$n/12"
    if [ "$n" -ge 12 ]; then
      log "opt AgentX waiter done ($n/12 profiles)"
      exit 0
    fi
    log "incomplete — will reclaim GPU and retry missing cells"
  fi
  if [ $((i % 10)) -eq 1 ]; then
    log "still waiting for >=${NEED_GIB}GiB free GPU (try $i, profiles=$(count_profiles)/12)"
  fi
  sleep 30
done
log "FAIL timed out waiting for free GPU (profiles=$(count_profiles)/12)"
exit 1
