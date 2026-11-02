#!/bin/bash
# Wait for a free GPU, then run DSpark UQ opt + adaptive verification AgentX
# (dense draft + ret50k + mnbt2048 + enable_adaptive_verification).
# v3 = post PR #53929 / 3462da7 capture-geometry port (smoke accept 72%).
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-dspark-opt-adaptive-v3-waiter.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
# shellcheck source=/dev/null
source "$HERE/gpu_claim.sh"

PORT="${PORT:-8470}"
NEED_GIB="${NEED_GIB:-250}"
UTIL="${UTIL:-0.70}"
CONCS="${CONCS:-8 16 20 24 30 32}"
STEM=dspark_qwen38_uq_opt_adaptive_v3

log() { echo "$(date -Is) $*"; }

count_profiles() {
  local n=0 d
  for d in /workspace/vllm/out/$STEM/sa/seed42_c*; do
    [ -f "$d/aiperf_artifacts/profile_export_aiperf.json" ] && n=$((n + 1))
  done
  echo "$n"
}

clear_empty_cells() {
  local d
  for d in /workspace/vllm/out/$STEM/sa/seed42_c*; do
    [ -d "$d" ] || continue
    if [ ! -f "$d/aiperf_artifacts/profile_export_aiperf.json" ]; then
      rm -rf "$d"
      log "cleared empty $d"
    fi
  done
}

log "dspark opt_adaptive_v3 AgentX waiter start need_gib=$NEED_GIB util=$UTIL port=$PORT concs=$CONCS"
clear_empty_cells

for i in $(seq 1 1440); do
  n=$(count_profiles)
  if [ "$n" -ge 6 ]; then
    log "dspark opt_adaptive_v3 done ($n/6 profiles)"
    date -Is >/workspace/logs/uq-dspark-opt-adaptive-v3-agentx.DONE
    exit 0
  fi

  if claim_gpu "$NEED_GIB"; then
    log "GPU $HIP_VISIBLE_DEVICES claimed — launching dspark opt_adaptive_v3 (have $n/6)"
    EXP=opt_adaptive CFG=dspark_uq PORT="$PORT" CONCS="$CONCS" UTIL="$UTIL" \
      bash "$HERE/run_uq_spec_ab_agentx.sh" || log "WARN dspark opt_adaptive_v3 rc=$?"
    release_gpu
    clear_empty_cells
    python3 /workspace/logs/refresh-dspark-adaptive-live.py 2>/dev/null || true
    n=$(count_profiles)
    log "pass finished profiles=$n/6"
    if [ "$n" -ge 6 ]; then
      log "dspark opt_adaptive_v3 done ($n/6 profiles)"
      date -Is >/workspace/logs/uq-dspark-opt-adaptive-v3-agentx.DONE
      exit 0
    fi
    log "incomplete — will reclaim GPU and retry missing cells"
  fi
  if [ $((i % 10)) -eq 1 ]; then
    log "still waiting for >=${NEED_GIB}GiB free GPU (try $i, profiles=$(count_profiles)/6)"
  fi
  sleep 30
done
log "FAIL timed out waiting for free GPU (profiles=$(count_profiles)/6)"
exit 1
