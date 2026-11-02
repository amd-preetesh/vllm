#!/bin/bash
# FP8 fair-FA adaptive pipeline: smoke → AgentX C-ladder → GSM/GPQA accuracy.
# Shares GPU claim flock with the UQ adaptive_v3 waiters.
trap '' HUP
set -uo pipefail
ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp
# shellcheck source=/dev/null
source "$HERE/gpu_claim.sh"

LOG=/workspace/logs/dspark-fp8-adaptive-pipeline.log
NEED_GIB="${NEED_GIB:-250}"
# Smoke fits in ~120GiB free (util tracks free); AgentX/acc still want a fat GPU.
SMOKE_NEED_GIB="${SMOKE_NEED_GIB:-120}"
SMOKE_PORT="${SMOKE_PORT:-8894}"
AGENTX_PORT="${AGENTX_PORT:-8480}"
ACC_PORT="${ACC_PORT:-8862}"
UTIL="${UTIL:-0.70}"
CONCS="${CONCS:-8 16 20 24 30 32}"
STEM=dspark_qwen38_fp8_fair_fa_adaptive

mkdir -p /workspace/logs
exec >>"$LOG" 2>&1
log() { echo "$(date -Is) $*"; }

smoke_done() { [ -f /workspace/logs/adaptive-fp8-smoke.DONE ]; }
agentx_done() {
  local n=0 d
  for d in /workspace/vllm/out/$STEM/sa/seed42_c*; do
    [ -f "$d/aiperf_artifacts/profile_export_aiperf.json" ] && n=$((n + 1))
  done
  [ "$n" -ge 6 ]
}
accuracy_done() {
  [ -f /workspace/vllm/out/$STEM/accuracy/gsm8k_seed7/summary.json ] \
    && [ -f /workspace/vllm/out/$STEM/accuracy/gpqa_seed7/summary.json ]
}

run_smoke() {
  log "SMOKE start hip=$HIP_VISIBLE_DEVICES"
  PORT="$SMOKE_PORT" bash "$HERE/debug_adaptive_fp8_smoke.sh"
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    date -Is >/workspace/logs/adaptive-fp8-smoke.DONE
    rm -f /workspace/logs/adaptive-fp8-smoke.FAIL
    log "SMOKE PASS"
  else
    echo "$(date -Is) FAIL rc=$rc" >/workspace/logs/adaptive-fp8-smoke.FAIL
    log "SMOKE FAIL rc=$rc"
  fi
  return "$rc"
}

run_agentx() {
  log "AGENTX start hip=$HIP_VISIBLE_DEVICES concs=$CONCS"
  EXP=adaptive CFG=dspark PORT="$AGENTX_PORT" CONCS="$CONCS" UTIL="$UTIL" \
    bash "$HERE/fp8_fair/run_fp8_fair_agentx.sh" || log "WARN agentx rc=$?"
  if agentx_done; then
    date -Is >/workspace/logs/dspark-fp8-adaptive-agentx.DONE
    log "AGENTX DONE 6/6"
    return 0
  fi
  log "AGENTX incomplete"
  return 1
}

run_accuracy() {
  log "ACCURACY start hip=$HIP_VISIBLE_DEVICES"
  ARM=dspark_fp8_adaptive PORT="$ACC_PORT" UTIL_CAP="$UTIL" \
    bash "$HERE/run_accuracy_fp8fa_uqbest.sh" || log "WARN accuracy rc=$?"
  if accuracy_done; then
    date -Is >/workspace/logs/dspark-fp8-adaptive-accuracy.DONE
    log "ACCURACY DONE"
    return 0
  fi
  log "ACCURACY incomplete"
  return 1
}

log "fp8 adaptive pipeline start smoke_need=$SMOKE_NEED_GIB need_gib=$NEED_GIB pid=$$"
for i in $(seq 1 2880); do
  if smoke_done && agentx_done && accuracy_done; then
    log "ALL DONE smoke+agentx+accuracy"
    date -Is >/workspace/logs/dspark-fp8-adaptive-pipeline.DONE
    exit 0
  fi
  need="$NEED_GIB"
  if ! smoke_done; then
    need="$SMOKE_NEED_GIB"
  fi
  if claim_gpu "$need"; then
    hip="$HIP_VISIBLE_DEVICES"
    echo "$(date -Is) pid=$$ hip=$hip need=$need role=fp8_adaptive_pipeline" \
      >"/workspace/logs/gpu_claims/hip${hip}.claim"
    if [ -n "${_CLAIM_FD:-}" ]; then
      flock -u "$_CLAIM_FD" 2>/dev/null || true
    fi
    export HIP_VISIBLE_DEVICES="$hip" CUDA_VISIBLE_DEVICES="$hip"
    unset ROCR_VISIBLE_DEVICES
    rc=0
    if ! smoke_done; then
      run_smoke || rc=1
    fi
    # AgentX/acc need a fatter GPU — release and wait if this claim is smoke-sized.
    if [ "$rc" -eq 0 ] && smoke_done && ! agentx_done; then
      free_ok=$(python - <<PY
import torch, os
need=float(os.environ.get("NEED_GIB","250"))*(1024**3)
free,_=torch.cuda.mem_get_info(0)
print("1" if free>=need else "0")
PY
)
      if [ "$free_ok" = "1" ]; then
        run_agentx || rc=1
      else
        log "smoke done; hip=$hip too small for AgentX (need ${NEED_GIB}GiB) — reclaim later"
        rc=1
      fi
    fi
    if [ "$rc" -eq 0 ] && agentx_done && ! accuracy_done; then
      run_accuracy || rc=1
    fi
    rm -f "/workspace/logs/gpu_claims/hip${hip}.claim" 2>/dev/null || true
    release_gpu 2>/dev/null || true
    if smoke_done && agentx_done && accuracy_done; then
      log "ALL DONE"
      date -Is >/workspace/logs/dspark-fp8-adaptive-pipeline.DONE
      exit 0
    fi
    log "pass ended rc=$rc — will reclaim if still incomplete"
  fi
  if [ $((i % 10)) -eq 1 ]; then
    log "waiting try=$i smoke=$(smoke_done && echo Y || echo N) agentx=$(agentx_done && echo Y || echo N) acc=$(accuracy_done && echo Y || echo N)"
  fi
  sleep 30
done
log "FAIL timed out"
exit 1
