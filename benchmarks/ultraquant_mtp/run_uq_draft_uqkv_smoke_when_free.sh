#!/bin/bash
# Weekend S0: draft Q × UltraQuant KV smoke (Idea A kernel path).
# Does not touch AgentX ports 8460-8470. Low VRAM bar.
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-draft-uqkv-smoke.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
# shellcheck source=/dev/null
source "$HERE/gpu_claim.sh"

ROOT=/workspace/vllm
VENV=/workspace/.venv-uq
AITER_SRC=/workspace/aiter
NEED_GIB="${NEED_GIB:-40}"
OUT="${OUT:-$ROOT/out/uq_draft_uqkv_smoke/summary.json}"

log() { echo "$(date -Is) $*"; }

if [ -f "$OUT" ]; then
  log "SKIP smoke — $OUT exists"
  exit 0
fi

log "draft×UQKV S0 smoke waiter start need_gib=$NEED_GIB"
for i in $(seq 1 1440); do
  if claim_gpu "$NEED_GIB"; then
    log "GPU $HIP_VISIBLE_DEVICES claimed — running S0 smoke"
    export PATH="$VENV/bin:$PATH"
    export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
    export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
    export PYTHONPATH="$AITER_SRC:$ROOT"
    python "$HERE/bench_draft_cross_uq_kv_smoke.py" \
      --ctx "${CTX:-4096,16384,32768}" \
      --qlens "${QLENS:-1,4,8}" \
      --out "$OUT"
    rc=$?
    release_gpu
    log "S0 smoke done rc=$rc"
    exit $rc
  fi
  if [ $((i % 10)) -eq 1 ]; then
    log "waiting for >=${NEED_GIB}GiB (try $i)"
  fi
  sleep 30
done
log "FAIL timed out"
exit 1
