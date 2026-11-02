#!/bin/bash
# Fresh Fused FlyDSL profile. One smoke launch, then the kernel trace,
# then counter passes. Stops if the smoke launch or the expected kernel fails.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/kernel-prof-fused.log 2>&1
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
OUT=/workspace/logs/kernel_prof_fused
SEQ="${FFDSL_PROF_SEQ:-131072}"
ITERS="${FFDSL_PROF_ITERS:-8}"

export PATH="$VENV/bin:/opt/rocm/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_VERIFY=1
export VLLM_ULTRAQUANT_PAIR_WAVES=2
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_SHARED_KV=0
export FFDSL_PROF_SEQ="$SEQ"
export FFDSL_PROF_ITERS="$ITERS"
export FFDSL_PROF_WARMUP=0

mkdir -p "$OUT"
cd "$ROOT"
log() { echo "$(date -Is) $*"; }

BENCH=/workspace/logs/bench_ffdsl_pair_rocprof.py
KERNEL_REGEX='ultraquant_decode_hd256_shared'
PMC_PREFIX=kernel_prof_fused
TRACE_PREFIX=kernel_prof_fused_trace
source "$(dirname "$0")/rocprof_pmc_groups.sh"

log "fused profile start pid=$$ hip=${HIP_VISIBLE_DEVICES} seq=${SEQ} iters=${ITERS}"
log "smoke launch"
if ! timeout -k 20 300 env FFDSL_PROF_ITERS=1 "$VENV/bin/python" "$BENCH" \
  > "$OUT/smoke.log" 2>&1; then
  log "FAIL smoke"
  tail -n 30 "$OUT/smoke.log" || true
  exit 1
fi
if ! grep -q "ffdsl pair rocprof done" "$OUT/smoke.log"; then
  log "FAIL smoke did not finish"
  exit 1
fi
log "smoke ok"

log "trace kernels and memory copies"
if ! timeout -k 20 300 rocprofv3 \
  --kernel-trace --memory-copy-trace --stats --truncate-kernels \
  -d "$OUT" -o "$TRACE_PREFIX" -f csv json \
  -- "$VENV/bin/python" "$BENCH" \
  > "$OUT/trace_stdout.log" 2>&1; then
  log "FAIL trace rc=$?"
  exit 1
fi
if ! grep -q "$KERNEL_REGEX" "$OUT/${TRACE_PREFIX}_kernel_stats.csv"; then
  log "FAIL kernel ${KERNEL_REGEX} missing from trace"
  cat "$OUT/${TRACE_PREFIX}_kernel_stats.csv" || true
  exit 1
fi
log "trace ok"
run_pmc_groups
log "fused profile complete"
touch /workspace/logs/kernel-prof-fused.ok
