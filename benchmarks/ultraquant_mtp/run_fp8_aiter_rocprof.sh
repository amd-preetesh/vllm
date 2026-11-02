#!/bin/bash
# Fresh FP8 AITER profile. Stops if the smoke launch or the 3D kernel is missing.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/kernel-prof-fp8.log 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
OUT=/workspace/logs/kernel_prof_fp8
SEQ="${FFDSL_PROF_SEQ:-131072}"
ITERS="${FFDSL_PROF_ITERS:-8}"

export PATH="$VENV/bin:/opt/rocm/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export FFDSL_PROF_SEQ="$SEQ"
export FFDSL_PROF_ITERS="$ITERS"
export FFDSL_PROF_WARMUP=0

mkdir -p "$OUT"
log() { echo "$(date -Is) $*"; }
BENCH=/workspace/logs/bench_fp8_aiter_rocprof.py
KERNEL_REGEX='kernel_unified_attention_3d'
PMC_PREFIX=kernel_prof_fp8
TRACE_PREFIX=kernel_prof_fp8_trace
source /workspace/vllm/benchmarks/ultraquant_mtp/rocprof_pmc_groups.sh

log "fp8 profile start pid=$$ hip=${HIP_VISIBLE_DEVICES} seq=${SEQ} iters=${ITERS}"
log "smoke launch"
if ! timeout -k 20 300 env FFDSL_PROF_ITERS=1 "$VENV/bin/python" "$BENCH" \
  > "$OUT/smoke.log" 2>&1; then
  log "FAIL smoke"
  tail -n 40 "$OUT/smoke.log" || true
  exit 1
fi
if ! grep -q "fp8 aiter rocprof done" "$OUT/smoke.log"; then
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
log "fp8 profile complete"
touch /workspace/logs/kernel-prof-fp8.ok
