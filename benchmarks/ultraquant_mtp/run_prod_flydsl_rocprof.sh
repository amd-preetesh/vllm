#!/bin/bash
# Fresh Prod FlyDSL profile. Stops if the smoke launch or the prod kernel is missing.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/kernel-prof-prod.log 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
OUT=/workspace/logs/kernel_prof_prod
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
export VLLM_ULTRAQUANT_PAIR_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_WAVES=2
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_SHARED_KV=0
export FFDSL_PROF_SEQ="$SEQ"
export FFDSL_PROF_ITERS="$ITERS"
export FFDSL_PROF_WARMUP=0

mkdir -p "$OUT"
log() { echo "$(date -Is) $*"; }
BENCH=/workspace/logs/bench_prod_flydsl_rocprof.py
KERNEL_REGEX='ultraquant_decode_hd256_kernel'
PMC_PREFIX=kernel_prof_prod
TRACE_PREFIX=kernel_prof_prod_trace
source /workspace/vllm/benchmarks/ultraquant_mtp/rocprof_pmc_groups.sh

log "prod profile start pid=$$ hip=${HIP_VISIBLE_DEVICES} seq=${SEQ} iters=${ITERS}"
log "smoke launch"
if ! timeout -k 20 300 env FFDSL_PROF_ITERS=1 "$VENV/bin/python" "$BENCH" \
  > "$OUT/smoke.log" 2>&1; then
  log "FAIL smoke"
  tail -n 40 "$OUT/smoke.log" || true
  exit 1
fi
if ! grep -q "prod flydsl rocprof done" "$OUT/smoke.log"; then
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
log "prod profile complete"
touch /workspace/logs/kernel-prof-prod.ok
