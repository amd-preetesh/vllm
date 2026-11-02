#!/bin/bash
# Extra rocprof passes: instruction mix, stalls, and L2 hits/misses.
# Writes into the existing kernel_prof_* directories.
trap '' HUP
mkdir -p /workspace/logs
PROFILE="${PROFILE:?fused|fp8|prod}"
exec >> "/workspace/logs/kernel-prof-gaps-${PROFILE}.log" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq

export PATH="$VENV/bin:/opt/rocm/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_SHARED_KV=0
export FFDSL_PROF_SEQ="${FFDSL_PROF_SEQ:-131072}"
export FFDSL_PROF_ITERS="${FFDSL_PROF_ITERS:-8}"
export FFDSL_PROF_WARMUP=0

case "$PROFILE" in
  fused)
    export VLLM_ULTRAQUANT_PAIR_VERIFY=1
    export VLLM_ULTRAQUANT_PAIR_WAVES=2
    OUT=/workspace/logs/kernel_prof_fused
    BENCH=/workspace/logs/bench_ffdsl_pair_rocprof.py
    KERNEL_REGEX='ultraquant_decode_hd256_shared'
    PMC_PREFIX=kernel_prof_fused
    ;;
  fp8)
    OUT=/workspace/logs/kernel_prof_fp8
    BENCH=/workspace/logs/bench_fp8_aiter_rocprof.py
    KERNEL_REGEX='kernel_unified_attention_3d'
    PMC_PREFIX=kernel_prof_fp8
    ;;
  prod)
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    export VLLM_ULTRAQUANT_PAIR_WAVES=2
    OUT=/workspace/logs/kernel_prof_prod
    BENCH=/workspace/logs/bench_prod_flydsl_rocprof.py
    KERNEL_REGEX='ultraquant_decode_hd256_kernel'
    PMC_PREFIX=kernel_prof_prod
    ;;
  *)
    echo "unknown PROFILE=$PROFILE" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT"
log() { echo "$(date -Is) $*"; }
source /workspace/vllm/benchmarks/ultraquant_mtp/rocprof_pmc_groups.sh

log "gaps start profile=${PROFILE} pid=$$ hip=${HIP_VISIBLE_DEVICES}"
run_pmc_detail_gaps
log "gaps complete profile=${PROFILE}"
touch "/workspace/logs/kernel-prof-gaps-${PROFILE}.ok"
