#!/bin/bash
# Detailed rocprofv3 of the 2-wave Fused FlyDSL decode kernel only.
# Isolated launches: 4 requests, query length 6, long cache. New output dir.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/ffdsl-pair-rocprof-detail.log 2>&1
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
OUT="${ROCPROF_OUT:-/workspace/logs/ffdsl_pair_rocprof_detail}"
SEQ="${FFDSL_PROF_SEQ:-131072}"
ITERS="${FFDSL_PROF_ITERS:-16}"

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

run_pmc() {
  local name="$1"
  shift
  log "PMC ${name} counters=$*"
  rocprofv3 \
    --kernel-include-regex 'ultraquant_decode_hd256_shared' \
    --pmc "$@" \
    -d "$OUT" \
    -o "ffdsl_pair_${name}" \
    -f csv json \
    -- \
    "$VENV/bin/python" /workspace/logs/bench_ffdsl_pair_rocprof.py \
    > "$OUT/${name}_stdout.log" 2>&1 \
    || log "WARN pmc ${name} rc=$?"
}

log "ffdsl pair rocprof detail start pid=$$ ppid=$PPID hip=${HIP_VISIBLE_DEVICES} seq=${SEQ} iters=${ITERS}"

rocprofv3 \
  --kernel-trace \
  --stats \
  --truncate-kernels \
  -d "$OUT" \
  -o ffdsl_pair_kernels \
  -f csv json \
  -- \
  "$VENV/bin/python" /workspace/logs/bench_ffdsl_pair_rocprof.py \
  > "$OUT/kernels_stdout.log" 2>&1 \
  || log "WARN kernel-trace rc=$?"

run_pmc issue SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_VMEM_RD SQ_INSTS_LDS SQ_INSTS_SALU
run_pmc stall SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_VMEM SQ_WAIT_INST_ANY SQ_BUSY_CYCLES SQ_WAVES
run_pmc cache TCC_HIT_sum TCC_MISS_sum SQ_INST_CYCLES_VMEM_RD SQ_WAIT_INST_LDS

log "ffdsl pair rocprof detail complete"
touch /workspace/logs/ffdsl-pair-rocprof-detail.ok
