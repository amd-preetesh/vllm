#!/bin/bash
# Decode-only PDV profiling matrix. All artifacts are new and restartable.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/pdv-bottleneck-rocprof.log 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
OUT=/workspace/logs/pdv_bottleneck_rocprof
BENCH=/workspace/logs/bench_pdv_rocprof.py
KERNEL_REGEX='pdv_flydsl_decode_hd256_kernel'

export PATH="$VENV/bin:/opt/rocm/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_ULTRAQUANT_PDV_VERIFY=1
export VLLM_ULTRAQUANT_PAIR_VERIFY=0
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_SHARED_KV=0
export PDV_PROF_WARMUP=0

mkdir -p "$OUT"
cd "$ROOT"
log() { echo "$(date -Is) $*"; }

shape_env() {
  export PDV_PROF_REQS="$1"
  export PDV_PROF_QLEN="$2"
  export PDV_PROF_SEQ="$3"
  export VLLM_ULTRAQUANT_KV_AUX="$4"
  export PDV_PROF_ITERS="${5:-8}"
  export PDV_PROF_MAX_SPLITS="${6:-512}"
}

run_trace() {
  local label="$1" reqs="$2" qlen="$3" seq="$4" aux="$5" max_splits="${6:-512}"
  local dir="$OUT/$label" prefix="${label}_trace"
  mkdir -p "$dir"
  if [ -f "$dir/${prefix}_kernel_stats.csv" ]; then
    log "SKIP trace $label"
    return
  fi
  shape_env "$reqs" "$qlen" "$seq" "$aux" 8 "$max_splits"
  log "TRACE $label reqs=$reqs qlen=$qlen seq=$seq aux=$aux max_splits=$max_splits"
  if ! timeout -k 20 600 rocprofv3 \
      --kernel-trace --stats --truncate-kernels \
      -d "$dir" -o "$prefix" -f csv json \
      -- "$VENV/bin/python" "$BENCH" > "$dir/trace_stdout.log" 2>&1; then
    log "WARN trace $label rc=$?"
    return
  fi
  if ! grep -q "$KERNEL_REGEX" "$dir/${prefix}_kernel_stats.csv"; then
    log "WARN trace $label missing PDV kernel"
  fi
}

run_pmc() {
  local label="$1" group="$2"
  shift 2
  local dir="$OUT/$label" prefix="${label}_${group}"
  mkdir -p "$dir"
  if [ -f "$dir/${prefix}_counter_collection.csv" ]; then
    log "SKIP pmc $label/$group"
    return
  fi
  log "PMC $label/$group counters=$*"
  timeout -k 20 300 rocprofv3 \
    --kernel-include-regex "$KERNEL_REGEX" \
    --pmc "$@" \
    -d "$dir" -o "$prefix" -f csv json \
    -- "$VENV/bin/python" "$BENCH" > "$dir/${group}_stdout.log" 2>&1 \
    || log "WARN pmc $label/$group rc=$?"
}

run_core_counters() {
  local label="$1" reqs="$2"
  shape_env "$reqs" 6 131072 0 8
  run_pmc "$label" pipes GRBM_GUI_ACTIVE GRBM_COUNT SQ_THREAD_CYCLES_VALU SQ_ACTIVE_INST_VALU SQ_VALU_MFMA_BUSY_CYCLES
  run_pmc "$label" occ SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_BUSY_CU_CYCLES SQ_LDS_IDX_ACTIVE SQ_LDS_BANK_CONFLICT
  run_pmc "$label" stall SQ_ACTIVE_INST_VMEM SQ_WAIT_INST_ANY SQ_BUSY_CYCLES SQ_WAVES SQ_WAIT_INST_LDS
  run_pmc "$label" l2 TCC_HIT_sum TCC_MISS_sum SQ_INST_CYCLES_VMEM_RD
}

run_full_baseline_counters() {
  local label=req32_q6_s131k_a0
  shape_env 32 6 131072 0 8
  run_pmc "$label" issue SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_VMEM_RD SQ_INSTS_LDS SQ_INSTS_SALU
  run_pmc "$label" mfma SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_INSTS_VALU_MFMA_MOPS_F16 SQ_INSTS_VALU_MFMA_MOPS_F32 SQ_INSTS_VALU_MFMA_MOPS_F64
  run_pmc "$label" vmem SQ_INSTS_VMEM SQ_INSTS_VMEM_WR SQ_ACTIVE_INST_LDS
  run_pmc "$label" rd_a TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum
  run_pmc "$label" rd_b TCC_EA0_RDREQ_64B_sum TCC_EA0_RDREQ_128B_sum
  run_pmc "$label" bubble TCC_BUBBLE_sum
  run_pmc "$label" wr TCC_EA0_WRREQ_sum TCC_EA0_WRREQ_64B_sum
  run_pmc "$label" dram TCC_EA0_RDREQ_DRAM_sum TCC_EA0_WRREQ_DRAM_sum
}

log "pdv bottleneck rocprof start pid=$$ ppid=$PPID hip=$HIP_VISIBLE_DEVICES"
log "correctness smoke qlen 2..8"
if ! timeout -k 20 600 "$VENV/bin/python" /workspace/logs/bench_pdv_smoke.py \
    > "$OUT/correctness_smoke.log" 2>&1; then
  log "FAIL correctness smoke"
  exit 1
fi
log "correctness smoke ok"

# Saturation and sequence scaling.
for reqs in 1 4 16 32; do
  label="req${reqs}_q6_s131k_a0"
  run_trace "$label" "$reqs" 6 131072 0
  run_core_counters "$label" "$reqs"
done
run_trace req32_q6_s32k_a0 32 6 32768 0
run_trace req32_q6_s262k_a0 32 6 262144 0

# Verify-width and cache-policy experiments. Baseline qlen 6/aux 0 is above.
for qlen in 2 4 8; do
  run_trace "req32_q${qlen}_s131k_a0" 32 "$qlen" 131072 0
done
for aux in 2 18; do
  run_trace "req32_q6_s131k_a${aux}" 32 6 131072 "$aux"
done
for max_splits in 128 256; do
  run_trace "req32_q6_s131k_a0_p${max_splits}" 32 6 131072 0 "$max_splits"
done

run_full_baseline_counters
log "pdv bottleneck rocprof complete"
touch /workspace/logs/pdv-bottleneck-rocprof.ok
