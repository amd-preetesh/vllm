#!/bin/bash
# Counter groups for one attention kernel. Source this; do not execute it.
# Caller sets: OUT, BENCH, KERNEL_REGEX, VENV, and a log() function.
# TCC counters are paired. Five of them in one pass abort with
# "exceeds the capabilities of the hardware" (error 38) and rocprof
# then hangs in its abort handler, so every pass is under timeout.

run_pmc() {
  local name="$1"
  shift
  log "PMC ${name} counters=$*"
  timeout -k 20 180 rocprofv3 \
    --kernel-include-regex "$KERNEL_REGEX" \
    --pmc "$@" \
    -d "$OUT" \
    -o "${PMC_PREFIX}_${name}" \
    -f csv json \
    -- \
    "$VENV/bin/python" "$BENCH" \
    > "$OUT/${name}_stdout.log" 2>&1 \
    || log "WARN pmc ${name} rc=$?"
}

run_pmc_groups() {
  run_pmc pipes GRBM_GUI_ACTIVE GRBM_COUNT SQ_THREAD_CYCLES_VALU SQ_ACTIVE_INST_VALU SQ_VALU_MFMA_BUSY_CYCLES
  run_pmc occ SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_BUSY_CU_CYCLES SQ_LDS_IDX_ACTIVE SQ_LDS_BANK_CONFLICT
  run_pmc mfma SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_INSTS_VALU_MFMA_MOPS_F16 SQ_INSTS_VALU_MFMA_MOPS_F32 SQ_INSTS_VALU_MFMA_MOPS_F64
  run_pmc vmem SQ_INSTS_VMEM SQ_INSTS_VMEM_WR SQ_ACTIVE_INST_LDS
  run_pmc rd_a TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum
  run_pmc rd_b TCC_EA0_RDREQ_64B_sum TCC_EA0_RDREQ_128B_sum
  run_pmc bubble TCC_BUBBLE_sum
  run_pmc wr TCC_EA0_WRREQ_sum TCC_EA0_WRREQ_64B_sum
  run_pmc dram TCC_EA0_RDREQ_DRAM_sum TCC_EA0_WRREQ_DRAM_sum
}

# Counters from the Fused detail page: instruction mix, stalls, L2.
# These groups already fit in one pass on gfx950. L2 is two TCC counters.
run_pmc_detail_gaps() {
  run_pmc issue SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_VMEM_RD SQ_INSTS_LDS SQ_INSTS_SALU
  run_pmc stall SQ_ACTIVE_INST_VMEM SQ_WAIT_INST_ANY SQ_BUSY_CYCLES SQ_WAVES SQ_WAIT_INST_LDS
  run_pmc l2 TCC_HIT_sum TCC_MISS_sum SQ_INST_CYCLES_VMEM_RD
}
