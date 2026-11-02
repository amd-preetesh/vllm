#!/bin/bash
# E6: PDV qlen-6/8 correctness + latency at long UQ contexts.
#   HIP_VISIBLE_DEVICES=0 bash benchmarks/ultraquant_mtp/run_uq_e6_pdv_qlen8.sh
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-e6-pdv-qlen8.log
exec >>"$LOG" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
VENV=/workspace/.venv-uq
AITER_SRC=/workspace/aiter
OUT="${OUT:-$ROOT/out/uq_e6_pdv_qlen8/summary.json}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP device}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$AITER_SRC:$ROOT"

echo "$(date -Is) E6 start hip=$HIP_VISIBLE_DEVICES out=$OUT"
python "$ROOT/benchmarks/ultraquant_mtp/bench_pdv_verify_qlen8.py" \
  --ctx "${CTX:-131072,196608,229376}" \
  --qlens "${QLENS:-6,8}" \
  --out "$OUT"
rc=$?
echo "$(date -Is) E6 done rc=$rc"
exit $rc
