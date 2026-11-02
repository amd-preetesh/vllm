#!/bin/bash
# Stream 2: rocprofv3 on the FFDSL pair kernel (two-wave shared verify).
# Isolated attention launches only. Does not start a vLLM server.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/ffdsl-pair-rocprof.log 2>&1
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
OUT="${ROCPROF_OUT:-/workspace/logs/ffdsl_pair_rocprof}"
SEQ="${FFDSL_PROF_SEQ:-131072}"
ITERS="${FFDSL_PROF_ITERS:-8}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-2}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_VERIFY=1
export VLLM_ULTRAQUANT_PAIR_WAVES=2
export FFDSL_PROF_SEQ="$SEQ"
export FFDSL_PROF_ITERS="$ITERS"
export FFDSL_PROF_WARMUP=0

mkdir -p "$OUT"
cd "$ROOT"

log() { echo "$(date -Is) $*"; }

log "ffdsl pair rocprof start pid=$$ ppid=$PPID hip=${HIP_VISIBLE_DEVICES} seq=${SEQ} iters=${ITERS} out=${OUT}"

# Kernel dispatch + stats first (always useful).
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

# Hardware counters that match the in-tree rocprof notes (occupancy / MFMA / VALU).
rocprofv3 \
  --kernel-include-regex 'ultraquant|flydsl|hd256|shared' \
  --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_MFMA TCC_EA_RDREQ_sum TCC_EA_WRREQ_sum \
  -d "$OUT" \
  -o ffdsl_pair_pmc \
  -f csv json \
  -- \
  "$VENV/bin/python" /workspace/logs/bench_ffdsl_pair_rocprof.py \
  > "$OUT/pmc_stdout.log" 2>&1 \
  || log "WARN pmc rc=$? (counter names may differ on this ROCm; kernel-trace still stands)"

python3 - <<'PY'
from pathlib import Path
import csv
out = Path("/workspace/logs/ffdsl_pair_rocprof")
stats = sorted(out.glob("**/*kernel_stats*.csv")) + sorted(out.glob("**/*stats*.csv"))
print(f"stats files: {len(stats)}")
for p in stats[:6]:
    print(f"--- {p}")
    rows = list(csv.DictReader(p.open()))
    rows.sort(key=lambda r: float(r.get("TotalDurationNs") or r.get("DurationNs") or 0), reverse=True)
    for r in rows[:12]:
        name = r.get("Name") or r.get("KernelName") or "?"
        dur = r.get("TotalDurationNs") or r.get("DurationNs") or "?"
        calls = r.get("Calls") or r.get("CallCount") or "?"
        print(f"{calls:>6}  {dur:>16}  {name[:100]}")
PY

log "ffdsl pair rocprof complete"
touch /workspace/logs/ffdsl-pair-rocprof.ok
