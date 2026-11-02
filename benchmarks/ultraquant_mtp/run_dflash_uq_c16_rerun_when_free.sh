#!/bin/bash
# Rerun DFlash UQ AgentX baseline c16 (suspected outlier: 4.0 tok/s vs neighbors).
# Quarantines the old cell, then fills seed42_c16 again when a GPU is free.
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-dflash-c16-rerun.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
OUT=/workspace/vllm/out/dflash_qwen38_uq/sa
PORT="${PORT:-8464}"
NEED_GIB="${NEED_GIB:-200}"
UTIL="${UTIL:-0.70}"

log() { echo "$(date -Is) $*"; }

pick_gpu() {
  python - <<PY
import torch
need = ${NEED_GIB} * (1024**3)
best = None
for i in range(8):
    try:
        free, total = torch.cuda.mem_get_info(i)
    except Exception:
        continue
    if free >= need and (best is None or free > best[1]):
        best = (i, free)
if best is None:
    raise SystemExit(1)
print(best[0])
PY
}

confirm_free() {
  python - <<PY
import time, torch
need = ${NEED_GIB} * (1024**3)
for sample in (1, 2):
    free, total = torch.cuda.mem_get_info(0)
    print(f"confirm sample{sample} free_gib={free/1024**3:.1f}", flush=True)
    if free < need:
        raise SystemExit(1)
    if sample == 1:
        time.sleep(30)
PY
}

# Quarantine old cell once
OLD="$OUT/seed42_c16"
SUS="$OUT/seed42_c16_suspect_20261021"
if [ -d "$OLD" ] && [ ! -d "$SUS" ]; then
  mv "$OLD" "$SUS"
  log "quarantined $OLD → $SUS"
elif [ -d "$SUS" ]; then
  log "suspect already quarantined at $SUS"
  # If a partial new cell exists without profile, leave it; else ensure empty for rerun
  if [ -f "$OLD/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "SKIP already have new c16 profile"
    exit 0
  fi
fi

log "dflash UQ c16 rerun waiter start need_gib=$NEED_GIB util=$UTIL port=$PORT"
for i in $(seq 1 1440); do
  if HIP=$(pick_gpu 2>/dev/null); then
    free=$(python -c "import torch; f,t=torch.cuda.mem_get_info($HIP); print(f'{f/1024**3:.1f}')")
    log "candidate GPU $HIP free=${free}GiB — confirming"
    export HIP_VISIBLE_DEVICES="$HIP"
    export CUDA_VISIBLE_DEVICES="$HIP"
    if ! confirm_free >>"$LOG" 2>&1; then
      log "WARN GPU $HIP failed confirm — retry"
      unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
      sleep 30
      continue
    fi
    log "GPU $HIP confirmed — launching dflash_uq baseline c16"
    EXP=baseline CFG=dflash_uq PORT="$PORT" CONCS="16" UTIL="$UTIL" \
      bash "$HERE/run_uq_spec_ab_agentx.sh" || log "WARN rerun rc=$?"
    if [ -f "$OUT/seed42_c16/aiperf_artifacts/profile_export_aiperf.json" ]; then
      log "dflash UQ c16 rerun DONE"
      exit 0
    fi
    log "FAIL no profile after rerun"
    exit 1
  fi
  if [ $((i % 10)) -eq 1 ]; then
    log "still waiting for >=${NEED_GIB}GiB (try $i)"
  fi
  sleep 30
done
log "FAIL timed out waiting for free GPU"
exit 1
