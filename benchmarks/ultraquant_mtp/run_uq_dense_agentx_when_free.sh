#!/bin/bash
# Wait for a GPU with enough free VRAM, then run dense AgentX (DSpark → DFlash c32).
# Does not kill other users' jobs.
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-dense-agentx-waiter.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
PORT="${PORT:-8450}"
NEED_GIB="${NEED_GIB:-200}"   # UTIL=0.70 needs ~201 GiB free on 288 GiB cards
UTIL="${UTIL:-0.70}"
CONCS="${CONCS:-32}"

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

log "dense AgentX waiter start need_gib=$NEED_GIB util=$UTIL port=$PORT"
for i in $(seq 1 720); do  # up to 6h @ 30s
  if HIP=$(pick_gpu 2>/dev/null); then
    free=$(python -c "import torch; f,t=torch.cuda.mem_get_info($HIP); print(f'{f/1024**3:.1f}')")
    log "GPU $HIP free=${free}GiB — launching dense AgentX"
    export HIP_VISIBLE_DEVICES="$HIP"
    EXP=dense CFG=dspark_uq PORT="$PORT" CONCS="$CONCS" UTIL="$UTIL" \
      bash "$HERE/run_uq_spec_ab_agentx.sh" || log "WARN dspark dense rc=$?"
    EXP=dense CFG=dflash_uq PORT="$PORT" CONCS="$CONCS" UTIL="$UTIL" \
      bash "$HERE/run_uq_spec_ab_agentx.sh" || log "WARN dflash dense rc=$?"
    log "dense AgentX waiter done"
    exit 0
  fi
  if [ $((i % 10)) -eq 1 ]; then
    log "still waiting for >=${NEED_GIB}GiB free GPU (try $i)"
  fi
  sleep 30
done
log "FAIL timed out waiting for free GPU"
exit 1
