#!/bin/bash
# Dual-GPU AgentX baseline suite: smoke → full ladders.
# Claims two free GPUs (≥NEED_GIB), FP8 lane on one, UQ lane on the other.
trap '' HUP
set -uo pipefail

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp/agentx_baseline_matrix
# shellcheck source=/dev/null
source "$ROOT/benchmarks/ultraquant_mtp/gpu_claim.sh"

LOG=/workspace/logs/agentx_baseline_qwen38/suite.log
NEED_GIB="${NEED_GIB:-250}"
FP8_PORT="${FP8_PORT:-8610}"
UQ_PORT="${UQ_PORT:-8620}"
mkdir -p /workspace/logs/agentx_baseline_qwen38
exec >>"$LOG" 2>&1
log() { echo "$(date -Is) $*"; }

pick_two() {
  NEED_GIB="$NEED_GIB" python3 - <<'PY'
import os, torch
need = float(os.environ["NEED_GIB"]) * (1024**3)
cands = []
for i in range(torch.cuda.device_count()):
    try:
        free, total = torch.cuda.mem_get_info(i)
    except Exception:
        continue
    if free >= need:
        score = free + (1e12 if free / total >= 0.85 else 0)
        cands.append((score, i, free))
cands.sort(reverse=True)
if len(cands) < 2:
    raise SystemExit(1)
print(cands[0][1], cands[1][1])
PY
}

log "suite start need_gib=$NEED_GIB pid=$$"

for try in $(seq 1 2880); do
  hips=$(pick_two 2>/dev/null) || {
    if [ $((try % 10)) -eq 1 ]; then
      log "waiting for 2×≥${NEED_GIB}GiB free GPUs try=$try"
    fi
    sleep 30
    continue
  }
  # shellcheck disable=SC2086
  set -- $hips
  FP8_HIP=$1
  UQ_HIP=$2
  echo "$(date -Is) pid=$$ hip=$FP8_HIP need=$NEED_GIB role=agentx_baseline_fp8" \
    >"$CLAIM_LOCK_DIR/hip${FP8_HIP}.claim"
  echo "$(date -Is) pid=$$ hip=$UQ_HIP need=$NEED_GIB role=agentx_baseline_uq" \
    >"$CLAIM_LOCK_DIR/hip${UQ_HIP}.claim"
  log "claimed fp8=hip$FP8_HIP uq=hip$UQ_HIP — SMOKE"

  # Parallel smoke
  HIP_VISIBLE_DEVICES=$FP8_HIP PORT=$FP8_PORT LANE=fp8 MODE=smoke \
    bash "$HERE/run_lane.sh" &
  FP8_PID=$!
  HIP_VISIBLE_DEVICES=$UQ_HIP PORT=$UQ_PORT LANE=uq MODE=smoke \
    bash "$HERE/run_lane.sh" &
  UQ_PID=$!
  wait "$FP8_PID"
  FP8_RC=$?
  wait "$UQ_PID"
  UQ_RC=$?
  log "smoke done fp8_rc=$FP8_RC uq_rc=$UQ_RC"
  python3 "$HERE/refresh_progress.py" 2>/dev/null || true

  if [ "$FP8_RC" -ne 0 ] || [ "$UQ_RC" -ne 0 ]; then
    log "WARN smoke had failures — continuing to full on healthy lane(s)"
  fi

  log "FULL ladders start"
  HIP_VISIBLE_DEVICES=$FP8_HIP PORT=$FP8_PORT LANE=fp8 MODE=full \
    bash "$HERE/run_lane.sh" &
  FP8_PID=$!
  HIP_VISIBLE_DEVICES=$UQ_HIP PORT=$UQ_PORT LANE=uq MODE=full \
    bash "$HERE/run_lane.sh" &
  UQ_PID=$!
  wait "$FP8_PID"
  FP8_RC=$?
  wait "$UQ_PID"
  UQ_RC=$?
  log "full done fp8_rc=$FP8_RC uq_rc=$UQ_RC"
  rm -f "$CLAIM_LOCK_DIR/hip${FP8_HIP}.claim" "$CLAIM_LOCK_DIR/hip${UQ_HIP}.claim"
  python3 "$HERE/refresh_progress.py" 2>/dev/null || true
  date -Is >/workspace/logs/agentx_baseline_qwen38/suite.DONE
  log "ALL DONE"
  exit 0
done
log "FAIL timed out waiting for 2 GPUs"
exit 1
