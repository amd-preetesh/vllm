#!/bin/bash
# After Phase-1 lit_match suite.DONE, free the GPU and run Phase 2 on the same HIP:
#   1) DFlash SWA accept-vs-length probe
#   2) AgentX recovery suite (geom64 + mnbt8k + SWA + adaptive)
trap '' HUP
set -uo pipefail

ROOT=/workspace/vllm
LIT=$ROOT/benchmarks/ultraquant_mtp/lit_match_dspark_dflash
REC=$ROOT/benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery
LOG_DIR=/workspace/logs/dspark_dflash_handoff
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/handoff.log" 2>&1
log() { echo "$(date -Is) handoff $*"; }

HIP="${HIP:-3}"
PORT_PROBE="${PORT_PROBE:-8730}"
PORT_FP8="${PORT_FP8:-8720}"
PORT_UQ="${PORT_UQ:-8721}"
SUITE_DONE=/workspace/logs/lit_match_qwen38/suite.DONE
HANDOFF_DONE="$LOG_DIR/handoff.DONE"
STATUS_JSON=/workspace/vllm/out/recovery_progress.json

write_status() {
  local phase="$1" detail="$2"
  python3 - "$STATUS_JSON" "$phase" "$detail" "$HIP" <<'PY'
import json, sys, time
from pathlib import Path
path, phase, detail, hip = sys.argv[1:5]
p = Path(path)
prev = {}
if p.exists():
    try:
        prev = json.loads(p.read_text())
    except Exception:
        prev = {}
prev.update({
    "phase": phase,
    "detail": detail,
    "hip": int(hip),
    "updated_unix": time.time(),
    "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
})
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(prev, indent=2) + "\n")
PY
}

if [ -f "$HANDOFF_DONE" ]; then
  log "already done — exit"
  exit 0
fi

write_status "wait_phase1" "polling suite.DONE on HIP=$HIP"
log "WAIT suite.DONE hip=$HIP"
while [ ! -f "$SUITE_DONE" ]; do
  sleep 60
done
log "Phase1 DONE — starting Phase2 handoff"

# Ensure lit-match server is gone before reclaiming VRAM.
pkill -f "lit_match_dspark_dflash/run_cell" 2>/dev/null || true
pkill -f "port 8710" 2>/dev/null || true
if command -v fuser >/dev/null 2>&1; then
  fuser -k 8710/tcp 2>/dev/null || true
  fuser -k 8711/tcp 2>/dev/null || true
fi
sleep 20
write_status "phase2_swa_probe" "DFlash SWA accept-vs-length on HIP=$HIP"

log "SWA probe START"
HIP_VISIBLE_DEVICES="$HIP" PORT="$PORT_PROBE" \
  bash "$REC/run_swa_probe.sh" || log "WARN swa probe rc=$?"
write_status "phase2_agentx" "AgentX recovery suite on HIP=$HIP (fp8 then uq)"

log "AgentX recovery START"
HIP_A="$HIP" HIP_B="$HIP" PORT_A="$PORT_FP8" PORT_B="$PORT_UQ" MODE=full \
  bash "$REC/run_suite.sh" || log "WARN recovery suite rc=$?"

python3 "$REC/summarize_gates.py" || true
python3 "$REC/refresh_canvas.py" || true
python3 "$LIT/refresh_canvas.py" || true
python3 "$LIT/summarize_gates.py" || true

write_status "done" "Phase2 handoff complete"
date -Is >"$HANDOFF_DONE"
log "ALL DONE"
