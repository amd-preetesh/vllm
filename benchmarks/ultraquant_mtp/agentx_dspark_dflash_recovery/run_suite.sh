#!/bin/bash
# AgentX DSpark/DFlash recovery suite — Phase 2 geometry + SWA + adaptive.
# Default: FP8 lane on HIP_A, UQ lane on HIP_B when both have enough free VRAM.
trap '' HUP
set -uo pipefail

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery
LOG_DIR=/workspace/logs/agentx_dspark_dflash_recovery
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/suite.log" 2>&1
log() { echo "$(date -Is) suite $*"; }

HIP_A="${HIP_A:-3}"
HIP_B="${HIP_B:-3}"
PORT_A="${PORT_A:-8720}"
PORT_B="${PORT_B:-8721}"
MODE="${MODE:-full}"

log "START recovery mode=$MODE hip_a=$HIP_A hip_b=$HIP_B"
if [ "$HIP_A" = "$HIP_B" ]; then
  # Serialize both lanes on one GPU.
  LANE=fp8 HIP_VISIBLE_DEVICES="$HIP_A" PORT="$PORT_A" MODE="$MODE" \
    bash "$HERE/run_lane.sh" || log "WARN fp8 lane rc=$?"
  LANE=uq HIP_VISIBLE_DEVICES="$HIP_B" PORT="$PORT_B" MODE="$MODE" \
    bash "$HERE/run_lane.sh" || log "WARN uq lane rc=$?"
else
  LANE=fp8 HIP_VISIBLE_DEVICES="$HIP_A" PORT="$PORT_A" MODE="$MODE" \
    bash "$HERE/run_lane.sh" &
  pid_a=$!
  LANE=uq HIP_VISIBLE_DEVICES="$HIP_B" PORT="$PORT_B" MODE="$MODE" \
    bash "$HERE/run_lane.sh" &
  pid_b=$!
  wait "$pid_a" || log "WARN fp8 lane rc=$?"
  wait "$pid_b" || log "WARN uq lane rc=$?"
fi
log "SUITE DONE"
python3 "$HERE/summarize_gates.py" || true
touch "$LOG_DIR/suite.DONE"
