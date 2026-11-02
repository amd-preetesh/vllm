#!/bin/bash
# Queue UQ bottleneck A/Bs on free HIP devices without touching Stream 2 (HIP3)
# or the P0-1 waiter (HIP1 / 8440).
#
# HIP0: E1 DSpark UQ mnbt2048 c32 → DFlash UQ mnbt2048 c32 → E4 dense smoke → dense AgentX c32
# HIP2: E2 DSpark UQ ret50k c32 → ret25k c32 → E5 accept probes
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-fix-queue.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
log() { echo "$(date -Is) $*"; }

log "uq fix queue start"

# --- lane A: HIP 0 / port 8450 ---
(
  export HIP_VISIBLE_DEVICES=0
  log "laneA start hip=0"
  EXP=mnbt2048 CFG=dspark_uq PORT=8450 CONCS="32" \
    bash "$HERE/run_uq_spec_ab_agentx.sh" || true
  EXP=mnbt2048 CFG=dflash_uq PORT=8450 CONCS="32" \
    bash "$HERE/run_uq_spec_ab_agentx.sh" || true
  MODE=dspark_uq_dense PORT=8452 \
    bash "$HERE/run_uq_dense_geometry_smoke.sh" || true
  if [ -f /workspace/vllm/out/p0_kv_geometry/dspark_uq_dense/geometry.json ]; then
    if python -c "import json,sys; d=json.load(open('/workspace/vllm/out/p0_kv_geometry/dspark_uq_dense/geometry.json')); sys.exit(0 if d.get('pass') else 1)"; then
      EXP=dense CFG=dspark_uq PORT=8450 CONCS="32" \
        bash "$HERE/run_uq_spec_ab_agentx.sh" || true
      EXP=dense CFG=dflash_uq PORT=8450 CONCS="32" \
        bash "$HERE/run_uq_spec_ab_agentx.sh" || true
    else
      log "laneA SKIP dense AgentX (geometry smoke failed)"
    fi
  fi
  log "laneA done"
) &
LANE_A=$!

# --- lane B: HIP 2 / port 8451 ---
(
  export HIP_VISIBLE_DEVICES=2
  log "laneB start hip=2"
  EXP=ret50k CFG=dspark_uq PORT=8451 CONCS="32" \
    bash "$HERE/run_uq_spec_ab_agentx.sh" || true
  EXP=ret25k CFG=dspark_uq PORT=8451 CONCS="32" \
    bash "$HERE/run_uq_spec_ab_agentx.sh" || true
  EXP=ret50k CFG=dflash_uq PORT=8451 CONCS="32" \
    bash "$HERE/run_uq_spec_ab_agentx.sh" || true
  TAG=dspark_uq PORT=8453 \
    bash "$HERE/run_uq_accept_probe.sh" || true
  TAG=dflash_uq PORT=8453 \
    bash "$HERE/run_uq_accept_probe.sh" || true
  TAG=dspark_fp8 PORT=8453 \
    bash "$HERE/run_uq_accept_probe.sh" || true
  log "laneB done"
) &
LANE_B=$!

wait "$LANE_A" || true
wait "$LANE_B" || true
log "uq fix queue done"
# Summarize vs baseline if helper exists
python3 /workspace/vllm/benchmarks/ultraquant_mtp/docs/uq_agentx_cell_analysis.py table \
  dspark_qwen38_uq/sa/seed42_c32 \
  dspark_qwen38_uq_mnbt2048/sa/seed42_c32 \
  dspark_qwen38_uq_ret50k/sa/seed42_c32 \
  2>/dev/null || true
