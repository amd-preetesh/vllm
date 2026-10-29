#!/bin/bash
# Queue lit-match FP8 cells across two free GPUs (default HIP 1 and 7).
# Arms × C={1,8,32}: nospec, mtp, dspark, dflash = 12 cells.
trap '' HUP
set -uo pipefail

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp/lit_match_dspark_dflash
LOG_DIR=/workspace/logs/lit_match_qwen38
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/suite.log" 2>&1
log() { echo "$(date -Is) suite $*"; }

# Default single GPU with enough free VRAM (override HIP_A/HIP_B if two free).
HIP_A="${HIP_A:-3}"
HIP_B="${HIP_B:-}"
PORT_A="${PORT_A:-8710}"
PORT_B="${PORT_B:-8711}"

# Prefer nospec early so speedup gates can be evaluated as soon as peers finish.
QUEUE=(
  "fp8_nospec:1"
  "fp8_mtp:1"
  "fp8_dspark:1"
  "fp8_dflash:1"
  "fp8_nospec:8"
  "fp8_mtp:8"
  "fp8_dspark:8"
  "fp8_dflash:8"
  "fp8_nospec:32"
  "fp8_mtp:32"
  "fp8_dspark:32"
  "fp8_dflash:32"
)

run_queue() {
  local hip="$1" port="$2"
  shift 2
  local cell arm conc
  for cell in "$@"; do
    arm="${cell%%:*}"
    conc="${cell##*:}"
    log "LAUNCH hip=$hip port=$port $arm c$conc"
    HIP_VISIBLE_DEVICES="$hip" PORT="$port" ARM="$arm" CONC="$conc" UTIL="${UTIL:-0.55}" \
      bash "$HERE/run_cell.sh" || log "WARN cell failed $arm c$conc"
  done
  log "LANE hip=$hip complete"
}

log "START lit_match hip_a=$HIP_A hip_b=${HIP_B:-none}"
if [ -n "$HIP_B" ] && [ "$HIP_B" != "$HIP_A" ]; then
  # Split: A gets odd indices, B even — keep nospec on A first.
  QA=(); QB=()
  i=0
  for cell in "${QUEUE[@]}"; do
    if (( i % 2 == 0 )); then QA+=("$cell"); else QB+=("$cell"); fi
    i=$((i + 1))
  done
  run_queue "$HIP_A" "$PORT_A" "${QA[@]}" &
  pid_a=$!
  run_queue "$HIP_B" "$PORT_B" "${QB[@]}" &
  pid_b=$!
  wait "$pid_a" || log "WARN lane A rc=$?"
  wait "$pid_b" || log "WARN lane B rc=$?"
else
  run_queue "$HIP_A" "$PORT_A" "${QUEUE[@]}"
fi
log "SUITE DONE"
python3 "$HERE/summarize_gates.py" || true
touch "$LOG_DIR/suite.DONE"
