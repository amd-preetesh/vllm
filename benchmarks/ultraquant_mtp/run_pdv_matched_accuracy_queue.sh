#!/bin/bash
# Queue all 4 matched accuracy cells. Uses two free GPUs in parallel:
#   HIP_A (default 5): fp8 then prod
#   HIP_B (default 6): pdv then hoist_k
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/pdv-matched-acc-queue.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
HIP_A="${HIP_A:-5}"
HIP_B="${HIP_B:-6}"
PORT_A="${PORT_A:-8401}"
PORT_B="${PORT_B:-8402}"

log() { echo "$(date -Is) $*"; }

run_pair() {
  local hip="$1" port="$2" m1="$3" m2="$4"
  log "PAIR start hip=$hip port=$port modes=$m1,$m2"
  HIP_VISIBLE_DEVICES="$hip" PDV_ACC_PORT="$port" MODE="$m1" \
    bash "$HERE/run_pdv_matched_accuracy_cell.sh"
  HIP_VISIBLE_DEVICES="$hip" PDV_ACC_PORT="$port" MODE="$m2" \
    bash "$HERE/run_pdv_matched_accuracy_cell.sh"
  log "PAIR done hip=$hip"
}

log "matched-acc QUEUE start hip_a=$HIP_A hip_b=$HIP_B"
run_pair "$HIP_A" "$PORT_A" fp8 prod &
pid_a=$!
run_pair "$HIP_B" "$PORT_B" pdv hoist_k &
pid_b=$!
wait "$pid_a"; rc_a=$?
wait "$pid_b"; rc_b=$?
log "matched-acc QUEUE done rc_a=$rc_a rc_b=$rc_b"

# Summary table
python - <<'PY'
import json, os
root="/workspace/vllm/out/pdv_matched_accuracy_seed7"
modes=["fp8","prod","pdv","hoist_k"]
print("mode       GSM8K%   GPQA%    gsm_ok gpqa_ok")
for m in modes:
  g=os.path.join(root,m,"gsm8k_seed7","summary.json")
  q=os.path.join(root,m,"gpqa_seed7","summary.json")
  ga=json.load(open(g))["accuracy"] if os.path.isfile(g) else None
  qa=json.load(open(q))["accuracy"] if os.path.isfile(q) else None
  print(f"{m:10} {ga if ga is not None else '—':>7} {qa if qa is not None else '—':>7}   {bool(ga):g} {bool(qa):g}")
PY
