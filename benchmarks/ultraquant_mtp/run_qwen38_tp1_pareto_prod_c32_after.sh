#!/bin/bash
# Start Prod FlyDSL c=32 on HIP 2 after the c=8 Prod cell releases that GPU.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/kernel-profile-tp1-pareto-prod-c32-wait.log 2>&1
set -u
PRIOR=/workspace/logs/kernel-profile-tp1-pareto-qwen3-8-27b_ultraquant_prod_mtp5_tp1_c8.log
echo "$(date -Is) waiting for prod c=8 to finish"
while ! grep -q "pareto cell complete" "$PRIOR" 2>/dev/null; do
  if grep -q "FAIL " "$PRIOR" 2>/dev/null; then
    echo "$(date -Is) prod c=8 failed, not starting c=32"
    exit 1
  fi
  sleep 20
done
echo "$(date -Is) prod c=8 complete, pausing for GPU teardown"
sleep 25
export HIP_VISIBLE_DEVICES=2
export PROFILE_PORT=8078
export PARETO_NAME=qwen3-8-27b_ultraquant_prod_mtp5_tp1_c32
export PARETO_KV=ultraquant_4bit
export PARETO_PAIR=0
export PARETO_CONC=32
exec bash /workspace/vllm/benchmarks/ultraquant_mtp/run_qwen38_tp1_pareto_cell.sh
