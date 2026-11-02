#!/bin/bash
# Launch Stream-2 AgentX conc sweep on free GPUs (default 1/2/3).
# Host-side: docker exec into preshah_vllm_ultraquant.
set -euo pipefail
CTR="${CTR:-preshah_vllm_ultraquant}"
SCRIPT=/workspace/vllm/benchmarks/ultraquant_mtp/run_dflash_dspark_agentx_conc_sweep.sh

HIP_A="${HIP_A:-1}"
HIP_B="${HIP_B:-2}"
HIP_C="${HIP_C:-3}"
PORT_A="${PORT_A:-8410}"
PORT_B="${PORT_B:-8420}"
PORT_C="${PORT_C:-8430}"

docker exec "$CTR" bash -lc "chmod +x $SCRIPT"
for role_hip_port in "a:${HIP_A}:${PORT_A}" "b:${HIP_B}:${PORT_B}" "c:${HIP_C}:${PORT_C}"; do
  role="${role_hip_port%%:*}"
  rest="${role_hip_port#*:}"
  hip="${rest%%:*}"
  port="${rest##*:}"
  echo "launch ROLE=$role HIP=$hip PORT=$port"
  docker exec -d "$CTR" bash -lc \
    "HIP_VISIBLE_DEVICES=$hip PORT=$port ROLE=$role nohup bash $SCRIPT >/dev/null 2>&1 &"
done

echo "logs:"
echo "  /workspace/logs/dflash-dspark-agentx-conc-a.log  # dflash_fp8"
echo "  /workspace/logs/dflash-dspark-agentx-conc-b.log  # dspark_fp8"
echo "  /workspace/logs/dflash-dspark-agentx-conc-c.log  # dflash_uq + dspark_uq"
echo "outs: vllm/out/{dflash,dspark}_qwen38_{fp8,uq}/sa/seed42_c{8,16,20,24,30}/"
