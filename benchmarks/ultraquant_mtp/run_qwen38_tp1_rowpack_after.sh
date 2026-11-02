#!/bin/bash
# Wait until the 2-wave weekend matrix releases HIP 0, then serve row-pack.
trap '' HUP
PRIOR=/workspace/logs/kernel-profile-tp1-weekend.log
echo "$(date -Is) rowpack follower waiting for weekend compare complete" >> /workspace/logs/kernel-profile-tp1-rowpack.log
while ! grep -q "weekend compare complete" "$PRIOR" 2>/dev/null; do
  sleep 60
done
echo "$(date -Is) rowpack follower starting" >> /workspace/logs/kernel-profile-tp1-rowpack.log
export WEEKEND_LOG=/workspace/logs/kernel-profile-tp1-rowpack.log
export WEEKEND_OUT=/workspace/vllm/out/kernel_profile_tp1_rowpack
export WEEKEND_MANIFEST=/workspace/vllm/benchmarks/ultraquant_mtp/weekend_rowpack_manifest.tsv
export WEEKEND_ROWPACK=1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
exec /workspace/vllm/benchmarks/ultraquant_mtp/run_qwen38_tp1_weekend.sh
