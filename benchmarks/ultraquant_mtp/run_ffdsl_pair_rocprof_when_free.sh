#!/bin/bash
# Start the detailed Fused FlyDSL rocprof on the first GPU that stays idle.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/ffdsl-pair-rocprof-detail-wait.log 2>&1
set -u

log() { echo "$(date -Is) $*"; }

# rocm-smi GPU index -> HIP index on this host.
hip_for_gpu() {
  case "$1" in
    0) echo 1 ;;
    1) echo 3 ;;
    2) echo 2 ;;
    3) echo 0 ;;
    4) echo 4 ;;
    5) echo 7 ;;
    6) echo 6 ;;
    7) echo 5 ;;
    *) return 1 ;;
  esac
}

idle_gpu() {
  rocm-smi --showuse | awk '
    /GPU\[[0-9]\]/ {
      idx=$1; gsub(/GPU\[|\]/,"",idx)
      use=$NF
      gsub(/%/,"",use)
      if (use+0 == 0) { print idx; exit }
    }
  '
}

log "waiting for an idle GPU"
last=""
while true; do
  gpu=$(idle_gpu || true)
  if [ -n "${gpu}" ] && [ "$gpu" = "$last" ]; then
    hip=$(hip_for_gpu "$gpu")
    log "GPU ${gpu} idle twice, starting detail rocprof on HIP ${hip}"
    export HIP_VISIBLE_DEVICES="$hip"
    exec bash /workspace/vllm/benchmarks/ultraquant_mtp/run_ffdsl_pair_rocprof_detail.sh
  fi
  last="$gpu"
  log "idle=${gpu:-none}"
  sleep 30
done
