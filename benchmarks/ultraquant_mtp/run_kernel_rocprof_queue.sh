#!/bin/bash
# Start the three kernel profiles on GPUs that stay idle. One profile per GPU.
trap '' HUP
mkdir -p /workspace/logs /workspace/logs/rocprof_gpu_locks
exec >> /workspace/logs/kernel-prof-queue.log 2>&1
set -u

log() { echo "$(date -Is) $*"; }

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

idle_gpus() {
  rocm-smi --showuse | awk '
    /GPU\[[0-9]\]/ {
      idx=$1; gsub(/GPU\[|\]/,"",idx)
      use=$NF; gsub(/%/,"",use)
      if (use+0 == 0) print idx
    }
  '
}

script_for() {
  case "$1" in
    fused) echo /workspace/vllm/benchmarks/ultraquant_mtp/run_ffdsl_pair_rocprof_e2e.sh ;;
    fp8) echo /workspace/vllm/benchmarks/ultraquant_mtp/run_fp8_aiter_rocprof.sh ;;
    prod) echo /workspace/vllm/benchmarks/ultraquant_mtp/run_prod_flydsl_rocprof.sh ;;
  esac
}

pending="fused fp8 prod"
declare -A seen_idle=()
log "queue start pending=${pending}"

while [ -n "${pending// /}" ]; do
  now=$(idle_gpus | tr '\n' ' ')
  log "idle=${now:-none} pending=${pending}"
  next_pending=""
  for job in $pending; do
    claimed=""
    for gpu in $now; do
      [ -n "${seen_idle[$gpu]:-}" ] || continue
      lock="/workspace/logs/rocprof_gpu_locks/gpu${gpu}"
      if mkdir "$lock" 2>/dev/null; then
        hip=$(hip_for_gpu "$gpu")
        log "GPU ${gpu} idle twice, starting ${job} on HIP ${hip}"
        HIP_VISIBLE_DEVICES="$hip" setsid bash "$(script_for "$job")" &
        echo $! > "${lock}/pid"
        echo "$job" > "${lock}/job"
        claimed=1
        unset "seen_idle[$gpu]"
        break
      fi
    done
    if [ -z "$claimed" ]; then
      next_pending="${next_pending} ${job}"
    fi
  done
  pending="${next_pending# }"
  for gpu in $now; do
    seen_idle[$gpu]=1
  done
  # Drop idle memory for GPUs that are busy again.
  if [ "${#seen_idle[@]}" -gt 0 ]; then
    for gpu in "${!seen_idle[@]}"; do
      case " $now " in
        *" $gpu "*) ;;
        *) unset "seen_idle[$gpu]" ;;
      esac
    done
  fi
  if [ -n "${pending// /}" ]; then
    sleep 30
  fi
done

log "all three profiles launched, waiting"
wait
log "queue complete"
touch /workspace/logs/kernel-rocprof-queue.ok
