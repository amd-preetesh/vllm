#!/bin/bash
# Method-match Spec-Bench + SPEED-747 queue.
# Hard caps: at most MAX_GPUS (default 2) concurrent TP=1 servers.
# Survives SSH/Cursor disconnect (trap HUP, nohup-friendly).
#
#   MAX_GPUS=2 REPS=3 CONCS="1 8 16 32 64 128 256" \
#     bash benchmarks/ultraquant_mtp/method_match/run_queue.sh
#
# Auto-handoff: workers claim free GPUs via gpu_claim.sh; finished cells
# release the card so the next job can pick it up.
trap '' HUP
set -uo pipefail

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp/method_match
CLAIM=$ROOT/benchmarks/ultraquant_mtp/gpu_claim.sh
VENV=/workspace/.venv-uq
LOG=/workspace/logs/method-match-queue.log
STATE=/workspace/logs/method-match-queue.state
JOBS=/workspace/logs/method-match-jobs.txt
LOCK=/workspace/logs/method-match-jobs.lock
DONEFLAG=/workspace/logs/method-match-queue.DONE
mkdir -p /workspace/logs "$HERE/datasets"
exec >>"$LOG" 2>&1

export PATH="$VENV/bin:$PATH"
export PYTHONPATH=/workspace/aiter:/workspace/vllm
export PYTHONNOUSERSITE=1

MAX_GPUS="${MAX_GPUS:-2}"
REPS="${REPS:-3}"
CONCS=(${CONCS:-1 8 16 32 64 128 256})
ARMS=(${ARMS:-dspark_fp8 dflash_fp8 dspark_uq_opt dflash_uq_swa})
WORKLOADS=(${WORKLOADS:-spec_bench speed747})
PORT_BASE="${PORT_BASE:-8700}"
NEED_GIB="${NEED_GIB:-200}"
# Prefer nearly-empty cards; claim_gpu already scores free VRAM.
POLL_SEC="${POLL_SEC:-45}"

log() { echo "$(date -Is) $*"; }
mark() { echo "$1" >"$STATE"; log "STATE $1"; }

source "$CLAIM"

# claim_gpu that refuses HIPs already owned by our method-match / other waiters.
claim_gpu_exclusive() {
  local need="$1" tries=0 hip
  while [ "$tries" -lt 8 ]; do
    tries=$((tries + 1))
    if ! claim_gpu "$need"; then
      return 1
    fi
    hip="$HIP_VISIBLE_DEVICES"
    # If another worker already owns this HIP (claim file with different pid), release and retry.
    local cf="/workspace/logs/gpu_claims/hip${hip}.claim"
    if [ -f "$cf" ]; then
      local owner
      owner=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^pid=/) print substr($i,5)}' "$cf" 2>/dev/null || true)
      if [ -n "$owner" ] && [ "$owner" != "$$" ] && kill -0 "$owner" 2>/dev/null; then
        log "hip=$hip already claimed by pid=$owner — release and retry"
        # Remove our overwrite and unlock
        release_gpu 2>/dev/null || true
        # restore foreign claim if we clobbered — best effort: skip this hip via busy file
        echo "busy" >"/workspace/logs/gpu_claims/hip${hip}.busy"
        sleep 2
        continue
      fi
    fi
    # Refresh claim with our orchestrator pid; worker will keep the file until done.
    echo "$(date -Is) pid=$$ hip=$hip need=$need role=method_match_queue" \
      >"/workspace/logs/gpu_claims/hip${hip}.claim"
    rm -f "/workspace/logs/gpu_claims/hip${hip}.busy" 2>/dev/null || true
    return 0
  done
  return 1
}

# ---- prepare datasets once ----
mark "prepare_datasets"
if [ ! -f "$HERE/datasets/MANIFEST.json" ] \
  || [ ! -f "$HERE/datasets/spec_bench_first_turn.jsonl" ] \
  || [ ! -f "$HERE/datasets/speed_bench_throughput_1k_balanced_747.jsonl" ]; then
  python3 "$HERE/prepare_datasets.py" || {
    log "FAIL prepare_datasets"
    mark "fail_prepare"
    exit 1
  }
else
  log "datasets already present"
fi

# ---- build / refresh job list (skip completed) ----
mark "build_jobs"
: >"$JOBS.tmp"
for wl in "${WORKLOADS[@]}"; do
  for arm in "${ARMS[@]}"; do
    for c in "${CONCS[@]}"; do
      for r in $(seq 1 "$REPS"); do
        stem=
        case "$arm" in
          dspark_fp8) stem=method_match_dspark_fp8 ;;
          dflash_fp8) stem=method_match_dflash_fp8 ;;
          dspark_uq_opt) stem=method_match_dspark_uq_opt ;;
          dflash_uq_swa) stem=method_match_dflash_uq_swa ;;
        esac
        out="$ROOT/out/$stem/$wl/c${c}_r${r}"
        if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ] \
          && [ -f "$out/accept_metrics.json" ]; then
          continue
        fi
        echo "$arm $wl $c $r" >>"$JOBS.tmp"
      done
    done
  done
done
mv -f "$JOBS.tmp" "$JOBS"
nj=$(wc -l <"$JOBS" | tr -d ' ')
log "queue start max_gpus=$MAX_GPUS reps=$REPS pending_jobs=$nj concs=${CONCS[*]} arms=${ARMS[*]} workloads=${WORKLOADS[*]} pid=$$"
mark "running:$nj"

pop_job() {
  # Atomic pop first line from JOBS under flock.
  python3 - <<'PY'
import fcntl, pathlib, os
path = pathlib.Path(os.environ["JOBS"])
lockp = pathlib.Path(os.environ["LOCK"])
lockp.touch(exist_ok=True)
with lockp.open("a+") as lf:
    fcntl.flock(lf, fcntl.LOCK_EX)
    if not path.exists():
        raise SystemExit(1)
    lines = path.read_text().splitlines()
    if not lines:
        raise SystemExit(1)
    job, rest = lines[0], lines[1:]
    path.write_text("\n".join(rest) + ("\n" if rest else ""))
    print(job)
PY
}

# Track worker PIDs → HIP
declare -A WORKER_HIP=()
declare -A WORKER_PORT=()
ACTIVE=0
PORT_I=0

launch_worker() {
  local job="$1" hip="$2" port="$3"
  local arm wl c r
  read -r arm wl c r <<<"$job"
  log "LAUNCH arm=$arm wl=$wl c=$c r=$r hip=$hip port=$port"
  (
    trap '' HUP
    export HIP_VISIBLE_DEVICES="$hip" CUDA_VISIBLE_DEVICES="$hip"
    unset ROCR_VISIBLE_DEVICES
    # Hold claim file for this HIP while cell runs (claim_gpu already wrote it).
    ARM="$arm" WORKLOAD="$wl" CONC="$c" REP="$r" PORT="$port" \
      bash "$HERE/run_cell.sh"
    rc=$?
    # Release this HIP
    rm -f "/workspace/logs/gpu_claims/hip${hip}.claim" 2>/dev/null || true
    echo "$(date -Is) WORKER_DONE arm=$arm wl=$wl c=$c r=$r hip=$hip rc=$rc" \
      >>/workspace/logs/method-match-worker-done.log
    exit "$rc"
  ) &
  local wpid=$!
  WORKER_HIP[$wpid]=$hip
  WORKER_PORT[$wpid]=$port
  ACTIVE=$((ACTIVE + 1))
  log "worker pid=$wpid active=$ACTIVE"
}

reap_workers() {
  local pid hip
  for pid in "${!WORKER_HIP[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      hip="${WORKER_HIP[$pid]}"
      log "reaped worker=$pid hip=$hip"
      unset "WORKER_HIP[$pid]"
      unset "WORKER_PORT[$pid]"
      ACTIVE=$((ACTIVE - 1))
      # ensure claim cleared
      rm -f "/workspace/logs/gpu_claims/hip${hip}.claim" 2>/dev/null || true
    fi
  done
}

# Main loop
while true; do
  reap_workers
  pending=$(wc -l <"$JOBS" 2>/dev/null | tr -d ' ')
  pending=${pending:-0}
  mark "running:pending=$pending:active=$ACTIVE"

  if [ "$pending" = 0 ] && [ "$ACTIVE" = 0 ]; then
    log "ALL JOBS DONE"
    mark "done"
    date -Is >"$DONEFLAG"
    python3 "$HERE/summarize.py" || true
    exit 0
  fi

  # At most one new claim per poll cycle so the previous server has time
  # to allocate VRAM before we pick the next free card (max 2 active).
  if [ "$ACTIVE" -lt "$MAX_GPUS" ] && [ "$pending" -gt 0 ]; then
    export JOBS LOCK
    if job=$(JOBS="$JOBS" LOCK="$LOCK" pop_job); then
      if claim_gpu_exclusive "$NEED_GIB"; then
        hip="$HIP_VISIBLE_DEVICES"
        if [ -n "${_CLAIM_FD:-}" ]; then
          flock -u "$_CLAIM_FD" 2>/dev/null || true
        fi
        unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
        echo "$(date -Is) pid=$$ hip=$hip need=$NEED_GIB role=method_match_worker" \
          >"/workspace/logs/gpu_claims/hip${hip}.claim"
        port=$((PORT_BASE + PORT_I))
        PORT_I=$((PORT_I + 1))
        launch_worker "$job" "$hip" "$port"
        # Give vLLM time to grab VRAM before the next claim poll.
        sleep 90
      else
        python3 - <<PY
import fcntl, pathlib
job = """$job"""
path = pathlib.Path("$JOBS")
lockp = pathlib.Path("$LOCK")
lockp.touch(exist_ok=True)
with lockp.open("a+") as lf:
    fcntl.flock(lf, fcntl.LOCK_EX)
    rest = path.read_text() if path.exists() else ""
    path.write_text(job.strip() + "\n" + rest)
PY
        log "no free GPU (need ${NEED_GIB}GiB); requeued: $job"
        release_gpu 2>/dev/null || true
      fi
    fi
  fi

  sleep "$POLL_SEC"
done
