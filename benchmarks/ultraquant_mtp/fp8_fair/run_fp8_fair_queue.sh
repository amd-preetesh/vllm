#!/bin/bash
# Queue fair FP8 KV8 AgentX for DSpark + DFlash (ret50k + mnbt2048 + block64;
# DFlash SWA draft). Up to MAX_GPUS=2 concurrent. Survives HUP.
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/fp8-fair-queue.log
STATE=/workspace/logs/fp8-fair-queue.state
DONEFLAG=/workspace/logs/fp8-fair-queue.DONE
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
ROOT=/workspace/vllm
# shellcheck source=/dev/null
source "$HERE/gpu_claim.sh"

MAX_GPUS="${MAX_GPUS:-2}"
NEED_GIB="${NEED_GIB:-200}"
UTIL="${UTIL:-0.85}"
CONCS="${CONCS:-8 16 20 24 30 32}"
PORT_BASE="${PORT_BASE:-8470}"
POLL_SEC="${POLL_SEC:-40}"
CFGS=(${CFGS:-dspark dflash})

log() { echo "$(date -Is) $*"; }
mark() { echo "$1" >"$STATE"; log "STATE $1"; }

cfg_done() {
  local cfg="$1" c out stem
  stem="${cfg}_qwen38_fp8_fair"
  for c in $CONCS; do
    out=$ROOT/out/$stem/sa/seed42_c${c}
    if [ ! -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
      return 1
    fi
  done
  return 0
}

pending_cfgs() {
  local c
  for c in "${CFGS[@]}"; do
    cfg_done "$c" || echo "$c"
  done
}

refresh_pareto() {
  python3 /home/preshah/workspace/logs/refresh-dflash-dspark-agentx-pareto.py 2>/dev/null || true
}

log "fp8 fair queue start max_gpus=$MAX_GPUS concs=$CONCS need=$NEED_GIB pid=$$"
mark "start"
rm -f "$DONEFLAG"

declare -A WPID=()
declare -A WHIP=()
ACTIVE=0
PORT_I=0

reap() {
  local pid
  for pid in "${!WPID[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      local hip="${WHIP[$pid]}"
      log "reaped cfg=${WPID[$pid]} hip=$hip"
      rm -f "/workspace/logs/gpu_claims/hip${hip}.claim" 2>/dev/null || true
      unset "WPID[$pid]" "WHIP[$pid]"
      ACTIVE=$((ACTIVE - 1))
    fi
  done
}

while true; do
  reap
  mapfile -t pend < <(pending_cfgs)
  if [ "${#pend[@]}" -eq 0 ] || [ -z "${pend[0]:-}" ]; then
    if [ "$ACTIVE" = 0 ]; then
      mark "done"
      date -Is >"$DONEFLAG"
      refresh_pareto
      log "fp8 fair queue ALL DONE"
      exit 0
    fi
  fi
  mark "running:pending=${pend[*]:-}:active=$ACTIVE"

  if [ "$ACTIVE" -lt "$MAX_GPUS" ] && [ "${#pend[@]}" -gt 0 ] && [ -n "${pend[0]:-}" ]; then
    # pick first pending not already running
    cfg=
    for cand in "${pend[@]}"; do
      running=0
      for pid in "${!WPID[@]}"; do
        if [ "${WPID[$pid]}" = "$cand" ]; then running=1; fi
      done
      if [ "$running" = 0 ]; then cfg="$cand"; break; fi
    done
    if [ -n "$cfg" ]; then
      if claim_gpu "$NEED_GIB"; then
        hip="$HIP_VISIBLE_DEVICES"
        if [ -n "${_CLAIM_FD:-}" ]; then
          flock -u "$_CLAIM_FD" 2>/dev/null || true
        fi
        unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
        echo "$(date -Is) pid=$$ hip=$hip cfg=$cfg role=fp8_fair" \
          >"/workspace/logs/gpu_claims/hip${hip}.claim"
        port=$((PORT_BASE + PORT_I))
        PORT_I=$((PORT_I + 1))
        log "LAUNCH cfg=$cfg hip=$hip port=$port"
        (
          trap '' HUP
          export HIP_VISIBLE_DEVICES="$hip" CUDA_VISIBLE_DEVICES="$hip"
          CFG="$cfg" PORT="$port" CONCS="$CONCS" UTIL="$UTIL" \
            bash "$HERE/fp8_fair/run_fp8_fair_agentx.sh"
          rc=$?
          rm -f "/workspace/logs/gpu_claims/hip${hip}.claim"
          echo "$(date -Is) DONE cfg=$cfg hip=$hip rc=$rc" \
            >>/workspace/logs/fp8-fair-worker-done.log
          exit "$rc"
        ) &
        wpid=$!
        WPID[$wpid]=$cfg
        WHIP[$wpid]=$hip
        ACTIVE=$((ACTIVE + 1))
        sleep 90
      else
        log "no free GPU; sleep $POLL_SEC"
        release_gpu 2>/dev/null || true
      fi
    fi
  fi
  sleep "$POLL_SEC"
done
