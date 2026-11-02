#!/bin/bash
# After baseline AgentX matrix finishes: full P0-1 FP8 conc matrix + DFlash accept bisect.
# Writes to out/*_fp8_p01/ (does not overwrite baseline). Safe to launch early.
#
#   HIP_VISIBLE_DEVICES=1 PORT=8440 SKIP_GEO=1 \
#     bash benchmarks/ultraquant_mtp/run_p0_post_matrix_queue.sh
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/p0-post-matrix-queue.log
exec >>"$LOG" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp
VENV=/workspace/.venv-uq
export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP device}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
PORT="${PORT:-8440}"
PORT_B="${PORT_B:-8441}"
# Full AgentX conc matrix for P0-1 (same as baseline roles a/b)
CONCS_P01="${CONCS_P01:-8 16 20 24 30 32}"

log() { echo "$(date -Is) $*"; }

matrix_cells_done() {
  # Wait for baseline UQ matrix c30 cells (last role-c targets) before P0-1.
  local f
  for f in \
    "$ROOT/out/dflash_qwen38_uq/sa/seed42_c30/aiperf_artifacts/profile_export_aiperf.json" \
    "$ROOT/out/dspark_qwen38_uq/sa/seed42_c30/aiperf_artifacts/profile_export_aiperf.json"
  do
    [ -f "$f" ] || return 1
  done
  return 0
}

wait_matrix_idle() {
  log "waiting for UQ matrix cells + ports 8410/8420/8430 clear"
  local i busy
  for i in $(seq 1 720); do  # up to 6h
    busy=0
    for p in 8410 8420 8430; do
      if curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1; then busy=1; fi
    done
    if pgrep -af 'vllm serve' | grep -E -- '--port 8430|--port 8410|--port 8420' >/dev/null 2>&1; then
      busy=1
    fi
    if [ "$busy" = 0 ] && matrix_cells_done; then
      log "matrix idle + UQ c30 cells present"
      return 0
    fi
    if [ "$busy" = 0 ]; then
      # Ports clear but cells missing (failed again) — still proceed after short settle
      if [ "$i" -ge 4 ]; then
        log "ports clear; some UQ cells still missing — proceeding with P0-1 matrix"
        return 0
      fi
    fi
    sleep 30
  done
  log "WARN timed out waiting; proceeding anyway"
}

run_geometry() {
  local mode="$1" port="$2"
  log "GEO start $mode port=$port"
  HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" PORT="$port" MODE="$mode" \
    bash "$HERE/run_p0_kv_geometry_smoke.sh"
  local rc=$?
  if [ "$rc" = 0 ]; then log "GEO PASS $mode"; else log "GEO FAIL $mode rc=$rc"; fi
  return "$rc"
}

run_p01_matrix() {
  # Full conc matrix: dflash_fp8_p01 + dspark_fp8_p01 (block-size 64).
  log "P0-1 AgentX matrix start concs=$CONCS_P01 hip=$HIP_VISIBLE_DEVICES port=$PORT"
  # shellcheck disable=SC2086
  ROLE=p01 \
  HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
  PORT="$PORT" \
  CONCS="$CONCS_P01" \
    bash "$HERE/run_dflash_dspark_agentx_conc_sweep.sh"
  local rc=$?
  log "P0-1 AgentX matrix done rc=$rc"
  return "$rc"
}

run_accept_bisect() {
  log "ACCEPT bisect start (window vs nowin) on port=$PORT_B"
  HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" PORT="$PORT_B" \
    bash "$HERE/run_dflash_accept_bisect.sh"
}

log "p0 post-matrix queue start hip=$HIP_VISIBLE_DEVICES port=$PORT SKIP_GEO=${SKIP_GEO:-0} CONCS_P01=$CONCS_P01"
geo_already_ok() {
  local m="$1"
  local g="$ROOT/out/p0_kv_geometry/$m/geometry.json"
  [ -f "$g" ] && python -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('kv_lcm_block_sizes')==1664 else 1)" "$g" 2>/dev/null
}
if [ "${SKIP_GEO:-0}" != 1 ]; then
  for m in dspark_fp8_bs64_kv8 dflash_fp8_bs64 dflash_fp8_bs64_kv8; do
    if geo_already_ok "$m"; then
      log "SKIP GEO $m (lcm=1664 already)"
    else
      run_geometry "$m" "$PORT" || true
    fi
  done
else
  log "SKIP_GEO=1 — using existing out/p0_kv_geometry/*"
fi

wait_matrix_idle
run_p01_matrix || true
run_accept_bisect || true
log "p0 post-matrix queue done"
