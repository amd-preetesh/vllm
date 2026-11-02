#!/bin/bash
# Research diag queue: E3 hitdbg, E5 accept probes (+ DFlash flip), H1
# null-block / no-window bisect. Waits for P1 (or opt+p0+p1 idle).
#
#   docker exec -d preshah_vllm_ultraquant bash -lc \
#     'nohup bash /workspace/vllm/benchmarks/ultraquant_mtp/run_uq_research_diag_queue_when_free.sh \
#        >>/workspace/logs/uq-research-diag-queue.log 2>&1 &'
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/uq-research-diag-queue.log
exec >>"$LOG" 2>&1
set -uo pipefail

HERE=/workspace/vllm/benchmarks/ultraquant_mtp
ROOT=/workspace/vllm
# shellcheck source=/dev/null
source "$HERE/gpu_claim.sh"

PORT="${PORT:-8466}"
PORT_PROBE="${PORT_PROBE:-8468}"
PORT_BISECT="${PORT_BISECT:-8470}"
NEED_GIB="${NEED_GIB:-260}"
NEED_GIB_PROBE="${NEED_GIB_PROBE:-200}"
UTIL="${UTIL:-0.70}"
CONCS_C32="${CONCS_C32:-32}"
WAIT_P1="${WAIT_P1:-1}"

log() { echo "$(date -Is) $*"; }

wait_for_p1_if_needed() {
  [ "$WAIT_P1" = 1 ] || return 0
  local done=/workspace/logs/uq-p1-queue.DONE
  log "waiting for P1 queue (WAIT_P1=1) or opt/p0/p1 idle"
  for i in $(seq 1 2880); do  # up to 24h @ 30s
    if [ -f "$done" ]; then
      log "P1 DONE file present — starting research diag queue"
      return 0
    fi
    if ! pgrep -f run_uq_p1_queue_when_free.sh >/dev/null 2>&1 \
      && ! pgrep -f run_uq_p0_code_queue_when_free.sh >/dev/null 2>&1 \
      && ! pgrep -f run_uq_opt_agentx_when_free.sh >/dev/null 2>&1 \
      && ! pgrep -f run_dflash_swa_when_free.sh >/dev/null 2>&1; then
      log "opt/p0/p1/swa idle — starting research diag queue"
      return 0
    fi
    if [ $((i % 10)) -eq 1 ]; then
      log "still waiting on P1/opt/p0/swa (try $i)"
    fi
    sleep 30
  done
  log "WARN P1 wait timed out; continuing diag queue anyway"
}

run_with_gpu() {
  local label="$1"
  local need="$2"
  shift 2
  for i in $(seq 1 1440); do
    if claim_gpu "$need"; then
      log "$label: GPU $HIP_VISIBLE_DEVICES claimed need=${need}GiB (flock held)"
      "$@"
      local rc=$?
      release_gpu
      return $rc
    fi
    if [ $((i % 10)) -eq 1 ]; then
      log "$label: waiting for >=${need}GiB (try $i)"
    fi
    sleep 30
  done
  log "FAIL $label timed out waiting for GPU"
  return 1
}

fix_nowin_symlink() {
  local nowin=/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2-nowin
  local win=/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2
  if [ ! -d "$nowin" ]; then
    log "WARN nowin draft dir missing: $nowin"
    return 1
  fi
  if [ ! -f "$win/model.safetensors" ]; then
    log "WARN window draft weights missing: $win/model.safetensors"
    return 1
  fi
  # Symlink must resolve inside the container (/workspace), not host /home/...
  if [ ! -f "$nowin/model.safetensors" ]; then
    log "fix nowin model.safetensors symlink → ../Qwen3.8-27B-DFlash2/model.safetensors"
    ln -sfn ../Qwen3.8-27B-DFlash2/model.safetensors "$nowin/model.safetensors"
  fi
  if [ -f "$nowin/model.safetensors" ]; then
    log "nowin weights OK: $(readlink -f "$nowin/model.safetensors" 2>/dev/null || true)"
    return 0
  fi
  log "FAIL nowin weights still missing after symlink fix"
  return 1
}

log "research diag queue start port=$PORT probe_port=$PORT_PROBE bisect_port=$PORT_BISECT need=$NEED_GIB"
wait_for_p1_if_needed
fix_nowin_symlink || true

# --- E3: per-group hitdbg AgentX c32 ---
run_with_gpu "E3 hitdbg dspark" "$NEED_GIB" bash -c '
  EXP=hitdbg CFG=dspark_uq PORT='"$PORT"' CONCS="'"$CONCS_C32"'" UTIL='"$UTIL"' \
    bash "'"$HERE"'/run_uq_spec_ab_agentx.sh"
' || log "WARN E3 dspark hitdbg rc=$?"

run_with_gpu "E3 hitdbg dflash" "$NEED_GIB" bash -c '
  EXP=hitdbg CFG=dflash_uq PORT='"$PORT"' CONCS="'"$CONCS_C32"'" UTIL='"$UTIL"' \
    bash "'"$HERE"'/run_uq_spec_ab_agentx.sh"
' || log "WARN E3 dflash hitdbg rc=$?"

# --- E5: accept probes (UQ + FP8; DFlash flipped twice) ---
for tag in dspark_uq dspark_fp8 dflash_fp8; do
  run_with_gpu "E5 probe $tag" "$NEED_GIB_PROBE" bash -c '
    TAG='"$tag"' PORT='"$PORT_PROBE"' \
      bash "'"$HERE"'/run_uq_accept_probe.sh"
  ' || log "WARN E5 $tag rc=$?"
done

# DFlash UQ: two fresh server starts (run-to-run flip)
for run in 1 2; do
  run_with_gpu "E5 probe dflash_uq r$run" "$NEED_GIB_PROBE" bash -c '
    TAG=dflash_uq RUN='"$run"' PORT='"$PORT_PROBE"' FORCE=1 \
      VLLM_DFLASH_DEBUG_NULL_BLOCK=1 \
      bash "'"$HERE"'/run_uq_accept_probe.sh"
  ' || log "WARN E5 dflash_uq r$run rc=$?"
done

# H1 A/B: same DFlash UQ probe with null-block zeroing
run_with_gpu "H1 null_block_zero probe" "$NEED_GIB_PROBE" bash -c '
  TAG=dflash_uq RUN=zero PORT='"$PORT_PROBE"' FORCE=1 \
    VLLM_DFLASH_DEBUG_NULL_BLOCK=1 VLLM_DFLASH_NULL_BLOCK_ZERO=1 \
    bash "'"$HERE"'/run_uq_accept_probe.sh"
' || log "WARN H1 null_zero probe rc=$?"

# --- Missing no-window bisect ---
if [ -f "$ROOT/out/dflash_accept_bisect/nowin/accept_vs_length.jsonl" ] \
  && [ -f "$ROOT/out/dflash_accept_bisect/window/accept_vs_length.jsonl" ]; then
  log "SKIP dflash accept bisect — outputs present"
else
  # Clear failed nowin server.log-only cell so bisect re-runs nowin
  if [ ! -f "$ROOT/out/dflash_accept_bisect/nowin/accept_vs_length.jsonl" ]; then
    rm -f "$ROOT/out/dflash_accept_bisect/nowin/server.log" 2>/dev/null || true
  fi
  run_with_gpu "H1 nowin bisect" "$NEED_GIB_PROBE" bash -c '
    PORT='"$PORT_BISECT"' \
      bash "'"$HERE"'/run_dflash_accept_bisect.sh"
  ' || log "WARN nowin bisect rc=$?"
fi

# Optional standalone aiter NaN repro (cheap; needs a free GPU briefly)
AITER_OUT=$ROOT/out/uq_accept_probe/aiter_null_block_nan_repro.txt
if [ -f "$AITER_OUT" ]; then
  log "SKIP aiter null-block repro — $AITER_OUT exists"
else
  run_with_gpu "H1 aiter nan repro" 8 bash -c '
    mkdir -p "'"$ROOT"'/out/uq_accept_probe"
    HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
      /workspace/.venv-uq/bin/python "'"$HERE"'/aiter_null_block_nan_repro.py" \
      >"'"$AITER_OUT"'" 2>&1 || true
  ' || log "WARN aiter repro rc=$?"
fi

log "research diag queue done"
touch /workspace/logs/uq-research-diag-queue.DONE
