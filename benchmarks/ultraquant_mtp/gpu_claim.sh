#!/bin/bash
# Shared GPU claim helpers for UQ waiters.
# Usage (MUST call directly — never HIP=$(claim_gpu), that runs in a subshell
# and drops the HIP_VISIBLE_DEVICES export):
#   source benchmarks/ultraquant_mtp/gpu_claim.sh
#   if claim_gpu 260; then
#     # HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES exported
#     ...; release_gpu
#   fi
#
# Serializes our waiters via flock so two jobs cannot confirm the same card.

CLAIM_LOCK_DIR="${CLAIM_LOCK_DIR:-/workspace/logs/gpu_claims}"
CLAIM_LOCK_FILE="${CLAIM_LOCK_FILE:-/workspace/logs/gpu_claim.lock}"
mkdir -p "$CLAIM_LOCK_DIR"

# Global flock FD — held for the whole serve+bench lifetime of one cell/matrix.
_CLAIM_FD=

claim_gpu() {
  local need_gib="${1:-260}"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  exec {_CLAIM_FD}>"$CLAIM_LOCK_FILE" || return 1
  if ! flock -n "$_CLAIM_FD"; then
    # Another of our waiters is claiming/running — wait briefly then fail pick.
    flock -w 5 "$_CLAIM_FD" || return 1
  fi

  local hip
  hip=$(NEED_GIB="$need_gib" python - <<'PY'
import os, torch
need = float(os.environ["NEED_GIB"]) * (1024**3)
best = None
for i in range(torch.cuda.device_count()):
    try:
        free, total = torch.cuda.mem_get_info(i)
    except Exception:
        continue
    if free < need:
        continue
    score = free
    if free / total >= 0.85:
        score += 1e12
    if best is None or score > best[1]:
        best = (i, score, free)
if best is None:
    raise SystemExit(1)
print(best[0])
PY
) || { flock -u "$_CLAIM_FD"; return 1; }

  export HIP_VISIBLE_DEVICES="$hip"
  export CUDA_VISIBLE_DEVICES="$hip"

  # Two samples 30s apart on the visible device.
  if ! NEED_GIB="$need_gib" python - <<'PY'
import os, sys, time, torch
need = float(os.environ["NEED_GIB"]) * (1024**3)
for sample in (1, 2):
    free, total = torch.cuda.mem_get_info(0)
    print(
        f"confirm sample{sample} hip={os.environ.get('HIP_VISIBLE_DEVICES')} "
        f"free_gib={free/1024**3:.1f}",
        file=sys.stderr,
        flush=True,
    )
    if free < need:
        raise SystemExit(1)
    if sample == 1:
        time.sleep(30)
PY
  then
    unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    flock -u "$_CLAIM_FD"
    return 1
  fi

  echo "$(date -Is) pid=$$ hip=$hip need=$need_gib" >"$CLAIM_LOCK_DIR/hip${hip}.claim"
  return 0
}

release_gpu() {
  local hip="${HIP_VISIBLE_DEVICES:-}"
  if [ -n "$hip" ]; then
    rm -f "$CLAIM_LOCK_DIR/hip${hip}.claim" 2>/dev/null || true
  fi
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  if [ -n "${_CLAIM_FD:-}" ]; then
    flock -u "$_CLAIM_FD" 2>/dev/null || true
  fi
}
