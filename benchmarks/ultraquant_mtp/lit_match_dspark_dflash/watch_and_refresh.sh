#!/bin/bash
# Poll lit_match outs; refresh canvas + summarize when cells land.
trap '' HUP
set -uo pipefail
HERE=/workspace/vllm/benchmarks/ultraquant_mtp/lit_match_dspark_dflash
LOG=/workspace/logs/lit_match_qwen38/watch.log
mkdir -p /workspace/logs/lit_match_qwen38
exec >>"$LOG" 2>&1
log() { echo "$(date -Is) watch $*"; }

while true; do
  python3 "$HERE/refresh_canvas.py" || true
  python3 "$HERE/summarize_gates.py" >/workspace/vllm/out/lit_match_qwen38/summary_gates.latest.txt 2>&1 || true
  if [ -f /workspace/logs/lit_match_qwen38/suite.DONE ]; then
    log "suite.DONE — final refresh"
    python3 "$HERE/refresh_canvas.py" || true
    python3 "$HERE/summarize_gates.py" || true
    break
  fi
  sleep 120
done
log "exit"
