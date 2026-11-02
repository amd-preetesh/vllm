#!/bin/bash
# Targeted re-run: DFlash FP8 SPEED-747 c1 x 3 reps on empty GPUs.
# Prior c1_r{1,2,3} failed fit_util / free-mem < util; quarantine then remeasure.
trap '' HUP
set -uo pipefail
export ARMS="dflash_fp8"
export WORKLOADS="speed747"
export CONCS="1"
export REPS=3
export MAX_GPUS=2
export NEED_GIB=250
export PORT_BASE=8830
export POLL_SEC=30
ROOT=/workspace/vllm
MM=$ROOT/benchmarks/ultraquant_mtp/method_match
LOG=/workspace/logs/method-match-dflash-fp8-speed-c1-rerun.log
STATE=/workspace/logs/method-match-dflash-fp8-speed-c1-rerun.state
JOBS=/workspace/logs/method-match-dflash-fp8-speed-c1-rerun.jobs
LOCK=/workspace/logs/method-match-dflash-fp8-speed-c1-rerun.lock
DONEFLAG=/workspace/logs/method-match-dflash-fp8-speed-c1-rerun.DONE
rm -f "$DONEFLAG" "$STATE"

# Quarantine failed empty/partial c1 cells so fresh outs land cleanly.
SRC=$ROOT/out/method_match_dflash_fp8/speed747
STAMP=$(date -u +%Y%m%dT%H%M%S)
Q=$SRC/c1_quarantine_bad_${STAMP}
mkdir -p "$Q"
for r in 1 2 3; do
  cell=$SRC/c1_r$r
  if [ -d "$cell" ] && [ ! -f "$cell/aiperf_artifacts/profile_export_aiperf.json" ]; then
    mv "$cell" "$Q/"
    echo "quarantined $cell -> $Q/"
  fi
done

sed \
  -e "s|^LOG=.*|LOG=$LOG|" \
  -e "s|^STATE=.*|STATE=$STATE|" \
  -e "s|^JOBS=.*|JOBS=$JOBS|" \
  -e "s|^LOCK=.*|LOCK=$LOCK|" \
  -e "s|^DONEFLAG=.*|DONEFLAG=$DONEFLAG|" \
  "$MM/run_queue.sh" > /tmp/run_queue_dflash_speed_c1.sh
chmod +x /tmp/run_queue_dflash_speed_c1.sh
exec bash /tmp/run_queue_dflash_speed_c1.sh
