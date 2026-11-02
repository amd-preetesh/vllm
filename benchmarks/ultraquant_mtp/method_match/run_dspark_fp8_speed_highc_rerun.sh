#!/bin/bash
# Targeted re-run: DSpark FP8 SPEED-747 c64/128/256 x 3 reps on empty GPUs.
trap '' HUP
set -uo pipefail
export ARMS="dspark_fp8"
export WORKLOADS="speed747"
export CONCS="64 128 256"
export REPS=3
export MAX_GPUS=2
export NEED_GIB=250
export PORT_BASE=8820
export POLL_SEC=30
# divert queue log/state for this rerun
ROOT=/workspace/vllm
MM=$ROOT/benchmarks/ultraquant_mtp/method_match
LOG=/workspace/logs/method-match-dspark-fp8-speed-highc-rerun.log
STATE=/workspace/logs/method-match-dspark-fp8-speed-highc-rerun.state
JOBS=/workspace/logs/method-match-dspark-fp8-speed-highc-rerun.jobs
LOCK=/workspace/logs/method-match-dspark-fp8-speed-highc-rerun.lock
DONEFLAG=/workspace/logs/method-match-dspark-fp8-speed-highc-rerun.DONE
rm -f "$DONEFLAG" "$STATE"
# Patch by exporting paths the queue script does not read — instead run inlined copy of build+loop
# Simplest: sed-copy run_queue with alternate paths
sed \
  -e "s|^LOG=.*|LOG=$LOG|" \
  -e "s|^STATE=.*|STATE=$STATE|" \
  -e "s|^JOBS=.*|JOBS=$JOBS|" \
  -e "s|^LOCK=.*|LOCK=$LOCK|" \
  -e "s|^DONEFLAG=.*|DONEFLAG=$DONEFLAG|" \
  "$MM/run_queue.sh" > /tmp/run_queue_dspark_speed_highc.sh
chmod +x /tmp/run_queue_dspark_speed_highc.sh
exec bash /tmp/run_queue_dspark_speed_highc.sh
