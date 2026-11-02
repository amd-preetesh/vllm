#!/bin/bash
# Resumable weekend serving queue.
# Qwen3.8-27B, TP 1, MTP 5, HIP 0, SemiAnalysis AgentX.
# Screen is 180s and confirm is 300s at concurrency 8, 16, and 32.
#
#   run_qwen38_tp1_weekend_queue.sh plan
#   run_qwen38_tp1_weekend_queue.sh run
#
# `plan` only prints the manifest expansion. `run` serves, and it refuses to
# start until logs/weekend_serving_gates/kernel_correctness.ok exists.
# Variants stay disabled until their manifest env object is non-empty and
# enabled is true. One invocation can supply that implementation without
# editing the manifest:
#
#   WEEKEND_VARIANT_NAME=rowpack \
#   WEEKEND_VARIANT_ENV='VLLM_ULTRAQUANT_ROWPACK=1' \
#   run_qwen38_tp1_weekend_queue.sh plan
#
# A failed cell stops immediately and the queue continues with the next cell.
# Completed cells (DONE plus profile export plus metrics) are skipped.
# trap '' HUP is installed first and re-asserted after the EXIT trap so an
# SSH hangup does not kill the queue. The EXIT trap does not catch HUP.

trap '' HUP
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
WS=$(cd "$ROOT/.." && pwd)
PLAN_PY="$SCRIPT_DIR/weekend_serving_plan.py"
MANIFEST="$SCRIPT_DIR/weekend_serving_manifest.json"
SA_DIR="$WS/SemiAnalysis"
INFERENCEX_DIR="$SA_DIR/InferenceX"
AIPERF="$SA_DIR/aiperf-venv/bin/aiperf"
AITER_SRC="$WS/aiter"
VENV="$WS/.venv-uq"
QUEUE_LOG="$WS/logs/weekend-serving-queue.log"
REFRESH_PY="$WS/logs/refresh_weekend_canvas.py"
PORT="${PROFILE_PORT:-8070}"

MODE=run
if [[ $# -ge 1 ]]; then
  case "$1" in
    plan|run) MODE=$1; shift ;;
    *)
      echo "usage: $0 [plan|run]" >&2
      exit 2
      ;;
  esac
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [plan|run]" >&2
  exit 2
fi

if [[ "$MODE" == "plan" ]]; then
  WEEKEND_WORKSPACE="$WS" python3 "$PLAN_PY" --manifest "$MANIFEST" --workspace "$WS"
  exit 0
fi

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
unset ROCR_VISIBLE_DEVICES || true
export PYTHONNOUSERSITE=1
export VLLM_ROCM_USE_AITER=1
export HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SA_DIR/.hf-cache}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export VLLM_ULTRAQUANT_FUSED_MTP=1
export VLLM_ULTRAQUANT_SHARED_KV=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_PAIR_VERIFY=0
unset VLLM_ULTRAQUANT_PAIR_WAVES || true

PRE_GATE="${WEEKEND_GATE_FILE:-$WS/logs/weekend_serving_gates/kernel_correctness.ok}"
if [[ ! -f "$PRE_GATE" ]]; then
  echo "refusing to serve: prerequisite marker missing: $PRE_GATE" >&2
  echo "Correctness and kernel gates are separate. Create that file only after they pass." >&2
  exit 2
fi

PLAN_DIR="$ROOT/out/weekend_serving_tp1/plan"
mkdir -p "$PLAN_DIR" "$(dirname "$QUEUE_LOG")"
WEEKEND_WORKSPACE="$WS" python3 "$PLAN_PY" --manifest "$MANIFEST" --workspace "$WS" --write "$PLAN_DIR" >/tmp/weekend-serving-plan.txt
# shellcheck disable=SC1091
source "$PLAN_DIR/common.env"
if [[ -n "${WEEKEND_GATE_FILE:-}" ]]; then
  GATE_MARKER="$WEEKEND_GATE_FILE"
fi
if [[ ! -f "$GATE_MARKER" ]]; then
  echo "refusing to serve: prerequisite marker missing: $GATE_MARKER" >&2
  echo "Correctness and kernel gates are separate. Create that file only after they pass." >&2
  exit 2
fi
if [[ ! -x "$AIPERF" ]]; then
  echo "refusing to serve: aiperf not executable: $AIPERF" >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/vllm" ]]; then
  echo "refusing to serve: vllm not executable: $VENV/bin/vllm" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"

exec 9>"$OUT_ROOT/.queue.lock"
if ! flock -n 9; then
  echo "weekend queue already running (lock $OUT_ROOT/.queue.lock)" >&2
  exit 2
fi

exec >>"$QUEUE_LOG" 2>&1
trap '' HUP

PIDFILE="$OUT_ROOT/.server.pid"
QUEUE_PIDFILE="$OUT_ROOT/queue.pid"
echo $$ >"$QUEUE_PIDFILE"

log() {
  echo "$(date -Is) $*"
}

cleanup_server() {
  local pid="" cmd tree c
  if [[ -f "$PIDFILE" ]]; then
    pid=$(cat "$PIDFILE" 2>/dev/null || true)
    rm -f "$PIDFILE"
  fi
  [[ -n "${pid}" ]] || return 0
  if [[ ! -d "/proc/${pid}" ]]; then
    return 0
  fi
  cmd=$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *vllm*|*VLLM*) ;;
    *)
      log "refuse to kill pid=${pid} because it is not vllm"
      return 0
      ;;
  esac
  tree="$pid"
  for c in $(pgrep -P "$pid" 2>/dev/null || true); do
    tree="$tree $c $(pgrep -P "$c" 2>/dev/null || true)"
  done
  # shellcheck disable=SC2086
  kill $tree 2>/dev/null || true
  sleep 15
  # shellcheck disable=SC2086
  kill -9 $tree 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup_queue() {
  cleanup_server
  rm -f "$QUEUE_PIDFILE"
}

trap cleanup_queue EXIT
trap '' HUP

refresh_canvas() {
  if [[ -f "$REFRESH_PY" ]]; then
    WEEKEND_WORKSPACE="$WS" python3 "$REFRESH_PY" || log "WARN canvas refresh failed"
  fi
}

run_aiperf() {
  local out_dir="$1" seconds="$2"
  shift 2
  (cd "$INFERENCEX_DIR" && "$AIPERF" profile --scenario inferencex-agentx-mvp \
    --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
    --model "$MODEL" --concurrency "$CONC" --benchmark-duration "$seconds" \
    --unsafe-override \
    --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-grace-period 300 \
    --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
    --max-context-length 262144 \
    --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir "$out_dir" \
    --public-dataset semianalysis_cc_traces_weka_062126_256k \
    "$@")
}

run_cell() {
  local cell_id="$1" variant="$2" conc="$3" phase="$4" seconds="$5" kv="$6" needs_screen="$7" env_file="$8"
  local out="$OUT_ROOT/$variant/c${conc}/$phase"
  local screen_done="$OUT_ROOT/$variant/c${conc}/screen/DONE"
  local profile="$out/aiperf_artifacts/profile_export_aiperf.json"
  mkdir -p "$out"
  if [[ -f "$out/DONE" && -s "$out/metrics_end.txt" && -s "$profile" ]]; then
    log "SKIP ${cell_id} reason=completed"
    return 0
  fi
  if [[ "$needs_screen" == "1" && ! -f "$screen_done" ]]; then
    log "SKIP ${cell_id} reason=screen_not_ok"
    printf 'blocked: screen not ok\n' >"$out/status.txt"
    return 0
  fi

  (
    # errexit does not survive this subshell: the parent invokes run_cell from
    # `if`, which suppresses it for the whole call. Fail with an explicit exit.
    set -uo pipefail
    trap '' HUP
    REASON="starting cell"
    SERVER_PID=""
    die() {
      REASON="$1"
      exit 1
    }
    on_exit() {
      local rc=$?
      trap - EXIT
      cleanup_server || true
      if [[ "$rc" -ne 0 ]]; then
        printf '%s\n' "$REASON" >"$out/FAILED"
        printf 'failed\n' >"$out/status.txt"
        rm -f "$out/DONE"
      fi
      exit "$rc"
    }
    trap on_exit EXIT
    trap '' HUP

    export HIP_VISIBLE_DEVICES=0
    export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES || true
    export VLLM_ULTRAQUANT_FUSED_MTP=1
    export VLLM_ULTRAQUANT_SHARED_KV=0
    export VLLM_ULTRAQUANT_V3=0
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    unset VLLM_ULTRAQUANT_PAIR_WAVES || true
    unset VLLM_ULTRAQUANT_ROWPACK || true
    unset VLLM_ULTRAQUANT_ROWPACK_NT || true
    unset VLLM_ULTRAQUANT_ROWPACK_PARTITIONS || true
    unset VLLM_ULTRAQUANT_PAIR_PRODUCER || true
    set -a
    # shellcheck disable=SC1090
    source "$env_file" || die "variant env failed to load"
    set +a

    rm -f "$out/DONE" "$out/FAILED"
    printf 'running\n' >"$out/status.txt"
    cat >"$out/cell.json" <<EOF
{
  "cell": "${cell_id}",
  "variant": "${variant}",
  "concurrency": ${conc},
  "phase": "${phase}",
  "duration_s": ${seconds},
  "model": "${MODEL}",
  "tp": ${TP},
  "mtp": ${MTP},
  "hip": "${HIP_DEVICE}",
  "kv_cache_dtype": "${kv}"
}
EOF
    cp "$env_file" "$out/variant.env" || die "could not record variant env"
    extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')
    CONC="$conc"
    log "START ${cell_id} hip=${HIP_VISIBLE_DEVICES} tp=${TP} mtp=${MTP} conc=${CONC} phase=${phase} seconds=${seconds} kv=${kv}"

    vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "$TP" \
      --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$kv" \
      --max-model-len 262144 --gpu-memory-utilization 0.90 \
      --enable-prefix-caching --no-enable-log-requests \
      --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
      --language-model-only --reasoning-parser qwen3 --trust-remote-code \
      --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}" \
      >"$out/server.log" 2>&1 &
    SERVER_PID=$!
    printf '%s\n' "$SERVER_PID" >"$PIDFILE"

    REASON="server not ready"
    ready=0
    for i in $(seq 1 180); do
      if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then
        ready=1
        break
      fi
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        die "server exited during launch"
      fi
      sleep 5
    done
    if [[ "$ready" != 1 ]]; then
      die "server not ready"
    fi
    log "READY ${cell_id}"

    run_aiperf "$out/aiperf_artifacts" "$seconds" "${extra[@]}" >"$out/benchmark.log" 2>&1 &
    bench=$!
    while true; do
      if grep -q "Phase profiling (profiling) started" "$out/benchmark.log" 2>/dev/null; then
        break
      fi
      if ! kill -0 "$bench" 2>/dev/null; then
        die "aiperf exited before measuring"
      fi
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        die "server exited before measuring"
      fi
      sleep 5
    done
    log "MEASURING ${cell_id}"
    bench_rc=0
    wait "$bench" || bench_rc=$?
    if [[ "$bench_rc" -ne 0 ]]; then
      die "aiperf rc=${bench_rc}"
    fi

    curl -sf --max-time 30 "http://127.0.0.1:$PORT/metrics" \
      | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' >"$out/metrics_end.txt" \
      || die "prometheus metrics missing"
    [[ -s "$out/aiperf_artifacts/profile_export_aiperf.json" ]] \
      || die "profile_export_aiperf.json missing"
    cleanup_server || die "server cleanup failed"
    printf '%s\n' "$(date -Is)" >"$out/DONE" || die "could not write DONE"
    printf 'ok\n' >"$out/status.txt" || die "could not write status"
    log "END ${cell_id} status=ok"
  )
}

log "weekend queue start pid=$$ ppid=$PPID hip=0 tp=${TP} mtp=${MTP} port=${PORT} gate=${GATE_MARKER}"
cat /tmp/weekend-serving-plan.txt
refresh_canvas
if [[ ! -s "$PLAN_DIR/cells.tsv" ]]; then
  log "no runnable cells; variants stay disabled until env is set and enabled"
  log "weekend queue complete"
  refresh_canvas
  exit 0
fi

while IFS=$'\t' read -r cell_id variant conc phase seconds kv needs_screen env_file; do
  [[ -n "${cell_id}" ]] || continue
  if ! run_cell "$cell_id" "$variant" "$conc" "$phase" "$seconds" "$kv" "$needs_screen" "$env_file"; then
    reason="cell failed"
    failed_file="$OUT_ROOT/$variant/c${conc}/$phase/FAILED"
    if [[ -f "$failed_file" ]]; then
      reason=$(head -n 1 "$failed_file")
    fi
    log "FAIL ${cell_id} reason=${reason}"
  fi
  refresh_canvas
done <"$PLAN_DIR/cells.tsv"

log "weekend queue complete"
refresh_canvas
