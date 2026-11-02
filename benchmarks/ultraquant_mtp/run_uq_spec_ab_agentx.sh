#!/bin/bash
# UQ DFlash/DSpark AgentX A/B cells (new outs only). Does not touch Stream 2 /
# P0-1 runners or baseline out dirs.
#
#   EXP=mnbt2048|ret50k|ret25k|dense|opt|mixed_pdv|dflash_swa|k6|cont_v2|hitdbg CFG=dspark_uq|dflash_uq \
#   HIP_VISIBLE_DEVICES=0 PORT=8450 CONCS="32" \
#     bash benchmarks/ultraquant_mtp/run_uq_spec_ab_agentx.sh
#
# EXP=opt = stacked best config fixes (no new kernel): dense draft backend +
#   --prefix-cache-retention-interval 50176 + --max-num-batched-tokens 2048
#   + --block-size 64 (already the UQ default here).
# EXP=mixed_pdv = ret50k + VLLM_ULTRAQUANT_MIXED_VERIFY=pdv (P0-2).
# EXP=dflash_swa = ret50k + ROCM_AITER_FA_DFLASH_SWA draft backend (P0-3; dflash only).
trap '' HUP
mkdir -p /workspace/logs
EXP="${EXP:?need EXP=mnbt2048|ret50k|ret25k|dense|opt|mixed_pdv|dflash_swa|k6|cont_v2|hitdbg|baseline}"
CFG="${CFG:?need CFG=dspark_uq|dflash_uq}"
exec >>"/workspace/logs/uq-spec-ab-${CFG}-${EXP}.log" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DFLASH_DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
DSPARK_DRAFT="${DSPARK_DRAFT_DIR:-/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP device}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SA_DIR/.hf-cache}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export VLLM_USE_V2_MODEL_RUNNER=1
# Same decode knobs as baseline UQ / MTP UQ
export VLLM_ULTRAQUANT_FUSED_MTP=1
export VLLM_ULTRAQUANT_SHARED_KV=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_WAVES=2
export VLLM_ULTRAQUANT_PAIR_VERIFY=0
export VLLM_ULTRAQUANT_PDV_VERIFY=1
unset VLLM_ULTRAQUANT_PDV_EXPERIMENT

case "$HIP_VISIBLE_DEVICES" in
  [0-7]) ;;
  *) echo "$(date -Is) FAIL HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"; exit 1 ;;
esac

PORT="${PORT:?need PORT}"
SEED="${SEED:-42}"
DURATION="${DURATION:-300}"
CONCS=(${CONCS:-32})
MAX_LEN=262144
# Default 0.90; set UTIL=0.70 on cards with stranded VRAM (~216/288 GiB free).
UTIL="${UTIL:-0.90}"
BLOCK=64
MNBT=8192
RETENTION=
SPEC_FILE=
DRAFT=
OUT_BASE=
ATTN=ULTRAQUANT_SPEC_COMPAT
KV=ultraquant_4bit

case "$CFG" in
  dspark_uq)
    DRAFT="$DSPARK_DRAFT"
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark.json
    OUT_STEM=dspark_qwen38_uq
    ;;
  dflash_uq)
    DRAFT="$DFLASH_DRAFT"
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38_uq/configs/uq_dflash2.json
    OUT_STEM=dflash_qwen38_uq
    ;;
  *) echo "$(date -Is) FAIL bad CFG=$CFG"; exit 1 ;;
esac

case "$EXP" in
  mnbt2048)
    MNBT=2048
    OUT_BASE=$ROOT/out/${OUT_STEM}_mnbt2048
    ;;
  ret50k)
    RETENTION=50176
    OUT_BASE=$ROOT/out/${OUT_STEM}_ret50k
    ;;
  ret25k)
    RETENTION=25088
    OUT_BASE=$ROOT/out/${OUT_STEM}_ret25k
    ;;
  dense)
    OUT_BASE=$ROOT/out/${OUT_STEM}_dense
    if [ "$CFG" = dspark_uq ]; then
      SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark_dense.json
    else
      SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38_uq/configs/uq_dflash2_dense.json
    fi
    ;;
  opt)
    # Fully stacked config: dense draft + ret50k + mnbt2048 (+ block 64).
    MNBT=2048
    RETENTION=50176
    OUT_BASE=$ROOT/out/${OUT_STEM}_opt
    if [ "$CFG" = dspark_uq ]; then
      SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark_dense.json
    else
      SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38_uq/configs/uq_dflash2_dense.json
    fi
    ;;
  opt_adaptive)
    # Same stack as opt, plus DSpark confidence-head adaptive verification.
    # DSpark only (candidate_selector is DFlash2; adaptive K is the DSpark twin).
    # OUT suffix _v3 after PR #53929 / 3462da7 capture-geometry + hash port
    # (v2 was conf-floor only; AgentX still ~0% accept).
    if [ "$CFG" != dspark_uq ]; then
      echo "$(date -Is) FAIL EXP=opt_adaptive requires CFG=dspark_uq"
      exit 1
    fi
    MNBT=2048
    RETENTION=50176
    OUT_BASE=$ROOT/out/${OUT_STEM}_opt_adaptive_v3
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark_dense_adaptive.json
    ;;
  mixed_pdv)
    # P0-2: keep PDV on mixed-batch verify (env-gated). Stack on ret50k.
    RETENTION=50176
    OUT_BASE=$ROOT/out/${OUT_STEM}_mixed_pdv
    export VLLM_ULTRAQUANT_MIXED_VERIFY=pdv
    ;;
  dflash_swa)
    # P0-3: DFlash SWA window page table. DFlash only; stack on ret50k.
    if [ "$CFG" != dflash_uq ]; then
      echo "$(date -Is) FAIL EXP=dflash_swa requires CFG=dflash_uq"
      exit 1
    fi
    RETENTION=50176
    OUT_BASE=$ROOT/out/${OUT_STEM}_dflash_swa
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38_uq/configs/uq_dflash2_swa.json
    ;;
  k6)
    # E7: K=6 speculative tokens (LCM 3072). Stack on ret50k.
    RETENTION=50176
    OUT_BASE=$ROOT/out/${OUT_STEM}_k6
    if [ "$CFG" = dspark_uq ]; then
      SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark_k6.json
    else
      SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38_uq/configs/uq_dflash2_k6.json
    fi
    ;;
  cont_v2)
    # P1-1: continuation-prefill v2 (persistent cu + context-aware threshold).
    RETENTION=50176
    OUT_BASE=$ROOT/out/${OUT_STEM}_cont_v2
    export VLLM_ULTRAQUANT_CONT_PREFILL=v2
    ;;
  hitdbg)
    # E3 / P1-4: per-group hit logger on a baseline-like UQ c32 cell.
    OUT_BASE=$ROOT/out/${OUT_STEM}_hitdbg
    export VLLM_KV_HIT_DEBUG=1
    ;;
  baseline)
    # Plain UQ baseline (same knobs as original matrix): block64, MNBT=8192, no retention.
    # Writes into the baseline out dir so a moved-aside cell can be replaced.
    OUT_BASE=$ROOT/out/${OUT_STEM}
    ;;
  *) echo "$(date -Is) FAIL bad EXP=$EXP"; exit 1 ;;
esac

cd "$ROOT"
log() { echo "$(date -Is) $*"; }

SERVER_PID=
cleanup_server() {
  [ -n "${SERVER_PID:-}" ] || return 0
  local pid="$SERVER_PID" c tree
  SERVER_PID=
  tree="$pid"
  for c in $(pgrep -P "$pid" 2>/dev/null || true); do
    tree="$tree $c $(pgrep -P "$c" 2>/dev/null || true)"
  done
  # shellcheck disable=SC2086
  kill $tree 2>/dev/null || true
  sleep 15
  # shellcheck disable=SC2086
  kill -9 $tree 2>/dev/null || true
}
trap cleanup_server EXIT

make_spec() {
  DRAFT="$DRAFT" MAX_LEN="$MAX_LEN" CFG="$SPEC_FILE" python - <<'PY'
import json, os
cfg = json.load(open(os.environ["CFG"]))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = int(os.environ["MAX_LEN"])
print(json.dumps(cfg))
PY
}

sample_idle() {
  local dest="$1" bench="$2"
  : >"$dest"
  while kill -0 "$bench" 2>/dev/null; do
    {
      echo "---- $(date -Is)"
      curl -s "http://127.0.0.1:$PORT/metrics" | awk '
        /^vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|num_preemptions_total|prefix_cache_hits_total|prefix_cache_queries_total|spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total|prompt_tokens_total|generation_tokens_total)\{/ { print }
      '
    } >>"$dest"
    sleep 10
  done
}

# Fit UTIL to currently free VRAM on cuda:0 (after HIP_VISIBLE_DEVICES).
# vLLM requires free >= util * total at startup; cards often lose headroom
# between the waiter confirm and serve.
fit_util_for_serve() {
  local want="$1"
  WANT_UTIL="$want" python - <<'PY'
import os, time, torch
want = float(os.environ["WANT_UTIL"])
time.sleep(3)  # reclaim after prior kill
for attempt in range(8):
    free, total = torch.cuda.mem_get_info(0)
    free_gib = free / 1024**3
    if free_gib < 120:
        import sys
        print(f"fit_util FAIL free={free_gib:.1f}GiB", file=sys.stderr, flush=True)
        raise SystemExit(1)
    # Leave 3% headroom under the free/total ratio.
    util = min(want, max(0.40, (free / total) - 0.03))
    if free >= util * total:
        import sys
        print(f"fit_util OK free={free_gib:.1f}GiB util={util:.3f} want={want}", file=sys.stderr, flush=True)
        print(f"{util:.4f}")
        raise SystemExit(0)
    time.sleep(10)
import sys
print("fit_util FAIL timeout", file=sys.stderr, flush=True)
raise SystemExit(1)
PY
}

run_cell() {
  local conc="$1"
  local out="$OUT_BASE/sa/seed${SEED}_c${conc}"
  if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "SKIP $CFG $EXP c$conc"
    return 0
  fi
  if [ ! -f "$DRAFT/config.json" ]; then
    log "FAIL missing draft $DRAFT"
    return 1
  fi
  cleanup_server
  # Drop anything still bound to our port from a prior crash.
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
  fi
  mkdir -p "$out"
  local cell_util fit_out
  fit_out=$(fit_util_for_serve "$UTIL" 2>"$out/fit_util.log") || {
    log "FAIL $CFG $EXP c$conc fit_util ($(tail -n 1 "$out/fit_util.log" 2>/dev/null))"
    return 1
  }
  cell_util=$(echo "$fit_out" | head -n 1)
  log "fit_util → $cell_util ($(tail -n 1 "$out/fit_util.log" 2>/dev/null))"
  local spec
  spec=$(make_spec)
  log "SA START $CFG $EXP c$conc hip=$HIP_VISIBLE_DEVICES port=$PORT mnbt=$MNBT ret=${RETENTION:-0} util=$cell_util spec=$SPEC_FILE"
  local serve_args=(
    serve "$MODEL" --port "$PORT" --tensor-parallel-size 1
    --attention-backend "$ATTN" --kv-cache-dtype "$KV"
    --block-size "$BLOCK"
    --max-model-len "$MAX_LEN" --gpu-memory-utilization "$cell_util"
    --enable-prefix-caching --no-enable-log-requests
    --max-num-seqs "$conc" --max-num-batched-tokens "$MNBT"
    --language-model-only --reasoning-parser qwen3 --trust-remote-code
    --speculative-config "$spec"
  )
  if [ -n "$RETENTION" ]; then
    serve_args+=(--prefix-cache-retention-interval "$RETENTION")
  fi
  vllm "${serve_args[@]}" >"$out/server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0 i
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL $CFG $EXP c$conc server exited"
      tail -n 80 "$out/server.log" || true
      cleanup_server
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL $CFG $EXP c$conc not ready"
    tail -n 80 "$out/server.log" || true
    cleanup_server
    return 1
  fi
  # Geometry breadcrumbs for dense
  grep -E "lcm block sizes|KV cache|GPU KV|block size|Padding mamba" "$out/server.log" \
    | head -n 40 >"$out/geometry_snip.txt" || true
  log "READY $CFG $EXP c$conc"

  local extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')
  (cd "$INFERENCEX_DIR" && "$AIPERF" profile --scenario inferencex-agentx-mvp \
    --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
    --model "$MODEL" --concurrency "$conc" --benchmark-duration "$DURATION" \
    --unsafe-override \
    --stats-interval 30 --random-seed "$SEED" --failed-request-threshold 0.10 \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-grace-period 300 \
    --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
    --max-context-length "$MAX_LEN" \
    --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir "$out/aiperf_artifacts" \
    --public-dataset semianalysis_cc_traces_weka_062126_256k \
    "${extra[@]}") >"$out/benchmark.log" 2>&1 &
  local bench=$!
  until grep -q "Phase profiling (profiling) started" "$out/benchmark.log" 2>/dev/null; do
    if ! kill -0 "$bench" 2>/dev/null; then
      log "FAIL $CFG $EXP c$conc aiperf exited early"
      tail -n 40 "$out/benchmark.log" || true
      cleanup_server
      return 1
    fi
    sleep 5
  done
  log "SA MEASURING $CFG $EXP c$conc"
  sample_idle "$out/idle_samples.txt" "$bench" &
  local sampler=$!
  wait "$bench" || log "WARN aiperf rc=$?"
  wait "$sampler" 2>/dev/null || true
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions|num_requests_running|num_requests_waiting|gpu_cache_usage|generation_tokens)' \
    >"$out/metrics_end.txt" || true
  {
    echo "cfg=$CFG exp=$EXP conc=$conc seed=$SEED mnbt=$MNBT retention=${RETENTION:-0}"
    echo "spec=$SPEC_FILE out=$out"
  } >"$out/cell_meta.txt"
  cleanup_server
  if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "END $CFG $EXP c$conc"
  else
    log "FAIL $CFG $EXP c$conc no profile"
    return 1
  fi
}

log "uq spec A/B start pid=$$ cfg=$CFG exp=$EXP hip=$HIP_VISIBLE_DEVICES port=$PORT concs=${CONCS[*]} out=$OUT_BASE"
for conc in "${CONCS[@]}"; do
  run_cell "$conc" || true
done
cleanup_server
log "uq spec A/B done cfg=$CFG exp=$EXP"
