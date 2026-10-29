#!/bin/bash
# Run one lane (fp8|uq) of the AgentX baseline matrix on a single GPU.
#   LANE=fp8 HIP_VISIBLE_DEVICES=1 PORT=8610 MODE=smoke|full \
#     bash benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery/run_lane.sh
trap '' HUP
set -uo pipefail

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
ARMS_JSON=$HERE/arms.json
OUT_ROOT=$ROOT/out/agentx_dspark_dflash_recovery
LOG_DIR=/workspace/logs/agentx_dspark_dflash_recovery

LANE="${LANE:?need LANE=fp8|uq}"
MODE="${MODE:-full}"  # smoke|full
PORT="${PORT:?need PORT}"
SEED="${SEED:-42}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need HIP}"
export HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES

mkdir -p "$LOG_DIR" "$OUT_ROOT"
exec >>"$LOG_DIR/lane_${LANE}.log" 2>&1
log() { echo "$(date -Is) lane=$LANE $*"; }

export PATH="$VENV/bin:$PATH"
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SA_DIR/.hf-cache}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export VLLM_USE_V2_MODEL_RUNNER=1

MODEL=$(python3 -c "import json; print(json.load(open('$ARMS_JSON'))['model'])")
CONCS=($(python3 -c "import json,os; print(' '.join(map(str,json.load(open('$ARMS_JSON'))['concs'])))"))
DURATION_FULL=$(python3 -c "import json; print(json.load(open('$ARMS_JSON'))['duration_s'])")
MAX_LEN_FULL=$(python3 -c "import json; print(json.load(open('$ARMS_JSON'))['max_model_len'])")
MNBT=$(python3 -c "import json; print(json.load(open('$ARMS_JSON'))['mnbt'])")
BLOCK=$(python3 -c "import json; print(json.load(open('$ARMS_JSON'))['block_size'])")
UTIL="${UTIL:-0.70}"

if [ "$MODE" = smoke ]; then
  # Weka AgentX traces need ≥~44k context; keep smoke short but dataset-eligible.
  DURATION=60
  MAX_LEN=65536
  CONCS=(8)
  SMOKE_TAG=smoke
else
  DURATION="$DURATION_FULL"
  MAX_LEN="$MAX_LEN_FULL"
  SMOKE_TAG=sa
fi

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
  sleep 10
  # shellcheck disable=SC2086
  kill -9 $tree 2>/dev/null || true
}
trap cleanup_server EXIT

apply_env() {
  # Clear UQ knobs then apply arm env.
  unset VLLM_ULTRAQUANT_FUSED_MTP VLLM_ULTRAQUANT_PDV_VERIFY VLLM_ULTRAQUANT_PDV_EXPERIMENT \
    VLLM_ULTRAQUANT_MIXED_VERIFY VLLM_ULTRAQUANT_CONT_PREFILL VLLM_ULTRAQUANT_PAIR_VERIFY \
    VLLM_ULTRAQUANT_SHARED_KV VLLM_ULTRAQUANT_V3 VLLM_ULTRAQUANT_ROWPACK_VERIFY \
    VLLM_ULTRAQUANT_PAIR_WAVES 2>/dev/null || true
  local kv="$1"
  shift
  # remaining are KEY=VAL pairs
  local pair
  for pair in "$@"; do
    [ -n "$pair" ] || continue
    export "$pair"
  done
  if [ "$kv" = ultraquant_4bit ]; then
    : "${VLLM_ULTRAQUANT_FUSED_MTP:=0}"
  fi
}

fit_util() {
  local want="$1"
  WANT_UTIL="$want" python3 - <<'PY'
import os, time, torch
want = float(os.environ["WANT_UTIL"])
time.sleep(2)
for _ in range(8):
    free, total = torch.cuda.mem_get_info(0)
    util = min(want, max(0.40, (free / total) - 0.03))
    if free >= util * total and free / (1024**3) >= 80:
        print(f"{util:.4f}")
        raise SystemExit(0)
    time.sleep(5)
raise SystemExit(1)
PY
}

EXTRA_JSON='{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}'

run_cell() {
  local arm_id="$1" label="$2" attn="$3" kv="$4" retention="$5" spec_json="$6"
  shift 6
  local env_pairs=("$@")
  local conc
  for conc in "${CONCS[@]}"; do
    local out="$OUT_ROOT/$arm_id/$SMOKE_TAG/seed${SEED}_c${conc}"
    if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
      log "SKIP $arm_id c$conc"
      continue
    fi
    cleanup_server
    if command -v fuser >/dev/null 2>&1; then
      fuser -k "${PORT}/tcp" 2>/dev/null || true
    fi
    mkdir -p "$out"
    apply_env "$kv" "${env_pairs[@]}"
    local cell_util
    cell_util=$(fit_util "$UTIL" 2>"$out/fit_util.log") || {
      log "FAIL $arm_id c$conc fit_util"
      echo FAIL >"$out/STATUS"
      continue
    }
    local serve_args=(
      serve "$MODEL" --port "$PORT" --tensor-parallel-size 1
      --attention-backend "$attn" --kv-cache-dtype "$kv"
      --block-size "$BLOCK"
      --max-model-len "$MAX_LEN" --gpu-memory-utilization "$cell_util"
      --enable-prefix-caching --no-enable-log-requests
      --max-num-seqs "$conc" --max-num-batched-tokens "$MNBT"
      --language-model-only --reasoning-parser qwen3 --trust-remote-code
    )
    if [ -n "$retention" ] && [ "$retention" != "0" ] && [ "$MODE" = full ]; then
      serve_args+=(--prefix-cache-retention-interval "$retention")
    fi
    if [ "$spec_json" != "null" ] && [ -n "$spec_json" ]; then
      serve_args+=(--speculative-config "$spec_json")
    fi
    log "START $arm_id ($label) c$conc hip=$HIP_VISIBLE_DEVICES attn=$attn kv=$kv util=$cell_util mode=$MODE"
    vllm "${serve_args[@]}" >"$out/server.log" 2>&1 &
    SERVER_PID=$!
    local ready=0 i
    for i in $(seq 1 240); do
      if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
        ready=1
        break
      fi
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "FAIL $arm_id c$conc server exited"
        tail -n 60 "$out/server.log" || true
        echo FAIL >"$out/STATUS"
        cleanup_server
        ready=0
        break
      fi
      sleep 5
    done
    if [ "$ready" != 1 ]; then
      [ -f "$out/STATUS" ] || { log "FAIL $arm_id c$conc not ready"; echo FAIL >"$out/STATUS"; }
      cleanup_server
      continue
    fi
    if [ "$arm_id" = uq_mtp_pdv ] && ! grep -q "PDV FlyDSL verify is enabled" "$out/server.log"; then
      log "FAIL $arm_id PDV line missing"
      echo FAIL >"$out/STATUS"
      cleanup_server
      continue
    fi
    log "READY $arm_id c$conc"
    echo RUNNING >"$out/STATUS"
    local extra=(--extra-inputs "$EXTRA_JSON")
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
      "${extra[@]}") >"$out/benchmark.log" 2>&1 || log "WARN aiperf rc=$?"
    curl -s "http://127.0.0.1:$PORT/metrics" \
      | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions|generation_tokens|gpu_cache)' \
      >"$out/metrics_end.txt" || true
    {
      echo "arm=$arm_id label=$label"
      echo "attn=$attn kv=$kv retention=$retention"
      echo "spec=$spec_json"
      echo "env=${env_pairs[*]}"
      echo "mode=$MODE conc=$conc util=$cell_util"
    } >"$out/cell_meta.txt"
    cleanup_server
    if [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
      echo DONE >"$out/STATUS"
      log "DONE $arm_id c$conc"
    else
      echo FAIL >"$out/STATUS"
      log "FAIL $arm_id c$conc no profile"
    fi
  done
}

log "lane start mode=$MODE hip=$HIP_VISIBLE_DEVICES port=$PORT concs=${CONCS[*]} mnbt=$MNBT block=$BLOCK"

export MAX_LEN_OVERRIDE="$MAX_LEN"
python3 - "$ARMS_JSON" "$LANE" <<'PY' >"$LOG_DIR/lane_${LANE}_plan.tsv"
import json, os, sys
arms = json.load(open(sys.argv[1]))
lane = sys.argv[2]
drafts = arms["drafts"]
max_len = int(os.environ["MAX_LEN_OVERRIDE"])
for a in arms["arms"]:
    if a["lane"] != lane:
        continue
    spec = a.get("spec")
    if spec is None:
        spec_s = "null"
    else:
        s = dict(spec)
        if "model_key" in s:
            s["model"] = drafts[s.pop("model_key")]
        s["max_model_len"] = max_len
        spec_s = json.dumps(s, separators=(",", ":"))
    env = a.get("env") or {}
    env_s = " ".join(f"{k}={v}" for k, v in env.items())
    print("\t".join([
        a["id"], a["label"], a["attn"], a["kv"], str(a.get("retention") or 0),
        spec_s, env_s,
    ]))
PY

# Smoke: first arm of this lane only (serve proof). Full: all arms.
SMOKE_DONE=0
while IFS=$'\t' read -r arm_id label attn kv retention spec_json env_s; do
  if [ "$MODE" = smoke ]; then
    if [ "$SMOKE_DONE" = 1 ]; then
      continue
    fi
    SMOKE_DONE=1
  fi
  # shellcheck disable=SC2206
  env_pairs=($env_s)
  run_cell "$arm_id" "$label" "$attn" "$kv" "$retention" "$spec_json" "${env_pairs[@]}"
done <"$LOG_DIR/lane_${LANE}_plan.tsv"

log "lane complete mode=$MODE"
date -Is >"$LOG_DIR/lane_${LANE}_${MODE}.DONE"
python3 "$HERE/summarize_gates.py" 2>/dev/null || true
