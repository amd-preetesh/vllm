#!/bin/bash
# After Stream 1 accuracy looks good, run the SemiAnalysis serving compare:
# FP8 MTP 5 c=32 vs FFDSL pair (UltraQuant, PAIR_VERIFY=1, ROWPACK=0).
trap '' HUP
ACCURACY_OK=/workspace/logs/ffdsl-pair-accuracy.ok
ACCURACY_LOG=/workspace/logs/ffdsl-pair-accuracy.log
SA_LOG=/workspace/logs/ffdsl-pair-sa.log
echo "$(date -Is) sa follower waiting for accuracy complete" >> "$SA_LOG"
while [ ! -f "$ACCURACY_OK" ]; do
  sleep 60
done

# Gate: require both primary summaries, and keep going unless FFDSL is clearly worse.
python3 - <<'PY' >> "$SA_LOG" 2>&1
import json, sys
from pathlib import Path
root = Path("/workspace/vllm/out/ffdsl_pair_accuracy")
need = [
    root / "gsm8k_fp8_seed7/summary.json",
    root / "gsm8k_ffdsl_seed7/summary.json",
]
missing = [str(p) for p in need if not p.exists()]
gpqa_seeds = []
for seed in (7, 11, 13):
    fp8 = root / f"gpqa_fp8_seed{seed}/summary.json"
    ffdsl = root / f"gpqa_ffdsl_seed{seed}/summary.json"
    if fp8.exists() and ffdsl.exists():
        gpqa_seeds.append((seed, json.loads(fp8.read_text())["accuracy"], json.loads(ffdsl.read_text())["accuracy"]))
    elif fp8.exists() or ffdsl.exists():
        missing.append(f"gpqa seed {seed} incomplete")
if not gpqa_seeds:
    missing.append("no complete gpqa seed pair")
if missing:
    print("sa gate FAIL missing", missing)
    sys.exit(2)
gsm8k_fp8 = json.loads((root / "gsm8k_fp8_seed7/summary.json").read_text())["accuracy"]
gsm8k_ffdsl = json.loads((root / "gsm8k_ffdsl_seed7/summary.json").read_text())["accuracy"]
gsm8k_gap = gsm8k_ffdsl - gsm8k_fp8
gpqa_gap = sum(ff - fp for _, fp, ff in gpqa_seeds) / len(gpqa_seeds)
print(f"sa gate gsm8k: fp8={gsm8k_fp8:.4f} ffdsl={gsm8k_ffdsl:.4f} gap={gsm8k_gap:+.4f}pp")
for seed, fp, ff in gpqa_seeds:
    print(f"sa gate gpqa seed {seed}: fp8={fp:.4f} ffdsl={ff:.4f} gap={ff-fp:+.4f}pp")
print(f"sa gate gpqa mean gap={gpqa_gap:+.4f}pp over {len(gpqa_seeds)} seeds")
# Seed 7 alone can swing GPQA at temperature 1. Judge the mean.
if gsm8k_gap < -2.0 or gpqa_gap < -2.0:
    print("sa gate FAIL: mean FFDSL more than 2pp below FP8; skipping SemiAnalysis")
    sys.exit(3)
print("sa gate PASS")
PY
gate_rc=$?
if [ "$gate_rc" != 0 ]; then
  echo "$(date -Is) sa follower stopped gate_rc=${gate_rc}" >> "$SA_LOG"
  exit "$gate_rc"
fi

echo "$(date -Is) sa follower starting SemiAnalysis FP8 vs FFDSL pair" >> "$SA_LOG"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-1}"
export PROFILE_PORT="${PROFILE_PORT:-8064}"
export PROFILE_CONC=32
export PROFILE_RUN_S=300
# Reuse the pair compare script layout but write a dedicated out/log.
export WEEKEND_LOG="$SA_LOG"
# pair_compare hardcodes its log; run a local thin cell script instead:
exec >> "$SA_LOG" 2>&1
ROOT=/workspace/vllm
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
OUT_ROOT=$ROOT/out/ffdsl_pair_sa
PORT="${PROFILE_PORT:-8064}"
CONC=32
RUN_S=300
mkdir -p "$OUT_ROOT"
export PATH="$VENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SA_DIR/.hf-cache}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export VLLM_ULTRAQUANT_FUSED_MTP=1
export VLLM_ULTRAQUANT_SHARED_KV=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_WAVES=2
cd "$ROOT"

log() { echo "$(date -Is) $*"; }
SERVER_PID=
cleanup_server() {
  [ -n "${SERVER_PID:-}" ] || return 0
  local pid="$SERVER_PID" c tree="$SERVER_PID"
  SERVER_PID=
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
  local name="$1" kv="$2" pair="$3"
  local out="$OUT_ROOT/$name"
  if [ -f "$out/metrics_end.txt" ] && [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "SKIP ${name} already complete"
    return 0
  fi
  cleanup_server
  mkdir -p "$out"
  export VLLM_ULTRAQUANT_PAIR_VERIFY="$pair"
  local extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')
  log "START ${name} hip=${HIP_VISIBLE_DEVICES} tp=1 conc=${CONC} kv=${kv} mtp=5 pair=${pair} rowpack=0"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$kv" \
    --max-model-len 262144 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching --no-enable-log-requests \
    --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
    --language-model-only --reasoning-parser qwen3 --trust-remote-code \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
    > "$out/server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0 i
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL ${name} server exited"
      cleanup_server
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL ${name} server not ready"
    cleanup_server
    return 1
  fi
  log "READY ${name}"
  run_aiperf "$out/aiperf_artifacts" "$RUN_S" "${extra[@]}" > "$out/benchmark.log" 2>&1 &
  local bench=$!
  until grep -q "Phase profiling (profiling) started" "$out/benchmark.log" 2>/dev/null; do
    if ! kill -0 "$bench" 2>/dev/null; then
      log "FAIL ${name} aiperf exited before measuring"
      cleanup_server
      return 1
    fi
    sleep 5
  done
  log "MEASURING ${name}"
  wait "$bench" || log "WARN ${name} aiperf rc=$?"
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' > "$out/metrics_end.txt" || true
  cleanup_server
  log "END ${name}"
}

log "ffdsl pair sa start pid=$$ ppid=$PPID hip=${HIP_VISIBLE_DEVICES} port=${PORT}"
run_cell qwen3-8-27b_fp8_mtp5_tp1_c32 fp8 0 || log "CELL fp8 stopped early"
run_cell qwen3-8-27b_ultraquant_ffdsl_pair_mtp5_tp1_c32 ultraquant_4bit 1 || log "CELL ffdsl stopped early"
log "ffdsl pair sa complete"
touch /workspace/logs/ffdsl-pair-sa.ok
