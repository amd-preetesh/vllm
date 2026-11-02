#!/bin/bash
# Stream 1: PDV FlyDSL is the verify default.
# GSM8K (short, 3 seeds) and GPQA-Diamond (long, 3 seeds), FP8 then PDV,
# then the same 300s SemiAnalysis AgentX window as the pair compare.
# One GPU. Draft steps stay on production FlyDSL.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/pdv-stream1.log 2>&1
set -uo pipefail

ROOT=/workspace/vllm
EVAL_HARNESS=/workspace/eval-harness
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
GSM_CONFIG="$EVAL_HARNESS/configs/benchmarks/gsm8k_ffdsl.json"
GPQA_CONFIG="$EVAL_HARNESS/configs/benchmarks/gpqa_diamond_qwen38_card.json"
PRIOR_ACC="$ROOT/out/ffdsl_pair_accuracy"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
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

PORT="${PDV_PORT:-8071}"
CONC=32
MTP=5
SEEDS="7 11 13"
ACC_ROOT="$ROOT/out/pdv_stream1_accuracy"
SA_ROOT="$ROOT/out/pdv_stream1_sa"
mkdir -p "$ACC_ROOT" "$SA_ROOT"
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

cell_done() {
  [ -f "$1/summary.json" ] && [ -f "$1/metrics_end.txt" ]
}

accuracy_of() {
  python3 -c "import json; print(json.load(open('$1/summary.json'))['accuracy'])"
}

reuse_prior() {
  local name="$1"
  local src="$PRIOR_ACC/$name"
  local dst="$ACC_ROOT/$name"
  if cell_done "$dst"; then
    return 0
  fi
  if cell_done "$src"; then
    mkdir -p "$dst"
    cp -a "$src/." "$dst/"
    log "REUSE ${name} acc=$(accuracy_of "$dst")"
  fi
}

start_server() {
  local kv="$1" mode="$2" out="$3" max_len="$4"
  cleanup_server
  mkdir -p "$out"
  if [ "$mode" = "pdv" ]; then
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
  else
    export VLLM_ULTRAQUANT_PDV_VERIFY=0
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
  fi
  log "SERVE kv=${kv} mode=${mode} max_len=${max_len} conc=${CONC} mtp=${MTP} hip=${HIP_VISIBLE_DEVICES} port=${PORT}"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$kv" \
    --max-model-len "$max_len" --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --no-enable-log-requests \
    --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
    --language-model-only --reasoning-parser qwen3 --trust-remote-code \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}" \
    > "$out/server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0 i
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAIL server exited; see ${out}/server.log"
      cleanup_server
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL server not ready; see ${out}/server.log"
    cleanup_server
    return 1
  fi
  if [ "$mode" = "pdv" ]; then
    if ! grep -q "PDV FlyDSL verify is enabled" "$out/server.log"; then
      log "FAIL PDV verify line missing from ${out}/server.log"
      cleanup_server
      return 1
    fi
  fi
  log "READY kv=${kv} mode=${mode}"
}

run_gsm8k() {
  local tag="$1" kv="$2" mode="$3" seed="$4"
  local out="$ACC_ROOT/gsm8k_${tag}_seed${seed}"
  if cell_done "$out"; then
    log "SKIP gsm8k ${tag} seed=${seed} acc=$(accuracy_of "$out")"
    return 0
  fi
  start_server "$kv" "$mode" "$out" 4096 || return 1
  log "EVAL gsm8k ${tag} seed=${seed}"
  python "$EVAL_HARNESS/drivers/gsm8k_native.py" \
    --port "$PORT" --model "$MODEL" --max_model_len 4096 \
    --bench_config "$GSM_CONFIG" \
    --out_dir "$out" --seed "$seed" \
    --concurrency_override "$CONC" \
    --enable_thinking false \
    > "$out/eval.log" 2>&1 || log "WARN gsm8k ${tag} seed=${seed} driver rc=$?"
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' > "$out/metrics_end.txt" || true
  cleanup_server
  if [ -f "$out/summary.json" ]; then
    log "END gsm8k ${tag} seed=${seed} acc=$(accuracy_of "$out")"
  else
    log "FAIL gsm8k ${tag} seed=${seed} missing summary.json"
    return 1
  fi
}

run_gpqa() {
  local tag="$1" kv="$2" mode="$3" seed="$4"
  local out="$ACC_ROOT/gpqa_${tag}_seed${seed}"
  if cell_done "$out"; then
    log "SKIP gpqa ${tag} seed=${seed} acc=$(accuracy_of "$out")"
    return 0
  fi
  start_server "$kv" "$mode" "$out" 262144 || return 1
  log "EVAL gpqa ${tag} seed=${seed}"
  python "$EVAL_HARNESS/drivers/gpqa_native.py" \
    --port "$PORT" --model "$MODEL" --max_model_len 262144 \
    --bench_config "$GPQA_CONFIG" \
    --out_dir "$out" --seed "$seed" \
    --concurrency_override "$CONC" \
    > "$out/eval.log" 2>&1 || log "WARN gpqa ${tag} seed=${seed} driver rc=$?"
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' > "$out/metrics_end.txt" || true
  cleanup_server
  if [ -f "$out/summary.json" ]; then
    log "END gpqa ${tag} seed=${seed} acc=$(accuracy_of "$out")"
  else
    log "FAIL gpqa ${tag} seed=${seed} missing summary.json"
    return 1
  fi
}

write_report() {
  python3 - <<'PY'
import json, re
from pathlib import Path
root = Path("/workspace/vllm/out/pdv_stream1_accuracy")
rows = []
for p in sorted(root.glob("*/summary.json")):
    j = json.loads(p.read_text())
    cell = p.parent.name
    mpath = p.parent / "metrics_end.txt"
    accept = None
    if mpath.exists():
        text = mpath.read_text()
        def grab(k, text=text):
            m = re.search(rf'vllm:{k}\{{[^}}]*\}} ([0-9.eE+-]+)', text)
            return float(m.group(1)) if m else None
        acc_n, draft = grab("spec_decode_num_accepted_tokens_total"), grab("spec_decode_num_draft_tokens_total")
        if acc_n is not None and draft:
            accept = 100.0 * acc_n / draft
    rows.append((cell, j.get("accuracy"), j.get("correct"), j.get("total"), accept))
lines = ["# PDV stream 1 accuracy", "",
         "| cell | accuracy | correct/total | draft accept |",
         "|---|---:|---:|---:|"]
for cell, acc, correct, total, accept in rows:
    acc_s = f"{acc:.4f}" if acc is not None else "-"
    ct = f"{correct}/{total}" if correct is not None else "-"
    ap = f"{accept:.1f}%" if accept is not None else "-"
    lines.append(f"| {cell} | {acc_s} | {ct} | {ap} |")
report = root / "COMPARE.md"
report.write_text("\n".join(lines) + "\n")
print(report.read_text())
PY
}

run_aiperf() {
  local out_dir="$1"
  shift
  (cd "$INFERENCEX_DIR" && "$AIPERF" profile --scenario inferencex-agentx-mvp \
    --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
    --model "$MODEL" --concurrency "$CONC" --benchmark-duration 300 \
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

run_sa_cell() {
  local name="$1" kv="$2" mode="$3"
  local out="$SA_ROOT/$name"
  if [ -f "$out/metrics_end.txt" ] && [ -f "$out/aiperf_artifacts/profile_export_aiperf.json" ]; then
    log "SKIP SA ${name} already complete"
    return 0
  fi
  cleanup_server
  mkdir -p "$out"
  if [ "$mode" = "pdv" ]; then
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
  else
    export VLLM_ULTRAQUANT_PDV_VERIFY=0
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
  fi
  log "SA START ${name} hip=${HIP_VISIBLE_DEVICES} kv=${kv} mode=${mode}"
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
      log "FAIL SA ${name} server exited"
      cleanup_server
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "FAIL SA ${name} server not ready"
    cleanup_server
    return 1
  fi
  if [ "$mode" = "pdv" ] && ! grep -q "PDV FlyDSL verify is enabled" "$out/server.log"; then
    log "FAIL SA ${name} PDV line missing"
    cleanup_server
    return 1
  fi
  local extra=(--extra-inputs '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}')
  run_aiperf "$out/aiperf_artifacts" "${extra[@]}" > "$out/benchmark.log" 2>&1 &
  local bench=$!
  until grep -q "Phase profiling (profiling) started" "$out/benchmark.log" 2>/dev/null; do
    if ! kill -0 "$bench" 2>/dev/null; then
      log "FAIL SA ${name} aiperf exited before measuring"
      cleanup_server
      return 1
    fi
    sleep 5
  done
  log "SA MEASURING ${name}"
  wait "$bench" || log "WARN SA ${name} aiperf rc=$?"
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' > "$out/metrics_end.txt" || true
  cleanup_server
  log "SA END ${name}"
}

log "pdv stream1 start pid=$$ ppid=$PPID hip=${HIP_VISIBLE_DEVICES} port=${PORT} conc=${CONC} mtp=${MTP}"

reuse_prior gsm8k_fp8_seed7
reuse_prior gpqa_fp8_seed7
reuse_prior gpqa_fp8_seed11
reuse_prior gpqa_fp8_seed13

for seed in $SEEDS; do
  run_gsm8k fp8 fp8 fp8 "$seed" || true
  run_gsm8k pdv ultraquant_4bit pdv "$seed" || true
done
for seed in $SEEDS; do
  run_gpqa fp8 fp8 fp8 "$seed" || true
  run_gpqa pdv ultraquant_4bit pdv "$seed" || true
done
write_report

python3 - <<'PY'
import json, sys
from pathlib import Path
root = Path("/workspace/vllm/out/pdv_stream1_accuracy")
missing = []
gaps = []
for bench in ("gsm8k", "gpqa"):
    for seed in (7, 11, 13):
        fp = root / f"{bench}_fp8_seed{seed}/summary.json"
        pd = root / f"{bench}_pdv_seed{seed}/summary.json"
        if not fp.exists() or not pd.exists():
            missing.append(f"{bench} seed {seed}")
            continue
        fa = json.loads(fp.read_text())["accuracy"]
        pa = json.loads(pd.read_text())["accuracy"]
        gaps.append((bench, seed, fa, pa, pa - fa))
        print(f"gate {bench} seed {seed}: fp8={fa:.4f} pdv={pa:.4f} gap={pa-fa:+.4f}pp")
if missing:
    print("sa gate FAIL missing", missing)
    sys.exit(2)
by = {}
for bench, seed, fa, pa, gap in gaps:
    by.setdefault(bench, []).append(gap)
for bench, vals in by.items():
    mean = sum(vals) / len(vals)
    print(f"gate {bench} mean gap={mean:+.4f}pp over {len(vals)} seeds")
    if mean < -2.0:
        print("sa gate FAIL: PDV more than 2pp below FP8")
        sys.exit(3)
print("sa gate PASS")
PY
gate_rc=$?
if [ "$gate_rc" != 0 ]; then
  log "pdv stream1 stopped before SemiAnalysis gate_rc=${gate_rc}"
  exit "$gate_rc"
fi

run_sa_cell qwen3-8-27b_fp8_mtp5_tp1_c32 fp8 fp8 || log "CELL fp8 sa stopped early"
run_sa_cell qwen3-8-27b_ultraquant_pdv_mtp5_tp1_c32 ultraquant_4bit pdv || log "CELL pdv sa stopped early"
log "pdv stream1 complete"
touch /workspace/logs/pdv-stream1.ok
