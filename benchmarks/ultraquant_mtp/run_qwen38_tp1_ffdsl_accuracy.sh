#!/bin/bash
# Stream 1: accuracy for Fused FlyDSL (FFDSL) pair vs FP8 AITER.
# GSM8K (short) and GPQA-Diamond (long). MTP 5, TP 1, concurrency 32.
# Row-pack stays off. Multi-seed only when the first-seed gap looks large.
trap '' HUP
mkdir -p /workspace/logs
exec >> /workspace/logs/ffdsl-pair-accuracy.log 2>&1
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EVAL_HARNESS=/workspace/eval-harness
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
GSM_CONFIG="$EVAL_HARNESS/configs/benchmarks/gsm8k_ffdsl.json"
GPQA_CONFIG="$EVAL_HARNESS/configs/benchmarks/gpqa_diamond_qwen38_card.json"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_ULTRAQUANT_FUSED_MTP=1
export VLLM_ULTRAQUANT_SHARED_KV=0
export VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
export VLLM_ULTRAQUANT_PAIR_WAVES=2

PORT="${ACCURACY_PORT:-8063}"
CONC="${ACCURACY_CONC:-32}"
MTP=5
PRIMARY_SEED="${ACCURACY_SEED:-7}"
EXTRA_SEEDS="${ACCURACY_EXTRA_SEEDS:-11 13}"
# Re-seed a bench when |fp8 - ffdsl| exceeds this many percentage points.
GAP_PP="${ACCURACY_GAP_PP:-1.0}"
OUT_ROOT="${ACCURACY_OUT:-$ROOT/out/ffdsl_pair_accuracy}"
mkdir -p "$OUT_ROOT"
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
  local out="$1"
  [ -f "$out/summary.json" ] && [ -f "$out/metrics_end.txt" ]
}

accuracy_of() {
  python3 -c "import json; print(json.load(open('$1/summary.json'))['accuracy'])"
}

start_server() {
  local kv="$1" pair="$2" out="$3" max_len="$4"
  cleanup_server
  mkdir -p "$out"
  export VLLM_ULTRAQUANT_PAIR_VERIFY="$pair"
  log "SERVE kv=${kv} pair=${pair} max_len=${max_len} conc=${CONC} mtp=${MTP} hip=${HIP_VISIBLE_DEVICES} port=${PORT}"
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
  log "READY kv=${kv} pair=${pair}"
}

run_gsm8k() {
  local tag="$1" kv="$2" pair="$3" seed="$4"
  local out="$OUT_ROOT/gsm8k_${tag}_seed${seed}"
  if cell_done "$out"; then
    log "SKIP gsm8k ${tag} seed=${seed} already complete acc=$(accuracy_of "$out")"
    return 0
  fi
  start_server "$kv" "$pair" "$out" 4096 || return 1
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
  local tag="$1" kv="$2" pair="$3" seed="$4"
  local out="$OUT_ROOT/gpqa_${tag}_seed${seed}"
  if cell_done "$out"; then
    log "SKIP gpqa ${tag} seed=${seed} already complete acc=$(accuracy_of "$out")"
    return 0
  fi
  start_server "$kv" "$pair" "$out" 262144 || return 1
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

maybe_extra_seeds() {
  local bench="$1"
  local fp8_out="$OUT_ROOT/${bench}_fp8_seed${PRIMARY_SEED}"
  local ffdsl_out="$OUT_ROOT/${bench}_ffdsl_seed${PRIMARY_SEED}"
  if [ ! -f "$fp8_out/summary.json" ] || [ ! -f "$ffdsl_out/summary.json" ]; then
    log "SEED ${bench}: primary incomplete; skip extras"
    return 0
  fi
  local gap
  gap=$(python3 -c "print(abs(float('$(accuracy_of "$fp8_out")')-float('$(accuracy_of "$ffdsl_out")')))")
  log "SEED ${bench}: primary gap=${gap}pp threshold=${GAP_PP}"
  python3 -c "import sys; sys.exit(0 if float('$gap') > float('$GAP_PP') else 1)" || {
    log "SEED ${bench}: gap within ${GAP_PP}pp; no extra seeds"
    return 0
  }
  log "SEED ${bench}: gap above ${GAP_PP}pp; running extra seeds: ${EXTRA_SEEDS}"
  local seed
  for seed in $EXTRA_SEEDS; do
    if [ "$bench" = "gsm8k" ]; then
      run_gsm8k fp8 fp8 0 "$seed" || true
      run_gsm8k ffdsl ultraquant_4bit 1 "$seed" || true
    else
      run_gpqa fp8 fp8 0 "$seed" || true
      run_gpqa ffdsl ultraquant_4bit 1 "$seed" || true
    fi
  done
}

write_report() {
  python3 - <<'PY'
import json, re
from pathlib import Path
root = Path("/workspace/vllm/out/ffdsl_pair_accuracy")
rows = []
for p in sorted(root.glob("*/summary.json")):
    j = json.loads(p.read_text())
    cell = p.parent.name
    mpath = p.parent / "metrics_end.txt"
    accept = None
    if mpath.exists():
        text = mpath.read_text()
        def grab(k):
            m = re.search(rf'vllm:{k}\{{[^}}]*\}} ([0-9.eE+-]+)', text)
            return float(m.group(1)) if m else None
        acc_n, draft = grab("spec_decode_num_accepted_tokens_total"), grab("spec_decode_num_draft_tokens_total")
        if acc_n is not None and draft:
            accept = 100.0 * acc_n / draft
    rows.append((cell, j.get("accuracy"), j.get("correct"), j.get("total"), accept))
report = root / "COMPARE.md"
lines = ["# FFDSL pair accuracy", "",
         "| cell | accuracy | correct/total | draft accept |",
         "|---|---:|---:|---:|"]
for cell, acc, correct, total, accept in rows:
    acc_s = f"{acc:.4f}" if acc is not None else "-"
    ct = f"{correct}/{total}" if correct is not None else "-"
    ap = f"{accept:.1f}%" if accept is not None else "-"
    lines.append(f"| {cell} | {acc_s} | {ct} | {ap} |")
report.write_text("\n".join(lines) + "\n")
print(report.read_text())
PY
}

log "ffdsl pair accuracy start pid=$$ ppid=$PPID hip=${HIP_VISIBLE_DEVICES} port=${PORT} conc=${CONC} mtp=${MTP} seed=${PRIMARY_SEED}"

# Short context first, then long context. FP8 baseline before FFDSL pair.
run_gsm8k fp8 fp8 0 "$PRIMARY_SEED" || true
run_gsm8k ffdsl ultraquant_4bit 1 "$PRIMARY_SEED" || true
maybe_extra_seeds gsm8k

run_gpqa fp8 fp8 0 "$PRIMARY_SEED" || true
run_gpqa ffdsl ultraquant_4bit 1 "$PRIMARY_SEED" || true
maybe_extra_seeds gpqa

write_report
log "ffdsl pair accuracy complete"
touch /workspace/logs/ffdsl-pair-accuracy.ok
