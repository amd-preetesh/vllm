#!/bin/bash
# One method-match cell: arm × workload × concurrency × rep.
# New experiment path — does not edit shipping FP8/UQ AgentX runners.
#
#   ARM=dspark_fp8|dflash_fp8|dspark_uq_opt|dflash_uq_swa \
#   WORKLOAD=spec_bench|speed747 CONC=32 REP=1 \
#   HIP_VISIBLE_DEVICES=0 PORT=8700 \
#     bash benchmarks/ultraquant_mtp/method_match/run_cell.sh
trap '' HUP
set -uo pipefail

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp/method_match
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DFLASH_DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
DSPARK_DRAFT="${DSPARK_DRAFT_DIR:-/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark}"

ARM="${ARM:?need ARM}"
WORKLOAD="${WORKLOAD:?need WORKLOAD}"
CONC="${CONC:?need CONC}"
REP="${REP:-1}"
PORT="${PORT:?need PORT}"
UTIL="${UTIL:-0.90}"
MAX_LEN="${MAX_LEN:-32768}"
MAX_SEQS="${MAX_SEQS:-256}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SA_DIR/.hf-cache}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export VLLM_USE_V2_MODEL_RUNNER=1

case "$HIP_VISIBLE_DEVICES" in
  [0-7]) ;;
  *) echo "$(date -Is) FAIL HIP=$HIP_VISIBLE_DEVICES"; exit 1 ;;
esac

ATTN=
KV=
SPEC_FILE=
DRAFT=
OUT_STEM=
MNBT=8192
RETENTION=
EXTRA_ENV=()

case "$ARM" in
  dspark_fp8)
    DRAFT="$DSPARK_DRAFT"
    SPEC_FILE=$HERE/configs/fp8_dspark_kv8_triton.json
    ATTN=ROCM_AITER_FA
    KV=fp8
    OUT_STEM=method_match_dspark_fp8
    unset VLLM_ULTRAQUANT_FUSED_MTP VLLM_ULTRAQUANT_PDV_VERIFY || true
    ;;
  dflash_fp8)
    DRAFT="$DFLASH_DRAFT"
    SPEC_FILE=$HERE/configs/fp8_dflash2_kv8_triton.json
    ATTN=ROCM_AITER_FA
    KV=fp8
    OUT_STEM=method_match_dflash_fp8
    unset VLLM_ULTRAQUANT_FUSED_MTP VLLM_ULTRAQUANT_PDV_VERIFY || true
    ;;
  dspark_uq_opt)
    DRAFT="$DSPARK_DRAFT"
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dspark_qwen38_uq/configs/uq_dspark_dense.json
    ATTN=ULTRAQUANT_SPEC_COMPAT
    KV=ultraquant_4bit
    OUT_STEM=method_match_dspark_uq_opt
    MNBT=2048
    RETENTION=50176
    export VLLM_ULTRAQUANT_FUSED_MTP=1
    export VLLM_ULTRAQUANT_SHARED_KV=0
    export VLLM_ULTRAQUANT_V3=0
    export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
    export VLLM_ULTRAQUANT_PAIR_WAVES=2
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
    unset VLLM_ULTRAQUANT_PDV_EXPERIMENT
    ;;
  dflash_uq_swa)
    DRAFT="$DFLASH_DRAFT"
    SPEC_FILE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38_uq/configs/uq_dflash2_swa.json
    ATTN=ULTRAQUANT_SPEC_COMPAT
    KV=ultraquant_4bit
    OUT_STEM=method_match_dflash_uq_swa
    MNBT=2048
    RETENTION=50176
    export VLLM_ULTRAQUANT_FUSED_MTP=1
    export VLLM_ULTRAQUANT_SHARED_KV=0
    export VLLM_ULTRAQUANT_V3=0
    export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0
    export VLLM_ULTRAQUANT_PAIR_WAVES=2
    export VLLM_ULTRAQUANT_PAIR_VERIFY=0
    export VLLM_ULTRAQUANT_PDV_VERIFY=1
    unset VLLM_ULTRAQUANT_PDV_EXPERIMENT
    ;;
  *) echo "$(date -Is) FAIL bad ARM=$ARM"; exit 1 ;;
esac

case "$WORKLOAD" in
  spec_bench)
    INPUT=$HERE/datasets/spec_bench_first_turn.jsonl
    ;;
  speed747)
    INPUT=$HERE/datasets/speed_bench_throughput_1k_balanced_747.jsonl
    ;;
  *) echo "$(date -Is) FAIL bad WORKLOAD=$WORKLOAD"; exit 1 ;;
esac
NREQ=$(grep -c . "$INPUT" 2>/dev/null || echo 0)

OUT=$ROOT/out/$OUT_STEM/$WORKLOAD/c${CONC}_r${REP}
mkdir -p "$OUT" /workspace/logs
LOG=/workspace/logs/method-match-${ARM}-${WORKLOAD}-c${CONC}-r${REP}.log
exec >>"$LOG" 2>&1
cd "$ROOT"
log() { echo "$(date -Is) $*"; }

if [ -f "$OUT/aiperf_artifacts/profile_export_aiperf.json" ] && [ -f "$OUT/accept_metrics.json" ]; then
  log "SKIP $ARM $WORKLOAD c$CONC r$REP"
  exit 0
fi
if [ ! -f "$INPUT" ]; then
  log "FAIL missing dataset $INPUT — run prepare_datasets.py"
  exit 1
fi
if [ ! -f "$DRAFT/config.json" ]; then
  log "FAIL missing draft $DRAFT"
  exit 1
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

fit_util_for_serve() {
  local want="$1"
  WANT_UTIL="$want" python - <<'PY'
import os, time, torch
want = float(os.environ["WANT_UTIL"])
time.sleep(3)
for _ in range(8):
    free, total = torch.cuda.mem_get_info(0)
    util = min(want, max(0.40, (free / total) - 0.03))
    if free >= util * total and free / (1024**3) >= 80:
        print(f"{util:.4f}")
        raise SystemExit(0)
    time.sleep(8)
raise SystemExit(1)
PY
}

make_spec() {
  DRAFT="$DRAFT" MAX_LEN="$MAX_LEN" CFG="$SPEC_FILE" python - <<'PY'
import json, os
cfg = json.load(open(os.environ["CFG"]))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = int(os.environ["MAX_LEN"])
print(json.dumps(cfg))
PY
}

write_accept() {
  local metrics="$1" dest="$2"
  METRICS="$metrics" DEST="$dest" ARM="$ARM" WORKLOAD="$WORKLOAD" CONC="$CONC" REP="$REP" python - <<'PY'
import json, os, re
from pathlib import Path
text = Path(os.environ["METRICS"]).read_text(errors="replace") if Path(os.environ["METRICS"]).exists() else ""
def grab(name):
    m = re.search(rf'^vllm:{name}\{{.*?\}}\s+([0-9.eE+-]+)', text, re.M)
    return float(m.group(1)) if m else None
drafts = grab("spec_decode_num_drafts_total")
acc = grab("spec_decode_num_accepted_tokens_total")
dtok = grab("spec_decode_num_draft_tokens_total")
tok = None
jp = Path(os.environ["DEST"]).parent / "aiperf_artifacts/profile_export_aiperf.json"
if jp.exists():
    d = json.load(open(jp))
    def find(o, k="output_token_throughput"):
        if isinstance(o, dict):
            if k in o:
                v = o[k]
                return v.get("avg") if isinstance(v, dict) else v
            for x in o.values():
                r = find(x, k)
                if r is not None:
                    return r
        if isinstance(o, list):
            for x in o:
                r = find(x, k)
                if r is not None:
                    return r
        return None
    tok = find(d)
rate = (acc / dtok) if acc is not None and dtok else None
al = (acc / drafts) if acc is not None and drafts else None
al_bonus = (1.0 + al) if al is not None else None
out = {
    "arm": os.environ["ARM"],
    "workload": os.environ["WORKLOAD"],
    "conc": int(os.environ["CONC"]),
    "rep": int(os.environ["REP"]),
    "output_tok_s": tok,
    "accept_rate": rate,
    "accept_length": al_bonus,  # 1 + accepted/draft_step (incl. bonus)
    "accept_length_draft_only": al,
    "num_drafts": drafts,
    "num_accepted_tokens": acc,
    "num_draft_tokens": dtok,
}
Path(os.environ["DEST"]).write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
PY
}

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi
cleanup_server

cell_util=$(fit_util_for_serve "$UTIL") || {
  log "FAIL fit_util"
  exit 1
}
spec=$(make_spec)
log "START arm=$ARM wl=$WORKLOAD c=$CONC r=$REP hip=$HIP_VISIBLE_DEVICES port=$PORT util=$cell_util attn=$ATTN kv=$KV"

serve_args=(
  serve "$MODEL" --port "$PORT" --tensor-parallel-size 1
  --attention-backend "$ATTN" --kv-cache-dtype "$KV"
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$cell_util"
  --enable-prefix-caching --no-enable-log-requests
  --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$MNBT"
  --language-model-only --reasoning-parser qwen3 --trust-remote-code
  --speculative-config "$spec"
)
# FP8 method-match uses block 16 default; UQ best uses 64
if [[ "$ARM" == *_uq_* ]]; then
  serve_args+=(--block-size 64)
fi
if [ -n "$RETENTION" ]; then
  serve_args+=(--prefix-cache-retention-interval "$RETENTION")
fi

vllm "${serve_args[@]}" >"$OUT/server.log" 2>&1 &
SERVER_PID=$!
ready=0
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen3.8-27B"; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "FAIL server exited"
    tail -n 80 "$OUT/server.log" || true
    exit 1
  fi
  sleep 5
done
if [ "$ready" != 1 ]; then
  log "FAIL not ready"
  tail -n 80 "$OUT/server.log" || true
  exit 1
fi
log "READY"

# Method-match sampling: temp=0, thinking disabled, max 1024 gen tokens.
extra=(--extra-inputs "{\"temperature\":0,\"max_completion_tokens\":$MAX_TOKENS,\"chat_template_kwargs\":{\"enable_thinking\":false}}")

(cd "$INFERENCEX_DIR" && "$AIPERF" profile \
  --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model "$MODEL" --concurrency "$CONC" \
  --request-count "$NREQ" \
  --input-file "$INPUT" --custom-dataset-type multi_turn \
  --dataset-sampling-strategy sequential \
  --unsafe-override \
  --stats-interval 30 --random-seed "$((42 + REP))" \
  --failed-request-threshold 0.15 \
  --warmup-request-count 8 \
  --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
  --output-artifact-dir "$OUT/aiperf_artifacts" \
  "${extra[@]}") >"$OUT/benchmark.log" 2>&1 &
bench=$!
wait "$bench" || log "WARN aiperf rc=$?"

curl -s "http://127.0.0.1:$PORT/metrics" \
  | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions|num_requests_running|generation_tokens)' \
  >"$OUT/metrics_end.txt" || true

{
  echo "arm=$ARM workload=$WORKLOAD conc=$CONC rep=$REP"
  echo "attn=$ATTN kv=$KV spec=$SPEC_FILE mnbt=$MNBT retention=${RETENTION:-0}"
  echo "max_len=$MAX_LEN max_seqs=$MAX_SEQS max_tokens=$MAX_TOKENS"
  echo "input=$INPUT nreq=$NREQ"
} >"$OUT/cell_meta.txt"

write_accept "$OUT/metrics_end.txt" "$OUT/accept_metrics.json" | tee "$OUT/accept_metrics.line" || true
cleanup_server

if [ -f "$OUT/aiperf_artifacts/profile_export_aiperf.json" ]; then
  log "END OK $ARM $WORKLOAD c$CONC r$REP"
  exit 0
fi
log "FAIL no profile"
exit 1
