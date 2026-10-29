#!/bin/bash
# One lit-match cell: ARM × CONC on one GPU.
# HF thinking card + short prompts (128 from SPEED-747 subsample).
#
#   ARM=fp8_nospec|fp8_mtp|fp8_dspark|fp8_dflash \
#   CONC=1|8|32 HIP_VISIBLE_DEVICES=1 PORT=8710 \
#     bash benchmarks/ultraquant_mtp/lit_match_dspark_dflash/run_cell.sh
trap '' HUP
set -uo pipefail

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp/lit_match_dspark_dflash
AITER_SRC=/workspace/aiter
VENV=/workspace/.venv-uq
SA_DIR=/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
ARMS_JSON=$HERE/arms.json
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DFLASH_DRAFT=/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2
DSPARK_DRAFT=/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark

ARM="${ARM:?need ARM}"
CONC="${CONC:?need CONC}"
PORT="${PORT:?need PORT}"
UTIL="${UTIL:-0.90}"
MAX_LEN="${MAX_LEN:-32768}"
MAX_SEQS="${MAX_SEQS:-256}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MNBT="${MNBT:-8192}"
NREQ="${NREQ:-128}"
INPUT="${INPUT:-$HERE/datasets/lit_match_128.jsonl}"

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

ATTN=ROCM_AITER_FA
KV=fp8
SPEC_FILE=
unset VLLM_ULTRAQUANT_FUSED_MTP VLLM_ULTRAQUANT_PDV_VERIFY 2>/dev/null || true

case "$ARM" in
  fp8_nospec)
    SPEC_FILE=
    ;;
  fp8_mtp)
    SPEC_FILE=$HERE/configs/fp8_mtp.json
    ;;
  fp8_dspark)
    SPEC_FILE=$HERE/configs/fp8_dspark.json
    ;;
  fp8_dflash)
    SPEC_FILE=$HERE/configs/fp8_dflash.json
    ;;
  *) echo "$(date -Is) FAIL bad ARM=$ARM"; exit 1 ;;
esac

OUT=$ROOT/out/lit_match_qwen38/$ARM/c${CONC}_r1
mkdir -p "$OUT" /workspace/logs/lit_match_qwen38
LOG=/workspace/logs/lit_match_qwen38/${ARM}_c${CONC}.log
exec >>"$LOG" 2>&1
cd "$ROOT"
log() { echo "$(date -Is) $*"; }

if [ -f "$OUT/aiperf_artifacts/profile_export_aiperf.json" ] && [ -f "$OUT/accept_metrics.json" ]; then
  log "SKIP $ARM c$CONC"
  exit 0
fi
if [ ! -f "$INPUT" ]; then
  log "FAIL missing dataset $INPUT"
  exit 1
fi
if [ "$ARM" = fp8_dspark ] && [ ! -f "$DSPARK_DRAFT/config.json" ]; then
  log "FAIL missing draft $DSPARK_DRAFT"; exit 1
fi
if [ "$ARM" = fp8_dflash ] && [ ! -f "$DFLASH_DRAFT/config.json" ]; then
  log "FAIL missing draft $DFLASH_DRAFT"; exit 1
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
  # Lit-match max_len=32k needs far less KV than AgentX 256k; allow ~48 GiB free.
  WANT_UTIL="$want" python - <<'PY'
import os, time, torch
want = float(os.environ["WANT_UTIL"])
time.sleep(3)
for _ in range(8):
    free, total = torch.cuda.mem_get_info(0)
    util = min(want, max(0.40, (free / total) - 0.05))
    if free >= util * total and free / (1024**3) >= 48:
        print(f"{util:.4f}")
        raise SystemExit(0)
    time.sleep(8)
raise SystemExit(1)
PY
}

write_accept() {
  local metrics="$1" dest="$2"
  METRICS="$metrics" DEST="$dest" ARM="$ARM" CONC="$CONC" python - <<'PY'
import json, os, re
from pathlib import Path
text = Path(os.environ["METRICS"]).read_text(errors="replace") if Path(os.environ["METRICS"]).exists() else ""
def grab(name):
    total = 0.0
    found = False
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if not line.startswith(f"vllm:{name}"):
            continue
        metric = line.split("{", 1)[0]
        if metric != f"vllm:{name}":
            continue
        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            try:
                total += float(parts[1])
                found = True
            except Exception:
                pass
    return total if found else None
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
    "conc": int(os.environ["CONC"]),
    "output_tok_s": tok,
    "accept_rate": rate,
    "accept_length": al_bonus,
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
  echo FAIL >"$OUT/STATUS"
  exit 1
}

log "START arm=$ARM c=$CONC hip=$HIP_VISIBLE_DEVICES port=$PORT util=$cell_util"

serve_args=(
  serve "$MODEL" --port "$PORT" --tensor-parallel-size 1
  --attention-backend "$ATTN" --kv-cache-dtype "$KV"
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$cell_util"
  --enable-prefix-caching --no-enable-log-requests
  --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$MNBT"
  --language-model-only --reasoning-parser qwen3 --trust-remote-code
)
SPEC_JSON=
if [ -n "$SPEC_FILE" ]; then
  # vLLM --speculative-config expects a JSON string (json.loads), not a path.
  SPEC_JSON=$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",",":")))' "$SPEC_FILE")
  serve_args+=(--speculative-config "$SPEC_JSON")
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
    echo FAIL >"$OUT/STATUS"
    exit 1
  fi
  sleep 5
done
if [ "$ready" != 1 ]; then
  log "FAIL not ready"
  tail -n 80 "$OUT/server.log" || true
  echo FAIL >"$OUT/STATUS"
  exit 1
fi
log "READY"
echo RUNNING >"$OUT/STATUS"

# HF thinking card (literature-matched)
EXTRA_JSON=$(python - <<PY
import json
print(json.dumps({
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.0,
  "reasoning_effort": "xhigh",
  "max_completion_tokens": $MAX_TOKENS,
  "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
}))
PY
)

(cd "$INFERENCEX_DIR" && "$AIPERF" profile \
  --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model "$MODEL" --concurrency "$CONC" \
  --request-count "$NREQ" \
  --input-file "$INPUT" --custom-dataset-type multi_turn \
  --dataset-sampling-strategy sequential \
  --unsafe-override \
  --stats-interval 30 --random-seed 42 \
  --failed-request-threshold 0.15 \
  --warmup-request-count 8 \
  --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
  --output-artifact-dir "$OUT/aiperf_artifacts" \
  --extra-inputs "$EXTRA_JSON") >"$OUT/benchmark.log" 2>&1 &
bench=$!
wait "$bench" || log "WARN aiperf rc=$?"

curl -s "http://127.0.0.1:$PORT/metrics" \
  | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions|num_requests_running|generation_tokens)' \
  >"$OUT/metrics_end.txt" || true

{
  echo "arm=$ARM conc=$CONC"
  echo "attn=$ATTN kv=$KV spec=${SPEC_FILE:-none} mnbt=$MNBT"
  echo "max_len=$MAX_LEN max_seqs=$MAX_SEQS max_tokens=$MAX_TOKENS"
  echo "input=$INPUT nreq=$NREQ"
  echo "sampling=HF_thinking_T1.0"
} >"$OUT/cell_meta.txt"

write_accept "$OUT/metrics_end.txt" "$OUT/accept_metrics.json" | tee "$OUT/accept_metrics.line" || true
cleanup_server

if [ -f "$OUT/aiperf_artifacts/profile_export_aiperf.json" ]; then
  echo DONE >"$OUT/STATUS"
  log "END OK $ARM c$CONC"
  exit 0
fi
echo FAIL >"$OUT/STATUS"
log "FAIL no profile"
exit 1
