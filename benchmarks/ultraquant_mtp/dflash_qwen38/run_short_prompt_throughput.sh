#!/bin/bash
# Fair-speed vs HF DFlash2 card: short prompts, TP=1, conc sweep.
# Not AgentX — GSM-like chat completions for tok/s at c1/c8/c32.
trap '' HUP
mkdir -p /workspace/logs
LOG=/workspace/logs/dflash-short-prompt-tput.log
exec >>"$LOG" 2>&1
set -uo pipefail

ROOT=/workspace/vllm
VENV=/workspace/.venv-uq
AITER_SRC=/workspace/aiter
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
HERE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38
OUT=$ROOT/out/dflash_qwen38_fp8_short_tput
PORT="${DFLASH_PORT:-8380}"
UTIL="${DFLASH_GPU_UTIL:-0.90}"
CONCS="${DFLASH_SHORT_CONCS:-1 8 32}"

export PATH="$VENV/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?need one HIP device}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$AITER_SRC:$ROOT"
export VLLM_USE_V2_MODEL_RUNNER=1

log() { echo "$(date -Is) $*"; }
mkdir -p "$OUT"

SPEC=$(DRAFT="$DRAFT" python - <<'PY'
import json, os
cfg = json.load(open("/workspace/vllm/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2.json"))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = 8192
print(json.dumps(cfg))
PY
)

log "SHORT TPUT start hip=$HIP_VISIBLE_DEVICES port=$PORT util=$UTIL tp=1 concs=$CONCS"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8 \
  --max-model-len 8192 --gpu-memory-utilization "$UTIL" \
  --enable-prefix-caching --no-enable-log-requests \
  --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  --speculative-config "$SPEC" \
  >"$OUT/server.log" 2>&1 &
SPID=$!
ready=0
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" | grep -q "Qwen3.8-27B"; then ready=1; break; fi
  if ! kill -0 "$SPID" 2>/dev/null; then log "FAIL server"; tail -n 40 "$OUT/server.log"; exit 1; fi
  sleep 5
done
[ "$ready" = 1 ] || { log "FAIL not ready"; exit 1; }
log "READY short-tput"

PORT="$PORT" MODEL="$MODEL" OUT="$OUT" CONCS="$CONCS" python - <<'PY'
import json, os, time, concurrent.futures, urllib.request
PORT=int(os.environ["PORT"]); MODEL=os.environ["MODEL"]; OUT=os.environ["OUT"]
concs=[int(x) for x in os.environ["CONCS"].split()]

def one(max_tokens=256):
    body={
        "model": MODEL,
        "messages":[{"role":"user","content":"Solve: If x+3=10, what is x? Show brief steps then the answer."}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "chat_template_kwargs":{"enable_thinking": True, "preserve_thinking": True},
        "reasoning_effort": "xhigh",
    }
    req=urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"},
    )
    t0=time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp=json.loads(r.read().decode())
    dt=time.time()-t0
    usage=resp.get("usage",{})
    return usage.get("completion_tokens",0), dt

rows=[]
for c in concs:
    # warm
    one(32)
    n = max(c*4, 8)
    t0=time.time()
    toks=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        futs=[ex.submit(one, 256) for _ in range(n)]
        for f in concurrent.futures.as_completed(futs):
            ct, _ = f.result()
            toks += ct
    wall=time.time()-t0
    tps = toks/wall if wall else 0
    row={"conc":c,"requests":n,"completion_tokens":toks,"wall_s":wall,"tok_s":tps}
    rows.append(row)
    print(json.dumps(row), flush=True)

path=os.path.join(OUT,"summary.json")
json.dump({"rows":rows,"note":"short-prompt fair-speed TP=1 vs HF card"}, open(path,"w"), indent=2)
print("WROTE", path, flush=True)
PY

kill "$SPID" 2>/dev/null || true
sleep 10
kill -9 "$SPID" 2>/dev/null || true
log "SHORT TPUT done"
