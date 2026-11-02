#!/bin/bash
# Adaptive smoke after PR #53929 / 3462da7 capture-geometry + hash fixes.
# Cold torch-compile cache (mandatory — shared cache masks the bug).
# Path A: UQ SPEC_COMPAT (our AgentX stack). Path B optional via ATTN=ROCM_AITER_FA.
trap '' HUP
set -uo pipefail
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
export PATH=/workspace/.venv-uq/bin:$PATH
export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH=/workspace/aiter:/workspace/vllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ULTRAQUANT_FUSED_MTP=1 VLLM_ULTRAQUANT_PDV_VERIFY=1
export VLLM_ULTRAQUANT_SHARED_KV=0 VLLM_ULTRAQUANT_V3=0
export VLLM_ULTRAQUANT_ROWPACK_VERIFY=0 VLLM_ULTRAQUANT_PAIR_WAVES=2
export VLLM_ULTRAQUANT_PAIR_VERIFY=0
unset VLLM_ULTRAQUANT_PDV_EXPERIMENT

ROOT=/workspace/vllm
PORT="${PORT:-8893}"
OUT=/workspace/vllm/out/debug_adaptive_pr53929_smoke
ATTN="${ATTN:-ULTRAQUANT_SPEC_COMPAT}"
KV="${KV:-ultraquant_4bit}"
mkdir -p "$OUT"
LOG=$OUT/server.log
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DRAFT=/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark

# Isolate compile cache so adaptive does not reuse non-adaptive artifacts.
export VLLM_CACHE_ROOT="$OUT/vllm_cache"
rm -rf "$VLLM_CACHE_ROOT"
mkdir -p "$VLLM_CACHE_ROOT"

SPEC=$(DRAFT="$DRAFT" python - <<'PY'
import json, os
cfg = {
  "method": "dspark",
  "model": os.environ["DRAFT"],
  "num_speculative_tokens": 7,
  "draft_sample_method": "probabilistic",
  "attention_backend": "ROCM_AITER_FA_UQ_DENSE" if os.environ.get("ATTN","ULTRAQUANT_SPEC_COMPAT")=="ULTRAQUANT_SPEC_COMPAT" else "TRITON_ATTN",
  "kv_cache_dtype": "auto",
  "enable_adaptive_verification": True,
  "max_model_len": 8192,
}
print(json.dumps(cfg))
PY
)
UTIL=$(python - <<'PY'
import torch
free,total=torch.cuda.mem_get_info(0)
print(f"{min(0.70, max(0.55,(free-8*(1024**3))/total)):.4f}")
PY
)
echo "$(date -Is) pr53929 smoke start hip=$HIP_VISIBLE_DEVICES util=$UTIL attn=$ATTN kv=$KV cache=$VLLM_CACHE_ROOT"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend "$ATTN" --kv-cache-dtype "$KV" \
  --block-size 64 --max-model-len 8192 --gpu-memory-utilization "$UTIL" \
  --enable-prefix-caching --no-enable-log-requests --max-num-seqs 8 \
  --max-num-batched-tokens 2048 --language-model-only --reasoning-parser qwen3 \
  --trust-remote-code --speculative-config "$SPEC" >"$LOG" 2>&1 &
PID=$!
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q Qwen3.8-27B && break
  kill -0 $PID 2>/dev/null || { echo FAIL_EXIT; tail -n 60 "$LOG"; exit 1; }
  sleep 5
done
echo READY

# Concurrent greedy gens — PR validation used coherent greedy; temp=0.
python - <<PY
import concurrent.futures, json, urllib.request
port=$PORT
model="/huggingface/pretrained_models/Qwen/Qwen3.8-27B"
prompt="Write a step by step solution. Put the answer in \\\\boxed{}.\n\nWhat is 17*19?"
def one(i):
  body=json.dumps({
    "model":model,
    "messages":[{"role":"user","content":prompt}],
    "max_tokens":128,
    "temperature":0,
  }).encode()
  req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
  with urllib.request.urlopen(req, timeout=300) as r:
    j=json.loads(r.read())
    txt=j["choices"][0]["message"].get("content") or ""
    return i, txt[:120].replace("\n"," ")
with concurrent.futures.ThreadPoolExecutor(8) as ex:
  outs=list(ex.map(one, range(8)))
for i,t in outs:
  print(f"GEN{i}: {t}")
print("GENS_DONE")
PY
sleep 15
echo "=== SpecDecoding metrics (last 8) ==="
grep "Avg Draft acceptance rate" "$LOG" | tail -n 8
echo "=== accept summary ==="
python - <<PY
import re
from pathlib import Path
text=Path("$LOG").read_text(errors="replace")
rates=[]
lens=[]
for m in re.finditer(r"Avg Draft acceptance rate: ([0-9.]+)%", text):
  rates.append(float(m.group(1)))
for m in re.finditer(r"Mean acceptance length: ([0-9.]+)", text):
  lens.append(float(m.group(1)))
print(f"n_metric_lines={len(rates)} last_accept%={rates[-1] if rates else None} last_mean_len={lens[-1] if lens else None}")
print(f"max_accept%={max(rates) if rates else None} max_mean_len={max(lens) if lens else None}")
ok = bool(rates) and max(rates) > 5.0
print("SMOKE_PASS" if ok else "SMOKE_FAIL")
raise SystemExit(0 if ok else 1)
PY
RC=$?
kill $PID 2>/dev/null || true
sleep 8
kill -9 $PID 2>/dev/null || true
exit $RC
