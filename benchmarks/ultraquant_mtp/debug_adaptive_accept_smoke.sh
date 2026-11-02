#!/bin/bash
# Short adaptive-on smoke: prove SpecDecoding accept and surface budget warnings.
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
PORT="${PORT:-8891}"
OUT=/workspace/vllm/out/debug_adaptive_accept_smoke
mkdir -p "$OUT"
LOG=$OUT/server.log
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DRAFT=/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark
SPEC=$(DRAFT="$DRAFT" python - <<'PY'
import json, os
cfg = {
  "method": "dspark",
  "model": os.environ["DRAFT"],
  "num_speculative_tokens": 7,
  "draft_sample_method": "probabilistic",
  "attention_backend": "ROCM_AITER_FA_UQ_DENSE",
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
echo "$(date -Is) smoke start hip=$HIP_VISIBLE_DEVICES util=$UTIL"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend ULTRAQUANT_SPEC_COMPAT --kv-cache-dtype ultraquant_4bit \
  --block-size 64 --max-model-len 8192 --gpu-memory-utilization "$UTIL" \
  --enable-prefix-caching --no-enable-log-requests --max-num-seqs 8 \
  --max-num-batched-tokens 2048 --language-model-only --reasoning-parser qwen3 \
  --trust-remote-code --speculative-config "$SPEC" >"$LOG" 2>&1 &
PID=$!
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q Qwen3.8-27B && break
  kill -0 $PID 2>/dev/null || { echo FAIL_EXIT; tail -n 40 "$LOG"; exit 1; }
  sleep 5
done
echo READY
# 8 concurrent short gens to exercise adaptive
python - <<PY
import concurrent.futures, json, urllib.request
port=$PORT
model="/huggingface/pretrained_models/Qwen/Qwen3.8-27B"
prompt="Write a step by step solution to the problem with rigorous reasoning and explanations (not just key points). Put the answer after the step-by-step solution, not before. Please put your final answer in \\\\boxed{}.\n\nWhat is 17*19?"
def one(i):
  body=json.dumps({
    "model":model,
    "messages":[{"role":"user","content":prompt}],
    "max_tokens":128,
    "temperature":0,
  }).encode()
  req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
  with urllib.request.urlopen(req, timeout=180) as r:
    return r.read()[:80]
with concurrent.futures.ThreadPoolExecutor(8) as ex:
  list(ex.map(one, range(8)))
print("GENS_DONE")
PY
sleep 12
echo "=== budget warnings ==="
grep -E "Adaptive verification draft_budget|DSpark cost tables" "$LOG" | head -40
echo "=== SpecDecoding sample ==="
grep -E "Avg Draft acceptance|Accepted:" "$LOG" | tail -20
python - <<'PY'
import re
from pathlib import Path
from collections import Counter
text=Path("/workspace/vllm/out/debug_adaptive_accept_smoke/server.log").read_text(errors="replace")
accs=[(int(a),int(d)) for a,d in re.findall(r"Accepted: (\d+) tokens, Drafted: (\d+)", text)]
print("n_windows", len(accs))
print("accepted_sum", sum(a for a,_ in accs), "drafted_sum", sum(d for _,d in accs))
if accs:
  print("accept_rate", sum(a for a,_ in accs)/max(1,sum(d for _,d in accs)))
  print("nonzero_windows", sum(1 for a,_ in accs if a>0), "/", len(accs))
PY
kill $PID 2>/dev/null; sleep 8; kill -9 $PID 2>/dev/null || true
echo SMOKE_END
