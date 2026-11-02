#!/bin/bash
# Sequential P1 GPU handoff on one HIP device:
#   1) longer-gen accept probe (serve held in bg, then killed)
#   2) AgentX c32 + prefix @ util=0.90 (full aiperf)
#   3) P2 UQ smoke
#   4) P3 FlyDSL smoke
# Each step frees the GPU before the next. HUP-safe.
trap '' HUP
set -uo pipefail
mkdir -p /workspace/logs
LOG=/workspace/logs/dflash-p1-gpu-queue.log
exec >>"$LOG" 2>&1

HIP="${HIP_VISIBLE_DEVICES:?need HIP}"
export HIP_VISIBLE_DEVICES="$HIP"
export CUDA_VISIBLE_DEVICES="$HIP"
unset ROCR_VISIBLE_DEVICES
export PATH=/workspace/.venv-uq/bin:$PATH
export PYTHONPATH=/workspace/aiter:/workspace/vllm
export PYTHONNOUSERSITE=1
export VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export VLLM_USE_V2_MODEL_RUNNER=1

ROOT=/workspace/vllm
HERE=$ROOT/benchmarks/ultraquant_mtp/dflash_qwen38
MODEL=/huggingface/pretrained_models/Qwen/Qwen3.8-27B
DRAFT="${DFLASH_DRAFT_DIR:-/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2}"
PORT_BASE="${DFLASH_QUEUE_PORT_BASE:-8361}"
UTIL="${DFLASH_GPU_UTIL:-0.90}"
STATE=/workspace/logs/dflash-p1-gpu-queue.state

log() { echo "$(date -Is) $*"; }
mark() { echo "$1" >"$STATE"; log "STATE $1"; }

kill_port() {
  local p="$1" pids
  pids=$(ps aux | awk -v pat="--port ${p}" '$0 ~ pat && $0 !~ /awk/ {print $2}')
  [ -n "$pids" ] && kill $pids 2>/dev/null || true
  sleep 10
  pids=$(ps aux | awk -v pat="--port ${p}" '$0 ~ pat && $0 !~ /awk/ {print $2}')
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
  # also kill orphan EngineCores writing to our outs
  sleep 5
}

wait_gpu_free() {
  local need_frac="${1:-0.90}" i ok
  for i in $(seq 1 90); do
    ok=$(python -c "import torch; f,t=torch.cuda.mem_get_info(0); print(1 if f>=0.98*${need_frac}*t else 0)")
    if [ "$ok" = 1 ]; then
      python -c "import torch; f,t=torch.cuda.mem_get_info(0); print(f'GPU free={f/2**30:.1f}GiB')"
      return 0
    fi
    log "waiting GPU reclaim i=$i need_frac=$need_frac"
    sleep 10
  done
  log "WARN GPU still tight; continuing"
}

make_spec() {
  local max_len="$1"
  DRAFT="$DRAFT" MAX_LEN="$max_len" python - <<'PY'
import json, os
cfg = json.load(open("/workspace/vllm/benchmarks/ultraquant_mtp/dflash_qwen38/configs/fp8_dflash2.json"))
cfg["model"] = os.environ["DRAFT"]
cfg["max_model_len"] = int(os.environ["MAX_LEN"])
print(json.dumps(cfg))
PY
}

log "QUEUE start hip=$HIP util=$UTIL port_base=$PORT_BASE pid=$$"
mark "start"
wait_gpu_free "$UTIL"

# ===================== 1) Accept (full GPU) =====================
mark "accept"
PORT=$PORT_BASE
kill_port "$PORT"
wait_gpu_free "$UTIL"
ACC_OUT=$ROOT/out/dflash_qwen38_fp8_agentx_debug/accept_full_gpu_u${UTIL}
mkdir -p "$ACC_OUT"
SPEC=$(make_spec 131072)
log "ACCEPT serve port=$PORT util=$UTIL"
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8 \
  --max-model-len 131072 --gpu-memory-utilization "$UTIL" \
  --enable-prefix-caching --no-enable-log-requests \
  --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  --enforce-eager \
  --speculative-config "$SPEC" \
  >"$ACC_OUT/server.log" 2>&1 &
ACC_PID=$!
ready=0
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" | grep -q "Qwen3.8-27B"; then ready=1; break; fi
  if ! kill -0 "$ACC_PID" 2>/dev/null; then
    log "FAIL accept serve exited"
    tail -n 40 "$ACC_OUT/server.log" || true
    break
  fi
  sleep 5
done
if [ "$ready" = 1 ]; then
  log "ACCEPT READY — longer-gen probe"
  PORT="$PORT" python - <<'PY' > /workspace/logs/dflash-accept-full-gpu.log 2>&1
import json, os, time, urllib.request
PORT = int(os.environ["PORT"])
MODEL = "/huggingface/pretrained_models/Qwen/Qwen3.8-27B"

def metrics():
    t = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30).read().decode()
    out = {}
    for line in t.splitlines():
        if line.startswith("#"):
            continue
        for k in (
            "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:spec_decode_num_draft_tokens_total",
        ):
            if line.startswith(k + "{") or line.startswith(k + " "):
                out[k] = float(line.split()[-1])
    return out

def chat(content, max_tokens=256):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read().decode())

print("ACCEPT_FULL_GPU start")
for n_words, mt in [(2040, 256), (8000, 256)]:
    prompt = ("word " * n_words) + (
        "Write a detailed 20-sentence explanation of binary search. Do not stop early."
    )
    m0 = metrics()
    t0 = time.time()
    resp = chat(prompt, mt)
    dt = time.time() - t0
    m1 = metrics()
    acc = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get(
        "vllm:spec_decode_num_accepted_tokens_total", 0
    )
    drf = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get(
        "vllm:spec_decode_num_draft_tokens_total", 0
    )
    rate = acc / drf if drf else float("nan")
    usage = resp.get("usage", {})
    print(
        f"words={n_words} dt={dt:.1f}s accepted={acc:.0f} drafted={drf:.0f} "
        f"rate={rate:.3f} usage={usage}"
    )
print("ACCEPT_DONE")
PY
  log "ACCEPT probe finished (see dflash-accept-full-gpu.log)"
  grep -E "words=|ACCEPT_|rate=" /workspace/logs/dflash-accept-full-gpu.log | tail -10 || true
else
  log "FAIL accept not ready — continuing to AgentX"
fi
kill_port "$PORT"
wait_gpu_free "$UTIL"

# ===================== 2) Full AgentX =====================
mark "agentx"
PORT=$PORT_BASE
kill_port "$PORT"
TAG="ml262144_c32_mnbt8192_u${UTIL}"
SA_OUT=$ROOT/out/dflash_qwen38_fp8_agentx_debug/$TAG
# Force a fresh AgentX (do not SKIP on prior artifact)
if [ -f "$SA_OUT/aiperf_artifacts/profile_export_aiperf.json" ]; then
  bak="$SA_OUT/aiperf_artifacts/profile_export_aiperf.json.bak_$(date +%Y%m%dT%H%M%SZ)"
  mv "$SA_OUT/aiperf_artifacts/profile_export_aiperf.json" "$bak"
  log "moved prior AgentX artifact → $bak"
fi
export DFLASH_PORT="$PORT" DFLASH_CONC=32 DFLASH_GPU_UTIL="$UTIL"
export DFLASH_EAGER=0 DFLASH_NO_PREFIX_CACHE=0 DFLASH_SPEC=1
export DFLASH_MODE=sa DFLASH_MAX_LEN=262144 DFLASH_MNBT=8192
export DFLASH_BENCH_DURATION=300 DFLASH_WARMUP_GRACE=300
log "AGENTX FULL start tag=$TAG hip=$HIP util=$UTIL prefix=ON"
bash "$HERE/run_qwen38_dflash2_fp8_agentx_debug.sh"
AX_RC=$?
log "AGENTX FULL finished rc=$AX_RC"
if [ -f "$SA_OUT/aiperf_artifacts/profile_export_aiperf.json" ]; then
  log "AGENTX artifact OK"
  python - <<PY
import json
p="$SA_OUT/aiperf_artifacts/profile_export_aiperf.json"
j=json.load(open(p))
def find(o, pred, depth=0):
    hits=[]
    if depth>8: return hits
    if isinstance(o, dict):
        for k,v in o.items():
            if pred(k) and isinstance(v,(int,float)): hits.append((k,v))
            elif pred(k) and isinstance(v,dict) and "avg" in v: hits.append((k+".avg", v["avg"]))
            else: hits += find(v, pred, depth+1)
    elif isinstance(o, list) and o:
        hits += find(o[0], pred, depth+1)
    return hits
toks=find(j, lambda k: k=="output_token_throughput" or k.endswith("output_token_throughput"))
itl=find(j, lambda k: "inter_token_latency" in k)
print("summary toks", toks[:3], "itl", itl[:3])
PY
else
  log "FAIL AGENTX missing artifact"
fi
# fault check
if grep -q "Memory access fault" "$SA_OUT/server.log" 2>/dev/null; then
  log "FAULT AgentX server.log has Memory access fault"
else
  log "NOFAULT AgentX"
fi
kill_port "$PORT"
wait_gpu_free "$UTIL"

# ===================== 3) P2 UQ smoke =====================
mark "p2_uq_smoke"
export DFLASH_PORT=$((PORT_BASE + 10)) DFLASH_GPU_UTIL="$UTIL" DFLASH_SMOKE_ONLY=1
rm -f $ROOT/out/dflash_qwen38_uq/smoke_ok.txt
bash "$HERE/run_qwen38_dflash2_uq.sh" || log "WARN P2 smoke rc=$?"
[ -f $ROOT/out/dflash_qwen38_uq/smoke_ok.txt ] && log "P2 SMOKE OK" || log "P2 SMOKE FAIL"
kill_port "$DFLASH_PORT"
wait_gpu_free "$UTIL"

# ===================== 4) P3 FlyDSL smoke =====================
mark "p3_flydsl_smoke"
export DFLASH_PORT=$((PORT_BASE + 11)) DFLASH_GPU_UTIL="$UTIL" DFLASH_SMOKE_ONLY=1
rm -f $ROOT/out/dflash_qwen38_flydsl/smoke_ok.txt
bash "$HERE/run_qwen38_dflash2_flydsl.sh" || log "WARN P3 smoke rc=$?"
[ -f $ROOT/out/dflash_qwen38_flydsl/smoke_ok.txt ] && log "P3 SMOKE OK" || log "P3 SMOKE FAIL"
kill_port "$DFLASH_PORT"

mark "done"
log "QUEUE ALL DONE hip=$HIP util=$UTIL"
