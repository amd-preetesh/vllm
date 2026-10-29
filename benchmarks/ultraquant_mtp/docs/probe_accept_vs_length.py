#!/usr/bin/env python3
"""Spec-decode acceptance as a function of context and generated length.

For each context length, sends one greedy request per output length (same
prompt, ignore_eos) and diffs the server's spec-decode counters around each
request, so each row is the acceptance over [0, max_tokens) of that request.
A drop in accept_len as max_tokens grows past ~2048 means acceptance dies
once the sequence leaves the DFlash window; a drop with ctx alone means the
long prompt is the trigger.

Run against an idle server only (counters are global).

Example:
  python probe_accept_vs_length.py --port 8361 \
      --model /huggingface/pretrained_models/Qwen/Qwen3.8-27B \
      --ctx 1024,32768,131072 --gen 512,1536,2560,4096
"""
import argparse
import json
import re
import urllib.request

FILLER = (
    "def handler(event, context):\n    items = event.get('items', [])\n"
    "    return [transform(x) for x in items if x is not None]\n"
)
TASK = (
    "Rewrite every function above with type hints and docstrings, then write "
    "exhaustive unit tests for each. Do not stop early."
)


def metrics(port):
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics").read().decode()
    def total(name):
        return sum(float(m.group(1)) for m in re.finditer(
            rf"^{name}(?:{{[^}}]*}})? ([\d.e+]+)$", text, re.M))
    per_pos = {}
    for m in re.finditer(
            r'^vllm:spec_decode_num_accepted_tokens_per_pos_total{[^}]*position="(\d+)"[^}]*} ([\d.e+]+)$',
            text, re.M):
        per_pos[int(m.group(1))] = per_pos.get(int(m.group(1)), 0.0) + float(m.group(2))
    return dict(acc=total("vllm:spec_decode_num_accepted_tokens_total"),
                drafts=total("vllm:spec_decode_num_drafts_total"),
                hits=total("vllm:prefix_cache_hits_total"),
                queries=total("vllm:prefix_cache_queries_total"),
                per_pos=per_pos)


def complete(port, model, prompt, max_tokens):
    body = dict(model=model, prompt=prompt, max_tokens=max_tokens,
                temperature=0.0, ignore_eos=True)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return out["usage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", default="1024,32768,131072")
    ap.add_argument("--gen", default="512,1536,2560,4096")
    ap.add_argument("--approx-tokens-per-filler", type=int, default=40)
    args = ap.parse_args()
    gens = [int(x) for x in args.gen.split(",")]
    for ctx in [int(x) for x in args.ctx.split(",")]:
        reps = max(1, ctx // args.approx_tokens_per_filler)
        prompt = f"# ctx={ctx}\n" + FILLER * reps + "\n" + TASK
        prev = metrics(args.port)
        for g in gens:
            usage = complete(args.port, args.model, prompt, g)
            cur = metrics(args.port)
            drafts = cur["drafts"] - prev["drafts"]
            acc = cur["acc"] - prev["acc"]
            pos = [round((cur["per_pos"].get(i, 0) - prev["per_pos"].get(i, 0))
                         / max(drafts, 1), 3) for i in sorted(cur["per_pos"])]
            hit = (cur["hits"] - prev["hits"]) / max(cur["queries"] - prev["queries"], 1)
            print(json.dumps(dict(
                ctx_tokens=usage["prompt_tokens"], max_tokens=g,
                accept_len=round(1 + acc / max(drafts, 1), 3), drafts=int(drafts),
                per_pos=pos, prefix_hit=round(hit, 3))), flush=True)
            prev = cur


if __name__ == "__main__":
    main()
