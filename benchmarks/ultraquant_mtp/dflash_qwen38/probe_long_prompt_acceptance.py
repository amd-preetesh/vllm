#!/usr/bin/env python3
"""Bisect long-prompt DFlash acceptance against a live vLLM server.

For each prompt length, send one temperature-0 completion and report:
  - accepted / drafted tokens from /metrics (spec decode counters)
  - output text preview
  - optional second request to exercise prefix-cache hits

Expects the server already running (use the agentx_debug launcher).
Does not edit shipping code.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _get(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def _post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _parse_spec_metrics(text: str) -> dict[str, float]:
    """Pull cumulative speculative-decoding counters from Prometheus text."""
    wanted = {
        "vllm:spec_decode_num_accepted_tokens_total": 0.0,
        "vllm:spec_decode_num_draft_tokens_total": 0.0,
    }
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, rest = line.partition(" ")
        base = name.split("{", 1)[0]
        if base in wanted:
            try:
                wanted[base] = float(rest.strip().split()[0])
            except ValueError:
                pass
    return wanted


def _synthetic_prompt(n_tokens: int, filler: str = "word") -> str:
    # Rough: one whitespace-separated token ≈ one tokenizer piece for ASCII.
    return (" ".join([filler] * max(n_tokens - 8, 1)) + " Reply with OK.").strip()


def _chat(
    port: int,
    model: str,
    content: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return _post_json(url, body, timeout=timeout)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8361)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--lengths",
        default="4096,16384,49152,131072,200000",
        help="Comma-separated prompt token lengths to probe",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument(
        "--repeat-for-prefix-hit",
        action="store_true",
        help="Send each prompt twice; second request should hit prefix cache",
    )
    p.add_argument(
        "--prompt-file",
        default="",
        help="Optional UTF-8 file; truncated/padded per length instead of synthetic",
    )
    p.add_argument(
        "--distinct-prefix",
        default="",
        help="If set, prepend '<value><len>' so prompts share no prefix",
    )
    args = p.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    base_text = ""
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            base_text = f.read()

    metrics_url = f"http://127.0.0.1:{args.port}/metrics"
    try:
        _get(f"http://127.0.0.1:{args.port}/v1/models", timeout=10)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL server not reachable on :{args.port}: {e}", flush=True)
        return 1

    print(
        f"probe start port={args.port} lengths={lengths} "
        f"max_tokens={args.max_tokens} prefix_hit={args.repeat_for_prefix_hit}",
        flush=True,
    )

    for n in lengths:
        if base_text:
            # Character approx; fine for relative length sweeps.
            content = (base_text * ((n // max(len(base_text.split()), 1)) + 1))[
                : max(n * 5, 32)
            ]
            content = content + "\nReply with OK."
        elif args.distinct_prefix:
            # Different leading token per length so no request can prefix-hit
            # another one; isolates "first real prefix hit" from length/pollution.
            content = f"{args.distinct_prefix}{n} " + _synthetic_prompt(n)
        else:
            content = _synthetic_prompt(n)

        passes = 2 if args.repeat_for_prefix_hit else 1
        for pass_i in range(passes):
            label = f"len={n} pass={pass_i + 1}/{passes}"
            before = _parse_spec_metrics(_get(metrics_url))
            t0 = time.time()
            try:
                data = _chat(
                    args.port, args.model, content, args.max_tokens, args.timeout
                )
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:500]
                print(f"{label} HTTP {e.code}: {body}", flush=True)
                continue
            except Exception as e:  # noqa: BLE001
                print(f"{label} ERROR {type(e).__name__}: {e}", flush=True)
                continue
            dt = time.time() - t0
            after = _parse_spec_metrics(_get(metrics_url))
            d_acc = (
                after["vllm:spec_decode_num_accepted_tokens_total"]
                - before["vllm:spec_decode_num_accepted_tokens_total"]
            )
            d_draft = (
                after["vllm:spec_decode_num_draft_tokens_total"]
                - before["vllm:spec_decode_num_draft_tokens_total"]
            )
            rate = (d_acc / d_draft) if d_draft > 0 else float("nan")
            msg = data.get("choices", [{}])[0].get("message", {})
            text = (msg.get("content") or msg.get("reasoning_content") or "")[:120]
            usage = data.get("usage") or {}
            print(
                f"{label} dt={dt:.1f}s "
                f"accepted={d_acc:.0f} drafted={d_draft:.0f} rate={rate:.3f} "
                f"prompt_tokens={usage.get('prompt_tokens')} "
                f"completion_tokens={usage.get('completion_tokens')} "
                f"text={text!r}",
                flush=True,
            )

    print("probe done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
