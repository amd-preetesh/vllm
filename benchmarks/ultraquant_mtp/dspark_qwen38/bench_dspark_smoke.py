#!/usr/bin/env python3
"""Minimal chat smoke for Qwen3.8 + DSpark (TP=1 serve)."""
from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Say OK in one word."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read().decode())
    text = resp["choices"][0]["message"].get("content") or ""
    print("smoke_ok", text[:80].replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
