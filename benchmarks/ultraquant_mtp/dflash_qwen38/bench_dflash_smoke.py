#!/usr/bin/env python3
"""Minimal DFlash smoke: one short chat completion against a live server."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8360)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Say OK in one word."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    text = data["choices"][0]["message"].get("content") or ""
    print("smoke_ok", text[:80].replace("\n", " "), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
