#!/usr/bin/env python3
"""Standalone aiter H1 repro: null page-table entries + NaN in block 0.

Calls ``flash_attn_with_kvcache`` with ``window_size=(2047, 0)`` on a page
table whose first N entries are 0 and whose block 0 V contains NaN. If the
output has NaNs, H1's aiter-side mechanism is confirmed without a vLLM server.

  HIP_VISIBLE_DEVICES=0 python benchmarks/ultraquant_mtp/aiter_null_block_nan_repro.py
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-blocks", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--null-prefix", type=int, default=16)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=2048)
    args = ap.parse_args()

    import torch

    try:
        from aiter.ops.triton.flash_attn_triton_amd import mha_v3
    except Exception:
        try:
            from aiter import flash_attn_with_kvcache as _fa

            mha_v3 = None
        except Exception as e:
            print(f"FAIL import aiter flash_attn: {e}", file=sys.stderr)
            return 2

    device = "cuda"
    B, H, D = 1, args.heads, args.head_dim
    bs = args.block_size
    nblk = args.num_blocks
    # [num_blocks, 2, H, block_size, D] common paged layout
    k_cache = torch.randn(nblk, H, bs, D, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(nblk, H, bs, D, device=device, dtype=torch.bfloat16)
    v_cache[0].fill_(float("nan"))

    # Page table: first null_prefix entries are 0 (null), then 1..
    n_pages = (args.seqlen + bs - 1) // bs
    page_table = torch.zeros(B, n_pages, device=device, dtype=torch.int32)
    for i in range(n_pages):
        if i < args.null_prefix:
            page_table[0, i] = 0
        else:
            page_table[0, i] = min(i, nblk - 1)
    cache_seqlens = torch.tensor([args.seqlen], device=device, dtype=torch.int32)
    q = torch.randn(B, 8, H, D, device=device, dtype=torch.bfloat16)
    window = (args.window - 1, 0)

    print(
        f"repro null_prefix={args.null_prefix} seqlen={args.seqlen} "
        f"window={window} block0_v_nan={bool(torch.isnan(v_cache[0]).any())}"
    )

    try:
        if mha_v3 is not None:
            out = mha_v3.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                causal=False,
                window_size=window,
            )
        else:
            out = _fa(
                q,
                k_cache,
                v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                causal=False,
                window_size=window,
            )
    except TypeError as e:
        # Signature drift — try kwargs subset.
        print(f"WARN signature retry: {e}")
        out = mha_v3.flash_attn_with_kvcache(
            q, k_cache, v_cache, page_table, cache_seqlens, None, None, False, window
        )

    if isinstance(out, (tuple, list)):
        out = out[0]
    nan_count = int(torch.isnan(out).sum().item())
    inf_count = int(torch.isinf(out).sum().item())
    print(f"out_shape={tuple(out.shape)} nan={nan_count} inf={inf_count}")
    if nan_count > 0 or inf_count > 0:
        print("RESULT: non-finite output — H1 aiter path CONFIRMED")
        return 0
    print("RESULT: finite output — H1 aiter path NOT observed with this call")
    return 1


if __name__ == "__main__":
    # Ensure aiter on path when run from host/container.
    root = os.environ.get("VLLM_ROOT", "/workspace/vllm")
    aiter = os.environ.get("AITER_SRC", "/workspace/aiter")
    sys.path[:0] = [aiter, root]
    raise SystemExit(main())
