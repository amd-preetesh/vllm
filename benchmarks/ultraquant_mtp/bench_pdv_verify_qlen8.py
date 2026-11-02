#!/usr/bin/env python3
"""E6: PDV qlen-6/8 correctness + latency at long UQ contexts (131k–230k).

Compares PDV shared-query FlyDSL against one Prod launch per verify token.
No shipping-kernel edits. Writes a JSON summary under --out.

  HIP_VISIBLE_DEVICES=0 python benchmarks/ultraquant_mtp/bench_pdv_verify_qlen8.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    _SEGM_POOL,
    flydsl_ultraquant_decode_attention,
)
from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.reference import hadamard_matrix
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store


def _zero_segm() -> None:
    for bufs in _SEGM_POOL._bufs.values():
        bufs["segm_out"].zero_()
        bufs["segm_max"].fill_(float("-inf"))
        bufs["segm_sum"].zero_()


def _run_case(
    *,
    num_queries: int,
    seq_len: int,
    warmup: int,
    iters: int,
) -> dict:
    set_random_seed(0xE600 + num_queries + (seq_len % 9973))
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hk, head = 1, 256
    block = 64
    scale = head**-0.5
    hadamard = hadamard_matrix(head, device)
    key = torch.randn(1, seq_len, hk, head, dtype=dtype, device=device)
    value = torch.randn_like(key)
    query = torch.randn(num_queries, hk * 6, head, dtype=dtype, device=device)
    blocks = (seq_len + block - 1) // block
    cache = torch.zeros(
        blocks, block, hk, slot_size(head), dtype=torch.uint8, device=device
    )
    table = torch.arange(blocks, dtype=torch.int32, device=device).view(1, blocks)
    slots = torch.arange(blocks * block, dtype=torch.int64, device=device)[:seq_len]
    ultraquant_store(
        key.reshape(-1, hk, head),
        value.reshape(-1, hk, head),
        cache,
        slots,
        PiT=hadamard,
    )
    token_lens = torch.arange(
        seq_len - num_queries + 1, seq_len + 1, dtype=torch.int32, device=device
    )
    qsl = torch.tensor([0, num_queries], dtype=torch.int32, device=device)
    seq_one = torch.tensor([seq_len], dtype=torch.int32, device=device)
    bt_exp = table.expand(num_queries, -1).contiguous()

    def _prod() -> torch.Tensor:
        _zero_segm()
        return flydsl_ultraquant_decode_attention(
            query=query,
            kv_cache=cache,
            block_table=bt_exp,
            seq_lens=token_lens,
            scale=scale,
            PiT=hadamard,
            max_seq_len=seq_len,
        )

    def _pdv() -> torch.Tensor:
        _zero_segm()
        return flydsl_ultraquant_decode_attention(
            query=query,
            kv_cache=cache,
            block_table=table,
            seq_lens=seq_one,
            scale=scale,
            PiT=hadamard,
            max_seq_len=seq_len,
            shared_queries=num_queries,
            use_shared_kernel=True,
            two_wave_pair=False,
            rowpack_verify=False,
            pdv_verify=True,
            query_start_loc=qsl,
            pair_num_reqs=1,
            pairs_per_req=1,
        )

    prod = _prod().clone()
    pdv = _pdv().clone()
    torch.cuda.synchronize()
    err = (pdv.float() - prod.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(
        pdv.float().flatten(), prod.float().flatten(), dim=0
    ).item()

    for _ in range(warmup):
        _pdv()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _pdv()
    torch.cuda.synchronize()
    pdv_ms = (time.perf_counter() - t0) * 1000.0 / max(1, iters)

    for _ in range(warmup):
        _prod()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _prod()
    torch.cuda.synchronize()
    prod_ms = (time.perf_counter() - t0) * 1000.0 / max(1, iters)

    ok = cosine > 0.999 and err.mean().item() < 0.02 and err.max().item() < 0.2
    row = {
        "qlen": num_queries,
        "seq_len": seq_len,
        "cosine": cosine,
        "err_mean": err.mean().item(),
        "err_max": err.max().item(),
        "pdv_ms": pdv_ms,
        "prod_ms": prod_ms,
        "ok": ok,
    }
    print(
        f"qlen={num_queries} seq={seq_len} cosine={cosine:.6f} "
        f"mean={err.mean().item():.5f} max={err.max().item():.5f} "
        f"pdv_ms={pdv_ms:.2f} prod_ms={prod_ms:.2f} ok={ok}",
        flush=True,
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ctx",
        default="131072,196608,229376",
        help="comma-separated context lengths",
    )
    ap.add_argument("--qlens", default="6,8", help="comma-separated query lengths")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument(
        "--out",
        default="/workspace/vllm/out/uq_e6_pdv_qlen8/summary.json",
    )
    args = ap.parse_args()
    ctxs = [int(x) for x in args.ctx.split(",") if x.strip()]
    qlens = [int(x) for x in args.qlens.split(",") if x.strip()]

    rows: list[dict] = []
    failed = False
    for seq in ctxs:
        for q in qlens:
            try:
                rows.append(
                    _run_case(
                        num_queries=q,
                        seq_len=seq,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                )
            except Exception as e:  # noqa: BLE001 — report and continue matrix
                failed = True
                print(f"FAIL qlen={q} seq={seq}: {type(e).__name__}: {e}", flush=True)
                rows.append(
                    {
                        "qlen": q,
                        "seq_len": seq,
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
            # Free between cases — long ctx caches are huge.
            torch.cuda.empty_cache()

    if any(not r.get("ok", False) for r in rows):
        failed = True
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
