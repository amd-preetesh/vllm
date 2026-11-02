#!/usr/bin/env python3
"""S0 smoke: bf16 draft Q attends UltraQuant paged KV (Idea A kernel path).

Proves the CrossAttn(Q, K_uq, V_uq) primitive vs dequant+SDPA reference.
No draft weights, no server. Writes out/uq_draft_uqkv_smoke/summary.json.

  HIP_VISIBLE_DEVICES=0 python benchmarks/ultraquant_mtp/bench_draft_cross_uq_kv_smoke.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    _SEGM_POOL,
    flydsl_ultraquant_decode_attention,
    is_ultraquant_flydsl_available,
)
from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.reference import hadamard_matrix
from vllm.v1.attention.ops.ultraquant.triton_dequant import ultraquant_full_dequant_kv
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store
from vllm.v1.attention.ops.ultraquant.triton_unified_attention import (
    ultraquant_unified_attention,
)


def _zero_segm() -> None:
    for bufs in _SEGM_POOL._bufs.values():
        bufs["segm_out"].zero_()
        bufs["segm_max"].fill_(float("-inf"))
        bufs["segm_sum"].zero_()


def _reference_cross_attn(
    query: torch.Tensor,  # [qlen, Hq, D] rotated
    k_full: torch.Tensor,  # [T, Hk, D] bf16 rotated
    v_full: torch.Tensor,  # [T, Hk, D]
    scale: float,
) -> torch.Tensor:
    """GliDe-style: every draft query attends the full target prefix (no causal)."""
    q = query.float()
    k = k_full.float()
    v = v_full.float()
    Hq = q.shape[1]
    Hk = k.shape[1]
    if Hq > Hk:
        rep = Hq // Hk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("qhd,thd->hqt", q, k) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqt,thd->qhd", probs, v)
    return out.to(query.dtype)


def run_case(
    *,
    seq_len: int,
    qlen: int,
    warmup: int,
    iters: int,
    use_flydsl: bool,
) -> dict:
    set_random_seed(0xA11A + seq_len + qlen)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hk, hq, head = 4, 24, 256  # Qwen3-ish GQA ratio 6
    block = 64
    scale = head**-0.5
    hadamard = hadamard_matrix(head, device)

    key = torch.randn(seq_len, hk, head, dtype=dtype, device=device)
    value = torch.randn_like(key)
    query = torch.randn(qlen, hq, head, dtype=dtype, device=device)

    blocks = (seq_len + block - 1) // block
    cache = torch.zeros(
        blocks, block, hk, slot_size(head), dtype=torch.uint8, device=device
    )
    table = torch.arange(blocks, dtype=torch.int32, device=device).view(1, blocks)
    slots = torch.arange(blocks * block, dtype=torch.int64, device=device)[:seq_len]
    ultraquant_store(key, value, cache, slots, PiT=hadamard)

    # Reference: dequant + explicit causal-at-end attention in rotated space.
    # Store rotates K; match by rotating Q the same way.
    alloc = blocks * block
    k_buf = torch.empty(1, hk, alloc, head, dtype=dtype, device=device)
    v_buf = torch.empty_like(k_buf)
    ultraquant_full_dequant_kv(cache, table, k_buf, v_buf, alloc)
    k_full = k_buf[0, :, :seq_len, :].transpose(0, 1).contiguous()  # [T,Hk,D]
    v_full = v_buf[0, :, :seq_len, :].transpose(0, 1).contiguous()
    q_rot = (query.float() @ hadamard).to(dtype)
    ref = _reference_cross_attn(q_rot, k_full, v_full, scale)

    # All draft queries see the full target prefix (cross-attn).
    seq_lens = torch.full((qlen,), seq_len, dtype=torch.int32, device=device)
    qsl = torch.arange(qlen + 1, dtype=torch.int32, device=device)
    bt = table.expand(qlen, -1).contiguous()

    def _uq_attn() -> torch.Tensor:
        # PDV FlyDSL only accepts qlen in [2, 8]; q=1 falls back to unified.
        use_pdv = (
            use_flydsl
            and is_ultraquant_flydsl_available()
            and hq == hk * 6
            and head == 256
            and 2 <= qlen <= 8
        )
        if use_pdv:
            _zero_segm()
            # Shared seq_len + multi-query = draft cross-attn over one UQ prefix.
            return flydsl_ultraquant_decode_attention(
                query=query,
                kv_cache=cache,
                block_table=table,
                seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
                scale=scale,
                PiT=hadamard,
                max_seq_len=seq_len,
                shared_queries=qlen,
                use_shared_kernel=True,
                two_wave_pair=False,
                rowpack_verify=False,
                pdv_verify=True,
                query_start_loc=torch.tensor(
                    [0, qlen], dtype=torch.int32, device=device
                ),
                pair_num_reqs=1,
                pairs_per_req=1,
            )
        return ultraquant_unified_attention(
            query=query,
            kv_cache=cache,
            block_table=bt,
            seq_lens=seq_lens,
            query_start_loc=qsl,
            scale=scale,
            PiT=hadamard,
            max_query_len=1,
            max_seq_len=seq_len,
        )

    out = _uq_attn()
    torch.cuda.synchronize()
    # FlyDSL/unified expect unrotated Q and rotate inside; ref used q_rot.
    # Compare in output space: both should match target attention.
    err = (out.float() - ref.float()).abs()
    cosine = F.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()

    for _ in range(warmup):
        _uq_attn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _uq_attn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / max(1, iters)

    ok = cosine > 0.99 and err.mean().item() < 0.05
    path = "flydsl_pdv" if (
        use_flydsl and is_ultraquant_flydsl_available() and hq == hk * 6
    ) else "unified"
    row = {
        "seq_len": seq_len,
        "qlen": qlen,
        "path": path,
        "cosine": cosine,
        "err_mean": err.mean().item(),
        "err_max": err.max().item(),
        "ms": ms,
        "ok": ok,
    }
    print(
        f"T={seq_len} q={qlen} path={path} cosine={cosine:.6f} "
        f"mean={err.mean().item():.4f} max={err.max().item():.4f} "
        f"ms={ms:.2f} ok={ok}",
        flush=True,
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="4096,16384,32768")
    ap.add_argument("--qlens", default="1,4,8")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--no-flydsl", action="store_true")
    ap.add_argument(
        "--out",
        default="/workspace/vllm/out/uq_draft_uqkv_smoke/summary.json",
    )
    args = ap.parse_args()
    ctxs = [int(x) for x in args.ctx.split(",") if x.strip()]
    qlens = [int(x) for x in args.qlens.split(",") if x.strip()]
    use_flydsl = not args.no_flydsl

    rows: list[dict] = []
    failed = False
    for t in ctxs:
        for q in qlens:
            try:
                rows.append(
                    run_case(
                        seq_len=t,
                        qlen=q,
                        warmup=args.warmup,
                        iters=args.iters,
                        use_flydsl=use_flydsl,
                    )
                )
            except Exception as e:  # noqa: BLE001
                failed = True
                print(f"FAIL T={t} q={q}: {type(e).__name__}: {e}", flush=True)
                rows.append(
                    {"seq_len": t, "qlen": q, "ok": False, "error": str(e)}
                )
            torch.cuda.empty_cache()

    if any(not r.get("ok", False) for r in rows):
        failed = True
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "idea": "draft_cross_uq_kv_S0",
                "flydsl_available": is_ultraquant_flydsl_available(),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
