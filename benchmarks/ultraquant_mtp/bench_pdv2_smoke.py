"""PDV2 smoke: compile + run packed verify; loose match vs shipping PDV.

FP4 Q (and, for pdv2_fp4pv, E4M3 P/V in PV) is intentionally coarser than
shipping, so we only require a live finite output and a reasonable cosine
against shipping PDV. Experiments that change only PV are also compared with
pdv2 (same FP4 QK) to isolate the PV error.

    VLLM_ULTRAQUANT_PDV_EXPERIMENT=pdv2_fp4pv \
        python benchmarks/ultraquant_mtp/bench_pdv2_smoke.py
"""

from __future__ import annotations

import os

import torch

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    _SEGM_POOL,
    flydsl_ultraquant_decode_attention,
)
from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.reference import hadamard_matrix
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store

_NEED = {
    "pdv2_ctrl": 0.999,
    "pdv2_fp4pv": 0.95,
    "fp4pv": 0.95,
    "pdv2_tr": 0.95,
}
# Experiments whose QK is pdv2's: also report cosine against pdv2.
_PV_ONLY = {"pdv2_fp4pv", "fp4pv", "pdv2_tr"}
_KNOWN = {
    "pdv2",
    "pdv2f",
    "pdv2g",
    "pdv2p",
    "pdv2_ctrl",
    "pdv2_fp4pv",
    "fp4pv",
    "pdv2_tr",
    "pdv2_iglp",
    "pdv2_32x32",
}


def _zero_segm():
    for bufs in _SEGM_POOL._bufs.values():
        bufs["segm_out"].zero_()
        bufs["segm_max"].fill_(float("-inf"))
        bufs["segm_sum"].zero_()


def _pdv(query, cache, table, seq_len, scale, hadamard, num_queries, exp: str):
    os.environ["VLLM_ULTRAQUANT_PDV_EXPERIMENT"] = exp
    _zero_segm()
    return flydsl_ultraquant_decode_attention(
        query=query,
        kv_cache=cache,
        block_table=table,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=query.device),
        scale=scale,
        PiT=hadamard,
        max_seq_len=seq_len,
        shared_queries=num_queries,
        use_shared_kernel=True,
        two_wave_pair=False,
        rowpack_verify=False,
        pdv_verify=True,
        query_start_loc=torch.tensor(
            [0, num_queries], dtype=torch.int32, device=query.device
        ),
        pair_num_reqs=1,
        pairs_per_req=1,
    ).clone()


def _cos(a, b) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _case(exp: str, seq_len: int, v_spread: bool) -> float:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hk, head, qg = 1, 256, 6
    block = 64
    num_queries = 6
    scale = head**-0.5
    hadamard = hadamard_matrix(head, device)
    key = torch.randn(1, seq_len, hk, head, dtype=dtype, device=device)
    value = torch.randn_like(key)
    if v_spread:
        # Per-token V magnitudes over ~2^±6 so UE8M0 V scales differ along
        # the PV contraction (stresses the per-group V reference).
        mag = torch.exp2(torch.randint(-6, 7, (1, seq_len, hk, 1), device=device))
        value = value * mag.to(dtype)
    query = torch.randn(num_queries, hk * qg, head, dtype=dtype, device=device)
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

    ship = _pdv(query, cache, table, seq_len, scale, hadamard, num_queries, "")
    out = _pdv(query, cache, table, seq_len, scale, hadamard, num_queries, exp)
    torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        raise SystemExit(f"{exp} produced non-finite values (seq_len={seq_len})")
    cosine = _cos(out, ship)
    err = (out.float() - ship.float()).abs()
    msg = (
        f"{exp} seq_len={seq_len} v_spread={int(v_spread)} vs shipping "
        f"cosine={cosine:.6f} mean={err.mean().item():.5f} "
        f"max={err.max().item():.5f}"
    )
    if exp in _PV_ONLY:
        ref = _pdv(query, cache, table, seq_len, scale, hadamard, num_queries, "pdv2")
        torch.cuda.synchronize()
        msg += f" | vs pdv2 cosine={_cos(out, ref):.6f}"
    print(msg, flush=True)
    return cosine


def main() -> None:
    exp = os.environ.get("VLLM_ULTRAQUANT_PDV_EXPERIMENT", "").strip()
    assert exp in _KNOWN, exp
    set_random_seed(0x0D02)
    # FP4 Q vs E4M3 Q: expect some gap; fail only on collapse.
    need = _NEED.get(exp, 0.90)
    worst = 1.0
    # 512: the original smoke. 1000: partial last tile. 3000: several
    # partitions with a partial tile pair.
    for seq_len, v_spread in ((512, False), (1000, False), (3000, True)):
        worst = min(worst, _case(exp, seq_len, v_spread))
    if worst < need:
        raise SystemExit(f"{exp} cosine too low: {worst} < {need}")
    print(f"{exp} smoke ok (worst cosine {worst:.6f}, need {need})", flush=True)


if __name__ == "__main__":
    main()
