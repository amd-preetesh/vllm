# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-KV verify kernel matches one FlyDSL launch per query."""

import sys

import pytest
import torch

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    flydsl_ultraquant_decode_attention,
    is_ultraquant_flydsl_available,
    ultraquant_flydsl_decode_eligible,
)
from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.reference import hadamard_matrix
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)


def _build_paged_cache(key, value, block_size):
    _batch, seq_len, num_kv_heads, head_size = key.shape
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    kv_cache = torch.zeros(
        blocks_per_seq,
        block_size,
        num_kv_heads,
        slot_size(head_size),
        dtype=torch.uint8,
        device=key.device,
    )
    block_table = torch.arange(
        blocks_per_seq, dtype=torch.int32, device=key.device
    ).view(1, blocks_per_seq)
    slot_mapping = torch.arange(
        blocks_per_seq * block_size, dtype=torch.int64, device=key.device
    )[:seq_len]
    ultraquant_store(
        key.reshape(-1, num_kv_heads, head_size),
        value.reshape(-1, num_kv_heads, head_size),
        kv_cache,
        slot_mapping,
    )
    return kv_cache, block_table


def _compare(actual, expected, tag):
    actual_f = actual.float()
    expected_f = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    ).item()
    err = (actual_f - expected_f).abs()
    print(
        f"{tag}: cosine={cosine:.6f} mean={err.mean().item():.5f} "
        f"max={err.max().item():.5f}",
        flush=True,
    )
    assert cosine > 0.999, tag
    assert err.mean() < 0.02, tag
    assert err.max() < 0.2, tag


def _run_case(num_queries, seq_len, query_group_size=6, two_wave_pair=False):
    if not ultraquant_flydsl_decode_eligible(
        head_size=256,
        num_kv_groups=query_group_size,
        has_sinks=False,
        sliding_window=None,
        flydsl_loaded=is_ultraquant_flydsl_available(),
    ):
        pytest.skip("UltraQuant D=256 FlyDSL kernel is unavailable")

    set_random_seed(0x51A2E + num_queries + seq_len)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_kv_heads = 1
    head_size = 256
    block_size = 64
    scale = head_size**-0.5
    key = torch.randn(
        1, seq_len, num_kv_heads, head_size, dtype=dtype, device=device
    )
    value = torch.randn_like(key)
    query = torch.randn(
        num_queries, query_group_size, head_size, dtype=dtype, device=device
    )
    kv_cache, block_table = _build_paged_cache(key, value, block_size)
    hadamard = hadamard_matrix(head_size, device)
    # Verify pattern: query j attends the first (seq_len - k + 1 + j) tokens.
    seq_lens = torch.arange(
        seq_len - num_queries + 1,
        seq_len + 1,
        dtype=torch.int32,
        device=device,
    )
    separate = flydsl_ultraquant_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table.expand(num_queries, -1).contiguous(),
        seq_lens=seq_lens,
        scale=scale,
        PiT=hadamard,
        max_seq_len=seq_len,
    )
    if two_wave_pair:
        qsl = torch.tensor([0, num_queries], dtype=torch.int32, device=device)
        req_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
        shared = flydsl_ultraquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=req_lens,
            scale=scale,
            PiT=hadamard,
            max_seq_len=seq_len,
            shared_queries=2,
            use_shared_kernel=True,
            two_wave_pair=True,
            query_start_loc=qsl,
            pair_num_reqs=1,
            pairs_per_req=1,
        )
    else:
        shared = flydsl_ultraquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=scale,
            PiT=hadamard,
            max_seq_len=seq_len,
            shared_queries=num_queries,
            use_shared_kernel=True,
            two_wave_pair=False,
        )
    _compare(shared, separate, f"k={num_queries} seq={seq_len}")
    return query, kv_cache, block_table, seq_lens, scale, hadamard, separate


@pytest.mark.parametrize("num_queries", [1, 5])
def test_shared_kv_matches_separate_flydsl(num_queries):
    _run_case(num_queries, seq_len=2048)


def test_two_wave_pair_matches_separate_flydsl():
    _run_case(2, seq_len=2048, two_wave_pair=True)


def _bench():
    import time

    num_queries = 5
    seq_len = 8192
    query, kv_cache, block_table, seq_lens, scale, hadamard, _ = _run_case(
        num_queries, seq_len
    )

    def once(shared):
        if shared:
            return flydsl_ultraquant_decode_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                scale=scale,
                PiT=hadamard,
                max_seq_len=seq_len,
                shared_queries=num_queries,
                use_shared_kernel=True,
            )
        return flydsl_ultraquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table.expand(num_queries, -1).contiguous(),
            seq_lens=seq_lens,
            scale=scale,
            PiT=hadamard,
            max_seq_len=seq_len,
        )

    for _ in range(3):
        once(False)
        once(True)
    torch.cuda.synchronize()

    def time_it(shared, iters=10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            once(shared)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1e3

    print(
        f"bench seq={seq_len} k={num_queries}: "
        f"separate {time_it(False):.2f} ms, shared {time_it(True):.2f} ms",
        flush=True,
    )


if __name__ == "__main__":
    try:
        _run_case(1, 2048)
        _run_case(5, 2048)
        _bench()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
