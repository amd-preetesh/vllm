# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PDV packed-verify FlyDSL matches one Prod FlyDSL launch per token."""

from __future__ import annotations

import os

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
    assert cosine > 0.995, tag
    assert err.mean() < 0.05, tag


@pytest.mark.parametrize("num_queries", [2, 5, 6])
@pytest.mark.parametrize(
    "pdv_experiment,tag",
    [
        ("", "pdv1.0"),
        ("hoist_k", "pdv1.1"),
    ],
)
def test_pdv_matches_separate_flydsl(num_queries, pdv_experiment, tag, monkeypatch):
    if not ultraquant_flydsl_decode_eligible(
        head_size=256,
        num_kv_groups=6,
        has_sinks=False,
        sliding_window=None,
        flydsl_loaded=is_ultraquant_flydsl_available(),
    ):
        pytest.skip("UltraQuant D=256 FlyDSL kernel is unavailable")

    if pdv_experiment:
        monkeypatch.setenv("VLLM_ULTRAQUANT_PDV_EXPERIMENT", pdv_experiment)
    else:
        monkeypatch.delenv("VLLM_ULTRAQUANT_PDV_EXPERIMENT", raising=False)

    set_random_seed(0x504456 + num_queries)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_len = 2048
    num_kv_heads = 1
    head_size = 256
    block_size = 64
    query_group_size = 6
    scale = head_size**-0.5
    key = torch.randn(1, seq_len, num_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn_like(key)
    query = torch.randn(
        num_queries, query_group_size, head_size, dtype=dtype, device=device
    )
    kv_cache, block_table = _build_paged_cache(key, value, block_size)
    hadamard = hadamard_matrix(head_size, device)
    seq_lens = torch.arange(
        seq_len - num_queries + 1,
        seq_len + 1,
        dtype=torch.int32,
        device=device,
    )

    separate = []
    for i in range(num_queries):
        separate.append(
            flydsl_ultraquant_decode_attention(
                query=query[i : i + 1],
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens[i : i + 1],
                scale=scale,
                PiT=hadamard,
                max_seq_len=seq_len,
                shared_queries=1,
                use_shared_kernel=False,
            )
        )
    separate = torch.cat(separate, dim=0)

    if pdv_experiment:
        os.environ["VLLM_ULTRAQUANT_PDV_EXPERIMENT"] = pdv_experiment
    else:
        os.environ.pop("VLLM_ULTRAQUANT_PDV_EXPERIMENT", None)
    pdv = flydsl_ultraquant_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        PiT=hadamard,
        max_seq_len=seq_len,
        shared_queries=num_queries,
        use_shared_kernel=True,
        pdv_verify=True,
    )
    _compare(pdv, separate, f"{tag} k={num_queries} seq={seq_len}")
