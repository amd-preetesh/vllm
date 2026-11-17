# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV non-causal DSpark draft decode against a dense reference."""

import pytest
import torch

from vllm.v1.attention.ops.aiter_noncausal_decode import noncausal_decode_attention


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Each query row attends to keys ``[0, seqused_k)`` of its request."""
    out = torch.zeros_like(q)
    block_size = k.shape[1]
    n_kv = k.shape[2]
    for i, seq_len in enumerate(seqused_k.tolist()):
        if seq_len <= 0:
            continue
        q_rows = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]
        n_blocks = (seq_len + block_size - 1) // block_size
        slots = block_table[i, :n_blocks].tolist()
        pages_k = k[slots].reshape(-1, n_kv, k.shape[-1])[:seq_len]
        pages_v = v[slots].reshape(-1, n_kv, v.shape[-1])[:seq_len]
        # GQA: one KV head is shared by q_heads / kv_heads query heads.
        group = q_rows.shape[1] // n_kv
        k_rep = pages_k.repeat_interleave(group, dim=1).float()
        v_rep = pages_v.repeat_interleave(group, dim=1).float()
        scores = torch.einsum("qhd,khd->qhk", q_rows.float(), k_rep) * scale
        weights = torch.softmax(scores, dim=-1)
        out[cu_seqlens_q[i] : cu_seqlens_q[i + 1]] = torch.einsum(
            "qhk,khd->qhd", weights, v_rep
        ).to(q.dtype)
    return out


def _cosine(got: torch.Tensor, ref: torch.Tensor) -> float:
    a = got.float().reshape(-1)
    b = ref.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@pytest.mark.parametrize("query_lens", [[8, 4], [7]])
def test_noncausal_split_matches_reference(query_lens: list[int]):
    if not torch.cuda.is_available():
        pytest.skip("ROCm device required")
    torch.manual_seed(0)
    device = "cuda"
    q_heads, kv_heads, head_size, block_size = 8, 2, 128, 64
    seq_lens = [96, 160][: len(query_lens)]
    cu = torch.tensor(
        [0, *torch.cumsum(torch.tensor(query_lens), 0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    n_tokens = int(cu[-1])
    q = torch.randn(n_tokens, q_heads, head_size, device=device, dtype=torch.bfloat16)
    n_blocks = 8
    k = torch.randn(
        n_blocks, block_size, kv_heads, head_size, device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    block_table = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]], device=device, dtype=torch.int32
    )[: len(query_lens)]
    seqused = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    scale = head_size**-0.5
    out = torch.empty_like(q)
    noncausal_decode_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu,
        max_seqlen_q=max(query_lens),
        seqused_k=seqused,
        block_table=block_table,
        softmax_scale=scale,
        num_segments=4,
    )
    ref = _reference(q, k, v, cu, seqused, block_table, scale)
    for i, qlen in enumerate(query_lens):
        rows = out[cu[i] : cu[i + 1]]
        expect = ref[cu[i] : cu[i + 1]]
        for j in range(qlen):
            cos = _cosine(rows[j], expect[j])
            assert cos > 0.995, (query_lens, i, j, cos)


def test_padded_query_rows_are_left_unwritten():
    """CUDA-graph padding past query_start_loc must not be scored."""
    if not torch.cuda.is_available():
        pytest.skip("ROCm device required")
    torch.manual_seed(1)
    device = "cuda"
    q_heads, kv_heads, head_size, block_size = 4, 1, 64, 16
    real = 4
    padded = 8
    q = torch.randn(padded, q_heads, head_size, device=device, dtype=torch.bfloat16)
    k = torch.randn(
        2, block_size, kv_heads, head_size, device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    cu = torch.tensor([0, real], device=device, dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    seqused = torch.tensor([20], device=device, dtype=torch.int32)
    out = torch.full_like(q, 7)
    noncausal_decode_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu,
        max_seqlen_q=real,
        seqused_k=seqused,
        block_table=block_table,
        softmax_scale=head_size**-0.5,
        num_segments=2,
    )
    assert torch.all(out[real:] == 7)
    ref = _reference(q[:real], k, v, cu, seqused, block_table, head_size**-0.5)
    assert _cosine(out[:real], ref) > 0.995
