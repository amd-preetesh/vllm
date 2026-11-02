# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P1-1: continuation-prefill v2 helpers (env-gated).

Gate: ``VLLM_ULTRAQUANT_CONT_PREFILL=v2`` (default unset = shipping path).

Changes vs shipping ``_ultraquant_continuation_prefill``:
  1. Persistent ``cu_seqlens`` buffers (no per-call ``torch.tensor(..., device=)``).
  2. Context-aware decode-vs-dequant threshold (short q at long L → dequant path).
  3. Dequant into workspace then a single transpose into the FA concat buffer
     (same workspace manager; no new Triton kernel in this pass).

Selected only from a thin branch in ``UltraQuantAttentionImpl``.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from vllm.v1.worker.workspace import current_workspace_manager

_CK_MAX_HEAD_DIM = 256

# Module-level persistent cu buffers keyed by device string.
_CU_Q: dict[str, torch.Tensor] = {}
_CU_K: dict[str, torch.Tensor] = {}
_SEQUSED_K: dict[str, torch.Tensor] = {}
_BLOCK_TABLE_1: dict[str, torch.Tensor] = {}
_LOGGED = False


def cont_prefill_v2_enabled() -> bool:
    return os.environ.get("VLLM_ULTRAQUANT_CONT_PREFILL", "").strip() == "v2"


def continuation_decode_threshold(cached_len: int, q_len: int) -> int:
    """Prefer dequant+FA once the unified per-row scan would dominate.

    Shipping uses a fixed 128. At long L the O(L) per-row scan is worse even
    for modest q; crossover is roughly 8–16 rows at L≈100k.
    """
    if cached_len < 8192:
        return 128
    if cached_len < 32768:
        return 32
    if cached_len < 100_000:
        return 16
    return 8


def _dev_key(device: torch.device) -> str:
    return str(device)


def _cu2(device: torch.device, *, q_len: int, seq_len: int | None = None) -> tuple:
    """Return (cu_q, cu_k_or_None, seqused_k_or_None) with persistent buffers."""
    key = _dev_key(device)
    if key not in _CU_Q:
        _CU_Q[key] = torch.zeros(2, device=device, dtype=torch.int32)
        _CU_K[key] = torch.zeros(2, device=device, dtype=torch.int32)
        _SEQUSED_K[key] = torch.zeros(1, device=device, dtype=torch.int32)
        _BLOCK_TABLE_1[key] = torch.zeros((1, 1), dtype=torch.int32, device=device)
    cu_q = _CU_Q[key]
    cu_q[1] = q_len
    if seq_len is None:
        return cu_q, None, None, _BLOCK_TABLE_1[key]
    cu_k = _CU_K[key]
    cu_k[1] = seq_len
    seqused = _SEQUSED_K[key]
    seqused[0] = seq_len
    return cu_q, cu_k, seqused, _BLOCK_TABLE_1[key]


def continuation_prefill_v2(
    *,
    impl: Any,
    layer: Any,
    query: torch.Tensor,
    key_chunk: torch.Tensor,
    val_chunk: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cached_len: int,
    seq_len: int,
    PiT: torch.Tensor,
    has_flash_attn: bool,
) -> torch.Tensor:
    """Large continuation chunk with persistent cu buffers (see module doc)."""
    from vllm.v1.attention.ops.ultraquant.triton_dequant import (
        ultraquant_full_dequant_kv,
    )

    global _LOGGED
    if not _LOGGED:
        from vllm.logger import init_logger

        init_logger(__name__).info_once(
            "VLLM_ULTRAQUANT_CONT_PREFILL=v2: persistent cu_seqlens + "
            "context-aware threshold active."
        )
        _LOGGED = True

    q_len, Hq, D = query.shape
    Hk = key_chunk.shape[1]
    device = query.device
    qdtype = query.dtype
    block_size = kv_cache.shape[1]
    alloc_len = math.ceil(cached_len / block_size) * block_size

    buf_shape = (1, Hk, alloc_len, D)
    full_cap = math.ceil(seq_len / block_size) * block_size
    full_shape = (full_cap, Hk, D)
    k_buf, v_buf, k_full_buf, v_full_buf = current_workspace_manager().get_simultaneous(
        (buf_shape, qdtype),
        (buf_shape, qdtype),
        (full_shape, qdtype),
        (full_shape, qdtype),
    )
    ultraquant_full_dequant_kv(
        kv_cache=kv_cache,
        block_table=block_table,
        k_out=k_buf,
        v_out=v_buf,
        alloc_len=alloc_len,
    )

    k_full = k_full_buf[:seq_len]
    v_full = v_full_buf[:seq_len]
    # Single transpose copy into concat buffer (avoid per-token Python loops).
    k_full[:cached_len].copy_(k_buf[0, :, :cached_len, :].transpose(0, 1))
    v_full[:cached_len].copy_(v_buf[0, :, :cached_len, :].transpose(0, 1))
    k_full[cached_len:] = (key_chunk.to(torch.float32) @ PiT).to(qdtype)
    v_full[cached_len:] = val_chunk
    q_rot = (query.to(torch.float32) @ PiT).to(qdtype)

    cu_q, cu_k, seqused_k, block_table_single = _cu2(
        device, q_len=q_len, seq_len=seq_len
    )

    if impl.sinks is not None or D > _CK_MAX_HEAD_DIM:
        from vllm.v1.attention.ops.triton_unified_attention import (
            unified_attention,
        )

        out = torch.empty_like(q_rot)
        window = (
            (impl.sliding_window - 1, 0)
            if impl.sliding_window and impl.sliding_window > 0
            else (-1, -1)
        )
        unified_attention(
            q=q_rot,
            k=k_full.unsqueeze(0),
            v=v_full.unsqueeze(0),
            out=out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=q_len,
            seqused_k=seqused_k,
            max_seqlen_k=seq_len,
            softmax_scale=impl.scale,
            causal=True,
            window_size=window,
            block_table=block_table_single,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            sinks=impl.sinks,
        )
        return out.to(qdtype)

    if has_flash_attn:
        return impl._flash_attn_varlen(
            q=q_rot,
            k=k_full,
            v=v_full,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q_len,
            max_seqlen_k=seq_len,
        )

    q_t = q_rot.transpose(0, 1).unsqueeze(0)
    k_t = k_full.transpose(0, 1).unsqueeze(0)
    v_t = v_full.transpose(0, 1).unsqueeze(0)
    if Hq > Hk:
        k_t = k_t.expand(1, Hq, -1, -1)
        v_t = v_t.expand(1, Hq, -1, -1)
    return (
        F.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=True, scale=impl.scale
        )
        .squeeze(0)
        .transpose(0, 1)
        .to(qdtype)
    )
