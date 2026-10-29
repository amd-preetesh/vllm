# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P0-3: DFlash SWA window-table draft FA (experiment).

Hypothesis H1 in ``docs/uq_kernel_bottlenecks.md``: long-context DFlash draft
reads null block 0 (overlaid with Mamba state) because aiter does not
window-mask V loads. This backend builds a positional window page table so
the draft never visits block 0 or out-of-window context.

Gate:
  - speculative_config ``attention_backend: ROCM_AITER_FA_DFLASH_SWA``, or
  - ``VLLM_DFLASH_DRAFT_ATTN_EXPERIMENT=win_table`` (compat dispatcher).

Default remains ``ROCM_AITER_FA_UQ_COMPAT`` / ``ROCM_AITER_FA``.
"""

from __future__ import annotations

import os
from dataclasses import replace

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionImpl
from vllm.v1.attention.backends.rocm_aiter_fa_uq_compat import (
    AiterFlashAttentionUQCompatBackend,
)
from vllm.v1.attention.ops.dflash_swa_window_table import (
    build_dflash_swa_window_tables,
)

logger = init_logger(__name__)
_LOGGED = False


def dflash_win_table_experiment_enabled() -> bool:
    return (
        os.environ.get("VLLM_DFLASH_DRAFT_ATTN_EXPERIMENT", "").strip() == "win_table"
    )


def _as_py_int(x, default: int = 0) -> int:
    """Python int without CUDA→CPU sync (unsafe under graph capture)."""
    if x is None:
        return default
    if isinstance(x, torch.Tensor):
        # Metadata fields used here are host ints; a CUDA 0-d would break capture.
        if x.device.type != "cpu":
            return default
        return int(x.item())
    return int(x)


def _with_window_decode_metadata(attn_metadata, *, block_size: int, window_left: int):
    """Replace decode rows' page table / seq_lens with the SWA window slice."""
    num_decodes = _as_py_int(getattr(attn_metadata, "num_decodes", 0), 0)
    if num_decodes <= 0 or window_left < 0:
        return attn_metadata

    max_q = _as_py_int(getattr(attn_metadata, "max_query_len", 1), 1) or 1
    # Decode-slice max query (spec verify / draft) — prefer decode token count.
    num_decode_tokens = _as_py_int(getattr(attn_metadata, "num_decode_tokens", 0), 0)
    if num_decodes > 0 and num_decode_tokens >= num_decodes:
        max_q = max(max_q, (num_decode_tokens + num_decodes - 1) // num_decodes)

    page_win, seq_win = build_dflash_swa_window_tables(
        attn_metadata.block_table[:num_decodes],
        attn_metadata.seq_lens[:num_decodes],
        block_size=block_size,
        window_left=window_left,
        max_query_len=max_q,
    )

    global _LOGGED
    if not _LOGGED:
        logger.info_once(
            "ROCM_AITER_FA_DFLASH_SWA: window page table enabled "
            "(window_left=%d, win_blocks=%d, block_size=%d).",
            window_left,
            page_win.shape[1],
            block_size,
        )
        _LOGGED = True

    # Common draft path: decode-only batch — swap tables in place (graph-safe).
    if num_decodes == attn_metadata.block_table.shape[0]:
        return replace(attn_metadata, block_table=page_win, seq_lens=seq_win)

    # Mixed batch: window-slice decode rows; keep prefill rows full-width.
    rest = attn_metadata.block_table[num_decodes:]
    win_w = page_win.shape[1]
    rest_w = rest.shape[1] if rest.numel() else 0
    out_w = max(win_w, rest_w)
    new_bt = torch.zeros(
        (attn_metadata.block_table.shape[0], out_w),
        device=page_win.device,
        dtype=page_win.dtype,
    )
    new_bt[:num_decodes, :win_w] = page_win
    if rest_w > 0:
        new_bt[num_decodes:, :rest_w] = rest

    src_seq = attn_metadata.seq_lens
    if src_seq.device != page_win.device:
        # Mixed-batch path is rare for DFlash draft; avoid H2D under capture.
        if page_win.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return attn_metadata
        src_seq = src_seq.to(device=page_win.device)
    new_seq = src_seq.clone()
    new_seq[:num_decodes] = seq_win
    return replace(attn_metadata, block_table=new_bt, seq_lens=new_seq)


class AiterFlashAttentionDFlashSwaImpl(AiterFlashAttentionImpl):
    """Draft FA Impl that window-slices the page table before SWA decode."""

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            attn_metadata is not None
            and self.sliding_window[0] != -1
            and int(getattr(attn_metadata, "num_decodes", 0) or 0) > 0
        ):
            # kv_cache: [num_blocks, 2, block_size, ...]
            block_size = int(kv_cache.shape[2])
            attn_metadata = _with_window_decode_metadata(
                attn_metadata,
                block_size=block_size,
                window_left=int(self.sliding_window[0]),
            )
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )


class AiterFlashAttentionDFlashSwaBackend(AiterFlashAttentionUQCompatBackend):
    """UQ-compat draft FA + SWA window page tables (P0-3 experiment)."""

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_FA_DFLASH_SWA"

    @staticmethod
    def get_impl_cls() -> type[AiterFlashAttentionDFlashSwaImpl]:
        return AiterFlashAttentionDFlashSwaImpl
