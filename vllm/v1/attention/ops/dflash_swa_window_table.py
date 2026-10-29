# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P0-3: build a fixed-width SWA window page table for DFlash draft FA.

Avoids scanning the full context (and null block 0 at the head of the page
table) by slicing each request's block table to the sliding window only.

``start_blk = max(0, (S - qlen - window_left) // B)``
``cache_seqlens_win = S - start_blk * B``
``win_blocks = ceil((window_left + qlen) / B) + 1`` (fixed, CUDA-graph safe)

All tensor work stays on-device — no ``.cpu()`` / ``.item()`` — so this is
safe under CUDA-graph capture (DFlash uses FULL graphs for UNIFORM_BATCH).

Used by ``ROCM_AITER_FA_DFLASH_SWA`` / ``VLLM_DFLASH_DRAFT_ATTN_EXPERIMENT=win_table``.
"""

from __future__ import annotations

import math

import torch


def window_num_blocks(window_left: int, max_query_len: int, block_size: int) -> int:
    """Fixed column count for the window page table."""
    span = max(1, int(window_left) + int(max_query_len))
    return math.ceil(span / int(block_size)) + 1


def build_dflash_swa_window_tables(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int,
    window_left: int,
    max_query_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(page_table_win, cache_seqlens_win)`` for decode requests.

    Args:
        block_table: ``[num_reqs, max_blocks]`` int page indices (may include
            leading null / PAD slots in the full-context table).
        seq_lens: ``[num_reqs]`` total context length per request.
        block_size: KV page size in tokens.
        window_left: ``sliding_window[0]`` (= W-1 for DFlash).
        max_query_len: max query rows in this decode batch (1+K for DFlash).
    """
    if window_left < 0:
        return block_table, seq_lens

    device = block_table.device
    dtype = block_table.dtype
    # H2D is illegal under CUDA graph capture. Metadata should already be on-device;
    # if not, leave the full page table unchanged rather than crashing capture.
    if seq_lens.device != device:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return block_table, seq_lens
        seq_lens = seq_lens.to(device=device)
    num_reqs = seq_lens.shape[0]
    max_blocks = block_table.shape[1]
    win_w = window_num_blocks(window_left, max_query_len, block_size)
    b = int(block_size)
    wl = int(window_left)
    mq = int(max_query_len)
    max_col = max(0, int(max_blocks) - 1)

    # start_blk = max(0, (S - qlen - window_left) // B)  — fully on-device
    start_blk = torch.div(
        seq_lens - mq - wl, b, rounding_mode="floor"
    ).clamp_min(0)

    cache_seqlens_win = seq_lens - start_blk * b

    # cols[i, j] = start_blk[i] + j
    offs = torch.arange(win_w, device=device, dtype=start_blk.dtype)
    cols = start_blk.unsqueeze(1) + offs.unsqueeze(0)
    valid = cols < max_blocks
    cols_clamped = cols.clamp(max=max_col)
    gathered = block_table.gather(1, cols_clamped.to(dtype=torch.long))
    zeros = torch.zeros((num_reqs, win_w), device=device, dtype=dtype)
    page_table_win = torch.where(valid, gathered, zeros)

    return page_table_win, cache_seqlens_win
