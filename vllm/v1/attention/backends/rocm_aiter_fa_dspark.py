# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft attention beside an UltraQuant target.

UltraQuant pages are LBNHC. Shipping ``ROCM_AITER_FA`` is LBHNC/LHBNC, so the
two backends have no layout in common and the draft cannot be allocated next
to the target. This backend accepts LBNHC and reads a StreamingLLM span: the
first 4 tokens plus the last 2048. That is the DSpark draft path. Target
verification stays on UltraQuant and still sees the full context.

Selected by default when the target KV dtype is ``ultraquant_4bit`` and the
speculative config does not name a draft attention backend.
"""

from dataclasses import replace

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionBackend,
    AiterFlashAttentionImpl,
    AiterFlashAttentionMetadata,
)
from vllm.v1.attention.ops.dspark_draft_window import build_draft_window_sink_tables
from vllm.v1.kv_cache_interface import KVCacheLayout

logger = init_logger(__name__)

# Measured DSpark draft span: a 4-token sink plus the last 2048 tokens.
DSPARK_DRAFT_SINK = 4
DSPARK_DRAFT_WINDOW = 2048


def _window_decode_metadata(
    attn_metadata: AiterFlashAttentionMetadata,
    num_decodes: int,
    page_win: torch.Tensor,
    seq_win: torch.Tensor,
) -> AiterFlashAttentionMetadata:
    """Point decode rows at the packed window. Prefill rows stay full-context."""
    if num_decodes == attn_metadata.block_table.shape[0]:
        return replace(attn_metadata, block_table=page_win, seq_lens=seq_win)

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
        if page_win.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return attn_metadata
        src_seq = src_seq.to(device=page_win.device)
    new_seq = src_seq.clone()
    new_seq[:num_decodes] = seq_win
    return replace(attn_metadata, block_table=new_bt, seq_lens=new_seq)


class AiterFlashAttentionDSparkImpl(AiterFlashAttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_decodes = int(getattr(attn_metadata, "num_decodes", 0) or 0)
        if attn_metadata is not None and num_decodes > 0:
            # kv_cache: [num_blocks, 2, block_size, ...]
            block_size = int(kv_cache.shape[2])
            page_win, seq_win = build_draft_window_sink_tables(
                attn_metadata.block_table[:num_decodes],
                attn_metadata.seq_lens[:num_decodes],
                block_size=block_size,
                window=DSPARK_DRAFT_WINDOW,
                sink=DSPARK_DRAFT_SINK,
            )
            logger.info_once(
                "DSpark draft attention is AITER FA with an UltraQuant "
                "layout: a %d-token sink plus the last %d tokens. Target "
                "verification is unchanged.",
                DSPARK_DRAFT_SINK,
                DSPARK_DRAFT_WINDOW,
            )
            attn_metadata = _window_decode_metadata(
                attn_metadata, num_decodes, page_win, seq_win
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


class AiterFlashAttentionDSparkBackend(AiterFlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_FA_DSPARK"

    @staticmethod
    def get_impl_cls() -> type[AiterFlashAttentionDSparkImpl]:
        return AiterFlashAttentionDSparkImpl

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # LBNHC first so the draft allocates next to an UltraQuant target.
        return (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC, KVCacheLayout.LHBNC)

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        del default_block_size
        return 64

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 64 == 0

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        del kv_cache_spec
        # 64 matches the UltraQuant page. Offering 16 lets the hybrid manager
        # settle on a 16-token block and breaks the UltraQuant kernel stride.
        return [64]
