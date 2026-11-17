# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_FA for a DSpark draft beside an UltraQuant target.

The draft reads its own KV cache, not the UltraQuant target cache. A DSpark
step scores a non-causal block (k=7 is 8 query rows) against the whole draft
context. ``ROCM_AITER_FA`` sends that batch to ``flash_attn_with_kvcache``,
which does not split the context. This backend sends that batch to
``noncausal_decode_attention``. Every other batch uses the AITER FA path.

Selected by default when the target KV dtype is ``ultraquant_4bit`` and the
speculative config does not name a draft attention backend. Target
verification stays on the UltraQuant backend and still sees the full context.
"""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionBackend,
    AiterFlashAttentionImpl,
    AiterFlashAttentionMetadata,
)
from vllm.v1.attention.ops.aiter_noncausal_decode import (
    default_num_segments,
    noncausal_decode_attention,
)
from vllm.v1.kv_cache_interface import is_quantized_kv_cache

logger = init_logger(__name__)


class AiterFlashAttentionDSparkImpl(AiterFlashAttentionImpl):
    def _use_noncausal_split(
        self,
        attn_metadata: AiterFlashAttentionMetadata,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> bool:
        decode_metadata = attn_metadata.decode_metadata
        return (
            attn_metadata.causal is False
            and attn_metadata.num_decodes > 0
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_extends == 0
            and decode_metadata is not None
            and decode_metadata.max_query_len > 1
            and self.sliding_window[0] == -1
            and self.sinks is None
            and self.alibi_slopes is None
            and not self.logits_soft_cap
            and self.kv_sharing_target_layer_name is None
            and output_scale is None
            and output_block_scale is None
            and not rocm_aiter_ops.is_shuffle_kv_cache_enabled()
        )

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
        if attn_metadata is None or not self._use_noncausal_split(
            attn_metadata, output_scale, output_block_scale
        ):
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

        key_cache, value_cache = self._split_kv_cache(kv_cache)
        quantized = is_quantized_kv_cache(self.kv_cache_dtype)
        if quantized:
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        decode_metadata = attn_metadata.decode_metadata
        assert decode_metadata is not None
        max_query_len = decode_metadata.max_query_len
        # Segments depend on the batch shape, not the context length, so a
        # captured graph keeps the split it needs at long context.
        num_segments = default_num_segments(
            num_decodes,
            self.num_kv_heads,
            max_query_len,
            query.device,
            self.num_queries_per_kv,
        )
        logger.info_once(
            "DSpark draft decode is using the split-KV kernel "
            "(query_len=%d, segments=%d).",
            max_query_len,
            num_segments,
        )
        noncausal_decode_attention(
            q=query[:num_decode_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_decode_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc[: num_decodes + 1],
            max_seqlen_q=max_query_len,
            seqused_k=attn_metadata.seq_lens[:num_decodes],
            block_table=attn_metadata.block_table[:num_decodes],
            softmax_scale=self.scale,
            k_descale=layer._k_scale if quantized else None,
            v_descale=layer._v_scale if quantized else None,
            num_segments=num_segments,
        )
        return output


class AiterFlashAttentionDSparkBackend(AiterFlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_FA_DSPARK"

    @staticmethod
    def get_impl_cls() -> type[AiterFlashAttentionDSparkImpl]:
        return AiterFlashAttentionDSparkImpl
