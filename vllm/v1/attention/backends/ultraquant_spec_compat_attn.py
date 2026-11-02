# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UltraQuant target backend for DFlash/DSpark UQ + FlyDSL experiments.

Shipping ``ULTRAQUANT`` alone prefers block size 64, so hybrid Qwen3.8 lands
on a mamba-aligned manager block divisible by 64 (historically 3008) and
virtual-splits to a FlyDSL kernel view of 64.

With a separate FA draft, ``Platform._preferred_block_size_for_backends``
falls back to default 16 when every backend accepts 16 — hybrid then lands
on 3088 (16-aligned only), which cannot split to 64.

This experiment backend:
1. Prefers / requires 64-aligned manager sizes (with FA_UQ_COMPAT).
2. Advertises the same exact kernel sizes as shipping UltraQuant so
   ``select_common_block_size`` picks 64 on a 3008-token manager.
3. Keeps shipping UltraQuant FlyDSL decode (no Triton-forced fallback).
4. When ``enable_adaptive_verification`` is on, reports
   ``AttentionCGSupport.ALWAYS`` so DSpark adaptive verify can capture varlen
   decode graphs (shipping TurboQuant/UltraQuant stay ``UNIFORM_BATCH``).

Does not edit shipping ``ultraquant_attn.py`` or FP8 DFlash/DSpark paths.
Pair with ``ROCM_AITER_FA_UQ_COMPAT`` / ``ROCM_AITER_FA_UQ_DENSE`` draft and
``--block-size 64``.
"""

from __future__ import annotations

from typing import ClassVar

from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionCGSupport, MultipleOf
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend
from vllm.v1.attention.backends.ultraquant_attn import (
    UltraQuantAttentionBackend,
    UltraQuantMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheSpec


class UltraQuantSpecCompatMetadataBuilder(UltraQuantMetadataBuilder):
    """UQ SPEC_COMPAT builder with adaptive-verification varlen CG support."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    @classmethod
    def get_cudagraph_support(  # type: ignore[override]
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        speculative_config = vllm_config.speculative_config
        if (
            speculative_config is not None
            and speculative_config.enable_adaptive_verification
        ):
            # Metadata already carries device query_start_loc; adaptive verify
            # needs varlen full graphs (DeepSeek / Kimi pattern).
            return AttentionCGSupport.ALWAYS
        return cls._cudagraph_support


class UltraQuantSpecCompatAttentionBackend(UltraQuantAttentionBackend):
    """UQ target for DFlash/DSpark UltraQuant + FlyDSL (experiment)."""

    @staticmethod
    def get_name() -> str:
        return "ULTRAQUANT_SPEC_COMPAT"

    @staticmethod
    def get_builder_cls() -> type[UltraQuantSpecCompatMetadataBuilder]:
        return UltraQuantSpecCompatMetadataBuilder

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 64

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        # Reject default 16 so multi-backend preferred-size selection cannot
        # skip UltraQuant's 64 preference (see Platform._preferred_block_size).
        if block_size is None:
            return True
        return block_size % 64 == 0

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec=None,
    ) -> list[int | MultipleOf]:
        # Same exact set as TurboQuant/UltraQuant shipping: enables virtual
        # split of a 64-aligned manager (e.g. 3008) down to kernel view 64.
        return list(TurboQuantAttentionBackend.get_supported_kernel_block_sizes())
