# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm AITER FA draft backend for UltraQuant + DFlash/DSpark experiments.

UltraQuant only supports ``LBNHC``. Shipping ``ROCM_AITER_FA`` only supports
``LBHNC``/``LHBNC``. Spec decode with UQ target + FA draft then fails layout
intersection.

Also: when FA accepts default block size 16 alongside UltraQuant, platform
preferred-size selection stays at 16 and hybrid Qwen3.8 lands on a manager
block that is only 16-aligned (3088), blocking FlyDSL's preferred 64-way
split. This backend prefers 64-aligned manager sizes and skips the shuffle
head-slot rewrite (UQ∩draft resolves to LBNHC).

FP8 DFlash/DSpark paths keep using ``ROCM_AITER_FA`` unchanged.
"""

from __future__ import annotations

from dataclasses import replace

from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout


class AiterFlashAttentionUQCompatBackend(AiterFlashAttentionBackend):
    """Draft attention for UltraQuant + DFlash/DSpark (experiment)."""

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_FA_UQ_COMPAT"

    @staticmethod
    def get_impl_cls():
        # Thin dispatcher: env selects the P0-3 window-table Impl without
        # changing the speculative_config backend string.
        from vllm.v1.attention.backends.rocm_aiter_fa_dflash_swa import (
            AiterFlashAttentionDFlashSwaImpl,
            dflash_win_table_experiment_enabled,
        )

        if dflash_win_table_experiment_enabled():
            return AiterFlashAttentionDFlashSwaImpl
        return AiterFlashAttentionBackend.get_impl_cls()

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Prefer LBNHC so UltraQuant target wins the intersection.
        return (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC, KVCacheLayout.LHBNC)

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 64

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 64 == 0

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec=None,
    ) -> list[int | MultipleOf]:
        # Align with UltraQuant's FlyDSL-legal set so a 3008 manager splits
        # to kernel view 64 for both target and draft groups.
        return [16, 32, 64, 128, 256]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # Skip shuffle head-slot rewrite from the parent. UQ∩draft resolves to
        # LBNHC; shuffle's dual head-slots assume FA's native page layout.
        if spec.block_size != 1 and spec.block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # DFlash SWA already picks the largest kernel block via
        # `_largest_kernel_block_within` (256), so FA draft page stays below
        # the UltraQuant hybrid manager page and unify pads FA — not UQ.
        # DSpark full-attn draft otherwise inherits the ~3k manager block,
        # FA page exceeds UQ, unify pads UQ, and FlyDSL's dense 64-split
        # fails (block stride != unpadded page). Cap to max kernel size.
        kernel_sizes = [
            s if isinstance(s, int) else s.base
            for s in cls.get_supported_kernel_block_sizes()
        ]
        max_kernel = max(kernel_sizes)
        if spec.block_size > max_kernel:
            return replace(spec, block_size=max_kernel)
        return spec
