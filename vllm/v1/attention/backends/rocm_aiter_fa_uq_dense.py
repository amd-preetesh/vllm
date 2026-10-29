# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense-page draft FA for UltraQuant + DFlash/DSpark (experiment).

Shipping ``ROCM_AITER_FA_UQ_COMPAT`` caps the draft page at the max aiter
kernel block (256). Against a UQ hybrid manager block of 3136 that leaves
draft util ~31% and scheduler LCM 12544.

This backend keeps the same LBNHC / 64-prefer layout rules but pads the draft
page to 784 (= 3136/4), the lcm-padded dense size from
``kv_geometry_calc.py`` (≈94% util, LCM 3136). Selected only via speculative
config ``attention_backend: ROCM_AITER_FA_UQ_DENSE`` — default stays on
``ROCM_AITER_FA_UQ_COMPAT``.
"""

from __future__ import annotations

from dataclasses import replace

from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.rocm_aiter_fa_uq_compat import (
    AiterFlashAttentionUQCompatBackend,
)
from vllm.v1.kv_cache_interface import AttentionSpec

# LCM-padded draft page vs UQ target 3136 (see kv_geometry_calc dense row).
_DENSE_DRAFT_BLOCK = 784


class AiterFlashAttentionUQDenseBackend(AiterFlashAttentionUQCompatBackend):
    """Draft FA with dense (lcm-padded) pages for UQ hybrid KV pools."""

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_FA_UQ_DENSE"

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec=None,
    ) -> list[int | MultipleOf]:
        # Keep FlyDSL-legal small sizes; add dense page size for allocator.
        return [16, 32, 64, 128, 256, _DENSE_DRAFT_BLOCK]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if spec.block_size != 1 and spec.block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # Cap large hybrid manager blocks to the dense draft page (not 256).
        if spec.block_size > _DENSE_DRAFT_BLOCK:
            return replace(spec, block_size=_DENSE_DRAFT_BLOCK)
        return spec
