# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P3 DFlash FlyDSL decode experiment kernel.

Selected with ``VLLM_ULTRAQUANT_PDV_EXPERIMENT=dflash`` (or the
``ULTRAQUANT_DFLASH`` attention backend, which forces that experiment).

This module currently builds the shipping PDV kernel as a correctness
baseline. Specialized DFlash decode math lands here later in a full copy
with its own LDS symbol — the shipping ``pdv_flydsl_decode_hd256`` file
is never edited.
"""

from __future__ import annotations

from vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256 import (
    build_pdv_flydsl_decode_hd256_module,
)

__all__ = ["build_pdv_flydsl_decode_hd256_module"]
