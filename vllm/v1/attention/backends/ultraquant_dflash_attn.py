# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UltraQuant + DFlash P3 experiment backend.

Forces the ``dflash`` PDV experiment (``dflash_flydsl_decode_hd256``) while
reusing UltraQuantAttentionBackend. Shipping UltraQuant / PDV paths stay
unchanged; default ``ULTRAQUANT`` still selects the shipping kernel.
"""

from __future__ import annotations

import os

# Force before UltraQuant imports resolve the PDV experiment module.
os.environ.setdefault("VLLM_ULTRAQUANT_PDV_VERIFY", "1")
os.environ["VLLM_ULTRAQUANT_PDV_EXPERIMENT"] = "dflash"

from vllm.v1.attention.backends.ultraquant_attn import (  # noqa: E402
    UltraQuantAttentionBackend,
)


class UltraQuantDFlashAttentionBackend(UltraQuantAttentionBackend):
    """Attention backend name for DFlash2 P3 FlyDSL experiments."""

    @staticmethod
    def get_name() -> str:
        return "ULTRAQUANT_DFLASH"
