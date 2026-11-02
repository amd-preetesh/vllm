# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Choose which PDV kernel module to build.

The shipping module stays the default. Set VLLM_ULTRAQUANT_PDV_EXPERIMENT
to one of the keys below (hoist_k, qblock_waves, pdv2, pdv2_fp4pv, ...)
to build one of the experiment copies.
"""

from __future__ import annotations

import importlib
import os

_MODULES = {
    "": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256",
    "shipping": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256",
    "hoist_k": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256_hoist_k",
    "qblock_waves": (
        "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256_qblock_waves"
    ),
    "pdv2": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256_pdv2_fp4q",
    "pdv2g": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256_pdv2_fp4q",
    "pdv2p": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256_pdv2_fp4q",
    "pdv2_ctrl": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256_pdv2_ctrl",
    "pdv2_fp4pv": (
        "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256_pdv2_fp4pv"
    ),
    "fp4pv": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256_pdv2_fp4pv",
    "pdv2_tr": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256_pdv2_tr",
    "pdv2_iglp": (
        "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256_pdv2_iglp"
    ),
    "pdv2_32x32": (
        "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256_pdv2_32x32"
    ),
    # DFlash2 P3: specialized decode lands in dflash_flydsl_*; currently aliases
    # shipping PDV via that module until experiment math is filled in.
    "dflash": "vllm.v1.attention.ops.flydsl_kernels.dflash_flydsl_decode_hd256",
}


def pdv_experiment_name() -> str:
    """Return the selected experiment name. Empty means the shipping kernel."""
    return os.environ.get("VLLM_ULTRAQUANT_PDV_EXPERIMENT", "").strip()


def load_pdv_module():
    """Import the PDV module for the current experiment."""
    name = pdv_experiment_name()
    module_name = _MODULES.get(name)
    if module_name is None:
        known = ", ".join(sorted(k for k in _MODULES if k))
        raise ValueError(
            "VLLM_ULTRAQUANT_PDV_EXPERIMENT must be empty, shipping, "
            f"{known}; got {name!r}"
        )
    return importlib.import_module(module_name)
