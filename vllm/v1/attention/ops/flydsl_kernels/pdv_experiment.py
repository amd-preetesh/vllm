# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select the PDV FlyDSL decode kernel module.

Default (empty / ``shipping`` / ``pdv1.0``) is PDV 1.0 — one wavefront, one
KV scan scores every packed-verify query.

Set ``VLLM_ULTRAQUANT_PDV_EXPERIMENT=hoist_k`` (or ``pdv1.1``) to opt into
PDV 1.1 (hoist K/scales once per tile). Shipping default stays 1.0.
"""

from __future__ import annotations

import importlib
import os

_MODULES = {
    "": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256",
    "shipping": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256",
    "pdv1.0": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256",
    # PDV 1.1 is added on the hoist-k branch; keep aliases reserved here so
    # an accidental env set fails clearly until that module is present.
}


def pdv_experiment_name() -> str:
    """Return the selected PDV variant. Empty means PDV 1.0."""
    return os.environ.get("VLLM_ULTRAQUANT_PDV_EXPERIMENT", "").strip()


def load_pdv_module():
    """Import the PDV kernel module for the current experiment env."""
    name = pdv_experiment_name()
    module_name = _MODULES.get(name)
    if module_name is None:
        known = ", ".join(sorted(k for k in _MODULES if k)) or "(default)"
        raise ValueError(
            "VLLM_ULTRAQUANT_PDV_EXPERIMENT must be empty, shipping, pdv1.0"
            + (f", {known}" if known != "(default)" else "")
            + f"; got {name!r}"
        )
    return importlib.import_module(module_name)
