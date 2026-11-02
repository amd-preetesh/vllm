# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P0-2: keep PDV on mixed-batch verify (experiment helper).

Shipping mixed-batch path in ``TurboQuantAttentionImpl.forward`` builds the
decode slice with ``max_query_len=1``. UltraQuant then falls out of the PDV
gate (``2 <= qlen <= 8``) into per-row ``_split_multi_query_decode``.

Gate: ``VLLM_ULTRAQUANT_MIXED_VERIFY=pdv`` (default unset = today).
Default environment keeps the shipping path unchanged.
"""

from __future__ import annotations

import os

import torch

_LOGGED = False


def mixed_verify_pdv_enabled() -> bool:
    return os.environ.get("VLLM_ULTRAQUANT_MIXED_VERIFY", "").strip() == "pdv"


def cap_reorder_batch_threshold(threshold: int | None) -> int | None:
    """Under the gate, classify only q ≤ 8 as decodes (PDV's max width)."""
    if threshold is None or not mixed_verify_pdv_enabled():
        return threshold
    return min(int(threshold), 8)


def decode_max_query_len(
    attn_metadata,
    num_rows: int,
) -> int:
    """Real max query length for a decode slice (no GPU sync).

    Prefer ``query_start_loc_cpu``; fall back to uniform ``num_rows / num_reqs``.
    """
    num_reqs = int(attn_metadata.seq_lens.shape[0])
    if num_reqs < 1:
        return int(attn_metadata.max_query_len)

    qsl_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    if qsl_cpu is not None and qsl_cpu.numel() >= num_reqs + 1:
        # Host tensor — diffs are free.
        qsl = qsl_cpu[: num_reqs + 1]
        if qsl.device.type != "cpu":
            qsl = qsl.cpu()
        lens = qsl[1:] - qsl[:-1]
        return int(lens.max().item())

    # Uniform MTP verify: num_rows == num_reqs * qlen.
    if num_rows % num_reqs == 0:
        return max(1, num_rows // num_reqs)
    return max(1, int(attn_metadata.max_query_len))


def resolve_pdv_qlen(attn_metadata, num_rows: int) -> int:
    """Query length used for the PDV gate, with mixed-batch repair when gated."""
    qlen = int(attn_metadata.max_query_len)
    if not mixed_verify_pdv_enabled():
        return qlen
    if qlen >= 2:
        return qlen
    # Mixed-batch shipping path forced max_query_len=1; recover from cu_seqlens.
    real = decode_max_query_len(attn_metadata, num_rows)
    global _LOGGED
    if not _LOGGED and real >= 2:
        from vllm.logger import init_logger

        init_logger(__name__).info_once(
            "VLLM_ULTRAQUANT_MIXED_VERIFY=pdv: repaired decode max_query_len "
            "%d → %d (mixed-batch PDV path).",
            qlen,
            real,
        )
        _LOGGED = True
    return real
