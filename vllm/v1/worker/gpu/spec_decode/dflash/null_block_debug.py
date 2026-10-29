# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash null-block (block 0) diagnostics for H1 (uq_kernel_bottlenecks §7).

Gates (default off — no behaviour change):
  - ``VLLM_DFLASH_DEBUG_NULL_BLOCK=1``: read-only finite/NaN counter on draft
    layer block 0 after the first long prefill and every N draft steps.
  - ``VLLM_DFLASH_NULL_BLOCK_ZERO=1``: zero draft-layer block 0 before each
    draft forward (A/B for H1).

Called from a thin hook in ``DFlashSpeculator.propose``.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEBUG: bool | None = None
_ZERO: bool | None = None
_STEP = 0
_DID_LONG_PREFILL = False
_EVERY_N = 32
_LONG_CTX = 4096


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "TRUE", "yes")


def debug_null_block_enabled() -> bool:
    global _DEBUG
    if _DEBUG is None:
        _DEBUG = _env_on("VLLM_DFLASH_DEBUG_NULL_BLOCK")
        if _DEBUG:
            try:
                _EVERY_N_local = int(
                    os.environ.get("VLLM_DFLASH_DEBUG_NULL_BLOCK_EVERY", "32")
                )
            except ValueError:
                _EVERY_N_local = 32
            globals()["_EVERY_N"] = max(1, _EVERY_N_local)
            try:
                _LONG_CTX_local = int(
                    os.environ.get("VLLM_DFLASH_DEBUG_NULL_BLOCK_LONG_CTX", "4096")
                )
            except ValueError:
                _LONG_CTX_local = 4096
            globals()["_LONG_CTX"] = max(1, _LONG_CTX_local)
            logger.info(
                "VLLM_DFLASH_DEBUG_NULL_BLOCK=1 every=%d long_ctx=%d",
                _EVERY_N,
                _LONG_CTX,
            )
    return bool(_DEBUG)


def null_block_zero_enabled() -> bool:
    global _ZERO
    if _ZERO is None:
        _ZERO = _env_on("VLLM_DFLASH_NULL_BLOCK_ZERO")
        if _ZERO:
            logger.info(
                "VLLM_DFLASH_NULL_BLOCK_ZERO=1: zeroing draft block 0 before "
                "each draft forward"
            )
    return bool(_ZERO)


def _iter_draft_attn_kv(model: Any) -> list[tuple[str, torch.Tensor]]:
    """Return (layer_name, kv_cache) for draft attention layers."""
    out: list[tuple[str, torch.Tensor]] = []
    # DFlashQwen3ForCausalLM → .model.layers[*].self_attn.attn
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return out
    for i, layer in enumerate(layers):
        attn = getattr(getattr(layer, "self_attn", None), "attn", None)
        if attn is None:
            continue
        kv = getattr(attn, "kv_cache", None)
        if kv is None:
            continue
        if isinstance(kv, (list, tuple)):
            for j, t in enumerate(kv):
                if isinstance(t, torch.Tensor) and t.numel() > 0:
                    name = getattr(attn, "layer_name", f"draft_layer_{i}") + f"/kv{j}"
                    out.append((name, t))
        elif isinstance(kv, torch.Tensor) and kv.numel() > 0:
            name = getattr(attn, "layer_name", f"draft_layer_{i}")
            out.append((name, kv))
    return out


def _block0_view(kv: torch.Tensor) -> torch.Tensor:
    """Expose block 0 as a floating tensor for isfinite / zeroing."""
    blk = kv[0]
    if blk.is_floating_point():
        return blk
    # Overlay bytes: reinterpret as bf16 (H1: mamba/fp32 state → bf16 NaN).
    return blk.view(torch.bfloat16)


def check_null_block_finite(model: Any, *, tag: str = "") -> dict[str, dict[str, int]]:
    """Read-only: per draft layer, count non-finite elements in block 0."""
    report: dict[str, dict[str, int]] = {}
    for name, kv in _iter_draft_attn_kv(model):
        view = _block0_view(kv)
        total = int(view.numel())
        finite = int(torch.isfinite(view).sum().item())
        nonfinite = total - finite
        report[name] = {
            "total": total,
            "finite": finite,
            "nonfinite": nonfinite,
            "shape0": int(kv.shape[0]) if kv.ndim >= 1 else 0,
        }
        logger.info(
            "NULL_BLOCK_DEBUG%s layer=%s block0_nonfinite=%d/%d shape=%s",
            f"[{tag}]" if tag else "",
            name,
            nonfinite,
            total,
            tuple(kv.shape),
        )
    return report


def zero_null_block(model: Any) -> int:
    """Zero block 0 of each draft attention KV tensor. Returns layers touched."""
    n = 0
    for _name, kv in _iter_draft_attn_kv(model):
        kv[0].zero_()
        n += 1
    return n


def maybe_null_block_hooks(model: Any, *, max_seq_len: int) -> None:
    """Thin entry from DFlashSpeculator.propose (gates evaluated inside)."""
    global _STEP, _DID_LONG_PREFILL
    if null_block_zero_enabled():
        try:
            zero_null_block(model)
        except Exception:
            logger.exception("VLLM_DFLASH_NULL_BLOCK_ZERO failed (ignored)")

    if not debug_null_block_enabled():
        return

    _STEP += 1
    long_now = max_seq_len >= _LONG_CTX
    should = False
    tag = ""
    if long_now and not _DID_LONG_PREFILL:
        _DID_LONG_PREFILL = True
        should = True
        tag = "first_long"
    elif _STEP % _EVERY_N == 0:
        should = True
        tag = f"step{_STEP}"
    if not should:
        return
    try:
        check_null_block_finite(model, tag=tag)
    except Exception:
        logger.exception("VLLM_DFLASH_DEBUG_NULL_BLOCK failed (ignored)")
