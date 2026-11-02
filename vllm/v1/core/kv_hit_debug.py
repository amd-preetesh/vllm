# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-group prefix-cache hit diagnostic (E3 / P1-4).

Gate: ``VLLM_KV_HIT_DEBUG=1`` (default off — no behaviour change).

When enabled, ``KVCacheManager.get_computed_blocks`` logs request id, prompt
length, per-group independent hit lengths, argmin group, alignment, free
blocks, and cached-block count so residual zero-hit events can be attributed
to mamba / SWA / full-attn groups.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request

logger = init_logger(__name__)

_ENABLED: bool | None = None


def kv_hit_debug_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("VLLM_KV_HIT_DEBUG", "").strip() in (
            "1",
            "true",
            "TRUE",
            "yes",
        )
    return _ENABLED


def _group_label(manager: KVCacheManager, group_id: int) -> str:
    groups = manager.kv_cache_config.kv_cache_groups
    if group_id < 0 or group_id >= len(groups):
        return f"g{group_id}"
    group = groups[group_id]
    kind = get_kv_cache_spec_kind(group.kv_cache_spec).value
    eagle = "eagle" if group.is_eagle_group else "tgt"
    return f"g{group_id}:{kind}:{eagle}"


def _alignment_tokens(coordinator: Any) -> int | None:
    prop = getattr(coordinator, "_cache_hit_alignment_tokens", None)
    if prop is None:
        return None
    try:
        return int(prop)
    except Exception:
        return None


def log_computed_blocks_hit(
    manager: KVCacheManager,
    request: Request,
    reconciled_hit: int,
    shared_prefix_boundary: int,
) -> None:
    """Log per-group independent hits after a prefix-cache lookup.

    Safe no-op when the gate is off. Never raises into the scheduler path.
    """
    if not kv_hit_debug_enabled():
        return
    try:
        _log_computed_blocks_hit_impl(
            manager, request, reconciled_hit, shared_prefix_boundary
        )
    except Exception:
        logger.exception("VLLM_KV_HIT_DEBUG logging failed (ignored)")


def _log_computed_blocks_hit_impl(
    manager: KVCacheManager,
    request: Request,
    reconciled_hit: int,
    shared_prefix_boundary: int,
) -> None:
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    coordinator = manager.coordinator
    prompt_len = int(getattr(request, "num_prompt_tokens", request.num_tokens))
    num_tokens = int(request.num_tokens)
    alignment = _alignment_tokens(coordinator)

    free_blocks = manager.block_pool.get_num_free_blocks()
    cached_blocks = len(manager.block_pool.cached_block_hash_to_block)

    per_group: list[tuple[int, int, str]] = []
    if isinstance(coordinator, HybridKVCacheCoordinator):
        _, hit_lengths = coordinator.find_longest_cache_hit_per_group(
            request.block_hashes, max(0, num_tokens - 1)
        )
        # Only cacheable groups participate in hit lookup; skip scratch
        # groups that stay at hit=0 by construction.
        cacheable_gids: set[int] = set()
        for ag in getattr(coordinator, "attention_groups", ()):
            cacheable_gids.update(ag.group_ids)
        for gid, hit in enumerate(hit_lengths):
            if cacheable_gids and gid not in cacheable_gids:
                continue
            group = manager.kv_cache_config.kv_cache_groups[gid]
            if not group.kv_cache_spec.prefix_cacheable:
                continue
            per_group.append((gid, int(hit), _group_label(manager, gid)))
    else:
        per_group.append((0, int(reconciled_hit), "g0:reconciled"))

    if per_group:
        _argmin_gid, argmin_hit, argmin_label = min(per_group, key=lambda x: x[1])
    else:
        argmin_hit, argmin_label = 0, "none"

    group_hits_str = ",".join(f"{label}={hit}" for _, hit, label in per_group)
    logger.info(
        "KV_HIT_DEBUG req=%s prompt_len=%d num_tokens=%d reconciled=%d "
        "boundary=%d alignment=%s argmin=%s(%d) per_group=[%s] "
        "free_blocks=%d cached_blocks=%d",
        request.request_id,
        prompt_len,
        num_tokens,
        int(reconciled_hit),
        int(shared_prefix_boundary),
        alignment if alignment is not None else "n/a",
        argmin_label,
        argmin_hit,
        group_hits_str,
        free_blocks,
        cached_blocks,
    )
