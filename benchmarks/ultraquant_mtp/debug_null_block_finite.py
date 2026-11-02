#!/usr/bin/env python3
"""H1 null-block finite check helper (uq_kernel_bottlenecks.md §7).

Server-side path: set ``VLLM_DFLASH_DEBUG_NULL_BLOCK=1`` (and optionally
``VLLM_DFLASH_NULL_BLOCK_ZERO=1``) when launching a DFlash server; the thin
hook in ``DFlashSpeculator.propose`` calls
``vllm.v1.worker.gpu.spec_decode.dflash.null_block_debug``.

This script is a small CLI that re-exports the same helpers for unit/smoke
use, and can print gate status.

  python debug_null_block_finite.py --status
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--status",
        action="store_true",
        help="Print whether null-block debug/zero gates are on",
    )
    args = ap.parse_args()
    if args.status:
        dbg = os.environ.get("VLLM_DFLASH_DEBUG_NULL_BLOCK", "")
        zero = os.environ.get("VLLM_DFLASH_NULL_BLOCK_ZERO", "")
        every = os.environ.get("VLLM_DFLASH_DEBUG_NULL_BLOCK_EVERY", "32")
        print(f"VLLM_DFLASH_DEBUG_NULL_BLOCK={dbg!r}")
        print(f"VLLM_DFLASH_NULL_BLOCK_ZERO={zero!r}")
        print(f"VLLM_DFLASH_DEBUG_NULL_BLOCK_EVERY={every!r}")
        print(
            "Hook: vllm.v1.worker.gpu.spec_decode.dflash.null_block_debug"
            ".maybe_null_block_hooks"
        )
        return 0

    # Import path used by the server hook (smoke that the module loads).
    sys.path.insert(0, os.environ.get("VLLM_ROOT", "/workspace/vllm"))
    from vllm.v1.worker.gpu.spec_decode.dflash import null_block_debug as nbd

    print("loaded", nbd.__name__)
    print("debug_enabled", nbd.debug_null_block_enabled())
    print("zero_enabled", nbd.null_block_zero_enabled())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
