# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft window: sink pages packed in front of the recent tokens."""

import torch

from vllm.v1.attention.backends.rocm_aiter_fa_dspark import (
    DSPARK_DRAFT_SINK,
    DSPARK_DRAFT_WINDOW,
)
from vllm.v1.attention.ops.dspark_draft_window import build_draft_window_sink_tables


def test_short_sequence_keeps_its_real_pages():
    block = 64
    table = torch.arange(8, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([100], dtype=torch.int32)
    pages, packed = build_draft_window_sink_tables(
        table,
        seq_lens,
        block_size=block,
        window=DSPARK_DRAFT_WINDOW,
        sink=DSPARK_DRAFT_SINK,
    )
    assert int(packed[0]) == 100
    assert int(pages[0, 0]) == 0
    assert int(pages[0, 1]) == 1


def test_long_sequence_keeps_the_sink_and_the_recent_window():
    block = 64
    n_blocks = 200
    table = torch.arange(n_blocks, dtype=torch.int32).view(1, -1)
    seq_len = 10_000
    pages, packed = build_draft_window_sink_tables(
        table,
        torch.tensor([seq_len], dtype=torch.int32),
        block_size=block,
        window=DSPARK_DRAFT_WINDOW,
        sink=DSPARK_DRAFT_SINK,
    )
    # One sink page (4 tokens round up to 64) plus the tokens from the window.
    win_start = max(1, (seq_len - DSPARK_DRAFT_WINDOW) // block)
    expect = block + (seq_len - win_start * block)
    assert int(packed[0]) == expect
    assert int(pages[0, 0]) == 0
    assert int(pages[0, 1]) == win_start
    used = (seq_len + block - 1) // block
    assert int(pages[0, used - win_start]) == used - 1
