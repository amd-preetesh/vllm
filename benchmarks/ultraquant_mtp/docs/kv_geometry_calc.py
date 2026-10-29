#!/usr/bin/env python3
"""Offline mirror of the hybrid KV block geometry for Qwen3.8-27B + draft.

Mirrors `_align_hybrid_block_size` (platforms/interface.py), the SWA block
choice in `Attention.get_kv_cache_spec`, and `unify_kv_cache_spec_page_size`
closely enough to predict group block sizes, draft page utilization, and the
scheduler LCM (prefix-cache hit granularity) for candidate flags.

Usage: kv_geometry_calc.py
"""
from math import gcd

MAMBA_PAGE_K7 = 3_350_645  # derived from logs: 1648*2048 / 1.0073 (K=7 drafts)
MAMBA_PAGE_K5 = 3_309_577  # derived from logs: 1664*2048 / 1.0297 (MTP K=5)


def cdiv(a, b):
    return -(-a // b)


def lcm(a, b):
    return a * b // gcd(a, b)


def attn_block(mamba_page, per_token, align):
    return align * cdiv(mamba_page, align * per_token)


def draft_geometry(target_block, target_per_tok, draft_per_tok, draft_kernel_blocks,
                   sliding, policy="largest"):
    target_page = target_block * target_per_tok
    if sliding:
        fits = [b for b in draft_kernel_blocks
                if b <= target_block and b * draft_per_tok <= target_page]
        start = max(fits)
        if policy == "divisor":
            divs = [b for b in fits if target_page % (b * draft_per_tok) == 0]
            start = max(divs) if divs else start
        elif policy == "lcm_pad":
            # Largest block dividing the target block whose page still fits.
            divs = [b for b in range(16, target_block + 1, 16)
                    if target_block % b == 0 and b * draft_per_tok <= target_page]
            start = max(divs)
            draft_page = start * draft_per_tok
            if target_page % draft_page:
                return dict(draft_block=start, target_block=target_block,
                            padded=True, util=draft_page / target_page,
                            mamba_pad_x=1)
    else:
        start = target_block
    draft_page = start * draft_per_tok
    if draft_page > target_page:
        ratio = draft_page // target_page
        return dict(draft_block=start, target_block=target_block * ratio,
                    padded=False, util=1.0, mamba_pad_x=ratio)
    if target_page % draft_page == 0:
        blk = start * (target_page // draft_page)
        return dict(draft_block=blk, target_block=target_block, padded=False,
                    util=1.0, mamba_pad_x=1)
    return dict(draft_block=start, target_block=target_block, padded=True,
                util=draft_page / target_page, mamba_pad_x=1)


def report(name, mamba_page, target_per_tok, align, draft_per_tok,
           draft_kernel_blocks, sliding, policy="largest"):
    tb = attn_block(mamba_page, target_per_tok, align)
    g = draft_geometry(tb, target_per_tok, draft_per_tok, draft_kernel_blocks,
                       sliding, policy)
    sched = lcm(g["target_block"], g["draft_block"])
    print(f"{name:44s} target_blk={g['target_block']:5d} "
          f"draft_blk={g['draft_block']:5d} padded={str(g['padded']):5s} "
          f"draft_page_util={g['util']:.1%} mamba_page_x{g['mamba_pad_x']} "
          f"sched_lcm={sched}")


FP8 = 4 * 256 * 2          # target fp8 B/token/layer
UQ = 4 * 272               # target ultraquant_4bit B/token/layer
D_BF16 = 8 * 128 * 2 * 2   # draft bf16
D_FP8 = 8 * 128 * 2        # draft fp8
D_UQ = 8 * 136             # draft in UltraQuant format
FA = [16, 32]
FA_UQ = [16, 32, 64, 128, 256]
DENSE16 = list(range(16, 4097, 16))  # hypothetical MultipleOf(16) draft backend

if __name__ == "__main__":
    print("-- FP8 target --")
    report("DFlash bf16 draft, block base 16 (today)", MAMBA_PAGE_K7, FP8, 16,
           D_BF16, FA, True)
    report("DFlash bf16 draft, --block-size 64", MAMBA_PAGE_K7, FP8, 64,
           D_BF16, FA, True)
    report("DFlash fp8 draft,  --block-size 64", MAMBA_PAGE_K7, FP8, 64,
           D_FP8, FA, True)
    report("DSpark bf16 draft, block base 16 (today)", MAMBA_PAGE_K7, FP8, 16,
           D_BF16, FA, False)
    report("DSpark fp8 draft,  --block-size 64", MAMBA_PAGE_K7, FP8, 64,
           D_FP8, FA, False)
    print("-- UltraQuant target --")
    report("DFlash bf16 draft, FA_UQ_COMPAT (today)", MAMBA_PAGE_K7, UQ, 64,
           D_BF16, FA_UQ, True)
    report("DFlash bf16 draft, dense draft, lcm_pad", MAMBA_PAGE_K7, UQ,
           64, D_BF16, DENSE16, True, "lcm_pad")
    report("DFlash UQ-format draft (largest)", MAMBA_PAGE_K7, UQ, 64, D_UQ,
           FA_UQ, True)
    report("DFlash UQ-format draft (divisor)", MAMBA_PAGE_K7, UQ, 64, D_UQ,
           FA_UQ, True, "divisor")
    report("DSpark UQ-format draft", MAMBA_PAGE_K7, UQ, 64, D_UQ, FA_UQ, False)
    print("-- MTP reference --")
    report("MTP (draft == target spec)", MAMBA_PAGE_K5, FP8, 64, FP8, [64], False)
