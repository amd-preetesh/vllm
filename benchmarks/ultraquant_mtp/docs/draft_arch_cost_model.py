#!/usr/bin/env python3
"""Bandwidth-floor cost model for one spec-decode step: target + draft.

Offline, read-only. Companion to `uq_draft_mlp_architecture_research.md`.
Counts HBM bytes per decode step and converts them to milliseconds with an
assumed effective bandwidth per stream. The output is a *floor* used to rank
draft architecture variants. It is not a profile: real steps also pay launch
gaps, split-K inefficiency, and Python overhead.

Shapes come from the checkpoints in this workspace:
  target  Qwen3.8-27B bf16 weights, 64 layers (16 full attn, 48 GDN),
          UQ KV 4 kv heads x 272 B per token per full-attn layer
  DSpark  5 layers, d=5120, MLP 17408, 32 q / 8 kv heads, hd128, bf16 KV,
          Markov rank 256, shared target lm_head (V=248320)

Usage:
  draft_arch_cost_model.py                       # default table, R=5.6, L=150k
  draft_arch_cost_model.py --reqs 6 --ctx 200000 --attn-bw 2.0
"""
import argparse
from dataclasses import dataclass, replace

V = 248_320
D = 5120
TARGET_PARAMS = 27.0e9
TARGET_FULL_ATTN = 16
TARGET_GDN = 48
UQ_BYTES_PER_TOK_LAYER = 4 * 272
GDN_STATE_BYTES = 48 * 128 * 128 * 4  # per layer per request, fp32


@dataclass(frozen=True)
class Draft:
    name: str
    layers: int = 5
    ctx_layers: int = 5           # layers that read context KV
    window: int | None = None     # None = full context
    kv_heads: int = 8
    head_dim: int = 128
    kv_bytes: float = 2.0         # bf16=2, fp8=1, UQ 4-bit ~0.53
    inter: int = 17408
    w_bytes: float = 2.0          # draft linear weights
    draft_vocab: int = V
    markov_rank: int = 256        # 0 = no sequential V-wide head
    sequential_steps: int = 7
    target_kv_xattn_layers: int = 0  # GliDe-style reads of the target UQ KV
    own_kv: bool = True


def draft_bytes(d: Draft, reqs: float, ctx: int) -> dict[str, float]:
    attn_params = D * (32 * 128) + 2 * D * d.kv_heads * d.head_dim + (32 * 128) * D
    mlp_params = 3 * D * d.inter
    eff_ctx = ctx if d.window is None else min(ctx, d.window)
    per_tok_layer = 2 * d.kv_heads * d.head_dim * d.kv_bytes
    out = {
        "draft MLP weights": d.layers * mlp_params * d.w_bytes,
        "draft attn-proj weights": d.layers * attn_params * d.w_bytes,
        "lm_head": d.draft_vocab * D * 2.0,
        "Markov W2 (per seq step)": d.sequential_steps * d.draft_vocab * d.markov_rank * 2.0
        if d.markov_rank else 0.0,
        "draft context KV read": reqs * d.ctx_layers * eff_ctx * per_tok_layer
        if d.own_kv else 0.0,
        "target UQ KV xattn read": reqs * d.target_kv_xattn_layers * ctx * UQ_BYTES_PER_TOK_LAYER,
    }
    return out


def target_bytes(reqs: float, ctx: int) -> dict[str, float]:
    return {
        "target weights (bf16)": TARGET_PARAMS * 2.0,
        "target UQ KV read": reqs * TARGET_FULL_ATTN * ctx * UQ_BYTES_PER_TOK_LAYER,
        "target GDN state r+w": reqs * TARGET_GDN * GDN_STATE_BYTES * 2,
    }


def pool_bytes_per_token(d: Draft) -> float:
    """Draft KV bytes that grow with context (0 when the draft KV is windowed)."""
    if not d.own_kv or d.window is not None:
        return 0.0
    return d.ctx_layers * 2 * d.kv_heads * d.head_dim * d.kv_bytes


def windowed_mb_per_req(d: Draft) -> float:
    if not d.own_kv or d.window is None:
        return 0.0
    return d.window * d.ctx_layers * 2 * d.kv_heads * d.head_dim * d.kv_bytes / 1e6


VARIANTS = [
    Draft("DSpark today (5L full, bf16 KV)"),
    Draft("DSpark + FP8 draft KV", kv_bytes=1.0),
    Draft("DSpark + UQ-format draft KV (P2-1)", kv_bytes=272 / 256 * 0.5),
    Draft("DSpark SWA 8192", window=8192),
    Draft("DSpark SWA 2048", window=2048),
    Draft("DSpark 3L full", layers=3, ctx_layers=3),
    Draft("DSpark 3L SWA 4096 + FP8 W + 64k vocab", layers=3, ctx_layers=3,
          window=4096, w_bytes=1.0, draft_vocab=65536),
    Draft("UQ-native: 3L SWA2048 + 1 target-UQ xattn, FP8 W, 64k vocab, selector",
          layers=3, ctx_layers=3, window=2048, w_bytes=1.0, draft_vocab=65536,
          markov_rank=0, target_kv_xattn_layers=1),
    Draft("GliDe-only: 3L, no own KV, 1 target-UQ xattn", layers=3, ctx_layers=0,
          own_kv=False, target_kv_xattn_layers=1),
]

ATTN_KEYS = {"draft context KV read", "target UQ KV xattn read", "target UQ KV read"}


def ms(nbytes: float, bw_tbs: float) -> float:
    return nbytes / (bw_tbs * 1e12) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", type=float, default=5.6, help="running requests per step")
    ap.add_argument("--ctx", type=int, default=150_000, help="mean context tokens")
    ap.add_argument("--weight-bw", type=float, default=5.5, help="TB/s for weight streams")
    ap.add_argument("--attn-bw", type=float, default=2.0,
                    help="TB/s for attention KV streams (PDV measured ~1.5)")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()

    def t(k, v):
        return ms(v, a.attn_bw if k in ATTN_KEYS else a.weight_bw)

    tgt = target_bytes(a.reqs, a.ctx)
    tgt_ms = sum(t(k, v) for k, v in tgt.items())
    print(f"R={a.reqs} ctx={a.ctx} weight_bw={a.weight_bw} TB/s attn_bw={a.attn_bw} TB/s")
    print(f"target floor: {tgt_ms:.1f} ms  ("
          + ", ".join(f"{k} {t(k, v):.1f}" for k, v in tgt.items()) + ")")
    print()
    print(f"{'variant':<72} {'draft GB':>8} {'draft ms':>8} {'share':>6} "
          f"{'pool B/tok':>10} {'SWA MB/req':>10}")
    for d in VARIANTS:
        b = draft_bytes(d, a.reqs, a.ctx)
        dms = sum(t(k, v) for k, v in b.items())
        print(f"{d.name:<72} {sum(b.values()) / 1e9:8.2f} {dms:8.2f} "
              f"{dms / (dms + tgt_ms):6.1%} {pool_bytes_per_token(d):10.0f} "
              f"{windowed_mb_per_req(d):10.1f}")
        if a.detail:
            for k, v in b.items():
                print(f"    {k:<34} {v / 1e9:7.2f} GB {t(k, v):6.2f} ms")


if __name__ == "__main__":
    main()
