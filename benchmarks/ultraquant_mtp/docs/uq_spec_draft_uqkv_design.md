# Design: UltraQuant-native speculative draft (weekend smoke → paper path)

Date: 2026-10-23. Implementation plan for Idea A: a draft that **reads the
target’s UltraQuant 4-bit paged KV** instead of keeping its own long bf16 KV.
Framing: a **new UltraQuant-specific** way to do 4-bit speculative decoding
(GliDe/LongSpec architecture + QuantSpec problem class), not “first 4-bit spec.”

Related: `uq_draft_mlp_architecture_research.md`, QuantSpec / GliDe / LongSpec.

---

## 1. Goal

**Serve** a lightweight draft whose long-range attention is:

```
Q_draft (bf16)  ×  K_uq, V_uq  (UltraQuant pages of one target layer)
```

No growing bf16 draft context KV. Local context optional via a short SWA
self-attn window (LongSpec-style). Train later so accept matches UQ target.

**AMD story:** CDNA runs the UltraQuant Triton/FlyDSL path; Instinct is the
eval platform, not the contribution.

---

## 2. Phased plan

| Phase | When | What | Success |
|---|---|---|---|
| **S0 smoke** | This weekend | Synthetic: Q × UltraQuant KV via existing FlyDSL/unified path vs dequant+SDPA reference | cosine ≥ 0.999 on ctx 4k–32k, qlen 1–8; latency printout |
| **S1 config** | After S0 / GPU free | P0-A: DSpark SWA override accept probe (no retrain) | W=8192 ≥ 95% of full-attn accept |
| **S2 kernel** | Next week | New module `draft_cross_uq_kv.py` (env-gated) wrapping PDV/FlyDSL for multi-q draft rows; unique LDS if new FlyDSL file | unit test + microbench vs S0 |
| **S3 stub draft** | Later | 1-layer draft forward in vLLM that cross-attends a chosen target layer’s UQ cache (random/untrained weights OK) | engine loads; propose() returns tokens |
| **S4 train** | Paper track | Distill/accept-train draft against UQ target (external SpecForge/NeMo) | accept vs DSpark UQ / MTP UQ on AgentX |

Do **not** block AgentX/opt/SWA/diag queues. Smoke uses NEED_GIB≈40 like E6.

---

## 3. Architecture (target design)

```
Target (Qwen3.8-27B, UltraQuant KV)
  └─ layer ℓ* full-attn KV pages  ──────────────────┐
                                                     │ cross-attn (FlyDSL/PDV)
Draft (small):                                       ▼
  embed(tokens) → [optional SWA self-attn, W≤512]
                → CrossAttn(Q ← h_draft, K/V ← UQ pages of ℓ*)
                → MLP → LM head (shared or small)
```

**S0 only exercises the CrossAttn × UQ pages box** (no draft weights).

Gates (future): `VLLM_ULTRAQUANT_DRAFT_UQKV=1` — default off.

---

## 4. Weekend smoke (S0) — exact procedure

Script: `benchmarks/ultraquant_mtp/bench_draft_cross_uq_kv_smoke.py`

1. Build random bf16 K,V `[T, Hk, D]`, Q `[qlen, Hq, D]` on one HIP device.
2. `ultraquant_store` into paged cache (block 64).
3. **UQ path:** `flydsl_ultraquant_decode_attention` (or unified fallback) with
   `pdv_verify` / shared query as available — bf16 Q, UQ cache.
4. **Reference:** dequant cache to bf16 + scaled_dot_product_attention (GQA).
5. Report cosine, err, ms for T ∈ {4096, 16384, 32768}, qlen ∈ {1, 4, 8}.
6. Write `out/uq_draft_uqkv_smoke/summary.json`.

Waiter: `run_uq_draft_uqkv_smoke_when_free.sh` — claim_gpu 40, no port conflict.

---

## 5. Related-work positioning (for the paper)

- **GliDe / LongSpec:** draft reuses target KV — we do too, but KV is **UltraQuant**.
- **QuantSpec:** 4-bit KV + self-spec hierarchical INT4 — we use a **different
  4-bit format** + **external lightweight draft**, not self-spec INT4 hierarchy.
- Claim: *new UltraQuant-native speculative drafting method*, not first 4-bit spec.

---

## 6. Non-goals for the weekend

- No draft weight training.
- No edits to shipping `pdv_flydsl_decode_hd256.py`.
- No killing ports 8460–8470.
- No AgentX full matrix for this idea yet.
