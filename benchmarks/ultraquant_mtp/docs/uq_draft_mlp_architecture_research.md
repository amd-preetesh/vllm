# Draft-model architecture for UltraQuant spec decode (Qwen3.8-27B, MI355X)

Date: 2026-10-23. Research note, no shipping code changed. The only new file
besides this note is the offline calculator `draft_arch_cost_model.py` in this
directory (pure Python, no GPU, no imports from vLLM).

Read with: `uq_dflash_dspark_bottlenecks.md`, `uq_kernel_bottlenecks.md`,
`dflash_dspark_agentx_bottlenecks.md`.

Tags: **[measured]** = from logs/probes in `out/`; **[model]** = from
`draft_arch_cost_model.py` (bandwidth floor, not a profile); **[lit]** = from a
cited paper/blog, measured on other hardware/targets; **[hyp]** = untested.

---

## 0. TL;DR

1. **The draft MLP is not where draft time goes on this stack.** At AgentX c32
   (about 5.6 running requests, about 150k context), DSpark's five draft layers
   stream about 3.2 GB of weights per step. The draft's own bf16 context KV is
   about 17 GB per step. The KV is about 5x the weights and about 36% of the
   decode step's bandwidth floor **[model]**. Making the MLP thinner or shallower
   saves well under 1 ms per step. Bounding what the draft attends to saves
   5–9 ms **[model]**. It also fixes the small DSpark UQ KV pool: the draft
   stores 20,480 B per context token, which is more than the whole UQ target
   (17,408 B).
2. **The best-supported architecture change is a draft that does not keep its
   own full-length KV.** Three examples. DFlash2 (sliding window 2048) beats
   DSpark by +1.2 accept on this exact target [lit]. DeepSeek's production DSpark
   uses a 128-token sliding window [lit]. LongSpec uses a 512-token window plus
   cross-attention to the target's KV [lit]. The target hidden states that
   DFlash/DSpark inject already carry the long-range information.
3. **Low expected value: UQ-aware distillation on its own.** DSpark accept under
   UQ is within noise of FP8 KV: matched greedy probe cells give 3.81 vs 3.91,
   and AgentX c16 gives 2.89 vs 2.68 **[measured]**. Fold UQ-target feature capture
   into any retrain, but do not fund a retrain just for it.
4. **The real accept gap on AgentX is at the first draft token and in
   intra-block coherence, not long context.** Draft position 0 accepts 0.67 for
   DSpark vs about 0.74 for native MTP. Conditional accept at positions 1–2 is
   about 0.58 vs about 0.73 **[measured]**. The cheap fix in the literature is
   DFlash2's two-tap convolution plus top-16 path selector: +0.6–1.3% latency,
   versus +9.6% for DSpark's full-vocabulary Markov correction [lit].

**Top 3 architecture ideas for this UQ stack (ranked):**

| rank | idea | why here | first validation |
|---|---|---|---|
| 1 | **Windowed draft attention** (SWA 2k–8k over the injected context features) | Removes about 90% of draft step bytes at 150k context [model]. Draft KV becomes O(window) per request, so the pool grows to MTP class and the planned FP8 / UQ-format draft-KV work (P2-1) may not be needed. DFlash2 and DeepSeek-V4 DSpark both ship it | zero-retrain DSpark SWA override + accept probe (section 7, P0-A) |
| 2 | **Thin, coherent backbone:** 3 layers + DFlash2 2-tap dynamic conv + low-rank path selector instead of the V-wide Markov loop, reduced draft vocab, FP8 draft weights | Closes the AgentX coherence gap (cond. accept 0.58 vs MTP 0.73) for about 1% latency. Removes the 7-step sequential 248k-vocab loop. Once rank 1 lands, weights dominate draft bytes | DFlash2 after the H1 fix *is* this architecture minus depth/vocab; run DFlash2 vs DSpark at matched config (P0-C) |
| 3 | **UQ-native cross-attention draft** (GliDe/LongSpec style): one draft layer reads the target's own UQ 4-bit KV of one full-attention layer through a PDV-style kernel, with SWA self-attention for local context | No long draft KV at all. Long-range retrieval at 1088 B/token (3.8x less than bf16). Draft is trained on the UQ-dequantized KV it reads at serving time, so it is UQ-native by construction | only if rank 1 shows a long-context accept loss; needs retrain + new kernel file (P2-A) |

**Cheapest validation experiment to queue next week:** P0-A. Serve DSpark UQ
from a config-only copy with `sliding_window` W ∈ {262144 (control), 32768,
8192, 2048} through the existing `ROCM_AITER_FA_DFLASH_SWA` backend. Then run
`probe_accept_vs_length.py` with the same contexts/gens as
`out/uq_accept_probe/dspark_uq`. There is no retrain and no code change. The
W=262144 control does double duty: it checks the window-table backend on a model
known to be healthy under full attention. Decision rule: if W=8192 keeps ≥95% of
control accept, rank 1 is the path.

**Blockers / unknowns:**

- **No kernel-level profile of a DSpark UQ decode step exists.** All step
  shares here are bandwidth floors. Measured steps are about 3x above the floor
  (section 2.3). `uq_kernel_bottlenecks.md` P0-1 is still the gating
  measurement.
- **The SWA draft path on UQ is unvalidated.** The DFlash null-block bug (H1) and
  the window-table backend are still open. `out/dflash_qwen38_uq_dflash_swa` has
  no cells yet. `out/dflash_accept_bisect/nowin` failed on missing weights (the
  symlink was added today).
- **Training recipe gaps.** DSpark training context length and data are
  unknown: the README states only the target and evaluation suites. No
  training pipeline for these drafters exists in this workspace. SpecForge,
  DeepSpec, NeMo AutoModel and AngelSlim are external.
- **Causality caveat for P0-A.** In this repo a layer with `layer_types =
  sliding_attention` defaults to *causal* (`qwen3_dflash.py`
  `_dflash_layer_causal`). The SWA copy must set `"is_causal": false`. Also, the
  inherited aiter window tuple `(W-1, 0)` makes the query block causal anyway
  (H3 in `uq_kernel_bottlenecks.md`). Expect a small accept bias from that,
  identical across W, so the control still isolates the window effect.

---

## 1. Current draft architecture in this repo (Q1)

### 1.1 Checkpoints (read from `config.json` + safetensors headers)

| | DSpark (`RadixArk/Qwen3.8-27B-DSpark`) | DFlash2 (`z-lab/Qwen3.8-27B-DFlash2`) |
|---|---|---|
| params (own) | 1.857B bf16 | 1.924B bf16 |
| layers | 5, all `full_attention` | 5, all `sliding_attention`, window 2048 |
| width / MLP | d=5120, SwiGLU 17408 (same as the target) | same |
| attention | GQA 32 q / 8 kv heads, hd128, q/k RMSNorm, non-causal block | same, `is_causal: false` |
| RoPE | YaRN factor 32 over 8192 (θ=1e7) | default θ=1e7 |
| context conditioning | `fc` 25600→5120 over target layers [5,19,33,47,61] + `hidden_norm`; injected as K/V in every layer (DFlash "KV injection") | same |
| embed / lm_head | **shared from target** (no own weights; V=248,320) | shared |
| extra heads | Markov head (low-rank bigram: `w1` V×256 embedding + `w2` 256×V), confidence head (linear on [h; Markov emb]) | `candidate_selector` (top-16 path selector, rank 256 codebooks) + per-layer 2-tap dynamic convs (`attention_conv`, `mlp_conv`) |
| K at serve | 7 (`block_size` 7, trained with 16-wide supervision) | 7 (block 8) |
| trained/evaluated against | NVFP4 and FP8 targets (README) | bf16 Qwen3.8-27B |

Parameter split for DSpark: about 1.34B in the five MLPs (72%). Attention
projections take 0.26B, `fc` takes 0.13B, and Markov `w1`+`w2` take 0.13B.
The MLP dominates parameters, not time (section 2).

### 1.2 How `propose()` works (DSpark → `DFlashSpeculator.propose`)

1. The target forward returns aux hidden states at layers 5/19/33/47/61. These
   are concatenated and passed through `fc` (`combine_hidden_states`).
2. `precompute_and_store_context_kv` (`qwen3_dflash.py:587`) runs RMSNorm, then
   **one fused GEMM** producing K and V for all 5 layers from the same normalized
   feature h. That is followed by a grouped k-norm, RoPE, and a per-layer KV-cache
   write. This runs eagerly over every new target token, so the context K/V is a
   *linear function of one shared vector h*, `K_l = RoPE(knorm_l(W_k,l·h))`,
   `V_l = W_v,l·h`. It never depends on draft-layer outputs. (This is why
   cross-layer KV sharing or latent KV is natural here; section 5.)
3. Query block: the anchor plus 6 mask tokens (DSpark "anchor-as-first", 7
   rows/request) run through 5 decoder layers under a FULL CUDA graph. Each
   layer attends to [context KV ∪ block KV], non-causal.
4. `_sample_sequential` (`dspark/speculator.py:156`) runs one shared-lm_head
   GEMM for all 7 positions, then a **7-step sequential loop**: Markov embedding
   of the previous token, a 256→248,320 GEMV bias, then Gumbel sampling over the
   full vocabulary.
5. The target verifies 8 rows per request with PDV (qlen 8 costs the same as 6,
   `uq_kernel_bottlenecks.md` 3.1).

DFlash2 is the same up to step 3, plus the convolutions. Step 4 becomes one
parallel top-16 path selection.

---

## 2. Where draft wall time goes (Q2)

### 2.1 What is measured

Opt stack (dense draft + ret50k + mnbt2048), AgentX seed 42 **[measured]**,
`uq_agentx_cell_analysis.py table`:

| cell | srv tok/s | ITL avg / p50 ms | accept | running | steps / 340 s | ms per step |
|---|---|---|---|---|---|---|
| DSpark UQ opt c16 | 87.8 | 15.0 / 15.0 | 2.33 | 1.9 | 6517 | ~52 |
| DSpark UQ opt c32 | 162.2 | 36.4 / 32.4 | 2.65 | 5.6 | 3752 | ~91 |
| MTP UQ/PDV c32 | 232.3 | 16.3 / 14.1 | 3.07 | 3.9 | 6627 | ~51 |

With the opt stack, prefix hits are 92–97%, so DSpark c32 is now mostly
decode-bound. Per running request per step, DSpark spends about 16 ms against
13 ms for MTP and yields 2.65 vs 3.05 tokens. So the remaining **~1.4x
per-token gap to MTP splits about evenly between step cost (~1.23x) and accept
(~1.15x)**. (These steps still include mnbt2048 prefill chunks, so treat them
as upper bounds.)

### 2.2 Bandwidth floor per step **[model]**

`python3 draft_arch_cost_model.py --detail` (R=5.6, ctx=150k, weights at
5.5 TB/s, attention at 2.0 TB/s; PDV measures about 1.5 TB/s):

| stream | GB / step | ms floor |
|---|---|---|
| target weights (bf16 27B) | 54.0 | 9.8 |
| target UQ KV (16 full-attn layers) | 14.6 | 7.3 |
| target GDN state | 0.6 | 0.3 |
| **draft context KV read (5 layers bf16, full ctx)** | **17.2** | **8.6** |
| draft MLP weights (5 × 3 × 5120 × 17408) | 2.67 | 0.49 |
| shared lm_head (248k × 5120 bf16) | 2.54 | 0.46 |
| Markov W2 × 7 sequential steps | 0.89 | 0.16 (may sit in the 256 MB Infinity Cache) |
| draft attn projections | 0.52 | 0.10 |

- The draft is about 36% of the step floor (9.8 of 27.2 ms). **88% of the draft
  floor is its own context-KV read.** The MLP is 5%.
- The sequential Markov loop is cheap in bytes. It costs 7 × (embedding gather
  + GEMV + full-vocab Gumbel + logits-cache write) of dependent launches inside
  the graph. Expect launch/latency-bound time of about 0.3–1 ms **[hyp]**.
  DFlash2 reports +9.6% cycle latency for the DSpark correction on H200 [lit].
- At short context (R=2, 30k) the draft is about 15% of the floor and weights
  dominate. There, a thin MLP / FP8 draft weights / a reduced vocab matter more.
  AgentX is the long-context regime.

### 2.3 Fraction an architecture change can move

- Measured steps (about 91 ms at c32) are about 3x the floor (27 ms). The
  unattributed time is launch gaps, prefill chunks, aiter split-K sized from
  `max_model_len` (`dflash_dspark_agentx_bottlenecks.md` 2.5), and Python.
  Until P0-1 profiling exists, **an architecture change can remove at most the
  draft share: 25–36% of the floor** (attention BW 4.0 vs 2.0 TB/s). If the aiter
  draft decode runs well below 2 TB/s at 150k, the share is larger.
- Bounded-window drafts take the draft from 9.8 ms to about 1.3–1.7 ms
  **[model]**. At fixed accept that is a 20–30% faster step, roughly the step-cost
  half of the MTP gap.
- MLP width/depth alone: 3 of 5 layers saves about 0.25 ms of weights plus 40% of
  the KV (3.4 ms). Depth reduction helps only *because* it also drops KV layers.

---

## 3. Accept evidence relevant to architecture

**DSpark vs MTP per position on AgentX c32** (`server.log` SpecDecoding lines,
unconditional per-position rates; conditional = ratio of neighbours)
**[measured]**:

| pos | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| DSpark opt, mean of 34 intervals | 0.668 | 0.388 | 0.223 | 0.147 | 0.103 | 0.075 | 0.058 |
| DSpark conditional | 0.668 | 0.58 | 0.57 | 0.66 | 0.70 | 0.73 | 0.78 |
| MTP UQ/PDV (5 sample intervals) | ~0.74 | ~0.55 | ~0.40 | ~0.31 | ~0.24 | – | – |
| MTP conditional | ~0.74 | ~0.74 | ~0.73 | ~0.77 | ~0.79 | – | – |

- Position 0 has no intra-block dependency, so the 0.67 vs 0.74 gap is
  backbone quality/domain. AgentX is tool/agent traffic. DSpark's README suite
  is chat/math/code, where it reaches about 3.4.
- Positions 1–2 are where parallel drafts collide on modes. DSpark's Markov head
  only partly fixes this: 0.58 conditional vs MTP's autoregressive 0.73. This is
  exactly the gap DFlash2's selector/conv targets [lit].
- Positions 5–6 contribute about 0.13 tokens together. K=7 verify is "free" in
  PDV (qlen 7–8 cost the same as 6), so leave K, but do not spend draft capacity
  on the tail.

**UQ vs FP8 target KV, DSpark greedy probe** (`out/uq_accept_probe/`,
111k/167k/195k × gen 256/1024/2048) **[measured]**: UQ mean 3.72, FP8 4.38.
Two FP8 cells (167k, gen 1024/2048: 5.75, 6.29) look like repetitive
generations. Excluding them, UQ 3.81 vs FP8 3.91 (−2.6%). On AgentX, UQ c16 is
2.89 vs FP8 c16 2.68. **No evidence that UQ KV materially hurts draft accept.**

**DFlash2 vs DSpark on this target** (DFlash2 model card, H200, block 8)
[lit]: mean accept MTP 4.28, DSpark 3.62, DFlash2 4.80. That is +1.18 over DSpark
with a SWA-2048 draft. DFlash2 UQ here is stuck at 1.00 because of the SWA
null-block fault (H1), not because of the architecture.

---

## 4. Literature survey (what each idea buys *here*)

Columns: latency mechanism · expected accept trade · UQ-target fit · engineering
cost on this vLLM stack.

| work | latency mechanism | accept trade | UQ fit | cost here |
|---|---|---|---|---|
| Speculative sampling (Leviathan 2023; Chen 2023) | draft k tokens, verify in one target pass; lossless rejection sampling | baseline | neutral | in place |
| Medusa (Cai 2024) | K independent heads on the target's last hidden; no draft KV, no attention | low (no intra-draft dependency); Hydra adds sequential heads | perfect (no KV) | dominated: native MTP already exists and accepts more |
| Hydra (Ankner 2024) | Medusa + sequentially dependent heads | +~1 accept over Medusa | no KV | same idea as DSpark Markov; superseded |
| EAGLE / EAGLE-2 / EAGLE-3 (Li 2024–25) | 1-layer autoregressive feature drafter; EAGLE-3 multi-layer target-feature fusion + training-time test; trees | strong per step; sequential K passes | own (small) KV per layer; 1 layer | the native Qwen3.8 MTP path is effectively this; trees blocked (below) |
| Sequoia (Chen 2024) | hardware-aware optimal trees | more tokens per verify | – | **blocked**: PDV packs chains of qlen ≤ 8; GDN targets need state forking per branch |
| REST (He 2023) | retrieval drafts from a datastore | domain-dependent, free compute | neutral | orthogonal; agent traces repeat a lot (suffix/n-gram add-on, not an architecture) |
| Cascade Speculative Drafting (Chen 2023) | draft-of-draft cascade | – | – | not useful: the draft is already parallel/one pass |
| DistillSpec (Zhou 2023), Online SD (Liu 2023) | align draft to target distribution (TV/KL; on-policy) | +10–45% accept [lit] | **the right vehicle for UQ-aware + AgentX-domain alignment** | needs a training pipeline (external) |
| LayerSkip / Draft&Verify / SWIFT / CLaSp / ConfLayers (self-spec layer skipping/early exit) | draft with a subset of the target's layers | 1.3–1.7x total speedup [lit] | poor: each draft token still streams ~27 GB of bf16 target weights at half depth; GDN state must be forked | non-goal |
| QuantSpec (Tiwari 2025), QSpec (Zhao 2024) | self-spec with 4-bit KV / 4-bit activations; hierarchical INT4/INT8 KV | >90% per-token accept [lit] | conceptually UQ-adjacent (UQ is already the 4-bit KV) | sequential full-model drafts are too expensive vs a 1-pass parallel draft at bf16 weights |
| SpecMQuant / HierSpec (2025) | trees cost too much on W4 targets → sequence drafts | – | supports chain-not-tree on quantized targets | design guidance only |
| GliDe (Du 2024) | draft cross-attends to the **target's KV cache** | better than a vanilla draft [lit] | **best fit:** read UQ KV with the existing PDV-class kernel; zero draft KV | retrain + new attention module + kernel file |
| LongSpec (2025/26) | SWA-512 self-attention + cross-attention to target KV; shared embed/lm_head; long-context position fix | up to 3.26x on long context [lit] | best fit (same as GliDe, adds the window) | same as GliDe |
| MagicDec (Sadhukhan 2024) | draft with sparse/StreamingLLM KV budget (256–2049) at large batch/long ctx | small accept loss, large draft speedup [lit] | supports windowed draft KV | config-level analog = P0-A |
| FR-Spec (Zhao 2025) | prune draft lm_head to top 32k frequent tokens; training-free | ~−3% accept, −75% lm_head cost [lit] | neutral | **already supported**: DSpark reduced draft vocab + `d2t` exists (`qwen3_dspark.py`); needs weight surgery only |
| DFlash (Chen 2026) | block-diffusion parallel draft, KV injection into every layer; 3L vs 5L vs 8L τ 5.64/5.99/6.33, same speedup | depth ~6% accept per +2 layers [lit]; base drafts degrade >4k ctx, long-ctx finetune recovers (+68% at 16k) | features carry long range | in place |
| DSpark (Cheng 2026) | DFlash backbone + V-wide Markov head + confidence-scheduled verify; 2L DSpark > 5L DFlash; **production: 3 MoE layers, SWA 128, γ=5** | +15–18% over DFlash at γ=7 [lit] | – | in place; confidence head exists but adaptive verify is off |
| DFlare (Zhang 2026) | per-draft-layer learned fusion over 9 target layers, separate ctx K/V proj | +5–11% speedup over DFlash [lit] | neutral | retrain; separate ctx K/V proj **breaks** the "one shared h" trick |
| DFlash2 (Inco 2026) | 2-tap dynamic conv (+3% params, +0.7% latency) ≈ 15-layer quality; top-16 path selector (+0.6% latency) beats DSpark correction (+9.6%) | +1.18 over DSpark on Qwen3.8-27B [lit] | SWA 2048 → bounded draft KV | in place; blocked by H1 on UQ |
| CLA / YOCO / MLA (cross-layer KV sharing, latent KV) | share or compress KV across layers | small loss when trained in [lit] | draft context KV is already one linear map of h, so it fits naturally | retrain |

Could not locate a paper named "SKIPDRAFT". It is treated as the layer-skip
self-spec family above (SWIFT, CLaSp, ConfLayers).

---

## 5. Answers to the architecture questions (Q3)

**Shallower/wider vs deeper/narrower.** Depth costs KV, not just GEMMs: every
DSpark layer reads the full context. Literature: DFlash 3L loses about 6% τ vs
5L at equal speedup on H200 [lit]. DSpark 2L beats 5L DFlash [lit]. DFlash2 5L
plus conv approaches 15L [lit]. Recommendation: **3 layers + conv**. Narrower
width (d<5120) needs its own up-projection before the shared lm_head and breaks
reuse of `fc`/target embeddings for little gain. Shrinking MLP intermediate
(17408 → 12288) saves about 0.15 ms at c32 **[model]**. Not worth a retrain on
its own.

**Shared LM head / tied embeddings.** Already shared (DSpark and DFlash2 carry
no embed/lm_head). The lever left is a **reduced draft vocab** (FR-Spec): the
lm_head drops from 2.54 GB to 0.65 GB per step at 64k, and the Markov loop runs
over 64k instead of 248k. vLLM already supports `draft_vocab_size` + `d2t` for
DSpark. Markov `w2` rows must be sliced the same way. Gain ≈ 0.4–0.8 ms/step;
accept −1–3% [lit]. Cheap, low priority.

**Full vs SWA attention in the draft.** The strongest lever (sections 2.2 and
3). The draft gets long-range information through the target's hidden states,
and DFlash2 ships SWA 2048 with higher accept than full-attention DSpark.
Bounded draft KV also removes the pool problem: full-attention DSpark stores
20,480 B per context token against 17,408 B for the whole UQ target. That makes
FP8 draft KV / UQ-format draft KV (P2-1) unnecessary *if* P0-A holds. Risk: the
DFlash paper shows base drafters lose accept beyond 4k without long-ctx
finetuning. Windowing at inference may actually *help* a draft trained on short
contexts, since it caps out-of-distribution attention spans [hyp].

**Multi-head (Medusa) vs sequential Markov (DSpark) vs path selection
(DFlash2).** Independent heads collapse on modes. The DSpark Markov loop fixes
part of this, at 7 dependent full-vocab steps. DFlash2's selector scores
adjacent top-16 candidate pairs in parallel and walks a precomputed table. It
beats the Markov correction with 40x fewer parameters and 16x less latency
overhead [lit]. On AgentX the coherence gap is large (cond. accept 0.58 vs MTP
0.73), so this is worth having, and DFlash2 already has it.

**Tree drafts vs chain.** Non-goal for this target. PDV packs one chain of
qlen ≤ 8 into 16-row MFMA blocks. The 48 GDN layers would need recurrent
state forked per branch. SGLang serves this model with `eagle-topk 1`.
DFlash2's selector captures most "top-k" benefit inside a chain.

**Self-speculative using target early layers.** Non-goal. Weights are bf16:
any draft built from target layers streams about 27 GB per draft token
(half the layers), vs about 7 GB of weights for a whole DSpark parallel step. GDN state also
cannot be reused across skip patterns.

**UQ-aware distillation.** Measured UQ penalty is ≤3% (section 3), so the
standalone gain is small. Do it for free inside any finetune: capture target
features with UQ KV on (the "train how you serve" rule). It becomes **mandatory**
for idea 3, where the draft reads the UQ-dequantized target KV directly.

---

## 6. What a "UQ-native draft" would look like (Q4)

Architecture sketch (all shapes reuse existing target/draft dims):

- **Backbone:** 3 decoder layers, d=5120, SwiGLU 17408 (or 12288), FP8 W8A8
  linears. Initialize from DSpark/DFlash2 layers 0–2 + `fc` + norms.
- **Context path A (local):** keep DFlash KV injection from the shared feature h,
  but **SWA 2048–4096** in all 3 layers. Draft KV per request is bounded at about
  25–50 MB **[model]**. Pick the SWA page as a divisor of the 3136-token target
  page (P1-3 divisor policy) so the LCM stays 3136.
- **Context path B (long range, optional = idea 3):** in layer 1 only, a
  GliDe-style cross-attention whose K/V *are the target's UQ cache* for the last
  full-attention layer (layer 63, 4 kv heads × hd256 × 272 B/token). Draft query
  = 24 heads (GQA 6, matching PDV's packing), with the target's partial-RoPE
  convention. Read cost: 1088 B/token for one layer, vs 20,480 B/token today.
  Kernel: a **new** FlyDSL file cloned from PDV (non-causal, qlen 7, own LDS
  symbol), selected by an env/config gate. The shipping PDV is unchanged.
- **Intra-block:** DFlash2 2-tap dynamic depthwise conv before/after each
  sublayer. Top-16 low-rank path selector instead of the V-wide Markov loop.
  Keep the DSpark confidence head for later load-aware K.
- **Vocab:** draft vocab 64k with `d2t`.
- **Training recipe:** DSpark loss (0.1 CE + 0.9 TV, position-decay weighting,
  confidence BCE) or DFlash CE with random anchors, block 8, supervision width
  16. Target = Qwen3.8-27B **served with UQ KV** for feature capture, and for
  idea 3 the cross-attention reads dequantized UQ KV during training. Data: an
  AgentX-like agent/tool trace mix plus long-context (LongAlign-style 32k–200k)
  with fixed anchors per sequence, which keeps long-ctx training cost bounded
  (DFlash §5.4). Use on-policy target generations (DistillSpec/OSD) for the
  domain gap at position 0. Finetune from DFlash2 (best accept on this target),
  not from scratch.
- **Floor per step at R=5.6, 150k:** about 2.7 GB / 0.8 ms vs 23.8 GB / 9.8 ms
  today **[model]**. Draft context pool ≈ 0 B/token, so the pool is MTP class
  (~9M tokens).

---

## 7. Ranked recommendations

Rules for all items: new out dirs, never reuse a baseline `OUT_BASE`, do not
touch ports 8460–8470 / AgentX waiters, kernel/backend changes go in new files
behind config/env gates with defaults unchanged.

### P0: no retrain, this/next week

**P0-A. DSpark windowed-draft accept probe (cheapest validation).**
- Effort: ~2 h setup + about 1 h GPU per W. Risk: low (config only). Expected:
  tells whether rank-1 idea is free.
- Setup: new dirs `/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark-swa{W}/`,
  following the `DFlash2-nowin` pattern. Symlink `model.safetensors`,
  `dspark.py`, `dflash.py`, `README.md`. `config.json` = original plus
  `"use_sliding_window": true`, `"sliding_window": W`,
  `"layer_types": ["sliding_attention"]*5`, **`"is_causal": false`**.
- Spec JSON (new): `dspark_qwen38_uq/configs/uq_dspark_swa{W}.json` = `uq_dspark.json`
  with `"model"` pointed at the copy and
  `"attention_backend": "ROCM_AITER_FA_DFLASH_SWA"`.
- Run: idle server, c1, then `probe_accept_vs_length.py --ctx 111419,167145,194991
  --gen 256,1024,2048` → `out/uq_accept_probe/dspark_uq_swa{W}/`.
- W = 262144 (control), 32768, 8192, 2048.
- Pass criteria:
  - Control within ±0.2 of the existing `dspark_uq` probe (3.72 mean).
    Otherwise the window-table backend or the causal plumbing is wrong; stop
    and report to the H1 track.
  - W=8192 ≥ 95% of control → schedule an AgentX c32 opt-stack cell.
- Watch: SWA draft groups disable fine-grained prefix hits
  (`SlidingWindowManager`) and change the LCM. Judge the probe on accept, and
  judge AgentX on server tok/s plus zero-hit share.

**P0-B. Attribute a DSpark decode step (the gating profile).**
- This is `uq_kernel_bottlenecks.md` P0-1. Add these buckets: draft context-KV
  attention (aiter `fwd_decode`), draft GEMMs, shared lm_head, the Markov loop
  (7 iterations), context precompute, target PDV/GDN/GEMMs, gaps.
- Effort: half a day. It turns section 2's floors into ms and decides how much
  rank-1/2 is worth.

**P0-C. DFlash2 vs DSpark at matched config, once the H1/window-table fix lands.**
- Same probe and AgentX c32 opt stack. DFlash2 *is* the "SWA + conv + selector"
  architecture.
- If it reproduces its +1.2 accept over DSpark on UQ, it should become the
  default drafter and the base for any finetune.
- Effort: config only (the configs already exist: `uq_dflash2_swa.json`,
  `uq_dflash2_dense_swa.json`).

**P0-D. Read-only accept decomposition.**
- Effort: 1 h, already partly done in section 3.
- Per-position, conditional accept for DSpark vs MTP vs DFlash2 per AgentX cell
  and per probe. Keep it in `uq_agentx_cell_analysis.py accept`.
- Used to decide whether a finetune targets position 0 (domain) or positions
  1–2 (coherence).

### P1: small engineering or short finetune

| id | what | effort | risk | expected gain | minimal experiment |
|---|---|---|---|---|---|
| P1-A | **SWA finetune** of DFlash2 (or DSpark) on AgentX-like + long-ctx data with UQ-target feature capture, only if P0-A loses >5% | 1–2 GPU-days on 8 GPUs + external pipeline | medium (pipeline doesn't exist here) | recovers windowing loss; position-0 domain gap (+0.07 abs at pos0 ≈ +0.3 accept) **[hyp]** | finetune 1 epoch, run P0-A probe + AgentX c32 |
| P1-B | **Reduced draft vocab** (FR-Spec) for DSpark: slice `lm_head` from target rows, slice Markov `w2`, write `d2t`, set `draft_vocab_size` 65536 | 1 day (weight surgery script, new dir) | low | 0.4–0.8 ms/step; −1–3% accept [lit] | token frequencies from AgentX outputs; probe + one AgentX cell |
| P1-C | **FP8 draft weights** (W8A8 dynamic) | config if the draft quant path accepts it; else a small loader change | low | halves 3.2 GB of draft GEMM bytes (matters after P0-A) | accept probe (expect <1% change) |
| P1-D | **Enable confidence-head adaptive K** (DSpark already ships the head; `use_confidence_head` needs `enable_adaptive_verification`) | config + check | low/medium | fewer wasted verify rows at high c; DeepSeek reports large gains at strict SLAs [lit] | AgentX c32 opt cell with adaptive verify on |
| P1-E | **3-layer distillation** (from 5L) with conv + selector | 2–4 GPU-days | medium | −40% draft KV/GEMM if still full-attn; with SWA, mostly GEMM | only after P0-A/P0-B show the draft is still >10% of the step |

### P2: new architecture + kernels

| id | what | effort | risk | expected gain |
|---|---|---|---|---|
| P2-A | **UQ-native cross-attention draft** (section 6, path B): new draft model file, new FlyDSL non-causal draft-reads-target-UQ kernel (cloned from PDV, own LDS symbols, env-gated), retrain | weeks | high (training + kernel + KV-group plumbing: the draft forward needs the target layer-63 block table) | zero draft long-KV, long-range retrieval at 1088 B/token. Only worth it if P0-A shows windowing loses accept at 100k+ |
| P2-B | **UQ-format draft KV (existing P2-1)** | weeks | high | becomes **unnecessary** if P0-A passes; still useful if a full-attention draft is kept |
| P2-C | **Latent / cross-layer-shared draft context KV** (exploit K_l, V_l = linear(h)): cache one latent per token, absorb per-layer up-projections into queries (MLA-style, decoupled RoPE) | weeks + retrain | high | 3–17x smaller draft KV while keeping full context; a fallback if windowing fails and P2-A is too invasive |

---

## 8. Explicit non-goals

- **Thinning the draft MLP as the first move.** It is 5% of the draft floor at
  long context (section 2.2). Revisit only after the KV term is gone.
- **Tree drafts / Sequoia / EAGLE-2 dynamic trees.** PDV chain packing and GDN
  state forking make them a kernel + model-state project with HierSpec-style
  evidence that trees hurt on quantized targets.
- **Self-speculative layer skip / early exit (LayerSkip, SWIFT, CLaSp,
  QuantSpec-style self drafts).** bf16 27B weights per draft token are too
  expensive.
- **Medusa/Hydra heads.** The native MTP head dominates them on this model.
- **A standalone UQ-aware distillation run.** The measured UQ penalty is within
  noise. Fold it into P1-A.
- **K > 7.** PDV qlen 8 is the packing limit, and the tail positions add about
  0.06 tokens each on AgentX.
- **Building UQ-format/FP8 draft KV before P0-A.** It optimizes bytes that a
  windowed draft never reads.
- **Training from scratch.** Finetune DFlash2 (best accept on this target) or
  DSpark.
- **Editing shipping kernels** (`pdv_flydsl_decode_hd256*.py`,
  `rocm_aiter_fa.py`, `ultraquant_attn.py`, `rocm_aiter_fa_uq_compat.py`).

---

## 9. Open items

- P0-B profile is the gating measurement for every percentage in section 2.
- The window-table SWA backend has no completed cell. P0-A's control doubles as
  its first test on a known-good draft.
- H3 (aiter `(W-1, 0)` window makes the non-causal block causal) affects any SWA
  draft. It needs the window-table backend to pass `num_query-1` as the right
  window, in the new backend file only.
- MTP per-position numbers in section 3 come from five sampled log intervals.
  Recompute them over the whole profile window with the cell-analysis helper.

---

## 10. References

- Leviathan, Kalman, Matias. Fast Inference from Transformers via Speculative Decoding. arXiv:2211.17192.
- Chen et al. Accelerating LLM Decoding with Speculative Sampling. arXiv:2302.01318.
- Cai et al. Medusa. arXiv:2401.10774. · Ankner et al. Hydra. arXiv:2402.05109.
- Li et al. EAGLE arXiv:2401.15077; EAGLE-2 arXiv:2406.16858; EAGLE-3 arXiv:2503.01840.
- Chen et al. Sequoia. arXiv:2402.12374. · He et al. REST. arXiv:2311.08252. · Chen et al. Cascade Speculative Drafting. arXiv:2312.11462.
- Zhou et al. DistillSpec. arXiv:2310.08461. · Liu et al. Online Speculative Decoding. arXiv:2310.07177.
- Elhoushi et al. LayerSkip. arXiv:2404.16710. · Zhang et al. Draft & Verify. arXiv:2309.08168. · Xia et al. SWIFT. arXiv:2410.06916. · Chen et al. CLaSp (ACL 2025). · Amer et al. ConfLayers. arXiv:2604.14612.
- Du et al. GliDe with a CaPE. arXiv:2402.02082. · LongSpec. arXiv:2502.17421.
- Sadhukhan et al. MagicDec. arXiv:2408.11049.
- Zhao et al. FR-Spec. arXiv:2502.14856.
- Tiwari et al. QuantSpec. arXiv:2502.10424. · Zhao et al. QSpec. arXiv:2410.11305. · Speculative Decoding Meets Quantization (HierSpec). arXiv:2505.22179.
- Chen, Liang, Liu. DFlash. arXiv:2602.06036. · Cheng et al. DSpark. arXiv:2607.05147. · Zhang et al. DFlare. arXiv:2606.02091.
- Inco AI. DFlash 2: Keep Drafting Parallel. https://inco.ai/blog/dflash2/ (Aug 2026), and the `z-lab/Qwen3.8-27B-DFlash2` model card.
- Brandon et al. Cross-Layer Attention. arXiv:2405.12981. · Sun et al. YOCO. arXiv:2405.05254. · DeepSeek-V2 (MLA). arXiv:2405.04434.
