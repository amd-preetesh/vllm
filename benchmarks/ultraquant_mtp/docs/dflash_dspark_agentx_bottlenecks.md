# DFlash2 / DSpark on AgentX (Qwen3.8-27B, MI355X): bottleneck diagnosis + fix plan

Date: 2026-10-21. Read-only investigation; no shipping code changed.
Helper scripts (this directory):

- `parse_spec_server_log.py`: running/waiting/KV%/hit/accept from `server.log`
- `kv_geometry_calc.py`: offline mirror of hybrid KV block geometry (reproduces logged group sizes exactly)
- `probe_accept_vs_length.py`: acceptance vs context length and generated length (replaces the 2-token "OK" probes)

## 1. Executive diagnosis

AgentX tok/s dies from two independent faults. Neither one is capacity pressure.
There are zero preemptions in every cell, average waiting is about 1–3, and
average KV usage is 6–41%.

1. **Prefix-cache reuse collapses: 96% for MTP vs 31–69% for DFlash/DSpark.**
   Every agent turn then re-prefills tens of thousands of tokens. Those 8k-token
   prefill chunks get co-scheduled with decode, so ITL rises to 230–430 ms
   (MTP is 20 ms) and TTFT rises to 21–69 s (MTP is 2.8 s). The root cause is
   KV block geometry, not memory size:
   - The draft adds a 5-layer bucket. The FA draft backend drops the common
     block base from 64 to 16, so the mamba-aligned target block becomes
     1648 = 16·103.
   - The DFlash sliding-window draft picks a 32-token block, which doesn't
     divide the target page. Its page is padded to the full ~3.4 MB target page,
     so it is 3.9% utilized on FP8 and 31% on UQ.
   - The DSpark full-attention bf16 draft has twice the target's bytes per
     token. That doubles the target block and pads mamba pages 2×.
   - The prefix-cache hit granularity (scheduler LCM) grows from 1664 to 3296
     on FP8 and from 3072 to 12544 on UQ.
   - Group size falls to 5 layers, giving 15 groups. Target attention gets 25%
     padding, and mamba has 10 groups instead of 3.
   - Net effect: every prefill churns many full-size pages through the free
     list and evicts other trajectories' cached prefixes. That is the
     "eviction + redo prefill" you suspected. It shows up as prefix-cache
     eviction, not preemption.
2. **DFlash acceptance is broken once the sequence outgrows the 2048-token
   sliding window.** This is a correctness bug, not a modeling limit.
   - DFlash UQ: GSM accept is 4.80, but GPQA decays from 4.27 to 1.0 within
     about 2 minutes as outputs grow. AgentX is 1.00 at both c8 and c32.
   - DFlash FP8: GPQA is fine (3.07), but AgentX (~140k) is 1.19.
   - DSpark (full-attention draft, same target backends) holds 2.7–3.5
     everywhere.
   - So the fault is in the DFlash sliding-window draft path: null/evicted
     window blocks, the padded page, or the windowed decode.

Fixing (1) alone should move DSpark to roughly the MTP regime. DFlash needs
(1) and (2).

## 2. Evidence

### 2.1 Profile-phase metrics (aiperf `server_metrics_export.json`, profile block only)

The cumulative "Prefix cache hit rate" in `server.log` includes warmup, which
is all misses. Use the export instead.

| cell (c32) | prefix hit | accept len | KV avg / max | waiting avg | preempt | ITL ms | TTFT s | tok/s |
|---|---|---|---|---|---|---|---|---|
| MTP FP8 (`pdv_stream1_sa/qwen3-8-27b_fp8_mtp5_tp1_c32`) | **96.0%** | 3.44 | 12 / 23% | 0.0 | 0 | 20.5 | 2.8 | 263 |
| PDV UQ (`…ultraquant_pdv_mtp5_tp1_c32`) | **94.3%** | 3.07 | 7 / 13% | 0.06 | 0 | 16.3 | 2.7 | 190 |
| DFlash FP8 (`dflash_qwen38_fp8/sa/seed42_c32`) | 69.4% | **1.19** | 31 / 49% | 0.96 | 0 | 339 | 21.4 | 11.8 |
| DFlash UQ (`dflash_qwen38_uq/sa/seed42_c32`) | 50.4% | **1.00** | 16 / 26% | 1.32 | 0 | 298 | 22.5 | 12.3 |
| DSpark FP8 (`dspark_qwen38_fp8/sa/seed42_c32`) | 46.8% | 2.78 | 33 / 54% | 2.74 | 0 | 429 | 68.8 | 6.8 |
| DSpark UQ (`dspark_qwen38_uq/sa/seed42_c32`) | 31.1% | 2.78 | 41 / 73% | 2.11 | 0 | 231 | 43.5 | 17.4 |

Recomputed tokens per request (queries × miss / requests) are about 5.5k for
MTP versus about 43k for DFlash FP8. That is 8× more prefill per turn.

### 2.2 Warmup eviction is visible directly (`server.log` stat lines)

aiperf warmup prefills about 3.84M tokens, one trajectory at a time.

- **MTP:** the first profile minute hits 44% cumulative, and measured prompt
  throughput drops to 1–3k tok/s.
- **DFlash/DSpark:** the first profile minute hits only 1–12%. They re-prefill
  at 10–22k tok/s.
- **KV usage:** one solo ~140k request uses 1.5% of the pool on MTP versus
  6.6% on DFlash FP8 and 3.3% on DSpark FP8. So the cached-prefix footprint
  per trajectory is 2–4× larger.

### 2.3 Block geometry (from `server.log`, reproduced by `kv_geometry_calc.py`)

| cell | `kv cache group sizes` | `kv lcm block sizes` | draft page util |
|---|---|---|---|
| MTP FP8 | [1664]×4 | 1664 | n/a |
| DFlash FP8 | [1648]×14 + [32] | 3296 | 3.9% (32 tok in a 1648-tok page) |
| DSpark FP8 | [1648]×10 + [3296]×4 + [1648] | 3296 | 100%, but mamba page padded 2× |
| DFlash UQ | [3136]×14 + [256] | 12544 | 30.7% |
| DSpark UQ | [3136]×14 + [256] | 12544 | 30.7% (full-attention draft: draft is about 80% of all KV) |

All four cells also log "Add 4 padding layers, may waste at most 25.00% KV
cache memory". DFlash additionally logs "Disabling fine-grained prefix-cache
hits … SlidingWindowManager".

Code path:

- `vllm/platforms/interface.py::_preferred_block_size_for_backends`: with
  {UNIFIED_ATTN, FA} the common base is 16 (MTP alone gets 64). Then
  `_align_hybrid_block_size` gives 16·cdiv(mamba, 16·2048) = 1648.
- `vllm/model_executor/layers/attention/attention.py::get_kv_cache_spec` (SWA
  branch): `_largest_kernel_block_within` picks 32 on FA and 256 on
  FA_UQ_COMPAT. These are the largest unsplit kernel blocks, but they don't
  divide the page.
- `vllm/v1/core/kv_cache_utils.py::unify_kv_cache_spec_page_size`: a
  non-divisible page triggers padding (`page_size_padded`).
- `vllm/v1/core/kv_cache_utils.py::_get_kv_cache_groups_uniform_page_size`:
  `group_size = min_num_layers = 5`.
- `vllm/v1/core/kv_cache_utils.py::resolve_kv_cache_block_sizes`: the
  scheduler block is the LCM of the group blocks.

### 2.4 DFlash acceptance breaks with sequence length on the sliding-window path

| run | accept len | per-pos (last) |
|---|---|---|
| DFlash FP8 short c32 (`dflash_qwen38_fp8_short_tput`) | 5.05 | 0.92…0.34 |
| DFlash FP8 GSM / GPQA | 4.80 / 3.07 | GPQA 0.79…0.20 |
| DFlash UQ GSM / GPQA | 4.80 / **1.08** | GPQA **0.000×7** |
| DSpark UQ GSM / GPQA | 4.88 / 3.52 | healthy |
| DFlash UQ AgentX c8 (stream-2, in progress) | **1.00** | KV max 6%, waiting 0 |
| DSpark FP8 AgentX c8 (stream-2, in progress) | 2.70 | |

- **DFlash UQ GPQA over time:** 4.27 → 3.16 → 1.95 → 1.3 → 1.0 over roughly
  the first 120 s. That window is 32 requests generating to about 2–4k
  tokens, which crosses the 2048 window.
- **Short-prompt run:** it uses the identical padded layout (1648/32) and
  accepts 5.05, so padding alone doesn't corrupt the cache. The trigger is
  positions outside the window, where block-table entries become null
  block 0, plus the backend specifics.
- **Existing probes are not evidence.** The `dflash-*-cliff*.log` and
  `dflash-ab-*.log` probes produce 2 tokens ("OK"), so `rate=0.143` is one
  draft step. They say nothing about acceptance.

### 2.5 Draft-attention cost scales with full context, not the window

- `rocm_aiter_fa.py` (non-causal multi-token decode) calls
  `aiter…mha_v3.flash_attn_with_kvcache(window_size=(2047, 0), page_table=…)`.
- In `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_decode.py`,
  the paged loop iterates all `ceil(cache_seqlen / BLOCK_SIZE_K)` blocks and
  masks out-of-window positions to -inf after computing QKᵀ. There is no
  window-start skip.
- `get_split_k` is sized from `block_table.shape[1] * block_size` (=
  `max_model_len`), not the real length.
- So a DFlash draft step does full-context work (about 4,375 blocks at 140k
  with 32-token blocks) for a 2048-token window.
- Also, the right window is 0 with `causal=False`. That makes the 8-query
  block causal, whereas DFlash2 is `is_causal: false`. Small effect, but it
  is a semantic mismatch.

### 2.6 Things that are NOT the problem (with evidence)

- **Preemption / recompute:** `num_preemptions` = 0 in every cell, and
  `request_num_preemptions` sum = 0.
- **Waiting for capacity:** waiting averages 1–3, and MTP has the same
  arrival pattern. AgentX at "c32" keeps only about 2–10 requests in flight
  because of trajectory pacing.
- **Raw KV capacity for DFlash UQ:** 6.3M tokens is plenty. It still misses
  50% because of the 12544 LCM and draft-page churn.

### 2.7 Infra note (stream-2)

Early DFlash FP8 cells on HIP 1 failed with stranded VRAM / a bad
`ROCR_VISIBLE_DEVICES` pin (`No CUDA GPUs are available`). After a GPU1 reset,
role A recovered: c8 completed and later concs continued. Streams b/c also
kept measuring (c8 aborted mid-flight by the reset but still wrote exports).

## 3. Ranked fixes

The rule followed throughout: new kernel or module files plus an env or config
switch, with defaults unchanged. Config-only items need no code.

### P0-1: Fix FP8 block geometry (config only, no code)

- **DFlash FP8 and DSpark FP8:** add `--block-size 64` to the serve line.
  Predicted (calculator):
  - Target block goes 1648 → 1664.
  - The DFlash bf16 sliding-window draft scales 32 → 832 with no padding
    (3.9% → 100% page utilization).
  - The LCM goes 3296 → 1664, matching MTP.
- **DSpark FP8:** also set the draft `"kv_cache_dtype": "fp8"` in
  `benchmarks/ultraquant_mtp/dspark_qwen38/configs/` (new file
  `fp8_dspark_kv8.json`; keep the old one). This keeps the draft page equal to
  the target page, which removes the 2× mamba padding and the 3296 target
  block. DFlash FP8 can try the same as a second variant
  (`fp8_dflash2_kv8.json`).
- **Runners:** add `DFLASH_BLOCK_SIZE` and `DSPARK_BLOCK_SIZE` env knobs to
  the runner scripts, defaulting to empty (today's behavior).
- **Verify in `server.log`:**
  - `kv cache group sizes` has no 32 or 3296 entries.
  - `kv lcm block sizes 1664`.
  - Solo-request KV% during warmup is about 1.5–2%.
- **Expected impact:**
  - Profile-phase hit goes from 47–69% to over 90%.
  - TTFT goes from 21–69 s to 3–5 s.
  - ITL goes from 300+ ms to 25–40 ms.
  - DSpark FP8 at accept 2.78 should land at roughly 150–230 tok/s (estimate
    scaled from MTP at 3.44 → 263).
  - DFlash FP8 is still capped by its acceptance (P0-2).

### P0-2: Root-cause the DFlash sliding-window acceptance collapse (diagnose first, then fix in new files)

Run `probe_accept_vs_length.py` on an idle server with
`--ctx 1024,32768,131072 --gen 512,1536,2560,4096` for DFlash FP8 and UQ.
Treat the hypotheses as ordered diagnostics:

1. **Sliding-window eviction path.** Run a draft-config copy with the window
   disabled: `/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2-nowin/`
   (symlink weights; `config.json` sets `use_sliding_window: false`,
   `layer_types: full_attention×5`, `sliding_window: null`). If UQ GPQA and
   the probe recover past 2k tokens, the bug is in the sliding-window
   machinery, not the model.
2. **Null-block / padded-page reads in `mha_v3` paged decode on the UQ
   compat layout (LBNHC + padded 256-token page).** Run the same probe with a
   new backend that keeps the draft page unpadded (P1-2). If accept recovers,
   it is a padded-stride bug for blocks beyond the window.
3. **Address overflow for high block IDs.** The padded page stride is about
   1.7M elements per block, which overflows int32 around block ID 1,270. This
   only matters after the pool is fully cycled, as in AgentX warmup. Test:
   DFlash FP8 probe at 131k after a warm-up that cycles the pool, compared
   with a fresh server.
4. **Right window 0 under non-causal.** Fix in the new kernel (P1-1) by
   passing `(window-1, num_query-1)`. Expect a small gain, not the collapse
   fix.

The fix lands in the new draft decode module from P1-1, not in
`rocm_aiter_fa.py` or aiter.

Expected impact: DFlash accept at long context goes from 1.0–1.2 back toward
its GPQA level (about 3). Combined with P0-1, that suggests DFlash FP8 in the
200+ tok/s range (estimate).

### P1-1: Window-aware draft decode kernel (new module + backend)

- **New kernel:** `vllm/v1/attention/ops/dflash_swa_decode_triton.py`, a paged
  non-causal multi-query decode.
  - The loop starts at `max(0, seqlen - window - q_len)` and never touches
    null blocks.
  - Split-K is sized from the real `seq_lens` max (CPU upper bound already
    exists: `seq_lens_cpu_upper_bound`).
  - It supports MultipleOf(16) block sizes and padded page strides with
    int64 addressing.
- **New backend:** `vllm/v1/attention/backends/rocm_aiter_fa_dflash_swa.py`
  (`ROCM_AITER_FA_DFLASH_SWA`, registered in
  `v1/attention/backends/registry.py`). It subclasses FA and overrides only
  the non-causal decode branch.
- **Selection:** draft `"attention_backend": "ROCM_AITER_FA_DFLASH_SWA"` in a
  new config JSON. An env alternative is
  `VLLM_DFLASH_DRAFT_ATTN_EXPERIMENT=swa_skip`. Default stays `ROCM_AITER_FA`.
- **Expected impact:** draft attention work at 140k drops about 70× (2048
  versus 140k keys per layer). That is a few ms per step at c32. It is also
  the vehicle for the P0-2 fix.

### P1-2: UQ geometry, dense draft page (new backend)

- **New backend:** `vllm/v1/attention/backends/rocm_aiter_fa_uq_dense.py`
  (`ROCM_AITER_FA_UQ_DENSE`). It doesn't edit `rocm_aiter_fa_uq_compat.py`.
  - It declares `MultipleOf(16)` kernel blocks so the draft can run a
    784-token page unsplit through the P1-1 kernel.
  - Its `customize_spec` selects the largest draft block that divides the
    target block (the `lcm_pad` policy).
- **Calculator result:**
  - Draft block 256 → 784, page utilization 31% → 94%.
  - LCM 12544 → 3136, matching the shipping UQ hit granularity.
  - DSpark UQ draft memory drops about 3×.
- **Selection:** the new spec JSONs `uq_dflash2_dense.json` and
  `uq_dspark_dense.json` set `"attention_backend": "ROCM_AITER_FA_UQ_DENSE"`.
- **Expected impact:** UQ hit goes from 31–50% to over 90%. DSpark UQ
  capacity per request improves about 2–3×.

### P1-3: Divisor-aware sliding-window block choice (opt-in)

- In `attention.py::get_kv_cache_spec` (SWA branch), add a thin branch behind
  `VLLM_SWA_BLOCK_POLICY=divisor` (default `largest` = today). It picks the
  largest kernel block whose page divides the shared page.
- This makes any page-compatible draft dtype scale instead of pad. For a
  UQ-format draft, the calculator gives 256 padded → 3136 dense.
- It is a non-kernel control-plane change, so a thin dispatcher branch is
  acceptable.

### P1-4: Hybrid group sizing (opt-in)

- In `kv_cache_utils.py::_get_kv_cache_groups_uniform_page_size`, add
  `VLLM_KV_GROUP_SIZE_POLICY=target_min`. It computes `group_size` from the
  target buckets only (16) and pads the small draft bucket instead.
- Result: target attention becomes 1 group with no 25% padding, mamba goes
  from 10 groups to 3, and the draft becomes 1 group padded to 16 layer-slots.
- This is good for DFlash, whose sliding-window draft holds a few blocks per
  request. It is bad for full-attention DSpark (11/16 of the draft group would
  be wasted per context token), so enable it only for sliding-window drafts.
- Expected impact: about 20% more cacheable tokens and about 3× less
  mamba-block churn per prefill chunk.

### P2-1: UltraQuant-format draft KV (new FlyDSL kernel)

- The draft's 8 heads × 136 B equals the target's 4 heads × 272 B, which is
  1088 B/token. So the page is identical, the draft block equals the target
  block (3136), the LCM is 3136, and the draft KV is about 4× smaller than
  bf16.
- **New kernel:**
  `vllm/v1/attention/ops/flydsl_kernels/dflash_flydsl_uq_draft_hd128.py`, a
  non-causal multi-query hd128 decode reading UQ codes. It has its own LDS
  symbol names.
- **Selection:** `VLLM_ULTRAQUANT_PDV_EXPERIMENT=dflash_uq_draft` via
  `pdv_experiment.py` plus draft `"kv_cache_dtype": "ultraquant_4bit"`.
- This is the long-term UQ answer, especially for DSpark, whose draft KV
  covers the full context.
- Acceptance must be re-measured, because 4-bit draft KV may cost accept.

### P2-2: Scheduler guard rails while misses persist

- Use `--long-prefill-token-threshold` / `--max-num-partial-prefills 1` (or
  `--max-num-batched-tokens 4096`) to cap decode stalls behind re-prefill.
  This trades TTFT for ITL.
- Only as a temporary mitigation; P0-1 removes the cause.

### P2-3: DSpark K at long context

- AgentX per-position acceptance is 0.69, 0.44, 0.25, 0.17, 0.10, 0.06, 0.05.
  K=5 keeps about 97% of the expected accepted length (2.65 vs 2.75) and
  verifies 25% fewer tokens.
- Try this only after P0-1, as `num_speculative_tokens: 5` in a new config.

## 4. What to measure (per fix)

Always report from the profile block of `server_metrics_export.json`, not
cumulative `server.log`. `parse_spec_server_log.py` only gives the time series.

| fix | proof metric | target |
|---|---|---|
| P0-1, P1-2, P1-3 | `kv cache group sizes`, `kv lcm block sizes`, no "padding layers"/padded draft in startup log | LCM = target block (1664 FP8 / 3136 UQ) |
| P0-1, P1-2 | solo-request KV% in warmup; `GPU KV cache size` | ≈ MTP (1.5–2%) |
| P0-1, P1-2 | profile `prefix_cache_hits / prefix_cache_queries`; first-minute hit after warmup | ≥ 90%; ≥ 40% within 60 s |
| P0-1 | recomputed tokens per request = (queries − hits) / requests | ~5–8k (MTP 5.5k) |
| P0-1, P2-2 | TTFT avg / ITL avg (aiperf) | ≤ 5 s / ≤ 40 ms |
| P0-2 | `probe_accept_vs_length.py` accept_len at max_tokens 512 → 4096, ctx 1k → 131k | flat (no drop past 2048) |
| P0-2 | DFlash UQ GPQA accept time series | stays ≥ 3 after minute 2 |
| P1-1 | draft attention time per step (rocprof on `flash_attn_with_kvcache` vs new kernel) at ctx 140k | ≥ 10× lower |
| all | `num_preemptions`, `num_requests_waiting_by_reason{capacity}` | stay 0 / ~0 (regression guard) |
| end-to-end | AgentX c8/c32 tok/s vs MTP FP8 263 / PDV 190 | DSpark ≥ 150, DFlash ≥ 150 after P0-2 |

Run order on one free GPU (do not touch 8410/8420/8430):

1. Start the DFlash FP8 server with `--block-size 64` and check the startup
   geometry. It takes about 3 minutes.
2. `probe_accept_vs_length.py` on DFlash FP8 and UQ, plus the no-window
   draft copy (P0-2 bisect).
3. AgentX c32 for DSpark FP8 with `--block-size 64` and an fp8 draft KV. This
   is the fastest proof of P0-1.

## 5. What NOT to do

- **Don't tune for preemption or capacity.** Preemptions are 0, so raising
  `--gpu-memory-utilization`, lowering `--max-num-seqs`, or adding swap won't
  touch the real cause.
- **Don't disable prefix caching to "avoid eviction".** MTP's 263 tok/s
  depends on 96% hits. Also, `DFLASH_NO_PREFIX_CACHE` was a fault workaround,
  not a perf knob.
- **Don't set `--prefix-cache-retention-interval` to dense (None).** It grows
  the sliding-window and mamba cache footprint and makes eviction worse. The
  default 0 is right.
- **Don't read acceptance from the `dflash-*cliff*` / `dflash-ab-*` probes**
  (2-token outputs). Use `probe_accept_vs_length.py`.
- **Don't use `server.log`'s cumulative "Prefix cache hit rate" for AgentX.**
  Warmup (all misses) halves it.
- **Don't compare short-prompt c32 (2359 tok/s) with AgentX.** Different
  regime (no prefix reuse, ISL in the tens).
- **Don't edit shipping files:** `rocm_aiter_fa.py`, aiter's `mha_v3` /
  `fwd_decode.py`, `ultraquant_attn.py`, `pdv_flydsl_decode_hd256.py`,
  `rocm_aiter_fa_uq_compat.py`. Add new backends or kernels and select them
  via config or env.
- **Don't tune K, temperature, or `draft_sample_method` for DFlash before
  P0-2.** Accept is 0.000 per position, which is a bug and not a tuning
  issue.
- **Don't run probes against servers that are serving AgentX.**
  `/metrics` counters are global.
