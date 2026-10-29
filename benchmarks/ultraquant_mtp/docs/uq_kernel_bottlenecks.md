# UltraQuant + DFlash/DSpark: remaining bottlenecks in code and kernels

Date: 2026-10-22. Read-only code and artifact review. No GPU was touched, no server was
started or stopped, and no existing file was edited. This is the only file written.

Builds on (read these first):
`dflash_dspark_agentx_bottlenecks.md`, `uq_dflash_dspark_bottlenecks.md`, `p0_status.md`.
Where this note disagrees with them, the section says so and cites the artifact.

Paths are relative to `/home/preshah/workspace/vllm` unless they start with `aiter/`
(`/home/preshah/workspace/aiter`) or `logs/` (`/home/preshah/workspace/logs`).

Evidence labels used below:

- **[measured]**: read from an existing artifact in `out/` or `logs/`.
- **[code]**: read directly from source. The behaviour follows from the code.
- **[est]**: arithmetic from measured numbers. Not directly profiled.
- **[hyp]**: hypothesis. Needs the listed test.

---

## 1. Executive summary: top 5 after config fixes

"Config fixes" means `--block-size 64`, `--prefix-cache-retention-interval 50176` and
`--max-num-batched-tokens 2048`. They are assumed applied or about to be.

Measured effect of the config fixes (aiperf tok/s at c32, from `out/*/sa/seed42_c32`):

| config | tok/s | accept | hit % | zero-hit events | note |
|---|---|---|---|---|---|
| DSpark UQ baseline | 17.4 | 2.78 | 31 | 11 of 20 | |
| DSpark UQ ret50k | **68.4** | 2.78 | 86 | 4 of 51 | zero-hit still 40% of miss tokens |
| DFlash UQ baseline | 12.3 | 1.00 | 50 | | |
| DFlash UQ ret50k | 22.5 | 1.00 | 82 | 1 of 36 | accept still 1.00 |
| DFlash UQ mnbt2048 | 31.8 | 1.00 | 93 | 0 of 37 | |

DSpark UQ is still about 2.4x below MTP UQ in server tok/s (98.7 against 232).
DFlash is stuck on accept.

1. **DFlash accept = 0.000 per position. This is a correctness fault in the sliding-window
   draft read, not tuning. It caps DFlash at one token per verify step.**
   - Existing probe data show it is deterministic and **context-length gated**, which
     contradicts the "not length-dependent" claim in `uq_dflash_dspark_bottlenecks.md` Q2.
     See section 4.1.
   - Best code-grounded hypothesis: the SWA draft reads the **null block (block 0)**. Block 0 is
     bytes shared with Mamba state in the overlaid KV pool. The aiter decode kernel does not
     window-mask V loads. Details and a decisive test are in section 4.1.
   - Expected gain if the hypothesis holds: DFlash accept 1.0 goes to roughly 2.7–4.8, about
     2.5–4x decode throughput **[est]**.

2. **Verify steps that share a batch with prefill lose the PDV kernel.**
   - In `TurboQuantAttentionImpl.forward` (`turboquant_attn.py:595-613`), the mixed-batch
     decode slice is built with `max_query_len=1`.
   - The UQ spec branch gates PDV on `2 <= qlen <= 8` (`ultraquant_attn.py:496`). So every
     verify request in a mixed step falls to `_split_multi_query_decode` (`ultraquant_attn.py:614`).
   - That runs one single-query decode per verify row: 8 KV scans per request instead of 1.
   - Recompute storms are exactly the mixed regime (~79% of wall time on DSpark UQ c32).
   - **[measured]** microbench: PDV 506 µs against 1044 µs for the per-row production path
     (4 reqs, q6, 131k). That is 2.06x on the attention kernel.
   - This is a dispatcher fix, not a kernel fix (P0-2).

3. **Residual zero-hit recompute (100k+ tokens) runs through an inefficient continuation prefill.**
   - Per chunk, per layer, per request it dequantises the whole cached prefix to bf16. It then
     copies it again into a concat buffer and runs dense FA2.
   - That is O(N²/chunk) work. It gets worse exactly when `--max-num-batched-tokens` is lowered.
   - It also issues two blocking H2D `torch.tensor(..., device=)` calls per layer per request
     (`ultraquant_attn.py:930-931`). Each one is a stream sync.
   - Rough size is 1–4% of a long prefill at 8192 chunks and 4–15% at 2048 **[est]** (section 5.3).
   - The bigger lever remains avoiding zero-hit (E3 hit logger).

4. **DSpark/DFlash decode-only step is ~110 ms against ~30–50 ms for MTP, and nobody has a
   kernel breakdown of it.**
   - The 110 ms is derived from logs in the prior note, not profiled.
   - Known pieces: PDV is only about 10 ms of it **[est]** (section 3.2). The bf16 draft KV read
     is another ~6 ms **[est]**. The rest is unattributed.
   - This is the largest unmeasured item. The ranking of everything else below depends on it
     (P0-1).

5. **The draft KV is bf16 and there is no FP8 or UQ-format draft read path.**
   - `ROCM_AITER_FA` + fp8 KV fails in aiter: `is_fp8([q, k_cache, v_cache])` raises "Mixed FP8
     and non-FP8 tensors" because Q is bf16
     (`aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py:203-216`,
     reached from `mha_v3.py:415`).
   - So DSpark's full-attention draft reads ~4096 B/token/layer over the whole context every step.
   - Dense backend `ROCM_AITER_FA_UQ_DENSE` fixes pool utilisation (31% to 94%). It does not
     reduce bytes read.

Not in the top 5, on purpose: **PDV kernel micro-tuning.** Every experiment variant is within
noise end to end, and the ceiling is about 6–10% of the decode step (section 3.2).

---

## 2. Code and kernel map

### 2.1 Target (UQ) path

| stage | file:line | what it does |
|---|---|---|
| backend select | `vllm/v1/attention/backends/ultraquant_spec_compat_attn.py` (`ULTRAQUANT_SPEC_COMPAT`) | prefers block 64; kernel block set is TurboQuant's |
| metadata builder | `ultraquant_attn.py:72-165`, `:83` | `reorder_batch_threshold = 1 + 2·K`: parallel_drafting is True for dflash/dspark (`config/speculative.py:1473-1474`, threshold from `attention/backend.py:626-643`). That is **15 for K=7** while verify is 8 rows. Requests with q ≤ 15 count as decodes |
| shared forward | `turboquant_attn.py:525-668` (inherited) | pure decode, pure prefill, or **mixed** (`:595-661`) |
| verify decode | `ultraquant_attn.py:478-527` | PDV when `2 ≤ qlen ≤ 8` and GQA = 6 |
| verify fallback | `ultraquant_attn.py:529-614` | pair / shared-KV (default off) / **`_split_multi_query_decode`** per-row |
| decode launcher | `vllm/v1/attention/ops/flydsl_ultraquant_decode.py:337-420`, `:956-1020` | builds the PDV module via `pdv_experiment.load_pdv_module()`. Grid is `(num_reqs, Hk, partitions)` |
| PDV kernel (shipping) | `vllm/v1/attention/ops/flydsl_kernels/pdv_flydsl_decode_hd256.py` (LDS `pdv_flydsl_hd256_smem_*`) | one wave64 per (seq, kv_head, partition). All verify rows are packed into 16-row MFMAs |
| experiment dispatch | `flydsl_kernels/pdv_experiment.py` (`VLLM_ULTRAQUANT_PDV_EXPERIMENT`) | keys: hoist_k, qblock_waves, pdv2*, fp4pv, pdv2_tr, pdv2_iglp, pdv2_32x32, dflash |
| prefill, first chunk | `ultraquant_attn.py:712-721` | `flash_attn_varlen` |
| continuation prefill | `ultraquant_attn.py:751-826` (Python loop per request), `:828-952` | `q_len ≤ 128`: per-row Triton decode (`:796-811`). Otherwise full-prefix dequant (`triton_dequant.py`) plus concat plus FA2 |
| dequant kernel | `vllm/v1/attention/ops/ultraquant/triton_dequant.py:46-190` | grid `(alloc_len, B·Hk)`: one program per token per head |

### 2.2 Draft path (DFlash / DSpark, model runner v2)

| stage | file:line | what it does |
|---|---|---|
| speculator | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` (DSpark subclasses it in `dspark/speculator.py`) | one forward over 1+K (DFlash) or K (DSpark) query rows per request |
| context KV precompute | `speculator.py:424-428` → `model_executor/models/qwen3_dflash.py:587-662` | GEMM + RoPE over **all** new target tokens, then `do_kv_cache_update` per layer. Out-of-window positions are PAD-slotted (`speculator.py:573-585`) but still computed |
| input prep | `speculator.py:487-688` | null-block guards at `:573` and `:605` |
| block table | `vllm/v1/worker/gpu/block_table.py:119-139`, `:184-` | kernel-block split `b·bpk+k` (null block 0 becomes `0..bpk-1`) |
| draft FA backend (default UQ) | `rocm_aiter_fa_uq_compat.py:58-76` | caps draft page at 256. Inherits SWA and non-causal logic from `rocm_aiter_fa.py` |
| draft FA backend (dense, new) | `rocm_aiter_fa_uq_dense.py` | caps at 784. LCM 3136, util 94% |
| non-causal decode call | `rocm_aiter_fa.py:1390-1419` | `aiter…mha_v3.flash_attn_with_kvcache(causal=False, window_size=(W-1, 0), page_table=…)` |
| aiter kernel | `aiter/…/flash_attn_triton_amd/fwd_decode.py:261-686` | paged split-K. Loops **all** blocks to `cache_seqlens` (`:502-512`) |
| fp8 draft KV | `rocm_aiter_fa.py:992-1004` | passes `k_descale`/`v_descale` with `q_descale=None`, so aiter raises |
| hybrid hit logic | `vllm/v1/core/kv_cache_coordinator.py:843-976`; `single_type_kv_cache_manager.py:971-1066` (SWA), `:1488-1565` (Mamba), `:1567-1624` (retention mask) | fixed-point over groups. Retention interval 0 keeps only the replay-boundary tail |
| working-tree coordinator change | `kv_cache_coordinator.py:152-165` | `drop_eagle_checkpoint_block` only when an eagle group is full attention. DSpark gets it. DFlash (SWA draft) does not |
| KV pool aliasing | `vllm/v1/worker/utils.py:389-455` (`allocate_kv_cache`) | "Cache groups overlay each other, so tensors may address the same bytes" |

### 2.3 What the existing experiments are (PDV)

All variants use unique LDS symbols. I checked every `global_sym_name`.

| key | file | change | status |
|---|---|---|---|
| (default) | `pdv_flydsl_decode_hd256.py` | shipping | |
| `hoist_k` | `…_hoist_k.py` | K fragments and scales read once per tile for all NT query blocks | real |
| `qblock_waves` | `…_qblock_waves.py` | one wave per 16-row query block | real |
| `pdv2` / `pdv2g` / `pdv2p` | `…_pdv2_fp4q.py` | FP4 Q, FP4×FP4 scaled QK, bf16 PV | real; three keys alias one module |
| `pdv2_fp4pv` / `fp4pv` | `…_pdv2_fp4pv.py` | plus FP8-MFMA PV | real |
| `pdv2_tr` | `…_pdv2_tr.py` | copy of fp4pv, tr4 TODO | scaffold |
| `pdv2_iglp` | `…_pdv2_iglp.py` | identical to fp4q | **stub** |
| `pdv2_32x32` | `…_pdv2_32x32.py` | identical to fp4q | **stub** |
| `pdv2_ctrl` | `…_pdv2_ctrl.py` | shipping math, new symbol | control |
| `dflash` | `dflash_flydsl_decode_hd256.py` | re-exports shipping | alias |

---

## 3. PDV / FlyDSL decode: where the time goes

### 3.1 Measured kernel facts (`logs/pdv_bottleneck_rocprof/*`, `logs/pdv_bottleneck_summary.json`)

32 requests, 131072 context, single layer, MI355X **[measured]**:

| verify tokens q | rows (6q) | 16-row MFMA blocks (NT) | kernel µs | effective GB/s over the quantised KV |
|---|---|---|---|---|
| 2 | 12 | 1 | 1598 | ~2850 |
| 4 | 24 | 2 | 2425 | ~1880 |
| 6 | 36 | 3 | 3051 | ~1500 |
| 8 | 48 | 3 | 3175 | ~1440 |

The KV bytes are 4.56 GB per dispatch (32 × 131072 × 4 heads × 272 B).

- Time steps with NT. It is not linear in q. **q = 7 and q = 8 cost the same as q = 6.**
- Context scaling is linear: q6 at 32k is 1182 µs, at 131k 3051 µs, at 262k 5532 µs.
- The kernel is at ~1.5 TB/s of ~8 TB/s peak. It is not bandwidth-bound.
  - Resources: VGPR 128 + AccVGPR 384 (512 total), so one wave per SIMD. LDS 21.5 KB.
  - VALU:MFMA instruction ratio is about 11.5:1 (649M against 57M).
  - The K=7 geometry is "free" relative to K=5, which settles the prior note's open
    question on K=7 verify cost.
- Matched comparison, 4 reqs, q6, 131k: PDV 506 µs, pair 733 µs, FP8 521 µs, per-row
  production 1044 µs.

### 3.2 What that means end to end **[est]**

- Per request, per layer at 131k: 3051/32 ≈ 95 µs. Times 16 full-attention layers is about
  1.5 ms per request-step at 131k, about 1.75 ms at 150k.
- At 6 running requests that is ~10 ms per decode step. DSpark decode-only steps are ~110 ms.
  MTP UQ steps are ~50 ms (ITL 16.3 ms × accept 3.07, `pdv_stream1_sa` c32). The 110 ms figure
  is carried over from the earlier note, not re-profiled.
- **Ceiling:** a 2x faster PDV kernel buys at most about 5–10% of a step.
- End-to-end experiment results agree (`out/pdv_exp_*/sa/seed42_c32`, `uq_agentx_cell_analysis.py table`):

  | variant | tok/s | ITL ms | accept |
  |---|---|---|---|
  | PDV baseline | 232 srv | 16.3 | 3.07 |
  | hoist_k | 260 | 16.6 | 3.36 |
  | pdv2 | 235 | 17.5 | 3.15 |
  | fp4pv | 245 | 17.4 | 3.20 |
  | qblock_waves | 208 srv | 21.2 | 2.86 |

  ITL is flat within about ±8% for hoist_k, pdv2 and fp4pv (16.6–17.5 against 16.3 ms). The
  tok/s spread tracks accept (noise), not step time. qblock_waves is a real regression
  (ITL 21.2 ms, +30%).
- Decode-heavy (i1k/o8k, c32, `out/pdv_decode_heavy/*/result.json`):

  | variant | output tok/s |
  |---|---|
  | FP8 | 315.7 |
  | PDV | 344.2 |
  | hoist_k | 336.5 |

  hoist_k is not better than shipping.

### 3.3 Conclusion on PDV

- Further PDV tuning (iglp, 32x32, tr4) is **P2**. It will not move AgentX.
- The only PDV-adjacent things that matter are:
  - Make sure verify rows actually reach PDV in mixed batches (section 5.2).
  - Keep q ≤ 8, which already holds at K=7.
- Do **not** reduce K to save attention. q4 saves 20% of the kernel but costs 14% of the
  expected accept length, using the AgentX per-position accepts 0.69, 0.44, 0.25, 0.17,
  0.10, 0.06, 0.05.

---

## 4. Evidence linking empirics to code

### 4.1 DFlash accept collapse

Existing artifacts:

- `out/dflash_accept_bisect/window/accept_vs_length.jsonl` (FP8 target, window draft).
- `out/uq_accept_probe/{dflash_uq,dspark_uq}/probe.txt`.

| probe | ctx tokens | accept | per-position accept |
|---|---|---|---|
| DFlash window, fresh server | 883, gen 512 to 4096 | 5.2–6.2 | healthy |
| DFlash window | 27,880 | **1.00** | 0.000 ×7 |
| DFlash window | 111,419 | **1.00** | 0.000 ×7 |
| DFlash UQ | 111k, 167k, 195k | 1.00–1.015 | 0.000–0.010 |
| DSpark UQ | 111k–195k | 3.1–4.6 | healthy |

- It is deterministic, independent of prefix-hit (0 and 0.98 both fail), and independent of
  generation length once the prompt is long.
- It is exactly 0.000 at **all** positions, including position 0. That looks like garbage
  logits or NaN, not a quality loss.
- Only the SWA draft shows it. DSpark has a full-attention draft and never visits null blocks.
- The long-generation 883-ctx run crossed the window (~5k tokens) and stayed healthy.
  So "more than 2048 tokens" alone is not the trigger.
- **The decisive bisect never ran.** `out/dflash_accept_bisect/nowin/server.log` ends in
  `FileNotFoundError: …Qwen3.8-27B-DFlash2-nowin/model.safetensors`. The no-window draft has
  no weight symlink. P0-2 in the earlier note is therefore still untested.

**Hypothesis H1 (null-block alias poison) [hyp], consistent with every datum above:**

1. The SWA manager replaces out-of-window blocks with `block_pool.null_block`, which is block
   ID 0 (`block_pool.py:185`). The draft block table therefore holds zeros for the old
   context, and `cache_seqlens` still spans the whole context.
2. `fwd_decode.py:502-512` iterates every block up to `cache_seqlens` and loads K and V for
   null blocks (`:580-581`). Only the sequence, split and tile masks gate the loads
   (`:557-562`). **The window is applied to QK only** (`:195-209`, `:210`).
3. A masked score gives p = 0, but `tl.dot(p, v)` is `0 × V`. If the V bytes in block 0 are
   NaN or Inf as bf16, the product is NaN. The softmax guard (`valid = m_i_new > -inf`,
   `:232`) does not protect V.
4. Block 0's bytes are **not** exclusive to the draft. `allocate_kv_cache`
   (`worker/utils.py:389-455`) overlays all groups on one buffer. Layer *l* of block *b*
   lives at `offset + l·layer_stride + b·block_stride`. The draft's layer *l* aliases the
   same bytes as layer *l* of the target and Mamba groups.
5. Mamba state is written through block 0. `block_table.py:184-196` says explicitly that dummy
   runs "write mamba state to the reserved null block". The align-mode prefill also leaves
   null gaps between states (`single_type_kv_cache_manager.py:1636-1640`). fp32 state bytes
   reinterpreted as bf16 pairs can decode to NaN/Inf (any bf16 with an all-ones exponent, about
   1 in 256 random patterns per element), so a large exposed region is likely poisoned
   **[hyp; the byte-level claim is arithmetic, not observed]**. The "exposure" percentages
   below (3.9%, 31%, 100%) are my estimate of how much of block 0's tile the draft layout
   covers, also **[hyp]**.

This explains the observed pattern:

- Short prompts fine (no null reads).
- Long prompts always bad.
- "Session flip" (depends on what was written into block 0 before the draft first read it).
- GPQA decay over time (null blocks appear as outputs grow, after block 0 is first written).
- UQ 6 of 6 bad.
- FP8 baseline 2 of 6 good. Its 32-token draft pages expose 3.9% of block 0's tile. P0-1
  832-token pages expose 100%, and P0-1 DFlash is 0 of 6 good (section 4.3). UQ 256-token
  pages expose 31%.
- DSpark healthy (never reads null).

**Not established.** This fits, but I did not read a poisoned block. Section 7 gives a
read-only counter test and an A/B that proves or kills it in one server start.

Related lower-confidence hypotheses:

- **H2:** a query block that straddles an unallocated next draft block reads null for its own
  K/V. The slot is PAD-guarded at `speculator.py:605`. This would hit the 256-token UQ draft
  6.4x more often than the FP8 1648-token one. That is a candidate for the UQ-only DSpark c8
  "pos ≥ 2 == 0" anomaly (15 of 31 intervals) **[hyp]**. It needs a per-step draft-block
  boundary log.
- **H3:** the window tuple `(W-1, 0)` makes the 8-query block causal while the draft is
  declared non-causal (`rocm_aiter_fa.py:1409` against `dflash.py` `requires_non_causal`). Small
  effect, not the collapse.
- **H4 (int32 stride overflow)** is **unlikely**. `fwd_decode.py:349-375` casts every stride
  to int64. The earlier note's "overflow at block ~1270" does not apply to this aiter copy.

### 4.2 Prefix-miss storm and why `ret50k` worked

- With `prefix_cache_retention_interval = 0`, `reachable_block_mask` keeps only the
  replay-boundary tails for SWA (`single_type_kv_cache_manager.py:1105-1146`) and a single
  Mamba state (`:1590-1622`). The fixed-point hit search (`kv_cache_coordinator.py:843-976`)
  needs all groups to agree at one aligned boundary. If that single checkpoint is evicted, the
  hit is zero. Retention 50176 (4 × 12544) adds a checkpoint every 50k tokens, so a lost tail
  degrades to a partial hit.
- Measured: DSpark UQ zero-hit events 11 of 20 become 4 of 51. Miss k/req 95 becomes 20.
  DFlash UQ zero-hit 10 of 24 becomes 1 of 36.
- ret25k is not better than ret50k (54.0 against 68.4 aip, hit 85.1 against 86.2%). Do not go
  finer without E3 data.
- DSpark ret50k still loses 40% of its miss tokens to the 4 zero-hit events. These are the
  remaining storms.

### 4.3 Why DFlash FP8 P0-1 regressed while DSpark FP8 P0-1 improved

All six P0-1 DFlash cells have accept = 1.00 **[measured]**:

| cell | accept |
|---|---|
| p01 c8 | 1.00 |
| p01 c16 | 1.00 |
| p01 c20 | 1.00 |
| p01 c24 | 1.00 |
| p01 c30 | 1.00 |
| p01 c32 | 1.00 |

- The baseline had c20 and c24 at 2.69 (whole cell good). The geometry fix turned the
  coin-flip into a constant.
- Baseline: 2 of 6 good. UQ: 0 of 6, and 0 of 12 with the ret and mnbt runs. P0-1: 0 of 6.
- Draft SWA page sizes: 32 tokens (3.9% of page used), 256 (31%), 832 (100%). Good-cell rate
  falls in that order. Under H1, exposure of block 0's tile to the draft read is what changes.
- Prefix hit also fell ~20 points versus baseline at matching concurrency:

  | conc | p01 hit % | baseline hit % |
  |---|---|---|
  | c16 | 68.0 | 88.2 |
  | c20 | 57.7 | 79.5 |
  | c24 | 55.4 | 75.6 |

  - Not explained by code reading. The retained tail is now 4 × 832-token blocks (3328
    tokens) instead of 65 × padded 32-token blocks.
  - It is confounded: accept 1.0 plus 8-row verify gives ITL 567 ms and TTFT 74 s, which
    raises concurrent prefill pressure.
  - n is small (12 admission events at c32). E3 (per-group hit logger) is the right tool.
- DSpark FP8 P0-1 improved because its draft is full attention. It never reads null blocks.
  Accept held at 2.54–3.03 across all six cells.
  - The geometry fix therefore showed up cleanly: hit 47% to 75% at c32, 6.8 to 26.3 aip tok/s.
  - Its remaining gap is the same recompute storm (6 zero-hit events of 26).

Net: P0-1 is a good geometry change for DSpark and a no-op or negative for DFlash until the
SWA read path is fixed. Do not judge DFlash geometry fixes until accept is fixed.

### 4.4 Why `mnbt2048` and `ret50k` helped differently

- `mnbt2048` raised DFlash UQ hit to 92.6% and tok/s to 31.8 (steps 1302 to 2272, miss k/req
  71 to 11, zero-hit events 10 of 24 to 0 of 37).
- On DSpark it did not help (11.8 aip tok/s, hit 48.3%, 3.7 waiting, miss 71 k/req).
- This is consistent with chunk size interacting with the mamba-align split and the 12544 LCM.
  Chunks below one block cannot checkpoint inside a chunk.
- It is also exactly where continuation prefill cost rises (section 5.3). Combine it with the
  dense draft, not with finer retention, until E3 data exist.

---

## 5. Ranked improvements

Rules for every item (workspace rule): new module or new file, an explicit env or config
gate, default environment keeps today's path, unique LDS symbol names for any FlyDSL file,
and a thin dispatcher branch only. Gains are **[est]** unless a measured number is cited.

### P0 (do first; each is small or decisive)

**P0-1. Profile one DSpark UQ decode-only step. No code change.**

- Use the existing runner `benchmarks/ultraquant_mtp/run_ffdsl_pair_rocprof_e2e.sh` pattern
  (`rocprofv3 --kernel-trace --stats`) on a free HIP device, one request pool at ~150k ctx,
  6 running. Compare with MTP UQ on the same pool.
- Goal: attribute the ~110 ms (draft layers, 7 sequential Markov samples plus lm_head,
  GDN × 48, GEMMs, graph gaps). Everything in P1 is ranked on guesses until this exists.
- Gain: decision value. Effort: hours.

**P0-2. Mixed-batch verify keeps PDV.**

- Problem: section 1 item 2 (`turboquant_attn.py:603-613`, `ultraquant_attn.py:496`).
- New file `vllm/v1/attention/backends/ultraquant_mixed_verify.py`: a helper that computes the
  decode slice's real max query length from `query_start_loc_cpu[:num_decodes+1]` (no GPU
  sync) and passes it down. Hook it with a thin branch in
  `UltraQuantAttentionImpl._decode_attention`.
- Gate: `VLLM_ULTRAQUANT_MIXED_VERIFY=pdv` (default empty = today).
- PDV already reads `query_start_loc` on device, so ragged decode rows are fine.
  Add a one-time log of the chosen path so the A/B is verifiable.
- Note a related edge: threshold 15 classifies q 9–15 requests as decodes. A pure-decode batch
  containing one q = 12 short-extend loses PDV for the whole batch (`qlen > 8`). Cap the
  classification at 8 under the same gate.
- Expected gain: attention part of mixed steps about 2x (506 against 1044 µs measured at 4
  reqs). Translates to 3–8% on prefill-heavy cells. More once recompute steps are frequent.
- A/B: same cell with `VLLM_ULTRAQUANT_MIXED_VERIFY=pdv`. Compare ITL p50 in steps with running ≥ 4
  and prefill tokens > 0. Unit check: PDV against per-row on a synthetic mixed batch (cosine 1.0
  expected, as in the existing correctness smoke).

**P0-3. DFlash null-block test, then window-table backend.**

- Test first (read-only counter, section 7). Then build:
- New backend `vllm/v1/attention/backends/rocm_aiter_fa_dflash_swa.py`
  (`ROCM_AITER_FA_DFLASH_SWA`), subclassing `AiterFlashAttentionUQCompatBackend`. Add an enum
  entry in `registry.py` like its siblings.
- New op `vllm/v1/attention/ops/dflash_swa_window_table.py`: a small Triton gather kernel
  that builds a fixed-width `[num_reqs, ceil((W+qlen)/B)+1]` window page table.
  - Per request: `start_blk = max(0, (S - qlen - (W-1)) // B)`.
  - It writes `cache_seqlens_win = S - start_blk·B`.
  - Fixed width keeps it CUDA-graph safe. DFlash runs FULL graphs when the backend supports
    UNIFORM_BATCH (`rocm_aiter_fa.py:456`).
- The impl calls the **same** aiter `flash_attn_with_kvcache` with `(page_table_win,
  cache_seqlens_win)`. The window is positional, so shifting q and k equally preserves it.
- Never visits null blocks, never scans masked context, and also fixes H3 by passing
  `(W-1, qlen-1)` if desired.
- Gate: speculative-config `"attention_backend": "ROCM_AITER_FA_DFLASH_SWA"`, or
  `VLLM_DFLASH_DRAFT_ATTN_EXPERIMENT=win_table` read by a dispatcher in the compat backend.
  Default stays `ROCM_AITER_FA_UQ_COMPAT` / `ROCM_AITER_FA`.
- Expected gain: if H1 holds, DFlash accept goes from 1.0 to roughly 2.7–4.8, about 2.5–4x
  decode throughput. DFlash UQ ret50k 22.5 aip could reach DSpark-class 50–70 **[est]**.
  Regardless of H1, draft attention bytes at 140k drop ~70x (saves ~5–15 ms per draft step
  at c32 **[est]**).
- A/B: `docs/probe_accept_vs_length.py` on `ctx 1024,32768,131072 × gen 512,2560`.
  Pass condition: accept at 131k within 10% of ctx-1k, on both FP8 and UQ.
  Then AgentX c32.
- Also run the missing no-window bisect (create the safetensors symlink). It is a free
  confirmation.

**P0-4. Run the already-queued combination: UQ DENSE draft + ret50k + mnbt2048 (config only).**

- `ROCM_AITER_FA_UQ_DENSE` smoke passed (LCM 3136). The AgentX dense cell has not been measured.
  Do ret50k with it, since LCM 3136 makes `50176 = 16 × 3136` valid.
- Watch the aiter paged kernel handle a 784-token page (not a multiple of 64).
  `fwd_decode.py:557-562` masks partial tiles, so it should pass. Confirm with the existing
  accept probe (DSpark UQ accept must stay at 2.7–3.5).
- Expected: DSpark UQ pool ~2.17M to ~4.4M tokens. Fewer evictions at c ≥ 24.

### P1

**P1-1. Continuation prefill v2 (new module).**

- New file `vllm/v1/attention/ops/ultraquant/continuation_prefill_v2.py` plus a thin branch in
  `UltraQuantAttentionImpl._prefill_attention`.
- Gate: `VLLM_ULTRAQUANT_CONT_PREFILL=v2`.
- Changes, in order of value:
  1. **Persistent `cu_seqlens` tensors** instead of `torch.tensor(..., device=)` at
     `ultraquant_attn.py:930-931`. Each call is a blocking H2D copy: 2 per layer per request
     per step, so ~32 stream syncs per prefilling request per step. The first-chunk path
     already has a `_cu_2` buffer.
  2. **Dequant straight into the `[L, Hk, D]` concat buffer** with a vectorised, multi-token
     per-program Triton kernel. Today: dequant to `[1, Hk, L, D]` (1.1 KB read, 4 KB write per
     token per layer), then transpose-copy into `k_full` (`:882-883`), another 4 KB read and
     4 KB write. That is ~3 passes where 1 is enough, and the kernel uses one program per token
     per head with byte-granular loads (`triton_dequant.py:96-103`).
  3. **Context-aware threshold** replacing the fixed `_CONTINUATION_DECODE_THRESHOLD = 128`
     (`turboquant_attn.py:84`, used at `ultraquant_attn.py:796`). The `q_len ≤ 128` path scans
     the full KV once per query row. At L = 100k that is ~110 MB per row per layer. The dequant
     path is O(L) once. Crossover is at roughly 8–16 rows at long context **[est]**, not 128.
- Size, honestly: dequant plus copy cost is ~5–10 ms per chunk per request at L ≈ 75k
  **[est]**. That is 1–4% of a long prefill at 8192 chunks and 4–15% at 2048 chunks. The
  sync and threshold fixes matter more for small chunks and for tails.
- Expected gain: 3–10% of prefill-step wall time at mnbt 2048. Effort: 1–2 days.
- A/B: new rocprof of one 8192-token continuation step at L = 100k (kernel time for
  dequant plus copy plus FA2), then AgentX c32 `*_cont_v2`.

**P1-2. UQ-aware mixed-verify and spec-as-decode threshold audit (config of the same item as P0-2).**

- After P0-2, check the `reorder_batch_threshold` of 15 for DFlash/DSpark. Requests with
  q 9–15 are decodes but PDV needs q ≤ 8.
- If E3 shows common short tails, cap the classification to 8 under the P0-2 gate.

**P1-3. FP8 draft KV via a new backend (unblocks `*_kv8`).**

- New backend `rocm_aiter_fa_kv8.py` (`ROCM_AITER_FA_KV8`) subclassing the compat backend.
  - Quantise Q per head to FP8 with `q_descale`, **or**
  - dequantise the touched KV window into bf16 scratch and call the bf16 aiter path.
- aiter requires `is_fp8([q, k_cache, v_cache])` to be uniform (`utils.py:203-216`). Today
  `rocm_aiter_fa.py:992-1004` passes `k_descale`, `v_descale` with `q_descale=None`.
- Gate: speculative-config `"attention_backend": "ROCM_AITER_FA_KV8"` with `"kv_cache_dtype": "fp8"`.
- Expected gain: halves draft KV bytes and draft pool. Most valuable for DSpark full attention
  (2.9 GB/request/step at 140k). Needs an accept re-check. Effort: 1–2 days.

**P1-4. Per-group hit logger (E3).** Still the cheapest way to separate the residual zero-hit
causes (mamba filter, SWA tail, draft eviction). New file `vllm/v1/core/kv_hit_debug.py`,
enabled by `VLLM_KV_HIT_DEBUG=1`, called from a thin hook in
`kv_cache_manager.get_computed_blocks`. It would also settle the p01 hit-rate drop for DFlash
(section 4.3).

**P1-5. Group-size and SWA block policies (opt-in)** from the earlier note (P1-3, P1-4). They
remain valid for DFlash after accept is fixed.

### P2

- **P2-1. UQ-format draft KV and a FlyDSL UQ draft decode kernel** (non-causal multi-query,
  hd128, 8 kv heads × 136 B). This is the long-term answer for DSpark's full-context draft
  read: 4096 to 1088 B/token/layer, page equal to target page, LCM 3136. Large effort.
  - New kernel `flydsl_kernels/dflash_flydsl_uq_draft_hd128.py` with its own LDS symbol.
  - Select with `VLLM_ULTRAQUANT_PDV_EXPERIMENT=dflash_uq_draft` plus draft
    `"kv_cache_dtype": "ultraquant_4bit"`.
  - Only after P0-1 shows the draft read is worth it.
- **P2-2. Fused UQ-aware continuation-prefill attention** (read FP4 in the FA tile loop, no
  bf16 prefix materialisation). Removes the whole dequant pass. Large effort, 3–10% of long
  prefills. Only after P1-1 data.
- **P2-3. PDV micro-tuning** (iglp, 32x32, tr4, occupancy). Ceiling 5–10% of the step.
  Stubs `pdv2_iglp` and `pdv2_32x32` exist. Do not prioritise.
- **P2-4. Skip draft context GEMM for out-of-window rows (DFlash).** `precompute_and_store_context_kv`
  computes K/V for the whole chunk then discards 75% of rows to PAD slots (`speculator.py:573-585`).
  Cost is ~15–30 ms per 150k recompute **[est]**, so low value.

---

## 6. How to A/B each item (exact selectors)

| item | selector | baseline | metric |
|---|---|---|---|
| P0-2 | `VLLM_ULTRAQUANT_MIXED_VERIFY=pdv` | unset | ITL p50 in mixed steps; one-time path log |
| P0-3 | spec JSON `"attention_backend": "ROCM_AITER_FA_DFLASH_SWA"` or `VLLM_DFLASH_DRAFT_ATTN_EXPERIMENT=win_table` | `ROCM_AITER_FA_UQ_COMPAT` | `probe_accept_vs_length.py` accept at 131k; AgentX c32 |
| P0-4 | spec JSON `"attention_backend": "ROCM_AITER_FA_UQ_DENSE"` plus `--prefix-cache-retention-interval 50176 --max-num-batched-tokens 2048` | compat + ret50k | KV pool tokens, hit %, srv tok/s |
| P1-1 | `VLLM_ULTRAQUANT_CONT_PREFILL=v2` | unset | rocprof of a 8192 × L=100k continuation step; `iteration_tokens_total` |
| P1-3 | spec JSON `"attention_backend": "ROCM_AITER_FA_KV8"`, `"kv_cache_dtype": "fp8"` | bf16 draft | accept per position; draft pool |
| P1-4 | `VLLM_KV_HIT_DEBUG=1` | off | which group has the argmin hit |
| P2-1 | `VLLM_ULTRAQUANT_PDV_EXPERIMENT=dflash_uq_draft` | unset | accept; draft read time |

Rules: new out dirs (`*_<tag>`), same seed 42 and concurrencies, report `server tok/s` next to
aiperf tok/s (aiperf drops unfinished requests), read the profile block of
`server_metrics_export.json`, and use `docs/uq_agentx_cell_analysis.py table|events|accept`.

---

## 7. Cheap decisive tests (read-only or one server start)

1. **H1 counter (no behaviour change).**
   - New file `benchmarks/ultraquant_mtp/debug_null_block_finite.py` plus an env-gated hook
     `VLLM_DFLASH_DEBUG_NULL_BLOCK=1` called once after the first long prefill and once every
     N draft steps.
   - It reports, per draft layer, `torch.isfinite(kv_cache[layer][0]).all()` over the exposed
     page region, plus the count of non-finite elements.
   - Expect non-finite in layers that alias GDN layers 0, 1, 2 and 4, and finite in layer 3 (a
     target full-attention layer, assuming standard Qwen3.8 layer ordering — verify from
     `kv_cache_config.kv_cache_tensors`).
   - Run it on an idle server only. It reads no live request.
2. **H1 A/B.** Same server, same prompt, with `VLLM_DFLASH_NULL_BLOCK_ZERO=1`, which zeroes
   block 0 of each draft layer before every draft forward. Accept recovering at 131k proves H1.
   Accept staying at 0 kills it, and the window-table backend then tests H2/H3/other.
3. **The missing no-window bisect.** Add the `model.safetensors` symlink in the `nowin`
   directory and rerun `run_dflash_accept_bisect.sh`.
4. **Context-length scan.** `docs/probe_accept_vs_length.py --ctx 1024,2100,4096,8192,16384,32768`
   on a fresh server, FP8 and UQ. It finds the onset between "no null blocks" and "null
   blocks present". That is stronger than the AgentX-cell argument.
5. **Standalone aiter reproducer.** Call `flash_attn_with_kvcache` on a page table whose first
   N entries are 0 and whose block 0 contains NaN, with `window_size=(2047, 0)`. Check for NaN
   output. This needs no vLLM server and runs in minutes.

---

## 8. Explicit non-goals

- Do not edit shipping files: `pdv_flydsl_decode_hd256.py`, `ultraquant_attn.py` kernels,
  `rocm_aiter_fa.py`, `rocm_aiter_fa_uq_compat.py`, aiter's `fwd_decode.py`, `mha_v3.py`.
  Every item above is a new module plus a gate and a thin dispatcher branch.
- Do not tune PDV kernels (hoist_k, qblock_waves, pdv2 family, iglp, 32x32) for AgentX.
  The ceiling is about 5–10% of the step and e2e is flat within ±5% **[measured]**.
- Do not change K or the verify width to save attention. q = 6, 7, 8 cost the same
  kernel time. Only q ≤ 4 and q ≤ 2 are cheaper, and accept loses more than the saving.
- Do not re-run `--block-size 64`, MNBT, or retention as "new" fixes. They are measured. Only
  new combinations are in scope (P0-4).
- Do not go below ret50k (ret25k was not better) before E3 data.
- Do not judge DFlash geometry changes (P0-1 style) until accept is fixed (section 4.3).
- Do not trust 2-token `dflash-*cliff*` probes for accept, as the earlier note already says.
- Do not touch ports 8410–8453, kill running jobs, or reset GPUs.
- Out of scope here: MTP UQ internals, FP8-only fixes, GDN/Mamba kernel work, model-quality
  effects of 4-bit KV on accept (section P2-1 only flags it).

---

## 9. Open items

- DFlash null-block hypothesis (H1) is consistent with all collected data but unproven.
- No kernel-level breakdown of a DSpark/DFlash decode-only step exists. The ~110 ms figure is
  derived (P0-1).
- The unexplained ~20 point prefix-hit drop for DFlash FP8 P0-1 versus baseline (section 4.3).
- DSpark UQ c8 pos ≥ 2 == 0 (H2) is untested.
- The dequant and continuation-prefill cost numbers in P1-1 are arithmetic estimates, not
  rocprof measurements.
