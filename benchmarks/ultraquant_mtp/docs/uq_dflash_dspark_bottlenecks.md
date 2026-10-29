# UltraQuant (UQ) DFlash2 / DSpark bottlenecks on AgentX

Qwen3.8-27B, MI355X (gfx950), vLLM TP=1, InferenceX `inferencex-agentx-mvp`,
seed 42, 300 s measure window. Companion to (and not a replacement for)
`dflash_dspark_agentx_bottlenecks.md`; read that first for the shared
hybrid-KV background. This note covers **UQ-only or UQ-worse** effects.

Status when written: `out/dflash_qwen38_uq` has c8/16/20/24/30/32 complete;
`out/dspark_qwen38_uq` has c8/16/32 complete, **c20/c24/c30 still running**
(server.log/benchmark.log only, no aiperf artifacts). Nothing here touched those
jobs. All numbers are reproducible with the read-only helper
`uq_agentx_cell_analysis.py` in this directory (`table`, `events`, `step`,
`accept`).

---

## 1. Executive summary

**What is already fixed for UQ (do not re-spend effort):**

- `--block-size 64` (P0-1) is applied to the UQ baseline; the UQ server logs
  show attention block 3136 (not the pre-fix 16-token-granular geometry).
- Decode-path knobs are the same fast set as MTP UQ/PDV
  (`FUSED_MTP=1`, `PDV_VERIFY=1`, `PAIR_WAVES=2`, FlyDSL decode). The
  decode-only gap to MTP UQ is only ~1.4-1.9x, not 10x (section 4).
- UQ prefill is ~1.7-1.8x faster than FP8 prefill. That is why DSpark UQ beats
  DSpark FP8 at c32 (17.4 vs 6.8 aiperf tok/s) and at c16 (44.5 vs 27.7).

**What is still broken, ranked (details in section 5):**

1. **All-or-nothing prefix-cache misses create a recompute storm.** Hit loss is
   *not* LCM granularity. On requests that do hit, UQ misses ~7-12k tokens,
   about the same as MTP UQ (7.9k). The damage is a minority of events that hit
   **zero** tokens and recompute 105-170k tokens each. Those events are 86% of
   all miss tokens on DFlash UQ c32 and DSpark UQ c32 (vs 11% for MTP UQ). At
   c32 DSpark UQ spends ~79% of wall time in 8192-token prefill steps that
   produce only ~30% of the generated tokens.
2. **DSpark UQ's KV pool is the smallest of any config (2.17M tokens)**, 4.1x
   smaller than MTP UQ (8.97M) and below DSpark FP8 (2.86M). The cause is the
   bf16 5-layer full-attention draft group padded to 3136/256 token granularity
   (31% utilisation in `kv_geometry_calc.py`). It is UQ-specific: UQ shrinks the
   target KV 3.7x but cannot shrink the bf16 draft pages, so the draft dominates
   the pool.
3. **K=7 mamba-page geometry gives scheduler LCM 12544** (3136 x 4 from the
   256-capped draft block). It is real but second-order (about 4-5k extra miss
   tokens per DFlash request; DSpark runs with partial hash hits, alignment 64).
4. **DFlash accept ~1.00 is a session-level flip, stuck "bad" in 6/6 UQ cells**
   (DFlash FP8 flips "good" in 2/6 cells and mid-cell in 2 more). Not caused by
   context length. UQ may be somewhat more prone; evidence is thin (n=6).
5. **DSpark UQ c8 has a UQ-only accept anomaly**: positions >= 2 accept exactly
   0.000 in 15 of 31 log intervals (FP8 DSpark: 0/22 at c8, 1/33 at c30). Needs a
   one-request probe.
6. aiperf `tok/s` undercounts UQ spec cells (tokens from requests unfinished at
   window end are dropped): DSpark UQ c32 aiperf 17.4 vs server-side 42.7.

**Cheapest high-value experiments** (all config-only or new-file, all into new
out dirs):

- `--max-num-batched-tokens 2048` on DSpark/DFlash UQ c32: modelled ~1.9x
  server tok/s from interleaving decode into prefill steps, no change to
  prefill work (section 6, E1).
- `--prefix-cache-retention-interval <k x 12544>`: replaces the single
  retained mamba/SWA checkpoint with periodic ones, so a lost tail checkpoint
  degrades to a partial hit instead of a 150k-token recompute (E2). The
  existing doc only warned against `None` (dense); a finite positive interval
  has not been tried.
- An env-gated per-group hit logger to find which KV group zeroes the hit (E3).

---

## 2. Evidence tables

All from `uq_agentx_cell_analysis.py table|events`. "Srv tok/s" = delta of
`vllm:generation_tokens` over the profiling phase (counts unfinished requests);
"aip tok/s" is aiperf's `output_token_throughput`. "Hit%" =
`prefix_cache_hits/queries` over the profiling phase; "Theo%" is aiperf's
theoretical hit for that dataset. "Acc" = mean acceptance length (1 + accepted
per draft). "Run/Wait" are time-averaged `num_requests_running/waiting`.
"Miss k/req" = (queries - hits) / (queries / mean ISL).

### 2.1 Per-cell results (UQ vs FP8 peers; FP8 = pre-p01 baselines)

| cell | aip tok/s | srv tok/s | ITL avg ms | ITL p50 | TTFT s | hit % | theo % | acc | KV avg/max % | run | wait | miss k/req |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **DFlash UQ** c8 | 13.2 | 25.1 | 49.9 | 41.2 | 11.6 | 70.7 | 96.5 | 1.00 | 4/6 | 1.5 | 0.02 | 56 |
| DFlash UQ c16 | 4.0 | 8.9 | 396.6 | 363.8 | 13.9 | 63.3 | 98.8 | 1.00 | 30/39 | 3.4 | 0.09 | 42 |
| DFlash UQ c20 | 27.3 | 37.4 | 115.6 | 74.8 | 7.4 | 67.2 | 97.9 | 1.00 | 9/15 | 4.0 | 0.04 | 44 |
| DFlash UQ c24 | 27.0 | 43.0 | 131.1 | 80.3 | 7.4 | 64.3 | 97.9 | 1.00 | 11/17 | 4.8 | 0.04 | 49 |
| DFlash UQ c30 | 16.3 | 41.6 | 210.8 | 104.3 | 10.5 | 62.7 | 98.5 | 1.00 | 16/23 | 7.1 | 0.43 | 48 |
| DFlash UQ c32 | 12.3 | 31.1 | 297.9 | 145.3 | 22.5 | 50.4 | 98.6 | 1.00 | 16/26 | 7.0 | 1.32 | 71 |
| **DSpark UQ** c8 | 21.5 | 61.6 | 22.1 | 13.4 | 9.9 | 74.3 | 97.1 | 2.18 | 11/18 | 1.2 | 0.02 | 47 |
| DSpark UQ c16 | 44.5 | 83.3 | 24.8 | 15.0 | 5.2 | 79.4 | 94.5 | 2.89 | 15/27 | 2.1 | 0.04 | 24 |
| DSpark UQ c20/24/30 | running | | | | | | | | | | | |
| DSpark UQ c32 | 17.4 | 42.7 | 231.2 | 214.5 | 43.5 | 31.1 | 98.2 | 2.78 | 41/73 | 6.2 | 2.11 | 95 |
| DFlash FP8 c16 | 26.6 | 32.0 | 77.4 | 66.9 | 6.3 | 88.2 | 98.2 | 1.03 | 12/20 | 2.6 | 0.10 | 15 |
| DFlash FP8 c20 | 44.6 | 58.4 | 65.4 | 30.2 | 14.0 | 79.5 | 98.3 | 2.69 | 15/29 | 3.4 | 0.44 | 27 |
| DFlash FP8 c24 | 43.6 | 62.1 | 78.6 | 34.4 | 15.3 | 75.6 | 98.2 | 2.69 | 18/29 | 4.0 | 0.45 | 32 |
| DFlash FP8 c32 | 11.8 | 31.7 | 339.1 | 214.4 | 21.4 | 69.4 | 98.4 | 1.19 | 31/49 | 7.4 | 0.96 | 44 |
| DSpark FP8 c16 | 27.7 | 44.0 | 48.1 | 32.3 | 6.3 | 84.9 | 98.3 | 2.68 | 14/22 | 2.4 | 0.22 | 19 |
| DSpark FP8 c24 | 24.8 | 39.5 | 135.9 | 69.4 | 16.3 | 70.3 | 98.0 | 2.48 | 26/41 | 4.8 | 0.29 | 39 |
| DSpark FP8 c32 | 6.8 | 17.8 | 428.6 | 496.7 | 68.8 | 46.8 | 98.9 | 2.78 | 33/54 | 6.1 | 2.74 | 70 |
| **MTP FP8** c32 | 263.1 | 263.1 | 20.5 | 16.0 | 2.8 | 96.0 | 97.7 | 3.44 | 12/23 | 4.2 | 0.00 | 5 |
| **MTP UQ/PDV** c32 | 190.0 | 232.3 | 16.3 | 14.1 | 2.7 | 94.3 | 97.4 | 3.07 | 7/13 | 3.9 | 0.06 | 8 |

Reading the table:

- MTP UQ at c32 sustains 232 server tok/s with ITL 16 ms and 8k miss tokens per
  request. DSpark UQ c32 (42.7 server tok/s, ITL 231 ms, 95k miss/request) is
  **12x more recompute per request and 5.4x fewer tokens per second.**
- UQ KV% average is *low* in the c20-c32 DFlash cells (9-16%): the pool is not
  full, yet hit% collapses. Only DSpark UQ c32 runs hot (41% avg / 73% max)
  because its pool is 2.9x smaller than DFlash UQ's.
- DSpark UQ accept is 2.2-2.9 (it varies with load, 2.18 at c8 is the anomaly in
  section 5.5). DFlash UQ is 1.00 in every cell.

### 2.2 Prefix-hit events: zero-hit vs good-hit

An "event" is a 1 s `server_metrics_export` timeslice with > 20k prefix-cache
queries (one request admitted). Zero-hit: hit < 5%. Good: hit >= 60%.

| cell | events | zero-hit | partial | good | zero-hit share of miss tokens | mean miss, good events | median miss, good |
|---|---|---|---|---|---|---|---|
| MTP FP8 c32 | 77 | 2 | 0 | 75 | 25% | 4.6k | 3.4k |
| MTP UQ/PDV c32 | 75 | 2 | 0 | 73 | 11% | 7.9k | 5.7k |
| DFlash FP8 c32 | 21 | 5 | 1 | 15 | 66% | 17.0k | 4.3k |
| DFlash UQ c8 | 8 | 2 | 0 | 6 | 75% | 18.3k | 14.2k |
| DFlash UQ c16 | 11 | 2 | 1 | 8 | 66% | 11.7k | 4.4k |
| DFlash UQ c20 | 23 | 4 | 1 | 18 | 71% | 12.3k | 7.3k |
| DFlash UQ c24 | 23 | 5 | 1 | 17 | 74% | 12.5k | 7.2k |
| DFlash UQ c30 | 28 | 8 | 1 | 19 | 80% | 11.0k | 6.8k |
| DFlash UQ c32 | 24 | 10 | 2 | 12 | 86% | 7.2k | 3.8k |
| DSpark FP8 c32 | 14 | 4 | 4 | 6 | 49% | 8.8k | 6.5k |
| DSpark UQ c8 | 9 | 2 | 0 | 7 | 78% | 14.0k | 9.6k |
| DSpark UQ c16 | 37 | 5 | 4 | 28 | 54% | 8.5k | 2.7k |
| DSpark UQ c32 | 20 | 11 | 3 | 6 | 86% | 10.8k | 10.3k |

Zero-hit event rate at c32: MTP 3%, DFlash FP8 24%, DFlash UQ 42%, DSpark FP8
29%, DSpark UQ 55%. The zero-hit events recompute ~150-170k tokens each
(DFlash UQ c32: 152k mean; DSpark UQ c32: 168k; DSpark UQ c16: 105k; MTP UQ:
37k).

Takeaway: the "good" events show UQ DFlash misses only ~1.5x more than FP8
(7-12k vs 6.5-7.5k) - that is the 12544-vs-1664 LCM granularity cost, about
half an LCM per request. It is tiny next to the zero-hit events.

### 2.3 Geometry and pool (server.log + `kv_geometry_calc.py`)

| config | attn block (tokens) | scheduler LCM | KV pool tokens | max conc @262k | draft attn backend / pages |
|---|---|---|---|---|---|
| MTP UQ/PDV (K=5) | 3072 | 3072 | 8,966,794 | 34.2x | in-model MTP head, no separate draft KV |
| MTP FP8 (K=5) | 1664 | 1664 | 5,390,793 | 20.6x | in-model |
| **DFlash UQ (K=7)** | **3136** | **12544** | 6,304,589 | 24.1x | `ROCM_AITER_FA_UQ_COMPAT`, bf16, block capped 256, SWA 2048 |
| **DSpark UQ (K=7)** | **3136** | **12544** | **2,169,829** | **8.3x** | same backend, bf16, full attention (5 layers) |
| DFlash FP8 (pre-p01) | 1648 | 3296 | 2,495,081 | 9.5x | `ROCM_AITER_FA` |
| DSpark FP8 (pre-p01) | 1648 | 3296 | 2,861,048 | 10.9x | `ROCM_AITER_FA` |

Calculator lines (`kv_geometry_calc.py`, UQ, K=7):

| variant | target block | draft block | draft util | LCM |
|---|---|---|---|---|
| bf16 draft today | 3136 | 256 | 30.7% | 12544 |
| dense lcm-padded draft | 3136 | 784 | 94.1% | 3136 |
| UQ-format draft (4-bit) divisor | 3136 | 3136 | 100% | 3136 |

Both UQ logs print `Padding mamba page size by 1.83%` and
`lcm block sizes 12544`. The UQ target page is 1088 B/token/layer (4 heads x
272 B); the mamba page depends on `num_speculative_tokens` through the conv
state width, so K=7 (3136) differs from K=5 (3072).

### 2.4 Acceptance anomalies

| cell | fraction of log intervals with accept pos>=2 == 0 | note |
|---|---|---|
| DSpark UQ c8 | 15 / 31 | UQ-only; in-flight ~213k-token request |
| DSpark UQ c16 | 0 / 32 | acc 2.89 |
| DSpark UQ c32 | 0 / 32 | acc 2.78 |
| DSpark FP8 c8..c32 | 0/22, 0/33, 0/32, 0/34, 1/33, 0/32 | |
| DFlash (any) | n/a, pos0 itself ~0 in bad cells | |

DFlash good/bad accept by cell (intervals with acc > 1.5):

| | c8 | c16 | c20 | c24 | c30 | c32 |
|---|---|---|---|---|---|---|
| DFlash UQ | 0/24 | 0/31 | 0/33 | 0/33 | 0/32 | 0/32 |
| DFlash FP8 | 0/30 | 0/33 | 32/32 | 31/32 | 3/34 | 9/33 |

---

## 3. Answers to the six questions

### Q1. Why DSpark UQ (~17) > DSpark FP8 (~7) but << MTP UQ/PDV (~190-240)?

Per-step engine time at c32 is not the problem. A step-time model from the server logs (derived, not directly profiled) gives decode-only steps of ~110 ms with 6 running requests and ~2.8
accepted tokens per verify, so decode-only throughput is ~150 server tok/s.
The 42.7 server tok/s actually achieved comes from time lost in prefill steps:

- DSpark UQ c32 completed 905 engine steps in the 340 s profile phase and
  generated ~14.5k tokens. Recompute (miss) tokens in the window are ~2.1M,
  i.e. ~260 steps at the 8192-token budget (~1 s each).
- ~79% of wall time sits inside those prefill steps, and every running request
  gets a single decode verify (~2.8 tokens) per step. Roughly 4.4k of the 14.5k
  tokens are produced in prefill steps, ~10k in the ~640 decode-only steps.
  (Model derived from miss-token counts and step counts; see E1 for a
  direct A/B.)
- MTP UQ avoids this because only 3% of its admissions are zero-hit (8k miss
  per request), so it is decode-bound almost all the time.

UQ beats FP8 for DSpark because the same recompute volume runs 1.7-1.8x faster
on UQ (dequant-whole-prefix + dense FA2 continuation prefill, fused MTP draft
metadata) and TTFT/queue time shrinks accordingly. It is still far below MTP
because every UQ DSpark request has a 24-55% chance of a ~150k-token recompute,
and because DSpark UQ's smaller pool (2.17M) evicts more at c>=24.

The decode-only gap to MTP UQ is 1.4-1.9x (K=7 verify of 8 rows through PDV vs
K=5/6 rows, plus the 5-layer full-attention draft at 200k+ context, plus the
Markov/confidence heads). That is real but is not what separates 17 from 190.

### Q2. Why is DFlash UQ accept ~1.0 while DSpark UQ is ~2.7? Anything UQ-specific?

- DFlash accept is **session-level, bimodal and not length-dependent**. FP8
  DFlash shows the same phenomenon: c8/c16 bad (1.00/1.03), c20/c24 good
  (2.69/2.69 for the whole cell), c30/c32 mostly bad with 3/34 and 9/33 good
  intervals. UQ is bad in 6/6 cells and 0/185 intervals. That is shared
  behaviour with a UQ-side bias (the chance of 6/6 bad at FP8's observed rate
  of ~67% bad cells is ~9%).
- DFlash's draft is SWA 2048 with `is_causal=false`; DSpark's is full attention.
  For both, the draft backend on UQ is `ROCM_AITER_FA_UQ_COMPAT`, a thin
  subclass of `AiterFlashAttentionBackend` (76 lines) that only changes the
  layout/kernel-block-size list and caps the block at 256. The sliding-window
  logic is inherited unchanged from `rocm_aiter_fa.py`
  (`sliding_window - 1, 0` window tuple), so nothing in the UQ subclass touches
  the SWA/accept path. The UQ target does change the target hidden states
  that DFlash draws its context from (4-bit KV perturbs attention outputs), but
  DSpark consumes the same hidden states and holds ~2.7, so this is not a
  sufficient explanation.
- What stays unexplained (hypotheses only): the cause of the flip itself.
  Candidates: draft SWA block table going stale after a prefix-hit/zero-hit
  transition (the draft group's SWA blocks have their own eviction), first
  request of a session fixing a state that is never reset, or draft position
  offsets after a mamba-align replay. Diagnostic: log per-request accept length
  with `cached_tokens`, and restart the same cell twice (E5).

### Q3. UQ geometry; is 12544 still the dominant prefix-hit killer?

Confirmed from logs: UQ DFlash and DSpark both have attn block 3136, draft
block 256 (capped by `customize_spec`), `lcm block sizes 12544`. MTP UQ is
3072/3072.

**No, 12544 is not the dominant killer after block-size 64.** Evidence:

1. Good-hit events miss 7-12k tokens on UQ DFlash vs 7.9k on MTP UQ (LCM 3072)
   and 6.5-7.5k on DFlash FP8 (LCM 3296). A 12544 LCM costs about half an LCM
   (~4-5k tokens/request) on top.
2. Zero-hit events are 54-86% of the miss tokens on every UQ cell and recompute
   100-170k tokens, an order of magnitude more than granularity can explain.
3. DSpark has no sliding-window manager, so `enable_partial_hash_hits` stays on
   and the hit alignment is 64, not 12544; it still loses 86% of c32 miss tokens
   to zero-hit events.
4. K=6 would give 3072 and a dense draft would give 3136; both are worth doing
   (section 6) but cannot fix the zero-hit events.

### Q4. Decode-path overhead: which knobs are on, what adds latency vs MTP UQ?

`set_uq_env` in `run_dflash_dspark_agentx_conc_sweep.sh` exports for DFlash/DSpark
UQ (identical to the MTP UQ/PDV recipe except for `PDV_EXPERIMENT` being unset):

```
VLLM_ULTRAQUANT_FUSED_MTP=1          # on: fused draft metadata path
VLLM_ULTRAQUANT_SHARED_KV=0          # off
VLLM_ULTRAQUANT_V3=0                 # off
VLLM_ULTRAQUANT_ROWPACK_VERIFY=0     # off
VLLM_ULTRAQUANT_PAIR_WAVES=2         # on
VLLM_ULTRAQUANT_PAIR_VERIFY=0        # off
VLLM_ULTRAQUANT_PDV_VERIFY=1         # on (one KV read scores 2-8 verify tokens)
# VLLM_ULTRAQUANT_PDV_EXPERIMENT unset
```

Attention: target `ULTRAQUANT_SPEC_COMPAT`; draft `ROCM_AITER_FA_UQ_COMPAT`
with draft `kv_cache_dtype=auto` (bf16).

Latency contributions that exist for DFlash/DSpark UQ but not for MTP UQ:

| item | applies to | cost | evidence / status |
|---|---|---|---|
| K=7 verify (8 rows) vs K=5 (6 rows) | both | PDV covers qlen 2-8; the rows pack into MFMA instructions, so 8 rows is near the packing limit | no direct profile; E6 validates kernel vs reference at qlen 8 |
| separate draft forward with 5 attention layers over 200k+ context | DSpark (full), DFlash (SWA 2048 bounds it) | full-attention draft reads the whole context in bf16 each step: bf16 draft KV is ~3.8x the bytes/token/layer of UQ (4096 B vs 1088 B), so each draft layer reads ~3.8x more than a UQ target layer over the same context | explains a decode-only step ~2x MTP; DSpark only |
| draft Markov/confidence heads, YaRN rope | DSpark | small | |
| continuation-prefill via dequant + dense FA2, per-request Python loop | all UQ | amortised in prefill steps; slow only when `_CONTINUATION_DECODE_THRESHOLD` (128) is crossed with many partial prefills | this is what the recompute storm runs through |
| draft FA_UQ_COMPAT block cap 256 | all draft | many small blocks: block table 12x longer than at 3136; padding waste in pool, not in decode | secondary |
| Triton JIT in inference | all UQ | one-off per new shape | seen only in early steps |

Knobs that add latency without helping: none identified. The PDV/pair-wave
stack is the same as the shipping MTP UQ run. `PDV_EXPERIMENT` unset means the
default kernel, which is the right baseline.

### Q5. Draft backends on UQ

- `ULTRAQUANT_SPEC_COMPAT` (target): prefers 64-aligned blocks, same kernel
  block sizes as TurboQuant. Fine.
- `ROCM_AITER_FA_UQ_COMPAT` (draft): `LBNHC` layout, kernel sizes
  `[16..256]`, `customize_spec` replaces the draft spec block size with
  `max_kernel=256`. This is the origin of the draft block 256 and therefore of
  the 30.7% pool utilisation and the 12544 LCM.
- **Mixed FP8 draft-kv:** `*_kv8` smoke runs failed with "Mixed FP8" in aiter FA
  (`p0_status.md`), so draft KV stays bf16 on UQ. So the UQ-format
  draft is not available as a config switch, and DSpark UQ's pool is
  dominated by draft pages.
- **Padding waste unique to UQ:** the UQ target page is 3.7x smaller per token
  than bf16, so the 3136-token target block is ~3.4 MB per layer. The bf16 draft
  is capped at a 256-token block (12x smaller than the 3136 target block); the
  allocator pads so that all groups share a page size. FP8 (3296 LCM, draft block ~64-256) pays
  much less for the same mismatch.
- Mamba page is padded by 1.83% (UQ) vs 0.73% (FP8): negligible.

### Q6. Ranked UQ-specific fix list

See section 6. Short form: P0 = E1 (chunk size), E2 (retention interval), E3
(hit logger); P1 = dense draft backend, K=6, accept-anomaly probe; P2 = UQ-format
draft kernel, continuation-prefill batching, per-request accept logging.

---

## 4. Decode-path gap vs MTP UQ (numbers)

| quantity | MTP UQ c32 | DSpark UQ c16 | DSpark UQ c32 |
|---|---|---|---|
| ITL p50 (ms) | 14.1 | 15.0 | 214.5 |
| ITL avg (ms) | 16.3 | 24.8 | 231.2 |
| acceptance length | 3.07 | 2.89 | 2.78 |
| engine steps / 340 s | 6627 | 5148 | 905 |
| avg running reqs | 3.9 | 2.1 | 6.2 |

At c16 DSpark UQ has ITL p50 15.0 ms, equal to MTP UQ (14.1 ms), and 2.89
accept. The decode path is **not** the bottleneck at c16; it only looks that
way at c32 because 79% of wall time is prefill. Server tok/s at c16 is 83 vs
232 for MTP because only ~2 requests run at a time (low concurrency in the
decode phase: the other requests are queued or prefilling).

---

## 5. UQ-specific root causes (ranked)

### 5.1 All-or-nothing prefix misses and prefill-bound decode starvation (dominant)

Symptoms: section 2.2 and Q1. Zero-hit rate 24-55% at c32 for spec-decode
configs vs 3% for MTP; each zero-hit event recomputes 100-170k tokens; KV% is
often low (9-16% in DFlash UQ c20-c30), so this is not simple pool exhaustion.

Why it is UQ-worse: UQ's KV is small, so a UQ run packs more concurrent
trajectories per GB and uses larger blocks (3136 vs 1648), meaning each
checkpoint covers more tokens and one lost checkpoint costs more. DSpark UQ is
additionally pool-limited (5.2).

Mechanism (hypothesis with a clear code pointer): with the default
`prefix_cache_retention_interval=0`, `cache.py` documents that only "semantic
checkpoints, including the latest replay boundary and shared-prefix junctions"
are retained for sliding-window and Mamba groups. A request therefore leaves
exactly **one** usable mamba/SWA checkpoint per trajectory. If that single
checkpoint is evicted, or its position is not an allowed cache-hit alignment
(`_cache_hit_alignment_tokens` = LCM for DFlash, 64 for DSpark), the coordinator's
`find_longest_cache_hit_per_group` fixed-point loop drops to zero because the
full-attention groups must agree with the mamba group's hit. That matches the
observed bimodal hit distribution (either ~95% or 0%). Other candidates the
logger in E3 will separate:

- the `MambaManager.find_longest_cache_hit` alignment filter rejecting the
  retained state (LCM-phase mismatch),
- align-mode interior-state retirement,
- tail-state eviction under unrelated churn (churn in the 5 s before a zero-hit
  event is not monotonically related to its probability: 258k tokens of churn
  gave a zero-hit in one event and 1.1M tokens of churn gave 95% hit in
  another),
- SWA draft-window block eviction (DFlash only).

### 5.2 DSpark UQ bf16 padded draft pool (UQ-specific, severe at c>=24)

2,169,829-token pool vs 8,966,794 (MTP UQ) and 2,861,048 (DSpark FP8). The UQ
target would allow ~9M tokens; the bf16 full-attention draft with 256-token
blocks inside a 3136-token page group cuts utilisation to 30.7%. DSpark UQ
c32 runs 41% average / 73% max KV usage vs 9-16% for DFlash UQ (6.3M pool).
This both evicts more cached prefixes and lowers the admission limit
(max concurrency 8.3x @262k).

Fix: dense (lcm-padded) draft pages, 784-token draft blocks, 94% utilisation,
LCM 3136; or UQ-format draft KV (section 6, E4/E8).

### 5.3 K=7 mamba page geometry: LCM 12544

Second-order: ~4-5k extra miss tokens per request for DFlash (alignment =
LCM 12544); DSpark alignment is 64. Also forces chunk boundaries onto 12544
multiples for DFlash under align mode. K=6 gives 3072; dense draft gives 3136.

### 5.4 DFlash session-level accept flip

See Q2. It caps DFlash UQ at ~1 token per verify and so ~3x lower decode
throughput when bad; it is also the main reason DFlash UQ c8/c16 look worse
than DSpark UQ even at low load.

### 5.5 DSpark UQ c8 accept anomaly (UQ-only)

Positions >= 2 are exactly 0.000 while pos0/pos1 are 0.56-0.71 / 0.04-0.30 in
15 of 31 intervals of the c8 cell, during a single ~213k-token request. FP8
DSpark does not show it (1/33 intervals across 6 cells). Exact zeros point to a
masking or position-id fault in the draft chain beyond the second token at very
long context (e.g. draft-slot mapping across a 256-token block boundary, an
int32 offset, or the confidence head truncating drafts), not a quality effect.
c16/c32 do not show it, so it needs a controlled repro: E5.

### 5.6 aiperf tok/s undercount (measurement)

aiperf drops tokens of requests unfinished at window end. In the DFlash/DSpark
UQ cells the ratio srv/aip is 1.6-3.1x (DSpark UQ c32: 2.45x) because each
request is hundreds of seconds long. Use `server tok/s` and ITL for A/B; report
both. Not a fix, but it changes how the UQ numbers compare against MTP (ratio
1.2x).

---

## 6. Recommended experiments

Rules for all experiments (workspace rule + user constraints):

- **Do not edit `run_dflash_dspark_agentx_conc_sweep.sh` while sweeps run** (bash
  reads scripts incrementally) and never reuse the baseline `OUT_BASE`. Copy it
  to a new runner (for example `run_uq_spec_ab_agentx.sh`) that takes `OUT_TAG`
  and writes `out/<cfg>_<tag>/sa/seed42_c<N>` (new dirs). Run only after the
  current jobs on ports 8410-8441 finish or on a free HIP device/port. Do not kill
  or interrupt running jobs.
- Kernel/backend changes: new file + env gate, shipping kernel files unchanged,
  default env keeps the existing kernel, new LDS symbol names. A dispatcher
  branch may select the new module.
- Same seed (42), same concurrencies; compare against the existing baseline cell
  using `uq_agentx_cell_analysis.py table|events|accept <cellA> <cellB>`.

| id | priority | what | files / gates | expected effect | A/B |
|---|---|---|---|---|---|
| **E1** | P0 | Smaller prefill chunk so decode interleaves with recompute | config only: `MNBT=2048` (and 4096) with `--max-num-seqs` unchanged; optionally `--long-prefill-token-threshold 2048` | ~1.9x server tok/s on DSpark/DFlash UQ c32 (model: same ~2.1M prefill tokens, but ~1050 steps instead of ~260, each carrying decode for the running requests; decode in prefill steps grows from ~4.4k to ~17k tokens), ITL p50 down ~4x. TTFT may rise; check `iteration_tokens_total` | new out dir `*_mnbt2048`; c24, c32 first. Watch the mamba align split when chunk (2048) < block (3136): confirm no hit-rate drop |
| **E2** | P0 | Periodic mamba/SWA checkpoints | config: `--prefix-cache-retention-interval 50176` (4 x 12544) and `25088` (2 x); must be a multiple of the scheduler block | turns zero-hit into partial hit (recompute <= interval, 25-50k instead of 105-170k); cost per checkpoint ~170 MB mamba (+~150 MB SWA for DFlash) per trajectory, +25-50% cache footprint at these intervals; expect miss tokens/request -50..-70% at c32 | new dirs `*_ret50k`, `*_ret25k`; compare `events` zero-hit share, hit%, KV% |
| **E3** | P0 | Per-group hit diagnostic | new `vllm/v1/core/kv_hit_debug.py`, enabled by `VLLM_KV_HIT_DEBUG=1`, called from a thin hook in `kv_cache_manager.get_computed_blocks`; logs request id, prompt length, per-group hit length, argmin group, alignment, free blocks, cached-block count | identifies which group zeroes the hit (mamba filter / eviction / SWA / full-attn) so E2 or a targeted fix can be chosen | one DSpark UQ c32 and one DFlash UQ c32 cell into `*_hitdbg` |
| **E4** | P1 | Dense draft backend | new `vllm/v1/attention/backends/rocm_aiter_fa_uq_dense.py` registered as `ROCM_AITER_FA_UQ_DENSE`; selected via a new spec JSON (`uq_dspark_dense.json`, `uq_dflash2_dense.json`) with draft block 784 (`customize_spec` returns lcm-padded size) | DSpark UQ pool 2.17M -> ~4.4M tokens (calculator: 94% util, LCM 3136); less eviction at c>=24, fewer zero-hit events, DFlash granularity cost removed | needs a smoke that aiter FA accepts block 784 (kernel block sizes are a multiple of 16; `p0_status.md` shows `geometry` smoke pattern) |
| **E5** | P1 | Accept anomaly / flip probe | run `probe_accept_vs_length.py` on DSpark UQ and FP8 at 131072, 196608, 229376 (c1, same prompt); log accept per position; for DFlash, restart the same cell twice | tells whether DSpark UQ pos>=2 == 0 is length-gated and UQ-only; whether the DFlash flip is run-to-run random | new out dir `*_probe`; no kernel change |
| **E6** | P1 | PDV qlen-8 correctness/latency at 200k+ | new script `benchmarks/ultraquant_mtp/bench_pdv_verify_qlen8.py`: PDV vs reference attention on UQ pages at ctx 131k-230k, qlen 6/8 | rules out a verify-kernel fault for the pos>=2 zeros and gives per-step cost for K=7 | read-only; no shipping kernel edit |
| **E7** | P1 | K=6 spec config | new spec JSON `uq_dspark_k6.json` / `uq_dflash2_k6.json` | LCM 12544 -> 3072; ~4-5k fewer miss tokens/request on DFlash; may trade ~0.2 accept length | new dirs `*_k6`; compare hit% and acc |
| **E8** | P2 | UQ-format draft KV | new FlyDSL draft decode kernel `pdv_flydsl_decode_draft_uq.py`, selected by `VLLM_ULTRAQUANT_PDV_EXPERIMENT=dflash_uq_draft`; unique LDS symbol; default stays on aiter FA | pool and LCM at 100% util/3136; removes bf16 draft read bandwidth for DSpark | large effort; only after E4 data |
| **E9** | P2 | Batch continuation prefill | new module (e.g. `ultraquant_cont_prefill_batched.py`) behind `VLLM_ULTRAQUANT_CONT_PREFILL=batched`; replaces the per-request Python loop | lowers per-chunk overhead when E1 raises chunk count | A/B on prefill-heavy c32 |

Suggested order: E3 and E1 in parallel on free devices (both small), then E2,
then E4/E5/E6. If E3 shows the single-checkpoint hypothesis is wrong, skip E2.

Measurement hygiene: record `server tok/s`, ITL p50/p95, TTFT, hit%, zero-hit
event count and miss tokens, accept per position, KV avg/max, step count; keep
`server.log`, `server_metrics_export.json`, `profile_export.jsonl`.

---

## 7. Explicit non-goals

- **"Re-run UQ with `--block-size 64`" is not a fix.** It is already in the UQ
  baseline (attn block 3136 in the logs). FP8 is catching up via `*_fp8_p01`.
- Do not set `prefix_cache_retention_interval=None` (dense retention); E2 tests
  finite positive intervals only.
- Do not edit shipping UQ kernels (`pdv_flydsl_decode_hd256*.py`, FlyDSL decode,
  `ultraquant_attn.py`) for these experiments; use new modules and env gates.
- Do not treat the 12544 LCM as the prefix-hit root cause. It is a ~4-5k token
  per request granularity cost.
- Do not interpret aiperf `tok/s` alone for UQ spec cells (section 5.6).
- Do not kill or interrupt anything on ports 8410-8441 or the HIP devices in use.
- Not covered here: MTP UQ internals, FP8-only fixes, DFlash SWA2048 accept
  collapse at long context (documented in the earlier note; this study finds
  the flip is not length-dependent).

---

## 8. Open items

- DSpark UQ c20/c24/c30 still running; re-run
  `uq_agentx_cell_analysis.py table events dspark_qwen38_uq/sa/seed42_c{20,24,30}`
  when they finish and add rows to section 2. Expectation: c24-c30 look like c32
  (zero-hit-dominated) if the pool theory holds.
- Root cause of structural zero-hit events at low KV% (5.1) is not proven.
- Root cause of the DFlash accept flip and of the DSpark UQ c8 pos>=2 zeros is
  not proven.
- E1's model assumes prefill efficiency holds at 2048-token chunks; verify.
