# DSpark / DFlash improvement backlog (quality-preserving)

Date: 2026-10-29 (merged with research agent pass).  
Goal: higher tok/s / accept on lit-match + AgentX **without** changing the target output distribution (unless Class C).

**Lit-match (in progress):** nospec C1 **93.5** / C8 **576** / C32 **1834** tok/s; MTP C1 **138** tok/s, accept **2.54** (~1.48×); DSpark C1 RUNNING.  
**Recovery:** geom64 / mnbt8k / SWA win_table / adaptive arms armed; AgentX paused until lit-match finishes.

Sources: [dflash_dspark_agentx_bottlenecks.md](./dflash_dspark_agentx_bottlenecks.md), [DSpark paper](https://arxiv.org/html/2607.05147), [vLLM adaptive verification](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification/), [PR 47808](https://github.com/vllm-project/vllm/pull/47808), [block verification](https://arxiv.org/abs/2403.10444), [SuffixDecoding](https://arxiv.org/abs/2411.04975), vLLM [#47930](https://github.com/vllm-project/vllm/issues/47930) / [#61761](https://github.com/vllm-project/vllm/issues/61761) / [#60560](https://github.com/vllm-project/vllm/issues/60560).

---

## Quality taxonomy

| Class | Rule | Examples |
|---|---|---|
| **A — lossless serving** | Same rejection sampler / target logits | geometry, SWA win_table, dense pages, mnbt, draft-context restore fix |
| **B — lossless draft policy** | Exact target verify; changes draft/verify *amount* or draft checkpoint | adaptive, block verify, windowed DSpark, DFlash2-Exp, suffix oracle |
| **C — quality risk** | Changes distribution / approximate accept | relaxed MTP thinking, 4-bit draft KV, mismatched top-p |

Default A/B gate: lit-match speedup+accept **and** card-sampling quality within noise. Class C needs explicit quality A/B.

---

## Top 10 (post-research rank)

| # | Idea | Class | Lever | A/B | Pass gate | Effort |
|---|---|---|---|---|---|---|
| 1 | **Draft-context on prefix-cache restore (H5)** — restored tokens never wrote draft KV; warm-hit accept→1 ([#47930](https://github.com/vllm-project/vllm/issues/47930)) | A | accept AgentX | 3-pass probe: cold / warm-hit / no-prefix-cache | warm ≥0.9× cold accept | Probe → env-gated fix |
| 2 | **Verifier × draft-sample grid** — `{greedy,probabilistic} × {standard,block}` rejection | B | accept, tok/s | lit_match C1/8/32 then AgentX | L + tok/s ≥ current | Spec JSON |
| 3 | **Windowed DSpark draft** W∈{8k,32k} via SWA backend (no retrain) | B | draft cost, LCM, ITL | probe then AgentX | W=8k accept ≥95% of full-attn | Config copy |
| 4 | **mnbt A/B {2048,4096,8192}** after geometry (earlier UQ favored 2048) | A | ITL vs hit | AgentX C16/32 | tok/s↑, hit drop ≤5pp | Config |
| 5 | **Adaptive verification finish** — DFlash adaptive + EMA tune + UQ ragged/PDV check | B | high-C verify cost | AgentX C24–36 adaptive vs fixed | ≥ fixed @C32; never < nospec | Spec JSON + env |
| 6 | **SWA hardening** — window margin {0,64,128}, symmetrize right window, NaN null-block | A | DFlash long accept | `run_swa_probe.sh` | accept≥2.5 @ ctx≥32k gen≥2560 | Backend env |
| 7 | **Draft attn: TRITON vs aiter FA / win_table** (MI355X often favors Triton) | A | draft step ms | lit_match + probe | accept ±0.1; step ≤ win_table | Spec JSON |
| 8 | **DFlash2-Exp draft** ([DaoCloud](https://huggingface.co/DaoCloud/Qwen3.8-27B-DFlash2-Exp); needs PR 54154) | B | long-ctx AL | lit_match then AgentX | AL & tok/s ≥ DFlash2 | PR + config |
| 9 | **GDN / mamba page vs K** — smaller LCM; deferred state under align | A | prefix-hit, pool | server.log LCM + AgentX | hit≥70%; ITL↓ ≥5% | Experiment path |
| 10 | **SuffixDecoding oracle → hybrid** on AgentX traces | B | agent-turn accept | offline oracle then AgentX arm | oracle AL ≥ DSpark ~2.65 | Script → new path |

**In-flight (recovery suite — keep running, not re-ranked as “new”):** geom64, mnbt8k, retention, FP8 draft kv8, `ROCM_AITER_FA_DFLASH_SWA`+`win_table`, UQ dense, DSpark adaptive.

---

## Demote / defer (vs earlier draft)

| Item | Why demote |
|---|---|
| **DSL `draft_confidence_threshold`** | Sequential draft early-exit; DSpark/DFlash emit K in one forward — little save. Adaptive already trims verify. |
| **Blind K sweep** | q=6–8 same PDV cost; K=5 ≈97% accept. Only as tie-breaker once adaptive+geometry green. |
| **UQ-format draft KV (P2-1)** | Heavy; windowed DSpark (#3) may remove pressure first. Accept risk → Class C-ish. |

---

## Run order (after lit-match + AgentX recovery)

1. **H5 three-pass probe** (separates restore vs null-block H1).  
2. Spec-JSON grid (#2) + mnbt (#4) — config only.  
3. Triton draft A/B (#7).  
4. Windowed DSpark (#3), adaptive finish (#5), SWA margin (#6).  
5. Medium: DFlash2-Exp (#8), GDN/LCM (#9), suffix oracle (#10).

---

## Stack knob → experiment map

| Knob (current path) | Where today | Experiments that touch it | Notes |
|---|---|---|---|
| **geometry / `--block-size 64`** | recovery `arms.json` + `run_lane.sh` | in-flight; #9 GDN/LCM | Prove via `kv lcm block sizes` in `server.log` |
| **`mnbt` / max_num_batched_tokens** | recovery `8192` | **#4** {2048,4096,8192} | Re-check after geometry; earlier UQ liked 2048 |
| **`prefix_cache_retention`** | 49920 FP8 / 50176 UQ | in-flight; keep LCM-aligned; never `0` on hybrid | |
| **SWA `win_table` / `ROCM_AITER_FA_DFLASH_SWA`** | recovery DFlash arms + `run_swa_probe.sh` | in-flight; **#6** margin/symmetrize | Correctness before any DFlash K/temp tune |
| **adaptive verification** | DSpark `*_adaptive_geom64` only | **#5** add DFlash + EMA | Only after geometry green |
| **draft sampling** | DSpark `probabilistic`; DFlash often default greedy | **#2** greedy/prob × standard/block | `rejection_sample_method` exists in `config/speculative.py` |
| **draft attn backend** | Triton (lit_match) / FA_DFLASH_SWA / UQ_DENSE | **#7** Triton vs aiter; **#3** windowed DSpark | MI355X often favors Triton for draft |
| **draft KV dtype** | FP8 draft kv8 on FP8 lane | in-flight; demote UQ draft KV | |
| **K / `num_speculative_tokens`** | 7 | demote blind sweep; adaptive caps effective K | Tie-breaker only |
| **prefix-cache × draft KV** | enabled everywhere | **#1 H5** restore probe | Novel diagnostic vs SWA H1 |

---

## Quality-preservation checks (required per experiment)

| Check | When | Pass |
|---|---|---|
| **Lossless sampler invariant** | Class A/B | Same target rejection path; no relaxed/synthetic accept |
| **Card sampling match** | After any draft/policy change | GSM8K + HumanEval (or lit_match 128) seed-fixed: pass@1 / exact within ±noise of nospec same card |
| **Thinking-card accept** | lit_match | Report `accept_length` from `/metrics`; do not use 2-token cliff probes |
| **AgentX quality smoke** | Before claiming AgentX win | Spot-check trajectories for repetition/degeneration (adaptive PR practice); no invalid rate spike |
| **Class C only** | relaxed MTP, 4-bit draft KV, vocab truncate | Full quality A/B required — not in Top 10 ship path |

---

## Gate criteria (copy from suite manifests)

### Lit-match (`lit_match_dspark_dflash/arms.json`) — vs same-card nospec

| Conc | DSpark speedup | DFlash speedup | Accept floor |
|---|---|---|---|
| C1 | ≥ **2.0×** | ≥ **2.5×** | DSpark ≥ **2.5**, DFlash ≥ **3.0** |
| C8 | ≥ **1.5×** | ≥ **1.5×** | same |
| C32 | ≥ **1.1×** | ≥ **1.1×** | same |

### AgentX recovery (`agentx_dspark_dflash_recovery/arms.json`)

| Metric | Gate |
|---|---|
| tok/s @ C32 | ≥ **1.15×** same-lane baseline nospec |
| prefix-cache hit (profile) | ≥ **70%** |
| mean accept length | ≥ **2.5** |
| ITL | not ~10× worse than MTP at same C |

### Probe-only (before full AgentX DFlash)

| Probe | Gate |
|---|---|
| `probe_accept_vs_length` (SWA / H5) | ctx≥32k & gen≥2560 → accept_len ≥ **2.5**; H5: warm-hit ≥ **0.9×** cold |

---

## Anti-goals

- Don't tune as capacity/preemption (they're ~0).  
- Don't disable prefix cache as a “fix”.  
- Don't retune K/temp for DFlash before SWA/H5 accept is fixed.  
- Don't edit shipping FA/FlyDSL in place — new experiment paths only.  
- Class C (relaxed accept, 4-bit draft KV, mismatched sampling) only with full quality harness.
