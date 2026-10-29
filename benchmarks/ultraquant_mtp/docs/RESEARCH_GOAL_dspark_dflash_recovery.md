# Research goal & full context — DSpark / DFlash literature recovery (ROCm)

**Last updated:** 2026-10-29 (GPU paused / released)  
**Branch intent:** resume lit-match Phase 1 → AgentX Phase 2 on a new MI355X (or similar) host  
**Owner machine (paused):** AMD Instinct node; physical GPU Unique ID `0xe81e8e9535780fc8`, PCI `0000:75:00.0` (was `HIP_VISIBLE_DEVICES=3` on that host only)

This document is the **single source of truth** to pull this repo on a new server and continue without tribal knowledge.

---

## 1. One-sentence goal

Recover **literature-class** DSpark / DFlash speedups on **ROCm + vLLM + UltraQuant** for Qwen3.8-27B: first prove the stack on a **HF thinking-card short-prompt** A/B (Phase 1), then fix **AgentX long-context** faults (geometry, SWA null-block, mnbt, adaptive) so dense drafts **beat same-lane nospec** without losing target-model accuracy.

---

## 2. Why we are here (diagnosis)

| Observation | Implication |
|---|---|
| Spec-Bench FP8 already hits accept ~3.4 DSpark / ~4.1 DFlash and beats AMD slide | Stack + drafts are **not** fundamentally broken on ROCm under short greedy prompts |
| AgentX same K=7, same drafts, TRITON draft attn still loses to nospec | AgentX pain is **long-ctx + sampling + hybrid KV**, not “draft model dead” |
| AgentX ISL ~185k, T=1 / thinking xhigh vs Spec-Bench ISL ~282, T=0 | Workload drives accept down to ~1.2–2.9 |
| AgentX DSpark often **&lt; nospec** tok/s despite accept ~2+ | Death is **re-prefill / ITL**, not zero accept |
| Prefix hit ~96% MTP vs **31–69%** DFlash/DSpark; DFlash UQ accept→**1.0** | Hybrid KV LCM + DFlash SWA null-block are the long-ctx bugs |

Full technical write-up: [`dflash_dspark_agentx_bottlenecks.md`](./dflash_dspark_agentx_bottlenecks.md).

Improvement backlog (novel + stack tunes, quality-preserving): [`dspark_dflash_improvement_backlog.md`](./dspark_dflash_improvement_backlog.md).

Pause / GPU release: [`PAUSE_REPORT_dspark_dflash_2026-10-29.md`](./PAUSE_REPORT_dspark_dflash_2026-10-29.md).

---

## 3. Phased plan (what we are executing)

### Phase 1 — Literature-matched A/B (prove stack health)

**Suite:** `benchmarks/ultraquant_mtp/lit_match_dspark_dflash/`  
**Out:** `out/lit_match_qwen38/` (local; not always in git — see §8)

| Item | Value |
|---|---|
| Model | Qwen3.8-27B TP=1 FP8 |
| Sampling | T=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, thinking on (HF card) |
| Data | 128 prompts (`datasets/lit_match_128.jsonl`), max_tokens≈2048 |
| Concurrency | {1, 8, 32} |
| Arms | nospec, MTP K=5, DSpark K=7 probabilistic, DFlash2 K=7 |

**Pass gates (vs same-card nospec):**

| Conc | DSpark | DFlash | Accept |
|---|---|---|---|
| C1 | ≥2.0× | ≥2.5× | DSpark≥2.5, DFlash≥3.0 |
| C8 | ≥1.5× | ≥1.5× | same |
| C32 | ≥1.1× | ≥1.1× | same |

**Critical bug fixed:** `--speculative-config` must be a **JSON string**, not a file path. See `run_cell.sh` (`SPEC_JSON=$(python3 -c '…json.dumps(json.load(…))')`).

### Phase 2 — AgentX long-ctx recovery

**Suite:** `benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery/`  
**Out:** `out/agentx_dspark_dflash_recovery/`

| Knob | Baseline (broken) | Recovery |
|---|---|---|
| block_size | FA base → 16 → bad LCM | **64** |
| mnbt | 2048 | **8192** (re-A/B 2048/4096 after geom) |
| retention | 49920 / 50176 | keep; verify LCM align |
| FP8 draft KV | often bf16 mismatch | **fp8** |
| DFlash draft attn | TRITON / broken SWA | `ROCM_AITER_FA_DFLASH_SWA` + `VLLM_DFLASH_DRAFT_ATTN_EXPERIMENT=win_table` |
| Adaptive | off / untested after geom | DSpark `enable_adaptive_verification` arms |

**AgentX C32 gates:** ≥1.15× baseline nospec; prefix hit ≥70%; accept ≥2.5; ITL not ~10× MTP.

**Handoff:** `lit_match_dspark_dflash/handoff_phase2.sh` waits for `suite.DONE`, runs SWA probe (`util` patched to **0.45**), then AgentX `run_suite.sh`.

---

## 4. What we have already done (evidence)

### 4.1 Infra / process

- Built lit-match suite + AgentX recovery suite + canvases + auto handoff.
- Paused prematurely started AgentX once Phase 1 failed on the speculative-config path bug; fixed JSON path; relaunched Phase 1 first.
- **Released GPU** 2026-10-29: no vllm/suite/handoff; Unique ID `0xe81e8e9535780fc8` free.

### 4.2 Lit-match results at pause

| Arm | C1 | C8 | C32 |
|---|---|---|---|
| fp8_nospec | **93.5** tok/s DONE | **576.4** DONE | **1834.8** DONE |
| fp8_mtp | **138.0** tok/s, accept **2.54** DONE | pending | pending |
| fp8_dspark | **166.4** tok/s, accept **3.00** (~**1.78×** nospec) DONE | pending | pending |
| fp8_dflash | killed at serve start | pending | pending |

C1 DSpark **accept passes** (≥2.5); **speedup misses** 2.0× gate (1.78×). Still valuable: stack + draft healthy under thinking card.

Snapshot JSON (if present on disk): `out/lit_match_qwen38/pause_snapshot.json`.

### 4.3 AgentX recovery partial

| Cell | Status |
|---|---|
| `fp8_dspark_geom64` seed42 **C8** | **DONE** (profile on disk if outs copied) |
| C16+ and other arms | not done |

### 4.4 Code / experiment paths (shipping kernels untouched)

| Path | Role |
|---|---|
| `vllm/v1/attention/backends/rocm_aiter_fa_dflash_swa.py` | DFlash SWA backend |
| `vllm/v1/attention/ops/dflash_swa_window_table.py` | win_table page table |
| `vllm/v1/attention/backends/rocm_aiter_fa_uq_dense.py` | UQ dense draft geometry |
| `vllm/v1/worker/gpu/spec_decode/dflash/null_block_debug.py` | `VLLM_DFLASH_DEBUG_NULL_BLOCK` |
| Kernel-experiments rule | New file + env dispatch; never edit shipping FlyDSL/FA in place |

### 4.5 Research backlog (post lit / AgentX)

Prioritized in [`dspark_dflash_improvement_backlog.md`](./dspark_dflash_improvement_backlog.md). Top next after recovery:

1. **H5** — draft KV missing on prefix-cache restore (vLLM #47930)  
2. `{greedy,probabilistic} × {standard,block}` verification grid  
3. Windowed DSpark (no retrain)  
4. mnbt A/B after geometry  
5. Adaptive on DFlash + UQ ragged/PDV  
… demote DSL early-exit for parallel drafters; demote blind K sweeps.

---

## 5. What we are trying to do next (resume checklist)

1. **Copy or regenerate outs** — git may not carry multi‑GB `out/`; copy `pause_snapshot.json` + DONE cell `accept_metrics.json` / AgentX C8 profile if you need exact numbers.
2. **Pick a free GPU** on the new host (`rocm-smi --showuniqueid` / `--showbus`); set `HIP_VISIBLE_DEVICES=<index>`.
3. **Resume lit-match** (DONE cells SKIP if `accept_metrics.json` + profile exist):

```bash
cd /path/to/vllm
# activate your ROCm venv; ensure aiter on PYTHONPATH as on old host
rm -f /workspace/logs/lit_match_qwen38/suite.PAUSED   # or local log path
HIP_A=<gpu> HIP_B= UTIL=0.55 \
  bash benchmarks/ultraquant_mtp/lit_match_dspark_dflash/run_suite.sh
```

4. **Re-arm handoff** (after suite, or in parallel waiting on `suite.DONE`):

```bash
HIP=<gpu> bash benchmarks/ultraquant_mtp/lit_match_dspark_dflash/handoff_phase2.sh
```

5. Publish / refresh canvases (`refresh_canvas.py`, `apply_canvas_host.py`).
6. Evaluate Phase 1 gates; if fail → triage draft sample / attn / feature capture before more AgentX.
7. After AgentX gates: run backlog #1 H5 probe, then config-only #2/#4.

### Paths / models expected on server

| Asset | Typical path (old host) |
|---|---|
| Target | `/huggingface/pretrained_models/Qwen/Qwen3.8-27B` |
| DSpark draft | `/workspace/pretrained_models/RadixArk/Qwen3.8-27B-DSpark` |
| DFlash2 draft | `/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2` |
| venv | `/workspace/.venv-uq` |
| aiter | `/workspace/aiter` |
| SemiAnalysis / aiperf | `/workspace/SemiAnalysis` |

Adjust paths in `arms.json` / env if the new server differs.

---

## 6. Quality / accuracy invariants

- Spec decode must stay **lossless** (exact target rejection) for shippable wins.
- Do **not** use relaxed MTP thinking accept or 4-bit draft KV without a Class C quality harness.
- Prefer profile-block metrics from aiperf exports, not cumulative `server.log` prefix-hit.
- Kernel rule: **new experiment module + env**, leave shipping kernels unchanged (`.cursor/rules/kernel-experiments.mdc`).

---

## 7. Out of scope

- Replacing AgentX with Spec-Bench as the only production metric  
- Claiming NVIDIA B200 absolute tok/s parity on MI355X  
- Editing shipping FA / FlyDSL math in place  
- Relying on MIP-encrypted AMD PDFs  

---

## 8. Git / transfer notes

- Commit this docs tree + suites + experiment backends to GitHub (see branch from push).
- **`out/`** aiperf dumps are large — prefer rsync separately:

```bash
rsync -avP user@old:/path/vllm/out/lit_match_qwen38/ ./out/lit_match_qwen38/
rsync -avP user@old:/path/vllm/out/agentx_dspark_dflash_recovery/ ./out/agentx_dspark_dflash_recovery/
```

- Canvases live under Cursor project `canvases/` (may need manual copy if not in this repo).

---

## 9. Quick command cheatsheet

```bash
# Gate summary
python3 benchmarks/ultraquant_mtp/lit_match_dspark_dflash/summarize_gates.py
python3 benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery/summarize_gates.py

# SWA accept probe only
HIP_VISIBLE_DEVICES=<i> PORT=8730 \
  bash benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery/run_swa_probe.sh

# AgentX FP8 lane only
LANE=fp8 HIP_VISIBLE_DEVICES=<i> PORT=8720 MODE=full \
  bash benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery/run_lane.sh
```

---

## 10. Success definition (end state)

1. Phase 1 lit-match gates pass (or documented triage if ROCm lands below NVIDIA lit ×).  
2. Phase 2 AgentX DSpark + DFlash ≥1.15× nospec @C32, accept≥2.5, prefix hit≥70%.  
3. Canvases + FINDINGS updated.  
4. Next research experiments from backlog queued with quality gates (H5 first).
