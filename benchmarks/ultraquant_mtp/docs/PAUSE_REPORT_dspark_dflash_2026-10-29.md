# Pause report — DFlash / DSpark lit-match + AgentX recovery

**Paused:** 2026-10-29 ~15:03 PDT (22:03 UTC)  
**Reason:** Release GPU for another user.  
**GPU (this host only):** index was `HIP_VISIBLE_DEVICES=3`.  
**Stable IDs:** ROCm Unique ID `0xe81e8e9535780fc8`, PCI `0000:75:00.0`.  
**GPU status after halt:** free ≈ **142 / 288 GiB**; no live `vllm serve` / lit-match suite / handoff.  
**Full context / resume:** [`RESEARCH_GOAL_dspark_dflash_recovery.md`](./RESEARCH_GOAL_dspark_dflash_recovery.md)

---

## What was stopped

| Process | Action |
|---|---|
| `lit_match_dspark_dflash/run_suite.sh` | TERM/KILL |
| `run_cell.sh` (was starting `fp8_dflash` C1) | TERM/KILL |
| `vllm serve` port **8710** (DFlash) | TERM/KILL |
| `handoff_phase2.sh` | stopped / disarmed |
| AgentX recovery | already paused earlier |

Markers:

- `/workspace/logs/lit_match_qwen38/suite.PAUSED`
- `/workspace/logs/dspark_dflash_handoff/handoff.PAUSED`
- Removed `suite.DONE` so handoff will **not** auto-start AgentX
- Snapshot: `out/lit_match_qwen38/pause_snapshot.json`

---

## Phase 1 lit-match progress (HF thinking card)

Suite: `benchmarks/ultraquant_mtp/lit_match_dspark_dflash/`  
Out: `out/lit_match_qwen38/`  
Card: T=1.0, top_p=0.95, top_k=20, thinking=xhigh, max_tokens=2048, N=128, HIP=3, util≈0.4432

| Arm | C1 | C8 | C32 |
|---|---|---|---|
| fp8_nospec | **DONE** 93.5 tok/s | **DONE** 576.4 | **DONE** 1834.8 |
| fp8_mtp | **DONE** 138.0 tok/s, accept **2.54** | pending | pending |
| fp8_dspark | **DONE** 166.4 tok/s, accept **3.00** (~**1.78×** nospec) | pending | pending |
| fp8_dflash | **not started** (killed at serve start) | pending | pending |

### Gate check so far (C1 only)

Pass gates: DSpark ≥2.0× / DFlash ≥2.5×; accept DSpark ≥2.5 / DFlash ≥3.0.

| Method | vs nospec C1 | accept | Notes |
|---|---|---|---|
| MTP | 1.48× | 2.54 | Reference |
| DSpark | **1.78×** (gate 2.0× → **miss** so far) | **3.00** (pass) | Healthy accept; speedup short of lit gate at C1 |
| DFlash | — | — | Not measured |

**Bug fixed mid-run (keep):** `run_cell.sh` must pass `--speculative-config` as **JSON string**, not file path (`SPEC_JSON=...`). Without this, all MTP/DSpark/DFlash cells failed argparse.

---

## Phase 2 AgentX (paused / partial)

Suite: `benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery/`  
Out: `out/agentx_dspark_dflash_recovery/`

Config deltas: `block_size=64`, `mnbt=8192`, retention 49920/50176, FP8 draft kv8, DFlash `ROCM_AITER_FA_DFLASH_SWA` + `win_table`, adaptive arms.

| Cell | Status |
|---|---|
| `fp8_dspark_geom64` **C8** | **DONE** (profile present) |
| `fp8_dspark_geom64` C16 | reset / incomplete (no profile) |
| Other arms / concs | not run |
| SWA probe | failed earlier (util 0.55 needed 158 GiB; only ~142 free) — probe script patched to **util 0.45** |

---

## Artifacts to keep (do not delete)

- `out/lit_match_qwen38/**/accept_metrics.json` + DONE cells  
- `out/lit_match_qwen38/pause_snapshot.json`  
- `out/agentx_dspark_dflash_recovery/fp8_dspark_geom64/sa/seed42_c8/`  
- Docs: `docs/dflash_dspark_agentx_bottlenecks.md`, `docs/dspark_dflash_improvement_backlog.md`  
- Canvases: `lit-match-dspark-dflash`, `dspark-dflash-recovery-progress`, `agentx-dspark-dflash-recovery`  
- Research backlog Top-1 next: **H5 prefix-cache draft-context restore** probe

---

## How to resume (when GPU free again)

```bash
# Inside container 4c08f4488bec (or equivalent), from /workspace/vllm:
# 1) Confirm HIP free (>= ~80–140 GiB depending on util)
# 2) Re-arm lit-match (DONE cells SKIP via accept_metrics.json)
rm -f /workspace/logs/lit_match_qwen38/suite.PAUSED
rm -f /workspace/logs/dspark_dflash_handoff/handoff.PAUSED
HIP_A=3 HIP_B= UTIL=0.55 bash benchmarks/ultraquant_mtp/lit_match_dspark_dflash/run_suite.sh &
# 3) Handoff waits for suite.DONE then SWA probe + AgentX (skips seed42_c8 DONE)
HIP=3 bash benchmarks/ultraquant_mtp/lit_match_dspark_dflash/handoff_phase2.sh &
```

Resume order of pending lit-match cells:  
`fp8_dflash:1` → `fp8_mtp:8` → `fp8_dspark:8` → `fp8_dflash:8` → `fp8_nospec:32` (skip) → `fp8_mtp:32` → `fp8_dspark:32` → `fp8_dflash:32`  
(Actual queue is nospec/mtp/dspark/dflash × C1 then C8 then C32; SKIP any with profile+accept.)

After lit-match: summarize_gates + refresh canvases, then AgentX recovery continues from C16.

---

## ETA when resumed (approx)

- Remaining lit-match: ~**2–3.5 h** (7 cells; C1 DFlash longest)  
- Then SWA + AgentX (~23 cells): ~**5–7 h**

---

## Owner notes

- Container: `4c08f4488bec` / `preshah_vllm_ultraquant`  
- Only **one GPU** was in use (HIP **3**).  
- Other GPUs were not claimed by this workstream.
