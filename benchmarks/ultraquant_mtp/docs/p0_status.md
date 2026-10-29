# P0 follow-ups status (MI355X)

Updated: 2026-10-21

## Geometry smoke (`out/p0_kv_geometry/`)

| Mode | Result | LCM | Notes |
|------|--------|-----|-------|
| `dflash_fp8_bs64` | **PASS** | **1664** | Target 1664 ×14 + draft **832** (no pad). Primary P0-1 win. |
| `dspark_fp8_bs64` | partial | **3328** | Target 1664 OK; draft bf16 groups still **3328** (2×). |
| `*_bs64_kv8` | **FAIL** | — | `ValueError: Mixed FP8 and non-FP8 tensors` in aiter FA (`utils.py`). Draft Q bf16 + KV fp8 rejected. |

## Artifacts ready for post-matrix runs

- DFlash nowin draft: `pretrained_models/z-lab/Qwen3.8-27B-DFlash2-nowin/`
- AgentX P0-1 DFlash: `run_dflash_fp8_p01_agentx.sh` → `out/dflash_qwen38_fp8_p01/`
- AgentX P0-1 DSpark (bs64): `run_dspark_fp8_p01_agentx.sh` → `out/dspark_qwen38_fp8_p01/`
- Full P0-1 conc matrix: `ROLE=p01` in `run_dflash_dspark_agentx_conc_sweep.sh` (concs 8…32)
- Accept bisect: `run_dflash_accept_bisect.sh` → `out/dflash_accept_bisect/{window,nowin}/`
- Queue (waits for baseline UQ, then full p01 matrix + bisect): `run_p0_post_matrix_queue.sh`

## Compare vs baseline matrix

Baseline lives under `out/{dflash,dspark}_qwen38_{fp8,uq}/sa/seed42_c*`.
P0-1 numbers go to `*_fp8_p01/` so they do not overwrite the matrix.
UQ already served with block-size 64; FP8 P0-1 is the geometry follow-up being re-measured.

## Next code fix (kv8)

Need draft FA path that accepts bf16 Q with fp8 paged KV (or dequant KV before FA), without editing shipping kernels — new backend experiment.

## UQ bottlenecks (follow-up research)

Full note: [`uq_dflash_dspark_bottlenecks.md`](./uq_dflash_dspark_bottlenecks.md) (helper: `uq_agentx_cell_analysis.py`).

**Already on UQ:** `--block-size 64` + fast decode knobs (same as MTP UQ/PDV). Do not re-run block-64 as a UQ “fix”.

**Still broken (ranked):** (1) all-or-nothing prefix zero-hit recompute storm, (2) tiny DSpark UQ KV pool from bf16 draft pages, (3) LCM 12544 second-order, (4) DFlash accept session flip stuck bad on UQ, (5) DSpark UQ c8 accept anomaly.

**UQ fix queue (running on HIP0/2 — does not touch Stream 2 / P0-1):**

| Exp | What | Out |
|-----|------|-----|
| E1 | `MNBT=2048` DSpark→DFlash UQ c32 | `out/*_uq_mnbt2048/` |
| E2 | `--prefix-cache-retention-interval` 50176 / 25088 | `out/*_uq_ret50k/`, `*_ret25k/` |
| E4 | `ROCM_AITER_FA_UQ_DENSE` (draft page 784) smoke → AgentX c32 | `out/*_uq_dense/` |
| E5 | accept-vs-length probe | `out/uq_accept_probe/` |

Runner: `run_uq_fix_queue.sh` · logs: `logs/uq-fix-queue.log`, `logs/uq-spec-ab-*.log`
