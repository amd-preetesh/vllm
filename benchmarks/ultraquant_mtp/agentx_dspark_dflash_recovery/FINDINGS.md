# DFlash / DSpark recovery findings

## Phase 1 — lit-matched (HF thinking, short prompts)

Suite: `benchmarks/ultraquant_mtp/lit_match_dspark_dflash/`
Out: `out/lit_match_qwen38/`

Card: T=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, thinking on.
Dataset: 128 prompts from SPEED-747 subsample. C∈{1,8,32}.
Arms: fp8_nospec, fp8_mtp, fp8_dspark, fp8_dflash.

Pass gates (vs same-card nospec): C1 DSpark≥2.0× / DFlash≥2.5×; C8 both≥1.5×; C32 both≥1.1×; accept DSpark≥2.5 / DFlash≥3.0.

See `out/lit_match_qwen38/summary_gates.json` after suite completes.

## Phase 2 — AgentX long-ctx

Suite: `benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery/`
Out: `out/agentx_dspark_dflash_recovery/`

Config deltas vs original AgentX baseline:
- `--block-size 64` (geometry; predicted LCM 1664 FP8)
- `mnbt=8192` (was 2048)
- retention 49920 / 50176 (LCM-aligned)
- FP8 draft `kv_cache_dtype=fp8`
- DFlash draft `ROCM_AITER_FA_DFLASH_SWA` + `VLLM_DFLASH_DRAFT_ATTN_EXPERIMENT=win_table`
- Adaptive arms after geometry

SWA probe: `run_swa_probe.sh` → `out/.../swa_probe/probe_accept_vs_length.txt`
Gate: ctx≥32k & gen≥2560 → accept_len ≥ 2.5

AgentX C32 gates: ≥1.15× baseline nospec, prefix hit ≥70%, accept ≥2.5.
