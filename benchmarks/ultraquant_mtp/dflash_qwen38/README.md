# Qwen3.8-27B × DFlash2 (new-code stream)

Bring-up stream for DFlash2 on Qwen3.8-27B. **New files only** — does not
edit shipping MTP, PDV, or existing DFlash modules.

## Locked choices

| Piece | Value |
| --- | --- |
| Target | `/huggingface/pretrained_models/Qwen/Qwen3.8-27B` |
| Draft | `/workspace/pretrained_models/z-lab/Qwen3.8-27B-DFlash2` (HF mount is RO) |
| Runner | `VLLM_USE_V2_MODEL_RUNNER=1` |
| Target attn / KV (P1) | `ROCM_AITER_UNIFIED_ATTN` + `fp8` |
| Draft attn / KV | `ROCM_AITER_FA` + `auto` (non-causal) |
| K | 7 speculative tokens |
| Phase 3 (later) | New `ULTRAQUANT_DFLASH` backend + `dflash_flydsl_*` |

## Phases

1. **P1 FP8** — `run_qwen38_dflash2_fp8.sh` (this directory)
2. **P2 UQ** — target `ultraquant_4bit`, shipping PDV if qlen fits
3. **P3 FlyDSL** — new backend + specialized kernel

## Fetch draft

```bash
bash benchmarks/ultraquant_mtp/dflash_qwen38/fetch_draft.sh
```

## P1 run (HIP 7)

```bash
HIP_VISIBLE_DEVICES=7 PDV_EXP_PORT=8360 \
  bash benchmarks/ultraquant_mtp/dflash_qwen38/run_qwen38_dflash2_fp8.sh
```

Logs: `/workspace/logs/dflash-qwen38-fp8.log`  
Out: `vllm/out/dflash_qwen38_fp8/`

## AgentX debug (0% accept / GPU fault bisect)

Diagnosis: long + `--enable-prefix-caching` faults (~81k+ cold prefill);
noprefix is fine. Root cause: DFlash was missing from
`_uses_trailing_mtp_layers`, so draft KV groups stayed unannotated and the
coordinator flagged **every** group (including GDN/Mamba). Fix: annotate
`method=="dflash"` like MTP (see `kv_cache_utils.py`). The draft-KV warning
was a symptom of that hole.

```bash
# Serve-only for the length bisect (conc=1, eager)
HIP_VISIBLE_DEVICES=7 DFLASH_PORT=8361 DFLASH_CONC=1 DFLASH_EAGER=1 \
  DFLASH_MODE=serve \
  bash benchmarks/ultraquant_mtp/dflash_qwen38/run_qwen38_dflash2_fp8_agentx_debug.sh

# In another shell, once READY:
python benchmarks/ultraquant_mtp/dflash_qwen38/probe_long_prompt_acceptance.py \
  --port 8361 --model /huggingface/pretrained_models/Qwen/Qwen3.8-27B \
  --lengths 4096,16384,49152,131072,200000 --repeat-for-prefix-hit

# Full AgentX retry with the same toggles: omit DFLASH_MODE (defaults to sa)
```

Useful toggles: `DFLASH_NO_PREFIX_CACHE=1`, `DFLASH_MNBT=32768`,
`DFLASH_SPEC=0` (target-only), `DFLASH_SERIALIZE=1` (stack on fault),
`DFLASH_MAX_LEN=65536` (quick health).

## P2 run (UltraQuant / shipping PDV)

```bash
HIP_VISIBLE_DEVICES=7 DFLASH_PORT=8370 DFLASH_GPU_UTIL=0.40 DFLASH_SMOKE_ONLY=1 \
  bash benchmarks/ultraquant_mtp/dflash_qwen38/run_qwen38_dflash2_uq.sh
```

Out: `vllm/out/dflash_qwen38_uq/`

## P3 run (ULTRAQUANT_DFLASH + dflash_flydsl)

```bash
HIP_VISIBLE_DEVICES=7 DFLASH_PORT=8371 DFLASH_GPU_UTIL=0.40 DFLASH_SMOKE_ONLY=1 \
  bash benchmarks/ultraquant_mtp/dflash_qwen38/run_qwen38_dflash2_flydsl.sh
```

Selects `VLLM_ULTRAQUANT_PDV_EXPERIMENT=dflash` → `dflash_flydsl_decode_hd256`
(currently aliases shipping PDV; specialized math lands in that file later).

Out: `vllm/out/dflash_qwen38_flydsl/`
