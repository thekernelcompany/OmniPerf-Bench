# yX4G — Bottleneck-family distribution of Level 1 (generated)

Fresh labeling pass over the 54 Level-1 commits: the collection pipeline never stored a bottleneck-area field, so each task was assigned one primary family from its commit subject and touched paths. Labels live in `scripts/rebuttal_analysis/bottleneck_categories.py` and are auditable per commit.

| Bottleneck family | vLLM | SGLang | Total |
|---|---|---|---|
| Sampling and logits | 6 | 1 | 7 |
| Attention kernels and backends | 5 | 2 | 7 |
| Scheduling and batching | 4 | 3 | 7 |
| General CPU overhead | 6 |  | 6 |
| Host-device memory traffic | 6 |  | 6 |
| Prefill-decode disaggregation |  | 5 | 5 |
| KV cache and block management | 4 |  | 4 |
| MoE and expert parallelism | 2 | 2 | 4 |
| Tokenization and frontend | 3 |  | 3 |
| Structured output | 2 |  | 2 |
| Quantization |  | 1 | 1 |
| Speculative decoding | 1 |  | 1 |
| LoRA |  | 1 | 1 |
| **Total** | **39** | **15** | **54** |

Families represented: 13. vLLM spans 10, SGLang 7.

Per-task labels:

### vllm

- `3b61cb45` **Attention kernels and backends** — [V1] Further reduce CPU overheads in flash-attn (#10989)
- `8c1e77fb` **Attention kernels and backends** — [Kernel] Update vllm-flash-attn version to reduce CPU overheads (#10742)
- `98f47f2a` **Attention kernels and backends** — [V1] Optimize the CPU overheads in FlashAttention custom op (#10733)
- `9f1710f1` **Attention kernels and backends** — Fix mla prefill context performance (#13897)
- `bc7c4d20` **Attention kernels and backends** — [Kernel][ROCM] Upstream prefix prefill speed up for vLLM V1 (#13305)
- `6a417b86` **General CPU overhead** — fix neuron performance issue (#13589)
- `6dd94dbe` **General CPU overhead** — [perf] fix perf regression from #12253 (#12380)
- `89a84b0b` **General CPU overhead** — [Core] Use array to speedup padding (#6779)
- `9badee53` **General CPU overhead** — Fix performance when `--generation-config` is not `None` (#14223)
- `9ed82e70` **General CPU overhead** — [Misc] Small perf improvements (#6520)
- `fc7b8d1e` **General CPU overhead** — [Performance] e2e overheads reduction: Small followup diff (#7364)
- `296f927f` **Host-device memory traffic** — [Model] RE: Mamba2 Prefill Performance Tweaks: Fixing Flurry of Unnecessary Memory Copie
- `310aca88` **Host-device memory traffic** — [perf]fix current stream (#11870)
- `70b808fe` **Host-device memory traffic** — [Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync (#14377)
- `b55ed6ef` **Host-device memory traffic** — [V1][Minor] Optimize token_ids_cpu copy (#11692)
- `b690e348` **Host-device memory traffic** — [Model] Mamba2 preallocate SSM output tensor to avoid d2d copy overhead (#21075)
- `fe66b347` **Host-device memory traffic** — [Model] Mamba2 Prefill Performance Tweaks: Fixing Flurry of Unnecessary Memory Copies  (
- `2deb029d` **KV cache and block management** — [Performance][BlockManagerV2] Mark prefix cache block as computed after schedule (#7822)
- `3476ed08` **KV cache and block management** — [Core] Optimize block_manager_v2 vs block_manager_v1 (to make V2 default)  (#5602)
- `660470e5` **KV cache and block management** — [Core] Optimize evictor-v2 performance (#7193)
- `9474e89b` **KV cache and block management** — [PREFIX CACHING FOLLOW UP] A bunch of fixes to block allocator performance when automati
- `19d98e0c` **MoE and expert parallelism** — [Kernel] Optimize moe intermediate_cache usage (#13625)
- `e7b20426` **MoE and expert parallelism** — Revert "[Performance] Performance improvements in non-blockwise fp8 CUTLASS MoE (#20762)
- `299ebb62` **Sampling and logits** — [Core] Speed up decode by remove synchronizing operation in sampler (#16436)
- `30172b49` **Sampling and logits** — [V1] Optimize handling of sampling metadata and req_ids list (#13244)
- `35fad35a` **Sampling and logits** — [V1][Sampler] Faster top-k only implementation (#15478)
- `99abb8b6` **Sampling and logits** — [V1][Spec Decode] Optimize Rejection Sampler with Triton Kernels (#14930)
- `a3223766` **Sampling and logits** — [Core] Optimize update checks in LogitsProcessor (#21245)
- `d7740ea4` **Sampling and logits** — [Core] Optimize sampler get_logprobs (#4594)
- `6e36f4fa` **Scheduling and batching** — improve chunked prefill performance
- `ad8d696a` **Scheduling and batching** — [Core] Scheduler perf fix (#4270)
- `e3580537` **Scheduling and batching** — [Performance] Enable chunked prefill and prefix caching together (#7753)
- `fa63e710` **Scheduling and batching** — [V1][Perf] Reduce scheduling overhead in model runner after cuda sync (#12094)
- `4c822298` **Speculative decoding** — [V1][Spec Decode] Optimize N-gram matching with Numba (#13365)
- `e206b543` **Structured output** — [v0][Core] Use xgrammar shared context to avoid copy overhead for offline engine (#13837
- `fc542144` **Structured output** — [Feature] Fix guided decoding blocking bitmask memcpy (#12563)
- `015069b0` **Tokenization and frontend** — [Misc] Optimize the Qwen3_ReasoningParser extract_reasoning_content (#17515)
- `22d33bac` **Tokenization and frontend** — [FrontEnd][Perf] `merge_async_iterators` fast-path for single-prompt requests (#15150)
- `58eee5f2` **Tokenization and frontend** — [PERF] Use faster way of decode in tokenizer: avoid useless list-to-list conversion (#20

### sglang

- `1acca3a2` **Attention kernels and backends** — FA3 speed up: skip len operation and get batch size directly from forward batch (#5969)
- `205d5cb4` **Attention kernels and backends** — perf: Optimize local attention memory allocation in FlashAttentionBackend (#6356)
- `021f76e4` **LoRA** — [Perf] Refactor LoRAManager to eliminate stream syncs and redundant computations  (#6994
- `c087ddd6` **MoE and expert parallelism** — Refine pre_reorder_triton_kernel slightly to improve performance (#6627)
- `df7f61ee` **MoE and expert parallelism** — Speed up rebalancing when using non-static dispatch algorithms (#6812)
- `132dad87` **Prefill-decode disaggregation** — [PD] Optimize transfer queue forward logic for dummy rank (#6922)
- `187b85b7` **Prefill-decode disaggregation** — [PD] Optimize custom mem pool usage and bump mooncake version (#7393)
- `2ed68d7a` **Prefill-decode disaggregation** — [PD Disaggregation] replace transfer with batch transfer for better performance (#7236)
- `6b231325` **Prefill-decode disaggregation** — [PD Perf] replace Queue to FastQueue (#6649)
- `dd1012fc` **Prefill-decode disaggregation** — [PD] Fix potential perf spike caused by tracker gc and optimize doc (#6764)
- `e3ec6bf4` **Quantization** — Minor speed up block_quant_dequant (#6814)
- `da47621c` **Sampling and logits** — Minor speedup topk postprocessing (#7058)
- `31589e17` **Scheduling and batching** — Speed up when having padding tokens two-batch overlap (#6668)
- `73b13e69` **Scheduling and batching** — Optimize DP attn scheduling for speculative decoding (#7285)
- `a191a0e4` **Scheduling and batching** — Improve performance of two batch overlap in some imbalanced cases (#6593)

