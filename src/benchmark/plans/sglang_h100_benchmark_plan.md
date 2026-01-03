# SGLang H100 Single GPU Benchmark Plan

## Overview
- **Total SGLang commits**: 74
- **H100 single GPU commits**: 57 (77%)
- **Estimated time per commit**: ~10-15 min (parallel 3-way)
- **Total estimated time**: ~10-14 hours (sequential) or ~3-4 hours (batched)

## Prerequisites
- [x] All 74 human commit Docker images on DockerHub
- [x] All 74 baseline commit Docker images on DockerHub
- [x] Updated benchmark code to use separate Docker images per phase
- [ ] Agent patches available for commits

## Execution Strategy

### Phase 1: Verify Agent Patches Exist
Check which H100 commits have Claude Code patches available.

### Phase 2: Run Benchmarks
Run in batches of 5-10 commits to avoid overwhelming Modal.

```bash
# Single commit test
python3 -m src.benchmark.run_single_commit <commit> --repo sglang --parallel

# Batch run (all H100 commits)
python3 -m src.benchmark.run_batch_commits --repo sglang --hardware H100 --parallel
```

### Phase 3: Collect Results
Results saved to `perf-agents-bench/state/runs/sglang/` with metrics.

## H100 Single GPU Commits (57 total)

| # | Commit | PR | Subject |
|---|--------|-----|---------|
| 1 | 021f76e4f498 | 6994 | Refactor LoRAManager to eliminate stream syncs |
| 2 | 09deb20deef8 | 420 | Optimize memory usage of logits processor |
| 3 | 132dad874d2e | 6922 | Optimize transfer queue forward logic |
| 4 | 148254d4db8b | 2705 | Improve moe reduce sum kernel performance |
| 5 | 187b85b7f384 | 7393 | Optimize custom mem pool usage |
| 6 | 1acca3a2c685 | 5969 | FA3 speed up: skip len operation |
| 7 | 1bf1cf195302 | 375 | Reduce overhead when fork(1) |
| 8 | 25c83fff6a80 | 5558 | Vocabulary Parallelism for LM Head |
| 9 | 2854a5ea9fbb | 1496 | Fix overhead due to penalizer |
| 10 | 2a413829f42b | 5955 | Add triton version as fused_moe config key |
| 11 | 2a754e57b052 | 579 | 2x performance for large prefill |
| 12 | 2bd18e2d767e | 2901 | Memory pool: Minor optimize |
| 13 | 2f42749184ca | 6474 | Fix topk inference performance |
| 14 | 3212c2ad3f7e | 6003 | VLM: optimize tensor transport |
| 15 | 4418f599a546 | 5624 | Fix FA3 DeepSeek prefill regression |
| 16 | 5239d79568f3 | 5188 | Speedup shared expert weight construction |
| 17 | 564a898ad975 | 619 | Optimize mem indices management |
| 18 | 62757db6f0f0 | 1010 | Reduce overhead when cache disabled |
| 19 | 6a2941f4d037 | 625 | Improve tensor parallel performance |
| 20 | 6b231325b978 | 6649 | Replace Queue to FastQueue |
| 21 | 6b7038babd56 | 4695 | Speedup warmup when DP > 1 |
| 22 | 6cb00c639812 | 6761 | Optimize time out logic |
| 23 | 6e2da5156176 | 6178 | Replace time.time() to perf_counter() |
| 24 | 6f560c761b2f | 117 | Improve streaming and first token latency |
| 25 | 6fc175968c3a | 5945 | Optimize pad operation to accelerate 25us |
| 26 | 73b13e69b420 | 7285 | Optimize DP attn scheduling |
| 27 | 79961afa8281 | 6077 | Optimize pad operations in FA3 |
| 28 | 7ce360689145 | 1738 | Faster overlap mode scheduler |
| 29 | 880221bd3b3e | 7968 | Revert batch transfer |
| 30 | 8f8f96a6217e | 1773 | Fix perf regression from stop_token_ids |
| 31 | 912788c095c9 | 6273 | Optimize local_block_table allocation |
| 32 | 9183c23eca51 | 2695 | Speed up update_weights_from_tensor |
| 33 | 9216b10678a0 | 394 | Improve full parallel performance |
| 34 | 9c064bf78af8 | 1587 | Speedup multi-LoRA serving |
| 35 | 9c088829ee2a | 5786 | Revert device_id in dist init |
| 36 | 9c745d078e29 | 2056 | Update xgrammar constrained decoding |
| 37 | a191a0e47c2f | 6593 | Improve two batch overlap |
| 38 | a37e1247c183 | 7724 | Use pybase64 instead of base64 |
| 39 | a99801e0750f | 8133 | Optimize TokenToKVPoolAllocator |
| 40 | ab4a83b25909 | 1339 | Optimize schedule |
| 41 | ac971ff633de | 658 | Reduce ttft and itl with stream_interval |
| 42 | b170930534ac | 1697 | Radix tree code optimize |
| 43 | b1e5a33ae337 | 6960 | Eliminate stream sync for LoRA batch init |
| 44 | b77a02cdfdb4 | 1752 | Support xgrammar and outlines |
| 45 | bb3a3b6675b1 | 137 | Faster JSON decoding for llava |
| 46 | c087ddd6865a | 6627 | Refine pre_reorder_triton_kernel |
| 47 | c2f212d672cc | 2966 | Optimize lightning_attn_decode triton |
| 48 | c98e84c21e43 | 1589 | Use torch.argmax for greedy sampling |
| 49 | d1112d8548eb | 2797 | Endpoint for file support |
| 50 | da47621ccc4f | 7058 | Minor speedup topk postprocessing |
| 51 | dc67d9769382 | 1319 | Speedup load safetensors |
| 52 | dd1012fcbe2a | 6764 | Fix perf spike from tracker gc |
| 53 | ddcf9fe3beac | 3731 | Optimize triton attention custom mask |
| 54 | df7f61ee7d23 | 6812 | Speed up rebalancing |
| 55 | e3ec6bf4b65a | 6814 | Minor speed up block_quant_dequant |
| 56 | e5db40dcbce6 | 1694 | ORJson faster serialization |
| 57 | e822e5900b98 | 364 | Optimize radix tree matching |

## Cost Estimate
- **GPU**: H100 @ ~$3/hr on Modal
- **Per commit**: 3 GPUs x 15 min = 0.75 GPU-hours = ~$2.25
- **57 commits**: ~$130 total

## Next Steps
1. Check which commits have agent patches
2. Run test on 1 commit to verify setup
3. Run batch of 5 commits
4. Scale to full 57 commits
