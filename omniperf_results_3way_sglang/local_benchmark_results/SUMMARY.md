# SGLang 3-Way Benchmark Results - Local Git Checkout Approach

## Executive Summary

Successfully benchmarked **1 out of 13** target commits using the local git checkout + uv pip install approach. The other commits fail due to complex dependency version conflicts between different eras of SGLang.

## Successful Benchmark: 2a754e57 (PR #579)

**Title**: "2x performance improvement for large prefill & Fix workspace conflicts"
**Era**: June 2024 (SGLang v0.1.17)
**Dependencies**: vllm==0.5.0, torch==2.3.0, flashinfer==0.2.1

### Results (Run 1)

| Variant | Total Throughput | vs Baseline |
|---------|------------------|-------------|
| baseline | 2,872 tok/s | - |
| human | 3,450 tok/s | +20.1% |
| claude_code | 2,951 tok/s | +2.8% |
| trae_gpt5 | 3,359 tok/s | +17.0% |

### Results (Run 2)

| Variant | Total Throughput | vs Baseline |
|---------|------------------|-------------|
| baseline | 2,893 tok/s | - |
| human | 2,928 tok/s | +1.2% |
| claude_code | 3,383 tok/s | +17.0% |
| trae_gpt5 | 3,424 tok/s | +18.3% |

**Note**: Run variance is significant (~20% for human optimization). Multiple runs recommended for reliable comparison.

## Failed Commits (12 total)

| Commit | PR# | Era | Failure Reason |
|--------|-----|-----|----------------|
| 187b85b7 | #7393 | Jan 2025 | Import error (sgl-kernel ABI) |
| 6b231325 | #6649 | Jan 2025 | Import error (transformers.masking_utils) |
| 6cb00c63 | #6761 | Jan 2025 | Import error (bench_one_batch deps) |
| 148254d4 | #2705 | Oct 2024 | Import error (outlines/pyarrow conflict) |
| 2bd18e2d | #2901 | Oct 2024 | Import error (bench_one_batch deps) |
| 880221bd | #7968 | Jan 2025 | Import error (bench_one_batch deps) |
| b1e5a33a | #6960 | Jan 2025 | Import error (bench_one_batch deps) |
| da47621c | #7058 | Jan 2025 | Not attempted |
| dd1012fc | #6764 | Jan 2025 | Not attempted |
| ddcf9fe3 | #3731 | Nov 2024 | Not attempted |
| df7f61ee | #6812 | Jan 2025 | Not attempted |
| e3ec6bf4 | #6814 | Jan 2025 | Not attempted |

## Root Cause Analysis

Different SGLang commits have incompatible dependency requirements:

### June 2024 Era (Works)
- `bench_latency.py` benchmark script
- vllm == 0.5.0
- torch == 2.3.0
- flashinfer == 0.2.1 (can disable with --disable-flashinfer)
- outlines == 0.0.44

### Oct-Nov 2024 Era (Fails)
- `bench_one_batch.py` benchmark script
- vllm == 0.6.x
- sgl-kernel (CUDA extensions)
- Newer outlines with pyarrow conflicts

### Jan 2025 Era (Fails)
- `bench_one_batch.py` benchmark script
- vllm == 0.6.4+
- sgl-kernel with SM90 support
- compressed-tensors
- transformers masking_utils (very recent)

## Recommendations

1. **For June 2024 commits**: Use the local approach with vllm 0.5.0 and torch 2.3.0
2. **For newer commits**: Build era-specific Docker images with pinned dependencies
3. **Alternative**: Use the overlay approach with pre-built Docker images (see overlay_benchmark_results/)

## Files Generated

```
local_benchmark_results/
├── 2a754e57_3way.json  # SUCCESS
├── 148254d4_3way.json  # FAILED
├── 187b85b7_3way.json  # FAILED
├── 2bd18e2d_3way.json  # FAILED
├── 6b231325_3way.json  # FAILED
├── 6cb00c63_3way.json  # FAILED
├── 880221bd_3way.json  # FAILED
├── b1e5a33a_3way.json  # FAILED
└── SUMMARY.md          # This file
```

## Hardware

- GPU: NVIDIA H100 PCIe (SM90), 80GB VRAM
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (dummy weights)
- Batch size: 4
- Input length: 512 tokens
- Output length: 64 tokens
