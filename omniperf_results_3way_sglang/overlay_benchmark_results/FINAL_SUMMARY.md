# SGLang 3-Way Benchmark Final Results

## Executive Summary

Out of **17 commits** attempted with the Python overlay approach, only **1 commit** (2a754e57) produced valid benchmark results. This is because the base Docker image (`sglang-rebuilt-2a754e57`) was built at a specific point in SGLang's development (June 2024, v0.1.17), and even commits from similar timeframes have incompatible code structures.

## Successful Benchmark: 2a754e57 (PR #579)

**Title**: "2x performance improvement for large prefill & Fix workspace conflicts"

### Configuration
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Batch size: 4
- Input length: 1024 tokens
- Output length: 256 tokens
- GPU: NVIDIA H100 PCIe

### Results Summary (3 Runs)

| Run | Baseline | Human | claude_code | codex | trae_gpt5 | trae_sonnet45 |
|-----|----------|-------|-------------|-------|-----------|---------------|
| 1 | 1735.85 | 2126.74 (+22.5%) | 2168.18 (+25.0%) | 2095.89 (+20.8%) | 2158.62 (+24.4%) | 1871.22 (+7.8%) |
| 2 | 1832.22 | 2141.03 (+16.9%) | 2168.18 (+18.3%) | 1755.96 (-4.2%) | 2130.12 (+16.3%) | 1871.22 (+2.1%) |
| 3 | 2134.22 | 2141.38 (+0.3%) | 2159.14 (+1.2%) | 2132.60 (-0.1%) | 2148.85 (+0.7%) | 2176.66 (+2.0%) |

### Key Observations

1. **Significant Run Variance**: Baseline throughput varies from 1735 to 2134 tokens/s between runs (~23% variance)
2. **Human vs Agent Performance**: In some runs, agent patches outperform human optimization
3. **Best Performers**:
   - Run 1: claude_code (+25.0%)
   - Run 2: claude_code (+18.3%)
   - Run 3: trae_sonnet45 (+2.0%)

## Failed Commits Analysis

### Why Other Commits Failed

The overlay approach works by:
1. Cloning SGLang repo inside Docker
2. Checking out specific commits
3. Using PYTHONPATH to override installed packages

This fails when:
- Code structure changed (different file paths, class names)
- Different dependencies required (vllm versions, flashinfer, etc.)
- Import paths changed between versions

### Commits Attempted (27 total across runs)

| Era | PR Range | Commits Attempted | Worked |
|-----|----------|-------------------|--------|
| June-July 2024 | <1000 | 10 | 1 (2a754e57) |
| Aug-Oct 2024 | 2000-3000 | 2 | 0 |
| Nov 2024 | 3000-5000 | 1 | 0 |
| Dec 2024 | 5000-7000 | 10 | 0 |
| Jan 2025 | >7000 | 4 | 0 |

## Recommendations for Benchmarking More Commits

### Option 1: Build Era-Specific Docker Images
- Build separate images at each major code version
- Time-consuming (~30-60 min per build)
- Most reliable approach

### Option 2: Use Pre-built Images with Fixes
- ayushnangia16 and shikhar481 repos have images
- Many are incomplete (missing vllm, etc.)
- Need to install missing dependencies

### Option 3: Era Grouping
- Identify commit clusters that share code structure
- Build one image per cluster
- Reduces total builds needed

## Files Generated

```
overlay_benchmark_results/
├── 2a754e57_3way.json      # Successful benchmark
├── 09deb20d_3way.json      # Failed (incompatible)
├── 1bf1cf19_3way.json      # Failed (incompatible)
├── 148254d4_3way.json      # Failed (incompatible)
├── ... (other failed results)
├── script_*.sh             # Generated Docker scripts
└── FINAL_SUMMARY.md        # This file
```

## Technical Details

### Docker Image Used
- Name: `sglang-rebuilt-2a754e57`
- SGLang Version: 0.1.17
- PyTorch: 2.3.0+cu121
- Python: 3.11
- Base: nvidia/cuda:12.4.1-devel-ubuntu22.04

### Agent Patches Tested
- **claude_code**: `/perf-agents-bench/state/runs/sglang/claude_code/default/2025-12-23_06-28-44/`
- **codex**: `/perf-agents-bench/state/runs/sglang/codex/gpt-5/389be848/`
- **trae_gpt5**: `/perf-agents-bench/state/runs/sglang/trae/gpt-5/2025-11-14_21-05-32/`
- **trae_sonnet45**: `/perf-agents-bench/state/runs/sglang/trae/claude-sonnet-45/2025-11-28_15-26-15/`

## Conclusion

The Python overlay approach is limited to commits that match the base Docker image's code structure. For comprehensive benchmarking across all 17+ commits, building commit-specific or era-specific Docker images is required. The 2a754e57 benchmark results show that AI agents (particularly claude_code and trae_gpt5) can match or slightly exceed human optimization performance on this particular commit.
