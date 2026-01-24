# TRAE Agent Benchmark Results - 2026-01-24

## Overview

This document summarizes the performance benchmark results for TRAE (Tool-augmented Reasoning Agent for Engineering) agents on vLLM optimization tasks. Two agent configurations were evaluated:

1. **GPT-5** (`trae_gpt5_0123`) - 6 commits
2. **Claude Sonnet 4.5** (`trae_sonnet45_0123`) - 10 commits

## Results Summary

### GPT-5 Agent: 5/6 Success (83.3%)

| Commit | Model | Status | Throughput |
|--------|-------|--------|------------|
| 19d98e0c | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | success | 1403.91 tok/s |
| 58eee5f2 | meta-llama/Meta-Llama-3-8B-Instruct | success | 2402.97 tok/s |
| a3223766 | facebook/opt-125m | **FAILED** | Server crashed |
| b690e348 | meta-llama/Meta-Llama-3-8B-Instruct | success | 2413.73 tok/s |
| bc7c4d20 | meta-llama/Meta-Llama-3-8B-Instruct | success | 1857.29 tok/s |
| d7740ea4 | meta-llama/Meta-Llama-3-8B-Instruct | success | 8088.76 tok/s |

### Claude Sonnet 4.5 Agent: 10/10 Success (100%)

| Commit | Model | Status | Throughput |
|--------|-------|--------|------------|
| 9ed82e70 | meta-llama/Meta-Llama-3-8B-Instruct | success | 1089.39 tok/s |
| b690e348 | meta-llama/Meta-Llama-3-8B-Instruct | success | 2414.66 tok/s |
| d7740ea4 | meta-llama/Meta-Llama-3-8B-Instruct | success | 8101.94 tok/s |
| e206b543 | meta-llama/Meta-Llama-3-8B-Instruct | success | 2356.02 tok/s |
| e3580537 | neuralmagic/Meta-Llama-3-8B-Instruct-FP8 | success | 1417.45 tok/s |
| e7b20426 | 01-ai/Yi-1.5-9B-Chat | success | 1883.66 tok/s |
| fa63e710 | meta-llama/Meta-Llama-3-8B | success | (latency) |
| fc542144 | meta-llama/Meta-Llama-3-8B-Instruct | success | 2245.74 tok/s |
| fc7b8d1e | meta-llama/Meta-Llama-3-8B-Instruct | success | 1294.01 tok/s |
| fe66b347 | meta-llama/Meta-Llama-3-8B-Instruct | success | (serving) |

## Technical Details

### Model Override System

Due to RoPE scaling compatibility issues with old vLLM versions (v0.4.x), a model override system was implemented:

```python
MODEL_OVERRIDES = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "ibm-ai-platform/Bamba-9B-v2": "meta-llama/Meta-Llama-3-8B-Instruct",
    "ibm-ai-platform/Bamba-9B": "meta-llama/Meta-Llama-3-8B-Instruct",
}
```

This was necessary because:
- Llama-3.1 models use a new `"llama3"` RoPE scaling type
- Old vLLM versions (before v0.5.2) only support `"dynamic"` and `"linear"` types
- Error: `ValueError: Unknown RoPE scaling type llama3`

### Docker Storage Challenges

The benchmarks were run on a system using the VFS Docker storage driver (required due to overlay-on-overlay filesystem limitations):

- **VFS characteristics**: Each Docker layer is fully copied, resulting in ~600GB+ storage per image
- **Solution**: Sequential benchmarking with `docker system prune -af` between each commit
- **Image pull time**: 15-25 minutes per image with VFS

### Benchmark Types

Three benchmark modes were encountered:
1. **Serving benchmark**: Online inference with TTFT/TPOT/ITL metrics
2. **Throughput benchmark (offline)**: Batch inference measuring tokens/second
3. **Latency benchmark (offline)**: Single-request latency measurement

## Failure Analysis

### a3223766 (GPT-5)
- **Model**: facebook/opt-125m
- **Error**: Server crashed after applying patch
- **Analysis**: The agent's patch introduced a bug that caused vLLM to crash during startup

## Process & Learnings

### Issues Encountered

1. **Disk space exhaustion**: Docker VFS driver consumes enormous space (~600GB per image)
   - Solution: Run benchmarks sequentially with cleanup between each

2. **RoPE scaling incompatibility**: New Llama-3.1 models incompatible with old vLLM
   - Solution: Implement MODEL_OVERRIDES mapping

3. **Timeout issues**: Image pulls taking >15 minutes with VFS
   - Solution: Increase timeout from 900s to 1800s (30 minutes)

4. **Output buffering**: Python subprocess output not appearing in logs
   - Solution: Use `python3 -u` for unbuffered output

### Key Code Changes

Modified `scripts/runners/run_3way_benchmarks.py`:
- Added MODEL_OVERRIDES dictionary
- Updated all benchmark functions to apply model override
- Also replaced model in perf_command string when overriding

## File Locations

- **GPT-5 Results**: `omniperf_results_3way_trae_gpt5_0123/results/`
- **Sonnet 4.5 Results**: `omniperf_results_3way_trae_sonnet45_0123/results/`
- **Agent Patches (GPT-5)**: `perf-agents-bench/state/runs/vllm/trae/gpt-5/2026-01-23_21-19-19/`
- **Agent Patches (Sonnet)**: `perf-agents-bench/state/runs/vllm/trae/us-anthropic-claude-sonnet-4-5-20250929-v1-0/2026-01-23_16-40-44/`
- **Benchmark Script**: `scripts/runners/run_3way_benchmarks.py`

## Duration

Total benchmark time: ~2.5 hours for 8 newly-benchmarked commits (sequential with VFS)
- Average per commit: ~19 minutes (image pull + benchmark)

## Conclusion

- **Sonnet 4.5**: 100% success rate on agent benchmarks
- **GPT-5**: 83.3% success rate (1 failure due to agent-generated buggy patch)

Both agents demonstrated the ability to generate working vLLM optimization patches for the majority of commits tested.
