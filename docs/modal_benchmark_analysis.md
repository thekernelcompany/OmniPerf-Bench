# Modal Multi-GPU Benchmark Analysis

**Date**: 2025-12-26 to 2025-12-27
**Branch**: feature/unified-trae-modal

## Overview

This document captures the work done to run OmniPerf benchmarks on Modal cloud GPUs for commits requiring multi-GPU configurations.

## Objective

Run 3-way benchmark comparisons (Baseline vs Human vs Agent patch) for 7 multi-GPU commits from the OmniPerf dataset that couldn't run on a single H100 80GB.

---

## Implementation Summary

### Files Modified

| File | Changes |
|------|---------|
| `src/eval/modal_benchmark.py` | Added 3-way benchmark functions, updated timeouts to 3600s, fixed GPU mappings |
| `src/eval/native_benchmark_runner.py` | Updated LARGE_MODEL_GPU_MAP for DeepSeek-V2 |

### Key Changes

1. **3-Way Benchmark Functions**: Created `run_3way_benchmark_2gpu()`, `run_3way_benchmark_4gpu()`, `run_3way_benchmark_8gpu()` that run:
   - Baseline wheel benchmark
   - Human wheel benchmark
   - Agent patch benchmark (applied to baseline)

2. **Timeout Fixes**: Increased `wait_for_server()` timeout from 600-900s to 3600s (1 hour) to accommodate large model loading.

3. **GPU Mapping Fixes**: Updated DeepSeek-V2 from H100:4 to H100:8 (236B MoE needs 8x H100 per HuggingFace model card).

---

## Multi-GPU Commits Tested

| Commit | Model | GPU Config | Parameters |
|--------|-------|------------|------------|
| 8aa1485f | meta-llama/Llama-4-Scout-17B-16E-Instruct | H100:2 | 17B x 16 experts |
| bd6028d6 | RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8 | H100:2 | 17B x 16 experts |
| 310aca88 | meta-llama/Meta-Llama-3-70B | H100:4 | 70B |
| ac45c44d | deepseek-ai/DeepSeek-V2 | H100:8 | 236B MoE |
| baeded25 | deepseek-ai/DeepSeek-V3 | H100:8 | 671B MoE |
| 4fb56914 | deepseek-ai/DeepSeek-V3-0324 | H100:8 | 671B MoE |
| 7661e92e | nvidia/Nemotron-4-340B-Instruct | H100:8 | 340B |

---

## Benchmark Results

### Summary

| GPU Config | Commits Tested | Success | Failed |
|------------|----------------|---------|--------|
| H100:2 | 2 | 0 | 2 |
| H100:4 | 1 | 0 | 1 |
| H100:8 | 4 | 0 | 4 |
| **Total** | **7** | **0** | **7** |

### Detailed Results

All benchmarks failed with similar errors:

```json
{
  "results": [
    {"commit_hash": "8aa1485f...", "status": "error", "error": "Baseline server failed to start"},
    {"commit_hash": "bd6028d6...", "status": "error", "error": "Baseline server failed to start"},
    {"commit_hash": "310aca88...", "status": "baseline_failed", "error": "Baseline benchmark produced no metrics"},
    {"commit_hash": "baeded25...", "status": "error", "error": "Baseline server failed to start"},
    {"commit_hash": "ac45c44d...", "status": "error", "error": "Baseline server failed to start"},
    {"commit_hash": "4fb56914...", "status": "error", "error": "Baseline server failed to start"},
    {"commit_hash": "7661e92e...", "status": "error", "error": "Baseline server failed to start"}
  ]
}
```

---

## Failure Analysis

### Error Categories

1. **"Baseline server failed to start"** (6/7 commits)
   - Server process died during initialization
   - vLLM server couldn't load the model

2. **"Baseline benchmark produced no metrics"** (1/7 commits - 310aca88)
   - Server started but benchmark client produced no parseable output

### Root Cause Hypotheses

1. **HuggingFace Gated Model Access**
   - DeepSeek-V3, Llama-4, Nemotron are gated models
   - HuggingFace token may not be correctly passed to Modal containers
   - Check: `modal.Secret.from_name("huggingface-secret")` configuration

2. **vLLM Version Compatibility**
   - Different commits use different vLLM versions (0.5.x to 0.10.x)
   - Older versions may not support newer model architectures
   - DeepSeek-V3 support was added in vLLM ~0.7.0

3. **Model Weight Download Timeout**
   - 671B models (DeepSeek-V3) take significant time to download
   - Modal container may timeout during download phase
   - Solution: Pre-download to Modal volume

4. **CUDA/Driver Compatibility**
   - Modal base image: `nvidia/cuda:12.4.0-devel-ubuntu22.04`
   - Some vLLM wheels may require different CUDA versions

### Debugging Steps Needed

1. **Check Modal Dashboard Logs**
   ```
   View Deployment: https://modal.com/apps/itssshikhar/main/deployed/omniperf-benchmark
   ```

2. **Verify HuggingFace Token**
   ```bash
   modal secret list
   modal secret show huggingface-secret
   ```

3. **Test with Non-Gated Model**
   - Run benchmark with `facebook/opt-125m` or `google/gemma-2b` to isolate auth issues

4. **Add Verbose Logging**
   - Capture full vLLM server startup logs
   - Log wheel installation output

---

## Timeline

| Time (UTC) | Event |
|------------|-------|
| 19:24 | First attempt with ac45c44d (H100:4) - OOM error |
| 19:32 | Updated GPU mapping: DeepSeek-V2 H100:4 -> H100:8 |
| 19:33 | Second attempt with ac45c44d (H100:8) - user requested reorder |
| 20:04 | Started H100:2 benchmarks (8aa1485f, bd6028d6) |
| 22:10 | H100:2 benchmarks completed - both failed |
| 22:28 | Started H100:4 benchmark (310aca88) |
| 22:40 | H100:4 benchmark completed - failed (no metrics) |
| 23:08 | Started H100:8 benchmarks (4 commits) |
| 00:12 | [1/4] baeded25 completed |
| 01:14 | [2/4] ac45c44d completed |
| 02:15 | [3/4] 4fb56914 completed |
| 03:16 | [4/4] 7661e92e completed - all failed |

**Total Runtime**: ~8 hours
**Estimated Modal Cost**: ~$100-150 (8x H100 for ~4 hours)

---

## Code Changes Detail

### modal_benchmark.py - GPU Mapping Fix

```python
# Before
LARGE_MODEL_GPU_MAP = {
    "deepseek-ai/DeepSeek-V2": "H100:4",  # OOM!
    ...
}

# After
LARGE_MODEL_GPU_MAP = {
    "deepseek-ai/DeepSeek-V2": "H100:8",  # 236B MoE needs 8x H100
    ...
}
```

### modal_benchmark.py - Timeout Fix

```python
# Before
if not wait_for_server(port=8000, timeout=900):  # 15 min

# After
if not wait_for_server(port=8000, timeout=3600):  # 1 hour
```

### native_benchmark_runner.py - GPU Mapping Sync

```python
LARGE_MODEL_GPU_MAP = {
    "deepseek-ai/DeepSeek-V2": "H100:8",  # Synced with modal_benchmark.py
    ...
}
```

---

## Recommendations

### Immediate Actions

1. **Debug HuggingFace Auth**
   - Verify Modal secret contains valid `HF_TOKEN`
   - Test with: `huggingface-cli whoami` in Modal container

2. **Add Server Logging**
   - Capture full vLLM startup logs to Modal output
   - Parse error messages for specific failure reasons

3. **Test with Smaller Models First**
   - Use gemma-2b or opt-125m to validate pipeline
   - Then scale up to larger models

### Architecture Improvements

1. **Pre-download Models to Modal Volume**
   - Create separate Modal function to download models
   - Mount volume with cached models to benchmark functions

2. **Add Retry Logic**
   - Retry server startup with exponential backoff
   - Handle transient failures gracefully

3. **Better Error Reporting**
   - Capture and return vLLM server stderr
   - Parse CUDA/torch error messages

---

## Files to Preserve

| File | Purpose |
|------|---------|
| `src/eval/modal_benchmark.py` | Modal benchmark functions |
| `src/eval/native_benchmark_runner.py` | Native runner with Modal integration |
| `omniperf_results_modal/` | Benchmark results |
| `docs/benchmark_analysis_full_run.md` | Previous single-GPU analysis |
| `docs/modal_benchmark_analysis.md` | This document |

---

## Next Steps

1. Investigate Modal dashboard logs for detailed error messages
2. Fix HuggingFace token passing to Modal containers
3. Test with smaller non-gated models
4. Consider pre-downloading model weights to Modal volume
5. Add verbose logging to capture vLLM startup errors
