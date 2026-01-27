# MIMO SGLang Agent Benchmark Implementation

## Status: BLOCKED - Docker Image Dependency Issues

## Goal
Run performance benchmarks for MIMO (xiaomi-mimo-v2-flash) agent patches on SGLang commits.

---

## Critical Analysis

### MIMO SGLang Runs Summary
- **Location**: `perf-agents-bench/state/runs/sglan/trae/xiaomi-mimo-v2-flash/2026-01-27_10-49-46/`
- **Total tasks**: 15
- **Successful with patches**: 7
- **Failed (max_steps_exceeded)**: 8

### Docker Image Availability

| Commit | Human Image | Baseline Image | Benchmarkable |
|--------|-------------|----------------|---------------|
| c087ddd6 | shikhar481:c087ddd6 | f4a8987f (**NO**) | **NO** |
| 6b231325 | ayushnangia16:full | b1c8d4e9 (YES) | **YES** |
| dd1012fc | ayushnangia16:full | 44aab7f9 (YES) | **YES** |
| e3ec6bf4 | shikhar481 + ayushnangia16 | b04df75a (YES) | **YES** |
| da47621c | ayushnangia16:full | 22a6b9fc (YES) | **YES** |
| 2ed68d7a | **NO** | e984d507 (**NO**) | **NO** |
| 187b85b7 | ayushnangia16:full | ceba0ce4 (YES) | **YES** |

### Coverage Summary
| Status | Count | Commits |
|--------|-------|---------|
| **Benchmarkable** | 5 | 6b231325, dd1012fc, e3ec6bf4, da47621c, 187b85b7 |
| Missing baseline image | 1 | c087ddd6 |
| Missing both images | 1 | 2ed68d7a |

---

## Implementation Steps

### Step 1: Add MIMO Agent Config to SGLang Benchmark Runner

Modify `scripts/runners/local_docker_sglang_benchmark.py`:

```python
# Add to AGENT_RUNS_DIRS (~line 58):
AGENT_RUNS_DIRS = {
    "claude_code": Path("/root/OmniPerf-Bench/perf-agents-bench/state/runs/sglang/claude_code"),
    "codex": Path("/root/OmniPerf-Bench/perf-agents-bench/state/runs/sglang/codex"),
    "trae_gpt5": Path("/root/OmniPerf-Bench/perf-agents-bench/state/runs/sglang/trae/gpt-5"),
    "trae_sonnet45": Path("/root/OmniPerf-Bench/perf-agents-bench/state/runs/sglang/trae/claude-sonnet-45"),
    "mimo": Path("perf-agents-bench/state/runs/sglan/trae/xiaomi-mimo-v2-flash/2026-01-27_10-49-46"),  # ADD THIS
}
```

### Step 2: Fix Hardcoded Paths

Replace `/root/OmniPerf-Bench` with `REPO_ROOT`:
```python
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
```

### Step 3: Create Commit Mapping for MIMO

The MIMO commits need to be mapped to their benchmark commands. Use the existing `sglang_commit_mapping.json`.

### Step 4: Run Benchmarks

```bash
# Test single commit first
python scripts/runners/local_docker_sglang_benchmark.py \
    --commit 6b231325 \
    --agent-only \
    --agents mimo \
    --timeout 900

# Run all benchmarkable commits
python scripts/runners/local_docker_sglang_benchmark.py \
    --commits 6b231325,dd1012fc,e3ec6bf4,da47621c,187b85b7 \
    --agent-only \
    --agents mimo \
    --timeout 900
```

---

## Key Differences from vLLM Benchmarks

1. **No wheel fallback**: SGLang has ABI compatibility constraints - can't use Python overlay approach
2. **Different image repos**:
   - `shikhar481/sglang-images:{commit}` (short hash)
   - `ayushnangia16/nvidia-sglang-docker:{full_hash}` (40-char hash)
3. **Official base image**: `lmsysorg/sglang:v0.4.6.post5-cu124` for fallback

---

## Expected Outcomes

| Outcome | Count |
|---------|-------|
| Successful benchmarks | ~5 |
| Skipped (no images) | 2 |
| Potential patch bugs | Unknown (TBD) |

---

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/runners/local_docker_sglang_benchmark.py` | Add MIMO to AGENT_RUNS_DIRS, fix paths |

## Output Directory

Results will be saved to:
```
omniperf_results_3way_sglang/agent_benchmark_results/{commit}_agent_mimo_serving.json
```

---

## BLOCKING ISSUE: Docker Image Dependency Problems

### Issue Summary
All available SGLang Docker images have Python dependency incompatibilities that prevent the SGLang server from starting.

### Errors Encountered

#### 1. shikhar481/sglang-images:v04x-fixed-* (Triton Issue)
```
ImportError: cannot import name 'default_cache_dir' from 'triton.runtime.cache'
```

#### 2. ayushnangia16/nvidia-sglang-docker:* (Transformers Issue)
```
ModuleNotFoundError: No module named 'transformers.masking_utils'
```

This error occurs in `compressed_tensors/modeling/attention.py` which requires a newer version of transformers that has `masking_utils`.

### Root Cause
SGLang has strict ABI compatibility requirements with:
- `sgl_kernel` (custom CUDA kernels)
- `flashinfer` (attention backend)
- `triton` (kernel compilation)
- `transformers` (model loading)

The Docker images were built at different times with different dependency versions, causing incompatibilities.

### Resolution Options
1. **Rebuild Docker images** with correct dependency pinning
2. **Use official SGLang image** (`lmsysorg/sglang:v0.4.6.post5-cu124`) and clone/checkout specific commits
3. **Wait for fixed images** from the image maintainers

### What Was Implemented
1. ✅ Moved MIMO SGLang runs to `feature/sglang-modal-benchmarks` branch
2. ✅ Added MIMO agent config to `local_docker_sglang_benchmark.py`
3. ✅ Fixed hardcoded paths to use `REPO_ROOT`
4. ✅ Updated image selection to check ayushnangia16 for baselines
5. ❌ Benchmark execution blocked by dependency issues

### Files Modified
- `scripts/runners/local_docker_sglang_benchmark.py` - Added MIMO config, fixed paths, updated image selection
- `perf-agents-bench/state/runs/sglan/trae/xiaomi-mimo-v2-flash/` - Copied from vllm branch
