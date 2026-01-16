# Fixable Commits Rerun Report

**Date:** 2026-01-16
**Dataset:** `Inferencebench/claude-code-vllm-benchmarks`
**Agent:** Claude Code (claude-sonnet-4-20250514)

## Overview

This report documents the rerun of 4 vLLM commits that had benchmark issues in the original run. The goal was to execute corrected benchmark commands and obtain proper baseline/human/agent comparisons.

## Commits Targeted for Rerun

| Task ID | Commit | Description | Original Issue | Fix Applied |
|---------|--------|-------------|----------------|-------------|
| vllm_core-0091 | fa63e710 | Reduce scheduling overhead after cuda sync | Effect size within noise | Add `--num-iters 100`, run 3 trials |
| vllm_core-0051 | 99abb8b6 | Optimize Rejection Sampler with Triton Kernels | Missing space in command | Fix: `[ngram]` spacing |
| vllm_core-0030 | 6ce01f30 | Optimize `get_seqs` | Only 100 prompts | Use 1000 prompts |
| vllm_core-0017 | 3476ed08 | Optimize block_manager_v2 vs v1 | Same command for baseline/test | Asymmetric commands |

## Results Summary

### Successfully Benchmarked (3 commits)

| Commit | Benchmark Type | Baseline | Human | Agent | Human Improvement | Agent Improvement |
|--------|---------------|----------|-------|-------|-------------------|-------------------|
| 3476ed08 | Latency (ms) | 169.20 | 175.44 | 184.16 | -3.69% | -8.85% |
| 99abb8b6 | Latency (ms) | 2177.47 | **N/A** | 2231.11 | N/A | -2.46% |
| 6ce01f30 | Throughput (req/s) | 9.18 | 9.21 | 9.20 | +0.33% | +0.22% |

### Not Run (1 commit)

| Commit | Reason |
|--------|--------|
| fa63e710 | Baseline Docker image not available (`shikhar481/vllm_fixed_human_images:baseline-2a0309a646b1` does not exist) |

## Detailed Results

### 3476ed08 - Optimize block_manager_v2 vs v1

**Benchmark Type:** Latency (standalone)
**Model:** facebook/opt-125m
**Asymmetric Commands:** Yes

- **Baseline command:** `python benchmarks/benchmark_latency.py --model facebook/opt-125m --input-len 1536 --output-len 50 --batch-size 8`
- **Test command:** `python benchmarks/benchmark_latency.py --model facebook/opt-125m --input-len 1536 --output-len 50 --batch-size 8 --use-v2-block-manager`

**Results:**
- Baseline: 169.20 ms
- Human (with v2 block manager): 175.44 ms (-3.69% slower)
- Agent (with v2 block manager): 184.16 ms (-8.85% slower)

**Analysis:** The v2 block manager shows slightly worse latency than v1 in this benchmark. Both human and agent patches show degradation, with the agent patch performing worse.

---

### 99abb8b6 - Optimize Rejection Sampler with Triton Kernels

**Benchmark Type:** Latency (standalone with speculative decoding)
**Model:** meta-llama/Llama-3.1-8B-Instruct
**Issue:** Human benchmark failed

**Command:**
```bash
python benchmarks/benchmark_latency.py --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-model [ngram] --ngram-prompt-lookup-min 5 --ngram-prompt-lookup-max 10 \
  --num-speculative-tokens 5 --input-len 550 --output-len 150
```

**Results:**
- Baseline: 2177.47 ms (throughput: 2639.30 tok/s)
- Human: **FAILED** (dependency conflict)
- Agent: 2231.11 ms (throughput: 2635.0 tok/s), -2.46% latency regression

**Human Benchmark Failure - Root Cause:**

The human Docker image (`ayushnangia16/nvidia-vllm-docker:99abb8b6...`) has a dependency conflict:

```
vllm 0.0.0+local requires transformers>=4.48.2, but you have transformers 4.44.2 which is incompatible.
...
ModuleNotFoundError: No module named 'transformers.models.mllama'
```

The vLLM in this image requires `transformers>=4.48.2` (which has `mllama` model support), but our compatibility fix downgrades to `transformers==4.44.2` to fix `LogitsWarper` imports. These requirements are mutually exclusive.

**Recommended Fix:** See [Docker Image Fix](#docker-image-fix-for-99abb8b6) section below.

---

### 6ce01f30 - Optimize get_seqs

**Benchmark Type:** Throughput
**Model:** meta-llama/Meta-Llama-3-8B-Instruct

**Command:**
```bash
python3 benchmarks/benchmark_throughput.py --backend vllm \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --input-len 1024 --output-len 256 --num-prompts 1000
```

**Results:**
- Baseline: 9.18 req/s (11745.11 tok/s)
- Human: 9.21 req/s (+0.33%)
- Agent: 9.20 req/s (+0.22%)

**Analysis:** Both human and agent show marginal improvements (~0.2-0.3%). The optimization has minimal measurable impact at this scale.

---

### fa63e710 - Reduce scheduling overhead after cuda sync (NOT RUN)

**Reason:** Baseline Docker image `shikhar481/vllm_fixed_human_images:baseline-2a0309a646b1` does not exist.

**Planned Configuration:**
- Model: meta-llama/Meta-Llama-3-8B
- Command: `python3 benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 32 --input-len 1000 --output-len 128 --num-iters 100`
- Trials: 3 (for statistical significance)

**To Run:** Build or locate the baseline image for parent commit `2a0309a646b1`.

---

## Docker Image Fix for 99abb8b6

### Problem

The human image has a newer vLLM that:
1. Requires `transformers>=4.48.2` (has `mllama` model support)
2. But our benchmark script blindly downgrades to `transformers==4.44.2` to fix `LogitsWarper` imports

### Solution Options

#### Option 1: Fix the Docker Image

Rebuild the Docker image to include benchmark scripts compatible with its vLLM version:

```dockerfile
# Ensure benchmark scripts are included and compatible
COPY benchmarks/ /workspace/benchmarks/
# Don't rely on external benchmark script cloning
```

The image should have its own `/workspace/benchmarks/` directory with scripts that work with `transformers>=4.48.2`.

#### Option 2: Fix the Benchmark Runner Script

Update `rerun_4_fixable_commits.py` to detect when vLLM requires newer transformers:

```python
def needs_transformers_downgrade(python_path: str) -> bool:
    """Check if vLLM needs transformers downgrade or has mllama dependency."""
    # If vLLM has mllama, it needs transformers>=4.48.2
    check_cmd = f"{python_path} -c \"import vllm.transformers_utils.configs.mllama\" 2>/dev/null"
    result = subprocess.run(check_cmd, shell=True)
    if result.returncode == 0:
        # Has mllama - DON'T downgrade transformers
        return False

    # Check if LogitsWarper import fails
    check_cmd = f"{python_path} -c \"from transformers.generation.logits_process import LogitsWarper\""
    result = subprocess.run(check_cmd, shell=True)
    return result.returncode != 0

# In apply_compatibility_fixes():
if needs_transformers_downgrade(python_path):
    # Apply transformers==4.44.2 fix
    ...
else:
    # Skip downgrade - use native benchmark scripts
    print("Skipping transformers downgrade (vLLM requires newer version)")
```

#### Option 3: Use Native Benchmark Scripts

For images with newer vLLM, use the benchmark scripts bundled with that vLLM installation instead of cloning old ones:

```python
# Check if /workspace/benchmarks exists in container
if os.path.exists("/workspace/benchmarks/benchmark_latency.py"):
    benchmarks_dir = "/workspace/benchmarks"
else:
    # Fall back to cloning compatible version
    clone_benchmark_scripts()
```

---

## Files Created

| File | Description |
|------|-------------|
| `scripts/reruns/rerun_4_fixable_commits.py` | Main benchmark runner script |
| `scripts/upload/upload_fixable_reruns.py` | HuggingFace upload script |
| `omniperf_results_3way_claude_code/reruns/4_fixable/3476ed08_result.json` | Results for block_manager_v2 |
| `omniperf_results_3way_claude_code/reruns/4_fixable/99abb8b6_result.json` | Results for Rejection Sampler |
| `omniperf_results_3way_claude_code/reruns/4_fixable/6ce01f30_result.json` | Results for get_seqs |
| `omniperf_results_3way_claude_code/reruns/4_fixable/hf_upload_ready.json` | Transformed data for HF upload |

---

## HuggingFace Upload

Results were uploaded to `Inferencebench/claude-code-vllm-benchmarks` using the 76-column schema (Schema v5).

**Upload Command:**
```bash
export HF_TOKEN=<write-token>
python scripts/upload/upload_fixable_reruns.py --update
```

**Records Updated:** 3 commits with new baseline metrics added to existing dataset (359 → 352 rows after deduplication and update).

---

## Next Steps

1. **Build missing baseline image** for fa63e710 (parent: 2a0309a646b1)
2. **Fix human image** for 99abb8b6 using one of the solutions above
3. **Re-run failed benchmarks** once images are fixed
4. **Consider adding statistical significance tests** for small improvements (<1%)
