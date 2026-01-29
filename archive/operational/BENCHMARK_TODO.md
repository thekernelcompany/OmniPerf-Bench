# vLLM Benchmark To-Do List

**Generated**: 2026-01-19
**Based on**: Analysis of Claude Code, Codex, TRAE Sonnet-4.5, and TRAE GPT-5 agent results

---

## Summary

| Category | Count | Priority |
|----------|-------|----------|
| Infrastructure Issues (Disk Full) | 6 | HIGH |
| Missing Baseline Benchmarks | 9 | HIGH |
| Missing Human Benchmarks | 8 | HIGH |
| Agent Crashes (Investigation Needed) | 17 | MEDIUM |
| Wrong Benchmark Type | 1 | LOW |

---

## 1. Infrastructure Issues - Disk Full (HIGH PRIORITY)

These commits failed due to "no space left on device" during Docker image pull. Need to rerun after cleaning disk space.

| Commit | Model | Benchmark Mode | Error |
|--------|-------|----------------|-------|
| `e3580537` | neuralmagic/Meta-Llama-3-8B-Instruct-FP8 | serving | `docker: write /ephemeral/docker-data/tmp/GetImageBlob: no space left on device` |
| `3a243095` | meta-llama/Llama-3.1-8B-Instruct | serving | `docker: write /ephemeral/docker-data/tmp/GetImageBlob: no space left on device` |
| `7c01f706` | meta-llama/Llama-3.1-8B-Instruct | serving | Disk full during image pull |
| `3476ed08` | meta-llama/Llama-3.1-8B-Instruct | serving | Disk full during image pull |
| `fc7b8d1e` | meta-llama/Llama-3.1-8B-Instruct | serving | Disk full during image pull |

**Action**: Clear Modal ephemeral storage and rerun these benchmarks.

---

## 2. Missing Baseline Benchmarks (HIGH PRIORITY)

These commits have agent results but are missing baseline benchmark results.

| Commit | Model | Parent Commit | Notes |
|--------|-------|---------------|-------|
| `89a84b0b` | Qwen/Qwen1.5-0.5B | 084a01fd3544 | "[Core] Use array to speedup padding" |
| `6e36f4fa` | meta-llama/Llama-3.1-8B-Instruct | dd2a6a82e3f4 | "improve chunked prefill performance" |
| `19d98e0c` | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | 2b04c209ee98 | Missing baseline |
| `0ec82edd` | Qwen/Qwen3-30B-A3B | 005ae9be6c22 | Missing baseline |
| `2f192835` | meta-llama/Llama-3.1-8B-Instruct | 95baec828f3e | Missing baseline |
| `526de822` | Various | - | Missing baseline |
| `660470e5` | meta-llama/Llama-3.1-8B-Instruct | 8d59dbb00044 | Missing baseline |
| `aea94362` | meta-llama/Llama-3.2-1B-Instruct | 7206ce4ce112 | Missing baseline |
| `bfdb1ba5` | meta-llama/Llama-2-7b-chat-hf | cf2f084d56a1 | Missing baseline |

**Docker Image Source**: `shikhar481/vllm_fixed_human_images:baseline-{12char_parent_hash}`

---

## 3. Missing Human Benchmarks (HIGH PRIORITY)

These commits have agent results but are missing human (optimized commit) benchmark results.

| Commit | Full Commit Hash | Model | Notes |
|--------|------------------|-------|-------|
| `89a84b0b` | 89a84b0bb7b30706a02836234a94493ea8f780bf | Qwen/Qwen1.5-0.5B | "[Core] Use array to speedup padding" |
| `6e36f4fa` | 6e36f4fa6ce64619b9ea94c88a157f5783a63a65 | meta-llama/Llama-3.1-8B-Instruct | "improve chunked prefill performance" |
| `19d98e0c` | 19d98e0c (short) | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | Human image may not exist |
| `0ec82edd` | 0ec82edda59aaf5cf3b07aadf4ecce1aa1131add | Qwen/Qwen3-30B-A3B | Check if human image exists |
| `2f192835` | 2f1928354903ae0c6edfe76cc90081eb513ead2c | meta-llama/Llama-3.1-8B-Instruct | Check if human image exists |
| `660470e5` | 660470e5a36b8e52083615ad7c85e9b4fd4c72ce | meta-llama/Llama-3.1-8B-Instruct | Check if human image exists |
| `aea94362` | aea94362c9bdd08ed2b346701bdc09d278e85f66 | meta-llama/Llama-3.2-1B-Instruct | Check if human image exists |
| `bfdb1ba5` | bfdb1ba5c3fb14387c69acb1f5067102d8028e56 | meta-llama/Llama-2-7b-chat-hf | Check if human image exists |

**Docker Image Source**: `ayushnangia16/nvidia-vllm-docker:{40char_full_commit_hash}`

---

## 4. Agent Crashes (MEDIUM PRIORITY)

These commits had agent patches that crashed during benchmark execution. May need investigation into patch quality.

### Claude Code Agent - Crashes
| Commit | Model | Error |
|--------|-------|-------|
| (Generally successful - 95% patch generation rate) | | |

### Codex Agent - Crashes
| Commit | Model | Error |
|--------|-------|-------|
| `89a84b0b` | Qwen/Qwen1.5-0.5B | Server crashed after applying patch |
| `6e36f4fa` | meta-llama/Llama-3.1-8B-Instruct | Server crashed after applying patch |
| Multiple others | Various | Server crashed after applying patch |

### TRAE Sonnet-4.5 Agent - Crashes
| Commit | Model | Error |
|--------|-------|-------|
| `89a84b0b` | Qwen/Qwen1.5-0.5B | Server crashed after applying patch |
| `6e36f4fa` | meta-llama/Llama-3.1-8B-Instruct | Server crashed after applying patch |
| Multiple others | Various | Server crashed after applying patch |

### TRAE GPT-5 Agent - Crashes
| Commit | Model | Error |
|--------|-------|-------|
| Multiple commits | Various | Server crashed after applying patch |

**Note**: Crashes may indicate:
1. Agent patch introduced breaking changes
2. Incompatible dependencies
3. Environment issues

---

## 5. Wrong Benchmark Type (LOW PRIORITY)

These commits are configured with incorrect benchmark types, resulting in mismatched metrics.

| Commit | Current Benchmark | Expected | Issue |
|--------|-------------------|----------|-------|
| `2deb029d` | `prefix_caching` | `serving` (if TTFT needed) | Outputs throughput metrics (tok/s) instead of TTFT metrics |

**Details for `2deb029d`**:
- **Model**: RedHatAI/Meta-Llama-3-8B-Instruct-FP8
- **Perf Command**: `python3 benchmarks/benchmark_prefix_caching.py --model RedHatAI/Meta-Llama-3-8B-Instruct-FP8 --output-len 200 --enable-prefix-caching`
- **Current Metrics**: `input_throughput_tok_s: 17564.27`, `throughput_tok_s: 5446.28`
- **Missing Metrics**: TTFT (Time to First Token), TPOT, ITL

**Action**: Determine if this commit should use serving benchmark for TTFT metrics or if throughput metrics are appropriate for prefix caching optimization.

---

## 6. Specific Commit Details

### 89a84b0b - "[Core] Use array to speedup padding (#6779)"

- **Author**: Peng Guanwen
- **Files Changed**:
  - `vllm/model_executor/layers/sampler.py`
  - `vllm/model_executor/sampling_metadata.py`
  - `vllm/sequence.py`
- **Model**: Qwen/Qwen1.5-0.5B
- **Benchmark Mode**: serving

**Agent Results**:
| Agent | Status | TTFT Mean | Throughput |
|-------|--------|-----------|------------|
| Claude Code | ✅ Success | 356.05ms | 54.61 req/s |
| Codex | ❌ Crashed | - | - |
| TRAE Sonnet-4.5 | ❌ Crashed | - | - |
| TRAE GPT-5 | TBD | - | - |

**Missing**: Baseline benchmark, Human benchmark

---

### 6e36f4fa - "improve chunked prefill performance"

- **PR Reference**: Fixes #7592 (vllm 0.5.4 enable_chunked_prefill throughput regression)
- **Author**: wang.yuqi
- **Files Changed**:
  - `tests/basic_correctness/test_chunked_prefill.py`
  - `vllm/core/scheduler.py`
- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Benchmark Mode**: serving

**Agent Results**:
| Agent | Status | TTFT Mean | Throughput |
|-------|--------|-----------|------------|
| Claude Code | ✅ Success | 1011.46ms | 35.94 req/s |
| Codex | ❌ Crashed | - | - |
| TRAE Sonnet-4.5 | ❌ Crashed | - | - |
| TRAE GPT-5 | TBD | - | - |

**Missing**: Baseline benchmark, Human benchmark

---

## 7. Action Items Checklist

### Immediate (Infrastructure)
- [ ] Clear Modal ephemeral storage
- [ ] Rerun disk-full commits: `e3580537`, `3a243095`, `7c01f706`, `3476ed08`, `fc7b8d1e`

### High Priority (Missing Benchmarks)
- [ ] Run baseline benchmarks for 9 commits listed above
- [ ] Verify human Docker images exist for 8 commits
- [ ] Build missing human Docker images if needed
- [ ] Run human benchmarks for 8 commits listed above

### Medium Priority (Investigation)
- [ ] Investigate why Codex and TRAE patches crash for `89a84b0b` and `6e36f4fa`
- [ ] Compare patch quality between Claude Code (working) vs Codex/TRAE (crashing)
- [ ] Review 17 crashed agent runs for patterns

### Low Priority (Configuration)
- [ ] Decide on correct benchmark type for `2deb029d` (prefix_caching vs serving)
- [ ] Update benchmark_mode_mapping.json if needed

---

## 8. Docker Image References

### Baseline Images
```
shikhar481/vllm_fixed_human_images:baseline-{12char_parent_hash}
```

### Human (Optimized) Images
```
ayushnangia16/nvidia-vllm-docker:{40char_full_commit_hash}
```

### Agent Images
Built dynamically during benchmark execution with applied patches.

---

## 9. Notes on Dataset Propagation

**Current Issue**: Agent crash status is NOT propagated to HuggingFace export. The dataset shows "Baseline install failed" errors which are different from agent crashes.

**Proposed Schema Update**:
```json
{
  "agent_status": "success|crashed|disk_full|no_metrics",
  "agent_error": "Error description if any",
  "benchmark_type": "serving|prefix_caching|standalone"
}
```

This would allow proper filtering and analysis of benchmark results by status.
