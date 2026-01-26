# Critical Data Leakage Analysis in Benchmark Prompts

**Date:** 2026-01-26
**Severity:** CRITICAL
**Impact:** All benchmark results are compromised

## Executive Summary

A critical data leakage issue has been identified in the `perf-agents-bench` benchmark framework. The task prompts sent to AI agents contain information derived directly from the human solution (reference optimization), including:

1. **Exact files modified** by the human solution
2. **Exact line count changes** (insertions/deletions)
3. **Diff statistics** showing the scope of changes

This fundamentally compromises the benchmark's validity as it tells agents exactly where to look and how much to change, rather than requiring them to discover optimization opportunities independently.

## Affected Components

### Source of Leakage: `bench/prepare.py`

#### 1. Target Files Derived from Human Diff (Lines 149-159)

```python
# Determine target files: if none provided, derive from pre..human diff
provided_targets = task_cfg["optimization_contract"].get("target_files", [])
derived_targets = []
if not provided_targets:
    try:
        # Use the base repo (not the worktree) to diff pre..human
        # This captures the exact change surface of the human commit
        derived_targets = get_changed_files(rm.base_dir, pre, human)
    except Exception:
        derived_targets = []
target_files = provided_targets or derived_targets
```

**Issue:** If no target files are explicitly configured, they are extracted directly from `git diff pre human`, revealing exactly which files the human solution modified.

#### 2. Diff Statistics Included in Prompt (Lines 356-363)

```python
# Add files modified statistics (shows scope without revealing exact changes)
if diff_stat:
    task_lines.append("")
    task_lines.append("## Files Modified (statistics)")
    task_lines.append("The following files were changed in the reference optimization:")
    task_lines.append("```")
    task_lines += diff_stat.splitlines()
    task_lines.append("```")
```

**Issue:** The `git diff --stat` output is included verbatim, showing file names and line counts.

#### 3. Target Files Listed Multiple Times (Lines 322-327, 341-344)

```python
if target_files:
    task_lines.append("")
    task_lines.append("Target files to optimize:")
    for f in target_files[:3]:
        task_lines.append(f"- {f}")

# ... later ...

if prompt["target_files"]:
    task_lines.append("")
    task_lines.append("## Target Files (ONLY modify these)")
    task_lines += [f"- `{t}`" for t in prompt["target_files"]]
```

## Evidence of Leakage in Task Prompts

### Example 1: vLLM vllm_core-0000

```
Target files to optimize:
- vllm/reasoning/qwen3_reasoning_parser.py

## Target Files (ONLY modify these)
- `vllm/reasoning/qwen3_reasoning_parser.py`

## Files Modified (statistics)
The following files were changed in the reference optimization:
```
vllm/reasoning/qwen3_reasoning_parser.py | 53 ++++++++++++++++----------------
 1 file changed, 27 insertions(+), 26 deletions(-)
```
```

### Example 2: vLLM vllm_core-0001

```
## Target Files (ONLY modify these)
- `vllm/model_executor/layers/fused_moe/configs/E=8,N=14336,device_name=AMD_Instinct_MI300X.json`
- `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=AMD_Instinct_MI300X.json`
- `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=AMD_Instinct_MI300X.json`

## Files Modified (statistics)
The following files were changed in the reference optimization:
```
.../configs/E=8,N=14336,device_name=AMD_Instinct_MI300X.json          | 4 ++--
 .../fused_moe/configs/E=8,N=3584,device_name=AMD_Instinct_MI300X.json | 4 ++--
 .../fused_moe/configs/E=8,N=7168,device_name=AMD_Instinct_MI300X.json | 2 +-
 3 files changed, 5 insertions(+), 5 deletions(-)
```
```

### Example 3: SGLang sglang_000_021f76e4

```
Target files to optimize:
- python/sglang/srt/lora/lora_manager.py
- python/sglang/srt/lora/mem_pool.py

## Target Files (ONLY modify these)
- `python/sglang/srt/lora/lora_manager.py`
- `python/sglang/srt/lora/mem_pool.py`

## Files Modified (statistics)
The following files were changed in the reference optimization:
```
python/sglang/srt/lora/lora_manager.py | 113 +++++++++++++++++++++++----------
 python/sglang/srt/lora/mem_pool.py     |   9 ++-
 2 files changed, 83 insertions(+), 39 deletions(-)
```
```

### Example 4: SGLang sglang_001_09deb20d

```
## Target Files (ONLY modify these)
- `python/sglang/srt/layers/logits_processor.py`
- `python/sglang/srt/managers/router/model_rpc.py`

## Files Modified (statistics)
The following files were changed in the reference optimization:
```
python/sglang/srt/layers/logits_processor.py   | 4 +++-
 python/sglang/srt/managers/router/model_rpc.py | 2 +-
 2 files changed, 4 insertions(+), 2 deletions(-)
```
```

## Quantitative Impact Assessment

### Affected Runs (2026-01-26 Kimi K2-Thinking Benchmark)

| Benchmark | Task Files Checked | Files with Leakage | Leakage Rate |
|-----------|-------------------|-------------------|--------------|
| vLLM | 34 | 34 | **100%** |
| SGLang | 7 | 7 | **100%** |
| **Total** | **41** | **41** | **100%** |

### Information Leaked Per Task

| Information Type | Impact Level | Description |
|-----------------|--------------|-------------|
| Target file names | **CRITICAL** | Agent knows exactly which files to modify |
| Number of files changed | HIGH | Agent knows the scope of changes |
| Lines per file | HIGH | Agent knows approximate change size |
| Insertions vs deletions | MEDIUM | Hints at nature of changes (refactor vs new code) |
| Diff visualization (+/-) | MEDIUM | Visual representation of change balance |

## Impact on Benchmark Validity

### What the Benchmark SHOULD Measure

1. Agent's ability to **explore and understand** large codebases
2. Agent's ability to **identify performance bottlenecks** through profiling
3. Agent's ability to **discover** which files need optimization
4. Agent's ability to **design** appropriate optimizations
5. Agent's ability to **implement** correct and effective changes

### What the Benchmark ACTUALLY Measures (with leakage)

1. Agent's ability to make changes to **explicitly specified files**
2. Agent's ability to make changes of a **known size**
3. Agent's ability to follow explicit instructions

### Comparison

| Capability | Without Leakage | With Leakage |
|------------|-----------------|--------------|
| Codebase exploration | Required | Not required |
| Bottleneck identification | Required | Skipped |
| File discovery | Required | Given in prompt |
| Change scope estimation | Required | Given in prompt |
| Implementation | Required | Required |

**Conclusion:** The benchmark difficulty is significantly reduced, making results incomparable to benchmarks without leakage.

## Recommendations

### Immediate Actions

1. **Remove `diff_stat` from prompts** - Delete lines 356-363 in `prepare.py`
2. **Remove derived target files** - Do not use `get_changed_files(pre, human)` to derive targets
3. **Use task-specific configuration only** - Require explicit target files in task YAML configs or omit entirely

### Long-term Fixes

1. **Redesign task specification** - Define tasks by performance scenario, not by files to change
2. **Blind evaluation** - Agent should discover optimization opportunities independently
3. **Re-run affected benchmarks** - All existing results should be considered invalid

### Proposed Code Changes

```python
# REMOVE THIS SECTION (lines 149-159):
# Determine target files: if none provided, derive from pre..human diff
# provided_targets = task_cfg["optimization_contract"].get("target_files", [])
# derived_targets = []
# if not provided_targets:
#     try:
#         derived_targets = get_changed_files(rm.base_dir, pre, human)
#     except Exception:
#         derived_targets = []
# target_files = provided_targets or derived_targets

# REPLACE WITH:
target_files = task_cfg["optimization_contract"].get("target_files", [])
# If no target files specified, agent must discover them

# REMOVE THIS SECTION (lines 356-363):
# if diff_stat:
#     task_lines.append("")
#     task_lines.append("## Files Modified (statistics)")
#     task_lines.append("The following files were changed in the reference optimization:")
#     task_lines.append("```")
#     task_lines += diff_stat.splitlines()
#     task_lines.append("```")
```

## Conclusion

This data leakage represents a fundamental flaw in the benchmark design. All benchmark results obtained with the current `prepare.py` implementation should be considered **invalid** for measuring agent performance on discovering and implementing optimizations.

The benchmark effectively transforms from a **discovery and optimization** task into a **guided modification** task, which is significantly easier and does not reflect real-world scenarios where developers must identify optimization opportunities themselves.

---

**Analysis conducted by:** Claude Code
**Files analyzed:**
- `bench/prepare.py` (source of leakage)
- 41 task.txt files from 2026-01-26 runs
- Multiple trajectory.json files confirming leakage in prompts sent to models
