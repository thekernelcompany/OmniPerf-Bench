# SGLang 3-Way Benchmark Session Documentation

**Date:** January 25, 2026
**GPU:** NVIDIA H100 PCIe (SM90, 80GB)
**Branch:** `feature/sglang-modal-benchmarks`

---

## Table of Contents

1. [Objective](#objective)
2. [Target Commits](#target-commits)
3. [Agent Patch Mapping](#agent-patch-mapping)
4. [Benchmark Approaches Tried](#benchmark-approaches-tried)
5. [Final Results](#final-results)
6. [Key Learnings](#key-learnings)
7. [File Structure](#file-structure)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## Objective

Run SGLang 3-way benchmarks comparing:
- **Baseline**: Original code at base commit (before PR)
- **Human**: Code after human's PR is merged
- **Agent Variants**: Code with AI agent patches applied to base commit
  - `claude_code`: Claude Code agent
  - `codex`: OpenAI Codex/GPT-5 agent
  - `trae_gpt5`: Trae with GPT-5
  - `trae_sonnet45`: Trae with Claude Sonnet 4.5

---

## Target Commits

17 performance-related SGLang commits were selected for benchmarking:

| Commit | PR | Subject |
|--------|-----|---------|
| `187b85b7` | #7393 | [PD] Optimize custom mem pool usage |
| `6b231325` | #6649 | [PD Perf] replace Queue to FastQueue |
| `4418f599` | #5624 | Fix FA3 DeepSeek prefill regression |
| `6cb00c63` | #6761 | [PD] Optimize time out logic |
| `148254d4` | #2705 | Improve moe reduce sum kernel |
| `2bd18e2d` | #2901 | Memory pool: Minor optimize |
| `2a413829` | - | Add triton version config key |
| `2a754e57` | #579 | 2x perf for large prefill (June 2024) |
| `5e023301` | #5662 | [perf] dsv3 bmm fallback to bf16 |
| `880221bd` | #7968 | Revert transfer batch |
| `b1e5a33a` | #6960 | Eliminate stream sync for LoRA |
| `c087ddd6` | #6627 | Refine pre_reorder_triton_kernel |
| `da47621c` | #7058 | Minor speedup topk postprocessing |
| `dd1012fc` | #6764 | [PD] Fix potential perf spike |
| `ddcf9fe3` | #3731 | Optimize triton attention mask |
| `df7f61ee` | #6812 | Speed up rebalancing |
| `e3ec6bf4` | #6814 | Minor speed up block_quant_dequant |

---

## Agent Patch Mapping

### Directory Structure

Agent patches are stored in two main directories:

```
perf-agents-bench/state/runs/
├── sglang/                    # Main SGLang patches
│   ├── claude_code/
│   ├── codex/
│   └── trae/
│       └── gpt-5/
└── sglan/                     # Trae Sonnet 4.5 SGLang patches (note: "sglan" not "sglang")
    └── trae/
        └── us-anthropic-claude-sonnet-4-5-20250929-v1-0/
```

### Agent-Specific Paths and Naming Conventions

#### 1. Claude Code (17/17 coverage)

**Path:** `perf-agents-bench/state/runs/sglang/claude_code/default/2025-12-23_06-28-44/`

**Naming Convention:** `sglang_XXX_<commit_hash>/model_patch.diff`

**Example:**
```
sglang_059_c087ddd6/model_patch.diff
sglang_027_6b231325/model_patch.diff
```

#### 2. Codex (17/17 coverage)

**Path:** `perf-agents-bench/state/runs/sglang/codex/gpt-5/389be848/`

**Naming Convention:** `sglang_core-XXXX/model_patch.diff`

**Commit Lookup:** Requires reading `prompt.json` in each folder to find the commit:
```json
{
  "commits": {
    "human": "c087ddd6865a52634326a05af66429cb5531cd16"
  }
}
```

**Example:**
```
sglang_core-0059/
├── model_patch.diff
└── prompt.json  # Contains commit hash
```

#### 3. Trae GPT-5 (17/17 coverage)

**Path:** `perf-agents-bench/state/runs/sglang/trae/gpt-5/`

**Subdirectories (searched in order):**
1. `2025-11-16_09-27-51/` - 74 commits with non-empty patches (primary)
2. `2025-11-14_21-05-32/` - 47 commits (many empty patches)

**Naming Convention:** `sglang_XXX_<commit_hash>/model_patch.diff`

**Important:** The first directory (2025-11-14) has many empty (0-byte) patches. The second directory (2025-11-16) was a re-run with actual patches. Always check file size > 0.

**Example:**
```
2025-11-16_09-27-51/sglang_059_c087ddd6/model_patch.diff  # 4455 bytes (use this)
2025-11-14_21-05-32/sglang_059_c087ddd6/model_patch.diff  # 0 bytes (skip)
```

#### 4. Trae Sonnet 4.5 (17/17 coverage)

**Primary Path:** `perf-agents-bench/state/runs/sglang/trae/claude-sonnet-45/2025-11-28_15-26-15/`

**Naming Convention:** `sglang_XXX_<commit_hash>/model_patch.diff`

**Example:**
```
sglang_059_c087ddd6/model_patch.diff
sglang_027_6b231325/model_patch.diff
```

> **Important:** This directory has ALL 17 target commits with non-empty patches!

**Backup Path (8 commits):** `perf-agents-bench/state/runs/sglan/trae/us-anthropic-claude-sonnet-4-5-20250929-v1-0/`

> **Note:** The backup directory uses `sglan` (typo) and different naming: `sglang_sonnet45_rerun_<commit>/`

### Patch Coverage Summary

| Agent | Coverage | Notes |
|-------|----------|-------|
| claude_code | 17/17 (100%) | All target commits have patches |
| codex | 17/17 (100%) | All target commits have patches |
| trae_gpt5 | 17/17 (100%) | Must search both subdirectories |
| trae_sonnet45 | 17/17 (100%) | Primary dir has all commits |

---

## Benchmark Approaches Tried

### 1. Docker-based Approach (Partially Successful)

Used pre-built Docker images from the dataset.

**Issues:**
- Images built for different GPU architectures (A100, not H100)
- FlashInfer compatibility issues with SM90
- Limited to specific commits with available images

### 2. Overlay Approach (Low Success Rate)

Overlay agent patches on existing SGLang installation.

**Issues:**
- Dependency conflicts between commits
- Patches generated against different base versions
- ~4% success rate

### 3. Isolated Venv Approach (Best Results)

Create fresh virtual environment per commit with appropriate dependencies.

**Implementation:** `run_isolated_benchmarks.py`

**Key Features:**
- Per-commit venv isolation at `/tmp/sglang-venvs/<commit>/`
- Era-based PyTorch version selection (June 2024 vs Late 2024)
- Automatic sgl_kernel installation
- Fallback model selection (DeepSeek-V3 -> Llama-3.1-8B)

**Success Rate:** ~27% (limited by SGLang server startup issues)

---

## Final Results

### Best Performing Commits

#### c087ddd6 (6/6 variants successful)

| Variant | Throughput (tokens/s) |
|---------|----------------------|
| baseline | 2675.65 |
| human | 2233.72 |
| claude_code | 2745.32 |
| codex | 2266.00 |
| trae_gpt5 | 2743.56 |
| trae_sonnet45 | 2684.83 |

**Winner:** claude_code (+2.6% over baseline)

#### 6b231325 (5/6 variants successful)

| Variant | Throughput (tokens/s) |
|---------|----------------------|
| baseline | 1251.61 |
| human | 1275.92 |
| claude_code | 1284.62 |
| codex | 1268.89 |
| trae_gpt5 | 1268.21 |
| trae_sonnet45 | N/A (no patch) |

**Winner:** claude_code (+2.6% over baseline)

### Overall Statistics

- **Total Benchmarks Attempted:** 102 (17 commits × 6 variants)
- **Successful:** 27 (26.5%)
- **Failed (Server Startup):** ~50%
- **Failed (Patch Application):** ~15%
- **N/A (No Patch):** ~8%

---

## Key Learnings

### 1. Agent Patch Directory Discovery

- **Always check multiple subdirectories** - Patches may be split across multiple run directories
- **Verify file size > 0** - Empty patch files exist from failed runs
- **Different naming conventions per agent** - No universal pattern
- **Typos exist** - `sglan` vs `sglang` directory names

### 2. Patch Application

- **Checkout BASE commit, not HUMAN commit** - Agent patches are generated as `base -> agent`, not `human -> agent`
- **Patches may not apply cleanly** - Context changes between commits
- **Use `git apply --check` first** - Validate before applying

### 3. SGLang Server Issues

- **sgl_kernel compatibility** - Different versions for different GPU architectures
- **FlashInfer issues** - Version constraints with `[srt]` extra
- **Memory management** - Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **Timeout handling** - Serving benchmarks can take >10 minutes

### 4. Dependency Management

- **Era-based PyTorch versions:**
  - June 2024 commits: `torch==2.3.0+cu121`, `triton==2.3.0`
  - Late 2024 commits: `torch==2.5.1+cu124`, `triton==3.1.0+`

- **Skip problematic dependencies:**
  - `xgrammar` - Build issues
  - `flashinfer` via `[srt]` - Version conflicts

### 5. Benchmark Types

- **bench_one_batch** - Faster, more reliable, measures raw inference
- **bench_serving** - Slower, requires server startup, measures end-to-end

---

## File Structure

```
omniperf_results_3way_sglang/
├── run_isolated_benchmarks.py     # Main benchmark runner
├── isolated_benchmark_results/    # Per-commit JSON results
│   ├── c087ddd6_isolated.json
│   ├── 6b231325_isolated.json
│   └── ...
├── docker_benchmark_results/      # Docker-based results
├── overlay_benchmark_results/     # Overlay approach results
├── local_benchmark_results/       # Local installation results
├── BENCHMARK_PLAN.md              # Implementation plan
├── H100_COMPATIBILITY_REPORT.md   # GPU compatibility notes
└── SESSION_DOCUMENTATION.md       # This file
```

### Result JSON Format

```json
{
  "commit": "c087ddd6",
  "full_commit": "c087ddd6865a52634326a05af66429cb5531cd16",
  "base_commit": "f4a8987f6904e4909adb473c52b443a62ba5a4b5",
  "pr_number": "6627",
  "subject": "Refine pre_reorder_triton_kernel slightly...",
  "timestamp": "2026-01-25T14:00:38.647851",
  "variants": {
    "baseline": {
      "status": "success",
      "metrics": {
        "total_throughput": 2675.65,
        "decode_throughput": 10.35,
        "prefill_throughput": 17030.87
      }
    },
    "human": { ... },
    "claude_code": { ... },
    "codex": { ... },
    "trae_gpt5": { ... },
    "trae_sonnet45": { ... }
  }
}
```

---

## Troubleshooting Guide

### "Failed to start SGLang server"

**Causes:**
1. sgl_kernel incompatibility with GPU
2. Missing dependencies
3. CUDA out of memory

**Solutions:**
```bash
# Check GPU compatibility
python -c "import torch; print(torch.cuda.get_device_capability())"

# Reinstall sgl_kernel
pip install --upgrade sgl_kernel

# Clear GPU memory
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill -9
```

### "Patch failed: already exists in working directory"

**Cause:** Checking out wrong commit (human instead of base)

**Solution:** For agent variants, checkout the BASE commit:
```python
if variant in ["claude_code", "codex", "trae_gpt5", "trae_sonnet45"]:
    target_commit = commit_info["base_commit"]  # Not human_commit!
```

### "trae_gpt5 patch not found" (but exists)

**Cause:** Only searching one subdirectory

**Solution:** Search both directories in order:
```python
TRAE_GPT5_SUBDIRS = [
    "2025-11-16_09-27-51",  # Non-empty patches
    "2025-11-14_21-05-32",  # Original (many empty)
]
```

### CUDA Out of Memory

**Solution:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Benchmark Timeout

**Cause:** Serving benchmarks are slow

**Solution:** Increase timeout or use `bench_one_batch` instead:
```python
timeout = 900  # 15 minutes for serving benchmarks
```

---

## Scripts Reference

### Run Single Commit
```bash
python run_isolated_benchmarks.py --commit c087ddd6 --variants baseline,human,claude_code
```

### Run All Commits
```bash
python run_isolated_benchmarks.py --all --variants baseline,human,claude_code,codex,trae_gpt5,trae_sonnet45
```

### Dry Run
```bash
python run_isolated_benchmarks.py --commit c087ddd6 --dry-run
```

### List Target Commits
```bash
python run_isolated_benchmarks.py --list
```

---

## Commits in This Session

1. `ac37ce92` - feat(sglang): Add multi-agent support and dependency fixes
2. `f09b439d` - feat(sglang): Add isolated venv benchmark runner and H100 results

---

*Generated by Claude Code on January 25, 2026*
