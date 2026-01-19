---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- benchmark
- performance
- vllm
- llm-inference
- optimization
- claude-code
- codex
- trae
- multi-agent
pretty_name: Multi-Agent vLLM Performance Benchmarks
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: commit_hash
    dtype: string
  - name: commit_short
    dtype: string
  - name: commit_subject
    dtype: string
  - name: repo
    dtype: string
  - name: perf_command
    dtype: string
  - name: files_changed
    list: string
  - name: pr_url
    dtype: string
  - name: models
    list: string
  - name: parent_commit
    dtype: string
  - name: gpu_config
    dtype: string
  - name: benchmark_mode
    dtype: string
  - name: agent_name
    dtype: string
  - name: agent_model
    dtype: string
  - name: benchmark_date
    dtype: string
  - name: model
    dtype: string
  - name: has_agent_patch
    dtype: bool
  - name: patch_path
    dtype: string
  - name: baseline_ttft_mean
    dtype: float64
  - name: baseline_ttft_median
    dtype: float64
  - name: baseline_ttft_p99
    dtype: float64
  - name: baseline_tpot_mean
    dtype: float64
  - name: baseline_tpot_median
    dtype: float64
  - name: baseline_tpot_p99
    dtype: float64
  - name: baseline_itl_mean
    dtype: float64
  - name: baseline_itl_median
    dtype: float64
  - name: baseline_itl_p99
    dtype: float64
  - name: baseline_latency_avg
    dtype: float64
  - name: baseline_throughput
    dtype: float64
  - name: human_ttft_mean
    dtype: float64
  - name: human_ttft_median
    dtype: float64
  - name: human_ttft_p99
    dtype: float64
  - name: human_tpot_mean
    dtype: float64
  - name: human_tpot_median
    dtype: float64
  - name: human_tpot_p99
    dtype: float64
  - name: human_itl_mean
    dtype: float64
  - name: human_itl_median
    dtype: float64
  - name: human_itl_p99
    dtype: float64
  - name: human_latency_avg
    dtype: float64
  - name: human_throughput
    dtype: float64
  - name: agent_ttft_mean
    dtype: float64
  - name: agent_ttft_median
    dtype: float64
  - name: agent_ttft_p99
    dtype: float64
  - name: agent_tpot_mean
    dtype: float64
  - name: agent_tpot_median
    dtype: float64
  - name: agent_tpot_p99
    dtype: float64
  - name: agent_itl_mean
    dtype: float64
  - name: agent_itl_median
    dtype: float64
  - name: agent_itl_p99
    dtype: float64
  - name: agent_latency_avg
    dtype: float64
  - name: agent_throughput
    dtype: float64
  - name: human_improvement_ttft_mean
    dtype: float64
  - name: human_improvement_tpot_mean
    dtype: float64
  - name: human_improvement_itl_mean
    dtype: float64
  - name: agent_improvement_ttft_mean
    dtype: float64
  - name: agent_improvement_tpot_mean
    dtype: float64
  - name: agent_improvement_itl_mean
    dtype: float64
  - name: agent_vs_human_ttft_mean
    dtype: float64
  - name: agent_vs_human_tpot_mean
    dtype: float64
  - name: agent_vs_human_itl_mean
    dtype: float64
  - name: human_improvement_ttft_median
    dtype: float64
  - name: human_improvement_ttft_p99
    dtype: float64
  - name: agent_improvement_ttft_median
    dtype: float64
  - name: agent_improvement_ttft_p99
    dtype: float64
  - name: agent_vs_human_ttft_median
    dtype: float64
  - name: agent_vs_human_ttft_p99
    dtype: float64
  - name: human_improvement_latency_avg
    dtype: float64
  - name: human_improvement_throughput
    dtype: float64
  - name: agent_improvement_latency_avg
    dtype: float64
  - name: agent_improvement_throughput
    dtype: float64
  - name: agent_vs_human_latency_avg
    dtype: float64
  - name: agent_vs_human_throughput
    dtype: float64
  - name: baseline_raw
    dtype: string
  - name: human_raw
    dtype: string
  - name: agent_raw
    dtype: string
  - name: test_script
    dtype: string
  - name: data_source
    dtype: string
  splits:
  - name: train
    num_bytes: 2197826
    num_examples: 361
  download_size: 500946
  dataset_size: 2197826
---

# Multi-Agent vLLM Performance Benchmarks

This dataset contains performance benchmark results from multiple AI coding agents on vLLM performance optimization tasks.

## Overview

| Metric | Value |
|--------|-------|
| **Total Results** | 361 |
| **Unique Commits** | 96 |
| **Agents** | 4 |
| **Schema Version** | v5 (76 columns) |
| **GPU** | H100 |
| **3-Way Complete** | 98 rows |

## Agents Included

| Agent | Model | Results | Description |
|-------|-------|---------|-------------|
| `claude_code` | `claude-sonnet-4-20250514` | 96 | Anthropic Claude Code |
| `codex` | `gpt-5` | 96 | OpenAI Codex |
| `trae` | `sonnet-4.5` | 96 | TRAE with Claude Sonnet 4.5 |
| `trae` | `gpt-5` | 73 | TRAE with GPT-5 |

## Agent Comparison Summary (as of 2026-01-19)

Based on analysis of 43 valid commits (excluding 9 with wrong perf command and 1 with documented data issue):

| Status | Claude Code | Codex | TRAE (Sonnet) | TRAE (GPT) |
|--------|-------------|-------|---------------|------------|
| Valid for comparison | 28 (65.1%) | 16 (37.2%) | 16 (37.2%) | 12 (27.9%) |
| Patch failure | 15 (34.9%) | 26 (60.5%) | 26 (60.5%) | 31 (72.1%) |
| Missing human_ttft | 0 | 1 | 1 | 0 |

**Key findings:**
- Claude Code has the highest patch success rate (65.1%)
- Codex and TRAE (Sonnet) have identical success patterns (16 VALID each on the same commits)
- TRAE (GPT) has the highest patch failure rate (72.1%)
- 12 commits fail for all agents (universal failures)

## Benchmark Modes

The `benchmark_mode` field determines which metrics are populated and how to interpret results.

### Serving Mode (`benchmark_mode = "serving"`)

Per-token latency metrics (lower is better):

| Metric | Description |
|--------|-------------|
| `ttft_mean/median/p99` | Time to First Token (ms) |
| `tpot_mean/median/p99` | Time per Output Token (ms) |
| `itl_mean/median/p99` | Inter-Token Latency (ms) |
| `throughput` | Output tokens per second (higher is better) |

### Standalone Mode (`benchmark_mode = "standalone"`)

Two sub-types exist based on the benchmark script:

**Latency benchmarks** (`benchmark_latency.py`):
| Metric | Description | Better |
|--------|-------------|--------|
| `latency_avg` | Average request latency (ms) | Lower |
| `throughput` | Tokens per second (may be null for latency-only benchmarks) | Higher |

**Throughput benchmarks** (`benchmark_throughput.py`, `benchmark_throughput_cache.py`):
| Metric | Description | Better |
|--------|-------------|--------|
| `throughput` | Tokens per second | Higher |

**Note:** Commit `3476ed08` is the only latency-only benchmark - it uses `benchmark_latency.py` but only produces `latency_avg`, not `throughput`. This is expected behavior.

### Prefix Caching Mode (`benchmark_mode = "prefix_caching"`)

| Metric | Description | Better |
|--------|-------------|--------|
| `throughput` | Output tokens per second | Higher |

Uses `benchmark_prefix_caching.py` which tests performance with `--enable-prefix-caching` flag.

## Metrics Coverage

### Baseline & Human Metrics (as of 2026-01-19)

| Metric Type | Mean | Median | P99 |
|-------------|------|--------|-----|
| Baseline TTFT | 123 | 123 | 123 |
| Baseline TPOT | 123 | 123 | 123 |
| Baseline ITL | 123 | 123 | 123 |
| Human TTFT | 123 | 123 | 123 |
| Human TPOT | 123 | 123 | 123 |
| Human ITL | 123 | 123 | 123 |

| Additional Metrics | Rows |
|--------------------|------|
| `baseline_throughput` | 117 |
| `baseline_latency_avg` | 56 |
| `human_throughput` | 180 |
| `human_latency_avg` | 48 |

### Agent Metrics

| Metric | Rows |
|--------|------|
| `agent_ttft_mean` | 83 |
| `agent_tpot_mean` | 83 |
| `agent_itl_mean` | 83 |
| `agent_throughput` | 119 |
| `agent_latency_avg` | 28 |

### 3-Way Completeness (Baseline + Human + Agent)

| Mode | Complete Rows |
|------|---------------|
| Serving | 72 |
| Standalone | 26 |
| **Total** | **98** |

## Known Data Gaps

### Commits with Incomplete Benchmark Data

Two commits have incomplete baseline/human data due to infrastructure issues:

| Commit | Issue | Baseline | Human | Details |
|--------|-------|----------|-------|---------|
| `6ce01f30` | Corrupted Docker image | Failed | Failed | Python environment corrupted in baseline image (`baseline-6a11fdfbb8d6`). Multiple `ModuleNotFoundError` for transformers, pyairports. Unfixable without rebuilding image. |
| `e7523c2e` | Hardware memory limit | Failed | Success | Model `google/gemma-3-12b-it` requires 131K context but baseline vLLM can only allocate 50K tokens in KV cache. Human's optimization enables larger context. |

**Note on `e7523c2e`:** The baseline failure is intentionally preserved. Reducing `--max-model-len` for baseline would create an unfair comparison (giving baseline artificial help the human didn't need). The human's optimization may specifically address memory efficiency.

### Commits with Partial Data

| Commit | Baseline | Human | Notes |
|--------|----------|-------|-------|
| `83450458` | Success | Failed | Human benchmark failed (no latency metrics in output) |
| `19d98e0c` | Success | Missing `human_ttft` | Prevents comparison for Codex/TRAE-Sonnet |

### Universal Failures (All Agents)

12 commits fail for all agents:
`7c01f706`, `ad8d696a`, `d7740ea4`, `660470e5`, `fc7b8d1e`, `ce6bf3a2`, `ccf02fcb`, `35fad35a`, `3a243095`, `6dd94dbe`, `e3580537`, `9ed82e70`

### Mode Classification Issues

| Commit | Issue |
|--------|-------|
| `ce6bf3a2`, `ccf02fcb` | `benchmark_mode = None` - mode not properly classified |

### Unbenchmarkable Commits (vLLM Version Incompatibility)

**7 commits cannot be benchmarked** due to Docker images containing old vLLM versions that don't support the configured model (`meta-llama/Llama-3.1-8B-Instruct`):

| Commit | PR | vLLM Version | Error |
|--------|-----|--------------|-------|
| `3a243095` | #3623 (Mar 2024) | ~0.3.x | `Unknown RoPE scaling type llama3` |
| `7c01f706` | #5974 (Jun 2024) | ~0.4.x | `Unknown RoPE scaling type llama3` |
| `80aa7e91` | #4971 (May 2024) | ~0.4.x | `Unknown RoPE scaling type llama3` |
| `8bc68e19` | #4208 (Apr 2024) | 0.4.2 | `Unknown RoPE scaling type llama3` |
| `9ed82e70` | #6520 (Jul 2024) | 0.5.2 | `Unknown RoPE scaling type llama3` |
| `ad8d696a` | #4270 (Apr 2024) | 0.4.1 | `Unknown RoPE scaling type llama3` |
| `cf2f084d` | #3279 (Mar 2024) | 0.3.3 | `Unknown RoPE scaling type llama3` |

**Root cause:** Llama 3.1 was released July 23, 2024 with a new "llama3" RoPE scaling type. These PRs predate that release. The Docker images were built with vLLM versions that don't support this RoPE type. The original PR authors likely used older models (Llama-2, Mistral-7B, etc.) but the specific models were not documented in the PRs.

**Resolution options:**
1. Rebuild Docker images with newer vLLM (changes what's being benchmarked)
2. Use compatible older models (e.g., `meta-llama/Llama-2-7b-hf`)
3. Mark as unbenchmarkable (current status)

### Special Cases: Non-Standard Benchmarks

Two commits have `benchmark_mode = None` because they don't fit standard benchmark patterns:

| Commit | PR | Issue | Human's Actual Benchmark |
|--------|-----|-------|-------------------------|
| `ccf02fcb` | #14848 (Mar 2025) | Mamba2-specific revert | Used `lm_eval` with `ibm-ai-platform/Bamba-9B` (accuracy test, not throughput/latency) |
| `ce6bf3a2` | #7898 (Aug 2024) | TPU-specific optimization | Used `benchmark_throughput.py` on **TPU** with `google/gemma-2b`. Cannot be benchmarked on GPU. |

These commits are included in the dataset but lack meaningful benchmark comparisons because:
- `ccf02fcb`: The human tested model accuracy, not inference performance
- `ce6bf3a2`: The optimization targets TPU-specific overhead (Dynamo guard evaluation), which doesn't apply to GPU

## Benchmark Design

Each row represents a benchmark run for a specific vLLM commit, comparing three versions:

| Version | Description |
|---------|-------------|
| **Baseline** | Performance before the optimization commit (parent commit) |
| **Human** | Performance with the human-authored optimization (ground truth) |
| **Agent** | Performance with the AI agent's optimization attempt |

## Filtering by Agent

```python
from datasets import load_dataset

ds = load_dataset("Inferencebench/claude-code-vllm-benchmarks", split="train")

# Filter by agent
claude_code = ds.filter(lambda x: x["agent_name"] == "claude_code")
codex = ds.filter(lambda x: x["agent_name"] == "codex")
trae_gpt5 = ds.filter(lambda x: x["agent_name"] == "trae" and x["agent_model"] == "gpt-5")
trae_sonnet = ds.filter(lambda x: x["agent_name"] == "trae" and x["agent_model"] == "sonnet-4.5")

print(f"Claude Code: {len(claude_code)} results")
print(f"Codex: {len(codex)} results")
print(f"TRAE GPT-5: {len(trae_gpt5)} results")
print(f"TRAE Sonnet: {len(trae_sonnet)} results")
```

## Key Columns

### Identity & Metadata
- `commit_hash`, `commit_short`, `commit_subject` - Git commit info
- `repo` - Repository name (e.g., "vllm-project/vllm")
- `agent_name` - Agent identifier (claude_code, codex, trae)
- `agent_model` - Model used by agent (claude-sonnet-4-20250514, gpt-5, sonnet-4.5)
- `benchmark_mode` - "serving", "standalone", or "prefix_caching"
- `data_source` - Origin of data (modal, docker, merged, etc.)

### Raw Metrics (per version)
- `baseline_*` - Baseline (pre-optimization) metrics
- `human_*` - Human patch metrics
- `agent_*` - Agent patch metrics

### Improvement Metrics
- `human_improvement_*` - % improvement of human patch over baseline
- `agent_improvement_*` - % improvement of agent patch over baseline
- `agent_vs_human_*` - How agent compares to human (positive = agent better)

**Note:** Improvement columns are `null` when baseline/human data is unavailable.

## Usage Examples

### Get 3-Way Complete Rows

```python
from datasets import load_dataset

ds = load_dataset("Inferencebench/claude-code-vllm-benchmarks", split="train")

# Serving mode with all 3 metrics
complete_serving = ds.filter(lambda x:
    x["baseline_tpot_mean"] is not None and
    x["human_tpot_mean"] is not None and
    x["agent_tpot_mean"] is not None
)
print(f"3-way complete (serving): {len(complete_serving)}")

# Standalone mode with all 3 metrics
complete_standalone = ds.filter(lambda x:
    x["baseline_latency_avg"] is not None and
    x["human_latency_avg"] is not None and
    x["agent_latency_avg"] is not None
)
print(f"3-way complete (standalone): {len(complete_standalone)}")
```

### Exclude Known Problem Commits

```python
# Commits with incomplete B/H data
problem_commits = ["6ce01f30", "e7523c2e", "83450458"]

clean_data = ds.filter(lambda x: x["commit_short"] not in problem_commits)
print(f"Clean rows: {len(clean_data)}")
```

### Compare Agents on Same Commits

```python
from collections import defaultdict

commits = defaultdict(list)
for row in ds:
    commits[row["commit_short"]].append(row)

multi_agent_commits = {k: v for k, v in commits.items() if len(v) > 1}
print(f"Commits with multiple agents: {len(multi_agent_commits)}")
```

## Schema (76 Columns)

| Category | Count | Examples |
|----------|-------|----------|
| Identity/Metadata | 18 | commit_hash, agent_name, agent_model, benchmark_mode |
| Baseline Metrics | 11 | baseline_ttft_mean/median/p99, baseline_throughput |
| Human Metrics | 11 | human_ttft_mean/median/p99, human_throughput |
| Agent Metrics | 11 | agent_ttft_mean/median/p99, agent_throughput |
| Improvement Metrics | 21 | agent_improvement_*, agent_vs_human_* |
| Raw Outputs | 4 | baseline_raw, human_raw, agent_raw, test_script |

## Changelog

- **2026-01-19**: Documented unbenchmarkable commits:
  - 7 commits with vLLM version incompatibility (RoPE scaling type llama3 not supported)
  - 2 commits with non-standard benchmarks (ccf02fcb: Mamba2/lm_eval, ce6bf3a2: TPU-specific)
  - Added human_ttft metrics for 7 serving commits (19d98e0c, 35fad35a, 660470e5, 6e36f4fa, 89a84b0b, e3580537, fc7b8d1e)
- **2026-01-19**: Fixed 8 Claude Code commits:
  - 7 serving commits corrected from throughput to ttft/tpot/itl metrics (99abb8b6, 22d33bac, 9badee53, e206b543, 89a84b0b, 19d98e0c, 6e36f4fa)
  - 1 prefix_caching commit fixed (2deb029d: benchmark_mode standalone→prefix_caching, throughput=5446.28 tok/s)
  - Added agent comparison summary and benchmark mode documentation
- **2026-01-19**: Added baseline median/p99 metrics (ttft, tpot, itl) for 123 rows. Documented known data gaps.
- **2026-01-18**: Added baseline/human metrics for 19 ONLY_AGENT commits (17 succeeded, 2 failed)
- **2026-01-15**: Schema v5 with 76 columns, multi-agent support

## Citation

```bibtex
@dataset{multi_agent_vllm_benchmarks,
  title={Multi-Agent vLLM Performance Benchmarks},
  author={Inferencebench},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/Inferencebench/claude-code-vllm-benchmarks}
}
```

## License

Apache 2.0
