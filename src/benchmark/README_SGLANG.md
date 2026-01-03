# SGLang Benchmarking System

A comprehensive guide to running performance benchmarks for SGLang optimizations on Modal cloud GPUs.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [System Architecture](#3-system-architecture)
4. [File Structure](#4-file-structure)
5. [Data Sources](#5-data-sources)
6. [Docker Images](#6-docker-images)
7. [Modal Volumes](#7-modal-volumes)
8. [The 3-Way Benchmark Flow](#8-the-3-way-benchmark-flow)
9. [Usage Examples](#9-usage-examples)
10. [Result Format](#10-result-format)
11. [Cost Estimate](#11-cost-estimate)
12. [Troubleshooting](#12-troubleshooting)
13. [Environment Variables](#13-environment-variables)
14. [Quick Reference](#14-quick-reference)

---

## 1. Overview

### What is this?

This system evaluates **AI agent performance** on real-world SGLang optimizations. It answers the question:

> "How well can an AI agent (like Claude Code) reproduce performance optimizations that human engineers wrote?"

### The 3-Way Comparison

For each commit, we run **three benchmarks**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   BASELINE          HUMAN               AGENT                       │
│   (Before PR)       (After PR)          (AI-generated)              │
│                                                                     │
│   Parent commit     Optimized commit    Parent commit +             │
│   Docker image      Docker image        Agent's patch               │
│                                                                     │
│   ───────────────────────────────────────────────────────────────   │
│                                                                     │
│   Performance       Performance         Performance                 │
│   before the        after the human     after the AI                │
│   optimization      applied their PR    applied its fix             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Metrics

We compare:
- **Request throughput** (req/s) - How many requests per second
- **Output throughput** (tokens/s) - How many tokens generated per second
- **TTFT** - Time to first token (latency)
- **TPOT** - Time per output token
- **ITL** - Inter-token latency

### Infrastructure

- **Cloud GPUs**: NVIDIA H100 on [Modal](https://modal.com)
- **Dataset**: 74 real SGLang PRs from HuggingFace
- **Docker Images**: Pre-built images for each commit
- **Execution**: Parallel 3-way benchmarks (3 GPUs simultaneously)

---

## 2. Quick Start

### Prerequisites

1. **Modal Account** - Sign up at [modal.com](https://modal.com)
2. **HuggingFace Token** - For downloading models
3. **Python 3.10+** with dependencies installed

### Installation

```bash
# Clone the repo (if not already done)
git clone https://github.com/your-org/OmniPerf-Bench.git
cd OmniPerf-Bench

# Install dependencies
uv venv && source .venv/bin/activate && uv sync

# Or with pip
pip install -r requirements.txt

# Authenticate with Modal
modal token new
```

### Run Your First Benchmark

```bash
# Run a single commit benchmark (parallel mode, 3 GPUs)
python -m src.benchmark.run_single_commit 09deb20d --repo sglang --parallel
```

This will:
1. Fetch commit info from HuggingFace dataset
2. Find agent patch from `perf-agents-bench/state/runs/`
3. Spin up 3 H100 GPUs on Modal
4. Run baseline, human, and agent benchmarks in parallel
5. Save results to `benchmark_results/sglang/09deb20d/{timestamp}/`

### View Results

```bash
# Check the results folder
ls benchmark_results/sglang/09deb20d/

# Read the summary
cat benchmark_results/sglang/09deb20d/*/summary.txt
```

---

## 3. System Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌─────────────┐     ┌─────────────────┐     ┌──────────────────────────┐   │
│  │ HuggingFace │     │   CLI Entry     │     │      Modal Cloud         │   │
│  │   Dataset   │────▶│   Point         │────▶│                          │   │
│  │             │     │                 │     │  ┌──────┐  ┌──────┐      │   │
│  │ commit_hash │     │ run_single_     │     │  │ H100 │  │ H100 │      │   │
│  │ perf_command│     │ commit.py       │     │  │      │  │      │      │   │
│  │ models      │     │                 │     │  │ BASE │  │HUMAN │      │   │
│  │ hardware    │     │ ────────────────│     │  └──────┘  └──────┘      │   │
│  └─────────────┘     │                 │     │                          │   │
│                      │ Fetches data,   │     │      ┌──────┐            │   │
│  ┌─────────────┐     │ finds patches,  │     │      │ H100 │            │   │
│  │ Agent Runs  │────▶│ launches Modal  │     │      │      │            │   │
│  │             │     │                 │     │      │AGENT │            │   │
│  │ model_patch │     └─────────────────┘     │      └──────┘            │   │
│  │ .diff       │                             │                          │   │
│  └─────────────┘                             └──────────────────────────┘   │
│                                                          │                  │
│                                                          ▼                  │
│                                              ┌──────────────────────────┐   │
│                                              │    Results Storage       │   │
│                                              │                          │   │
│                                              │  Local: benchmark_       │   │
│                                              │         results/         │   │
│                                              │                          │   │
│                                              │  Modal: sglang-          │   │
│                                              │         benchmark-       │   │
│                                              │         results volume   │   │
│                                              └──────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| CLI Entry Point | `run_single_commit.py` | User-facing script to run benchmarks |
| Modal Worker | `modal/sglang_benchmark.py` | GPU execution logic (3,387 lines) |
| Batch Runner | `runners/hero_sglang.py` | Run multiple commits sequentially |
| Docker Builder | `tools/build_sglang_images.py` | Build Docker images for commits |
| Commit Data | `plans/sglang_h100_commits.jsonl` | 57 H100-compatible commits |

---

## 4. File Structure

```
src/benchmark/
│
├── run_single_commit.py           # CLI entry point
│                                  # Usage: python -m src.benchmark.run_single_commit <commit>
│
├── modal/
│   └── sglang_benchmark.py        # Main GPU worker (3,387 lines)
│                                  # Contains:
│                                  #   - run_3way_benchmark_docker_parallel()
│                                  #   - _create_single_phase_script()
│                                  #   - _run_single_phase_sandbox()
│                                  #   - parse_benchmark_output()
│
├── runners/
│   └── hero_sglang.py             # Batch orchestrator
│                                  # Usage: python -m src.benchmark.runners.hero_sglang --limit 10
│
├── tools/
│   └── build_sglang_images.py     # Docker image builder
│                                  # Usage: python tools/build_sglang_images.py --commit abc123
│
├── plans/
│   ├── sglang_h100_commits.jsonl  # 57 H100-compatible commits (JSONL)
│   │                              # Schema: {commit_hash, commit_subject, pr_url, models, perf_command, hardware}
│   │
│   └── sglang_h100_benchmark_plan.md  # Planning document
│
├── fixes/
│   ├── sglang_commit_mapping_verified.json  # Commit → Docker image mapping
│   │
│   └── sglang_docker_builds_full.csv        # Docker build status (74 commits)
│                                            # Columns: human_commit, base_commit, pr_number, status, subject
│
└── README_SGLANG.md               # This file!


perf-agents-bench/
└── state/
    └── runs/
        └── sglang/
            └── claude_code/
                └── default/
                    └── {date}/
                        └── sglang_{idx}_{commit}/
                            ├── model_patch.diff    # Agent's generated patch
                            └── journal.json        # Run metadata


benchmark_results/                  # Local results storage
└── sglang/
    └── {commit_short}/
        └── {timestamp}/
            ├── metrics.json        # Parsed benchmark metrics
            ├── metadata.json       # Run metadata
            ├── baseline_raw.txt    # Full baseline output
            ├── human_raw.txt       # Full human output
            ├── agent_raw.txt       # Full agent output
            ├── agent_patch.diff    # The agent's patch
            ├── perf_command.txt    # The benchmark command
            ├── summary.txt         # Human-readable summary
            └── complete_result.json # Full result (for debugging)
```

---

## 5. Data Sources

### HuggingFace Dataset

**Dataset**: [`Ayushnangia/omniperf_v1`](https://huggingface.co/datasets/Ayushnangia/omniperf_v1)
**Config**: `sglang`
**Split**: `train`

```python
from datasets import load_dataset
ds = load_dataset("Ayushnangia/omniperf_v1", "sglang", split="train")
```

### Dataset Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `commit_hash` | string | Full 40-char commit hash | `09deb20deef8181a23f66c933ea74b86fee47366` |
| `commit_subject` | string | Commit message title | `Optimize memory usage of logits processor (#420)` |
| `pr_url` | string | GitHub PR URL | `https://github.com/sgl-project/sglang/pull/420` |
| `models` | list | Models to benchmark | `["meta-llama/Llama-3.1-8B-Instruct"]` |
| `perf_command` | string | Benchmark command | `python -m sglang.bench_serving --model ...` |
| `hardware` | string | Required hardware | `H100`, `H100-TP8`, `AMD-MI300X` |
| `has_serving` | bool | Uses serving benchmark | `true` |
| `diff_text` | string | The human's PR patch | `--- a/file.py\n+++ b/file.py\n...` |

### Commit Statistics

```
Total SGLang commits:     74
H100 single GPU:          57 (77%)  ← Focus of this system
H100 multi-GPU (TP8):     10 (13%)
AMD MI300X:                4 (5%)
Other (Blackwell, etc.):   3 (4%)
```

---

## 6. Docker Images

### DockerHub Repository

**Registry**: `ayushnangia16/sglang-docker`
**Total Images**: ~144 (74 human + 74 baseline, with some overlap)

### Image Naming Convention

```
ayushnangia16/sglang-docker:{full_40_char_commit_hash}
```

**Examples**:
```
ayushnangia16/sglang-docker:09deb20deef8181a23f66c933ea74b86fee47366  # Human commit
ayushnangia16/sglang-docker:33b242df303e03886835d08a583fefe979a3ee88  # Base commit
```

### What's in Each Image?

Each Docker image contains:
- SGLang installed at the specific commit
- All dependencies (PyTorch, Triton, etc.)
- Pre-compiled wheels for faster startup

### How Images Are Used

```
BASELINE phase:  Uses base commit Docker image
                 ayushnangia16/sglang-docker:{base_commit}

HUMAN phase:     Uses human commit Docker image
                 ayushnangia16/sglang-docker:{human_commit}

AGENT phase:     Uses base commit Docker image + applies agent patch
                 ayushnangia16/sglang-docker:{base_commit} + agent_patch.diff
```

### Building New Images

If an image is missing:

```bash
# Build Docker image for a specific commit
python tools/build_sglang_images.py --commit 09deb20deef8181a23f66c933ea74b86fee47366

# Check if image exists
docker pull ayushnangia16/sglang-docker:09deb20deef8181a23f66c933ea74b86fee47366
```

### Image Status CSV

Check `fixes/sglang_docker_builds_full.csv` for build status:

```csv
index,human_commit,base_commit,pr_number,status,subject
1,021f76e4f49861b2e9ea9ccff06a46d577e3c548,777688b8929c877e4e28c2eac208d776abe4c3af,6994,OK,[Perf] Refactor LoRAManager
2,09deb20deef8181a23f66c933ea74b86fee47366,33b242df303e03886835d08a583fefe979a3ee88,420,OK,Optimize memory usage
...
```

---

## 7. Modal Volumes

Modal [volumes](https://modal.com/docs/guide/volumes) provide persistent storage across runs.

### Available Volumes

| Volume Name | Mount Path | Purpose |
|-------------|------------|---------|
| `sglang-model-cache` | `/root/.cache/huggingface` | HuggingFace model weights (Llama, DeepSeek, etc.) |
| `sglang-build-cache` | `/build` | Compiled SGLang wheels (faster reinstalls) |
| `sglang-benchmark-results` | `/results` | Benchmark result JSONs (persistent storage) |

### How Volumes Are Created

Volumes are created automatically if they don't exist:

```python
# In sglang_benchmark.py
model_cache = modal.Volume.from_name("sglang-model-cache", create_if_missing=True)
build_cache = modal.Volume.from_name("sglang-build-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("sglang-benchmark-results", create_if_missing=True)
```

### Checking Volume Contents

```bash
# List files in a volume (via Modal CLI)
modal volume ls sglang-benchmark-results

# Download file from volume
modal volume get sglang-benchmark-results /09deb20d/20250103_120000/human_result.json ./
```

### Volume Cost

- Volumes have a small storage cost (~$0.10/GB/month)
- Model cache can be ~50GB (for Llama-3.1-8B, DeepSeek, etc.)
- Results are small (~100KB per commit)

---

## 8. The 3-Way Benchmark Flow

### What Happens in Each Phase

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        3-WAY PARALLEL BENCHMARK                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: BASELINE                                                          │
│  ─────────────────                                                          │
│  1. Pull base commit Docker image                                           │
│  2. Create Modal sandbox with H100 GPU                                      │
│  3. Start SGLang server                                                     │
│  4. Run benchmark command (bench_serving)                                   │
│  5. Parse metrics from output                                               │
│  6. Save results to Modal volume                                            │
│                                                                             │
│  Phase 2: HUMAN                                                             │
│  ──────────────                                                             │
│  1. Pull human commit Docker image (with optimization applied)              │
│  2. Create Modal sandbox with H100 GPU                                      │
│  3. Start SGLang server                                                     │
│  4. Run benchmark command (bench_serving)                                   │
│  5. Parse metrics from output                                               │
│  6. Save results to Modal volume                                            │
│                                                                             │
│  Phase 3: AGENT                                                             │
│  ──────────────                                                             │
│  1. Pull base commit Docker image                                           │
│  2. Apply agent's patch (git apply model_patch.diff)                        │
│  3. Create Modal sandbox with H100 GPU                                      │
│  4. Start SGLang server                                                     │
│  5. Run benchmark command (bench_serving)                                   │
│  6. Parse metrics from output                                               │
│  7. Save results to Modal volume                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Parallel Execution

With `--parallel` flag, all 3 phases run simultaneously on separate GPUs:

```
Time ─────────────────────────────────────────────────────────────────▶

GPU 1:  [====== BASELINE (10-15 min) ======]
GPU 2:  [======= HUMAN (10-15 min) ========]
GPU 3:  [======= AGENT (10-15 min) ========]

Total time: ~15 min (vs ~45 min sequential)
```

### The Benchmark Command

Each phase runs the same command from the dataset:

```bash
python -m sglang.bench_serving \
    --backend sglang \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-prompts 100
```

This:
1. Starts an SGLang server
2. Sends 100 prompts to it
3. Measures throughput and latency
4. Outputs metrics in a parseable format

---

## 9. Usage Examples

### Single Commit Benchmark

```bash
# Basic usage (parallel mode recommended)
python -m src.benchmark.run_single_commit 09deb20d --repo sglang --parallel

# Sequential mode (1 GPU, slower)
python -m src.benchmark.run_single_commit 09deb20d --repo sglang

# Dry run (show what would happen)
python -m src.benchmark.run_single_commit 09deb20d --repo sglang --dry-run

# Human-only mode (no baseline/agent)
python -m src.benchmark.run_single_commit 09deb20d --repo sglang --human-only

# Different GPU config
python -m src.benchmark.run_single_commit 09deb20d --repo sglang --gpu H100:2
```

### Batch Benchmarks

```bash
# Run batch of commits (uses hero_sglang.py)
python -m src.benchmark.runners.hero_sglang --limit 10

# Run all H100 commits
python -m src.benchmark.runners.hero_sglang --hardware H100

# Resume from specific commit
python -m src.benchmark.runners.hero_sglang --start-from 09deb20d
```

### Check Available Commits

```bash
# View H100 commits JSONL
head -5 src/benchmark/plans/sglang_h100_commits.jsonl | jq .

# Count commits by hardware
cat src/benchmark/plans/sglang_h100_commits.jsonl | jq -r .hardware | sort | uniq -c
```

### Check Docker Images

```bash
# Check if image exists
docker pull ayushnangia16/sglang-docker:09deb20deef8181a23f66c933ea74b86fee47366

# List all available images
docker search ayushnangia16/sglang-docker --limit 100
```

### Modal Commands

```bash
# Check Modal volumes
modal volume ls

# Check sandbox status
modal app list

# View logs from recent run
modal app logs sglang-benchmark

# Stop all running sandboxes
modal app stop sglang-benchmark
```

---

## 10. Result Format

### Local Storage

Results are saved to:
```
benchmark_results/sglang/{commit_short}/{timestamp}/
```

### Files Generated

| File | Size | Content |
|------|------|---------|
| `metrics.json` | ~2 KB | Parsed metrics (throughput, latency) |
| `metadata.json` | ~1 KB | Run metadata (commit, PR, model) |
| `baseline_raw.txt` | ~50 KB | Full baseline benchmark output |
| `human_raw.txt` | ~50 KB | Full human benchmark output |
| `agent_raw.txt` | ~50 KB | Full agent benchmark output |
| `agent_patch.diff` | ~1-10 KB | The agent's generated patch |
| `perf_command.txt` | ~200 B | The benchmark command |
| `summary.txt` | ~1 KB | Human-readable summary |
| `complete_result.json` | ~200 KB | Full result (for debugging) |

### metrics.json Schema

```json
{
  "status": "success",
  "baseline_metrics": {
    "request_throughput": 12.34,
    "output_throughput": 4567.89,
    "median_ttft": 0.123,
    "median_tpot": 0.045,
    "median_itl": 0.012
  },
  "human_metrics": {
    "request_throughput": 15.67,
    "output_throughput": 5678.90,
    "median_ttft": 0.098,
    "median_tpot": 0.038,
    "median_itl": 0.010
  },
  "agent_metrics": {
    "request_throughput": 14.23,
    "output_throughput": 5234.56,
    "median_ttft": 0.105,
    "median_tpot": 0.041,
    "median_itl": 0.011
  },
  "human_improvement": {
    "request_throughput_pct": 27.0,
    "output_throughput_pct": 24.3
  },
  "agent_improvement": {
    "request_throughput_pct": 15.3,
    "output_throughput_pct": 14.6
  },
  "agent_vs_human": {
    "request_throughput_ratio": 0.91,
    "output_throughput_ratio": 0.92
  },
  "duration_s": 847.5,
  "gpu_config": "H100:1",
  "benchmark_mode": "parallel_3way"
}
```

### Modal Volume Storage

Results are also saved to the `sglang-benchmark-results` volume:
```
/results/{commit_short}/{timestamp}/
├── baseline_result.json
├── human_result.json
└── agent_result.json
```

---

## 11. Cost Estimate

### Per-Commit Cost

| Resource | Rate | Usage | Cost |
|----------|------|-------|------|
| H100 GPU | ~$3.00/hr | 3 GPUs × 15 min | $2.25 |
| Modal volumes | ~$0.10/GB/mo | Negligible | ~$0 |

**Total per commit**: ~$2.25 (parallel mode)

### Full Dataset Cost

| Scope | Commits | Cost |
|-------|---------|------|
| H100 single GPU | 57 | ~$128 |
| All SGLang commits | 74 | ~$167 |

### Cost Optimization Tips

1. **Use parallel mode** (`--parallel`) - Same cost, 3x faster
2. **Run dry-run first** - Verify setup before using GPUs
3. **Pre-cache models** - Models download once, reused across runs
4. **Batch commits** - Reduce Modal startup overhead

### Free Tier

Modal offers $30/month free credits - enough for ~13 commits.

---

## 12. Troubleshooting

### Common Issues

#### Docker Image Not Found

```
Error: Could not pull image ayushnangia16/sglang-docker:abc123...
```

**Solution**: Build the image or check the commit hash
```bash
# Check if commit exists in dataset
python -c "from datasets import load_dataset; ds = load_dataset('Ayushnangia/omniperf_v1', 'sglang', split='train'); print([r['commit_hash'][:12] for r in ds if 'abc123' in r['commit_hash']])"

# Build image if needed
python tools/build_sglang_images.py --commit <full_40_char_hash>
```

#### Patch Application Fails

```
error: patch failed: python/sglang/file.py:123
```

**Solution**: Check patch compatibility
```bash
# View the patch
cat perf-agents-bench/state/runs/sglang/claude_code/default/*/sglang_*_abc123*/model_patch.diff

# Verify base commit matches
jq .commits.pre perf-agents-bench/state/runs/sglang/claude_code/default/*/sglang_*_abc123*/journal.json
```

#### Model Download Timeout

```
TimeoutError: Model download exceeded timeout
```

**Solution**: Pre-cache the model
```bash
# SSH into Modal sandbox and download
modal shell

# Or increase timeout in sglang_benchmark.py
gpu_cfg['timeout'] = 3600  # 1 hour instead of default
```

#### No Agent Patch Found

```
Warning: No agent patch found for abc12345
```

**Solution**: Run the agent first
```bash
# Check if agent runs exist
ls perf-agents-bench/state/runs/sglang/claude_code/default/

# The commit needs an agent run in perf-agents-bench
```

#### Modal Authentication Error

```
modal.exception.AuthError: No Modal token found
```

**Solution**: Authenticate with Modal
```bash
modal token new
```

#### Out of GPU Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution**: Use larger GPU config
```bash
python -m src.benchmark.run_single_commit abc123 --repo sglang --gpu H100:2
```

### Debug Mode

For detailed logs:

```python
# In run_single_commit.py, change log level
logging.basicConfig(level=logging.DEBUG, ...)
```

### Check Modal Status

```bash
# View running apps
modal app list

# View recent logs
modal app logs sglang-benchmark --last 100

# Kill stuck sandbox
modal app stop sglang-benchmark
```

---

## 13. Environment Variables

### Required

```bash
# Modal authentication (get from modal.com dashboard)
export MODAL_TOKEN_ID=ak-xxxxxxxxxxxxxx
export MODAL_TOKEN_SECRET=as-xxxxxxxxxxxxxx

# HuggingFace token (for model downloads)
export HF_TOKEN=hf_xxxxxxxxxxxxxx
```

### Optional

```bash
# GitHub token (for commit data, usually not needed)
export GHAPI_TOKEN=ghp_xxxxxxxxxxxxxx

# Custom results directory
export BENCHMARK_RESULTS_DIR=/custom/path/results
```

### Setting Up Modal Secret

HuggingFace token is stored as a Modal secret:

```bash
# Create the secret (one-time)
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxx
```

The benchmark code references it as:
```python
secrets=[modal.Secret.from_name("huggingface-secret")]
```

---

## 14. Quick Reference

### Commands Cheat Sheet

```bash
# Single commit (recommended)
python -m src.benchmark.run_single_commit <commit> --repo sglang --parallel

# Dry run
python -m src.benchmark.run_single_commit <commit> --repo sglang --dry-run

# Batch run
python -m src.benchmark.runners.hero_sglang --limit 10

# Check results
ls benchmark_results/sglang/<commit>/
cat benchmark_results/sglang/<commit>/*/summary.txt

# Modal commands
modal volume ls sglang-benchmark-results
modal app list
modal app stop sglang-benchmark
```

### File Locations

| What | Where |
|------|-------|
| CLI entry point | `src/benchmark/run_single_commit.py` |
| Modal worker | `src/benchmark/modal/sglang_benchmark.py` |
| Commit list | `src/benchmark/plans/sglang_h100_commits.jsonl` |
| Docker status | `src/benchmark/fixes/sglang_docker_builds_full.csv` |
| Agent patches | `perf-agents-bench/state/runs/sglang/claude_code/` |
| Local results | `benchmark_results/sglang/{commit}/` |

### Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `run_3way_benchmark_docker_parallel()` | `sglang_benchmark.py` | Main entry point |
| `_run_single_phase_sandbox()` | `sglang_benchmark.py` | Run one phase |
| `_create_single_phase_script()` | `sglang_benchmark.py` | Generate bash script |
| `parse_benchmark_output()` | `sglang_benchmark.py` | Extract metrics |
| `find_dataset_row()` | `run_single_commit.py` | Query HuggingFace |
| `find_agent_patch()` | `run_single_commit.py` | Find agent's patch |

### Metrics Glossary

| Metric | Description | Unit | Good Direction |
|--------|-------------|------|----------------|
| `request_throughput` | Requests processed per second | req/s | Higher |
| `output_throughput` | Tokens generated per second | tokens/s | Higher |
| `median_ttft` | Time to first token | seconds | Lower |
| `median_tpot` | Time per output token | seconds | Lower |
| `median_itl` | Inter-token latency | seconds | Lower |

---

## Need Help?

1. **Check this README** - Most answers are here
2. **Check the code** - `sglang_benchmark.py` is well-documented
3. **Check Modal docs** - [modal.com/docs](https://modal.com/docs)
4. **Check SGLang docs** - [sgl-project.github.io](https://sgl-project.github.io/)

---

*Last updated: January 2026*
