# Claude Code vLLM Benchmarks

This directory contains benchmark results from running Claude Code (claude-sonnet-4) on vLLM performance optimization tasks.

## Overview

The benchmark evaluates Claude Code's ability to reproduce performance optimizations from real vLLM commits. For each commit:

1. **Baseline**: Run benchmark on parent commit (before optimization)
2. **Human**: Run benchmark on the actual commit (human's optimization)
3. **Agent**: Run benchmark with Claude Code's attempted optimization patch

## Results Summary

- **Total commits tested**: 94
- **Successful benchmarks**: 20 (with full 3-way comparison)
- **Agent matched/exceeded human performance**: 75% (TPOT metric)

## Running the Benchmarks

### Prerequisites

1. **Modal account**: Sign up at https://modal.com
2. **HuggingFace token**: For model access
3. **Python environment**:
   ```bash
   cd /home/ubuntu/OmniPerf-Bench
   source .venv/bin/activate
   ```

### Step 1: Deploy Modal App

Deploy the benchmark infrastructure to Modal:

```bash
modal deploy src/eval/modal_benchmark.py
```

This creates:
- `omniperf-benchmark` app with GPU functions (H100:1, H100:2, H100:4, H100:8)
- CPU-only build functions for wheel compilation
- Model cache and build cache volumes

### Step 2: Run Single Benchmark

```python
from src.eval.modal_benchmark import run_3way_modal_benchmark

result = run_3way_modal_benchmark(
    baseline_wheel_url="https://vllm-wheels.s3.us-west-2.amazonaws.com/{parent_commit}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl",
    human_wheel_url="https://vllm-wheels.s3.us-west-2.amazonaws.com/{commit}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl",
    agent_patch="<unified diff from agent>",
    perf_command="python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100",
    model="meta-llama/Llama-3.1-8B-Instruct",
    gpu_config="H100:1",
    base_commit="<parent_commit_hash>",
    human_commit="<commit_hash>",
)
```

### Step 3: Run Batch Benchmarks

Use the hero benchmark runner for batch processing:

```bash
# Create commit plan
python hero_benchmark_runner.py --dry-run

# Run all commits
python hero_benchmark_runner.py

# Resume from specific commit
python hero_benchmark_runner.py --start-from abc12345
```

## Architecture

### Key Components

1. **modal_benchmark.py**: Main benchmark infrastructure
   - `run_3way_modal_benchmark()`: Entry point for 3-way comparison
   - `build_vllm_cpu_only()`: Build wheels on cheap CPU instances
   - `ensure_vllm_build_cached()`: Cache management for builds
   - GPU benchmark functions (`run_3way_benchmark_1gpu`, etc.)

2. **hero_benchmark_runner.py**: Batch runner
   - Processes multiple commits from HuggingFace dataset
   - Handles GPU configuration selection
   - Saves incremental results

### Build/Wheel System

The system uses a 2-phase architecture:

1. **Phase 1 (CPU)**: Build vLLM wheels on $0.20/hr CPU instances
   - Wheels cached in Modal volume for reuse
   - No GPU time wasted on compilation

2. **Phase 2 (GPU)**: Run benchmarks on H100 instances
   - Install pre-built wheels
   - Execute benchmark commands
   - Compare baseline/human/agent performance

### Wheel URL Validation (Fix Applied)

The system now validates S3 wheel URLs before use. If a wheel doesn't exist:
- Triggers CPU build instead of failing
- Caches built wheel in Modal volume
- Uses Python overlay for old commits without wheels

### Blocked Models

The following models are blocked (too large/unstable):
- DeepSeek-V3, DeepSeek-R1
- Nemotron-4-340B
- DBRX
- Llama-4-Scout-17B-16E-Instruct

### CLI Version Compatibility

Old vLLM versions don't support new CLI arguments. The system automatically strips:
- `--backend vllm`
- `--enable-prefix-caching`
- `--use-v2-block-manager`
- And other version-specific arguments

## Result Format

Each benchmark result is saved as JSON:

```json
{
  "instance": {
    "commit_hash": "abc123...",
    "commit_subject": "[Perf] Optimization description",
    "perf_command": "python benchmarks/benchmark_serving.py ..."
  },
  "result": {
    "status": "success",
    "baseline_metrics": {"ttft_mean": 100.0, "tpot_mean": 10.0, ...},
    "human_metrics": {"ttft_mean": 90.0, "tpot_mean": 9.0, ...},
    "agent_metrics": {"ttft_mean": 92.0, "tpot_mean": 9.2, ...},
    "human_improvement": {"ttft_mean": 10.0, "tpot_mean": 10.0},
    "agent_improvement": {"ttft_mean": 8.0, "tpot_mean": 8.0},
    "agent_vs_human": {"ttft_mean": -2.0, "tpot_mean": -2.0}
  }
}
```

## Metrics Explained

- **TTFT (Time To First Token)**: Latency until first token generated
- **TPOT (Time Per Output Token)**: Average inter-token latency
- **ITL (Inter-Token Latency)**: Similar to TPOT
- **Throughput**: Tokens per second

Improvement percentages are calculated as:
```
improvement = (baseline - optimized) / baseline * 100
```

Positive values indicate improvement (lower is better for latency metrics).

## Troubleshooting

### "No wheel available and no ancestor wheel found"
- S3 wheels don't exist for old commits
- System will now auto-build from source (fix applied)

### "unrecognized arguments: --backend vllm"
- Old vLLM version doesn't support `--backend`
- System now strips incompatible arguments (fix applied)

### Server startup timeout
- Default increased to 20 minutes
- Large models may need more time to load weights

### Container OOM/crash
- Blocked models list prevents running known-problematic models
- Check GPU memory requirements match model size
