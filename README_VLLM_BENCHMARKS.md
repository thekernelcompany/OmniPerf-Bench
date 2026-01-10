# vLLM Performance Benchmark Infrastructure

## Overview

This infrastructure benchmarks human vs AI agent performance optimizations on vLLM. It compares:
- **Baseline**: Parent commit (before optimization)
- **Human**: The actual PR commit (human-written optimization)
- **Agent**: Claude's generated patch applied to baseline

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support (tested on A100/H100)
- Docker with NVIDIA Container Toolkit
- Python 3.10+
- ~50GB disk space for Docker images

### Installation

```bash
# Clone the repository
git clone https://github.com/thekernelcompany/OmniPerf-Bench.git
cd OmniPerf-Bench

# Install dependencies
pip install requests tqdm

# Verify Docker access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Running Benchmarks

### Option 1: Run All 3-Way Benchmarks

```bash
python run_3way_benchmarks.py
```

This will:
1. Pull Docker images from `shikhar481/vllm_fixed_human_images`
2. Run baseline, human, and agent benchmarks for each commit
3. Save results to `omniperf_results_3way_claude_code/`

### Option 2: Run Single Commit Benchmark

```bash
python local_docker_benchmark.py --commit <commit_hash> --type <baseline|human|agent>
```

Example:
```bash
# Run human benchmark for commit 015069b0
python local_docker_benchmark.py --commit 015069b0 --type human
```

### Option 3: Custom Context Length (for OOM issues)

```bash
python run_benchmark_custom_context.py --commit <commit_hash> --max-model-len 2048
```

## Docker Images

### Repository
```
Docker Hub: shikhar481/vllm_fixed_human_images
```

### Image Naming Convention

| Type | Tag Format | Example |
|------|------------|---------|
| Human | `<full_commit_hash>` | `015069b01741e9ecb9e604c7fe87fbdfc306ebe5` |
| Baseline | `baseline-<parent_hash_12>` | `baseline-fbefc8a78d22` |

### List Available Images

```bash
# List all human images
curl -s "https://hub.docker.com/v2/repositories/shikhar481/vllm_fixed_human_images/tags?page_size=100" | \
  python3 -c "import sys,json; [print(r['name']) for r in json.load(sys.stdin)['results'] if not r['name'].startswith('baseline-')]"

# List all baseline images
curl -s "https://hub.docker.com/v2/repositories/shikhar481/vllm_fixed_human_images/tags?page_size=100" | \
  python3 -c "import sys,json; [print(r['name']) for r in json.load(sys.stdin)['results'] if r['name'].startswith('baseline-')]"
```

## Benchmark Methodology

### Test Parameters

| Parameter | Value |
|-----------|-------|
| Prompts | 100 |
| Output tokens | 64 |
| Concurrency | 16 |
| Warmup requests | 10 |
| Timeout | 600s |

### Metrics Collected

- **Throughput** (tokens/second) - Primary metric
- **TTFT** (Time to First Token)
- **TPOT** (Time Per Output Token)
- **Request latency**

### Benchmark Script

The benchmark uses vLLM's official `benchmark_serving.py`:

```bash
python benchmark_serving.py \
  --backend openai-chat \
  --model <model_name> \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --num-prompts 100 \
  --random-input-len 512 \
  --random-output-len 64 \
  --request-rate inf
```

## Results Structure

```
omniperf_results_3way_claude_code/
├── baseline_benchmark_results/
│   ├── 015069b0_baseline_result.json
│   └── ...
├── agent_benchmark_results/
│   ├── 015069b0_agent_result.json
│   ├── 015069b0_human_result.json
│   └── ...
└── docker_benchmark_results/
    └── ...
```

### Result JSON Format

```json
{
  "human_commit": "015069b0",
  "human_commit_full": "015069b01741e9ecb9e604c7fe87fbdfc306ebe5",
  "parent_commit": "fbefc8a78d22...",
  "model": "Qwen/Qwen3-1.7B",
  "status": "success",
  "metrics": {
    "throughput_tok_s": 198.29,
    "mean_ttft_ms": 45.2,
    "mean_tpot_ms": 12.3
  },
  "duration_s": 120.5,
  "timestamp": "2026-01-10 12:00:00"
}
```

## Reproducing Results

### Step 1: Verify Docker Images Exist

```bash
# Check if commit has both human and baseline images
COMMIT="015069b0"
HUMAN_TAG="${COMMIT}1741e9ecb9e604c7fe87fbdfc306ebe5"  # Full hash

docker pull shikhar481/vllm_fixed_human_images:${HUMAN_TAG}
docker pull shikhar481/vllm_fixed_human_images:baseline-fbefc8a78d22
```

### Step 2: Run Benchmark

```bash
# Run all three benchmarks
python run_3way_benchmarks.py --commits 015069b0

# Or run individually
python local_docker_benchmark.py --commit 015069b0 --type baseline
python local_docker_benchmark.py --commit 015069b0 --type human
python local_docker_benchmark.py --commit 015069b0 --type agent
```

### Step 3: Compare Results

```python
import json

baseline = json.load(open('omniperf_results_3way_claude_code/baseline_benchmark_results/015069b0_baseline_result.json'))
human = json.load(open('omniperf_results_3way_claude_code/agent_benchmark_results/015069b0_human_result.json'))
agent = json.load(open('omniperf_results_3way_claude_code/agent_benchmark_results/015069b0_agent_result.json'))

print(f"Baseline: {baseline['metrics'].get('throughput_tok_s', 'N/A')} tok/s")
print(f"Human: {human['metrics'].get('throughput_tok_s', 'N/A')} tok/s")
print(f"Agent: {agent['metrics'].get('throughput_tok_s', 'N/A')} tok/s")
```

## Available Commits

### Complete 3-Way Benchmarks (22 commits)

| Commit | Model | Human (tok/s) | Agent (tok/s) | Diff |
|--------|-------|---------------|---------------|------|
| 015069b0 | Qwen/Qwen3-1.7B | 198 | 198 | 0% |
| 22d33bac | Meta-Llama-3-8B-Instruct | 3946 | 3985 | +1.0% |
| 6e36f4fa | Meta-Llama-3-8B-Instruct | 2414 | 2784 | +15.4% |
| e3580537 | Meta-Llama-3-8B-FP8 | 2497 | 3107 | +24.4% |
| fc7b8d1e | Meta-Llama-3-8B-Instruct | 2214 | 2598 | +17.3% |
| 89a84b0b | Qwen/Qwen1.5-0.5B | 3559 | 2967 | -16.6% |
| ... | ... | ... | ... | ... |

See `DOCKER_IMAGE_ANALYSIS.md` for full list.

### Agent Failure Cases (2 commits)

| Commit | Human Result | Agent Result |
|--------|--------------|--------------|
| 35fad35a | 3173 tok/s | CRASHED |
| ad8d696a | 2383 tok/s | CRASHED |

## Troubleshooting

### OOM Errors

Reduce context length:
```bash
python run_benchmark_custom_context.py --commit <hash> --max-model-len 2048
```

### Server Startup Timeout

Increase timeout in `local_docker_benchmark.py`:
```python
SERVER_STARTUP_TIMEOUT = 300  # seconds
```

### Missing Docker Image

Check if image exists:
```bash
docker manifest inspect shikhar481/vllm_fixed_human_images:<tag>
```

If missing, the commit cannot be benchmarked without building the image.

### CUDA/Driver Mismatch

Ensure CUDA version compatibility:
```bash
nvidia-smi  # Check driver version
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Analysis Files

| File | Description |
|------|-------------|
| `DOCKER_IMAGE_ANALYSIS.md` | Complete Docker image availability analysis |
| `CRITICAL_FAILURE_ANALYSIS.md` | Breakdown of all 96 commits by failure category |
| `COMPREHENSIVE_BENCHMARK_ANALYSIS.md` | Full benchmark results and statistics |
| `comprehensive_commit_analysis.json` | Structured data of all commits |
| `detailed_failure_analysis.json` | Detailed failure reasons per commit |

## Contributing

To add new commits to benchmark:

1. Build Docker images for human commit and its parent (baseline)
2. Push to `shikhar481/vllm_fixed_human_images`
3. Add commit to `matched_commits_for_benchmark.json`
4. Run benchmark using `run_3way_benchmarks.py`

## License

MIT

---

*Generated: 2026-01-10*
