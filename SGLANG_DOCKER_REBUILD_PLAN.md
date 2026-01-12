# Plan: Rebuild SGLang Docker Images with sgl-kernel

## Problem Analysis

### Critical Findings

The current Docker images at `shikhar481/sglang-images` are **broken**. The SGLang server crashes with:
```
ModuleNotFoundError: No module named 'deep_gemm'
```

### Root Cause Investigation

**My initial assumption was wrong.** I proposed to add `pip install sgl-kernel` from PyPI. This would NOT work because:

1. **Version Mismatch**: The commits require specific sgl-kernel versions that **don't exist on PyPI**:
   | Commit | Required sgl-kernel | Required torch | PyPI Available? |
   |--------|---------------------|----------------|-----------------|
   | d1112d85 | 0.0.5.post2 | 2.5.1 | NO (earliest is 0.1.0) |
   | 93470a14 | 0.0.8 | 2.5.1 | NO |
   | 9c088829 | 0.0.9.post2 | 2.6.0 | NO |

2. **Torch Version**: Even if we used latest PyPI sgl-kernel (0.3.20), it requires torch==2.9.1, not 2.4.0/2.5.1/2.6.0

3. **The Real Solution**: sgl-kernel must be **built from source** at each commit using git submodules

### sgl-kernel Build Requirements

At those commits, sgl-kernel uses `setup.py` with torch CUDA extensions and requires these git submodules:
```
sgl-kernel/3rdparty/cutlass     → https://github.com/NVIDIA/cutlass.git
sgl-kernel/3rdparty/cccl        → https://github.com/NVIDIA/cccl.git
sgl-kernel/3rdparty/flashinfer  → https://github.com/flashinfer-ai/flashinfer.git
sgl-kernel/3rdparty/deepgemm    → https://github.com/deepseek-ai/DeepGEMM
```

The build copies `deepgemm` to Python's site-packages as the `deep_gemm` module.

## Corrected Overview

Rebuild all 6 Docker images with **sgl-kernel built from source** (not from PyPI), then run 3-way benchmarks.

## The 6 Docker Images

| Type | Commit | Model | Benchmark Command |
|------|--------|-------|-------------------|
| Human | `d1112d85` | `google/gemma-2-2b` | `python -m sglang.bench_serving --model google/gemma-2-2b --num-prompts 100` |
| Parent | `48efec7b` | (same) | (same) |
| Human | `93470a14` | `meta-llama/Llama-3.1-8B-Instruct` | `python3 -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct` |
| Parent | `db452760` | (same) | (same) |
| Human | `9c088829` | `meta-llama/Llama-3.1-8B-Instruct` | `python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100` |
| Parent | `005aad32` | (same) | (same) |

## Agent Patches Available

- `sglang_064_d1112d85/model_patch.diff`
- `sglang_043_93470a14/model_patch.diff`
- `sglang_045_9c088829/model_patch.diff`

## Architecture

Following the vLLM local Docker benchmark pattern from `scripts/runners/local_docker_benchmark.py`:

### 3-Way Benchmark Flow

For each commit pair (human + parent):

1. **BASELINE Phase**: Run benchmark on parent commit Docker image
2. **HUMAN Phase**: Run benchmark on human commit Docker image
3. **AGENT Phase**: Run benchmark on parent commit Docker image + apply agent patch

### Key Differences from vLLM

| Aspect | vLLM | SGLang |
|--------|------|--------|
| Server | `python -m vllm.entrypoints.openai.api_server` | `python -m sglang.launch_server` |
| Benchmark | `benchmark_serving.py` | `sglang.bench_serving` |
| Health check | `/v1/models` | `/health` or port check |
| Docker registry | `ayushnangia16/nvidia-vllm-docker` | `shikhar481/sglang-images` |

## Implementation Plan

### Step 1: Fix the Dockerfile Template in `build_sglang_images.py`

The current Dockerfile template is wrong. Update to:

```dockerfile
ARG CUDA_VERSION=12.4.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH="${PATH}:/usr/local/cuda/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git curl wget build-essential cmake ninja-build \
    libopenmpi-dev libnuma-dev \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Clone SGLang at specific commit WITH SUBMODULES
ARG COMMIT_HASH
WORKDIR /opt
RUN git clone --recursive https://github.com/sgl-project/sglang.git sglang \
    && cd sglang \
    && git checkout ${COMMIT_HASH} \
    && git submodule update --init --recursive

# Install PyTorch (version from commit's pyproject.toml)
ARG TORCH_VERSION=2.5.1
RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu124

# Build sgl-kernel FROM SOURCE (not from PyPI!)
WORKDIR /opt/sglang/sgl-kernel
RUN pip install scikit-build-core ninja
RUN pip install -e . --no-build-isolation

# Verify sgl-kernel build
RUN python -c "import sgl_kernel; import deep_gemm; print('sgl-kernel OK')"

# Install flashinfer
ARG TORCH_MINOR=2.5
RUN pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch${TORCH_MINOR}/

# Install SGLang dependencies and package
WORKDIR /opt/sglang
RUN pip install transformers huggingface_hub tokenizers accelerate numpy<2.0 \
    requests aiohttp triton packaging vllm datasets pandas tqdm
RUN pip install -e "python" --no-deps

# Verify full installation
RUN python -c "import sglang; import deep_gemm; print(f'SGLang {sglang.__version__} ready')"

WORKDIR /workspace
CMD ["python", "-c", "import sglang; print('ready')"]
```

**Key changes:**
1. `git clone --recursive` to get submodules
2. `git submodule update --init --recursive` after checkout
3. Build sgl-kernel from source with `pip install -e .`
4. Use correct torch version per commit (2.5.1 or 2.6.0)
5. Verify `deep_gemm` import works

### Step 2: Modify `build_sglang_images.py` to Support Per-Commit Config

Add per-commit torch version detection:

```python
COMMIT_CONFIG = {
    "d1112d85": {"torch": "2.5.1", "torch_minor": "2.5"},
    "48efec7b": {"torch": "2.5.1", "torch_minor": "2.5"},
    "93470a14": {"torch": "2.5.1", "torch_minor": "2.5"},
    "db452760": {"torch": "2.5.1", "torch_minor": "2.5"},
    "9c088829": {"torch": "2.6.0", "torch_minor": "2.6"},
    "005aad32": {"torch": "2.6.0", "torch_minor": "2.6"},
}
```

### Step 3: Rebuild All 6 Images

```bash
# Rebuild each image with correct config
for commit in d1112d85 48efec7b 93470a14 db452760 9c088829 005aad32; do
    python src/benchmark/tools/build_sglang_images.py --commit $commit
done
```

### Step 4: Benchmark Script (already created)

The `scripts/runners/local_docker_sglang_benchmark.py` script was already created. It will work once images are fixed.

A new script (~800 lines) modeled after the vLLM version with these functions:

```python
# Configuration
DOCKER_REPO = "shikhar481/sglang-images"
RESULTS_DIR = Path("omniperf_results_3way_sglang_local")
AGENT_PATCHES_DIR = Path("perf-agents-bench/state/runs/sglang/claude_code")

# Commit mapping (hardcoded for these 6 images)
COMMIT_PAIRS = [
    {
        "human_commit": "d1112d8548eb13c842900b3a8d622345f9737759",
        "parent_commit": "48efec7b052354865aa2f0605a5bf778721f3cbb",
        "model": "google/gemma-2-2b",
        "perf_command": "python -m sglang.bench_serving --model google/gemma-2-2b --num-prompts 100",
    },
    {
        "human_commit": "93470a14116a60fe5dd43f0599206e8ccabdc211",
        "parent_commit": "db452760e5b2378efd06b1ceb9385d2eeb6d217c",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "perf_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
    {
        "human_commit": "9c088829ee2a28263f36d0814fde448c6090b5bc",
        "parent_commit": "005aad32ad45ce27d73fd39aa1f7e9ba5d8ebb8f",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "perf_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
]

# Core functions
def run_sglang_benchmark(docker_image: str, model: str, perf_command: str, ...) -> BenchmarkResult
def parse_sglang_metrics(output: str) -> Dict[str, float]
def run_3way_benchmark(commit_pair: dict, agent_patch: Optional[Path]) -> dict
def main()
```

### Step 2: Docker Execution Pattern

```bash
docker run --rm --gpus all \
    -e HF_TOKEN=$HF_TOKEN \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    --shm-size=16g \
    --entrypoint bash \
    shikhar481/sglang-images:${COMMIT} \
    -c "BENCHMARK_SCRIPT"
```

### Step 3: Benchmark Script Template (inside container)

```bash
# Start SGLang server
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --port 30000 \
    --host 127.0.0.1 &

# Wait for server (check /health endpoint or port)
for i in {1..300}; do
    if curl -s http://127.0.0.1:30000/health 2>/dev/null; then
        echo "SERVER_READY"
        break
    fi
    sleep 1
done

# Run benchmark
python -m sglang.bench_serving \
    --backend sglang \
    --model $MODEL \
    --num-prompts 100 \
    --port 30000

# Kill server
pkill -f sglang.launch_server
```

### Step 4: Agent Patch Application (for AGENT phase)

```bash
# Clone SGLang repo
git clone https://github.com/sgl-project/sglang.git /tmp/sglang_patch
cd /tmp/sglang_patch
git checkout $PARENT_COMMIT

# Apply agent patch
git apply /tmp/agent.patch

# Overlay Python files onto installed SGLang
cp -r /tmp/sglang_patch/python/sglang/* /path/to/installed/sglang/
```

### Step 5: Metric Parsing

Parse same metrics as Modal version:
- `request_throughput` (req/s)
- `output_throughput` (tok/s)
- `ttft_mean`, `ttft_median`, `ttft_p99` (ms)
- `tpot_mean`, `tpot_median`, `tpot_p99` (ms)
- `itl_mean`, `itl_median` (ms)

## Files to Create

1. **`scripts/runners/local_docker_sglang_benchmark.py`** (~800 lines)
   - Main benchmark runner
   - Adapts vLLM pattern for SGLang

## Usage

```bash
# Run all 3 commit pairs
python scripts/runners/local_docker_sglang_benchmark.py

# Run specific commit
python scripts/runners/local_docker_sglang_benchmark.py --commit d1112d85

# Dry run
python scripts/runners/local_docker_sglang_benchmark.py --dry-run

# Skip agent phase (baseline + human only)
python scripts/runners/local_docker_sglang_benchmark.py --no-agent
```

## Execution Plan

Run **all 3 commit pairs** with **full 3-way benchmark** (baseline + human + agent):

1. **Commit 1: d1112d85** (gemma-2-2b, ~8GB VRAM, fastest)
2. **Commit 2: 93470a14** (Llama-3.1-8B, ~16GB VRAM)
3. **Commit 3: 9c088829** (Llama-3.1-8B, ~16GB VRAM)

Estimated time: ~30-60 minutes per commit pair (3 phases each)

## Files to Modify

1. **`src/benchmark/tools/build_sglang_images.py`**
   - Replace BENCHMARK_DOCKERFILE with corrected version
   - Add per-commit torch version config
   - Add submodule initialization step

## Verification

### Phase 1: Verify Image Build
```bash
# Test build for one commit first
python src/benchmark/tools/build_sglang_images.py --commit d1112d85 --no-push

# Verify sgl-kernel works
docker run --rm --gpus all shikhar481/sglang-images:d1112d85... \
    python -c "import sgl_kernel; import deep_gemm; print('OK')"
```

### Phase 2: Verify Server Starts
```bash
docker run --rm --gpus all -p 30000:30000 shikhar481/sglang-images:d1112d85... \
    python -m sglang.launch_server --model google/gemma-2-2b --port 30000
```

### Phase 3: Run Benchmarks
```bash
python scripts/runners/local_docker_sglang_benchmark.py --commit d1112d85
```

## Critical Considerations

1. **Build Time**: Each image takes 30-60 min due to sgl-kernel CUDA compilation
2. **Disk Space**: Use `/ephemeral` for Docker builds (~20GB per image during build)
3. **Submodule Cloning**: May hit GitHub rate limits - consider shallow clone with `--depth 1`
4. **Torch Version**: Must match exactly - 2.5.1 for commits 1-4, 2.6.0 for commits 5-6
5. **GPU Required**: sgl-kernel build requires GPU access during compilation
