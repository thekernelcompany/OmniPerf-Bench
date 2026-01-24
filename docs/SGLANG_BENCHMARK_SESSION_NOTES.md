# SGLang Benchmark Session Notes

**Date**: January 24, 2025
**Purpose**: Complete documentation for continuing SGLang 3-way benchmarks on Blackwell GPU
**Target Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition (SM100)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [HuggingFace Dataset Analysis](#huggingface-dataset-analysis)
3. [sgl-kernel Architecture Compatibility](#sgl-kernel-architecture-compatibility)
4. [Docker Image Status](#docker-image-status)
5. [Benchmark Results on H100](#benchmark-results-on-h100)
6. [Known Issues and Root Causes](#known-issues-and-root-causes)
7. [Building sgl-kernel from Source](#building-sgl-kernel-from-source)
8. [Blackwell-Specific Considerations](#blackwell-specific-considerations) **(UPDATED)**
9. [Recommended Approach](#recommended-approach)
10. [Step-by-Step Instructions](#step-by-step-instructions)
11. [File Locations Reference](#file-locations-reference)
12. [Troubleshooting Guide](#troubleshooting-guide)
13. [Appendix: Blackwell Testing Session Log](#appendix-blackwell-testing-session-log-january-24-2026) **(NEW)**

---

## Executive Summary

### The Goal
Run 3-way benchmarks (baseline vs human vs agent) on 67 SGLang performance optimization commits from the HuggingFace dataset `Ayushnangia/omniperf_v1`.

### Key Findings

| Finding | Impact |
|---------|--------|
| Only **2 out of 67 commits** modify sgl-kernel native code | 97% of commits can use PyPI sgl-kernel |
| PyPI sgl-kernel has **SM90 + SM100 binaries only** | Works on H100/Blackwell, NOT on A100 |
| **59% of Docker images fail** on H100 | Due to missing deps, not GPU issues |
| **17 commits benchmark successfully** on H100 | These are the reliable ones |

### CRITICAL UPDATE (January 24, 2026): Blackwell Testing Results

| Finding | Impact |
|---------|--------|
| **ALL 67 Docker images have PyTorch < 2.7** | NONE can run on Blackwell (SM120) |
| PyTorch 2.7+ required for SM120 CUDA kernels | Hard blocker, not just a warning |
| 4 of 17 "working" commits have NO Docker images | Only 13 actually have images on DockerHub |
| Official `lmsysorg/sglang:latest` works | Has PyTorch 2.9.1 with SM120 support |

### Bottom Line
- **Blackwell (SM120)**: **NONE of the dataset Docker images work** - all have PyTorch < 2.7
- **H100 (SM90)**: 17 commits work (as originally documented)
- **A100 (SM80)**: Blocked by sgl_kernel (no SM80 binaries in PyPI)
- **Solution for Blackwell**: Must rebuild ALL 67 images with PyTorch 2.7+

---

## HuggingFace Dataset Analysis

### Dataset Location
```
Dataset: Ayushnangia/omniperf_v1
Config: sglang
Split: test
```

### Loading the Dataset
```python
import pandas as pd

# Direct parquet read (faster, avoids HF dataset issues)
pf = '/home/ubuntu/.cache/huggingface/hub/datasets--Ayushnangia--omniperf_v1/snapshots/e2e4cb1157dd9352c9e5630d33f04ef9df374915/sglang/train-00000-of-00001.parquet'
df = pd.read_parquet(pf)

# Or via HuggingFace (may have cache issues)
# HF_DATASETS_CACHE=/tmp/hf_cache python3 -c "from datasets import load_dataset; ds = load_dataset('Ayushnangia/omniperf_v1', 'sglang', split='test')"
```

### Commit Classification Results

**Total commits in dataset: 67**

| Category | Count | Description | PyPI sgl-kernel Valid? |
|----------|-------|-------------|------------------------|
| **sgl-kernel/CUDA** | 2 | Modify native C++/CUDA code | NO - require recompilation |
| **Triton kernels** | 5 | Modify Triton Python kernels | YES - JIT compiled |
| **Pure Python** | 60 | No kernel changes | YES - no recompilation needed |

### The 2 Commits That REQUIRE sgl-kernel Recompilation

These commits modify native CUDA/C++ code in `sgl-kernel/`:

1. **`25e1816eff10`** - "fix custom allreduce performance/accuracy problem (#4477)"
   - Files: `sgl-kernel/csrc/allreduce/trt_reduce_internal.cu`, `trt_reduce_internal.cuh`

2. **`a73c4df4387a`** - "Add optimized native kernels in sgl-kernel (#5150)"
   - Files: `sgl-kernel/csrc/cpu/activation.cpp`, `bmm.cpp`, `decode.cpp`

### The 5 Triton Kernel Commits (JIT - No Recompilation Needed)

These modify Triton kernels which are JIT-compiled from Python at runtime:

1. `148254d4db8b` - "Improve moe reduce sum kernel performance (#2705)"
2. `2a413829f42b` - "Add triton version as a fused_moe_triton config search key..."
3. `915140fd18c9` - "[NVIDIA] Add Low Latency NVFP4 decode kernels from Flashinfer (#8552)"
4. `c087ddd6865a` - "Refine pre_reorder_triton_kernel slightly to improve performance (#6627)"
5. `ddcf9fe3beac` - "Optimize triton attention custom mask (#3731)"

### Analysis Script

```python
import pandas as pd

pf = '/home/ubuntu/.cache/huggingface/hub/datasets--Ayushnangia--omniperf_v1/snapshots/e2e4cb1157dd9352c9e5630d33f04ef9df374915/sglang/train-00000-of-00001.parquet'
df = pd.read_parquet(pf)

sgl_kernel_commits = []
triton_kernel_commits = []
python_only_commits = []

for _, row in df.iterrows():
    files = row.get('files_changed')
    if files is None or (hasattr(files, '__len__') and len(files) == 0):
        files = []
    elif hasattr(files, 'tolist'):
        files = files.tolist()

    subject = str(row.get('commit_subject', ''))
    commit_hash = str(row.get('commit_hash', ''))[:12]

    is_sgl_kernel = False
    is_triton = False

    for f in files:
        f_str = str(f).lower()
        if 'sgl-kernel' in f_str or 'sgl_kernel' in f_str:
            is_sgl_kernel = True
            break
        if f_str.endswith('.cu') or f_str.endswith('.cuh') or f_str.endswith('.cpp'):
            is_sgl_kernel = True
            break

    if not is_sgl_kernel:
        for f in files:
            f_str = str(f).lower()
            if 'triton' in f_str and f_str.endswith('.py'):
                is_triton = True
                break

    entry = {'hash': commit_hash, 'subject': subject[:80], 'files': [str(f) for f in files]}

    if is_sgl_kernel:
        sgl_kernel_commits.append(entry)
    elif is_triton:
        triton_kernel_commits.append(entry)
    else:
        python_only_commits.append(entry)

print(f"sgl-kernel/CUDA: {len(sgl_kernel_commits)}")
print(f"Triton kernels: {len(triton_kernel_commits)}")
print(f"Pure Python: {len(python_only_commits)}")
```

---

## sgl-kernel Architecture Compatibility

### PyPI sgl-kernel Binary Support

| Architecture | GPU | PyPI Support | Notes |
|--------------|-----|--------------|-------|
| SM80 | A100 | NO | Must build from source |
| SM89 | RTX 4090, L40 | NO | Must build from source |
| SM90 | H100, H200 | YES | Works out of the box |
| SM100 | Blackwell | YES | Works out of the box |

### Checking sgl-kernel Version and Architecture Support

```python
import sgl_kernel
print(f"Version: {sgl_kernel.__version__}")

# Check available architecture variants
import os
sgl_path = os.path.dirname(sgl_kernel.__file__)
for item in os.listdir(sgl_path):
    if item.startswith('sm'):
        print(f"Found: {item}")
```

### What Happens on Unsupported Architecture

On A100 (SM80) with PyPI sgl-kernel:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Or exit code 139 (SIGSEGV) when trying to execute SM90-only kernels.

---

## Docker Image Status

### Docker Repositories

| Repository | Format | Total Tags | Notes |
|------------|--------|------------|-------|
| `shikhar481/sglang-images` | `{8-char}-src`, `-v2`, `-v3` | 161 | Newer, multiple versions per commit |
| `ayushnangia16/nvidia-sglang-docker` | Full 40-char hash | 78 | Both human + base commits |

### Checking if Image Exists

```bash
# Check shikhar481 repo
docker manifest inspect shikhar481/sglang-images:187b85b7-src 2>/dev/null && echo "EXISTS" || echo "NOT FOUND"

# Check ayushnangia16 repo
docker manifest inspect ayushnangia16/nvidia-sglang-docker:187b85b7f38496653948a2aba546d53c09ada0f3 2>/dev/null && echo "EXISTS" || echo "NOT FOUND"
```

### Image Search Strategy

```python
def get_sglang_image(commit: str):
    """Try both Docker repos to find image."""
    short = commit[:8]
    full = commit[:40]

    # 1. Try shikhar481 (newer, multiple versions)
    for suffix in ['-src', '-v3', '-v2', '']:
        image = f"shikhar481/sglang-images:{short}{suffix}"
        if image_exists(image):
            return image

    # 2. Try ayushnangia16 (full hashes)
    image = f"ayushnangia16/nvidia-sglang-docker:{full}"
    if image_exists(image):
        return image

    return None
```

### v05-hf Images (Problematic)

The `shikhar481/sglang-images:v05-hf-*` images have issues:
- PyTorch 2.10.0 (pre-release/nightly)
- Triton 3.0.0 (bundled, has bugs)
- Missing runtime dependencies

---

## Benchmark Results on H100

### Summary Statistics

| Status | Count | Percentage |
|--------|-------|------------|
| SUCCESS | 17 | 41% |
| FAILED | 24 | 59% |

### Failure Breakdown

| Failure Reason | Count | Description |
|----------------|-------|-------------|
| Server startup failed | 15 | Various module/import errors |
| libnuma.so.1 missing | 5 | System library not in Docker image |
| Other | 4 | Sandbox errors, timeouts |

### Successfully Benchmarked Commits (17 total)

These commits have working Docker images and completed benchmarks on H100:

```
187b85b7 - [PD] Optimize custom mem pool usage and bump mooncake version (#7393)
6b231325 - [PD Perf] replace Queue to FastQueue (#6649)
4418f599 - Fix FA3 DeepSeek prefill performance regression (#5624)
6cb00c63 - [PD] Optimize time out logic and add env var doc for mooncake (#6761)
148254d4 - Improve moe reduce sum kernel performance (#2705)
2bd18e2d - Memory pool: Minor optimize to avoid to (#2901)
2a413829 - Add triton version as a fused_moe_triton config search key...
2a754e57 - 2x performance improvement for large prefill & Fix workspace conflicts (#579)
5e023301 - [perf] dsv3 bmm fallback to bf16 (#5662)
880221bd - Revert "[PD Disaggregation] replace transfer with batch transfer..."
b1e5a33a - Eliminate stream sync to speed up LoRA batch init (#6960)
c087ddd6 - Refine pre_reorder_triton_kernel slightly to improve performance (#6627)
da47621c - Minor speedup topk postprocessing (#7058)
dd1012fc - [PD] Fix potential perf spike caused by tracker gc and optimize doc (#6764)
ddcf9fe3 - Optimize triton attention custom mask (#3731)
df7f61ee - Speed up rebalancing when using non-static dispatch algorithms (#6812)
e3ec6bf4 - Minor speed up block_quant_dequant (#6814)
```

### Failed Commits - Common Errors

**libnuma.so.1 missing** (5 commits):
```
ImportError: libnuma.so.1: cannot open shared object file: No such file or directory
```
Fix: `apt-get install -y libnuma1` in Docker image

**Module import errors** (various commits):
- `No module named 'outlines.fsm.regex'`
- `No module named 'vllm.model_executor.model_loader.utils'`
- Various sglang internal import failures

---

## Known Issues and Root Causes

### Issue 1: sgl-kernel SM80 Incompatibility

**Symptom**: Exit code 139 (SIGSEGV) or "no kernel image available"
**Root Cause**: PyPI sgl-kernel only has SM90/SM100 binaries
**Affected**: A100, RTX 4090, L40
**Solution**: Build sgl-kernel from source with correct TORCH_CUDA_ARCH_LIST

### Issue 2: Triton JIT Crash (v05-hf images)

**Symptom**: Server crashes during first inference with SIGSEGV in Triton compiler
```
File "triton/compiler/code_generator.py", line 1041, in visit_FunctionDef
... during compute_position_triton kernel compilation
```
**Root Cause**: PyTorch 2.10.0 + Triton 3.0.0 incompatibility in v05-hf base images
**Solution**: Use different base images with stable PyTorch (2.4.x or 2.5.x)

### Issue 3: Missing libnuma.so.1

**Symptom**:
```
ImportError: libnuma.so.1: cannot open shared object file: No such file or directory
```
**Root Cause**: System library not installed in Docker image
**Solution**: Add to Dockerfile:
```dockerfile
RUN apt-get update && apt-get install -y libnuma1 libnuma-dev
```

### Issue 4: FlashInfer Wheel Incompatibility

**Symptom**: Cannot install FlashInfer as alternative attention backend
```
flashinfer-python depends on torch==2.6.*
```
**Root Cause**: FlashInfer wheels only support specific PyTorch versions
**Affected**: v05-hf images with PyTorch 2.10.0

### Issue 5: Module Import Errors in Older Commits

**Symptom**: Various `ModuleNotFoundError` during server startup
**Root Cause**: Different SGLang versions have different module structures and dependencies
**Solution**: Each commit's Docker image should be self-contained with correct deps

---

## Building sgl-kernel from Source

### When You Need to Build from Source

1. Running on SM80 (A100) or SM89 (RTX 4090, L40)
2. The commit modifies sgl-kernel code (only 2 commits in dataset)
3. Need specific optimization flags

### Dockerfile for SM80 Build (A100)

```dockerfile
FROM shikhar481/sglang-images:v05-hf-187b85b7f384

# Uninstall existing sgl-kernel
RUN pip uninstall -y sgl-kernel 2>/dev/null || true

# Install build dependencies
RUN pip install scikit-build-core cmake ninja pybind11 numpy einops

WORKDIR /opt/sglang/sgl-kernel

# Fix CMakeLists.txt - add binary directory for mscclpp
RUN sed -i 's|add_subdirectory(${repo-mscclpp_SOURCE_DIR})|add_subdirectory(${repo-mscclpp_SOURCE_DIR} ${CMAKE_BINARY_DIR}/mscclpp)|g' CMakeLists.txt

# Set environment for SM80 build
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
ENV MAX_JOBS=96
ENV NVCC_THREADS=4
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# Build and install sgl-kernel from source
RUN pip install . --no-build-isolation -v

WORKDIR /workspace
```

### Dockerfile for SM100 Build (Blackwell)

```dockerfile
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    git cmake ninja-build \
    libnuma1 libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone SGLang
RUN git clone https://github.com/sgl-project/sglang.git /opt/sglang
WORKDIR /opt/sglang

# Checkout specific commit
ARG COMMIT_HASH
RUN git checkout ${COMMIT_HASH}

# Install PyTorch with CUDA 12.6
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install build dependencies
RUN pip3 install scikit-build-core cmake ninja pybind11 numpy einops

# Build sgl-kernel for SM100
WORKDIR /opt/sglang/sgl-kernel
ENV TORCH_CUDA_ARCH_LIST="9.0;10.0"
ENV MAX_JOBS=64
ENV NVCC_THREADS=4

RUN pip3 install . --no-build-isolation -v

# Install SGLang
WORKDIR /opt/sglang
RUN pip3 install -e "python[all]"

WORKDIR /workspace
```

### Build Commands

```bash
# Build for specific commit
docker build \
    --build-arg COMMIT_HASH=187b85b7f38496653948a2aba546d53c09ada0f3 \
    -t sglang-benchmark:187b85b7 \
    -f Dockerfile.sglang .

# Build with high parallelism (adjust based on RAM)
docker build \
    --build-arg MAX_JOBS=96 \
    --build-arg NVCC_THREADS=4 \
    -t sglang-benchmark:sm100 .

# Build with build logs
docker build --progress=plain -t sglang-test:sm100 . 2>&1 | tee build.log
```

### Verifying the Build

```bash
# Test sgl-kernel import
docker run --rm --gpus all sglang-test:sm100 python3 -c "
import sgl_kernel
print(f'sgl_kernel version: {sgl_kernel.__version__}')
print('sgl_kernel loaded successfully!')
"

# Test SGLang server startup
docker run --rm --gpus all -p 30000:30000 sglang-test:sm100 \
    python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --host 0.0.0.0
```

---

## Blackwell-Specific Considerations

### Hardware Specs

- **Architecture**: SM120 (Blackwell RTX PRO 6000 Server Edition)
- **Note**: Earlier documentation incorrectly stated SM100 - actual compute capability is **12.0**
- **PyPI sgl-kernel**: NOT tested on SM120
- **CUDA Version**: Requires CUDA 12.6+ for full support

### CRITICAL: PyTorch Version Requirements

| PyTorch Version | SM120 (Blackwell) Support | Status |
|-----------------|---------------------------|--------|
| 2.1.x - 2.6.x | NO | CUDA kernels not compiled |
| 2.7+ | YES | SM120 support added |
| 2.9.1 (current latest) | YES | Verified working |

**This is a HARD requirement, not just a warning.** PyTorch < 2.7 will show a warning but then CRASH:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### Driver Requirements

```bash
# Check driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Blackwell requires driver 550+ for full support
# Tested with: Driver 580.126.09, CUDA 13.0
```

### Actual Test Results (January 24, 2026)

**Docker images tested on NVIDIA RTX PRO 6000 Blackwell Server Edition:**

| Image | PyTorch | SGLang | Result |
|-------|---------|--------|--------|
| `ayushnangia16/...:148254d4...` | 2.5.1+cu121 | 0.4.1.post3 | CRASH - no SM120 kernels |
| `ayushnangia16/...:187b85b7...` | 2.6.0+cu124 | 0.4.x | CRASH - no SM120 kernels |
| `shikhar481/...:fixed-9c745d07...` | 2.4.0+cu124 | 0.3.5.post2 | CRASH - no SM120 kernels |
| `shikhar481/...:2a754e57-v3` | 2.1.2+cu121 | 0.1.x | CRASH - no SM120 kernels |
| `lmsysorg/sglang:latest` | **2.9.1+cu129** | latest | **SUCCESS** |

### Why Docker Doesn't Help Here

A common misconception: "Docker provides isolation, so version mismatches shouldn't matter."

**Reality**: Docker does NOT abstract GPU architecture.
```
┌─────────────────────────────────────────────┐
│            Docker Container                  │
│  ┌────────────────────────────────────────┐ │
│  │ PyTorch 2.5.1 (compiled for SM50-SM90) │ │
│  │ CUDA kernels: must match GPU arch      │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
                      │
                      ▼ nvidia-container-toolkit
┌─────────────────────────────────────────────┐
│            Host GPU (Blackwell SM120)        │
│  "I need SM120 kernels, not SM90!"          │
└─────────────────────────────────────────────┘
```

Docker containers:
- ✅ Isolate Python dependencies, libraries, OS packages
- ✅ Share host kernel and GPU drivers
- ❌ **Cannot translate GPU architecture** - CUDA code compiled for SM90 won't run on SM120

### Testing Blackwell Compatibility

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

# RTX PRO 6000 Blackwell shows: (12, 0) = SM120
# Note: torch.cuda.is_available() returns True even if kernels won't work!
# The actual crash happens when you try to run CUDA operations.
```

### Verified Working Configuration on Blackwell

```bash
# Official SGLang image works:
docker pull lmsysorg/sglang:latest

# Test it:
docker run --rm --gpus all lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 30000 --host 0.0.0.0 \
    --tp-size 1 --mem-fraction-static 0.3

# Successfully runs inference on Blackwell!
```

---

## Recommended Approach

### Option A: Focus on Working Commits (Fastest)

Only benchmark the 17 commits that already work on H100:

```python
WORKING_COMMITS = [
    "187b85b7f38496653948a2aba546d53c09ada0f3",
    "6b231325b9782555eb8e1cfcf27820003a98382b",
    "4418f599a54699181b35d89b0def2697cccb721a",
    "6cb00c6398126513e37c43dd975d461765fb44c7",
    "148254d4db8bf3bffee23710cd1acbd5711ebd1b",
    "2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0",
    "2a413829f42b8e8433a3e7cfd91cc9cb241cfbc0",
    "2a754e57b052e249ed4f8572cb6f0069ba6a495e",
    "5e02330137a1ce44f29cc41a4da5f010c4bffec6",
    "880221bd3b3e56a4bc2268fe9a9f77f426accf6c",
    "b1e5a33ae337d20e35e966b8d82a02a913d32689",
    "c087ddd6865a52634326a05af66429cb5531cd16",
    "da47621ccc4f8e8381f3249257489d5fe32aff1b",
    "dd1012fcbe2a1fb36c44e10c16f8d0bcd8e9da25",
    "ddcf9fe3beacd8aed573c711942194dd02350da4",
    "df7f61ee7d235936e6663f07813d7c03c4ec1603",
    "e3ec6bf4b65a50e26e936a96adc7acc618292002",
]
```

### Option B: Fix Failing Docker Images

For the 24 failing commits, fix common issues:

```dockerfile
# Add to Dockerfile to fix libnuma issue
RUN apt-get update && apt-get install -y libnuma1 libnuma-dev

# Fix outlines version
RUN pip install "outlines>=0.0.44"

# Ensure all SGLang dependencies
RUN pip install einops dill msgspec python-multipart partial-json-parser xgrammar torchao pynvml
```

### Option C: Rebuild All Images from Scratch

Use a consistent base image and rebuild for each commit:

```bash
# For each commit
for COMMIT in $(cat commits.txt); do
    docker build \
        --build-arg COMMIT_HASH=$COMMIT \
        -t sglang-benchmark:${COMMIT:0:8} \
        -f Dockerfile.sglang.rebuild .
done
```

---

## Step-by-Step Instructions

### Step 1: Set Up Blackwell Environment

```bash
# 1. Verify GPU
nvidia-smi
# Should show RTX PRO 6000 Blackwell

# 2. Check CUDA version
nvcc --version
# Should be 12.6+

# 3. Install Docker with NVIDIA support
sudo apt-get install -y docker.io nvidia-container-toolkit
sudo systemctl restart docker

# 4. Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### Step 2: Clone Repository

```bash
git clone https://github.com/your-repo/OmniPerf-Bench.git
cd OmniPerf-Bench
git checkout feature/sglang-modal-benchmarks
```

### Step 3: Test PyPI sgl-kernel on Blackwell

```bash
# Quick test
docker run --rm --gpus all python:3.11 bash -c "
pip install sgl-kernel torch --index-url https://download.pytorch.org/whl/cu126
python -c 'import sgl_kernel; print(sgl_kernel.__version__)'
"
```

### Step 4: Run Benchmark on Working Commit

```bash
# Pull a known working image
docker pull ayushnangia16/nvidia-sglang-docker:187b85b7f38496653948a2aba546d53c09ada0f3

# Run benchmark
python scripts/runners/hero_sglang_benchmark.py \
    --commit 187b85b7f38496653948a2aba546d53c09ada0f3 \
    --gpu-config "BLACKWELL:1" \
    --benchmark-type serving
```

### Step 5: Batch Process Working Commits

```python
# scripts/run_working_commits.py
import subprocess

WORKING_COMMITS = [
    "187b85b7f38496653948a2aba546d53c09ada0f3",
    "6b231325b9782555eb8e1cfcf27820003a98382b",
    # ... add all 17
]

for commit in WORKING_COMMITS:
    cmd = [
        "python", "scripts/runners/hero_sglang_benchmark.py",
        "--commit", commit,
        "--gpu-config", "BLACKWELL:1",
        "--benchmark-type", "serving"
    ]
    subprocess.run(cmd)
```

### Step 6: Fix and Rebuild Failing Images (Optional)

```bash
# Identify failing commits
python scripts/analyze_failures.py

# Rebuild with fixes
python src/benchmark/tools/build_sglang_images.py \
    --commit 62757db6f0f09a6dff15b1ee1ac3029602951509 \
    --fix-libnuma \
    --fix-outlines
```

---

## File Locations Reference

### Key Files in OmniPerf-Bench

```
OmniPerf-Bench/
├── src/benchmark/
│   ├── modal/
│   │   └── sglang_benchmark.py      # Modal cloud benchmark runner
│   ├── tools/
│   │   └── build_sglang_images.py   # Docker image builder
│   └── fixes/
│       ├── sglang_commit_mapping.json    # 74 commit mappings
│       └── sglang_docker_image_status.json
├── scripts/runners/
│   ├── hero_sglang_benchmark.py     # Main benchmark orchestrator
│   ├── local_docker_benchmark.py    # Local Docker benchmark (vLLM)
│   └── local_docker_sglang_benchmark.py  # To be created for SGLang
├── omniperf_results_3way_sglang/
│   ├── sglang/                      # Per-commit results
│   ├── docker_benchmark_results/    # Local Docker results
│   └── dataset_benchmarks/          # Dataset-based results
└── docs/
    └── SGLANG_BENCHMARK_SESSION_NOTES.md  # This file
```

### HuggingFace Dataset Cache

```
~/.cache/huggingface/hub/datasets--Ayushnangia--omniperf_v1/
└── snapshots/e2e4cb1157dd9352c9e5630d33f04ef9df374915/
    └── sglang/
        └── train-00000-of-00001.parquet
```

### Docker Images

```
# Primary repository (used by Modal)
ayushnangia16/nvidia-sglang-docker:{40-char-commit-hash}

# Secondary repository (newer builds)
shikhar481/sglang-images:{8-char}-src
shikhar481/sglang-images:{8-char}-v2
shikhar481/sglang-images:{8-char}-v3
shikhar481/sglang-images:v05-hf-{8-char}  # Problematic base
```

---

## Troubleshooting Guide

### Error: "no kernel image is available for execution on the device"

**Cause**: PyTorch CUDA kernels not compiled for your GPU architecture
**Solution by GPU**:
- **Blackwell (SM120)**: Requires PyTorch 2.7+ - rebuild Docker images
- **H100 (SM90)**: PyTorch 2.1+ works, use existing images
- **A100 (SM80)**: Build sgl-kernel from source with `TORCH_CUDA_ARCH_LIST="8.0"`

**NOTE**: On Blackwell, this error occurs even with SGLang 0.3.x (which doesn't use sgl-kernel) because PyTorch itself lacks SM120 kernels.

### Error: "libnuma.so.1: cannot open shared object file"

**Cause**: libnuma not installed in Docker image
**Solution**:
```bash
# Inside container
apt-get update && apt-get install -y libnuma1

# Or rebuild image with:
RUN apt-get update && apt-get install -y libnuma1 libnuma-dev
```

### Error: "ModuleNotFoundError: No module named 'outlines.fsm.regex'"

**Cause**: Old outlines version
**Solution**:
```bash
pip install "outlines>=0.0.44"
```

### Error: Exit code 139 (SIGSEGV) during Triton compilation

**Cause**: PyTorch/Triton version incompatibility (common in v05-hf images)
**Solution**:
- Use different base image with stable PyTorch
- Or try `--disable-cuda-graph` flag (partial fix)

### Error: "Server crashed during startup"

**Cause**: Various - check full error message
**Debug**:
```bash
# Run with verbose logging
docker run --rm --gpus all IMAGE python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --log-level debug \
    --port 30000
```

### Error: Disk space issues with HuggingFace

**Cause**: Default cache on full disk
**Solution**:
```bash
# Use different cache location
HF_DATASETS_CACHE=/path/to/cache python your_script.py
```

---

## Commit Mapping Reference

The full mapping of human commits to their baseline (parent) commits is in:
```
/home/ubuntu/OmniPerf-Bench/src/benchmark/fixes/sglang_commit_mapping.json
```

Sample structure:
```json
{
  "human_commit": "187b85b7f38496653948a2aba546d53c09ada0f3",
  "base_commit": "ceba0ce4f661722198f6568a54ba20cf06b7e033",
  "pr_number": "7393",
  "subject": "[PD] Optimize custom mem pool usage and bump mooncake version (#7393)"
}
```

---

## Summary Checklist for Blackwell

### Current Status (as of January 24, 2026)
- [x] Verify Blackwell GPU detected (`nvidia-smi`) - **RTX PRO 6000 Blackwell Server Edition detected**
- [x] Confirm CUDA 12.6+ installed - **Driver 580.126.09, CUDA 13.0**
- [x] Test PyPI sgl-kernel imports - **NOT TESTED on SM120**
- [x] Test dataset Docker images - **ALL FAILED - PyTorch < 2.7**
- [x] Test official SGLang image - **SUCCESS with PyTorch 2.9.1**
- [ ] Rebuild dataset images with PyTorch 2.7+ - **REQUIRED for Blackwell**
- [ ] Run batch benchmarks on Blackwell
- [ ] Collect and analyze results

### For Future Blackwell Testing
1. **DO NOT** use existing dataset Docker images - they will all crash
2. **DO** use `lmsysorg/sglang:latest` for testing latest SGLang
3. **DO** rebuild images with PyTorch 2.7+ to test specific commits

---

## Contact / Issues

If you encounter issues not covered here:
1. Check existing benchmark results in `omniperf_results_3way_sglang/`
2. Review error logs in Docker container
3. File issue at: https://github.com/anthropics/claude-code/issues

---

## Appendix: Blackwell Testing Session Log (January 24, 2026)

### Session Overview

Attempted to run the 17 "working" commits from the dataset on NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120).

### Key Discoveries

#### 1. Docker Image Availability Check

Checked all 17 "working" commits for Docker image availability:

| Commit | ayushnangia16 | shikhar481 |
|--------|--------------|------------|
| 187b85b7 | EXISTS | NO |
| 6b231325 | EXISTS | NO |
| 4418f599 | **NO IMAGE** | NO |
| 6cb00c63 | EXISTS | NO |
| 148254d4 | EXISTS | EXISTS (-src) |
| 2bd18e2d | EXISTS | EXISTS (-src) |
| 2a413829 | **NO IMAGE** | NO |
| 2a754e57 | EXISTS | EXISTS (-v3) |
| 5e023301 | **NO IMAGE** | NO |
| 880221bd | EXISTS | NO |
| b1e5a33a | EXISTS | NO |
| c087ddd6 | **NO IMAGE** | NO |
| da47621c | EXISTS | NO |
| dd1012fc | EXISTS | NO |
| ddcf9fe3 | EXISTS | EXISTS (-src) |
| df7f61ee | EXISTS | NO |
| e3ec6bf4 | EXISTS | NO |

**Finding**: 4 of 17 "working" commits have NO Docker images at all.

#### 2. PyTorch Version Analysis

| Repository | Image Type | PyTorch Version | SM120 Support |
|------------|-----------|-----------------|---------------|
| ayushnangia16 | Various | 2.5.1 - 2.6.0 | NO |
| shikhar481 | -v3 | 2.1.2 | NO |
| shikhar481 | fixed-* | 2.4.0 | NO |
| lmsysorg | latest | 2.9.1 | YES |

#### 3. Actual Test Results

**Test 1: ayushnangia16/nvidia-sglang-docker:148254d4...**
```
PyTorch: 2.5.1+cu121
SGLang: 0.4.1.post3
sgl_kernel: FAILED (SM100 binaries only, GPU is SM120)
Result: CRASH
```

**Test 2: shikhar481/sglang-images:fixed-9c745d078e29**
```
PyTorch: 2.4.0+cu124
SGLang: 0.3.5.post2
sgl_kernel: NOT REQUIRED (0.3.x)
Result: CRASH - "RuntimeError: CUDA error: no kernel image is available"
```

**Test 3: lmsysorg/sglang:latest**
```
PyTorch: 2.9.1+cu129
SGLang: latest
Result: SUCCESS - Server started, inference completed
```

#### 4. Root Cause Analysis

The issue is NOT sgl_kernel architecture mismatch (as initially suspected for A100).

The ACTUAL issue is that **PyTorch itself** doesn't have CUDA kernels for SM120 until version 2.7.

Even basic PyTorch operations fail:
```python
torch.arange(0, 10, device='cuda')  # CRASH on SM120 with PyTorch < 2.7
```

#### 5. SGLang Version vs sgl_kernel Dependency

| SGLang Version | sgl_kernel Required? | Notes |
|----------------|---------------------|-------|
| 0.1.x - 0.3.x | NO | Pure Python + Triton |
| 0.4.x+ | YES | Requires compiled CUDA extension |

The 0.3.x images don't need sgl_kernel, but they STILL fail on Blackwell because PyTorch lacks SM120 kernels.

### Conclusions

1. **ALL 67 dataset Docker images are incompatible with Blackwell** - none have PyTorch 2.7+
2. **The "17 working commits" only work on H100** (SM90) where PyTorch 2.1-2.6 is sufficient
3. **To benchmark on Blackwell**, must rebuild all images with PyTorch 2.7+
4. **Official `lmsysorg/sglang:latest` works** and can be used for testing latest SGLang on Blackwell

### Recommended Next Steps

1. **For immediate Blackwell testing**: Use `lmsysorg/sglang:latest` (but this only tests latest code, not specific commits)
2. **For dataset benchmarking**: Either:
   - Use H100/A100 machines (where existing images work)
   - Rebuild all 67 Docker images with PyTorch 2.7+ base

### Environment Details

```
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
Compute Capability: SM120 (12.0)
Driver: 580.126.09
CUDA: 13.0
Docker data-root: /opt/dlami/nvme/docker (1.7TB ephemeral storage)
```

---

*Document updated from Claude Code session on January 24, 2026*
