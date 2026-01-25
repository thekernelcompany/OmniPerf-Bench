# SGLang H100 Docker Benchmark - Implementation Status

## Summary
Implementation of the SGLang Docker benchmark plan has been completed with Docker and NVIDIA Container Toolkit installed. However, significant ABI compatibility issues prevent successful benchmark execution with the existing Docker images.

## Environment Setup (COMPLETED)

### Docker Installation
- Docker 28.2.2 installed
- NVIDIA Container Toolkit 1.18.2 installed
- GPU access verified: H100 PCIe (SM90)

### CUDA/Driver Versions
- Host Driver: 570.148.08
- Host CUDA: 12.8
- Container CUDA: 12.1.1 - 12.6.1 (varies by image)
- Container PyTorch: 2.5.1 - 2.6.0 (varies by image)

## Docker Images Downloaded (5 of 14)

| # | Commit | Repository | Size | Status |
|---|--------|------------|------|--------|
| 1 | 187b85b7 | ayushnangia16 | 20.8GB | sgl_kernel ABI error |
| 2 | 6b231325 | ayushnangia16 | 15GB | Downloaded |
| 3 | 6cb00c63 | ayushnangia16 | 15GB | Downloaded |
| 4 | 148254d4 | ayushnangia16 | 21.3GB | Transformers error |
| 5 | 148254d4-src | shikhar481 | 21.1GB | sgl_kernel version mismatch |

### Remaining images not downloaded (disk space constraint)
- 2bd18e2d, 2a754e57, 880221bd, b1e5a33a
- c087ddd6, da47621c, dd1012fc, ddcf9fe3
- df7f61ee, e3ec6bf4

## Compatibility Issues Identified

### 1. sgl_kernel ABI Mismatch
**Images affected:** 187b85b7, newer images with PyTorch 2.6.0
**Error:** 
```
ImportError: ...common_ops.abi3.so: undefined symbol: _ZN3c108ListType3get...
```
**Cause:** sgl_kernel compiled against different PyTorch C++ ABI
**Status:** Partial mitigation via `pip install --upgrade sgl_kernel` in benchmark script

### 2. libnuma.so Missing
**Images affected:** 148254d4
**Error:**
```
ImportError: libnuma.so.1: cannot open shared object file
```
**Fix:** Added `apt-get install -y libnuma-dev` to benchmark script
**Status:** FIXED in benchmark script

### 3. Transformers Version
**Images affected:** Older images with SGLang < 0.4
**Error:**
```
ImportError: cannot import name 'AutoProcessor' from 'transformers'
```
**Cause:** Older transformers version missing required classes

### 4. vLLM API Changes
**Images affected:** Very old images (09deb20d with SGLang 0.1.14)
**Error:**
```
ModuleNotFoundError: No module named 'vllm.model_executor.model_loader.utils'
```
**Cause:** vLLM internal API changed between versions

## Code Updates Completed

### local_docker_sglang_benchmark.py
1. Updated paths from `/home/ubuntu/` to `/root/`
2. Added multi-agent support (claude_code, codex)
3. Added libnuma installation in Docker commands
4. Added sgl_kernel upgrade attempt for ABI issues
5. Added HF_CACHE_PATH configuration

### run_all_sglang_3way_benchmarks.py (NEW)
Batch runner script for all 14 commits with:
- Progress tracking
- Result checking
- Skip existing results option
- Parallel execution support

## Disk Space Status
- Total: 993GB
- Used: 692GB (includes ~87GB Docker images)
- Available: 301GB

## Recommendations

### Option 1: Rebuild Images (Recommended)
Build fresh Docker images with:
- PyTorch 2.6.0+cu124 base
- Latest sgl_kernel
- libnuma-dev pre-installed
- Compatible transformers version

### Option 2: Source Installation
Skip Docker, install SGLang from source on host:
```bash
pip install sglang[all]
```
Then run benchmarks directly.

### Option 3: Modal-based Execution
Use Modal platform which may have better managed environments.

## Files Created/Modified

```
scripts/runners/
├── local_docker_sglang_benchmark.py (MODIFIED)
└── run_all_sglang_3way_benchmarks.py (NEW)

docs/
├── sglang_h100_benchmark_status.md (NEW)

omniperf_results_3way_sglang/
├── docker_benchmark_results/
│   ├── 09deb20d_human_serving.json
│   ├── 187b85b7_human_serving.json
│   └── 148254d4_human_serving.json
```

## Next Steps
1. Investigate building fresh compatible Docker images
2. Or proceed with source-based installation approach
3. Run benchmarks once compatibility is resolved

