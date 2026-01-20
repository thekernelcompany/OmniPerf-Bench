# OmniPerf Dataset Coverage Analysis

**Date:** 2026-01-20
**Dataset:** https://huggingface.co/datasets/Ayushnangia/omniperf_v1

---

## Executive Summary

This document analyzes the coverage of the OmniPerf HuggingFace dataset with working Docker images ready for benchmarking.

| Metric | Count | Percentage |
|--------|-------|------------|
| Total OmniPerf Commits | 169 | 100% |
| SGLang Commits | 74 | 44% |
| vLLM Commits | 95 | 56% |
| **SGLang Images Ready** | **11** | **15% of SGLang** |
| SGLang Pending GPU Test | 3 | 4% of SGLang |
| SGLang Buildable (v0.4.x) | 56 | 76% of SGLang |
| SGLang Unfixable (v0.1.x) | 7 | 9% of SGLang |

---

## OmniPerf Dataset Overview

The OmniPerf dataset contains 169 performance benchmark commits:
- **SGLang:** 74 commits (target of Docker image builds)
- **vLLM:** 95 commits (not targeted in this project)

All benchmark results are stored at:
```
/home/ubuntu/sglang-images/OmniPerf-Bench/archive/results/omniperf_results_v2/sglang/
```

---

## Working Docker Images (GPU-Confirmed)

### Phase 6 `-src` Builds (Built from Source)

| Commit | Version | PR | Subject | Status |
|--------|---------|-----|---------|--------|
| `62757db6-src` | v0.2.11 | #1010 | Cache disabled overhead | **WORKING** |
| `ab4a83b2-src` | v0.3.0 | #1339 | Optimize schedule | **WORKING** |
| `c98e84c2-src` | v0.3.2 | #1589 | torch.argmax optimization | **WORKING** |
| `2854a5ea-src` | v0.3.1.post3 | #1496 | bench_latency fix | **WORKING** |
| `9c064bf7-src` | v0.3.2 | #1587 | LoRA Step 1 | **WORKING** |
| `b77a02cd-src` | v0.3.4.post2 | #1752 | Grammar backends | **WORKING** |

### v3 Rebuilds (Working)

| Commit | Version | PR | Status |
|--------|---------|-----|--------|
| `9c745d07-v3` | v0.3.5.post2 | #2056 | **WORKING** |
| `10189d08-v3` | v0.3.6 | #2171 | **WORKING** |

### Original Images

| Commit | PR | Subject | Parent | Status |
|--------|-----|---------|--------|--------|
| `021f76e4` | #6994 | LoRA Manager stream sync | `777688b8` | **WORKING** |
| `777688b8` | - | (Baseline) | - | **WORKING** |
| `c087ddd6` | #6627 | Triton kernel optimization | `f4a8987f` | **WORKING** |

---

## Pending GPU Testing (3 Images)

| Commit | Version | PR | Subject | Status |
|--------|---------|-----|---------|--------|
| `e5db40dc-src` | v0.3.3.post1 | #1694 | ORJson serialization | PENDING |
| `b1709305-src` | v0.3.3.post1 | #1697 | Radix tree optimization | PENDING |
| `8f8f96a6-src` | v0.3.4.post1 | #1773 | stop_token_ids fix | PENDING |

---

## Complete 74 SGLang Commits Breakdown

### By Version

| Version Range | Count | % | Status | Notes |
|---------------|-------|---|--------|-------|
| **v0.4.x** | 56 | 76% | Buildable | Requires sgl-kernel from source |
| **v0.3.x** | 10 | 14% | Working | 8 GPU-confirmed, 2 pending |
| **v0.2.x** | 1 | 1% | Working | GPU-confirmed |
| **v0.1.x** | 7 | 9% | Unfixable | API incompatibilities |

### Detailed v0.4.x Problem Analysis

The 56 v0.4.x commits (76% of dataset) previously failed because:

1. **sgl-kernel ABI mismatch**: Pre-built wheels incompatible with container PyTorch
2. **Missing dependencies**: uvloop, libnuma, etc.
3. **Dockerfiles didn't build sgl-kernel from source**

**Solution:** Build sgl-kernel from source using CUDA stubs.

The key insight from the official sgl-kernel Dockerfile (line 76):
```dockerfile
ln -sf /usr/local/cuda-${CUDA_VERSION}/targets/${_LIB}-linux/lib/stubs/libcuda.so \
       /usr/lib/${ARCH}-linux-gnu/libcuda.so
```

This creates a CUDA stub that allows NVCC compilation without a GPU.

---

## OmniPerf Benchmark Results Summary

From `/archive/results/omniperf_results_v2/omniperf_report_sglang.json`:

| Category | Count |
|----------|-------|
| Success | 28 |
| Baseline Failed | 46 |
| Total | 74 |

### Failure Analysis

| Error Type | Count | Root Cause |
|------------|-------|------------|
| huggingface-hub version | 20 | Dependency version mismatch |
| sgl_kernel missing | 8 | v0.4.x without proper kernel build |
| outlines.fsm.regex | 3 | Outlines package API change |
| Baseline JSON output | 10 | Benchmark script incompatibility |
| Other | 5 | Various import/module errors |

---

## Docker Images Available

### Current Images (from `docker images`)

```
sglang:62757db6-src   (v0.2.11)   19.1GB
sglang:ab4a83b2-src   (v0.3.0)    16GB
sglang:2854a5ea-src   (v0.3.1)    16GB
sglang:c98e84c2-src   (v0.3.2)    16GB
sglang:9c064bf7-src   (v0.3.2)    16GB
sglang:e5db40dc-src   (v0.3.3)    16GB
sglang:b1709305-src   (v0.3.3)    16GB
sglang:b77a02cd-src   (v0.3.4)    16GB
sglang:8f8f96a6-src   (v0.3.4)    16GB
```

Also pushed to DockerHub: `shikhar481/sglang-images:<tag>`

---

## Priority v0.4.x Commits for Building

These are 3-way candidates from OmniPerf dataset that need v0.4.x Docker builds:

| Commit | Version | PR | Subject | Claimed Speedup |
|--------|---------|-----|---------|-----------------|
| `79961afa` | v0.4.6.post2 | #6077 | FA3 metadata init | 21% (530us→418us) |
| `93470a14` | v0.4.5 | #5090 | Refactor FA3 code | - |
| `3212c2ad` | v0.4.9.post4 | #6003 | VLM tensor transport | 16% |
| `2bd18e2d` | v0.4.1.post6 | #2901 | Memory pool optimization | - |
| `d1112d85` | v0.4.4.post1 | #2797 | input_embeds endpoint | - |

---

## Dockerfiles Reference

### Available Dockerfiles

```
dockerfiles/
├── Dockerfile.v0.2.11              # 62757db6
├── Dockerfile.v0.3.0               # ab4a83b2
├── Dockerfile.v0.3.1.post3         # 2854a5ea
├── Dockerfile.v0.3.2               # c98e84c2
├── Dockerfile.v0.3.2.9c064bf7      # 9c064bf7
├── Dockerfile.v0.3.3.post1.e5db40dc # e5db40dc
├── Dockerfile.v0.3.3.post1.b1709305 # b1709305
├── Dockerfile.v0.3.4.post1         # 8f8f96a6
├── Dockerfile.v0.3.4.post2         # b77a02cd
├── Dockerfile.v0.4.x.template      # Template for v0.4.x builds
└── README.md
```

### Dependency Matrix

| Version | Commit | vLLM | Torch | FlashInfer |
|---------|--------|------|-------|------------|
| v0.2.11 | 62757db6 | 0.5.4 | 2.3.0 | 0.1.6 |
| v0.3.0 | ab4a83b2 | 0.5.5 | 2.4.0 | 0.1.6 |
| v0.3.1 | 2854a5ea | 0.5.5 | 2.4.0 | 0.1.6 |
| v0.3.2 | c98e84c2 | 0.5.5 | 2.4.0 | 0.1.6 |
| v0.3.3 | e5db40dc | 0.5.5 | 2.4.0 | 0.1.6 |
| v0.3.4 | b77a02cd | 0.5.5 | 2.4.0 | 0.1.6 |
| **v0.4.x** | various | N/A | 2.5.0+ | 0.2.1+ |

---

## Building v0.4.x Images (New Approach)

### Key Insight

**sgl-kernel CAN be built on CPU-only machines using CUDA stubs!**

The `nvidia/cuda:12.4.1-devel-ubuntu22.04` image includes:
- NVCC compiler
- CUDA headers
- libcuda.so stubs at `/usr/local/cuda/targets/x86_64-linux/lib/stubs/`

### Template Usage

```bash
# Copy template
cp dockerfiles/Dockerfile.v0.4.x.template dockerfiles/Dockerfile.v0.4.6.79961afa

# Edit placeholders:
# - COMMIT_SHORT=79961afa
# - COMMIT_FULL=79961afa8281f98f380d11db45c8d4b6e66a574f
# - VERSION=v0.4.6.post2

# Build (takes ~30-60 minutes due to sgl-kernel compilation)
docker build -f dockerfiles/Dockerfile.v0.4.6.79961afa -t sglang:79961afa-src .

# Test
docker run --rm --gpus all sglang:79961afa-src \
    python3 -c "import sglang; print(sglang.__version__)"
```

---

## Current State Summary (Updated 2026-01-20)

```
OmniPerf SGLang Dataset:           74 commits
├── v0.4.x:                        56 (76%)
│   └── BUILT & WORKING:            2 (2bd18e2d, d1112d85)
│   └── BLOCKED (torch version):    2 (79961afa, 3212c2ad)
│   └── BLOCKED (FetchContent):     1 (93470a14)
│   └── NOT YET ATTEMPTED:         51
├── v0.3.x (WORKING):              10 (14%)
│   └── GPU confirmed:              8
│   └── Pending test:               2
├── v0.2.x (WORKING):               1 (1%)
│   └── GPU confirmed:              1
└── v0.1.x (UNFIXABLE):             7 (9%)

READY FOR BENCHMARKING:    13 commits (18%)
  - v0.4.x: 2bd18e2d, d1112d85
  - v0.3.x: 62757db6, ab4a83b2, c98e84c2, 2854a5ea, 9c064bf7, b77a02cd, 9c745d07, 10189d08
  - v0.3.x pending test: e5db40dc, b1709305, 8f8f96a6
PERMANENTLY BROKEN:         7 commits (9%)
```

---

## Next Steps

### Phase 1: Test Pending Images
Test remaining 3 images on GPU: `e5db40dc-src`, `b1709305-src`, `8f8f96a6-src`

### Phase 2: Build Priority v0.4.x Images
1. Create Dockerfiles for 5 priority commits
2. Build using CUDA stubs approach
3. Push to DockerHub
4. Test on H100/L40S

### Phase 3: Comprehensive v0.4.x Build
1. Generate Dockerfiles for all 56 v0.4.x commits
2. Batch build (may take several hours per image)
3. Validate with import tests
4. GPU validation on H100

---

## Files Reference

| File | Description |
|------|-------------|
| `archive/results/omniperf_results_v2/sglang/` | Benchmark results per commit |
| `data/mappings/omniperf_dataset.json` | Full dataset with commit metadata |
| `dockerfiles/` | All Docker build files |
| `dockerfiles/README.md` | Dockerfile documentation |
| `docs/OMNIPERF_COVERAGE_ANALYSIS.md` | This document |
