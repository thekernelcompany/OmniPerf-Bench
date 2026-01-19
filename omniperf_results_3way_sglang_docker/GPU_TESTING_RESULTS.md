# SGLang Docker Image Testing Results

**Date:** 2026-01-19
**GPU:** NVIDIA L40S (46GB VRAM)
**Host Driver:** 535.274.02
**Host CUDA:** 12.2
**Docker Version:** 29.1.5

---

## Executive Summary

**Total Testing: 26 images tested (8 original + 18 rebuilt), only 3 are functional:**

| Phase | Images | Working | Broken |
|-------|--------|---------|--------|
| Phase 1 (Original) | 8 | 3 | 5 |
| Phase 2 (Rebuilt) | 18 | 0 | 18 |
| **Total** | **26** | **3 (11.5%)** | **23 (88.5%)** |

**The documentation claims 30 images are ready - this is incorrect. Only 3 images work.**

---

## Detailed Test Results

| Commit | SGLang Ver | sgl_kernel | uvloop | Server | Status | Issue |
|--------|-----------|-----------|--------|--------|--------|-------|
| `021f76e4` | latest | OK | OK | OK | **WORKING** | - |
| `777688b8` | latest | OK | OK | OK | **WORKING** | Parent commit |
| `c087ddd6` | latest | OK | OK | OK | **WORKING** | - |
| `79961afa` | 0.4.6.post2 | FAILED | OK | - | BROKEN | ABI mismatch |
| `9216b106` | 0.1.14 | FAILED | OK | - | BROKEN | vllm API incompatibility |
| `2a754e57` | 0.1.17 | MISSING | OK | - | BROKEN | outlines.fsm missing |
| `b1e5a33a` | 0.4.6.post5 | FAILED | OK | - | BROKEN | ABI mismatch |
| `ddcf9fe3` | 0.4.3.post2 | FAILED | OK | - | BROKEN | ABI mismatch |

---

## Quick Benchmark Results (TinyLlama)

**Note:** Full Llama-3.1-8B benchmarks require gated model access. This is a sanity test with TinyLlama-1.1B.

### Test Configuration
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Prompts: 50
- Request Rate: 4

### Results Comparison

| Metric | Baseline (777688b8) | Human (021f76e4) | Change |
|--------|---------------------|------------------|--------|
| **TTFT Mean** | 40.97 ms | 39.36 ms | **+3.9% better** |
| TTFT Median | 24.99 ms | 25.19 ms | ~same |
| **TTFT P99** | 331.80 ms | 277.01 ms | **+16.5% better** |
| ITL Mean | 4.07 ms | 4.10 ms | ~same |
| ITL Median | 3.82 ms | 3.82 ms | same |
| Throughput | 4.23 req/s | 4.23 req/s | same |
| E2E Latency Mean | 952.28 ms | 957.72 ms | ~same |

**Observation:** Human commit shows improvement in TTFT P99 (tail latency), consistent with the LoRA stream sync optimization claim.

---

## Working Images

### 021f76e4 - LoRA Stream Sync Optimization
- **PR:** #6994
- **Claimed:** 15-20% improvement in TTFT/ITL
- **Status:** FULLY FUNCTIONAL
- **Benchmark:** Confirmed working

### 777688b8 - Parent/Baseline Commit
- **Status:** FULLY FUNCTIONAL
- **Use as:** Baseline for 021f76e4 comparisons

### c087ddd6 - Triton Kernel Optimization
- **PR:** #6627
- **Claimed:** 10-15% kernel improvement
- **Status:** FULLY FUNCTIONAL
- **Note:** Needs parent commit (f4a8987f) for comparison

---

## Broken Images - Detailed Failure Analysis

### 79961afa - FA3 Metadata Init (SGLang 0.4.6.post2)
**Claimed:** 21% faster (530us -> 418us)
**Error:** sgl_kernel ABI mismatch
```
[sgl_kernel] CRITICAL: Could not load any common_ops library!
ImportError: undefined symbol: _ZN3c108ListType3getE...
```

### 9216b106 - 40% Scheduler Improvement (SGLang 0.1.14)
**Claimed:** 40% faster (90s -> 53s)
**Error:** vllm API incompatibility
```
ImportError: cannot import name '_set_default_torch_dtype' from 'vllm.model_executor.model_loader'
```
**Root Cause:** SGLang 0.1.14 expects older vllm API, but container has vllm 0.4.1

### 2a754e57 - 2x Prefill Performance (SGLang 0.1.17)
**Claimed:** 2x prefill improvement
**Error:** outlines dependency mismatch
```
ModuleNotFoundError: No module named 'outlines.fsm'
```
**Root Cause:** outlines library API changed, old SGLang code incompatible

### b1e5a33a - 13% LoRA ITL Improvement (SGLang 0.4.6.post5)
**Claimed:** 13% ITL improvement
**Error:** sgl_kernel ABI mismatch
- sgl_kernel: FAILED to load
- Other dependencies OK

### ddcf9fe3 - Triton Attention Optimization (SGLang 0.4.3.post2)
**Claimed:** 5% TTFT improvement
**Error:** sgl_kernel ABI mismatch
- sgl_kernel: FAILED to load
- Other dependencies OK

---

## Root Cause Analysis

### Issue 1: sgl_kernel ABI Mismatch (affects 4 images)
sgl_kernel was installed from PyPI pre-built wheel instead of compiled from source.

**Fix:**
```bash
cd /sglang/sgl-kernel && pip install -e . --no-build-isolation
```

### Issue 2: vllm API Incompatibility (affects 9216b106)
SGLang 0.1.14 expects older vllm API functions that don't exist in vllm 0.4.1.

**Fix:** Rebuild with matching vllm version or skip this commit.

### Issue 3: outlines API Change (affects 2a754e57)
Old SGLang versions use `outlines.fsm.guide` which no longer exists.

**Fix:** Pin outlines to older version or skip this commit.

---

## Comparison with Documentation Claims

| Document Claim | Reality |
|---------------|---------|
| "30 images ready" | Only 3 work |
| "All verified with uvloop + sgl_kernel" | Most have broken sgl_kernel |
| "5/5 tested FAILED" (FAILURES_ANALYSIS.md) | Confirmed - documentation was accurate |

**BENCHMARK_ANALYSIS.md is misleading. BENCHMARK_FAILURES_ANALYSIS.md was correct.**

---

## Recommendations

### Immediate Actions
1. **Use only working images:** `021f76e4`, `777688b8`, `c087ddd6`
2. **Get Llama-3.1-8B access** to run full benchmarks
3. **Skip old SGLang images** (0.1.x) - too many dependency issues

### Rebuild Priority
| Priority | Commit | Issue | Effort |
|----------|--------|-------|--------|
| 1 | `79961afa` | sgl_kernel rebuild | Low |
| 2 | `b1e5a33a` | sgl_kernel rebuild | Low |
| 3 | `ddcf9fe3` | sgl_kernel rebuild | Low |
| Skip | `9216b106` | vllm incompatibility | High |
| Skip | `2a754e57` | outlines incompatibility | High |

---

## Test Commands Reference

### Full Import Test
```bash
docker run --rm --gpus all shikhar481/sglang-images:<COMMIT> python3 -c "
import sgl_kernel
import uvloop
import flashinfer
import sglang
from sglang.srt.server_args import ServerArgs
print('ALL OK')
"
```

### Server Startup Test
```bash
docker run -d --rm --gpus all -p 30000:30000 \
    shikhar481/sglang-images:<COMMIT> \
    python3 -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 30000 --host 0.0.0.0
```

### Benchmark Test
```bash
docker run --rm --network host shikhar481/sglang-images:<COMMIT> \
    python3 -m sglang.bench_serving \
    --backend sglang \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --num-prompts 50 --request-rate 4
```

---

## Infrastructure Setup (L40S Machine)

Commands used to set up this machine from scratch:

```bash
# Install NVIDIA driver
sudo apt update && sudo apt install -y nvidia-driver-535 nvidia-utils-535
sudo modprobe nvidia

# Install Docker
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh && sudo sh /tmp/get-docker.sh

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

---

---

## Phase 2: Rebuilt Image Testing (2026-01-19)

After the initial testing revealed issues, 18 images were rebuilt and pushed to DockerHub. This section documents the results of testing those rebuilt images.

### Rebuilt Images Summary

| Category | Count | Result |
|----------|-------|--------|
| v0.1.x images | 7 | **ALL FAILED** - vllm API incompatibility |
| v0.2.x images | 1 | **FAILED** - Missing sgl_kernel & flashinfer |
| v0.3.x images | 10 | **ALL FAILED** - huggingface-hub version mismatch |
| **Total** | **18** | **0 working (0%)** |

---

### v0.1.x Rebuilt Images (7 images)

All v0.1.x images fail with the same vllm API incompatibility error:

```
ImportError: cannot import name '_set_default_torch_dtype' from 'vllm.model_executor.model_loader'
```

| Commit | SGLang Ver | Error |
|--------|-----------|-------|
| `9216b106` | 0.1.14 | vllm API incompatibility |
| `2a754e57` | 0.1.17 | vllm API + outlines.fsm missing |
| `9f63f95a` | 0.1.x | vllm API incompatibility |
| `67b10a05` | 0.1.x | vllm API incompatibility |
| `2591d99e` | 0.1.x | vllm API incompatibility |
| `2d9cdf8d` | 0.1.x | vllm API incompatibility |
| `91bb5c87` | 0.1.x | vllm API incompatibility |

**Root Cause:** SGLang 0.1.x expects vllm functions like `_set_default_torch_dtype` that were removed in newer vllm versions. The containers have vllm 0.4.1+ installed but SGLang 0.1.x code requires older vllm API.

**Fix Required:** Must rebuild with compatible vllm version (vllm ~0.2.x or older) or patch SGLang code to work with newer vllm.

---

### v0.2.x Rebuilt Image (1 image)

| Commit | SGLang Ver | sgl_kernel | flashinfer | uvloop | Status |
|--------|-----------|------------|------------|--------|--------|
| `62757db6` | 0.2.11 | MISSING | MISSING | OK | **BROKEN** |

**Error:**
```
sgl_kernel: FAILED - No module named 'sgl_kernel'
flashinfer: FAILED - No module named 'flashinfer'
```

**Root Cause:** The rebuild did not install sgl_kernel or flashinfer. These modules are critical for SGLang performance but were omitted from the container.

**Fix Required:** Rebuild with sgl_kernel and flashinfer installed.

---

### v0.3.x Rebuilt Images (10 images)

All v0.3.x images fail with dependency version mismatches:

**Error Pattern 1 (5 images):**
```
huggingface-hub>=0.34.0,<1.0 is required but found huggingface-hub==1.3.2
```

**Error Pattern 2 (5 images):**
```
ImportError: cannot import name 'AutoProcessor' from 'transformers'
```

| Commit | SGLang Ver | Error Type |
|--------|-----------|------------|
| `593f80e8` | 0.3.x | huggingface-hub version |
| `35f53ba7` | 0.3.x | huggingface-hub version |
| `5bc1b8c4` | 0.3.x | huggingface-hub version |
| `7e68c9e0` | 0.3.x | huggingface-hub version |
| `45da9d5e` | 0.3.x | huggingface-hub version |
| `2854a5ea` | 0.3.x | AutoProcessor import |
| `9c064bf7` | 0.3.x | AutoProcessor import |
| `c98e84c2` | 0.3.x | AutoProcessor import |
| `b1709305` | 0.3.x | AutoProcessor import |
| `0628ef1b` | 0.3.x | AutoProcessor import |

**Root Cause:** The containers have newer versions of huggingface-hub (1.3.2) and transformers than SGLang 0.3.x was designed for. The older transformers library doesn't have `AutoProcessor`, and the huggingface-hub version constraint `<1.0` conflicts with installed version.

**Fix Required:** Pin huggingface-hub to `<1.0` and transformers to a compatible version when building.

---

### Phase 2 Conclusions

The rebuild attempt did not fix the underlying dependency issues:

1. **v0.1.x images** - Need older vllm version, not just sgl_kernel fix
2. **v0.2.x images** - Missing critical dependencies (sgl_kernel, flashinfer)
3. **v0.3.x images** - Need older huggingface-hub/transformers versions

### Recommendations for Next Rebuild

| Version | Required Fix |
|---------|-------------|
| v0.1.x | Install vllm ~0.2.x (not 0.4.x) |
| v0.2.x | Install sgl_kernel and flashinfer from source |
| v0.3.x | Pin `huggingface-hub<1.0` and compatible transformers |
| v0.4.x | Build sgl_kernel from source (requires GPU) |

### Working Images (Unchanged)

Only the original 3 images remain functional:
- `021f76e4` - LoRA Stream Sync Optimization
- `777688b8` - Parent/Baseline Commit
- `c087ddd6` - Triton Kernel Optimization

---

*Updated: 2026-01-19 11:30 UTC*
*Tester: Claude Code on L40S instance*
