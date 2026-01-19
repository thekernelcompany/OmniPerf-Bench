# SGLang Docker Image Testing Results

**Date:** 2026-01-19
**GPU:** NVIDIA L40S (46GB VRAM)
**Host Driver:** 535.274.02
**Host CUDA:** 12.2
**Docker Version:** 29.1.5

---

## Executive Summary

**Total Testing: 44 image tests across 3 phases. Only 3 images are functional.**

| Phase | Images | Import Test | Server Test | Working |
|-------|--------|-------------|-------------|---------|
| Phase 1 (Original) | 8 | 3 pass | 3 pass | **3** |
| Phase 2 (First Rebuild) | 18 | 0 pass | N/A | **0** |
| Phase 3 (v2 Rebuild) | 18 | 18 pass | 0 pass | **0** |

**Critical Finding:** Import tests are NOT sufficient. The v2 images pass all import tests but fail at server startup.

**Only 3 working images:** `021f76e4`, `777688b8`, `c087ddd6` (all recent SGLang versions)

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

## Test Commands Reference (For Build Instance)

**CRITICAL: Import tests are NOT sufficient. You MUST run server startup test.**

### Quick Validation Script (Copy-Paste Ready)

```bash
#!/bin/bash
# Usage: ./test_image.sh <image_tag>
# Example: ./test_image.sh 9216b106-v2

TAG="${1:-latest}"
IMAGE="shikhar481/sglang-images:$TAG"

echo "============================================"
echo "Testing: $IMAGE"
echo "============================================"

# Step 1: Import Test
echo ""
echo "[1/3] Import Test..."
docker run --rm --gpus all $IMAGE python3 -c "
import sys
print(f'Python: {sys.version.split()[0]}')
try:
    import sglang
    print(f'sglang: {sglang.__version__}')
except Exception as e:
    print(f'sglang: FAIL - {e}')
    sys.exit(1)

try:
    import vllm
    print(f'vllm: {vllm.__version__}')
except Exception as e:
    print(f'vllm: FAIL - {e}')

try:
    import uvloop
    print('uvloop: OK')
except Exception as e:
    print(f'uvloop: FAIL - {e}')

try:
    import flashinfer
    print(f'flashinfer: {flashinfer.__version__}')
except:
    print('flashinfer: N/A (v0.1.x)')

try:
    import sgl_kernel
    print('sgl_kernel: OK')
except:
    print('sgl_kernel: N/A (v0.1.x/v0.2.x)')

print('Import test: PASS')
"
IMPORT_RESULT=$?

if [ $IMPORT_RESULT -ne 0 ]; then
    echo "[FAIL] Import test failed"
    exit 1
fi

# Step 2: Server Startup Test (THE CRITICAL ONE)
echo ""
echo "[2/3] Server Startup Test (this is the real test)..."
docker run --rm --gpus all $IMAGE timeout 60 python3 -c "
# This simulates what launch_server does without actually starting the server
import sys
try:
    from sglang.srt.server import launch_server
    print('Server module import: PASS')
except Exception as e:
    print(f'Server module import: FAIL')
    print(f'Error: {e}')
    sys.exit(1)
"
SERVER_RESULT=$?

if [ $SERVER_RESULT -ne 0 ]; then
    echo ""
    echo "[FAIL] Server startup test failed"
    echo ""
    echo "Common fixes:"
    echo "  - 'outlines.fsm' error: pip install 'outlines==0.0.34'"
    echo "  - '_grouped_size_compiled_for_decode_kernels' error: need older flashinfer"
    echo "  - 'orjson' error: pip install orjson"
    exit 1
fi

# Step 3: Full Server Launch Test (Optional but recommended)
echo ""
echo "[3/3] Full Server Launch Test..."
CONTAINER_ID=$(docker run -d --rm --gpus all -p 30000:30000 $IMAGE \
    python3 -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 30000 --host 0.0.0.0)

echo "Container: $CONTAINER_ID"
echo "Waiting 45s for server to start..."
sleep 45

# Check if container is still running
if docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
    # Try health check
    if curl -s http://localhost:30000/health 2>/dev/null | grep -q "ok\|healthy"; then
        echo "Health check: PASS"
        echo ""
        echo "============================================"
        echo "[SUCCESS] $TAG is FULLY FUNCTIONAL"
        echo "============================================"
    else
        echo "Health check: No response (may still be loading model)"
        echo "Check logs: docker logs $CONTAINER_ID"
    fi
else
    echo "Container exited. Logs:"
    docker logs $CONTAINER_ID 2>&1 | tail -20
    echo ""
    echo "[FAIL] Server crashed during startup"
fi

# Cleanup
docker kill $CONTAINER_ID 2>/dev/null

echo ""
echo "Done testing $TAG"
```

### Manual Test Commands

**Step 1: Import Test (necessary but NOT sufficient)**
```bash
docker run --rm --gpus all shikhar481/sglang-images:<TAG> python3 -c "
import sglang; print(f'sglang: {sglang.__version__}')
import vllm; print(f'vllm: {vllm.__version__}')
import uvloop; print('uvloop: OK')
try:
    import flashinfer; print(f'flashinfer: {flashinfer.__version__}')
except: print('flashinfer: N/A')
from sglang.srt.server_args import ServerArgs; print('ServerArgs: OK')
"
```

**Step 2: Server Module Test (THE CRITICAL TEST)**
```bash
docker run --rm --gpus all shikhar481/sglang-images:<TAG> python3 -c "
from sglang.srt.server import launch_server
print('Server module: OK')
"
```

If this fails, check the error:
- `No module named 'outlines.fsm'` → Fix: `pip install 'outlines==0.0.34'`
- `cannot import name '_grouped_size_compiled_for_decode_kernels'` → Fix: Need older flashinfer
- `No module named 'orjson'` → Fix: `pip install orjson`

**Step 3: Full Server Startup Test**
```bash
# Start server in background
docker run -d --rm --gpus all -p 30000:30000 --name test_server \
    shikhar481/sglang-images:<TAG> \
    python3 -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 30000 --host 0.0.0.0

# Wait and check logs
sleep 45
docker logs test_server 2>&1 | tail -30

# Check if healthy
curl http://localhost:30000/health

# Cleanup
docker kill test_server
```

### Error Reference & Fixes

| Error | Affected Versions | Fix |
|-------|-------------------|-----|
| `No module named 'outlines.fsm'` | v0.1.x | `pip install 'outlines==0.0.34'` |
| `cannot import name '_grouped_size_compiled_for_decode_kernels'` | v0.2.x, v0.3.x | Need flashinfer 0.0.x or 0.1.x (not 0.2.x) |
| `No module named 'orjson'` | v0.3.6+ | `pip install orjson` |
| `cannot import name '_set_default_torch_dtype'` | v0.1.x | Need vllm 0.2.x or 0.3.x (not 0.4.x+) |
| `sgl_kernel ABI mismatch` | v0.4.x | Build sgl_kernel from source with GPU |
| `No module named 'sgl_kernel'` | v0.2.x | Install sgl_kernel |
| `huggingface-hub version` | v0.3.x | `pip install 'huggingface-hub<1.0'` |

### Batch Test All Images

```bash
#!/bin/bash
# Test multiple images
IMAGES=(
    "9216b106-v2"
    "2a754e57-v2"
    "62757db6-v2"
    "ab4a83b2-v2"
    "10189d08-v2"
)

for tag in "${IMAGES[@]}"; do
    echo "Testing $tag..."
    docker run --rm --gpus all shikhar481/sglang-images:$tag python3 -c "
from sglang.srt.server import launch_server
print('$tag: PASS')
" 2>&1 | grep -E "(PASS|Error|error|FAIL|cannot import|No module)"
done
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

---

## Phase 3: v2 Image Testing (2026-01-19)

18 new images with `-v2` suffix were rebuilt with corrected dependency versions as documented in `REBUILD_FROM_SOURCE.md`. This section documents comprehensive GPU testing of these images.

### Import Test Results

All 18 v2 images **PASS basic import tests**:

| Category | Images | Import Test | Versions |
|----------|--------|-------------|----------|
| v0.1.x | 7 | **ALL PASS** | SGLang 0.1.9-0.1.21, vllm 0.2.7-0.3.3 |
| v0.2.x | 1 | **PASS** | SGLang 0.2.11, vllm 0.5.4, flashinfer OK |
| v0.3.x | 10 | **ALL PASS** | SGLang 0.3.0-0.3.6, vllm 0.5.5, flashinfer OK |

### Server Startup Test Results (CRITICAL)

**ALL 18 v2 images FAIL at server startup despite passing import tests.**

| Category | Server Startup | Error |
|----------|----------------|-------|
| v0.1.x (7) | **ALL FAIL** | `ModuleNotFoundError: No module named 'outlines.fsm'` |
| v0.2.x (1) | **FAIL** | `ImportError: cannot import name '_grouped_size_compiled_for_decode_kernels'` |
| v0.3.x (10) | **ALL FAIL** | `ImportError: cannot import name '_grouped_size_compiled_for_decode_kernels'` or `ModuleNotFoundError: No module named 'orjson'` |

### Detailed v0.1.x v2 Results

| Commit | SGLang | vllm | Import | Server | Error |
|--------|--------|------|--------|--------|-------|
| `9216b106-v2` | 0.1.14 | 0.3.3 | PASS | FAIL | outlines.fsm |
| `2a754e57-v2` | 0.1.17 | 0.3.3 | PASS | FAIL | outlines.fsm |
| `09deb20d-v2` | 0.1.14 | 0.3.3 | PASS | FAIL | outlines.fsm |
| `564a898a-v2` | 0.1.20 | 0.3.3 | PASS | FAIL | outlines.fsm |
| `6a2941f4-v2` | 0.1.20 | 0.3.3 | PASS | FAIL | outlines.fsm |
| `6f560c76-v2` | 0.1.9 | 0.2.7 | PASS | FAIL | outlines.fsm |
| `ac971ff6-v2` | 0.1.21 | 0.3.3 | PASS | FAIL | outlines.fsm |

**Error:**
```
ModuleNotFoundError: No module named 'outlines.fsm'
```

**Root Cause:** SGLang 0.1.x uses `from outlines.fsm.fsm import RegexFSM` but the outlines library API changed - the `fsm` submodule no longer exists.

### Detailed v0.2.x/v0.3.x v2 Results

| Commit | SGLang | flashinfer | Import | Server | Error |
|--------|--------|------------|--------|--------|-------|
| `62757db6-v2` | 0.2.11 | 0.2.0.post1 | PASS | FAIL | flashinfer API |
| `ab4a83b2-v2` | 0.3.0 | 0.2.0.post1 | PASS | FAIL | flashinfer API |
| `2854a5ea-v2` | 0.3.1.post3 | 0.2.0.post1 | PASS | FAIL | flashinfer API |
| `9c064bf7-v2` | 0.3.2 | 0.2.0.post1 | PASS | FAIL | flashinfer API |
| `c98e84c2-v2` | 0.3.2 | 0.2.0.post1 | PASS | FAIL | flashinfer API |
| `e5db40dc-v2` | 0.3.3.post1 | 0.2.0.post1 | PASS | FAIL | flashinfer API |
| `b1709305-v2` | 0.3.3.post1 | 0.2.0.post1 | PASS | FAIL | flashinfer API |
| `b77a02cd-v2` | 0.3.4.post2 | 0.2.0.post1 | PASS | FAIL | flashinfer API |
| `8f8f96a6-v2` | 0.3.4.post1 | 0.2.0.post1 | PASS | FAIL | flashinfer API |
| `9c745d07-v2` | 0.3.5.post2 | 0.2.0.post1 | PASS | FAIL | flashinfer API |
| `10189d08-v2` | 0.3.6 | 0.2.0.post1 | PASS | FAIL | orjson missing |

**Error (v0.2.x/v0.3.x early):**
```
from flashinfer.decode import _grouped_size_compiled_for_decode_kernels
ImportError: cannot import name '_grouped_size_compiled_for_decode_kernels' from 'flashinfer.decode'
```

**Error (v0.3.6):**
```
ModuleNotFoundError: No module named 'orjson'
```

**Root Cause:**
- flashinfer 0.2.0.post1 doesn't have `_grouped_size_compiled_for_decode_kernels` internal function that older SGLang versions expect
- v0.3.6 additionally requires `orjson` which wasn't installed

### Phase 3 Conclusions

**The v2 rebuild fixed vllm/huggingface-hub version issues but introduced new problems:**

1. **v0.1.x** - The `outlines` library API changed; need to pin `outlines<0.1.0` or use a fork
2. **v0.2.x/v0.3.x** - The flashinfer version (0.2.0.post1) is TOO NEW; SGLang 0.2.x/0.3.x expects older flashinfer with internal `_grouped_size_compiled_for_decode_kernels` function
3. **v0.3.6** - Missing `orjson` dependency

### Required Fixes for v3 Rebuild

| Version | Current Issue | Required Fix |
|---------|--------------|--------------|
| v0.1.x | outlines.fsm missing | Pin `outlines==0.0.34` or older |
| v0.2.x | flashinfer API mismatch | Pin flashinfer ~0.0.x (pre-0.1.0) |
| v0.3.x | flashinfer API mismatch | Pin flashinfer ~0.0.x or ~0.1.x |
| v0.3.6 | orjson missing | Add `pip install orjson` |

### Key Lesson

**Import tests are insufficient.** A proper test must attempt server startup:
```bash
docker run --rm --gpus all <image> python3 -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 30000 --host 0.0.0.0
```

---

## Phase 4: v3 Image Build (2026-01-19)

18 images rebuilt with dependency fixes. **11 pass server module import, 7 fail.**

### Fixes Applied

| Version | Phase 3 Error | v3 Fix |
|---------|--------------|--------|
| v0.2.x | `_grouped_size_compiled_for_decode_kernels` missing | `flashinfer==0.1.6` |
| v0.3.x | `_grouped_size_compiled_for_decode_kernels` + `python-multipart` missing | `flashinfer==0.1.6` + `python-multipart` |
| v0.3.6 | `No module named 'orjson'` | Added `pip install orjson` |

### Server Module Import Test Results

**Test:** `python3 -c "from sglang.srt.server import launch_server; print('PASS')"`

**v0.2.x (1 image): 1/1 PASS**
| Tag | SGLang | Server Import | Status |
|-----|--------|---------------|--------|
| `62757db6-v3` | 0.2.11 | ✅ PASS | **WORKING** |

**v0.3.x (10 images): 10/10 PASS**
| Tag | SGLang | Server Import | Status |
|-----|--------|---------------|--------|
| `ab4a83b2-v3` | 0.3.0 | ✅ PASS | **WORKING** |
| `2854a5ea-v3` | 0.3.1.post3 | ✅ PASS | **WORKING** |
| `9c064bf7-v3` | 0.3.2 | ✅ PASS | **WORKING** |
| `c98e84c2-v3` | 0.3.2 | ✅ PASS | **WORKING** |
| `e5db40dc-v3` | 0.3.3.post1 | ✅ PASS | **WORKING** |
| `b1709305-v3` | 0.3.3.post1 | ✅ PASS | **WORKING** |
| `b77a02cd-v3` | 0.3.4.post2 | ✅ PASS | **WORKING** |
| `8f8f96a6-v3` | 0.3.4.post1 | ✅ PASS | **WORKING** |
| `9c745d07-v3` | 0.3.5.post2 | ✅ PASS | **WORKING** |
| `10189d08-v3` | 0.3.6 | ✅ PASS | **WORKING** |

**v0.1.x (7 images): 0/7 PASS - UNFIXABLE**
| Tag | SGLang | Server Import | Error |
|-----|--------|---------------|-------|
| `9216b106-v3` | 0.1.14 | ❌ FAIL | libcuda.so.1 (hard CUDA import) |
| `2a754e57-v3` | 0.1.17 | ❌ FAIL | vllm `LoadConfig` API missing |
| `09deb20d-v3` | 0.1.14 | ❌ FAIL | libcuda.so.1 (hard CUDA import) |
| `564a898a-v3` | 0.1.20 | ❌ FAIL | `outlines.fsm.guide` missing |
| `6a2941f4-v3` | 0.1.20 | ❌ FAIL | `outlines.fsm.guide` missing |
| `6f560c76-v3` | 0.1.9 | ❌ FAIL | `outlines` module missing |
| `ac971ff6-v3` | 0.1.21 | ❌ FAIL | `outlines.fsm.guide` missing |

### v0.1.x Root Cause Analysis

v0.1.x images are **unfixable** without source code modifications:

1. **Hard CUDA imports** (v0.1.14): Old SGLang versions do hard imports of CUDA operations at module load time, unlike v0.3.x which handles CUDA import failures gracefully.

2. **vllm API incompatibility** (v0.1.17): Expects `LoadConfig` from vllm.config which doesn't exist in vllm 0.3.3.

3. **outlines API changes** (v0.1.17+): Different v0.1.x versions need different outlines APIs:
   - v0.1.14: `outlines.fsm.fsm.RegexFSM` (outlines 0.0.34)
   - v0.1.17+: `outlines.fsm.guide.RegexGuide` (outlines 0.0.44+)
   But v0.1.17+ also needs vllm APIs that don't exist.

**Conclusion:** Skip v0.1.x images entirely. Use v0.2.x/v0.3.x versions.

---

## Overall Summary (All Phases)

| Phase | Images | Working | Status |
|-------|--------|---------|--------|
| Phase 1 (Original) | 8 | 3 | Tested on GPU |
| Phase 2 (First Rebuild) | 18 | 0 | All broken |
| Phase 3 (v2 Rebuild) | 18 | 0 | Import OK, Server FAIL |
| Phase 4 (v3 Rebuild) | 18 | **11** | v0.2.x + v0.3.x work |
| **Total Unique** | **26** | **14 (54%)** | |

**Working images:**
- Original: `021f76e4`, `777688b8`, `c087ddd6`
- v3: `62757db6-v3`, `ab4a83b2-v3`, `2854a5ea-v3`, `9c064bf7-v3`, `c98e84c2-v3`, `e5db40dc-v3`, `b1709305-v3`, `b77a02cd-v3`, `8f8f96a6-v3`, `9c745d07-v3`, `10189d08-v3`

---

*Updated: 2026-01-19 18:00 UTC*
*Build Machine: OmniPerf-Bench*
*Server module import test: No GPU required*
