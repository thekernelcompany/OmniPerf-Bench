# SGLang Docker Images - Benchmark Failure Analysis

**Date:** 2026-01-18
**Tested on:** NVIDIA H100 PCIe, Driver 535.183.06, CUDA 12.2

## Executive Summary

The 30 Docker images claimed to work in BENCHMARK_ANALYSIS.md **do not function as advertised**. Multiple critical dependency and compatibility issues prevent benchmarks from running.

## Environment

| Component | Version |
|-----------|---------|
| Host GPU | NVIDIA H100 PCIe |
| Host Driver | 535.183.06 |
| Host CUDA | 12.2 |
| Container CUDA | 12.4.1 |
| Docker Storage | VFS (DinD) |

## All 30 Images Status

| # | Commit | PR | Subject | Tested? | Status |
|---|--------|-----|---------|---------|--------|
| 1 | 2a754e57 | #579 | 2x prefill | YES | FAILED - missing rpyc, outlines, pyairports |
| 2 | 9216b106 | #394 | 40% scheduler | YES | FAILED - missing pyairports |
| 3 | b1e5a33a | #6960 | 13% LoRA ITL | NO | - |
| 4 | 79961afa | #6077 | 21% FA3 faster | YES | FAILED - sgl_kernel ABI mismatch |
| 5 | 3212c2ad | #6003 | 16% VLM faster | NO | - |
| 6 | c087ddd6 | #6627 | 10-15% kernel | YES | FAILED - segfault during CUDA graph |
| 7 | 1acca3a2 | #5969 | FA3 len() removal | NO | - |
| 8 | 2bd18e2d | #2901 | Memory pool | NO | - |
| 9 | d1112d85 | #2797 | Input embeds | NO | - |
| 10 | 10189d08 | #2171 | CPU affinity | NO | - |
| 11 | ddcf9fe3 | #3731 | Triton attention | NO | - |
| 12 | 93470a14 | #5090 | FA3 optimization | NO | - |
| 13 | f4a8987f | - | Parent baseline | NO | - |
| 14 | 09deb20d | #420 | Logits memory | NO | - |
| 15 | 2854a5ea | #1496 | bench_latency fix | NO | - |
| 16 | 564a898a | #619 | Mem indices | NO | - |
| 17 | 62757db6 | #1010 | Cache disabled | NO | - |
| 18 | 6a2941f4 | #625 | TP overhead | NO | - |
| 19 | 6f560c76 | #117 | First token latency | NO | - |
| 20 | 8f8f96a6 | #1773 | stop_token_ids | NO | - |
| 21 | 9183c23e | #2695 | Weights update | NO | - |
| 22 | 9c064bf7 | #1587 | LoRA Step 1 | NO | - |
| 23 | 9c745d07 | #2056 | xgrammar | NO | - |
| 24 | ab4a83b2 | #1339 | Optimize schedule | YES | FAILED - missing pyairports |
| 25 | ac971ff6 | #658 | stream_interval | NO | - |
| 26 | b1709305 | #1697 | Radix tree | NO | - |
| 27 | b77a02cd | #1752 | Grammar backends | NO | - |
| 28 | c98e84c2 | #1589 | torch.argmax | NO | - |
| 29 | e3ec6bf4 | #6814 | FP8 quant | NO | - |
| 30 | e5db40dc | #1694 | ORJson | NO | - |

**Tested: 5/30 | All 5 tested FAILED**

## Images Tested (Details)

### Image: 2a754e57 (sglang 0.1.17)
**Claimed:** 2x prefill improvement
**Status:** FAILED

**Errors:**
```
ModuleNotFoundError: No module named 'rpyc'
ModuleNotFoundError: No module named 'outlines'
ModuleNotFoundError: No module named 'pyairports'
```

### Image: 9216b106 (sglang 0.1.14)
**Claimed:** 40% scheduler improvement
**Status:** FAILED

**Errors:**
```
ModuleNotFoundError: No module named 'pyairports'
```

### Image: 79961afa (sglang 0.4.6.post2)
**Claimed:** 21% FA3 faster
**Status:** FAILED

**Errors:**
- sgl_kernel ABI mismatch: `undefined symbol: _ZN3c108ListType...`
- ModuleNotFoundError: No module named 'sentencepiece'
- ModuleNotFoundError: No module named 'outlines'

### Image: c087ddd6 (sglang 0.4.6.post5, torch 2.6.0)
**Claimed:** 10-15% kernel improvement
**Status:** FAILED

**Errors:**
- All dependencies present (uvloop, sentencepiece, sgl_kernel OK)
- **Segmentation fault (exit code 139)** during CUDA graph capture
- Likely CUDA 12.4 vs host CUDA 12.2 driver incompatibility

### Image: ab4a83b2 (sglang 0.3.0)
**Claimed:** Optimize schedule
**Status:** FAILED

**Errors:**
```
ModuleNotFoundError: No module named 'pyairports'
```

## Root Causes

### 1. Broken Outlines Dependency
The installed `outlines` package requires `pyairports` which is not installed in any image:
```python
from outlines.types.airports import ...
from pyairports.airports import AIRPORT_LIST  # <-- Missing
```

This affects images with sglang 0.1.x, 0.3.x

### 2. sgl_kernel ABI Mismatch
Newer images (0.4.x) have sgl_kernel compiled against different PyTorch version:
```
undefined symbol: _ZN3c108ListType3getE...
```

### 3. CUDA Version Mismatch
- Containers built with CUDA 12.4.1
- Host driver supports CUDA 12.2
- Results in segmentation faults during CUDA graph capture

### 4. Missing Dependencies
Various images missing:
- rpyc
- sentencepiece
- outlines (correct version)
- pyairports

## Conclusion

**BENCHMARK_ANALYSIS.md claims are NOT accurate.** The 30 Docker images cannot run benchmarks without significant fixes:

1. Install missing Python packages in containers
2. Rebuild sgl_kernel from source against container's PyTorch
3. Either use CUDA 12.2 base images or upgrade host driver to support CUDA 12.4

## Recommendations

1. **Rebuild all images** with correct dependencies installed
2. **Pin outlines version** to pre-0.1.0 that doesn't require pyairports
3. **Build sgl_kernel from source** during image build, not from PyPI
4. **Use CUDA 12.2** base images for compatibility with driver 535.x
5. **Test images locally** before marking as "ready" in documentation

## Testing Methodology

### Version-Specific Differences

Different sglang versions have different module structures and dependencies:

| SGLang Version | Benchmark Module | Key Dependencies |
|----------------|------------------|------------------|
| 0.1.x | `sglang.bench_latency` | rpyc, outlines (old API) |
| 0.3.x | `sglang.bench_latency` | outlines (different API) |
| 0.4.x | `sglang.bench_one_batch` | sgl_kernel, sentencepiece |

**Important:** A generic dependency check will NOT work for all images. Each version has different import paths and requirements.

### Step 1: Check Version and Available Modules (No GPU Required)

First, identify what sglang version and benchmark modules exist in an image:

```bash
COMMIT="2a754e57"
docker pull shikhar481/sglang-images:$COMMIT

docker run --rm shikhar481/sglang-images:$COMMIT python3 -c "
import sglang
print(f'Version: {sglang.__version__}')

# List available benchmark modules
import os
sgl_path = os.path.dirname(sglang.__file__)
print('Available modules:')
for f in os.listdir(sgl_path):
    if 'bench' in f:
        print(f'  {f}')
"
```

### Step 2: Test Import Chain (No GPU Required)

Test if the benchmark module can be imported - this reveals missing dependencies:

```bash
# For sglang 0.1.x and 0.3.x images:
docker run --rm shikhar481/sglang-images:$COMMIT python3 -c "
from sglang.bench_latency import *
print('bench_latency imports OK')
"

# For sglang 0.4.x images:
docker run --rm shikhar481/sglang-images:$COMMIT python3 -c "
from sglang.bench_one_batch import *
print('bench_one_batch imports OK')
"
```

This is where most failures occur - the import chain pulls in dependencies like outlines/pyairports/sgl_kernel and fails.

### Step 3: Test Individual Dependencies (No GPU Required)

Check specific dependencies that commonly fail:

```bash
docker run --rm shikhar481/sglang-images:$COMMIT python3 -c "
import sys

def check(name):
    try:
        __import__(name)
        print(f'{name}: OK')
    except Exception as e:
        print(f'{name}: FAILED - {e}')

check('rpyc')
check('uvloop')
check('sentencepiece')
check('outlines')
check('pyairports')
check('sgl_kernel')
"
```

### Step 4: Run Actual Benchmark (GPU Required)

Only after steps 1-3 pass, attempt the actual benchmark:

```bash
# For 0.1.x/0.3.x with bench_latency:
docker run --rm --gpus all \
    -e HF_TOKEN="your_token" \
    -v /path/to/cache:/root/.cache/huggingface \
    shikhar481/sglang-images:$COMMIT \
    python3 -m sglang.bench_latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 32 --input-len 1024 --output-len 128

# For 0.4.x with bench_one_batch:
docker run --rm --gpus all \
    -e HF_TOKEN="your_token" \
    -v /path/to/cache:/root/.cache/huggingface \
    shikhar481/sglang-images:$COMMIT \
    python3 -m sglang.bench_one_batch \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 32 --input-len 1024 --output-len 128
```

### Per-Image Benchmark Commands

The correct benchmark command for each image is defined in `/tmp/sglang_3way_candidates.json`. Example entry:

```json
{
  "human": "2a754e57",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "perf_command": "python3 -m sglang.bench_latency --batch-size 1 --input-len 8192 --output-len 1 --model meta-llama/Llama-3.1-8B-Instruct",
  "subject": "2x prefill (#579)"
}
```

To properly test each image, use its specific `perf_command` from the candidates file.

## Appendix: VFS Storage Impact

VFS storage driver (required for DinD) consumed 738GB for just 4 images (~16GB each visible). This makes testing all 30 images impractical without overlay2 support or direct host Docker access.
