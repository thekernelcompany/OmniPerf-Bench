# GPU Testing Guide for SGLang Images

**Last Updated:** 2026-01-20
**GPU Tested:** NVIDIA L40S (SM89, 46GB VRAM)

---

## CRITICAL: Use `-src` Images

The original images (`2a754e57`, `ab4a83b2`, `79961afa`) are **BROKEN**. Use the `-src` suffix images instead.

| Original Image | Status | Replacement |
|----------------|--------|-------------|
| `2a754e57` | ❌ BROKEN (outlines.fsm missing) | Skip - v0.1.x unfixable |
| `ab4a83b2` | ❌ BROKEN (sglang not installed) | ✅ `ab4a83b2-src` |
| `79961afa` | ❌ BROKEN (sgl_kernel ABI mismatch) | Needs GPU rebuild |

---

## Working Images

### v0.4.x Images (Built from Source with CUDA Stubs)

| Tag | Version | Torch | FlashInfer | vLLM | Status |
|-----|---------|-------|------------|------|--------|
| `2bd18e2d-src` | v0.4.1.post6 | 2.4.0 | 0.1.6 | 0.6.3.post1 | ❌ BROKEN (torchao/torch mismatch) |
| `d1112d85-src` | v0.4.4.post1 | 2.5.1 | 0.2.3 | 0.7.2 | ✅ GPU WORKING |

### v0.3.x Source Builds (`-src`) - ALL WORKING

| Tag | SGLang | vLLM | Status |
|-----|--------|------|--------|
| `62757db6-src` | v0.2.11 | 0.5.4 | ✅ GPU WORKING |
| `ab4a83b2-src` | v0.3.0 | 0.5.5 | ✅ GPU WORKING |
| `2854a5ea-src` | v0.3.1.post3 | 0.5.5 | ✅ GPU WORKING |
| `c98e84c2-src` | v0.3.2 | 0.5.5 | ✅ GPU WORKING |
| `9c064bf7-src` | v0.3.2 | 0.5.5 | ✅ GPU WORKING |
| `e5db40dc-src` | v0.3.3.post1 | 0.5.5 | ✅ GPU WORKING |
| `b1709305-src` | v0.3.3.post1 | 0.5.5 | ✅ GPU WORKING |
| `8f8f96a6-src` | v0.3.4.post1 | 0.5.5 | ✅ GPU WORKING |
| `b77a02cd-src` | v0.3.4.post2 | 0.5.5 | ✅ GPU WORKING |

### v3 Builds - WORKING

| Tag | SGLang | vLLM | Status |
|-----|--------|------|--------|
| `9c745d07-v3` | v0.3.5.post2 | 0.6.3.post1 | ✅ GPU WORKING |
| `10189d08-v3` | v0.3.6 | 0.6.3.post1 | ✅ GPU WORKING |

### Original Images - WORKING

| Tag | SGLang | Status |
|-----|--------|--------|
| `021f76e4` | latest | ✅ GPU WORKING |
| `777688b8` | latest | ✅ GPU WORKING |
| `c087ddd6` | latest | ✅ GPU WORKING |

---

## BLOCKED v0.4.x Images

| Commit | Version | Status | Notes |
|--------|---------|--------|-------|
| `79961afa` | v0.4.6.post2 | BLOCKED | Requires torch>=2.6.0 |
| `93470a14` | v0.4.5 | BLOCKED | CMake FetchContent can't access flashinfer fork |
| `3212c2ad` | v0.4.9.post4 | BLOCKED | Requires torch>=2.7.1 |

---

## Quick Test Commands

### Pull Working v0.4.x Image

```bash
# Only d1112d85-src works; 2bd18e2d-src has torchao/torch version conflict
docker pull shikhar481/sglang-images:d1112d85-src
```

### Test v0.4.x Imports

```bash
docker run --rm --gpus all shikhar481/sglang-images:d1112d85-src python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import sglang, sgl_kernel
print(f'SGLang {sglang.__version__} - All imports OK')
from sglang.srt.server import launch_server
print('Server module: OK')
"
```

### Test Server Startup

```bash
# Start server
docker run -d --gpus all --name test-sglang \
    -p 30000:30000 \
    -e HF_HOME=/root/.cache/huggingface \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    shikhar481/sglang-images:ab4a83b2-src \
    python3 -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host 0.0.0.0 --port 30000

# Wait for startup
sleep 45

# Verify
curl http://localhost:30000/get_model_info

# Test generation
curl http://localhost:30000/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 10}}'

# Cleanup
docker stop test-sglang && docker rm test-sglang
```

---

## Benchmark Commands

### Latency Benchmark

```bash
# v0.4.x
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN \
    shikhar481/sglang-images:2bd18e2d-src \
    python3 -m sglang.bench_latency \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --batch-size 1 --input-len 128 --output-len 32

# v0.3.x
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN \
    shikhar481/sglang-images:ab4a83b2-src \
    python3 -m sglang.bench_latency \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --batch-size 1 --input-len 128 --output-len 32
```

---

## Known Issues

### v0.1.x Images - UNFIXABLE

All v0.1.x images (including `2a754e57`) fail with:
```
ModuleNotFoundError: No module named 'outlines.fsm'
```

**Root Cause:** The outlines library API changed completely. These commits require source code modifications to work.

**Recommendation:** Skip v0.1.x images. Use v0.2.x+ instead.

### v0.4.x sgl_kernel ABI Mismatch

Pre-built v0.4.x images may fail with:
```
sgl_kernel: CRITICAL: Could not load any common_ops library!
- ImportError: undefined symbol: _ZN3c108ListType3get...
```

**Root Cause:** sgl_kernel was compiled for SM100 (Blackwell) but running on SM89 (L40S/Ada).

**Fix:** Use `-src` images built with CUDA stubs, or rebuild sgl_kernel from source:
```bash
cd /sglang/sgl-kernel
pip install -e . --no-build-isolation
```

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Working `-src` v0.4.x images | 1 | ✅ d1112d85 (v0.4.4) |
| Working `-src` v0.3.x images | 9 | ✅ v0.2.x to v0.3.4 |
| Working `-v3` images | 2 | ✅ v0.3.5 to v0.3.6 |
| Working original images | 3 | ✅ Latest |
| **Total Working** | **15** | Ready for benchmarks |
| Broken v0.1.x | ~15 | ❌ Skip |
| Broken v0.4.x | 4 | ⚠️ Dependency issues (2bd18e2d, 79961afa, 93470a14, 3212c2ad) |

---

## All Available Images

```bash
# v0.4.x (only d1112d85 works)
docker pull shikhar481/sglang-images:d1112d85-src  # v0.4.4.post1 - WORKING
# docker pull shikhar481/sglang-images:2bd18e2d-src  # v0.4.1.post6 - BROKEN (torchao conflict)

# v0.3.x source builds
docker pull shikhar481/sglang-images:62757db6-src  # v0.2.11
docker pull shikhar481/sglang-images:ab4a83b2-src  # v0.3.0
docker pull shikhar481/sglang-images:2854a5ea-src  # v0.3.1.post3
docker pull shikhar481/sglang-images:c98e84c2-src  # v0.3.2
docker pull shikhar481/sglang-images:9c064bf7-src  # v0.3.2
docker pull shikhar481/sglang-images:e5db40dc-src  # v0.3.3.post1
docker pull shikhar481/sglang-images:b1709305-src  # v0.3.3.post1
docker pull shikhar481/sglang-images:8f8f96a6-src  # v0.3.4.post1
docker pull shikhar481/sglang-images:b77a02cd-src  # v0.3.4.post2

# v3 builds
docker pull shikhar481/sglang-images:9c745d07-v3   # v0.3.5.post2
docker pull shikhar481/sglang-images:10189d08-v3  # v0.3.6
```

---

*Last tested: 2026-01-20 on NVIDIA L40S (SM89)*
