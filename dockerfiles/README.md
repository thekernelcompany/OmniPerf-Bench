# SGLang Docker Build from Source

This directory contains Dockerfiles for building SGLang from source at specific commits.
These are needed because the pre-built images had incorrect code versions.

## Why Build from Source?

Testing revealed that the v3 images available on DockerHub were NOT built from the correct git commits.
For example, `ab4a83b2-v3` contained `vocab_parallel_embedding.py` which doesn't exist at commit ab4a83b2.

## Dockerfiles Summary

| Dockerfile | Commit | Version | Patches |
|------------|--------|---------|---------|
| `Dockerfile.v0.2.11` | 62757db6 | v0.2.11 | Event loop fix |
| `Dockerfile.v0.3.0` | ab4a83b2 | v0.3.0 | None |
| `Dockerfile.v0.3.1.post3` | 2854a5ea | v0.3.1.post3 | Model registry |
| `Dockerfile.v0.3.2` | c98e84c2 | v0.3.2 | Model registry |
| `Dockerfile.v0.3.2.9c064bf7` | 9c064bf7 | v0.3.2 | Model registry |
| `Dockerfile.v0.3.3.post1.e5db40dc` | e5db40dc | v0.3.3.post1 | Model registry + orjson |
| `Dockerfile.v0.3.3.post1.b1709305` | b1709305 | v0.3.3.post1 | Model registry + orjson |
| `Dockerfile.v0.3.4.post2` | b77a02cd | v0.3.4.post2 | Model registry + orjson |
| `Dockerfile.v0.3.4.post1` | 8f8f96a6 | v0.3.4.post1 | Model registry + orjson |

## Dockerfiles Detail

### Dockerfile.v0.2.11 (commit 62757db6)
- **Version:** v0.2.11
- **Dependencies:** torch 2.3.0, vllm 0.5.4, flashinfer 0.1.6
- **Patches:** Event loop fix in detokenizer_manager.py
  - Problem: `RuntimeError: There is no current event loop in thread 'MainThread'`
  - Fix: Replace `asyncio.get_event_loop()` with `asyncio.new_event_loop()`

### Dockerfile.v0.3.0 (commit ab4a83b2)
- **Version:** v0.3.0
- **Dependencies:** torch 2.4.0, vllm 0.5.5, flashinfer 0.1.6
- **Patches:** None required

### Dockerfile.v0.3.1.post3 (commit 2854a5ea)
- **Version:** v0.3.1.post3
- **Dependencies:** torch 2.4.0, vllm 0.5.5, flashinfer 0.1.6
- **Patches:** Model registry assertion fix

### Dockerfile.v0.3.2 (commit c98e84c2)
- **Version:** v0.3.2
- **Dependencies:** torch 2.4.0, vllm 0.5.5, flashinfer 0.1.6
- **Patches:** Model registry assertion fix
  - Problem: `AssertionError: Duplicated model implementation for LlamaForCausalLM`
  - Fix: Convert assertion to debug log

### Dockerfile.v0.3.2.9c064bf7 (commit 9c064bf7)
- **Version:** v0.3.2 (different PR)
- **Dependencies:** torch 2.4.0, vllm 0.5.5, flashinfer 0.1.6
- **Patches:** Model registry assertion fix

### Dockerfile.v0.3.3.post1.e5db40dc (commit e5db40dc)
- **Version:** v0.3.3.post1
- **PR:** #1694 - ORJson serialization
- **Dependencies:** torch 2.4.0, vllm 0.5.5, flashinfer 0.1.6, orjson
- **Patches:** Model registry assertion fix

### Dockerfile.v0.3.3.post1.b1709305 (commit b1709305)
- **Version:** v0.3.3.post1
- **PR:** #1697 - Radix tree optimization
- **Dependencies:** torch 2.4.0, vllm 0.5.5, flashinfer 0.1.6, orjson
- **Patches:** Model registry assertion fix

### Dockerfile.v0.3.4.post2 (commit b77a02cd)
- **Version:** v0.3.4.post2
- **PR:** #1752 - Grammar backends
- **Dependencies:** torch 2.4.0, vllm 0.5.5, flashinfer 0.1.6, orjson
- **Patches:** Model registry assertion fix

### Dockerfile.v0.3.4.post1 (commit 8f8f96a6)
- **Version:** v0.3.4.post1
- **PR:** #1773 - stop_token_ids fix
- **Dependencies:** torch 2.4.0, vllm 0.5.5, flashinfer 0.1.6, orjson
- **Patches:** Model registry assertion fix

## Build Commands

```bash
# Build all images
cd /path/to/OmniPerf-Bench

# v0.2.11 (62757db6)
docker build -f dockerfiles/Dockerfile.v0.2.11 -t sglang:62757db6-src .

# v0.3.0 (ab4a83b2)
docker build -f dockerfiles/Dockerfile.v0.3.0 -t sglang:ab4a83b2-src .

# v0.3.1.post3 (2854a5ea)
docker build -f dockerfiles/Dockerfile.v0.3.1.post3 -t sglang:2854a5ea-src .

# v0.3.2 (c98e84c2)
docker build -f dockerfiles/Dockerfile.v0.3.2 -t sglang:c98e84c2-src .

# v0.3.2 (9c064bf7)
docker build -f dockerfiles/Dockerfile.v0.3.2.9c064bf7 -t sglang:9c064bf7-src .

# v0.3.3.post1 (e5db40dc)
docker build -f dockerfiles/Dockerfile.v0.3.3.post1.e5db40dc -t sglang:e5db40dc-src .

# v0.3.3.post1 (b1709305)
docker build -f dockerfiles/Dockerfile.v0.3.3.post1.b1709305 -t sglang:b1709305-src .

# v0.3.4.post2 (b77a02cd)
docker build -f dockerfiles/Dockerfile.v0.3.4.post2 -t sglang:b77a02cd-src .

# v0.3.4.post1 (8f8f96a6)
docker build -f dockerfiles/Dockerfile.v0.3.4.post1 -t sglang:8f8f96a6-src .
```

## Test Commands

### 1. Quick Test (Server Module Import)
```bash
docker run --rm --gpus all sglang:<commit>-src \
    python3 -c "from sglang.srt.server import launch_server; print('Import OK')"
```

### 2. Full Server Startup Test (CRITICAL - Use This)
```bash
# Start server
docker run -d --rm --gpus all -p 30000:30000 --name test_server \
    sglang:<commit>-src \
    python3 -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 30000 --host 0.0.0.0

# Wait for startup
sleep 60

# Verify health
curl http://localhost:30000/health

# Check logs if issues
docker logs test_server

# Cleanup
docker kill test_server
```

## Dependency Matrix

| Version | Commit | vLLM | Torch | FlashInfer | orjson |
|---------|--------|------|-------|------------|--------|
| v0.2.11 | 62757db6 | 0.5.4 | 2.3.0 | 0.1.6 (torch2.3) | No |
| v0.3.0 | ab4a83b2 | 0.5.5 | 2.4.0 | 0.1.6 (torch2.4) | No |
| v0.3.1.post3 | 2854a5ea | 0.5.5 | 2.4.0 | 0.1.6 (torch2.4) | No |
| v0.3.2 | c98e84c2 | 0.5.5 | 2.4.0 | 0.1.6 (torch2.4) | No |
| v0.3.2 | 9c064bf7 | 0.5.5 | 2.4.0 | 0.1.6 (torch2.4) | No |
| v0.3.3.post1 | e5db40dc | 0.5.5 | 2.4.0 | 0.1.6 (torch2.4) | Yes |
| v0.3.3.post1 | b1709305 | 0.5.5 | 2.4.0 | 0.1.6 (torch2.4) | Yes |
| v0.3.4.post2 | b77a02cd | 0.5.5 | 2.4.0 | 0.1.6 (torch2.4) | Yes |
| v0.3.4.post1 | 8f8f96a6 | 0.5.5 | 2.4.0 | 0.1.6 (torch2.4) | Yes |

## Pushing to DockerHub

After building and testing, push to DockerHub:

```bash
# Tag with your DockerHub username
docker tag sglang:ab4a83b2-src YOUR_USERNAME/sglang:ab4a83b2-src
docker push YOUR_USERNAME/sglang:ab4a83b2-src

# Or batch push all
for commit in 62757db6 ab4a83b2 2854a5ea c98e84c2 9c064bf7 e5db40dc b1709305 b77a02cd 8f8f96a6; do
    docker tag sglang:${commit}-src YOUR_USERNAME/sglang:${commit}-src
    docker push YOUR_USERNAME/sglang:${commit}-src
done
```

## Applied Patches

### 1. Event Loop Fix (v0.2.x)
```python
# Before: asyncio.get_event_loop() fails in multiprocessing
loop = asyncio.get_event_loop()

# After: Create new event loop
loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
```

### 2. Model Registry Fix (v0.3.x)
```python
# Before: Assertion fails when both SGLang and vllm register same model
assert (arch not in MODEL_REGISTRY), f"Duplicated model implementation for {arch}"

# After: Allow override with debug log
if not (arch not in MODEL_REGISTRY):
    logger.debug(f"Overriding model implementation for {arch}")
```

## Known Issues

1. **PYTHONPATH required:** The sglang package at these commits doesn't install correctly with pip due to setuptools package discovery issues. PYTHONPATH is set in the Dockerfiles to work around this.

2. **Event loop (v0.2.x):** Older versions have asyncio event loop issues when using uvloop. The Dockerfiles apply a patch.

3. **Model registry (v0.3.x):** v0.3.0-v0.3.4 have assertion failures when both SGLang and vllm define the same model class. The Dockerfiles apply a patch to disable these assertions.

4. **pyairports mock module:** The outlines package requires pyairports but the PyPI package is a placeholder. The Dockerfiles create a mock module using airportsdata.

5. **v0.3.5+ works without patches:** 9c745d07 (v0.3.5.post2) and 10189d08 (v0.3.6) work with the v3 images on DockerHub. No Dockerfiles needed.

## Build Status (2026-01-20)

All 9 images built and pushed to DockerHub with `-src` suffix. All pass server module import test.

| Image | Status |
|-------|--------|
| `62757db6-src` | Built, pushed, import test PASS |
| `ab4a83b2-src` | Built, pushed, import test PASS |
| `2854a5ea-src` | Built, pushed, import test PASS |
| `c98e84c2-src` | Built, pushed, import test PASS |
| `9c064bf7-src` | Built, pushed, import test PASS |
| `e5db40dc-src` | Built, pushed, import test PASS |
| `b1709305-src` | Built, pushed, import test PASS |
| `b77a02cd-src` | Built, pushed, import test PASS |
| `8f8f96a6-src` | Built, pushed, import test PASS |

---

## v0.4.x Dockerfiles (NEW - 2026-01-20)

v0.4.x SGLang commits require building `sgl-kernel` from source. The key insight is that
sgl-kernel can be built on CPU-only machines using CUDA stubs.

### CUDA Stubs Approach

The `nvidia/cuda:*-devel` images include CUDA stubs at:
```
/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so
```

We symlink this to `/usr/lib/x86_64-linux-gnu/libcuda.so` to allow NVCC compilation without a GPU.

### v0.4.x Dockerfiles Summary

| Dockerfile | Commit | Version | PR | Subject |
|------------|--------|---------|-----|---------|
| `Dockerfile.v0.4.x.template` | - | - | - | Template for new builds |
| `Dockerfile.v0.4.6.79961afa` | 79961afa | v0.4.6.post2 | #6077 | FA3 pad optimization |
| `Dockerfile.v0.4.5.93470a14` | 93470a14 | v0.4.5 | #5090 | FA3 refactor and optimize |
| `Dockerfile.v0.4.9.3212c2ad` | 3212c2ad | v0.4.9.post4 | #6003 | VLM tensor transport |
| `Dockerfile.v0.4.1.2bd18e2d` | 2bd18e2d | v0.4.1.post6 | #2901 | Memory pool optimization |
| `Dockerfile.v0.4.4.d1112d85` | d1112d85 | v0.4.4.post1 | #2797 | input_embeds endpoint |

### Building v0.4.x Images

```bash
# Build v0.4.6 (79961afa) - ~30-60 min build time
docker build -f dockerfiles/Dockerfile.v0.4.6.79961afa -t sglang:79961afa-src .

# Build v0.4.5 (93470a14)
docker build -f dockerfiles/Dockerfile.v0.4.5.93470a14 -t sglang:93470a14-src .

# Build v0.4.9 (3212c2ad)
docker build -f dockerfiles/Dockerfile.v0.4.9.3212c2ad -t sglang:3212c2ad-src .

# Build v0.4.1 (2bd18e2d)
docker build -f dockerfiles/Dockerfile.v0.4.1.2bd18e2d -t sglang:2bd18e2d-src .

# Build v0.4.4 (d1112d85)
docker build -f dockerfiles/Dockerfile.v0.4.4.d1112d85 -t sglang:d1112d85-src .
```

### v0.4.x Dependency Matrix

| Version | Commit | Torch | FlashInfer | sgl-kernel |
|---------|--------|-------|------------|------------|
| v0.4.1 | 2bd18e2d | 2.5.0 | 0.2.1 | Built from source |
| v0.4.4 | d1112d85 | 2.5.0 | 0.2.1 | Built from source |
| v0.4.5 | 93470a14 | 2.5.0 | 0.2.1 | Built from source |
| v0.4.6 | 79961afa | 2.5.0 | 0.2.1 | Built from source |
| v0.4.9 | 3212c2ad | 2.5.0 | 0.2.1 | Built from source |

### Using the Template

To create a Dockerfile for a new v0.4.x commit:

```bash
# Copy template
cp Dockerfile.v0.4.x.template Dockerfile.v0.4.X.COMMIT_SHORT

# Edit and replace:
# - COMMIT_SHORT (e.g., abcd1234)
# - git checkout line
# - LABEL fields

# Build
docker build -f Dockerfile.v0.4.X.COMMIT_SHORT -t sglang:COMMIT_SHORT-src .
```

### v0.4.x Build Notes

- **Build time:** 30-60 minutes (sgl-kernel compilation is slow)
- **No GPU required:** Uses CUDA stubs for compilation
- **GPU required for runtime:** Needs NVIDIA GPU to actually run SGLang
- **Build parallelism:** Adjust `CMAKE_BUILD_PARALLEL_LEVEL` if OOM occurs
