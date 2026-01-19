# SGLang Docker Images - Rebuilt From Source

**Date:** 2026-01-19
**Updated:** 2026-01-19 (18 images now pushed)

## Summary

**18 images successfully rebuilt and pushed to `shikhar481/sglang-images`.**

All v0.1.x, v0.2.x, and v0.3.x commits have been fixed. Only v0.4.x commits remain blocked (require GPU to build sgl-kernel).

---

## All Pushed Images

### v0.1.x Images (7 commits)

| Commit | Version | PR | Subject | Status |
|--------|---------|-----|---------|--------|
| `2a754e57` | v0.1.17 | #579 | 2x prefill improvement | PUSHED |
| `9216b106` | v0.1.14 | #394 | 40% scheduler improvement | PUSHED |
| `09deb20d` | v0.1.14 | #420 | Logits memory optimization | PUSHED |
| `564a898a` | v0.1.20 | #619 | Mem indices optimization | PUSHED |
| `6a2941f4` | v0.1.20 | #625 | TP overhead improvement | PUSHED |
| `6f560c76` | v0.1.9 | #117 | First token latency | PUSHED |
| `ac971ff6` | v0.1.21 | #658 | stream_interval optimization | PUSHED |

### v0.2.x Images (1 commit)

| Commit | Version | PR | Subject | Status |
|--------|---------|-----|---------|--------|
| `62757db6` | v0.2.11 | #1010 | Cache disabled overhead | PUSHED |

### v0.3.x Images (10 commits)

| Commit | Version | PR | Subject | Status |
|--------|---------|-----|---------|--------|
| `ab4a83b2` | v0.3.0 | #1339 | Optimize schedule | PUSHED |
| `2854a5ea` | v0.3.1.post3 | #1496 | bench_latency fix | PUSHED |
| `9c064bf7` | v0.3.2 | #1587 | LoRA Step 1 | PUSHED |
| `c98e84c2` | v0.3.2 | #1589 | torch.argmax optimization | PUSHED |
| `e5db40dc` | v0.3.3.post1 | #1694 | ORJson serialization | PUSHED |
| `b1709305` | v0.3.3.post1 | #1697 | Radix tree optimization | PUSHED |
| `b77a02cd` | v0.3.4.post2 | #1752 | Grammar backends | PUSHED |
| `8f8f96a6` | v0.3.4.post1 | #1773 | stop_token_ids fix | PUSHED |
| `9c745d07` | v0.3.5.post2 | #2056 | xgrammar optimization | PUSHED |
| `10189d08` | v0.3.6 | #2171 | CPU affinity | PUSHED |

### v0.4.x Images (BLOCKED - Need GPU)

| Commit | Version | PR | Subject | Status |
|--------|---------|-----|---------|--------|
| `79961afa` | v0.4.6.post2 | #6077 | 21% FA3 faster | BLOCKED |
| `93470a14` | v0.4.5 | #5090 | FA3 optimization | BLOCKED |
| `3212c2ad` | v0.4.9.post4 | #6003 | 16% VLM faster | BLOCKED |
| (others) | v0.4.x | - | Various | BLOCKED |

**Reason:** v0.4.x commits require sgl-kernel built from source, which needs GPU for CUDA compilation.

---

## Build Process

### For v0.1.x - v0.3.x (No GPU Required)

All older commits were built using:
1. CUDA 12.1.1 base image
2. torch 2.3.0 or 2.4.0 depending on version
3. flashinfer from flashinfer.ai wheels
4. pyairports mock using airportsdata package
5. numpy<2 for outlines compatibility

**Common Dockerfile pattern:**
```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install PyTorch
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Pin numpy<2 for outlines compatibility
RUN pip install "numpy<2"

# Install flashinfer
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Copy and install sglang
COPY python/ /sglang/python/
RUN pip install -e "python[all]"

# Install uvloop
RUN pip install uvloop

# Fix pyairports (PyPI package is broken placeholder)
RUN pip install airportsdata && \
    pip uninstall -y pyairports || true && \
    rm -rf /usr/local/lib/python3.10/dist-packages/pyairports* && \
    mkdir -p /usr/local/lib/python3.10/dist-packages/pyairports && \
    echo "" > /usr/local/lib/python3.10/dist-packages/pyairports/__init__.py && \
    python3 -c "import airportsdata; d=airportsdata.load(); print('AIRPORT_LIST =', list(d.keys()))" > /usr/local/lib/python3.10/dist-packages/pyairports/airports.py
```

### For v0.4.x (GPU Required)

v0.4.x commits need sgl-kernel built from source:
```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.04-py3-min

RUN pip install torch --index-url https://download.pytorch.org/whl/cu124

# Build sgl-kernel from source (requires GPU for CUDA compilation)
RUN cd sgl-kernel && pip install . -v

RUN pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python
RUN pip install uvloop sentencepiece
```

---

## Key Fixes Applied

### 1. pyairports Placeholder Issue
The PyPI `pyairports` package was replaced with an empty placeholder (Author: "John Doe").
**Fix:** Create mock module using `airportsdata` package.

### 2. numpy 2.x Incompatibility
outlines versions have a numpy.lib.function_base import that doesn't exist in numpy 2.x.
**Fix:** Pin `numpy<2` before installing outlines.

### 3. Missing uvloop
Server startup fails without uvloop.
**Fix:** Explicitly install uvloop.

### 4. CUDA/torch Version Matching

| SGLang Version | vLLM Version | CUDA | torch | flashinfer |
|----------------|--------------|------|-------|------------|
| v0.1.9 | 0.3.3 | 12.1 | 2.3.0 | N/A |
| v0.1.14-0.1.21 | 0.4.1-0.5.1 | 12.1 | 2.3.0 | cu121/torch2.3 |
| v0.2.x | 0.5.4 | 12.1 | 2.3.0 | cu121/torch2.3 |
| v0.3.x | 0.5.5 | 12.1 | 2.4.0 | cu121/torch2.4 |
| v0.4.x | 0.6+ | 12.4 | 2.6.0 | cu124/torch2.6 |

---

## Verification

### Quick Test (No GPU)
```bash
docker pull shikhar481/sglang-images:<commit>
docker run --rm shikhar481/sglang-images:<commit> python3 -c "
import sglang; print(f'SGLang {sglang.__version__}')
import uvloop; print('uvloop: OK')
import outlines; print('outlines: OK')
"
```

### Full Test (GPU Required)
```bash
docker run --rm --gpus all shikhar481/sglang-images:<commit> python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import sglang; print(f'SGLang {sglang.__version__}')
import uvloop; print('uvloop: OK')
import outlines; print('outlines: OK')
"
```

### Run Benchmark
```bash
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN \
    shikhar481/sglang-images:<commit> \
    python3 -m sglang.bench_latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 1 --input-len 512 --output-len 64
```

---

## Pull Commands

```bash
# v0.1.x
docker pull shikhar481/sglang-images:2a754e57
docker pull shikhar481/sglang-images:9216b106
docker pull shikhar481/sglang-images:09deb20d
docker pull shikhar481/sglang-images:564a898a
docker pull shikhar481/sglang-images:6a2941f4
docker pull shikhar481/sglang-images:6f560c76
docker pull shikhar481/sglang-images:ac971ff6

# v0.2.x
docker pull shikhar481/sglang-images:62757db6

# v0.3.x
docker pull shikhar481/sglang-images:ab4a83b2
docker pull shikhar481/sglang-images:2854a5ea
docker pull shikhar481/sglang-images:9c064bf7
docker pull shikhar481/sglang-images:c98e84c2
docker pull shikhar481/sglang-images:e5db40dc
docker pull shikhar481/sglang-images:b1709305
docker pull shikhar481/sglang-images:b77a02cd
docker pull shikhar481/sglang-images:8f8f96a6
docker pull shikhar481/sglang-images:9c745d07
docker pull shikhar481/sglang-images:10189d08
```
