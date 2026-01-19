# SGLang Docker Images - Rebuilt From Source (v2)

**Date:** 2026-01-19
**Updated:** 2026-01-19 - v2 images with correct dependency versions

## Summary

**18 images successfully rebuilt and pushed to `shikhar481/sglang-images` with `-v2` suffix.**

All v0.1.x, v0.2.x, and v0.3.x commits have been fixed with correct vllm/huggingface-hub versions. Only v0.4.x commits remain blocked (require GPU to build sgl-kernel).

---

## Critical Fixes in v2 Images

The original images failed GPU testing due to dependency version mismatches. The v2 images fix these issues:

| Version | Original Issue | v2 Fix |
|---------|---------------|--------|
| v0.1.x | vllm 0.4.1 API incompatibility | Use vllm 0.3.3 (v0.1.14+) or vllm 0.2.7 (v0.1.9) |
| v0.2.x | Missing flashinfer | Add flashinfer from flashinfer.ai wheels |
| v0.3.x | huggingface-hub >1.0 | Pin huggingface-hub<1.0 |

---

## All Pushed v2 Images

### v0.1.x Images (7 commits)

| Commit | Version | vllm | PR | Subject | Tag |
|--------|---------|------|-----|---------|-----|
| `2a754e57` | v0.1.17 | 0.3.3 | #579 | 2x prefill improvement | `2a754e57-v2` |
| `9216b106` | v0.1.14 | 0.3.3 | #394 | 40% scheduler improvement | `9216b106-v2` |
| `09deb20d` | v0.1.14 | 0.3.3 | #420 | Logits memory optimization | `09deb20d-v2` |
| `564a898a` | v0.1.20 | 0.3.3 | #619 | Mem indices optimization | `564a898a-v2` |
| `6a2941f4` | v0.1.20 | 0.3.3 | #625 | TP overhead improvement | `6a2941f4-v2` |
| `6f560c76` | v0.1.9 | 0.2.7 | #117 | First token latency | `6f560c76-v2` |
| `ac971ff6` | v0.1.21 | 0.3.3 | #658 | stream_interval optimization | `ac971ff6-v2` |

### v0.2.x Images (1 commit)

| Commit | Version | vllm | PR | Subject | Tag |
|--------|---------|------|-----|---------|-----|
| `62757db6` | v0.2.11 | 0.5.4 | #1010 | Cache disabled overhead | `62757db6-v2` |

### v0.3.x Images (10 commits)

| Commit | Version | vllm | PR | Subject | Tag |
|--------|---------|------|-----|---------|-----|
| `ab4a83b2` | v0.3.0 | 0.5.5 | #1339 | Optimize schedule | `ab4a83b2-v2` |
| `2854a5ea` | v0.3.1.post3 | 0.5.5 | #1496 | bench_latency fix | `2854a5ea-v2` |
| `9c064bf7` | v0.3.2 | 0.5.5 | #1587 | LoRA Step 1 | `9c064bf7-v2` |
| `c98e84c2` | v0.3.2 | 0.5.5 | #1589 | torch.argmax optimization | `c98e84c2-v2` |
| `e5db40dc` | v0.3.3.post1 | 0.5.5 | #1694 | ORJson serialization | `e5db40dc-v2` |
| `b1709305` | v0.3.3.post1 | 0.5.5 | #1697 | Radix tree optimization | `b1709305-v2` |
| `b77a02cd` | v0.3.4.post2 | 0.5.5 | #1752 | Grammar backends | `b77a02cd-v2` |
| `8f8f96a6` | v0.3.4.post1 | 0.5.5 | #1773 | stop_token_ids fix | `8f8f96a6-v2` |
| `9c745d07` | v0.3.5.post2 | 0.5.5 | #2056 | xgrammar optimization | `9c745d07-v2` |
| `10189d08` | v0.3.6 | 0.5.5 | #2171 | CPU affinity | `10189d08-v2` |

### v0.4.x Images (BLOCKED - Need GPU)

| Commit | Version | PR | Subject | Status |
|--------|---------|-----|---------|--------|
| `79961afa` | v0.4.6.post2 | #6077 | 21% FA3 faster | BLOCKED |
| `93470a14` | v0.4.5 | #5090 | FA3 optimization | BLOCKED |
| `3212c2ad` | v0.4.9.post4 | #6003 | 16% VLM faster | BLOCKED |
| (others) | v0.4.x | - | Various | BLOCKED |

**Reason:** v0.4.x commits require sgl-kernel built from source, which needs GPU for CUDA compilation.

---

## Build Process Details

### v0.1.x Dockerfile (vllm 0.3.3)

```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install torch 2.1.2 (compatible with vllm 0.3.3)
RUN pip3 install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Pin numpy<2 for outlines compatibility
RUN pip3 install "numpy<2"

# Pin transformers to version compatible with torch 2.1.2 and vllm 0.3.3
RUN pip3 install "transformers>=4.38.0,<4.40.0"

# Install vllm 0.3.3 dependencies
RUN pip3 install accelerate sentencepiece ray "pynvml==11.5.0" \
    prometheus-client ninja cupy-cuda12x==12.1.0
RUN pip3 install xformers==0.0.23.post1

# Install vllm 0.3.3
RUN pip3 install vllm==0.3.3 --no-deps

# Install sglang and dependencies
RUN pip3 install aiohttp fastapi psutil rpyc uvloop uvicorn pyzmq interegular pydantic pillow
RUN pip3 install "outlines>=0.0.27"
```

### v0.1.9 Dockerfile (vllm 0.2.7)

v0.1.9 uses different dependencies (lark, numba instead of outlines):

```dockerfile
# Install vllm 0.2.7
RUN pip3 install vllm==0.2.7

# Install sglang dependencies for v0.1.9
RUN pip3 install aiohttp fastapi psutil rpyc uvloop uvicorn pyzmq interegular \
    lark numba pydantic diskcache cloudpickle pillow
```

### v0.2.x Dockerfile (vllm 0.5.4)

```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install torch 2.3.0
RUN pip3 install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Pin huggingface-hub to <1.0
RUN pip3 install "huggingface-hub>=0.25.0,<1.0"

# Install flashinfer for torch 2.3
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

# Install vllm 0.5.4
RUN pip3 install vllm==0.5.4
```

### v0.3.x Dockerfile (vllm 0.5.5)

```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install torch 2.4.0
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Pin huggingface-hub to <1.0 (v0.3.x requirement)
RUN pip3 install "huggingface-hub>=0.25.0,<1.0"

# Pin transformers to compatible version (with AutoProcessor)
RUN pip3 install "transformers>=4.45.0,<4.50.0"

# Install flashinfer for torch 2.4
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install vllm 0.5.5
RUN pip3 install vllm==0.5.5
```

---

## Dependency Matrix

| SGLang Ver | vLLM | CUDA | torch | transformers | huggingface-hub | flashinfer |
|------------|------|------|-------|--------------|-----------------|------------|
| v0.1.9 | 0.2.7 | 12.1 | 2.1.2 | <4.40.0 | any | N/A |
| v0.1.14-0.1.21 | 0.3.3 | 12.1 | 2.1.2 | <4.40.0 | any | N/A |
| v0.2.x | 0.5.4 | 12.1 | 2.3.0 | <4.45.0 | <1.0 | cu121/torch2.3 |
| v0.3.x | 0.5.5 | 12.1 | 2.4.0 | <4.50.0 | <1.0 | cu121/torch2.4 |
| v0.4.x | 0.6+ | 12.4 | 2.6.0 | latest | latest | cu124/torch2.6 |

---

## Pull Commands (v2 Images)

```bash
# v0.1.x
docker pull shikhar481/sglang-images:2a754e57-v2
docker pull shikhar481/sglang-images:9216b106-v2
docker pull shikhar481/sglang-images:09deb20d-v2
docker pull shikhar481/sglang-images:564a898a-v2
docker pull shikhar481/sglang-images:6a2941f4-v2
docker pull shikhar481/sglang-images:6f560c76-v2
docker pull shikhar481/sglang-images:ac971ff6-v2

# v0.2.x
docker pull shikhar481/sglang-images:62757db6-v2

# v0.3.x
docker pull shikhar481/sglang-images:ab4a83b2-v2
docker pull shikhar481/sglang-images:2854a5ea-v2
docker pull shikhar481/sglang-images:9c064bf7-v2
docker pull shikhar481/sglang-images:c98e84c2-v2
docker pull shikhar481/sglang-images:e5db40dc-v2
docker pull shikhar481/sglang-images:b1709305-v2
docker pull shikhar481/sglang-images:b77a02cd-v2
docker pull shikhar481/sglang-images:8f8f96a6-v2
docker pull shikhar481/sglang-images:9c745d07-v2
docker pull shikhar481/sglang-images:10189d08-v2
```

---

## Verification Commands

### Quick Test (No GPU)
```bash
docker pull shikhar481/sglang-images:<commit>-v2
docker run --rm shikhar481/sglang-images:<commit>-v2 python3 -c "
import sglang; print(f'SGLang {sglang.__version__}')
import uvloop; print('uvloop: OK')
import vllm; print(f'vllm {vllm.__version__}')
"
```

### Full Test (GPU Required)
```bash
docker run --rm --gpus all shikhar481/sglang-images:<commit>-v2 python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import sglang; print(f'SGLang {sglang.__version__}')
import uvloop; print('uvloop: OK')
import vllm; print(f'vllm {vllm.__version__}')
try:
    import flashinfer; print('flashinfer: OK')
except: print('flashinfer: N/A')
"
```

### Server Test
```bash
docker run -d --rm --gpus all -p 30000:30000 \
    shikhar481/sglang-images:<commit>-v2 \
    python3 -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 30000 --host 0.0.0.0
```

---

*Updated: 2026-01-19 14:15 UTC*
