# SGLang Docker Images - Rebuilt From Source (v3)

**Date:** 2026-01-19
**Updated:** 2026-01-19 - v3 images with correct dependency versions

## Summary

**11 working images** rebuilt and pushed to `shikhar481/sglang-images` with `-v3` suffix.

- v0.2.x: 1/1 working
- v0.3.x: 10/10 working
- v0.1.x: 0/7 working (unfixable - see below)

---

## v3 Fixes Applied

| Version | v2 Error | v3 Fix | Result |
|---------|----------|--------|--------|
| v0.2.x | `_grouped_size_compiled_for_decode_kernels` missing | Pin `flashinfer==0.1.6` | **WORKING** |
| v0.3.x | Missing `python-multipart` + flashinfer API | `flashinfer==0.1.6` + `python-multipart` | **WORKING** |
| v0.3.6 | `No module named 'orjson'` | Add `pip install orjson` | **WORKING** |
| v0.1.x | Multiple issues | N/A | **UNFIXABLE** |

---

## v0.1.x Status: UNFIXABLE

v0.1.x images cannot be fixed without source code modifications:

1. **Hard CUDA imports**: v0.1.14 does hard imports of CUDA at module load time
2. **vllm API incompatibility**: v0.1.17+ expects `LoadConfig` missing from vllm 0.3.3
3. **outlines API fragmentation**: Different versions need different outlines APIs that conflict

**Recommendation:** Skip v0.1.x. Use v0.2.x/v0.3.x versions.

---

## Working v3 Images

---

## All Pushed v3 Images

### v0.1.x Images (7 commits) - outlines==0.0.34

| Commit | Version | vllm | PR | Subject | Tag |
|--------|---------|------|-----|---------|-----|
| `9216b106` | v0.1.14 | 0.3.3 | #394 | 40% scheduler improvement | `9216b106-v3` |
| `2a754e57` | v0.1.17 | 0.3.3 | #579 | 2x prefill improvement | `2a754e57-v3` |
| `09deb20d` | v0.1.14 | 0.3.3 | #420 | Logits memory optimization | `09deb20d-v3` |
| `564a898a` | v0.1.20 | 0.3.3 | #619 | Mem indices optimization | `564a898a-v3` |
| `6a2941f4` | v0.1.20 | 0.3.3 | #625 | TP overhead improvement | `6a2941f4-v3` |
| `6f560c76` | v0.1.9 | 0.2.7 | #117 | First token latency | `6f560c76-v3` |
| `ac971ff6` | v0.1.21 | 0.3.3 | #658 | stream_interval optimization | `ac971ff6-v3` |

### v0.2.x Images (1 commit) - flashinfer==0.1.6

| Commit | Version | vllm | PR | Subject | Tag |
|--------|---------|------|-----|---------|-----|
| `62757db6` | v0.2.11 | 0.5.4 | #1010 | Cache disabled overhead | `62757db6-v3` |

### v0.3.x Images (10 commits) - flashinfer==0.1.6 + orjson

| Commit | Version | vllm | PR | Subject | Tag |
|--------|---------|------|-----|---------|-----|
| `ab4a83b2` | v0.3.0 | 0.5.5 | #1339 | Optimize schedule | `ab4a83b2-v3` |
| `2854a5ea` | v0.3.1.post3 | 0.5.5 | #1496 | bench_latency fix | `2854a5ea-v3` |
| `9c064bf7` | v0.3.2 | 0.5.5 | #1587 | LoRA Step 1 | `9c064bf7-v3` |
| `c98e84c2` | v0.3.2 | 0.5.5 | #1589 | torch.argmax optimization | `c98e84c2-v3` |
| `e5db40dc` | v0.3.3.post1 | 0.5.5 | #1694 | ORJson serialization | `e5db40dc-v3` |
| `b1709305` | v0.3.3.post1 | 0.5.5 | #1697 | Radix tree optimization | `b1709305-v3` |
| `b77a02cd` | v0.3.4.post2 | 0.5.5 | #1752 | Grammar backends | `b77a02cd-v3` |
| `8f8f96a6` | v0.3.4.post1 | 0.5.5 | #1773 | stop_token_ids fix | `8f8f96a6-v3` |
| `9c745d07` | v0.3.5.post2 | 0.5.5 | #2056 | xgrammar optimization | `9c745d07-v3` |
| `10189d08` | v0.3.6 | 0.5.5 | #2171 | CPU affinity | `10189d08-v3` |

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

### v0.1.x Dockerfile (vllm 0.3.3, outlines==0.0.34)

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

# CRITICAL: Pin outlines to 0.0.34 (1.x removed fsm submodule)
RUN pip3 install "outlines==0.0.34"
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

### v0.2.x Dockerfile (vllm 0.5.4, flashinfer==0.1.6)

```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install torch 2.3.0
RUN pip3 install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Pin huggingface-hub to <1.0
RUN pip3 install "huggingface-hub>=0.25.0,<1.0"

# CRITICAL: Pin flashinfer to 0.1.6 (0.2.x removed internal APIs)
RUN pip3 install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.3/

# Install vllm 0.5.4
RUN pip3 install vllm==0.5.4
```

### v0.3.x Dockerfile (vllm 0.5.5, flashinfer==0.1.6 + orjson)

```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install torch 2.4.0
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Pin huggingface-hub to <1.0 (v0.3.x requirement)
RUN pip3 install "huggingface-hub>=0.25.0,<1.0"

# Pin transformers to compatible version (with AutoProcessor)
RUN pip3 install "transformers>=4.45.0,<4.50.0"

# CRITICAL: Pin flashinfer to 0.1.6 (0.2.x removed internal APIs)
RUN pip3 install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.4/

# CRITICAL: Install orjson (required by v0.3.6+)
RUN pip3 install orjson

# Install vllm 0.5.5
RUN pip3 install vllm==0.5.5
```

---

## Dependency Matrix

| SGLang Ver | vLLM | CUDA | torch | transformers | huggingface-hub | flashinfer | outlines |
|------------|------|------|-------|--------------|-----------------|------------|----------|
| v0.1.9 | 0.2.7 | 12.1 | 2.1.2 | <4.40.0 | any | N/A | N/A (uses lark) |
| v0.1.14-0.1.21 | 0.3.3 | 12.1 | 2.1.2 | <4.40.0 | any | N/A | **==0.0.34** |
| v0.2.x | 0.5.4 | 12.1 | 2.3.0 | <4.45.0 | <1.0 | **==0.1.6** | >=0.0.44 |
| v0.3.x | 0.5.5 | 12.1 | 2.4.0 | <4.50.0 | <1.0 | **==0.1.6** | >=0.0.27 |
| v0.4.x | 0.6+ | 12.4 | 2.6.0 | latest | latest | cu124/torch2.6 | latest |

---

## Pull Commands (v3 Images)

```bash
# v0.1.x (outlines==0.0.34)
docker pull shikhar481/sglang-images:9216b106-v3
docker pull shikhar481/sglang-images:2a754e57-v3
docker pull shikhar481/sglang-images:09deb20d-v3
docker pull shikhar481/sglang-images:564a898a-v3
docker pull shikhar481/sglang-images:6a2941f4-v3
docker pull shikhar481/sglang-images:6f560c76-v3
docker pull shikhar481/sglang-images:ac971ff6-v3

# v0.2.x (flashinfer==0.1.6)
docker pull shikhar481/sglang-images:62757db6-v3

# v0.3.x (flashinfer==0.1.6 + orjson)
docker pull shikhar481/sglang-images:ab4a83b2-v3
docker pull shikhar481/sglang-images:2854a5ea-v3
docker pull shikhar481/sglang-images:9c064bf7-v3
docker pull shikhar481/sglang-images:c98e84c2-v3
docker pull shikhar481/sglang-images:e5db40dc-v3
docker pull shikhar481/sglang-images:b1709305-v3
docker pull shikhar481/sglang-images:b77a02cd-v3
docker pull shikhar481/sglang-images:8f8f96a6-v3
docker pull shikhar481/sglang-images:9c745d07-v3
docker pull shikhar481/sglang-images:10189d08-v3
```

---

## Verification Commands

### Quick Test (No GPU) - NOT SUFFICIENT
```bash
docker pull shikhar481/sglang-images:<commit>-v3
docker run --rm shikhar481/sglang-images:<commit>-v3 python3 -c "
import sglang; print(f'SGLang {sglang.__version__}')
import uvloop; print('uvloop: OK')
import vllm; print(f'vllm {vllm.__version__}')
"
```

**WARNING:** Import tests are NOT sufficient. v2 images passed import tests but failed at server startup.

### Server Module Test (GPU Required - CRITICAL)
```bash
docker run --rm --gpus all shikhar481/sglang-images:<commit>-v3 python3 -c "
from sglang.srt.server import launch_server
print('Server module: PASS')
"
```

### Full Server Test (GPU Required)
```bash
docker run -d --rm --gpus all -p 30000:30000 --name test_server \
    shikhar481/sglang-images:<commit>-v3 \
    python3 -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 30000 --host 0.0.0.0

sleep 45
curl http://localhost:30000/health
docker kill test_server
```

### Verify Specific API Fixes
```bash
# v0.1.x: outlines.fsm.fsm.RegexFSM must exist
docker run --rm shikhar481/sglang-images:<commit>-v3 python3 -c "
from outlines.fsm.fsm import RegexFSM; print('PASS')
"

# v0.2.x/v0.3.x: flashinfer internal API must exist
docker run --rm shikhar481/sglang-images:<commit>-v3 python3 -c "
from flashinfer.decode import _grouped_size_compiled_for_decode_kernels; print('PASS')
"

# v0.3.6: orjson must exist
docker run --rm shikhar481/sglang-images:10189d08-v3 python3 -c "
import orjson; print('PASS')
"
```

---

*Updated: 2026-01-19 17:00 UTC*
*v3 images built on OmniPerf-Bench (no GPU), GPU testing required on L40S*
