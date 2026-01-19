# SGLang Docker Build from Source

This directory contains Dockerfiles for building SGLang from source at specific commits.
These are needed because the pre-built images had incorrect code versions.

## Why Build from Source?

Testing revealed that the v3 images available on DockerHub were NOT built from the correct git commits.
For example, `ab4a83b2-v3` contained `vocab_parallel_embedding.py` which doesn't exist at commit ab4a83b2.

## Dockerfiles

### Dockerfile.v0.3.0 (commit ab4a83b2)
- **Commit date:** 2024-09-05
- **Dependencies:** torch 2.4.0, vllm 0.5.5, flashinfer 0.1.6
- **Patches:** None required

### Dockerfile.v0.2.11 (commit 62757db6)
- **Commit date:** 2024-08-09
- **Dependencies:** torch 2.3.0, vllm 0.5.4, flashinfer 0.1.6
- **Patches:** Event loop fix in detokenizer_manager.py
  - Problem: `RuntimeError: There is no current event loop in thread 'MainThread'`
  - Fix: Replace `asyncio.get_event_loop()` with `asyncio.new_event_loop()`

### Dockerfile.v0.3.2 (commit c98e84c2)
- **Commit date:** 2024-10-06
- **Dependencies:** torch 2.4.0, vllm 0.5.5, flashinfer 0.1.6
- **Patches:** Model registry assertion fix in model_runner.py
  - Problem: `AssertionError: Duplicated model implementation for LlamaForCausalLM`
  - Fix: Convert assertion to debug log (allow SGLang to override vllm model implementations)

## Build Commands

```bash
# Build v0.3.0
docker build -f dockerfiles/Dockerfile.v0.3.0 -t sglang:ab4a83b2-src .

# Build v0.2.11
docker build -f dockerfiles/Dockerfile.v0.2.11 -t sglang:62757db6-src .

# Build v0.3.2
docker build -f dockerfiles/Dockerfile.v0.3.2 -t sglang:c98e84c2-src .
```

## Test Commands

### 1. Import Test
```bash
docker run --rm --gpus all sglang:ab4a83b2-src \
    python3 -c "from sglang.srt.server import launch_server; print('Import OK')"
```

### 2. Server Startup Test
```bash
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN \
    sglang:ab4a83b2-src \
    python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --host 0.0.0.0 --port 30000 &

# Wait for startup and test
sleep 60
curl http://localhost:30000/health
```

## Dependency Matrix

| Version | Commit | vLLM | Torch | FlashInfer |
|---------|--------|------|-------|------------|
| v0.2.11 | 62757db6 | 0.5.4 | 2.3.0 | 0.1.6 (torch2.3) |
| v0.3.0 | ab4a83b2 | 0.5.5 | 2.4.0 | 0.1.6 (torch2.4) |
| v0.3.2 | c98e84c2 | 0.5.5 | 2.4.0 | 0.1.6 (torch2.4) |

## Pushing to DockerHub

After building and testing, push to DockerHub:

```bash
# Tag with your DockerHub username
docker tag sglang:ab4a83b2-src YOUR_USERNAME/sglang:ab4a83b2-src
docker push YOUR_USERNAME/sglang:ab4a83b2-src
```

## Known Issues

1. **PYTHONPATH required:** The sglang package at these commits doesn't install correctly with pip due to setuptools package discovery issues. PYTHONPATH is set in the Dockerfiles to work around this.

2. **Event loop (v0.2.x):** Older versions have asyncio event loop issues when using uvloop. The Dockerfiles apply a patch.

3. **Model registry (v0.3.x):** Later v0.3.x versions have assertion failures when both SGLang and vllm define the same model class. The Dockerfiles apply a patch to allow overrides.
