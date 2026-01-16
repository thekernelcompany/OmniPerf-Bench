# Docker Image Fix for 99abb8b6

## Image to Fix

```
ayushnangia16/nvidia-vllm-docker:99abb8b650c66664cdc84d815b7f306f33bd9881
```

**Commit:** `99abb8b650c66664cdc84d815b7f306f33bd9881`
**PR:** [V1][Spec Decode] Optimize Rejection Sampler with Triton Kernels

---

## What's Wrong

The Docker image is missing the `vllm.benchmarks` module. When we try to run benchmarks:

1. **vLLM CLI fails:**
   ```
   ModuleNotFoundError: No module named 'vllm.benchmarks'
   ```

2. **Cloned benchmark scripts are deprecated:**
   ```
   DEPRECATED: This script has been moved to the vLLM CLI.
   Please use: vllm bench latency
   ```

3. **The CLI can't run because it needs `vllm.benchmarks`** - circular problem.

### Root Cause

The image was likely built with:
```bash
pip install vllm
```

This installs only the core vLLM package, **not** the benchmarks module. The benchmarks are part of the source repository but not included in the pip package.

---

## What Needs to be Done

Rebuild the Docker image with vLLM installed **from source** so that the benchmarks module is included.

### Option A: Install vLLM from Source with Benchmarks

```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# ... base setup ...

# Clone vLLM at the specific commit
RUN git clone https://github.com/vllm-project/vllm.git /workspace && \
    cd /workspace && \
    git checkout 99abb8b650c66664cdc84d815b7f306f33bd9881

# Install vLLM from source (this includes benchmarks)
RUN cd /workspace && \
    pip install -e ".[all]"

# Verify benchmarks are available
RUN python -c "from vllm.benchmarks.serve import main; print('Benchmarks OK')"
```

### Option B: Copy Benchmarks into Existing Image

If you want to patch the existing image without full rebuild:

```dockerfile
FROM ayushnangia16/nvidia-vllm-docker:99abb8b650c66664cdc84d815b7f306f33bd9881

# Clone vLLM to get benchmarks
RUN git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm-src && \
    cd /tmp/vllm-src && \
    git checkout 99abb8b650c66664cdc84d815b7f306f33bd9881

# Copy benchmarks module to installed vLLM
RUN cp -r /tmp/vllm-src/vllm/benchmarks /opt/venv/lib/python3.12/site-packages/vllm/ && \
    cp -r /tmp/vllm-src/benchmarks /workspace/

# Clean up
RUN rm -rf /tmp/vllm-src

# Verify
RUN /opt/venv/bin/python3 -c "from vllm.benchmarks.serve import main; print('Benchmarks OK')"
```

---

## Verification Steps

After rebuilding, verify the image works:

```bash
# 1. Check vllm.benchmarks module exists
docker run --rm <new_image> python3 -c "from vllm.benchmarks.serve import main; print('OK')"

# 2. Check vLLM CLI works
docker run --rm <new_image> python3 -m vllm.entrypoints.cli.main bench latency --help

# 3. Check benchmark scripts exist
docker run --rm <new_image> ls -la /workspace/benchmarks/
```

---

## Benchmark Command to Run

Once the image is fixed, run:

```bash
python benchmarks/benchmark_latency.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-model [ngram] \
  --ngram-prompt-lookup-min 5 \
  --ngram-prompt-lookup-max 10 \
  --num-speculative-tokens 5 \
  --input-len 550 \
  --output-len 150
```

Or with the new CLI:

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-model [ngram] \
  --ngram-prompt-lookup-min 5 \
  --ngram-prompt-lookup-max 10 \
  --num-speculative-tokens 5 \
  --input-len 550 \
  --output-len 150
```

---

## Current Image State

| Component | Status |
|-----------|--------|
| vLLM core | Installed (v0.0.0+local) |
| transformers | 4.56.0 |
| vllm.benchmarks | **MISSING** |
| /workspace/benchmarks/ | **MISSING** |
| vLLM CLI | Present but broken |

---

## Expected Image State After Fix

| Component | Status |
|-----------|--------|
| vLLM core | Installed |
| transformers | 4.56.0 (keep as-is) |
| vllm.benchmarks | Installed |
| /workspace/benchmarks/ | Present |
| vLLM CLI | Working |

---

## Notes

- **DO NOT downgrade transformers** - vLLM in this image requires transformers>=4.48.2
- The parent commit for baseline is: `3a1e6481586ed7f079275b5d5072a6e246af691e`
- Baseline image works fine: `shikhar481/vllm_fixed_human_images:baseline-3a1e6481586e`
