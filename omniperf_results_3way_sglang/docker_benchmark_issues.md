# SGLang Local Docker Benchmark - Issues Log

## Date: 2026-01-23

## Summary

Implementation of local Docker-based 3-way benchmark for SGLang completed. During testing, discovered issues with pre-built Docker images.

## Files Created/Modified

1. **Created**: `scripts/runners/local_docker_sglang_benchmark.py`
   - Main local Docker benchmark runner for SGLang
   - Supports human-only and 3-way benchmarks
   - Searches both `shikhar481/sglang-images` and `ayushnangia16/nvidia-sglang-docker` repos

2. **Modified**: `src/benchmark/tools/build_sglang_images.py`
   - Added `--build-baselines` flag for building baseline images
   - Added `--baseline` flag for single baseline builds
   - Baseline images tagged as `baseline-{commit[:12]}`
   - Status tracking in `sglang_docker_builds_full.csv`

3. **Modified**: `scripts/runners/hero_sglang_benchmark.py`
   - Added `--use-local-docker` flag
   - Added `--human-only` and `--3way` flags
   - Integrated `run_3way_benchmark_local_docker()` function

## Docker Image Issues Found

### shikhar481/sglang-images

**Image tested**: `shikhar481/sglang-images:09deb20d-v3`

**Error**:
```
ModuleNotFoundError: No module named 'vllm.model_executor.model_loader.utils';
'vllm.model_executor.model_loader' is not a package
```

**Root cause**: SGLang 0.1.14 installed in image has vLLM dependency version mismatch. The vLLM version installed doesn't have the expected module structure.

### ayushnangia16/nvidia-sglang-docker

**Image tested**: `ayushnangia16/nvidia-sglang-docker:09deb20deef8181a23f66c933ea74b86fee47366`

**SGLang version**: 0.1.14 (imports correctly)

**Errors found**:

1. **Llama-3.1 model**: Transformers version incompatibility
   ```
   ValueError: `rope_scaling` must be a dictionary with two fields, `type` and `factor`
   ```
   The old transformers version doesn't understand the new rope_scaling format.

2. **Llama-2 model**: uvloop/asyncio issue
   ```
   RuntimeError: There is no current event loop in thread 'MainThread'.
   AssertionError (proc_router.is_alive() and proc_detoken.is_alive())
   ```
   Event loop issue in detokenizer manager.

## Recommendations

1. **Rebuild Docker images** with compatible dependency versions
2. **Use wheel-based approach** from Modal benchmark (builds SGLang from source)
3. **Test images before deployment** with basic server startup check

## Machine Configuration

- 8x NVIDIA A100-SXM4-40GB
- Ephemeral storage: 6.8TB at `/opt/dlami/nvme`
- Docker reconfigured to use ephemeral storage

## Next Steps

1. Test with ayushnangia16 images (after disk space fix)
2. Build fresh Docker images if pre-built ones continue to fail
3. Consider using Modal's wheel-based approach for reliability
