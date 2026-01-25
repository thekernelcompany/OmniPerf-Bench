# SGLang Benchmark Status

## Environment
- GPU: NVIDIA H100 PCIe (SM90)
- CUDA Driver: 570.148.08 (CUDA 12.8)
- Host CUDA: 12.6

## Docker Images Downloaded
1. ayushnangia16/nvidia-sglang-docker:187b85b7f38496653948a2aba546d53c09ada0f3 (20.8GB)
2. ayushnangia16/nvidia-sglang-docker:148254d4db8bf3bffee23710cd1acbd5711ebd1b (21.3GB)
3. ayushnangia16/nvidia-sglang-docker:6cb00c6398126513e37c43dd975d461765fb44c7 (15GB)
4. ayushnangia16/nvidia-sglang-docker:6b231325b9782555eb8e1cfcf27820003a98382b (15GB)
5. shikhar481/sglang-images:148254d4-src (21.1GB)

## Known Compatibility Issues

### sgl_kernel ABI Mismatch
- Images built with older PyTorch versions have ABI incompatibility with host
- Error: `undefined symbol: _ZN3c108ListType3get...`
- Attempted fix: Upgrade sgl_kernel via pip - partial success

### Missing System Dependencies
- libnuma.so.1 missing in some images
- Fix: Install libnuma-dev in container
- Status: Implemented in benchmark script

### Transformers Version
- Older images missing AutoProcessor
- Error: `cannot import name 'AutoProcessor' from 'transformers'`
- Some images have transformers 4.57.6 which should work

## Benchmark Results

| Commit | Status | Error |
|--------|--------|-------|
| 187b85b7 | Error | sgl_kernel ABI mismatch |
| 148254d4 | Error | Transformers version issue |
| 09deb20d | Error | vLLM model_loader import |

## Next Steps
1. Consider building fresh Docker images with compatible dependencies
2. Test with source installation instead of Docker
3. Check if Modal-based benchmarks work better

## Scripts Updated
- `scripts/runners/local_docker_sglang_benchmark.py` - Added libnuma install, sgl_kernel upgrade
- `scripts/runners/run_all_sglang_3way_benchmarks.py` - Batch runner for all commits

