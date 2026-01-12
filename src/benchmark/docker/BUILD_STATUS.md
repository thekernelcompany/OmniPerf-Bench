# SGLang Docker Image Build Status

## Overview

Building Docker images at `shikhar481/sglang-images` for SGLang benchmarking. Images build sgl-kernel FROM SOURCE with git submodules to include the `deep_gemm` module (not available in PyPI versions).

## Build Status Summary

| Commit | Type | Model | torch | Status | Notes |
|--------|------|-------|-------|--------|-------|
| d1112d85 | human | gemma-2-2b | 2.5.1 | **SUCCESS** | Pushed to DockerHub |
| 48efec7b | parent | gemma-2-2b | 2.5.1 | **SUCCESS** | Pushed to DockerHub |
| 93470a14 | human | Llama-3.1-8B | 2.5.1 | **SKIPPED** | Requires deleted sgl-project/flashinfer fork |
| db452760 | parent | Llama-3.1-8B | 2.5.1 | **SKIPPED** | Requires deleted sgl-project/flashinfer fork |
| 9c088829 | human | Llama-3.1-8B | 2.6.0 | **IN PROGRESS** | Building with FA3 disabled |
| 005aad32 | parent | Llama-3.1-8B | 2.6.0 | **PENDING** | Will build after 9c088829 |

## Full Commit Hashes

- `d1112d8548eb13c842900b3a8d622345f9737759` (d1112d85)
- `48efec7b052354865aa2f0605a5bf778721f3cbb` (48efec7b)
- `93470a14116a60fe5dd43f0599206e8ccabdc211` (93470a14)
- `db452760e5b2378efd06b1ceb9385d2eeb6d217c` (db452760)
- `9c088829ee2a28263f36d0814fde448c6090b5bc` (9c088829)
- `005aad32ad45ce27d73fd39aa1f7e9ba5d8ebb8f` (005aad32)

## Detailed Build Notes

### d1112d85 / 48efec7b (SUCCESS)

**Configuration:**
- torch version: 2.5.1
- CUDA: 12.4.1
- flashinfer: cu124/torch2.5

**Build process:**
1. Clone SGLang at specific commit
2. Initialize git submodules (including 3rdparty/deepgemm)
3. Build sgl-kernel from source with `pip install -e . --no-build-isolation -v`
4. Install flashinfer from prebuilt wheels
5. Install SGLang with `--no-deps` to avoid PyPI sgl-kernel

**Result:** Images successfully built and pushed to DockerHub.

---

### 93470a14 / db452760 (SKIPPED)

**Issue:** CMakeLists.txt at these commits references `sgl-project/flashinfer` repository which no longer exists (returns 404).

```cmake
# flashinfer
FetchContent_Declare(
    repo-flashinfer
    GIT_REPOSITORY https://github.com/sgl-project/flashinfer  # 404 - DELETED
    GIT_TAG        sgl-kernel
    GIT_SHALLOW    OFF
)
```

**Error:**
```
fatal: invalid reference: sgl-kernel
CMake Error at repo-flashinfer-subbuild/.../repo-flashinfer-populate-gitclone.cmake:61 (message):
  Failed to checkout tag: 'sgl-kernel'
```

**Resolution:** These commits cannot be built. The `sgl-project/flashinfer` fork has been deleted from GitHub. Would need to:
- Find an archived copy of the fork, OR
- Patch CMakeLists.txt to use `flashinfer-ai/flashinfer` with a compatible commit, OR
- Skip these benchmarks entirely

---

### 9c088829 / 005aad32 (IN PROGRESS / PENDING)

**Configuration:**
- torch version: 2.6.0
- CUDA: 12.4.1
- flashinfer: cu124/torch2.6

**Issue encountered:** FA3 (Flash Attention 3) hopper kernels fail to compile with CUDA 12.4 due to CUTLASS template errors:

```
error: no instance of function template "cute::make_tiled_copy_C" matches the argument list
        argument types are: (...)
```

**Fix applied:** Dockerfile patches CMakeLists.txt to disable FA3:
```dockerfile
RUN sed -i 's/set(SGL_KERNEL_ENABLE_FA3 ON)/#set(SGL_KERNEL_ENABLE_FA3 ON) # DISABLED/g' CMakeLists.txt
```

**Previous build errors:**
1. v1-v2: FA3 CUTLASS template compilation errors
2. v3-v4: Incorrect sed commands patched flash-attention GIT_TAG instead of just FA3
3. v5 (current): Corrected Dockerfile with only FA3 disable patch

---

## Docker Images on DockerHub

Repository: `shikhar481/sglang-images`

Available tags:
- `d1112d8548eb13c842900b3a8d622345f9737759`
- `48efec7b052354865aa2f0605a5bf778721f3cbb`

---

## Dockerfile Configuration

Key features of the build Dockerfile:
1. Base: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
2. Python 3.11
3. Build sgl-kernel FROM SOURCE (not PyPI) to include deep_gemm
4. FA3 hopper kernels disabled for CUDA 12.4 compatibility
5. flashinfer installed from prebuilt wheels matching torch version
6. SGLang installed with `--no-deps` to prevent PyPI sgl-kernel override

---

## Build Commands

```bash
# Build single image
cd /tmp/sglang_build/sglang
DOCKER_BUILDKIT=0 docker build -f Dockerfile.benchmark \
  -t shikhar481/sglang-images:<COMMIT_HASH> \
  --build-arg COMMIT_HASH=<COMMIT_HASH> \
  --build-arg TORCH_VERSION=<2.5.1|2.6.0> \
  --build-arg TORCH_MINOR=<2.5|2.6> \
  .

# Push to DockerHub
docker push shikhar481/sglang-images:<COMMIT_HASH>
```

---

## Known Issues

1. **deep_gemm import**: Requires GPU at runtime; cannot be verified during Docker build
2. **FA3 compilation**: CUTLASS template errors with CUDA 12.4 on sm_90a (H100)
3. **Deleted flashinfer fork**: Some older commits reference `sgl-project/flashinfer` which no longer exists
4. **torch version compatibility**: flashinfer wheels must match torch major.minor version

---

*Last updated: 2025-01-12*
