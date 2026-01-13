# SGLang Docker Image Build Status

## Overview

Building Docker images at `shikhar481/sglang-images` for SGLang benchmarking. Images build sgl-kernel FROM SOURCE with git submodules to include the `deep_gemm` module (not available in PyPI versions).

## Build Status Summary

| Commit | Type | Model | torch | Build | Runtime | Notes |
|--------|------|-------|-------|-------|---------|-------|
| d1112d85 | human | gemma-2-2b | 2.5.1 | **SUCCESS** | **BROKEN** | torchao 0.15.0 requires torch.int1 (torch 2.6+) |
| 48efec7b | parent | gemma-2-2b | 2.5.1 | **SUCCESS** | **BROKEN** | torchao 0.15.0 requires torch.int1 (torch 2.6+) |
| 93470a14 | human | Llama-3.1-8B | 2.5.1 | **SKIPPED** | N/A | Requires deleted sgl-project/flashinfer fork |
| db452760 | parent | Llama-3.1-8B | 2.5.1 | **SKIPPED** | N/A | Requires deleted sgl-project/flashinfer fork |
| 9c088829 | human | Llama-3.1-8B | 2.6.0 | **SUCCESS** | **BROKEN** | Missing runtime deps (IPython, orjson, uvicorn) |
| 005aad32 | parent | Llama-3.1-8B | 2.6.0 | **SUCCESS** | **BROKEN** | Missing runtime deps (IPython, orjson, uvicorn) |

## Runtime Testing Results (2026-01-13)

### Test Environment
- Ubuntu 22.04.5 LTS
- NVIDIA H100 PCIe (81GB)
- Docker 28.2.2 with NVIDIA Container Toolkit 1.18.1
- CUDA 12.6 (host) / 12.4.1 (containers)

### Import Verification (ALL PASS)

All 4 built images have sgl_kernel and deep_gemm importable:

```
d1112d85: torch=2.5.1+cu124, sgl_kernel=FOUND, deep_gemm=FOUND
48efec7b: torch=2.5.1+cu124, sgl_kernel=FOUND, deep_gemm=FOUND
9c088829: torch=2.6.0+cu124, sgl_kernel=FOUND, deep_gemm=FOUND
005aad32: torch=2.6.0+cu124, sgl_kernel=FOUND, deep_gemm=FOUND
```

### Server Startup Tests (ALL FAIL)

#### d1112d85 / 48efec7b (torch 2.5.1)

**Error:** torchao version mismatch
```
AttributeError: module 'torch' has no attribute 'int1'
```

**Root Cause:** torchao 0.15.0 was installed, which requires `torch.int1` only available in torch 2.6+.

**Fix Required:** Either:
1. Downgrade torchao to 0.14.x compatible with torch 2.5.1, OR
2. Upgrade torch to 2.6.0

#### 9c088829 / 005aad32 (torch 2.6.0)

**Error:** Missing runtime dependencies
```
ModuleNotFoundError: No module named 'IPython'
ModuleNotFoundError: No module named 'orjson'
ModuleNotFoundError: No module named 'uvicorn'
```

**Additional Issue:** Installing missing packages via pip causes segfault due to ABI mismatch between PyPI sglang packages and source-built sgl_kernel.

**Fix Required:** Rebuild images with full `sglang[srt]` dependencies installed BEFORE building sgl_kernel from source.

### 3-Way Benchmark Results

All benchmark phases failed due to server startup crashes:

```
d1112d85:
  baseline: [FAIL] Server crashed during startup (torchao.int1 error)
  human: [FAIL] Server crashed during startup
  agent: [FAIL] Server crashed during startup

9c088829 (with runtime deps installed):
  Model loaded successfully ✓
  KV cache allocated ✓
  Inference: [FAIL] Segfault in Triton JIT compiler
```

### Root Cause Analysis

| Image | torch | triton | Issue |
|-------|-------|--------|-------|
| d1112d85 | 2.5.1 | 3.1.0 | torchao 0.15.0 requires torch.int1 (torch 2.6+) |
| 9c088829 | 2.6.0 | 3.2.0 | Segfault in triton code_generator.py during inference |

### Required Fixes for Rebuild

**For d1112d85/48efec7b (torch 2.5.1):**
```dockerfile
# Pin torchao to version compatible with torch 2.5.1
RUN pip install "torchao<0.15"
```

**For 9c088829/005aad32 (torch 2.6.0):**
```dockerfile
# Install ALL runtime dependencies before sgl-kernel build
RUN pip install compressed-tensors datasets decord fastapi hf_transfer huggingface_hub \
    interegular "llguidance>=0.7.11,<0.8.0" modelscope ninja orjson packaging pillow \
    "prometheus-client>=0.20.0" psutil pydantic pynvml python-multipart "pyzmq>=25.1.2" \
    "soundfile==0.13.1" "torchao>=0.7.0" uvicorn uvloop "xgrammar==0.1.17" IPython \
    setproctitle "outlines>=0.0.44,<=0.1.11" partial_json_parser einops sse-starlette \
    httptools msgspec

# May need to pin triton version for compatibility
RUN pip install "triton==3.1.0"  # or test with 3.2.0 after other fixes
```

---

## Rebuild Instructions (2026-01-13)

### Summary: Dockerfile vs Docker Image Status

| Commit | Dockerfile Status | Docker Image Status | Action Required |
|--------|-------------------|---------------------|-----------------|
| d1112d85 | **FIXED** (torchao<0.15 already added) | **STALE** (built from old Dockerfile) | Rebuild only |
| 48efec7b | **FIXED** (uses same Dockerfile) | **STALE** (built from old Dockerfile) | Rebuild only |
| 9c088829 | **NEEDS UPDATE** (missing runtime deps) | **STALE** | Update Dockerfile + Rebuild |
| 005aad32 | **NEEDS UPDATE** (uses same Dockerfile) | **STALE** | Update Dockerfile + Rebuild |

### d1112d85 / 48efec7b (torch 2.5.1) - REBUILD ONLY

The `Dockerfile.d1112d85` already contains the fix:
```dockerfile
# CRITICAL: Pin torchao<0.15 for torch 2.5.1 compatibility (0.15+ requires torch.int1 from torch 2.6+)
RUN pip install "torchao<0.15" "transformers>=4.40.0" ...
```

**Why images are broken:** The Docker images on DockerHub were built from an **older version** of the Dockerfile before this fix was added. The current Dockerfile is correct.

**Rebuild commands:**
```bash
cd /path/to/OmniPerf-Bench/src/benchmark/docker/sglang_commits

# Build d1112d85
DOCKER_BUILDKIT=0 docker build \
  -f Dockerfile.d1112d85 \
  -t shikhar481/sglang-images:d1112d8548eb13c842900b3a8d622345f9737759 \
  .

# Build 48efec7b (same Dockerfile, different commit arg)
DOCKER_BUILDKIT=0 docker build \
  -f Dockerfile.d1112d85 \
  --build-arg COMMIT_HASH=48efec7b052354865aa2f0605a5bf778721f3cbb \
  -t shikhar481/sglang-images:48efec7b052354865aa2f0605a5bf778721f3cbb \
  .

# Push to DockerHub
docker push shikhar481/sglang-images:d1112d8548eb13c842900b3a8d622345f9737759
docker push shikhar481/sglang-images:48efec7b052354865aa2f0605a5bf778721f3cbb
```

### 9c088829 / 005aad32 (torch 2.6.0) - UPDATE DOCKERFILE + REBUILD

The `Dockerfile.9c088829` needs the following changes:

**Problem 1: Missing runtime dependencies**

Current (broken):
```dockerfile
RUN pip install "transformers>=4.40.0" "huggingface_hub>=0.23.0" "tokenizers>=0.19.0" \
    "accelerate>=0.30.0" "numpy<2.0" requests aiohttp \
    triton packaging xgrammar pydantic fastapi uvicorn vllm || true
```

Fixed (add ALL runtime deps):
```dockerfile
# Install ALL runtime dependencies BEFORE sgl-kernel build
# NOTE: Do NOT install vllm - it will upgrade torch and break sgl-kernel ABI!
RUN pip install "torchao>=0.7.0,<0.15" "transformers>=4.40.0" "huggingface_hub>=0.23.0" \
    "tokenizers>=0.19.0" "accelerate>=0.30.0" "numpy<2.0" requests aiohttp \
    triton packaging xgrammar pydantic fastapi uvicorn uvloop httptools \
    orjson setproctitle IPython pyzmq pillow psutil compressed-tensors \
    python-multipart sse-starlette interegular "outlines>=0.0.44,<=0.1.11" \
    partial_json_parser einops msgspec prometheus-client decord soundfile \
    cuda-python hf_transfer modelscope pynvml datasets pandas tqdm pybase64 || true
```

**Problem 2: Triton segfault during inference**

The torch 2.6.0 images crash with a segfault in Triton's JIT compiler during the first inference request. This may be due to triton 3.2.0 incompatibility.

Potential fix (add before sgl-kernel build):
```dockerfile
# Pin triton to 3.1.0 for stability (3.2.0 causes segfaults)
RUN pip install "triton==3.1.0"
```

**Problem 3: SGLang install overwrites sgl-kernel**

Current (problematic):
```dockerfile
RUN pip install -e "python[srt]" || pip install -e "python"
```

Fixed (use --no-deps to prevent overwriting source-built sgl-kernel):
```dockerfile
RUN pip install -e "python[srt]" --no-deps || pip install -e "python" --no-deps
```

**Rebuild commands (after updating Dockerfile):**
```bash
cd /path/to/OmniPerf-Bench/src/benchmark/docker/sglang_commits

# Build 9c088829
DOCKER_BUILDKIT=0 docker build \
  -f Dockerfile.9c088829 \
  -t shikhar481/sglang-images:9c088829ee2a28263f36d0814fde448c6090b5bc \
  .

# Build 005aad32 (same Dockerfile, different commit arg)
DOCKER_BUILDKIT=0 docker build \
  -f Dockerfile.9c088829 \
  --build-arg COMMIT_HASH=005aad32ad45ce27d73fd39aa1f7e9ba5d8ebb8f \
  -t shikhar481/sglang-images:005aad32ad45ce27d73fd39aa1f7e9ba5d8ebb8f \
  .

# Push to DockerHub
docker push shikhar481/sglang-images:9c088829ee2a28263f36d0814fde448c6090b5bc
docker push shikhar481/sglang-images:005aad32ad45ce27d73fd39aa1f7e9ba5d8ebb8f
```

### Post-Rebuild Verification

After rebuilding, verify each image:

```bash
# 1. Check imports (should all pass)
for commit in d1112d85 48efec7b 9c088829 005aad32; do
  echo "=== $commit ==="
  docker run --rm --gpus all shikhar481/sglang-images:${commit}* python -c "
import importlib.util
import torch
for mod in ['sgl_kernel', 'deep_gemm']:
    spec = importlib.util.find_spec(mod)
    print(f'{mod}: found={spec is not None}')
print(f'torch: {torch.__version__}')
import torchao; print(f'torchao: {torchao.__version__}')
"
done

# 2. Test server startup (quick smoke test)
docker run --rm --gpus all -p 30000:30000 \
  -e HF_TOKEN=<your_token> \
  shikhar481/sglang-images:d1112d8548eb13c842900b3a8d622345f9737759 \
  python -m sglang.launch_server --model google/gemma-2-2b-it --port 30000

# 3. Test inference (in another terminal)
curl http://localhost:30000/generate -d '{"text": "Hello", "max_new_tokens": 10}'
```

---

## Full Commit Hashes

- `d1112d8548eb13c842900b3a8d622345f9737759` (d1112d85)
- `48efec7b052354865aa2f0605a5bf778721f3cbb` (48efec7b)
- `93470a14116a60fe5dd43f0599206e8ccabdc211` (93470a14)
- `db452760e5b2378efd06b1ceb9385d2eeb6d217c` (db452760)
- `9c088829ee2a28263f36d0814fde448c6090b5bc` (9c088829)
- `005aad32ad45ce27d73fd39aa1f7e9ba5d8ebb8f` (005aad32)

## Detailed Build Notes

### d1112d85 / 48efec7b (SUCCESS - REBUILT 2025-01-13)

**Configuration:**
- torch version: 2.5.1
- CUDA: 12.4.1
- flashinfer: cu124/torch2.5
- sgl-kernel: 0.0.5.post2 (built from source)

**Build process:**
1. Clone SGLang at specific commit
2. Initialize git submodules (including 3rdparty/deepgemm)
3. Build sgl-kernel from source with `pip install . --no-build-isolation -v` (NON-EDITABLE)
4. Install flashinfer from prebuilt wheels
5. Install SGLang dependencies (WITHOUT vllm to preserve torch 2.5.1)
6. Install SGLang from source

**Result:** Images successfully rebuilt and pushed to DockerHub.

**Docker image digests:**
- `d1112d85`: sha256:8825b87f980215ecc09a7d155ad42dbd2885054e5bdbb4141e3b8a8b31816606
- `48efec7b`: sha256:c8d208e05c6b183af071a9759f5a50372de934b2208f07b9d7cc7bd3555bc2ce

## Critical Bug Fix: deep_gemm Missing (2025-01-13)

### The Problem

The original d1112d85 and 48efec7b images were **BROKEN** - they had `sgl_kernel` but were **missing `deep_gemm`**.

**Verification results before fix:**

| Image | sgl_kernel | deep_gemm | Status |
|-------|------------|-----------|--------|
| d1112d85 (torch 2.5.1) | FOUND | **NOT FOUND** | BROKEN |
| 48efec7b (torch 2.5.1) | FOUND | **NOT FOUND** | BROKEN |
| 9c088829 (torch 2.6.0) | FOUND | FOUND | OK |
| 005aad32 (torch 2.6.0) | FOUND | FOUND | OK |

### Root Cause Analysis

**The Problem: Editable Install + setup.py = No deep_gemm**

The original Dockerfile used **editable install**:
```dockerfile
RUN pip install -e . --no-build-isolation -v  # BROKEN for setup.py commits
```

For **OLD commits** (d1112d85, 48efec7b) using `setup.py`:

1. `setup.py` has a `CustomBuildPy` class that calls `copy_deepgemm_to_build_lib()`:
```python
class CustomBuildPy(build_py):
    def run(self):
        self.copy_deepgemm_to_build_lib()  # Copies to self.build_lib
        build_py.run(self)
```

2. `copy_deepgemm_to_build_lib()` copies deep_gemm to `self.build_lib`:
```python
def copy_deepgemm_to_build_lib(self):
    dst_dir = os.path.join(self.build_lib, "deep_gemm")  # TEMP DIRECTORY!
    src_dir = os.path.join(str(deepgemm.resolve()), "deep_gemm")
    shutil.copytree(src_dir, dst_dir)
```

3. **With editable install (`-e`):**
   - `build_lib` is a **temporary directory** (e.g., `/tmp/pip-xxx/build`)
   - deep_gemm gets copied there
   - Temp directory is **deleted** after install completes
   - Editable install creates `.pth` file pointing to source directory
   - **deep_gemm is NOT in the source directory's Python path** -> ModuleNotFoundError

### Why Newer Commits (9c088829, 005aad32) Work

For **NEW commits** using `pyproject.toml` + CMake:

CMakeLists.txt explicitly installs deep_gemm:
```cmake
install(DIRECTORY "${repo-deepgemm_SOURCE_DIR}/deep_gemm/"
        DESTINATION "deep_gemm"
        ...)
```

This CMake `install()` directive puts deep_gemm directly into the wheel/site-packages, which works correctly even with editable installs.

### The Fix

**Change 1: Non-editable install for sgl-kernel**

```dockerfile
# OLD (broken for setup.py commits):
RUN pip install -e . --no-build-isolation -v

# NEW (works for all commits):
RUN pip install . --no-build-isolation -v
```

**Change 2: Remove vllm from dependencies**

The original Dockerfile had:
```dockerfile
RUN pip install ... "vllm>=0.6.4.post1" || true
```

This upgraded torch from 2.5.1 to 2.9.0, breaking sgl-kernel ABI compatibility.

Fixed by removing vllm:
```dockerfile
# NOTE: Do NOT install vllm here - it will upgrade torch and break sgl-kernel ABI!
RUN pip install "transformers>=4.40.0" ... || true
```

### Verification After Fix

```
$ docker run --rm <image> python -c "import importlib.util; ..."

sgl_kernel: found=True
deep_gemm: found=True
torch: 2.5.1+cu124
deep_gemm location: /usr/local/lib/python3.11/dist-packages/deep_gemm/__init__.py
```

The key indicator: deep_gemm is now in `/usr/local/lib/python3.11/dist-packages/` (site-packages) instead of a deleted temp directory.

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

### 9c088829 / 005aad32 (SUCCESS)

**Configuration:**
- torch version: 2.6.0
- CUDA: 12.4.1
- flashinfer: cu124/torch2.6
- sgl-kernel: 0.1.0 (built from source)
- SGLang: 0.4.5.post3

## Debugging Journey for 9c088829

### Error 1: FA3 CUTLASS Template Compilation Failure (v1-v2 builds)

**Root Cause:** CMakeLists.txt automatically enables FA3 (Flash Attention 3) hopper kernels when CUDA >= 12.4:

```cmake
# sgl-kernel/CMakeLists.txt lines 142-147
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" OR SGL_KERNEL_ENABLE_SM90A)
    set(SGL_KERNEL_ENABLE_FA3 ON)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_90a,code=sm_90a"
    )
endif()
```

FA3 kernels require sm_90a (H100) architecture and use CUTLASS templates that had compatibility issues with the CUDA 12.4 toolchain.

**Error Message:**
```
/tmp/.../hopper/instantiations/flash_fwd_hdimall_bf16_...sm90.cu: In instantiation of ...
error: no instance of function template "cute::make_tiled_copy_C" matches the argument list
        argument types are: (cute::Copy_Atom<cute::SM90_TMA_REDUCE_ADD,
        cute::bfloat16_t>,...
```

**Fix:** Patch CMakeLists.txt to disable FA3:
```dockerfile
RUN sed -i 's/set(SGL_KERNEL_ENABLE_FA3 ON)/#set(SGL_KERNEL_ENABLE_FA3 ON) # DISABLED/g' CMakeLists.txt
```

---

### Error 2: Incorrect GIT_TAG Replacement (v3-v4 builds)

**Root Cause:** An earlier version of the Dockerfile included sed commands to redirect `sgl-project/flashinfer` URLs to `flashinfer-ai/flashinfer`. These sed commands used a pattern that was too broad:

```dockerfile
# PROBLEMATIC (removed in v5)
RUN find . -name "CMakeLists.txt" -exec sed -i 's|GIT_TAG.*sgl-kernel|GIT_TAG 9220fb3443b5a5d274f00ca5552f798e225239b7|g' {} \;
```

This accidentally matched BOTH repositories in CMakeLists.txt:

1. **flashinfer** (line 62): `GIT_TAG 9220fb3443b5a5d274f00ca5552f798e225239b7` - CORRECT
2. **flash-attention** (line 70): `GIT_TAG sgl-kernel` - INCORRECTLY CHANGED

The flash-attention repo (`sgl-project/sgl-attn`) uses a branch named `sgl-kernel`, NOT a commit hash.

**Error Message:**
```
Cloning into 'repo-flash-attention-src'...
fatal: reference is not a tree: 9220fb3443b5a5d274f00ca5552f798e225239b7
CMake Error at repo-flash-attention-subbuild/.../repo-flash-attention-populate-gitclone.cmake:61 (message):
  Failed to checkout tag: '9220fb3443b5a5d274f00ca5552f798e225239b7'
```

**Fix:** Remove ALL flashinfer URL patching. The 9c088829 commit already uses `flashinfer-ai/flashinfer` (the correct upstream repo), so no URL patching is needed.

---

### Successful Build (v5)

**Final Dockerfile Key Section:**
```dockerfile
WORKDIR /opt/sglang/sgl-kernel
RUN pip install scikit-build-core ninja cmake packaging

# CRITICAL: Disable FA3 hopper kernels to avoid CUTLASS template errors
# Comment out the line that sets FA3 ON when CUDA >= 12.4
RUN cat CMakeLists.txt | head -150 && \
    sed -i 's/set(SGL_KERNEL_ENABLE_FA3 ON)/#set(SGL_KERNEL_ENABLE_FA3 ON) # DISABLED/g' CMakeLists.txt && \
    echo "Patched CMakeLists.txt to disable FA3"

# Build sgl-kernel from source
RUN pip install -e . --no-build-isolation -v 2>&1 | tee /tmp/sgl_kernel_build.log || (tail -100 /tmp/sgl_kernel_build.log && exit 1)
```

**What the patch does:**

Before patch (CMakeLists.txt line 143):
```cmake
set(SGL_KERNEL_ENABLE_FA3 ON)
```

After patch:
```cmake
#set(SGL_KERNEL_ENABLE_FA3 ON) # DISABLED
```

This keeps FA3 at its default `OFF` state, preventing the hopper kernel compilation.

**Build Output Verification:**
```
Successfully built sgl-kernel
Installing collected packages: sgl-kernel
Successfully installed sgl-kernel-0.1.0

pip list | grep -E "sgl|deep":
sgl-kernel               0.1.0       /opt/sglang/sgl-kernel
```

The path `/opt/sglang/sgl-kernel` confirms it was built from source (not PyPI).

---

## Docker Images on DockerHub

Repository: `shikhar481/sglang-images`

Available tags:
- `d1112d8548eb13c842900b3a8d622345f9737759`
- `48efec7b052354865aa2f0605a5bf778721f3cbb`
- `9c088829ee2a28263f36d0814fde448c6090b5bc`
- `005aad32ad45ce27d73fd39aa1f7e9ba5d8ebb8f`

---

## Dockerfile Configuration

Key features of the build Dockerfile:
1. Base: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
2. Python 3.11
3. Build sgl-kernel FROM SOURCE (not PyPI) to include deep_gemm
4. **Non-editable install** (`pip install .`) for setup.py-based commits to properly install deep_gemm
5. FA3 hopper kernels disabled for CUDA 12.4 compatibility (torch 2.6.0 images only)
6. flashinfer installed from prebuilt wheels matching torch version
7. **vllm NOT installed** in direct dependencies to prevent torch upgrade
8. SGLang installed from source with sgl-kernel already present

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
2. **FA3 compilation**: CUTLASS template errors with CUDA 12.4 on sm_90a (H100) - WORKAROUND: disable FA3
3. **Deleted flashinfer fork**: Some older commits reference `sgl-project/flashinfer` which no longer exists
4. **torch version compatibility**: flashinfer wheels must match torch major.minor version
5. **sed pattern matching**: Be careful with sed replacements - avoid patterns that match multiple targets
6. **Editable install breaks deep_gemm**: For setup.py-based commits, `pip install -e .` copies deep_gemm to a temp directory that gets deleted. Use non-editable install instead.
7. **vllm upgrades torch**: Installing vllm will upgrade torch to the latest version, breaking sgl-kernel ABI compatibility. Either skip vllm or pin torch version carefully.

---

## Lessons Learned

1. **Test sed patterns carefully**: Use `grep` first to see all matches before running `sed -i`
2. **Check CMakeLists.txt thoroughly**: Multiple FetchContent declarations may have similar patterns
3. **Verify build outputs**: Check `pip list` paths to confirm source vs PyPI installation
4. **Keep URL patching minimal**: Only patch what's necessary; newer commits may not need patching
5. **Verify deep_gemm presence**: Always run verification after build:
   ```bash
   docker run --rm <image> python -c "
   import importlib.util
   for mod in ['sgl_kernel', 'deep_gemm']:
       spec = importlib.util.find_spec(mod)
       print(f'{mod}: found={spec is not None}')
   "
   ```
6. **Editable vs non-editable install**: For commits using setup.py (older sgl-kernel), use `pip install .` (non-editable). For commits using pyproject.toml + CMake (newer), either works.
7. **Watch for transitive dependencies**: Installing packages like vllm can upgrade torch and break ABI compatibility with pre-built CUDA extensions.

---

*Last updated: 2026-01-13*
