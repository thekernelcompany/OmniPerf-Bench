# SGLang Docker Images - Rebuilt From Source

**Date:** 2026-01-19

## Summary

**CRITICAL: Only 2 of 3 images are ready. The 79961afa image needs GPU rebuild.**

Two Docker images were successfully rebuilt from source:
1. Checkout the specific commit
2. Use the commit's own source files
3. Build dependencies that match the commit's era
4. Build `sgl-kernel` from source (for newer commits) to avoid ABI mismatch

## Images Rebuilt

### 1. Commit 2a754e57 (v0.1.17)

**PR:** #579 - 2x performance improvement for large prefill
**Status:** REBUILT - Fixed missing rpyc, outlines

**Original Issue:**
- Missing `rpyc`, `outlines`, `pyairports` modules

**Fix Applied:**
- Built from commit's own source with `python[all]` dependencies
- Uses vllm==0.5.0, CUDA 12.1, torch 2.3.0

**Build Command:**
```bash
cd /tmp/sglang_src
git checkout 2a754e57b052e249ed4f8572cb6f0069ba6a495e

cat > Dockerfile << 'EOF'
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
# ... (see /tmp/sglang_src/Dockerfile for full content)
COPY python/ /sglang/python/
RUN pip install -e "python[all]"
EOF

docker build -t sglang:2a754e57 .
docker tag sglang:2a754e57 shikhar481/sglang-images:2a754e57
```

**Verification:**
```
SGLang version: 0.1.17
rpyc: OK
outlines: OK
```

---

### 2. Commit 79961afa (v0.4.6.post2)

**PR:** #6077 - Optimize FA3 metadata init (21% faster - 530us -> 418us)
**Status:** NOT FIXED - Requires GPU to build sgl-kernel from source

**Original Issue:**
- `sgl_kernel` ABI mismatch: `undefined symbol: _ZN3c108ListType...`
- Missing `sentencepiece`, `outlines`, `uvloop`

**Problem:**
- sgl-kernel source build FAILED on non-GPU machine (CUDA compilation errors)
- PyPI wheel was installed as fallback (same ABI mismatch risk)
- Cannot verify fix without H100 GPU

**To Fix (run on H100):**
```bash
# On H100 machine with nvidia-docker:
cd /tmp/sglang_src
git checkout 79961afa8281f98f380d11db45c8d4b6e66a574f

cat > Dockerfile << 'EOF'
FROM nvcr.io/nvidia/tritonserver:24.04-py3-min
RUN pip install scikit-build-core cmake ninja
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124
COPY . /sgl-workspace/sglang/

# Build sgl-kernel from source (requires GPU for CUDA compilation)
RUN cd sgl-kernel && pip install . -v

RUN pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python
RUN pip install uvloop sentencepiece
EOF

docker build -t sglang:79961afa .
# Verify sgl_kernel works:
docker run --gpus all sglang:79961afa python3 -c "from sgl_kernel import common_ops; print('OK')"
```

**Current Image Status:** Uses PyPI wheel - likely still has ABI mismatch

---

### 3. Commit ab4a83b2 (v0.3.0)

**PR:** #1339 - Optimize schedule
**Status:** REBUILT - Fixed pyairports

**Original Issue:**
- Missing `pyairports` module (required by `outlines`)
- The `pyairports` package on PyPI is a placeholder (yanked/replaced)

**Fix Applied:**
- Created mock `pyairports` module using `airportsdata` package
- The mock provides `AIRPORT_LIST` which outlines expects
- Uses vllm==0.5.5, CUDA 12.1, torch 2.4

**Build Command:**
```bash
cd /tmp/sglang_src
git checkout ab4a83b25909aa98330b838a224e4fe5c943e483

cat > Dockerfile << 'EOF'
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
COPY . /sgl-workspace/sglang/
RUN pip install -e "python[all]"

# Fix pyairports (PyPI package is broken)
RUN pip install airportsdata && \
    pip uninstall -y pyairports || true && \
    rm -rf /usr/local/lib/python3.10/dist-packages/pyairports* && \
    mkdir -p /usr/local/lib/python3.10/dist-packages/pyairports && \
    echo "" > /usr/local/lib/python3.10/dist-packages/pyairports/__init__.py && \
    python3 -c "import airportsdata; d=airportsdata.load(); print('AIRPORT_LIST =', list(d.keys()))" > /usr/local/lib/python3.10/dist-packages/pyairports/airports.py
EOF

docker build -t sglang:ab4a83b2 .
docker tag sglang:ab4a83b2 shikhar481/sglang-images:ab4a83b2
```

**Verification:**
```
SGLang version: 0.3.0
uvloop: OK
outlines: OK
pyairports mock: OK (has 28270 airports)
```

---

## Key Lessons Learned

### 1. Don't Use Generic Build Scripts for All Commits
The original `rebuild_sglang_fixed.sh` used:
- Hardcoded CUDA 12.4.1 for ALL commits (wrong for older versions)
- Installed sgl-kernel from PyPI (pre-built wheel → ABI mismatch)
- Fixed dependency versions regardless of commit era

### 2. Build sgl-kernel From Source
For commits that have `sgl-kernel/` directory:
- MUST build from source: `cd sgl-kernel && pip install -e . -v`
- PyPI wheels are compiled for specific torch/CUDA versions
- ABI mismatch causes `undefined symbol` errors at runtime

### 3. The pyairports Package is Broken
- The original `pyairports` package was replaced with a placeholder on PyPI
- outlines versions 0.0.43+ require pyairports for airport type validation
- Workaround: Create mock module using `airportsdata` package

### 4. Match CUDA/torch Versions to Commit Era
| SGLang Version | vLLM Version | Recommended CUDA | Recommended torch |
|----------------|--------------|------------------|-------------------|
| 0.1.x | 0.5.0 | 12.1 | 2.3.0 |
| 0.3.x | 0.5.5 | 12.1 | 2.4.0 |
| 0.4.x | 0.6+ | 12.4 | 2.6.0 |

---

## Verification

Run on H100:
```bash
chmod +x tools/verify_rebuilt_images.sh
HF_TOKEN=your_token ./tools/verify_rebuilt_images.sh
```

## Push to Docker Hub

```bash
docker push shikhar481/sglang-images:2a754e57
docker push shikhar481/sglang-images:2a754e57b052e249ed4f8572cb6f0069ba6a495e

docker push shikhar481/sglang-images:79961afa
docker push shikhar481/sglang-images:79961afa8281f98f380d11db45c8d4b6e66a574f

docker push shikhar481/sglang-images:ab4a83b2
docker push shikhar481/sglang-images:ab4a83b25909aa98330b838a224e4fe5c943e483
```
