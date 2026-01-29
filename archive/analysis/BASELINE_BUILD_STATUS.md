# Baseline Image Build Status Report

**Generated**: 2026-01-12 (Updated)
**Analysis by**: Claude Code

---

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| **HuggingFace Dataset (vLLM)** | 95 | - |
| **V1 Mapping commits** | 45 | - |
| **V2 Mapping commits** | 33 | - |
| **Total unique parent commits** | 78 | 100% |
| **Existing baseline images** | 48 | 62% |
| **Missing baseline images** | 30 | 38% |
| **Permanently failed** | 4 | 5% |

---

## Recent Builds (2026-01-12)

Successfully built with commit-specific fixes:

| Parent Commit | Fix Applied | Status |
|---------------|-------------|--------|
| `9bde5ba12709` | Python version constraint (<=3.12 -> <3.13) | SUCCESS |
| `bc8a8ce5ec37` | OOM retry with max_jobs=32 | SUCCESS |
| `a869baca73eb` | Lower parallelism (max_jobs=24) | SUCCESS |
| `f728ab8e3578` | Lower parallelism (max_jobs=24) | SUCCESS |
| `029c71de11bc` | Standard build | SUCCESS |
| `7206ce4ce112` | CUDA stubs library path fix | SUCCESS |

### Permanently Failed

| Parent Commit | Error | Reason |
|---------------|-------|--------|
| `0e74d797ce86` | Build infrastructure | Incompatible vLLM version |
| `2a0309a646b1` | Build infrastructure | Incompatible vLLM version |
| `51c31bc10ca7` | CUDA API mismatch | vLLM v0.4.0 requires CUDA 11 |

---

## Existing Baseline Images (47)

These baseline images exist on Docker Hub (`shikhar481/vllm_fixed_human_images:baseline-*`):

### Original 27 + Newly Built 20

| Parent Commit | Status |
|---------------|--------|
| `a4d577b37944` | `baseline-a4d577b37944` |
| `95baec828f3e` | `baseline-95baec828f3e` |
| `2b04c209ee98` | `baseline-2b04c209ee98` |
| `f508e03e7f2d` | `baseline-f508e03e7f2d` |
| `1d35662e6dc1` | `baseline-1d35662e6dc1` |
| `f721096d48a7` | `baseline-f721096d48a7` |
| `51f8aa90ad40` | `baseline-51f8aa90ad40` |
| `6dd55af6c9dd` | `baseline-6dd55af6c9dd` |
| `beebf4742af8` | `baseline-beebf4742af8` |
| `5c04bb8b863b` | `baseline-5c04bb8b863b` |
| `70363bccfac1` | `baseline-70363bccfac1` |
| `388596c91437` | `baseline-388596c91437` |
| `0fca3cdcf265` | `baseline-0fca3cdcf265` |
| `084a01fd3544` | `baseline-084a01fd3544` |
| `bd43973522ea` | `baseline-bd43973522ea` |
| `51e971d39e12` | `baseline-51e971d39e12` |
| `dd2a6a82e3f4` | `baseline-dd2a6a82e3f4` |
| `95a178f86120` | `baseline-95a178f86120` |
| `6a11fdfbb8d6` | `baseline-6a11fdfbb8d6` |
| `64172a976c8d` | `baseline-64172a976c8d` |
| `ebce310b7433` | `baseline-ebce310b7433` |
| `b0e96aaebbfb` | `baseline-b0e96aaebbfb` |
| `fbefc8a78d22` | `baseline-fbefc8a78d22` |
| `f1c852014603` | `baseline-f1c852014603` |
| `0032903a5bb7` | `baseline-0032903a5bb7` |
| `36fb68f94792` | `baseline-36fb68f94792` |
| `54600709b6d4` | `baseline-54600709b6d4` |
| `3cdfe1f38b2c` | `baseline-3cdfe1f38b2c` (NEW) |
| `005ae9be6c22` | `baseline-005ae9be6c22` (NEW) |
| `067c34a15594` | `baseline-067c34a15594` (NEW) |
| `10904e6d7550` | `baseline-10904e6d7550` (NEW) |
| `1da8f0e1ddda` | `baseline-1da8f0e1ddda` (NEW) |
| `25373b6c6cc2` | `baseline-25373b6c6cc2` (NEW) |
| `270a5da495d2` | `baseline-270a5da495d2` (NEW) |
| `2f385183f354` | `baseline-2f385183f354` (NEW) |
| `20478c4d3abc` | `baseline-20478c4d3abc` (NEW) |
| `3014c920dae5` | `baseline-3014c920dae5` (NEW) |
| `333681408fea` | `baseline-333681408fea` (NEW) |
| `3a1e6481586e` | `baseline-3a1e6481586e` (NEW) |
| `3cd91dc9555e` | `baseline-3cd91dc9555e` (NEW) |
| `526078a96c52` | `baseline-526078a96c52` (NEW) |
| `5b8a1fde8422` | `baseline-5b8a1fde8422` (NEW) |
| `5fc5ce0fe45f` | `baseline-5fc5ce0fe45f` (NEW) |
| `9bde5ba12709` | `baseline-9bde5ba12709` (NEW - Python fix) |
| `bc8a8ce5ec37` | `baseline-bc8a8ce5ec37` (NEW - OOM retry) |
| `a869baca73eb` | `baseline-a869baca73eb` (NEW - Low parallelism) |
| `f728ab8e3578` | `baseline-f728ab8e3578` (NEW - Low parallelism) |
| `029c71de11bc` | `baseline-029c71de11bc` (NEW - User requested) |
| `7206ce4ce112` | `baseline-7206ce4ce112` (NEW - CUDA stubs fix) |

---

## Commit-Specific Build Configurations

These configurations are defined in `build_baseline_images_v2.py`:

```python
RETRY_CONFIGS = {
    "9bde5ba12709": {
        "max_jobs": 32,
        "pre_build_patch": """
sed -i 's/<=3.12/<3.13/g' pyproject.toml 2>/dev/null || true
sed -i 's/<=3.12/<3.13/g' setup.py 2>/dev/null || true
"""
    },
    "bc8a8ce5ec37": {"max_jobs": 32},
    "a869baca73eb": {"max_jobs": 24},
    "f728ab8e3578": {"max_jobs": 24},
}
```

---

## Missing Baseline Images (30)

### High Priority - Needed for Benchmarks

| # | Parent Commit | Human Commit | Model |
|---|---------------|--------------|-------|
| 1 | `a6d795d59304` | `22dd9c27` | meta-llama/Llama-3.1-8B-Instruct |
| 3 | `733e7c9e95f5` | `35fad35a` | meta-llama/Llama-3.1-8B-Instruct |
| 4 | `8d59dbb00044` | `660470e5` | meta-llama/Llama-3.1-8B-Instruct |
| 5 | `3d925165f2b1` | `ad8d696a` | meta-llama/Llama-3.1-8B-Instruct |
| 6 | `8bb43b9c9ee8` | `c0569dbc` | Qwen/Qwen3-30B-A3B-FP8 |
| 7 | `acaea3bb0788` | `ccf02fcb` | ibm-ai-platform/Bamba-9B |
| 8 | `ec82c3e388b9` | `d55e446d` | meta-llama/Meta-Llama-3-8B |
| 9 | `cc466a32903d` | `d7740ea4` | meta-llama/Llama-3.1-8B-Instruct |
| 10 | `dd572c0ab3ef` | `dcc6cfb9` | Qwen/Qwen3-30B-A3B-FP8 |
| 11 | `4ce64e2df486` | `e493e485` | microsoft/phi-1_5 |
| 12 | `90f1e55421f1` | `e7b20426` | 01-ai/Yi-1.5-9B-Chat |
| 13 | `67abdbb42fdb` | `fc7b8d1e` | meta-llama/Llama-3.1-8B-Instruct |

### V2 Mapping - Remaining

| # | Parent Commit | Human Commit | Model |
|---|---------------|--------------|-------|
| 14 | `d263bd9df7b2` | `25ebed2f` | meta-llama/Llama-3.1-8B-Instruct |
| 15 | `edc4fa31888b` | `3b61cb45` | meta-llama/Llama-3.1-8B-Instruct |
| 16 | `c8d70e2437fe` | `4c822298` | meta-llama/Llama-3.1-8B-Instruct |
| 17 | `7d94577138e3` | `6d0734c5` | mistralai/Mistral-7B-Instruct-v0.3 |
| 18 | `63d635d17962` | `70b808fe` | Qwen/Qwen2-VL-7B |
| 19 | `f168b8572520` | `7661e92e` | nvidia/Nemotron-4-340B-Instruct |
| 20 | `76b494444fd8` | `8a4e5c5f` | meta-llama/Llama-3.1-8B-Instruct |
| 21 | `8c1e77fb585c` | `98f47f2a` | facebook/opt-125m |
| 22 | `e642ec962cf2` | `9f1710f1` | deepseek-ai/DeepSeek-V2-Lite-Chat |
| 23 | `4a18fd14ba4a` | `b2e0ad3b` | meta-llama/Llama-3.1-8B-Instruct |
| 24 | `3e1c76cf3a87` | `baeded25` | deepseek-ai/DeepSeek-V3 |
| 25 | `cf2f084d56a1` | `bfdb1ba5` | meta-llama/Llama-2-7b-chat-hf |
| 26 | `c34eeec58d3a` | `dae68969` | deepseek-ai/DeepSeek-R1 |
| 27 | `88faa466d788` | `eefbf4a6` | Qwen/Qwen3-30B-A3B-FP8 |
| 28 | `eb5741ad422f` | `fc542144` | meta-llama/Llama-3.1-8B-Instruct |

---

## Build Instructions

### Prerequisites

1. **GPU Machine**: H100 recommended (TORCH_CUDA_ARCH_LIST="9.0")
2. **Docker**: With NVIDIA runtime
3. **Docker Hub Access**: Login to push images
4. **Disk Space**: ~50GB per image build

### Quick Start

```bash
# Login to Docker Hub
docker login

# Build all missing baseline images
python3 build_baseline_images_v2.py --all

# Or dry-run first
python3 build_baseline_images_v2.py --dry-run
```

### Build Settings (Optimized)

```python
MAX_JOBS = 56          # Parallel compilation jobs (default)
NVCC_THREADS = 2       # NVCC threads per job
TORCH_CUDA_ARCH_LIST = "9.0"  # H100
BUILD_TIMEOUT = 5400   # 90 minutes per image
```

### Build Single Commit

```bash
# Build specific parent commit
python3 build_baseline_images_v2.py --commit 005ae9be

# Without pushing (local only)
python3 build_baseline_images_v2.py --commit 005ae9be --no-push
```

---

## Files Generated

| File | Description |
|------|-------------|
| `baseline_build_list.json` | All commits needing baselines |
| `build_baseline_images_v2.py` | Optimized build script with retry configs |
| `build_progress.json` | Real-time build progress |
| `build_errors/*.log` | Error logs for failed builds |

---

## Docker Hub Repositories

| Repository | Purpose |
|------------|---------|
| `ayushnangia16/nvidia-vllm-docker` | Original human commit images (105 images) |
| `shikhar481/vllm_fixed_human_images` | Fixed human images + baseline images |

### Image Naming Convention

- Human images: `shikhar481/vllm_fixed_human_images:{commit_hash_40char}`
- Baseline images: `shikhar481/vllm_fixed_human_images:baseline-{parent_hash_12char}`

---

## Next Steps: Running Benchmarks

After building baseline images, to get 3-way comparison data:

1. **Run baseline benchmark** using the new Docker image
2. **Compare against existing human/agent results**
3. **Update COMPREHENSIVE_BENCHMARK_ANALYSIS.md** with results

Example benchmark command:
```bash
# Inside baseline Docker container
python3 benchmarks/benchmark_serving.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --num-prompts 100 --request-rate 10
```

---

*Report updated 2026-01-12 by Claude Code*
