# vLLM-style SGLang Builds Summary

## Input vs Output

| Metric | Count |
|--------|-------|
| **Dates provided** | 4 |
| **Total commits provided** | 8 (2 per date) |
| **Commits built** | 4 |
| **Commits skipped** | 4 |
| **Images pushed** | 4 |

---

## What You Provided (4 dates, 8 commits)

| Date | Commit | Type | Status | Reason |
|------|--------|------|--------|--------|
| 2025-05-02 | 1acca3a2 | human | **BUILT** | - |
| 2025-05-02 | 6ea1e6ac | parent | **BUILT** | - |
| 2025-06-11 | 021f76e4 | human | SKIPPED | Hung indefinitely on Modal |
| 2025-06-11 | 777688b8 | parent | SKIPPED | Hung indefinitely on Modal |
| 2025-07-08 | 136c6e04 | human | **BUILT** | - |
| 2025-07-08 | a37e1247 | parent | **BUILT** | - |
| 2025-07-26 | 3212c2ad | human | SKIPPED | InternVL model not supported |
| 2025-07-26 | 53475674 | parent | SKIPPED | InternVL model not supported |

---

## Images Pushed to DockerHub

All images at `shikhar481/sglang-images`:

| Image Tag | SGLang | torch | triton | Digest |
|-----------|--------|-------|--------|--------|
| `1acca3a2-vllm-style` | 0.4.6.post2 | 2.6.0+cu124 | 3.2.0 | `sha256:4568bda8298a3bafbaa4551e641cff4c2bde3e60f73593e8e4af89d26fb33c8d` |
| `6ea1e6ac-vllm-style` | 0.4.6.post2 | 2.6.0+cu124 | 3.2.0 | `sha256:dbc22924b940275260e907505ef3bba700378cf80688bfbf3a9ef89785c729ab` |
| `136c6e04-vllm-style` | 0.4.9 | 2.7.1+cu126 | 3.3.1 | `sha256:3a1e3e253874323fe62890a91ade5cc4ac6c3c1bd37cffea32e9e3fec504e6fe` |
| `a37e1247-vllm-style` | 0.4.9 | 2.7.1+cu126 | 3.3.1 | `sha256:8f264d1bfebc21ab657278682e8cc11e3f749087e98945da6164ffedf0e2f517` |

---

## Why Commits Were Skipped

### 2025-06-11 (021f76e4, 777688b8)
From `benchmark_notes.txt`:
> No build errors occurred for both Dockerfiles. Both benchmarks are running indefinitely on Modal without completing.

**Decision**: Likely a memory leak or infinite loop - not worth building.

### 2025-07-26 (3212c2ad, 53475674)
From `benchmark_notes.txt`:
> ValueError: InternVLChatModel architecture is not registered as a processor in this SGLang version.

**Decision**: Model architecture not supported in that SGLang version - benchmark would fail anyway.

---

## Key Technical Details

### Approach Differences

| Aspect | Our Original | vLLM-style |
|--------|--------------|------------|
| sgl-kernel | Build from source | PyPI wheel |
| flashinfer | Pre-built wheel | Build from source |
| Python | 3.11 | 3.10 |
| deep_gemm | Included | May be missing |

### CUDA Matching

| Date | torch | CUDA wheel | Base Image | Match? |
|------|-------|------------|------------|--------|
| 2025-05-02 | 2.6.0 | cu124 | 12.4 | **YES** |
| 2025-07-08 | 2.7.1 | cu126 | 12.4 | NO (mismatch) |

**Recommendation**: Prioritize 2025-05-02 builds for testing (proper CUDA matching).

---

## Test Commands

### Priority 1: torch 2.6.0 + triton 3.2.0 (proper CUDA match)
```bash
docker run --rm --gpus all -p 30000:30000 -e HF_TOKEN=<token> \
  shikhar481/sglang-images:1acca3a2-vllm-style \
  python -m sglang.launch_server --model google/gemma-2-2b-it --port 30000
```

### Priority 2: torch 2.7.1 + triton 3.3.1 (newer triton)
```bash
docker run --rm --gpus all -p 30000:30000 -e HF_TOKEN=<token> \
  shikhar481/sglang-images:136c6e04-vllm-style \
  python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --port 30000
```

---

## Dockerfiles Created

```
src/benchmark/docker/vllm_commits/
├── Dockerfile.1acca3a2   # 2025-05-02, torch 2.6.0
├── Dockerfile.6ea1e6ac   # 2025-05-02, torch 2.6.0 (parent)
├── Dockerfile.136c6e04   # 2025-07-08, torch 2.7.1
├── Dockerfile.a37e1247   # 2025-07-08, torch 2.7.1 (parent)
└── VLLM_BUILDS_SUMMARY.md  # This file
```

---

*Generated: 2026-01-13*
