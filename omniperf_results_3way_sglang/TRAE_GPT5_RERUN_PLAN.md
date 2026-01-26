# trae_gpt5 Benchmark Rerun Plan

## Status: ✅ COMPLETED (2026-01-26)

## Executive Summary

**Problem:** 6 commits for trae_gpt5 used empty (0-byte) patches from old runs instead of valid patches from newer runs. This caused "unrecognized input" errors during benchmark.

**Root Cause:** Benchmarks ran at ~16:00-17:00 before the script fix was committed at 21:47. The fix added proper directory ordering and `st_size > 0` checks.

**Solution:** Re-run benchmarks for 6 commits with the now-fixed script.

## ✅ RERUN RESULTS (Completed 2026-01-26)

| Commit | Before | After | Throughput |
|--------|--------|-------|------------|
| e3ec6bf4 | EMPTY_PATCH | ✅ SUCCESS | 900.72 tok/s |
| da47621c | EMPTY_PATCH | ✅ SUCCESS | 1013.95 tok/s |
| a191a0e4 | EMPTY_PATCH | ✅ SUCCESS | 1288.50 tok/s |
| 31589e17 | EMPTY_PATCH | ✅ SUCCESS | 1422.07 tok/s |
| 73b13e69 | EMPTY_PATCH | ✅ SUCCESS | 1215.77 tok/s |
| 205d5cb4 | EMPTY_PATCH | ✅ SUCCESS | 1288.33 tok/s |

### Success Rate Improvement
- **Before:** 2/17 (12%)
- **After:** 8/17 (47%)
- **Improvement:** +6 commits (+35 percentage points)

### Fixes Applied During Rerun
1. Fixed NVIDIA driver mismatch (reloaded kernel modules)
2. Installed `uv` package manager
3. Cloned and unshallowed SGLang repo
4. Pinned transformers to `>=4.44.0,<5.0.0` to avoid API breaking changes

---

## Commits to Re-run

| Commit | PR# | Subject | Patch Size | Expected Outcome |
|--------|-----|---------|------------|------------------|
| e3ec6bf4 | 6814 | Minor speed up block_quant_dequant | 2,850 bytes | Should work |
| da47621c | 7058 | Minor speedup topk postprocessing | 4,525 bytes | Should work |
| a191a0e4 | 6593 | Improve performance of two batch overlap | 2,473 bytes | Should work |
| 31589e17 | 6668 | Speed up when having padding tokens | 5,022 bytes | Should work |
| 73b13e69 | 7285 | Optimize DP attn scheduling | 9,080 bytes | Should work |
| 205d5cb4 | 6356 | Optimize local attention memory allocation | 7,106 bytes | Should work |

**NOT included:** b1e5a33a (ABI_MISMATCH - would fail regardless)

---

## Verification Completed

Patch finding logic verified to work correctly:
```
e3ec6bf4: FOUND (2,850 bytes) in 2025-11-16_09-27-51
da47621c: FOUND (4,525 bytes) in 2025-11-16_09-27-51
a191a0e4: FOUND (2,473 bytes) in 2025-11-16_09-27-51
31589e17: FOUND (5,022 bytes) in 2025-11-16_09-27-51
73b13e69: FOUND (9,080 bytes) in 2025-11-16_09-27-51
205d5cb4: FOUND (7,106 bytes) in 2025-11-16_09-27-51
```

---

## Pre-requisites

### 1. Machine Requirements
- **GPU:** NVIDIA H100 (80GB) or equivalent
- **Python:** 3.10+
- **CUDA:** 12.x with working NVML drivers

### 2. Script Path Fix
The script uses `/root/OmniPerf-Bench` but repo is at `/home/ubuntu/OmniPerf-Bench`.

**Option A:** Run as root user
**Option B:** Create symlink: `sudo ln -s /home/ubuntu/OmniPerf-Bench /root/OmniPerf-Bench`
**Option C:** Modify script paths (sed command below)

```bash
# Option C: Update paths in script
cd /home/ubuntu/OmniPerf-Bench/omniperf_results_3way_sglang
sed -i 's|/root/OmniPerf-Bench|/home/ubuntu/OmniPerf-Bench|g' run_isolated_benchmarks.py
```

---

## Benchmark Commands

### Single Command (All 6 commits sequentially)
```bash
cd /home/ubuntu/OmniPerf-Bench/omniperf_results_3way_sglang

# Run all 6 commits for trae_gpt5 only
for commit in e3ec6bf4 da47621c a191a0e4 31589e17 73b13e69 205d5cb4; do
    echo "=========================================="
    echo "Running: $commit"
    echo "=========================================="
    python3 run_isolated_benchmarks.py --commit "$commit" --variants trae_gpt5 --skip-clone
done
```

### Individual Commands (if running one at a time)
```bash
cd /home/ubuntu/OmniPerf-Bench/omniperf_results_3way_sglang

# 1. e3ec6bf4 - Minor speed up block_quant_dequant
python3 run_isolated_benchmarks.py --commit e3ec6bf4 --variants trae_gpt5 --skip-clone

# 2. da47621c - Minor speedup topk postprocessing
python3 run_isolated_benchmarks.py --commit da47621c --variants trae_gpt5 --skip-clone

# 3. a191a0e4 - Improve performance of two batch overlap
python3 run_isolated_benchmarks.py --commit a191a0e4 --variants trae_gpt5 --skip-clone

# 4. 31589e17 - Speed up when having padding tokens
python3 run_isolated_benchmarks.py --commit 31589e17 --variants trae_gpt5 --skip-clone

# 5. 73b13e69 - Optimize DP attn scheduling
python3 run_isolated_benchmarks.py --commit 73b13e69 --variants trae_gpt5 --skip-clone

# 6. 205d5cb4 - Optimize local attention memory allocation
python3 run_isolated_benchmarks.py --commit 205d5cb4 --variants trae_gpt5 --skip-clone
```

### Dry Run (verify without running)
```bash
python3 run_isolated_benchmarks.py --commit e3ec6bf4 --variants trae_gpt5 --dry-run
```

---

## Expected Benchmark Commands (from PERF_COMMAND_CONFIG)

The script uses these serving benchmark commands:

| Commit | Original Command |
|--------|------------------|
| e3ec6bf4 | `python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100` |
| da47621c | `python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100` |
| a191a0e4 | `python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100` |
| 31589e17 | `python3 -m sglang.bench_serving --backend sglang --model deepseek-ai/DeepSeek-V3-0324 ...` (falls back to Llama-3.1-8B) |
| 73b13e69 | `python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100` |
| 205d5cb4 | `python3 -m sglang.bench_serving --backend sglang --model meta-llama/Llama-4-Maverick-17B...` (falls back to Llama-3.1-8B) |

**Note:** Large models (DeepSeek-V3, Llama-4-Maverick) fall back to `meta-llama/Llama-3.1-8B-Instruct` on single GPU.

---

## Post-Run Verification

After running, verify results in JSON files:

```bash
# Check results for each commit
for commit in e3ec6bf4 da47621c a191a0e4 31589e17 73b13e69 205d5cb4; do
    echo "=== $commit ==="
    jq '.variants.trae_gpt5 | {status, patch_applied, patch_message, throughput: .metrics.throughput_tokens_per_sec}' \
        isolated_benchmark_results/${commit}_isolated.json 2>/dev/null || echo "Not found"
done
```

### Expected Success Criteria
- `patch_applied: true`
- `status: "success"`
- `throughput_tokens_per_sec: <number>` (should be ~1000-1300 tok/s for Llama-3.1-8B)

---

## Patch Quality Analysis

Sample patch review (e3ec6bf4):

**Human PR approach:**
```python
# Vectorized with repeat_interleave
x_scale_repeat = x_s.repeat_interleave(block_n, dim=-2).repeat_interleave(block_k, dim=-1)
return (x_q_block.to(torch.float32) * x_scale_repeat).to(dtype)
```

**trae_gpt5 approach:**
```python
# Also vectorized with repeat_interleave
x_scale_repeat = x_s.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
x_dq_block = x_dq_block * x_scale_repeat[:n, :k]
```

The approaches are semantically equivalent. Both eliminate nested loops with vectorized operations.

---

## Timeline Estimate

Per commit (typical):
- Venv creation: ~2 min
- SGLang install: ~5-10 min
- Server startup: ~2-3 min
- Benchmark: ~2-3 min
- Total: ~12-18 min per commit

**Total for 6 commits: ~1.5-2 hours**

---

## Troubleshooting

### If patch fails to apply
Check git status and ensure base commit is correct:
```bash
cd /home/ubuntu/OmniPerf-Bench/omniperf_results_3way_sglang/sglang-repo
git status
git log --oneline -3
```

### If server fails to start
Check server logs:
```bash
cat /tmp/sglang-venvs/<commit>/server_startup.log
```

### If benchmark times out
Increase SERVER_TIMEOUT in script (currently 300s):
```python
SERVER_TIMEOUT = 600  # 10 minutes
```

---

## Summary

| What | Status |
|------|--------|
| Commits identified | 6 of 7 (b1e5a33a excluded due to ABI issue) |
| Patches verified | All 6 found and valid |
| Script fix verified | Correct directory ordering + size check |
| GPU required | Yes (H100 or equivalent) |
| Estimated time | ~2 hours |

**Next step:** Run on a machine with GPU access.
