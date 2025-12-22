# Evaluation Results Analysis

**Generated:** 2024-12-22

**Total Runs Evaluated:** 833

---

## Executive Summary

| Outcome | Count | Percentage |
|---------|------:|------------|
| Successful with performance data | 109 | 13.1% |
| Agent failed to generate patch | 366 | 43.9% |
| Baseline test produced no output | 143 | 17.2% |
| Test environment issues | 133 | 16.0% |
| Git/infrastructure errors | 33 | 4.0% |
| No test script available | 13 | 1.6% |
| Other issues | 36 | 4.3% |

---

## 1. Successful Evaluations (109 runs)

These runs completed the full evaluation pipeline: the agent generated a patch,
the test script ran successfully on both baseline and patched code, and
performance timing was captured.

### 1.1 Performance Improvements (12 runs)

Runs where the agent's patch resulted in >5% speedup over baseline.

| Speedup | Agent | Model | Repo | Commit | Baseline (ms) | Patched (ms) |
|--------:|-------|-------|------|--------|-------------:|-------------:|
| 1.639x | trae | gpt-5 | vllm | `2f192835` | 0.81 | 0.49 |
| 1.585x | codex | gpt-5 | vllm | `2f192835` | 0.78 | 0.49 |
| 1.577x | trae | gpt-5 | vllm | `2f192835` | 0.79 | 0.50 |
| 1.568x | trae | claude-sonnet-45 | vllm | `2f192835` | 0.78 | 0.50 |
| 1.509x | codex | gpt-5 | vllm | `fc542144` | 0.80 | 0.53 |
| 1.482x | trae | claude-sonnet-45 | vllm | `2f192835` | 0.79 | 0.53 |
| 1.334x | trae | gpt-5 | vllm | `3a243095` | 0.10 | 0.08 |
| 1.322x | codex | gpt-5 | vllm | `3a243095` | 0.10 | 0.08 |
| 1.317x | trae | claude-sonnet-45 | vllm | `3a243095` | 0.10 | 0.07 |
| 1.300x | trae | gpt-5 | vllm | `3a243095` | 0.10 | 0.08 |
| 1.166x | trae | claude-sonnet-45 | vllm | `b55ed6ef` | 0.03 | 0.03 |
| 1.103x | trae | gpt-5 | vllm | `b55ed6ef` | 0.03 | 0.02 |

### 1.2 Performance Regressions (12 runs)

Runs where the agent's patch resulted in >5% slowdown compared to baseline.
These represent cases where the agent's "optimization" actually hurt performance.

| Speedup | Agent | Model | Repo | Commit | Baseline (ms) | Patched (ms) |
|--------:|-------|-------|------|--------|-------------:|-------------:|
| 0.134x | trae | claude-sonnet-45 | vllm | `7c01f706` | 1.13 | 8.44 |
| 0.233x | trae | gpt-5 | vllm | `7c01f706` | 1.13 | 4.84 |
| 0.237x | trae | gpt-5 | vllm | `7c01f706` | 1.14 | 4.80 |
| 0.242x | codex | gpt-5 | vllm | `7c01f706` | 1.14 | 4.70 |
| 0.422x | trae | gpt-5 | sglang | `bb3a3b66` | 0.22 | 0.53 |
| 0.433x | trae | claude-sonnet-45 | sglang | `bb3a3b66` | 0.23 | 0.53 |
| 0.511x | trae | gpt-5 | vllm | `83450458` | 3.99 | 7.81 |
| 0.513x | trae | claude-sonnet-45 | vllm | `83450458` | 4.00 | 7.81 |
| 0.517x | trae | gpt-5 | vllm | `83450458` | 3.98 | 7.70 |
| 0.877x | trae | gpt-5 | vllm | `89a84b0b` | 0.56 | 0.64 |
| 0.937x | trae | gpt-5 | sglang | `6f560c76` | 0.60 | 0.64 |
| 0.949x | trae | claude-sonnet-45 | vllm | `22d33bac` | 0.98 | 1.04 |

### 1.3 Neutral Results (85 runs)

Runs where performance changed by less than 5% (within measurement noise).

| Speedup | Agent | Model | Repo | Commit |
|--------:|-------|-------|------|--------|
| 1.036x | trae | gpt-5 | sglang | `a73c4df4` |
| 1.034x | trae | gpt-5 | sglang | `6b7038ba` |
| 1.028x | trae | gpt-5 | sglang | `e5db40dc` |
| 1.022x | trae | gpt-5 | vllm | `f26c4aee` |
| 1.020x | trae | claude-sonnet-45 | sglang | `c98e84c2` |
| 1.015x | trae | gpt-5 | vllm | `88693683` |
| 1.014x | trae | claude-sonnet-45 | vllm | `25ebed2f` |
| 1.013x | trae | claude-sonnet-45 | vllm | `93e5f3c5` |
| 1.013x | trae | claude-sonnet-45 | sglang | `9c064bf7` |
| 1.013x | trae | claude-sonnet-45 | vllm | `58eee5f2` |
| 1.013x | trae | claude-sonnet-45 | vllm | `89a84b0b` |
| 1.011x | trae | claude-sonnet-45 | sglang | `6fc17596` |
| 1.011x | trae | gpt-5 | vllm | `4c822298` |
| 1.010x | trae | claude-sonnet-45 | sglang | `6b7038ba` |
| 1.009x | trae | gpt-5 | vllm | `310aca88` |
| 1.007x | trae | gpt-5 | sglang | `6e2da515` |
| 1.006x | trae | gpt-5 | sglang | `2bd18e2d` |
| 1.006x | trae | gpt-5 | vllm | `9f1710f1` |
| 1.005x | trae | claude-sonnet-45 | vllm | `88693683` |
| 1.005x | trae | gpt-5 | sglang | `dd1012fc` |
| ... | | | | |
| *(showing 20 of 85 neutral runs)* | | | | |

---

## 2. Agent Did Not Generate Patch (366 runs)

**Root Cause:** The agent (TRAE, Codex, or OpenHands) did not produce a
`model_patch.diff` file. In 365 of 366 cases, the agent status was `error`,
indicating the agent crashed, timed out, or encountered an unrecoverable error.

**Impact:** These runs could not be evaluated because there was no patch to test.

### Breakdown by Agent/Model

| Agent | Model | Count | % of Category |
|-------|-------|------:|--------------:|
| trae | gpt-5 | 223 | 60.9% |
| trae | claude-sonnet-45 | 112 | 30.6% |
| trae | gpt-4o | 11 | 3.0% |
| codex | gpt-4o | 9 | 2.5% |
| openhands | gpt-5 | 6 | 1.6% |
| codex | gpt-5 | 4 | 1.1% |
| trae | o4-mini | 1 | 0.3% |

### Sample Runs

| Item ID | Agent | Model | Commit | Agent Status |
|---------|-------|-------|--------|--------------|
| sglang_073_e822e590 | trae | claude-sonnet-45 | `e822e590` | error |
| sglang_077_f0815419 | trae | claude-sonnet-45 | `f0815419` | error |
| sglang_079_ff00895c | trae | claude-sonnet-45 | `ff00895c` | error |
| sglang_073_e822e590 | trae | gpt-5 | `e822e590` | error |
| sglang_077_f0815419 | trae | gpt-5 | `f0815419` | error |
| sglang_079_ff00895c | trae | gpt-5 | `ff00895c` | error |
| sglang_045_9c088829 | trae | gpt-5 | `9c088829` | error |
| sglang_035_7ce36068 | trae | gpt-5 | `7ce36068` | error |
| sglang_028_6b7038ba | trae | gpt-5 | `6b7038ba` | error |
| sglang_064_d1112d85 | trae | gpt-5 | `d1112d85` | error |
| ... | | | | |
| *(showing 10 of 366 runs)* | | | | |

---

## 3. Baseline Test Produced No Output (143 runs)

**Root Cause:** The test script executed but did not print any JSON output.
This typically happens when:
- The script crashes before reaching the timing code
- Import errors prevent the script from running
- Exceptions are raised but not caught

**Impact:** Cannot measure baseline performance, so no comparison is possible.

### Breakdown by Agent/Model

| Agent | Model | Count |
|-------|-------|------:|
| trae | gpt-5 | 75 |
| trae | claude-sonnet-45 | 41 |
| codex | gpt-5 | 25 |
| trae | gpt-4o | 2 |

### Sample Runs

| Item ID | Repo | Commit | Error |
|---------|------|--------|-------|
| sglang_023_564a898a | sglang | `564a898a` | Baseline test failed to produce output |
| sglang_040_915140fd | sglang | `915140fd` | Baseline test failed to produce output |
| sglang_052_ac971ff6 | sglang | `ac971ff6` | Baseline test failed to produce output |
| sglang_029_6cb00c63 | sglang | `6cb00c63` | Baseline test failed to produce output |
| sglang_066_dc188132 | sglang | `dc188132` | Baseline test failed to produce output |
| sglang_054_b1709305 | sglang | `b1709305` | Baseline test failed to produce output |
| sglang_056_b77a02cd | sglang | `b77a02cd` | Baseline test failed to produce output |
| sglang_024_5e023301 | sglang | `5e023301` | Baseline test failed to produce output |
| sglang_046_9c745d07 | sglang | `9c745d07` | Baseline test failed to produce output |
| sglang_040_915140fd | sglang | `915140fd` | Baseline test failed to produce output |

---

## 4. Import Errors (76 runs)

**Root Cause:** The test script attempted to import a Python module that is not
installed in the evaluation environment.

**Impact:** Test cannot run; no performance data collected.

### Missing Modules

| Module | Affected Runs | Notes |
|--------|-------------:|-------|
| `transformers` | 24 | HuggingFace transformers not installed |
| `vllm._C` | 14 | Native C extension not compiled |
| `decord` | 6 | Audio/video processing library |
| `outlines` | 6 |  |
| `librosa` | 6 | Audio/video processing library |
| `pybase64` | 4 |  |
| `transformers_neuronx` | 4 |  |
| `vllm` | 3 |  |
| `deep_ep` | 3 |  |
| `benchmark.kernels.minmax_text_01_lighting_attention` | 2 |  |
| `einops` | 2 |  |
| `outlines_core.fsm` | 1 |  |
| `vllm.core.block` | 1 |  |

### Sample Runs

| Item ID | Repo | Commit | Missing Module |
|---------|------|--------|----------------|
| sglang_039_912788c0 | sglang | `912788c0` | `transformers` |
| sglang_043_93470a14 | sglang | `93470a14` | `decord` |
| sglang_021_4418f599 | sglang | `4418f599` | `transformers` |
| sglang_061_c2f212d6 | sglang | `c2f212d6` | `benchmark.kernels.minmax_text_01_lighting_attention` |
| sglang_000_021f76e4 | sglang | `021f76e4` | `transformers` |
| sglang_063_cd7e32e2 | sglang | `cd7e32e2` | `transformers` |
| sglang_014_2a413829 | sglang | `2a413829` | `transformers` |
| sglang_038_8f8f96a6 | sglang | `8f8f96a6` | `transformers` |
| sglang_008_205d5cb4 | sglang | `205d5cb4` | `transformers` |
| sglang_026_6a2941f4 | sglang | `6a2941f4` | `outlines` |

---

## 5. Target Not Resolved (55 runs)

**Root Cause:** The test script could not import or locate the optimization target
(the specific function/class being benchmarked). This often happens due to:
- API changes between vLLM versions
- Missing dependencies
- Renamed or moved modules

### Error Messages

| Error | Count |
|-------|------:|
| `cannot import name 'default_cache_dir' from 'triton.runtime....` | 22 |
| `Failed to import vLLM components: cannot import name 'Prompt...` | 15 |
| `type object 'P2pNcclEngine' has no attribute 'extract_kv_fro...` | 5 |
| `module 'vllm.model_executor.models.utils' has no attribute '...` | 3 |
| `module 'vllm.model_executor.layers.quantization.utils.fp8_ut...` | 2 |
| `cannot import name 'build_regex_from_schema' from 'outlines....` | 2 |
| `module 'vllm.core.block_manager' has no attribute 'UncachedB...` | 2 |
| `module 'vllm.v1.engine.output_processor' has no attribute 'R...` | 2 |
| `Failed to import required classes: cannot import name 'XGram...` | 1 |
| `cannot import name 'cuda_utils' from partially initialized m...` | 1 |

---

## 6. Optimization Path Not Triggered (2 runs)

**Root Cause:** The test ran successfully, but the specific optimization code path
was not executed during the benchmark. For example, "Custom allreduce not initialized"
indicates the multi-GPU communication optimization wasn't active.

**Impact:** Cannot measure the optimization because it wasn't used.

| Item ID | Repo | Commit | Message |
|---------|------|--------|---------|
| sglang_011_25e1816e | sglang | `25e1816e` | Custom allreduce not initialized |
| sglang_011_25e1816e | sglang | `25e1816e` | Custom allreduce not initialized |

---

## 7. Git Worktree Errors (33 runs)

**Root Cause:** The evaluation system could not create a git worktree at the
specified pre-commit hash. This typically happens when:
- The commit doesn't exist in the local repository clone
- The repository needs to be fetched/updated
- Git worktree limits have been reached

**Impact:** Cannot check out the baseline code for comparison.

### Affected Commits

| Pre-Commit | Count |
|------------|------:|
| `067c34a1` | 1 |
| `005ae9be` | 1 |
| `25373b6c` | 1 |
| `0df4d9b0` | 1 |
| `89ac266b` | 1 |
| `a869baca` | 1 |
| `526078a9` | 1 |
| `1d35662e` | 1 |
| `3e1c76cf` | 1 |
| `733e7c9e` | 1 |

All 33 affected runs are from **codex/gpt-5**.

---

## 8. No Test Script Available (27 runs)

**Root Cause:** The HuggingFace test script dataset does not contain a test
script for the specified human commit.

**Impact:** Cannot evaluate without a benchmark script.

### Commits Without Test Scripts

| Human Commit | Affected Runs |
|--------------|-------------:|
| `f06e90c2` | 2 |
| `fbcbb263` | 2 |
| `e88dd482` | 2 |
| `f0653886` | 2 |
| `5e5c8e09` | 2 |
| `81ede99c` | 2 |
| `b9986454` | 1 |

---

## 9. Evaluation Marked as No Patch (31 runs)

**Root Cause:** The agent generated a patch file, but the evaluation system
still classified the result as "no_patch". This typically happens when:
- The patch file was empty
- The patch only modified non-Python files (e.g., README, configs)
- The patch format was invalid

### Breakdown by Agent/Model

| Agent | Model | Count |
|-------|-------|------:|
| codex | gpt-5 | 19 |
| trae | gpt-5 | 8 |
| trae | claude-sonnet-45 | 4 |

---

## 10. Patch Apply Failed (2 runs)

**Root Cause:** The agent's patch could not be applied to the codebase.
This typically happens due to:
- Patch conflicts with the target code
- Invalid patch format
- File paths in the patch don't match the repository

| Item ID | Agent | Model | Commit |
|---------|-------|-------|--------|
| vllm_core-0016 | codex | gpt-5 | `310aca88` |
| vllm_core-0029 | codex | gpt-5 | `660470e5` |

---

## 11. Other Issues

**Other test failures:** 0 runs
**Uncategorized:** 3 runs

---

## Recommendations for Improving Evaluation Success Rate

### High Priority

1. **Improve Agent Reliability (would fix 366 runs / 44%)**
   - Investigate why agents are failing with `error` status
   - Add better error handling and recovery in agent code
   - Consider increasing agent timeouts if applicable

2. **Fix Test Environment (would fix ~200 runs / 24%)**
   - Install missing modules: `transformers`, `librosa`, `decord`, `outlines`
   - Ensure `vllm._C` extension is compiled
   - Update test scripts to handle API changes between versions

### Medium Priority

3. **Fix Git Repository Issues (would fix 33 runs / 4%)**
   - Fetch all remote commits before evaluation
   - Clean up stale worktrees with `git worktree prune`

4. **Add Missing Test Scripts (would fix 27 runs / 3%)**
   - Generate test scripts for commits: `5e5c8e09`, `81ede99c`, `b9986454`, `e88dd482`, `f0653886`, `f06e90c2`, `fbcbb263`

### Low Priority

5. **Investigate Empty Patches (31 runs)**
   - Check if agents are producing non-Python changes
   - Validate patch format before evaluation
