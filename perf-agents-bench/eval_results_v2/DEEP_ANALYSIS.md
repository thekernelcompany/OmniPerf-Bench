# Deep Evaluation Analysis - All 833 Runs

**Generated:** 2024-12-22

This report provides the **actual root cause** for every evaluation run, not just high-level categories.

---

## Summary

| Category | Count | % | Description |
|----------|------:|--:|-------------|
| SUCCESS_IMPROVEMENT | 12 | 1.4% | Performance improved >5% |
| SUCCESS_NEUTRAL | 85 | 10.2% | Performance within 5% |
| SUCCESS_REGRESSION | 12 | 1.4% | Performance degraded >5% |
| AGENT_TIMEOUT | 1 | 0.1% | Agent timed out (>1 hour) |
| AGENT_NO_PATCH | 365 | 43.8% | Agent errored without producing patch |
| BASELINE_OOM | 2 | 0.2% | GPU out of memory |
| BASELINE_CUDA_ERROR | 1 | 0.1% | CUDA/GPU error during test |
| BASELINE_IMPORT_ERROR | 29 | 3.5% | Missing Python module |
| BASELINE_ATTRIBUTE_ERROR | 8 | 1.0% | AttributeError in test |
| BASELINE_TYPE_ERROR | 55 | 6.6% | TypeError in test (API mismatch) |
| BASELINE_RUNTIME_ERROR | 6 | 0.7% | RuntimeError in test |
| BASELINE_ASSERTION | 5 | 0.6% | AssertionError in test |
| BASELINE_EXCEPTION | 36 | 4.3% | Other exception in test |
| BASELINE_UNKNOWN | 1 | 0.1% | Unknown test failure |
| TEST_IMPORT_ERROR | 76 | 9.1% | Test script import error |
| TARGET_NOT_RESOLVED | 55 | 6.6% | Cannot find optimization target |
| OPT_PATH_NOT_HIT | 2 | 0.2% | Optimization path not triggered |
| GIT_WORKTREE_FAILED | 33 | 4.0% | Git commit not in local repo |
| NO_TEST_SCRIPT | 13 | 1.6% | No test script for this commit |
| PATCH_INVALID | 31 | 3.7% | Patch marked as invalid |
| PATCH_APPLY_FAILED | 2 | 0.2% | Patch failed to apply |
| UNKNOWN | 3 | 0.4% | Uncategorized |

---

## SUCCESS_IMPROVEMENT (12 runs, 1.4%)

**Description:** Performance improved >5%

### Speedup: 1.568x (1 runs)

**Root Cause:** 1.568x improvement

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0013 | trae | claude-sonnet-45 | vllm | `2f192835` |

### Speedup: 1.166x (1 runs)

**Root Cause:** 1.166x improvement

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0066 | trae | claude-sonnet-45 | vllm | `b55ed6ef` |

### Speedup: 1.482x (1 runs)

**Root Cause:** 1.482x improvement

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0013 | trae | claude-sonnet-45 | vllm | `2f192835` |

### Speedup: 1.317x (1 runs)

**Root Cause:** 1.317x improvement

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0021 | trae | claude-sonnet-45 | vllm | `3a243095` |

### Speedup: 1.334x (1 runs)

**Root Cause:** 1.334x improvement

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0020 | trae | gpt-5 | vllm | `3a243095` |

### Speedup: 1.577x (1 runs)

**Root Cause:** 1.577x improvement

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0012 | trae | gpt-5 | vllm | `2f192835` |

### Speedup: 1.300x (1 runs)

**Root Cause:** 1.300x improvement

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0020 | trae | gpt-5 | vllm | `3a243095` |

### Speedup: 1.639x (1 runs)

**Root Cause:** 1.639x improvement

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0012 | trae | gpt-5 | vllm | `2f192835` |

### Speedup: 1.103x (1 runs)

**Root Cause:** 1.103x improvement

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0063 | trae | gpt-5 | vllm | `b55ed6ef` |

### Speedup: 1.509x (1 runs)

**Root Cause:** 1.509x improvement

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0097 | codex | gpt-5 | vllm | `fc542144` |

### Speedup: 1.322x (1 runs)

**Root Cause:** 1.322x improvement

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0021 | codex | gpt-5 | vllm | `3a243095` |

### Speedup: 1.585x (1 runs)

**Root Cause:** 1.585x improvement

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0013 | codex | gpt-5 | vllm | `2f192835` |

---

## SUCCESS_NEUTRAL (85 runs, 10.2%)

**Description:** Performance within 5%

### Speedup: 1.000x (7 runs)

**Root Cause:** 1.000x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_058_bc3f6db2 | trae | claude-sonnet-45 | sglang | `bc3f6db2` |
| sglang_045_9c088829 | trae | claude-sonnet-45 | sglang | `9c088829` |
| vllm_bedrock_sonnet45-0016 | trae | claude-sonnet-45 | vllm | `310aca88` |
| vllm_bedrock_sonnet45-0019 | trae | claude-sonnet-45 | vllm | `35fad35a` |
| vllm_core-0018 | trae | gpt-5 | vllm | `35fad35a` |
| vllm_core-0067 | trae | gpt-5 | vllm | `bc7c4d20` |
| vllm_core-0048 | trae | gpt-5 | vllm | `93e5f3c5` |

### Speedup: 1.003x (6 runs)

**Root Cause:** 1.003x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_067_dc67d976 | trae | claude-sonnet-45 | sglang | `dc67d976` |
| sglang_068_dd1012fc | trae | claude-sonnet-45 | sglang | `dd1012fc` |
| sglang_044_9c064bf7 | trae | gpt-5 | sglang | `9c064bf7` |
| vllm_core-0095 | trae | gpt-5 | vllm | `fe66b347` |
| vllm_core-0056 | codex | gpt-5 | vllm | `9badee53` |
| vllm_core-0099 | codex | gpt-5 | vllm | `fe66b347` |

### Speedup: 0.999x (5 runs)

**Root Cause:** 0.999x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_030_6e2da515 | trae | claude-sonnet-45 | sglang | `6e2da515` |
| sglang_037_880221bd | trae | claude-sonnet-45 | sglang | `880221bd` |
| sglang_037_880221bd | trae | gpt-5 | sglang | `880221bd` |
| sglang_032_6fc17596 | trae | gpt-5 | sglang | `6fc17596` |
| vllm_core-0040 | trae | gpt-5 | vllm | `88693683` |

### Speedup: 0.998x (5 runs)

**Root Cause:** 0.998x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_042_9216b106 | trae | claude-sonnet-45 | sglang | `9216b106` |
| sglang_007_1bf1cf19 | trae | gpt-5 | sglang | `1bf1cf19` |
| sglang_027_6b231325 | trae | gpt-5 | sglang | `6b231325` |
| vllm_bedrock_sonnet45-0059 | trae | claude-sonnet-45 | vllm | `9f1710f1` |
| vllm_bedrock_sonnet45-0016 | trae | claude-sonnet-45 | vllm | `310aca88` |

### Speedup: 1.005x (4 runs)

**Root Cause:** 1.005x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_064_d1112d85 | trae | claude-sonnet-45 | sglang | `d1112d85` |
| sglang_068_dd1012fc | trae | gpt-5 | sglang | `dd1012fc` |
| sglang_017_2ed68d7a | trae | gpt-5 | sglang | `2ed68d7a` |
| vllm_bedrock_sonnet45-0043 | trae | claude-sonnet-45 | vllm | `88693683` |

### Speedup: 1.002x (4 runs)

**Root Cause:** 1.002x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_034_79961afa | trae | claude-sonnet-45 | sglang | `79961afa` |
| sglang_007_1bf1cf19 | trae | claude-sonnet-45 | sglang | `1bf1cf19` |
| sglang_010_25c83fff | trae | gpt-5 | sglang | `25c83fff` |
| vllm_bedrock_sonnet45-0023 | trae | claude-sonnet-45 | vllm | `4c822298` |

### Speedup: 1.013x (4 runs)

**Root Cause:** 1.013x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_044_9c064bf7 | trae | claude-sonnet-45 | sglang | `9c064bf7` |
| vllm_bedrock_sonnet45-0051 | trae | claude-sonnet-45 | vllm | `93e5f3c5` |
| vllm_bedrock_sonnet45-0044 | trae | claude-sonnet-45 | vllm | `89a84b0b` |
| vllm_bedrock_sonnet45-0026 | trae | claude-sonnet-45 | vllm | `58eee5f2` |

### Speedup: 0.989x (3 runs)

**Root Cause:** 0.989x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_033_73b13e69 | trae | claude-sonnet-45 | sglang | `73b13e69` |
| sglang_034_79961afa | trae | gpt-5 | sglang | `79961afa` |
| sglang_062_c98e84c2 | trae | gpt-5 | sglang | `c98e84c2` |

### Speedup: 1.001x (2 runs)

**Root Cause:** 1.001x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_010_25c83fff | trae | claude-sonnet-45 | sglang | `25c83fff` |
| vllm_core-0066 | codex | gpt-5 | vllm | `b55ed6ef` |

### Speedup: 1.004x (2 runs)

**Root Cause:** 1.004x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_016_2bd18e2d | trae | claude-sonnet-45 | sglang | `2bd18e2d` |
| vllm_bedrock_sonnet45-0071 | trae | claude-sonnet-45 | vllm | `bc7c4d20` |

### Speedup: 0.988x (2 runs)

**Root Cause:** 0.988x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_022_5239d795 | trae | claude-sonnet-45 | sglang | `5239d795` |
| sglang_033_73b13e69 | trae | gpt-5 | sglang | `73b13e69` |

### Speedup: 1.011x (2 runs)

**Root Cause:** 1.011x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_032_6fc17596 | trae | claude-sonnet-45 | sglang | `6fc17596` |
| vllm_core-0022 | trae | gpt-5 | vllm | `4c822298` |

### Speedup: 0.982x (2 runs)

**Root Cause:** 0.982x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_031_6f560c76 | trae | claude-sonnet-45 | sglang | `6f560c76` |
| sglang_001_09deb20d | trae | gpt-5 | sglang | `09deb20d` |

### Speedup: 0.994x (2 runs)

**Root Cause:** 0.994x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_025_62757db6 | trae | claude-sonnet-45 | sglang | `62757db6` |
| sglang_035_7ce36068 | trae | claude-sonnet-45 | sglang | `7ce36068` |

### Speedup: 0.990x (2 runs)

**Root Cause:** 0.990x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_072_e5db40dc | trae | claude-sonnet-45 | sglang | `e5db40dc` |
| sglang_064_d1112d85 | trae | gpt-5 | sglang | `d1112d85` |

### Speedup: 0.996x (2 runs)

**Root Cause:** 0.996x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_027_6b231325 | trae | claude-sonnet-45 | sglang | `6b231325` |
| sglang_067_dc67d976 | trae | gpt-5 | sglang | `dc67d976` |

### Speedup: 0.983x (2 runs)

**Root Cause:** 0.983x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_017_2ed68d7a | trae | claude-sonnet-45 | sglang | `2ed68d7a` |
| sglang_042_9216b106 | trae | gpt-5 | sglang | `9216b106` |

### Speedup: 0.997x (2 runs)

**Root Cause:** 0.997x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_058_bc3f6db2 | trae | gpt-5 | sglang | `bc3f6db2` |
| vllm_core-0025 | trae | gpt-5 | vllm | `58eee5f2` |

### Speedup: 1.006x (2 runs)

**Root Cause:** 1.006x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_016_2bd18e2d | trae | gpt-5 | sglang | `2bd18e2d` |
| vllm_core-0056 | trae | gpt-5 | vllm | `9f1710f1` |

### Speedup: 0.959x (2 runs)

**Root Cause:** 0.959x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0056 | trae | claude-sonnet-45 | vllm | `9badee53` |
| vllm_core-0025 | trae | gpt-5 | vllm | `58eee5f2` |

### Speedup: 0.995x (2 runs)

**Root Cause:** 0.995x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0041 | trae | gpt-5 | vllm | `89a84b0b` |
| vllm_core-0022 | trae | gpt-5 | vllm | `4c822298` |

### Speedup: 1.010x (1 runs)

**Root Cause:** 1.010x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_028_6b7038ba | trae | claude-sonnet-45 | sglang | `6b7038ba` |

### Speedup: 0.979x (1 runs)

**Root Cause:** 0.979x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_001_09deb20d | trae | claude-sonnet-45 | sglang | `09deb20d` |

### Speedup: 0.986x (1 runs)

**Root Cause:** 0.986x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_049_a73c4df4 | trae | claude-sonnet-45 | sglang | `a73c4df4` |

### Speedup: 1.020x (1 runs)

**Root Cause:** 1.020x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_062_c98e84c2 | trae | claude-sonnet-45 | sglang | `c98e84c2` |

### Speedup: 1.034x (1 runs)

**Root Cause:** 1.034x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_028_6b7038ba | trae | gpt-5 | sglang | `6b7038ba` |

### Speedup: 1.036x (1 runs)

**Root Cause:** 1.036x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_049_a73c4df4 | trae | gpt-5 | sglang | `a73c4df4` |

### Speedup: 1.028x (1 runs)

**Root Cause:** 1.028x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_072_e5db40dc | trae | gpt-5 | sglang | `e5db40dc` |

### Speedup: 0.993x (1 runs)

**Root Cause:** 0.993x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_048_a37e1247 | trae | gpt-5 | sglang | `a37e1247` |

### Speedup: 1.007x (1 runs)

**Root Cause:** 1.007x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_030_6e2da515 | trae | gpt-5 | sglang | `6e2da515` |

### Speedup: 0.966x (1 runs)

**Root Cause:** 0.966x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_022_5239d795 | trae | gpt-5 | sglang | `5239d795` |

### Speedup: 0.971x (1 runs)

**Root Cause:** 0.971x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_025_62757db6 | trae | gpt-5 | sglang | `62757db6` |

### Speedup: 1.014x (1 runs)

**Root Cause:** 1.014x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0008 | trae | claude-sonnet-45 | vllm | `25ebed2f` |

### Speedup: 0.981x (1 runs)

**Root Cause:** 0.981x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0005 | trae | gpt-4o | vllm | `22d33bac` |

### Speedup: 1.009x (1 runs)

**Root Cause:** 1.009x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0015 | trae | gpt-5 | vllm | `310aca88` |

### Speedup: 1.015x (1 runs)

**Root Cause:** 1.015x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0040 | trae | gpt-5 | vllm | `88693683` |

### Speedup: 0.969x (1 runs)

**Root Cause:** 0.969x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0015 | trae | gpt-5 | vllm | `310aca88` |

### Speedup: 0.952x (1 runs)

**Root Cause:** 0.952x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0005 | trae | gpt-5 | vllm | `22d33bac` |

### Speedup: 1.022x (1 runs)

**Root Cause:** 1.022x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0090 | trae | gpt-5 | vllm | `f26c4aee` |

### Speedup: 0.991x (1 runs)

**Root Cause:** 0.991x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0053 | trae | gpt-5 | vllm | `9badee53` |

### Speedup: 0.980x (1 runs)

**Root Cause:** 0.980x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0007 | trae | gpt-5 | vllm | `25ebed2f` |

### Speedup: 0.954x (1 runs)

**Root Cause:** 0.954x (no significant change)

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0006 | codex | gpt-5 | vllm | `22d33bac` |

---

## SUCCESS_REGRESSION (12 runs, 1.4%)

**Description:** Performance degraded >5%

### Speedup: 0.433x (1 runs)

**Root Cause:** 0.433x REGRESSION

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_057_bb3a3b66 | trae | claude-sonnet-45 | sglang | `bb3a3b66` |

### Speedup: 0.422x (1 runs)

**Root Cause:** 0.422x REGRESSION

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_057_bb3a3b66 | trae | gpt-5 | sglang | `bb3a3b66` |

### Speedup: 0.937x (1 runs)

**Root Cause:** 0.937x REGRESSION

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_031_6f560c76 | trae | gpt-5 | sglang | `6f560c76` |

### Speedup: 0.134x (1 runs)

**Root Cause:** 0.134x REGRESSION

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0039 | trae | claude-sonnet-45 | vllm | `7c01f706` |

### Speedup: 0.513x (1 runs)

**Root Cause:** 0.513x REGRESSION

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0042 | trae | claude-sonnet-45 | vllm | `83450458` |

### Speedup: 0.949x (1 runs)

**Root Cause:** 0.949x REGRESSION

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0006 | trae | claude-sonnet-45 | vllm | `22d33bac` |

### Speedup: 0.877x (1 runs)

**Root Cause:** 0.877x REGRESSION

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0041 | trae | gpt-5 | vllm | `89a84b0b` |

### Speedup: 0.517x (1 runs)

**Root Cause:** 0.517x REGRESSION

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0039 | trae | gpt-5 | vllm | `83450458` |

### Speedup: 0.233x (1 runs)

**Root Cause:** 0.233x REGRESSION

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0037 | trae | gpt-5 | vllm | `7c01f706` |

### Speedup: 0.511x (1 runs)

**Root Cause:** 0.511x REGRESSION

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0039 | trae | gpt-5 | vllm | `83450458` |

### Speedup: 0.237x (1 runs)

**Root Cause:** 0.237x REGRESSION

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0037 | trae | gpt-5 | vllm | `7c01f706` |

### Speedup: 0.242x (1 runs)

**Root Cause:** 0.242x REGRESSION

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0039 | codex | gpt-5 | vllm | `7c01f706` |

---

## AGENT_TIMEOUT (1 runs, 0.1%)

**Description:** Agent timed out (>1 hour)

### Agent timeout after 60 min (1 runs)

**Root Cause:** Agent ran for over 1 hour without producing a patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_012_27168308 | trae | gpt-5 | sglang | `27168308` |

---

## AGENT_NO_PATCH (365 runs, 43.8%)

**Description:** Agent errored without producing patch

### Agent error, 0.0min (153 runs)

**Root Cause:** Agent status=error, ran for 0.0 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| moe_align_opt-0000 | openhands | gpt-5 | vllm | `0ec82edd` |
| moe_align_opt-0000 | openhands | gpt-5 | vllm | `0ec82edd` |
| prefix_caching_opt-0000 | openhands | gpt-5 | vllm | `2deb029d` |
| prefix_caching_opt-0000 | openhands | gpt-5 | vllm | `2deb029d` |
| prefix_caching_opt-0000 | openhands | gpt-5 | vllm | `2deb029d` |
| prefix_caching_opt-0000 | openhands | gpt-5 | vllm | `2deb029d` |
| moe_align_opt-0000 | trae | o4-mini | vllm | `0ec82edd` |
| moe_align_opt-0000 | trae | gpt-5 | vllm | `0ec82edd` |
| moe_align_opt-0000 | trae | gpt-5 | vllm | `0ec82edd` |
| moe_align_opt-0000 | trae | gpt-5 | vllm | `0ec82edd` |
| ... | | | | |
| *(143 more runs)* | | | | |

### Agent error, 2.9min (18 runs)

**Root Cause:** Agent status=error, ran for 2.9 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_073_e822e590 | trae | claude-sonnet-45 | sglang | `e822e590` |
| sglang_079_ff00895c | trae | claude-sonnet-45 | sglang | `ff00895c` |
| sglang_045_9c088829 | trae | gpt-5 | sglang | `9c088829` |
| sglang_044_9c064bf7 | trae | gpt-5 | sglang | `9c064bf7` |
| vllm_bedrock_sonnet45-0092 | trae | claude-sonnet-45 | vllm | `eefbf4a6` |
| vllm_bedrock_sonnet45-0080 | trae | claude-sonnet-45 | vllm | `d4bc1a4d` |
| vllm_bedrock_sonnet45-0085 | trae | claude-sonnet-45 | vllm | `e206b543` |
| vllm_bedrock_sonnet45-0054 | trae | claude-sonnet-45 | vllm | `99abb8b6` |
| vllm_bedrock_sonnet45-0059 | trae | claude-sonnet-45 | vllm | `9f1710f1` |
| vllm_bedrock_sonnet45-0086 | trae | claude-sonnet-45 | vllm | `e3580537` |
| ... | | | | |
| *(8 more runs)* | | | | |

### Agent error, 3.2min (15 runs)

**Root Cause:** Agent status=error, ran for 3.2 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_068_dd1012fc | trae | gpt-5 | sglang | `dd1012fc` |
| sglang_040_915140fd | trae | gpt-5 | sglang | `915140fd` |
| sglang_057_bb3a3b66 | trae | gpt-5 | sglang | `bb3a3b66` |
| sglang_049_a73c4df4 | trae | gpt-5 | sglang | `a73c4df4` |
| sglang_077_f0815419 | trae | gpt-5 | sglang | `f0815419` |
| vllm_bedrock_sonnet45-0091 | trae | claude-sonnet-45 | vllm | `ed250545` |
| vllm_bedrock_sonnet45-0094 | trae | claude-sonnet-45 | vllm | `f26c4aee` |
| vllm_bedrock_sonnet45-0058 | trae | claude-sonnet-45 | vllm | `9ed82e70` |
| vllm_bedrock_sonnet45-0045 | trae | claude-sonnet-45 | vllm | `8a4e5c5f` |
| vllm_bedrock_sonnet45-0066 | trae | claude-sonnet-45 | vllm | `b55ed6ef` |
| ... | | | | |
| *(5 more runs)* | | | | |

### Agent error, 2.7min (13 runs)

**Root Cause:** Agent status=error, ran for 2.7 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_077_f0815419 | trae | claude-sonnet-45 | sglang | `f0815419` |
| sglang_075_f0653886 | trae | gpt-5 | sglang | `f0653886` |
| sglang_041_9183c23e | trae | gpt-5 | sglang | `9183c23e` |
| sglang_046_9c745d07 | trae | gpt-5 | sglang | `9c745d07` |
| sglang_050_a99801e0 | trae | gpt-5 | sglang | `a99801e0` |
| vllm_bedrock_sonnet45-0046 | trae | claude-sonnet-45 | vllm | `8aa1485f` |
| vllm_bedrock_sonnet45-0097 | trae | claude-sonnet-45 | vllm | `fc542144` |
| vllm_bedrock_sonnet45-0080 | trae | claude-sonnet-45 | vllm | `d4bc1a4d` |
| vllm_bedrock_sonnet45-0029 | trae | claude-sonnet-45 | vllm | `660470e5` |
| vllm_bedrock_sonnet45-0027 | trae | claude-sonnet-45 | vllm | `5e5c8e09` |
| ... | | | | |
| *(3 more runs)* | | | | |

### Agent error, 2.6min (13 runs)

**Root Cause:** Agent status=error, ran for 2.6 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0082 | trae | claude-sonnet-45 | vllm | `d7740ea4` |
| vllm_bedrock_sonnet45-0089 | trae | claude-sonnet-45 | vllm | `e7b20426` |
| vllm_bedrock_sonnet45-0083 | trae | claude-sonnet-45 | vllm | `dae68969` |
| vllm_bedrock_sonnet45-0064 | trae | claude-sonnet-45 | vllm | `b10e5198` |
| vllm_bedrock_sonnet45-0031 | trae | claude-sonnet-45 | vllm | `6a417b86` |
| vllm_bedrock_sonnet45-0077 | trae | claude-sonnet-45 | vllm | `ccf02fcb` |
| vllm_bedrock_sonnet45-0098 | trae | claude-sonnet-45 | vllm | `fc7b8d1e` |
| vllm_bedrock_sonnet45-0091 | trae | claude-sonnet-45 | vllm | `ed250545` |
| vllm_bedrock_sonnet45-0053 | trae | claude-sonnet-45 | vllm | `98f47f2a` |
| vllm_bedrock_sonnet45-0055 | trae | claude-sonnet-45 | vllm | `9a3b8832` |
| ... | | | | |
| *(3 more runs)* | | | | |

### Agent error, 2.5min (11 runs)

**Root Cause:** Agent status=error, ran for 2.5 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_074_e88dd482 | trae | gpt-5 | sglang | `e88dd482` |
| vllm_bedrock_sonnet45-0095 | trae | claude-sonnet-45 | vllm | `fa63e710` |
| vllm_bedrock_sonnet45-0093 | trae | claude-sonnet-45 | vllm | `f092153f` |
| vllm_bedrock_sonnet45-0096 | trae | claude-sonnet-45 | vllm | `fb0acb6c` |
| vllm_bedrock_sonnet45-0087 | trae | claude-sonnet-45 | vllm | `e493e485` |
| vllm_bedrock_sonnet45-0063 | trae | claude-sonnet-45 | vllm | `aea94362` |
| vllm_bedrock_sonnet45-0074 | trae | claude-sonnet-45 | vllm | `c0569dbc` |
| vllm_bedrock_sonnet45-0072 | trae | claude-sonnet-45 | vllm | `bd6028d6` |
| vllm_core-0047 | trae | gpt-5 | vllm | `9323a315` |
| vllm_core-0047 | trae | gpt-5 | vllm | `9323a315` |
| ... | | | | |
| *(1 more runs)* | | | | |

### Agent error, 2.4min (10 runs)

**Root Cause:** Agent status=error, ran for 2.4 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_073_e822e590 | trae | gpt-5 | sglang | `e822e590` |
| sglang_066_dc188132 | trae | gpt-5 | sglang | `dc188132` |
| sglang_054_b1709305 | trae | gpt-5 | sglang | `b1709305` |
| vllm_bedrock_sonnet45-0081 | trae | claude-sonnet-45 | vllm | `d55e446d` |
| vllm_bedrock_sonnet45-0084 | trae | claude-sonnet-45 | vllm | `dcc6cfb9` |
| vllm_bedrock_sonnet45-0070 | trae | claude-sonnet-45 | vllm | `baeded25` |
| vllm_bedrock_sonnet45-0032 | trae | claude-sonnet-45 | vllm | `6ce01f30` |
| vllm_bedrock_sonnet45-0052 | trae | claude-sonnet-45 | vllm | `9474e89b` |
| vllm_bedrock_sonnet45-0088 | trae | claude-sonnet-45 | vllm | `e7523c2e` |
| vllm_bedrock_sonnet45-0062 | trae | claude-sonnet-45 | vllm | `ad8d696a` |

### Agent error, 2.8min (10 runs)

**Root Cause:** Agent status=error, ran for 2.8 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_061_c2f212d6 | trae | gpt-5 | sglang | `c2f212d6` |
| sglang_058_bc3f6db2 | trae | gpt-5 | sglang | `bc3f6db2` |
| sglang_035_7ce36068 | trae | gpt-5 | sglang | `7ce36068` |
| sglang_048_a37e1247 | trae | gpt-5 | sglang | `a37e1247` |
| vllm_bedrock_sonnet45-0075 | trae | claude-sonnet-45 | vllm | `c45f3c3a` |
| vllm_bedrock_sonnet45-0092 | trae | claude-sonnet-45 | vllm | `eefbf4a6` |
| vllm_bedrock_sonnet45-0041 | trae | claude-sonnet-45 | vllm | `81ede99c` |
| vllm_core-0046 | trae | gpt-5 | vllm | `8d75fe48` |
| vllm_core-0048 | trae | gpt-5 | vllm | `93e5f3c5` |
| vllm_core-0093 | trae | gpt-5 | vllm | `fc542144` |

### Agent error, 3.4min (10 runs)

**Root Cause:** Agent status=error, ran for 3.4 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_065_da47621c | trae | gpt-5 | sglang | `da47621c` |
| vllm_bedrock_sonnet45-0040 | trae | claude-sonnet-45 | vllm | `80aa7e91` |
| vllm_bedrock_sonnet45-0096 | trae | claude-sonnet-45 | vllm | `fb0acb6c` |
| vllm_bedrock_sonnet45-0079 | trae | claude-sonnet-45 | vllm | `cf2f084d` |
| vllm_bedrock_sonnet45-0003 | trae | claude-sonnet-45 | vllm | `0ec82edd` |
| vllm_bedrock_sonnet45-0068 | trae | claude-sonnet-45 | vllm | `b6d10354` |
| vllm_bedrock_sonnet45-0034 | trae | claude-sonnet-45 | vllm | `6d646d08` |
| vllm_bedrock_sonnet45-0047 | trae | claude-sonnet-45 | vllm | `8bc68e19` |
| vllm_bedrock_sonnet45-0050 | trae | claude-sonnet-45 | vllm | `9323a315` |
| vllm_bedrock_sonnet45-0038 | trae | claude-sonnet-45 | vllm | `7661e92e` |

### Agent error, 3.1min (9 runs)

**Root Cause:** Agent status=error, ran for 3.1 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_043_93470a14 | trae | gpt-5 | sglang | `93470a14` |
| sglang_060_c2bd094d | trae | gpt-5 | sglang | `c2bd094d` |
| sglang_055_b1e5a33a | trae | gpt-5 | sglang | `b1e5a33a` |
| sglang_071_e3ec6bf4 | trae | gpt-5 | sglang | `e3ec6bf4` |
| vllm_bedrock_sonnet45-0086 | trae | claude-sonnet-45 | vllm | `e3580537` |
| vllm_bedrock_sonnet45-0032 | trae | claude-sonnet-45 | vllm | `6ce01f30` |
| vllm_bedrock_sonnet45-0030 | trae | claude-sonnet-45 | vllm | `67da5720` |
| vllm_bedrock_sonnet45-0078 | trae | claude-sonnet-45 | vllm | `ce6bf3a2` |
| vllm_bedrock_sonnet45-0043 | trae | claude-sonnet-45 | vllm | `88693683` |

### Agent error, 3.5min (9 runs)

**Root Cause:** Agent status=error, ran for 3.5 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_038_8f8f96a6 | trae | gpt-5 | sglang | `8f8f96a6` |
| sglang_059_c087ddd6 | trae | gpt-5 | sglang | `c087ddd6` |
| sglang_047_a191a0e4 | trae | gpt-5 | sglang | `a191a0e4` |
| sglang_053_adca585b | trae | gpt-5 | sglang | `adca585b` |
| vllm_bedrock_sonnet45-0098 | trae | claude-sonnet-45 | vllm | `fc7b8d1e` |
| vllm_bedrock_sonnet45-0076 | trae | claude-sonnet-45 | vllm | `ca7a2d5f` |
| vllm_bedrock_sonnet45-0028 | trae | claude-sonnet-45 | vllm | `61b8cea3` |
| vllm_bedrock_sonnet45-0025 | trae | claude-sonnet-45 | vllm | `526de822` |
| vllm_core-0006 | trae | gpt-5 | vllm | `22dd9c27` |

### Agent error, 3.0min (8 runs)

**Root Cause:** Agent status=error, ran for 3.0 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_079_ff00895c | trae | gpt-5 | sglang | `ff00895c` |
| sglang_062_c98e84c2 | trae | gpt-5 | sglang | `c98e84c2` |
| vllm_bedrock_sonnet45-0087 | trae | claude-sonnet-45 | vllm | `e493e485` |
| vllm_bedrock_sonnet45-0099 | trae | claude-sonnet-45 | vllm | `fe66b347` |
| vllm_bedrock_sonnet45-0057 | trae | claude-sonnet-45 | vllm | `9d72daf4` |
| vllm_bedrock_sonnet45-0061 | trae | claude-sonnet-45 | vllm | `ac45c44d` |
| vllm_bedrock_sonnet45-0075 | trae | claude-sonnet-45 | vllm | `c45f3c3a` |
| vllm_bedrock_sonnet45-0094 | trae | claude-sonnet-45 | vllm | `f26c4aee` |

### Agent error, 3.7min (8 runs)

**Root Cause:** Agent status=error, ran for 3.7 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_064_d1112d85 | trae | gpt-5 | sglang | `d1112d85` |
| sglang_079_ff00895c | trae | gpt-5 | sglang | `ff00895c` |
| sglang_036_86a876d8 | trae | gpt-5 | sglang | `86a876d8` |
| vllm_bedrock_sonnet45-0069 | trae | claude-sonnet-45 | vllm | `b9986454` |
| vllm_bedrock_sonnet45-0033 | trae | claude-sonnet-45 | vllm | `6d0734c5` |
| vllm_bedrock_sonnet45-0004 | trae | claude-sonnet-45 | vllm | `19d98e0c` |
| vllm_core-0001 | trae | gpt-5 | vllm | `0d243f2a` |
| vllm_core-0002 | trae | gpt-5 | vllm | `0ec82edd` |

### Agent error, 3.3min (8 runs)

**Root Cause:** Agent status=error, ran for 3.3 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_076_f06e90c2 | trae | gpt-5 | sglang | `f06e90c2` |
| sglang_078_fbcbb263 | trae | gpt-5 | sglang | `fbcbb263` |
| sglang_034_79961afa | trae | gpt-5 | sglang | `79961afa` |
| sglang_051_ab4a83b2 | trae | gpt-5 | sglang | `ab4a83b2` |
| sglang_072_e5db40dc | trae | gpt-5 | sglang | `e5db40dc` |
| sglang_042_9216b106 | trae | gpt-5 | sglang | `9216b106` |
| vllm_bedrock_sonnet45-0093 | trae | claude-sonnet-45 | vllm | `f092153f` |
| vllm_core-0001 | trae | gpt-5 | vllm | `0d243f2a` |

### Agent error, 4.0min (6 runs)

**Root Cause:** Agent status=error, ran for 4.0 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_052_ac971ff6 | trae | gpt-5 | sglang | `ac971ff6` |
| sglang_070_df7f61ee | trae | gpt-5 | sglang | `df7f61ee` |
| vllm_bedrock_sonnet45-0076 | trae | claude-sonnet-45 | vllm | `ca7a2d5f` |
| vllm_core-0005 | trae | gpt-5 | vllm | `22d33bac` |
| vllm_core-0004 | trae | gpt-5 | vllm | `21d93c14` |
| vllm_core-0004 | trae | gpt-5 | vllm | `21d93c14` |

### Agent error, 2.3min (6 runs)

**Root Cause:** Agent status=error, ran for 2.3 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0037 | trae | claude-sonnet-45 | vllm | `70b808fe` |
| vllm_bedrock_sonnet45-0073 | trae | claude-sonnet-45 | vllm | `bfdb1ba5` |
| vllm_bedrock_sonnet45-0051 | trae | claude-sonnet-45 | vllm | `93e5f3c5` |
| vllm_bedrock_sonnet45-0044 | trae | claude-sonnet-45 | vllm | `89a84b0b` |
| vllm_bedrock_sonnet45-0084 | trae | claude-sonnet-45 | vllm | `dcc6cfb9` |
| vllm_bedrock_sonnet45-0099 | trae | claude-sonnet-45 | vllm | `fe66b347` |

### Agent error, 3.8min (5 runs)

**Root Cause:** Agent status=error, ran for 3.8 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_039_912788c0 | trae | gpt-5 | sglang | `912788c0` |
| sglang_069_ddcf9fe3 | trae | gpt-5 | sglang | `ddcf9fe3` |
| sglang_067_dc67d976 | trae | gpt-5 | sglang | `dc67d976` |
| vllm_bedrock_sonnet45-0003 | trae | claude-sonnet-45 | vllm | `0ec82edd` |
| vllm_core-0005 | trae | gpt-5 | vllm | `22d33bac` |

### Agent error, 4.2min (4 runs)

**Root Cause:** Agent status=error, ran for 4.2 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_035_7ce36068 | trae | gpt-5 | sglang | `7ce36068` |
| vllm_core-0026 | trae | gpt-5 | vllm | `61b8cea3` |
| vllm_core-0008 | trae | gpt-5 | vllm | `296f927f` |
| vllm_core-0007 | trae | gpt-5 | vllm | `25ebed2f` |

### Agent error, 4.3min (4 runs)

**Root Cause:** Agent status=error, ran for 4.3 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_037_880221bd | trae | gpt-5 | sglang | `880221bd` |
| vllm_bedrock_sonnet45-0067 | trae | claude-sonnet-45 | vllm | `b690e348` |
| vllm_core-0003 | trae | gpt-5 | vllm | `19d98e0c` |
| vllm_core-0006 | trae | gpt-5 | vllm | `22dd9c27` |

### Agent error, 3.9min (4 runs)

**Root Cause:** Agent status=error, ran for 3.9 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0046 | trae | claude-sonnet-45 | vllm | `8aa1485f` |
| vllm_bedrock_sonnet45-0058 | trae | claude-sonnet-45 | vllm | `9ed82e70` |
| vllm_bedrock_sonnet45-0040 | trae | claude-sonnet-45 | vllm | `80aa7e91` |
| vllm_core-0002 | trae | gpt-5 | vllm | `0ec82edd` |

### Agent error, 1.8min (3 runs)

**Root Cause:** Agent status=error, ran for 1.8 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_073_e822e590 | trae | gpt-5 | sglang | `e822e590` |
| vllm_bedrock_sonnet45-0048 | trae | claude-sonnet-45 | vllm | `8c1e77fb` |
| vllm_core-0048 | trae | gpt-5 | vllm | `93e5f3c5` |

### Agent error, 3.6min (3 runs)

**Root Cause:** Agent status=error, ran for 3.6 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_056_b77a02cd | trae | gpt-5 | sglang | `b77a02cd` |
| vllm_bedrock_sonnet45-0090 | trae | claude-sonnet-45 | vllm | `ec3b5ce9` |
| vllm_core-0008 | trae | gpt-5 | vllm | `296f927f` |

### Agent error, 2.0min (3 runs)

**Root Cause:** Agent status=error, ran for 2.0 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0069 | trae | claude-sonnet-45 | vllm | `b9986454` |
| vllm_bedrock_sonnet45-0060 | trae | claude-sonnet-45 | vllm | `a3223766` |
| vllm_bedrock_sonnet45-0049 | trae | claude-sonnet-45 | vllm | `8d75fe48` |

### Agent error, 4.1min (2 runs)

**Root Cause:** Agent status=error, ran for 4.1 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_063_cd7e32e2 | trae | gpt-5 | sglang | `cd7e32e2` |
| vllm_bedrock_sonnet45-0057 | trae | claude-sonnet-45 | vllm | `9d72daf4` |

### Agent error, 2.2min (2 runs)

**Root Cause:** Agent status=error, ran for 2.2 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0088 | trae | claude-sonnet-45 | vllm | `e7523c2e` |
| vllm_bedrock_sonnet45-0095 | trae | claude-sonnet-45 | vllm | `fa63e710` |

### Agent error, 6.4min (2 runs)

**Root Cause:** Agent status=error, ran for 6.4 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0008 | trae | gpt-4o | vllm | `296f927f` |
| vllm_core-0092 | trae | gpt-5 | vllm | `fb0acb6c` |

### Agent error, 7.2min (2 runs)

**Root Cause:** Agent status=error, ran for 7.2 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0001 | trae | gpt-4o | vllm | `0d243f2a` |
| vllm_core-0018 | trae | gpt-5 | vllm | `35fad35a` |

### Agent error, 5.3min (1 runs)

**Root Cause:** Agent status=error, ran for 5.3 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_077_f0815419 | trae | gpt-5 | sglang | `f0815419` |

### Agent error, 4.8min (1 runs)

**Root Cause:** Agent status=error, ran for 4.8 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_045_9c088829 | trae | gpt-5 | sglang | `9c088829` |

### Agent error, 17.9min (1 runs)

**Root Cause:** Agent status=error, ran for 17.9 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_028_6b7038ba | trae | gpt-5 | sglang | `6b7038ba` |

### Agent error, 35.8min (1 runs)

**Root Cause:** Agent status=error, ran for 35.8 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_008_205d5cb4 | trae | gpt-5 | sglang | `205d5cb4` |

### Agent error, 21.0min (1 runs)

**Root Cause:** Agent status=error, ran for 21.0 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_033_73b13e69 | trae | gpt-5 | sglang | `73b13e69` |

### Agent error, 12.0min (1 runs)

**Root Cause:** Agent status=error, ran for 12.0 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_019_31589e17 | trae | gpt-5 | sglang | `31589e17` |

### Agent error, 1.9min (1 runs)

**Root Cause:** Agent status=error, ran for 1.9 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0097 | trae | claude-sonnet-45 | vllm | `fc542144` |

### Agent error, 1.5min (1 runs)

**Root Cause:** Agent status=error, ran for 1.5 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0071 | trae | claude-sonnet-45 | vllm | `bc7c4d20` |

### Agent error, 8.6min (1 runs)

**Root Cause:** Agent status=error, ran for 8.6 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0006 | trae | gpt-4o | vllm | `22dd9c27` |

### Agent error, 30.8min (1 runs)

**Root Cause:** Agent status=error, ran for 30.8 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0007 | trae | gpt-4o | vllm | `25ebed2f` |

### Agent error, 45.0min (1 runs)

**Root Cause:** Agent status=error, ran for 45.0 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0000 | trae | gpt-4o | vllm | `8aa1485f` |

### Agent error, 9.0min (1 runs)

**Root Cause:** Agent status=error, ran for 9.0 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0000 | trae | gpt-4o | vllm | `8aa1485f` |

### Agent error, 0.7min (1 runs)

**Root Cause:** Agent status=error, ran for 0.7 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0005 | trae | gpt-4o | vllm | `22d33bac` |

### Agent error, 1.6min (1 runs)

**Root Cause:** Agent status=error, ran for 1.6 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0006 | trae | gpt-4o | vllm | `22dd9c27` |

### Agent error, 5.2min (1 runs)

**Root Cause:** Agent status=error, ran for 5.2 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0001 | trae | gpt-4o | vllm | `0d243f2a` |

### Agent error, 1.1min (1 runs)

**Root Cause:** Agent status=error, ran for 1.1 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0004 | trae | gpt-4o | vllm | `21d93c14` |

### Agent error, 0.1min (1 runs)

**Root Cause:** Agent status=error, ran for 0.1 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0067 | trae | gpt-4o | vllm | `bc7c4d20` |

### Agent error, 4.7min (1 runs)

**Root Cause:** Agent status=error, ran for 4.7 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0003 | trae | gpt-5 | vllm | `19d98e0c` |

### Agent error, 12.3min (1 runs)

**Root Cause:** Agent status=error, ran for 12.3 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0044 | trae | gpt-5 | vllm | `8bc68e19` |

### Agent success, 41.8min (1 runs)

**Root Cause:** Agent status=success, ran for 41.8 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0023 | trae | gpt-5 | vllm | `4fb56914` |

### Agent error, 4.9min (1 runs)

**Root Cause:** Agent status=error, ran for 4.9 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0007 | trae | gpt-5 | vllm | `25ebed2f` |

### Agent error, 10.5min (1 runs)

**Root Cause:** Agent status=error, ran for 10.5 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0023 | trae | gpt-5 | vllm | `4fb56914` |

### Agent error, 6.2min (1 runs)

**Root Cause:** Agent status=error, ran for 6.2 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0083 | trae | gpt-5 | vllm | `e493e485` |

### Agent error, 6.5min (1 runs)

**Root Cause:** Agent status=error, ran for 6.5 minutes but did not produce patch

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0057 | trae | gpt-5 | vllm | `a3223766` |

---

## BASELINE_OOM (2 runs, 0.2%)

**Description:** GPU out of memory

### GPU OOM (2 runs)

**Root Cause:** GPU ran out of memory during baseline test execution

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_023_564a898a | trae | claude-sonnet-45 | sglang | `564a898a` |
| sglang_023_564a898a | trae | gpt-5 | sglang | `564a898a` |

---

## BASELINE_CUDA_ERROR (1 runs, 0.1%)

**Description:** CUDA/GPU error during test

### CUDA: an illegal memory access was encountered (1 runs)

**Root Cause:** CUDA error during test: an illegal memory access was encountered

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0072 | trae | gpt-5 | vllm | `ca7a2d5f` |

---

## BASELINE_IMPORT_ERROR (29 runs, 3.5%)

**Description:** Missing Python module

### Missing: vllm.compilation (6 runs)

**Root Cause:** Test requires module 'vllm.compilation' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0073 | trae | claude-sonnet-45 | vllm | `bfdb1ba5` |
| vllm_bedrock_sonnet45-0062 | trae | claude-sonnet-45 | vllm | `ad8d696a` |
| vllm_core-0069 | trae | gpt-5 | vllm | `bfdb1ba5` |
| vllm_core-0059 | trae | gpt-5 | vllm | `ad8d696a` |
| vllm_core-0062 | codex | gpt-5 | vllm | `ad8d696a` |
| vllm_core-0073 | codex | gpt-5 | vllm | `bfdb1ba5` |

### Missing: vllm.attention.backends.dual_chunk_flash_attn (6 runs)

**Root Cause:** Test requires module 'vllm.attention.backends.dual_chunk_flash_attn' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0053 | trae | claude-sonnet-45 | vllm | `98f47f2a` |
| vllm_bedrock_sonnet45-0022 | trae | claude-sonnet-45 | vllm | `3b61cb45` |
| vllm_core-0021 | trae | gpt-5 | vllm | `3b61cb45` |
| vllm_core-0021 | trae | gpt-5 | vllm | `3b61cb45` |
| vllm_core-0050 | trae | gpt-5 | vllm | `98f47f2a` |
| vllm_core-0053 | codex | gpt-5 | vllm | `98f47f2a` |

### Missing: vllm.commit_id (5 runs)

**Root Cause:** Test requires module 'vllm.commit_id' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0030 | trae | gpt-5 | vllm | `6ce01f30` |
| vllm_core-0030 | trae | gpt-5 | vllm | `6ce01f30` |
| vllm_core-0094 | trae | gpt-5 | vllm | `fc7b8d1e` |
| vllm_core-0082 | trae | gpt-5 | vllm | `e3580537` |
| vllm_core-0032 | codex | gpt-5 | vllm | `6ce01f30` |

### Missing: vllm._version (4 runs)

**Root Cause:** Test requires module 'vllm._version' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0004 | trae | claude-sonnet-45 | vllm | `19d98e0c` |
| vllm_core-0003 | trae | gpt-4o | vllm | `19d98e0c` |
| vllm_core-0003 | trae | gpt-4o | vllm | `19d98e0c` |
| vllm_core-0003 | trae | gpt-5 | vllm | `19d98e0c` |

### Cannot import: default_cache_dir (2 runs)

**Root Cause:** Cannot import 'default_cache_dir' - API may have changed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_029_6cb00c63 | trae | claude-sonnet-45 | sglang | `6cb00c63` |
| sglang_029_6cb00c63 | trae | gpt-5 | sglang | `6cb00c63` |

### Missing: vllm (2 runs)

**Root Cause:** Test requires module 'vllm' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_066_dc188132 | trae | claude-sonnet-45 | sglang | `dc188132` |
| sglang_066_dc188132 | trae | gpt-5 | sglang | `dc188132` |

### Missing: outlines (2 runs)

**Root Cause:** Test requires module 'outlines' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_056_b77a02cd | trae | claude-sonnet-45 | sglang | `b77a02cd` |
| sglang_056_b77a02cd | trae | gpt-5 | sglang | `b77a02cd` |

### Missing: vllm.v1.spec_decode.metadata (2 runs)

**Root Cause:** Test requires module 'vllm.v1.spec_decode.metadata' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0054 | trae | claude-sonnet-45 | vllm | `99abb8b6` |
| vllm_core-0051 | trae | gpt-5 | vllm | `99abb8b6` |

---

## BASELINE_ATTRIBUTE_ERROR (8 runs, 1.0%)

**Description:** AttributeError in test

### 'numpy.ndarray' object has no attribute 'typecode' (4 runs)

**Root Cause:** AttributeError: 'numpy.ndarray' object has no attribute 'typecode'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0034 | trae | claude-sonnet-45 | vllm | `6d646d08` |
| vllm_core-0032 | trae | gpt-5 | vllm | `6d646d08` |
| vllm_core-0032 | trae | gpt-5 | vllm | `6d646d08` |
| vllm_core-0034 | codex | gpt-5 | vllm | `6d646d08` |

### 'CpuGpuBlockAllocator' object has no attribute 'allocate_mutable_block'. Did you mean: 'allocate_mut (4 runs)

**Root Cause:** AttributeError: 'CpuGpuBlockAllocator' object has no attribute 'allocate_mutable_block'. Did you mean: 'allocate_mut

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0018 | trae | claude-sonnet-45 | vllm | `3476ed08` |
| vllm_core-0017 | trae | gpt-5 | vllm | `3476ed08` |
| vllm_core-0017 | trae | gpt-5 | vllm | `3476ed08` |
| vllm_core-0018 | codex | gpt-5 | vllm | `3476ed08` |

---

## BASELINE_TYPE_ERROR (55 runs, 6.6%)

**Description:** TypeError in test (API mismatch)

### MambaMixer2.__init__() got an unexpected keyword argument 'd_model' (5 runs)

**Root Cause:** TypeError: MambaMixer2.__init__() got an unexpected keyword argument 'd_model'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0077 | trae | claude-sonnet-45 | vllm | `ccf02fcb` |
| vllm_bedrock_sonnet45-0009 | trae | claude-sonnet-45 | vllm | `296f927f` |
| vllm_bedrock_sonnet45-0009 | trae | claude-sonnet-45 | vllm | `296f927f` |
| vllm_core-0008 | trae | gpt-5 | vllm | `296f927f` |
| vllm_core-0073 | trae | gpt-5 | vllm | `ccf02fcb` |

### triton_scaled_mm() got an unexpected keyword argument 'use_heuristic' (4 runs)

**Root Cause:** TypeError: triton_scaled_mm() got an unexpected keyword argument 'use_heuristic'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0025 | trae | claude-sonnet-45 | vllm | `526de822` |
| vllm_core-0024 | trae | gpt-5 | vllm | `526de822` |
| vllm_core-0024 | trae | gpt-5 | vllm | `526de822` |
| vllm_core-0025 | codex | gpt-5 | vllm | `526de822` |

### ModelConfig.__init__() missing 1 required positional argument: 'task' (4 runs)

**Root Cause:** TypeError: ModelConfig.__init__() missing 1 required positional argument: 'task'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0035 | trae | claude-sonnet-45 | vllm | `6dd94dbe` |
| vllm_core-0033 | trae | gpt-5 | vllm | `6dd94dbe` |
| vllm_core-0033 | trae | gpt-5 | vllm | `6dd94dbe` |
| vllm_core-0089 | trae | gpt-5 | vllm | `f092153f` |

### SchedulerConfig.__init__() got an unexpected keyword argument 'chunked_prefill_enabled' (4 runs)

**Root Cause:** TypeError: SchedulerConfig.__init__() got an unexpected keyword argument 'chunked_prefill_enabled'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0036 | trae | claude-sonnet-45 | vllm | `6e36f4fa` |
| vllm_core-0034 | trae | gpt-5 | vllm | `6e36f4fa` |
| vllm_core-0034 | trae | gpt-5 | vllm | `6e36f4fa` |
| vllm_core-0036 | codex | gpt-5 | vllm | `6e36f4fa` |

### Unexpected keyword argument 'logprobs' (3 runs)

**Root Cause:** TypeError: Unexpected keyword argument 'logprobs'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0063 | trae | claude-sonnet-45 | vllm | `aea94362` |
| vllm_core-0060 | trae | gpt-5 | vllm | `aea94362` |
| vllm_core-0063 | codex | gpt-5 | vllm | `aea94362` |

### cutlass_scaled_mm_dq() got an unexpected keyword argument 'scale_a' (3 runs)

**Root Cause:** TypeError: cutlass_scaled_mm_dq() got an unexpected keyword argument 'scale_a'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0049 | trae | claude-sonnet-45 | vllm | `8d75fe48` |
| vllm_core-0046 | trae | gpt-5 | vllm | `8d75fe48` |
| vllm_core-0049 | codex | gpt-5 | vllm | `8d75fe48` |

### apply_penalties() takes 6 positional arguments but 8 were given (3 runs)

**Root Cause:** TypeError: apply_penalties() takes 6 positional arguments but 8 were given

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0010 | trae | claude-sonnet-45 | vllm | `299ebb62` |
| vllm_core-0009 | trae | gpt-5 | vllm | `299ebb62` |
| vllm_core-0009 | trae | gpt-5 | vllm | `299ebb62` |

### PagedAttention.forward_decode() got an unexpected keyword argument 'context_lens' (3 runs)

**Root Cause:** TypeError: PagedAttention.forward_decode() got an unexpected keyword argument 'context_lens'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0038 | trae | gpt-5 | vllm | `80aa7e91` |
| vllm_core-0038 | trae | gpt-5 | vllm | `80aa7e91` |
| vllm_core-0040 | codex | gpt-5 | vllm | `80aa7e91` |

### ServerArgs.__init__() missing 1 required positional argument: 'model_path' (2 runs)

**Root Cause:** TypeError: ServerArgs.__init__() missing 1 required positional argument: 'model_path'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_052_ac971ff6 | trae | claude-sonnet-45 | sglang | `ac971ff6` |
| sglang_052_ac971ff6 | trae | gpt-5 | sglang | `ac971ff6` |

### RadixCache.__init__() missing 2 required positional arguments: 'req_to_token_pool' and 'token_to_kv_ (2 runs)

**Root Cause:** TypeError: RadixCache.__init__() missing 2 required positional arguments: 'req_to_token_pool' and 'token_to_kv_

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_054_b1709305 | trae | claude-sonnet-45 | sglang | `b1709305` |
| sglang_054_b1709305 | trae | gpt-5 | sglang | `b1709305` |

### SamplingBatchInfo.__init__() missing 2 required positional arguments: 'is_all_greedy' and 'need_min_ (2 runs)

**Root Cause:** TypeError: SamplingBatchInfo.__init__() missing 2 required positional arguments: 'is_all_greedy' and 'need_min_

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_046_9c745d07 | trae | claude-sonnet-45 | sglang | `9c745d07` |
| sglang_046_9c745d07 | trae | gpt-5 | sglang | `9c745d07` |

### BlockPool.__init__() got an unexpected keyword argument 'max_free_blocks' (2 runs)

**Root Cause:** TypeError: BlockPool.__init__() got an unexpected keyword argument 'max_free_blocks'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0064 | trae | claude-sonnet-45 | vllm | `b10e5198` |
| vllm_core-0061 | trae | gpt-5 | vllm | `b10e5198` |

### Unexpected keyword argument 'all_stop_token_ids' (2 runs)

**Root Cause:** TypeError: Unexpected keyword argument 'all_stop_token_ids'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0060 | trae | claude-sonnet-45 | vllm | `a3223766` |
| vllm_core-0060 | codex | gpt-5 | vllm | `a3223766` |

### 'int' object is not subscriptable (2 runs)

**Root Cause:** TypeError: 'int' object is not subscriptable

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0000 | trae | gpt-5 | vllm | `0ec82edd` |
| vllm_core-0000 | trae | gpt-5 | vllm | `0ec82edd` |

### SequenceGroupToSample.__init__() got an unexpected keyword argument 'do_sample' (2 runs)

**Root Cause:** TypeError: SequenceGroupToSample.__init__() got an unexpected keyword argument 'do_sample'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0078 | trae | gpt-5 | vllm | `d7740ea4` |
| vllm_core-0082 | codex | gpt-5 | vllm | `d7740ea4` |

### setup.<locals>.<lambda>() got an unexpected keyword argument 'prev_block' (2 runs)

**Root Cause:** TypeError: setup.<locals>.<lambda>() got an unexpected keyword argument 'prev_block'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0055 | trae | gpt-5 | vllm | `9ed82e70` |
| vllm_core-0058 | codex | gpt-5 | vllm | `9ed82e70` |

### KVCacheBlock.__init__() got an unexpected keyword argument 'prev_token_id' (2 runs)

**Root Cause:** TypeError: KVCacheBlock.__init__() got an unexpected keyword argument 'prev_token_id'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0087 | trae | gpt-5 | vllm | `ed250545` |
| vllm_core-0091 | codex | gpt-5 | vllm | `ed250545` |

### InputMetadata.__init__() got an unexpected keyword argument 'num_prompts' (2 runs)

**Root Cause:** TypeError: InputMetadata.__init__() got an unexpected keyword argument 'num_prompts'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0076 | trae | gpt-5 | vllm | `d4bc1a4d` |
| vllm_core-0080 | codex | gpt-5 | vllm | `d4bc1a4d` |

### Unexpected keyword argument 'logprob_token_ids' (1 runs)

**Root Cause:** TypeError: Unexpected keyword argument 'logprob_token_ids'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0091 | trae | gpt-5 | vllm | `fa63e710` |

### selective_state_update() got an unexpected keyword argument 'out' (1 runs)

**Root Cause:** TypeError: selective_state_update() got an unexpected keyword argument 'out'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0064 | trae | gpt-5 | vllm | `b690e348` |

### silu_mul_fp8_quant_deep_gemm() takes from 2 to 4 positional arguments but 5 were given (1 runs)

**Root Cause:** TypeError: silu_mul_fp8_quant_deep_gemm() takes from 2 to 4 positional arguments but 5 were given

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0080 | trae | gpt-5 | vllm | `dcc6cfb9` |

### cutlass_moe_fp8() got an unexpected keyword argument 'per_out_ch' (1 runs)

**Root Cause:** TypeError: cutlass_moe_fp8() got an unexpected keyword argument 'per_out_ch'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0085 | trae | gpt-5 | vllm | `e7b20426` |

### TritonExperts.__init__() got an unexpected keyword argument 'per_channel_quant' (1 runs)

**Root Cause:** TypeError: TritonExperts.__init__() got an unexpected keyword argument 'per_channel_quant'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0070 | trae | gpt-5 | vllm | `c0569dbc` |

### RotaryEmbedding.__init__() missing 1 required positional argument: 'dtype' (1 runs)

**Root Cause:** TypeError: RotaryEmbedding.__init__() missing 1 required positional argument: 'dtype'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0079 | trae | gpt-5 | vllm | `dae68969` |

---

## BASELINE_RUNTIME_ERROR (6 runs, 0.7%)

**Description:** RuntimeError in test

### "normal_kernel_cuda" not implemented for 'Int' (4 runs)

**Root Cause:** RuntimeError: "normal_kernel_cuda" not implemented for 'Int'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0037 | trae | claude-sonnet-45 | vllm | `70b808fe` |
| vllm_core-0035 | trae | gpt-5 | vllm | `70b808fe` |
| vllm_core-0035 | trae | gpt-5 | vllm | `70b808fe` |
| vllm_core-0037 | codex | gpt-5 | vllm | `70b808fe` |

### "normal_kernel_cuda" not implemented for 'Float8_e4m3fn' (2 runs)

**Root Cause:** RuntimeError: "normal_kernel_cuda" not implemented for 'Float8_e4m3fn'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0031 | trae | gpt-5 | vllm | `6d0734c5` |
| vllm_core-0031 | trae | gpt-5 | vllm | `6d0734c5` |

---

## BASELINE_ASSERTION (5 runs, 0.6%)

**Description:** AssertionError in test

### AssertionError (5 runs)

**Root Cause:** Test assertion failed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0065 | trae | claude-sonnet-45 | vllm | `b2e0ad3b` |
| vllm_core-0071 | trae | gpt-5 | vllm | `c45f3c3a` |
| vllm_core-0062 | trae | gpt-5 | vllm | `b2e0ad3b` |
| vllm_core-0065 | codex | gpt-5 | vllm | `b2e0ad3b` |
| vllm_core-0075 | codex | gpt-5 | vllm | `c45f3c3a` |

---

## BASELINE_EXCEPTION (36 runs, 4.3%)

**Description:** Other exception in test

### ValueError: 'aimv2' is already used by a Transformers config, pick another name. (12 runs)

**Root Cause:** ValueError: 'aimv2' is already used by a Transformers config, pick another name.

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0030 | trae | claude-sonnet-45 | vllm | `67da5720` |
| vllm_bedrock_sonnet45-0055 | trae | claude-sonnet-45 | vllm | `9a3b8832` |
| vllm_bedrock_sonnet45-0038 | trae | claude-sonnet-45 | vllm | `7661e92e` |
| vllm_bedrock_sonnet45-0007 | trae | claude-sonnet-45 | vllm | `22dd9c27` |
| vllm_core-0028 | trae | gpt-5 | vllm | `67da5720` |
| vllm_core-0036 | trae | gpt-5 | vllm | `7661e92e` |
| vllm_core-0028 | trae | gpt-5 | vllm | `67da5720` |
| vllm_core-0036 | trae | gpt-5 | vllm | `7661e92e` |
| vllm_core-0077 | trae | gpt-5 | vllm | `d55e446d` |
| vllm_core-0006 | trae | gpt-5 | vllm | `22dd9c27` |
| ... | | | | |
| *(2 more runs)* | | | | |

### NameError: name 'vllm_ops' is not defined (9 runs)

**Root Cause:** NameError: name 'vllm_ops' is not defined

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0011 | trae | claude-sonnet-45 | vllm | `2a052011` |
| vllm_bedrock_sonnet45-0011 | trae | claude-sonnet-45 | vllm | `2a052011` |
| vllm_bedrock_sonnet45-0020 | trae | claude-sonnet-45 | vllm | `379da6dc` |
| vllm_core-0019 | trae | gpt-5 | vllm | `379da6dc` |
| vllm_core-0010 | trae | gpt-5 | vllm | `2a052011` |
| vllm_core-0019 | trae | gpt-5 | vllm | `379da6dc` |
| vllm_core-0010 | trae | gpt-5 | vllm | `2a052011` |
| vllm_core-0020 | codex | gpt-5 | vllm | `379da6dc` |
| vllm_core-0011 | codex | gpt-5 | vllm | `2a052011` |

### ValueError: top_k must be -1 (disable), or at least 1, got 0. (5 runs)

**Root Cause:** ValueError: top_k must be -1 (disable), or at least 1, got 0.

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0014 | trae | claude-sonnet-45 | vllm | `30172b49` |
| vllm_bedrock_sonnet45-0014 | trae | claude-sonnet-45 | vllm | `30172b49` |
| vllm_core-0013 | trae | gpt-5 | vllm | `30172b49` |
| vllm_core-0013 | trae | gpt-5 | vllm | `30172b49` |
| vllm_core-0014 | codex | gpt-5 | vllm | `30172b49` |

### NotImplementedError: "normal_kernel_cuda" not implemented for 'Float8_e4m3fn' (4 runs)

**Root Cause:** NotImplementedError: "normal_kernel_cuda" not implemented for 'Float8_e4m3fn'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_040_915140fd | trae | claude-sonnet-45 | sglang | `915140fd` |
| sglang_024_5e023301 | trae | claude-sonnet-45 | sglang | `5e023301` |
| sglang_040_915140fd | trae | gpt-5 | sglang | `915140fd` |
| sglang_024_5e023301 | trae | gpt-5 | sglang | `5e023301` |

### NameError: name 'vocab_size' is not defined (3 runs)

**Root Cause:** NameError: name 'vocab_size' is not defined

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0050 | trae | claude-sonnet-45 | vllm | `9323a315` |
| vllm_core-0047 | trae | gpt-5 | vllm | `9323a315` |
| vllm_core-0050 | codex | gpt-5 | vllm | `9323a315` |

### UnboundLocalError: cannot access local variable 'num_prefills' where it is not a (2 runs)

**Root Cause:** UnboundLocalError: cannot access local variable 'num_prefills' where it is not associated with a value

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0028 | trae | claude-sonnet-45 | vllm | `61b8cea3` |
| vllm_core-0026 | trae | gpt-5 | vllm | `61b8cea3` |

### OSError: [Errno 28] No space left on device: '/tmp/tmpjxkcscix' (1 runs)

**Root Cause:** OSError: [Errno 28] No space left on device: '/tmp/tmpjxkcscix'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0098 | codex | gpt-5 | vllm | `fc7b8d1e` |

---

## BASELINE_UNKNOWN (1 runs, 0.1%)

**Description:** Unknown test failure

### stderr: ...ffffffffffffffffffffffffffffffffffffffffff_155858.log: No space left on device

 (1 runs)

**Root Cause:** Unknown error, stderr ends with: 25-12-22_15-52-55_505683_155858/logs/python-core-driver-01000000ffffffffffffffffffffffffffffffffffffffffffffffff_155858.log: No space left on device



| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0094 | codex | gpt-5 | vllm | `f26c4aee` |

---

## TEST_IMPORT_ERROR (76 runs, 9.1%)

**Description:** Test script import error

### Missing: transformers (24 runs)

**Root Cause:** Test requires module 'transformers' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_039_912788c0 | trae | claude-sonnet-45 | sglang | `912788c0` |
| sglang_021_4418f599 | trae | claude-sonnet-45 | sglang | `4418f599` |
| sglang_000_021f76e4 | trae | claude-sonnet-45 | sglang | `021f76e4` |
| sglang_063_cd7e32e2 | trae | claude-sonnet-45 | sglang | `cd7e32e2` |
| sglang_014_2a413829 | trae | claude-sonnet-45 | sglang | `2a413829` |
| sglang_038_8f8f96a6 | trae | claude-sonnet-45 | sglang | `8f8f96a6` |
| sglang_008_205d5cb4 | trae | claude-sonnet-45 | sglang | `205d5cb4` |
| sglang_065_da47621c | trae | claude-sonnet-45 | sglang | `da47621c` |
| sglang_053_adca585b | trae | claude-sonnet-45 | sglang | `adca585b` |
| sglang_070_df7f61ee | trae | claude-sonnet-45 | sglang | `df7f61ee` |
| ... | | | | |
| *(14 more runs)* | | | | |

### Missing: vllm._C (14 runs)

**Root Cause:** Test requires module 'vllm._C' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0068 | trae | claude-sonnet-45 | vllm | `b6d10354` |
| vllm_bedrock_sonnet45-0074 | trae | claude-sonnet-45 | vllm | `c0569dbc` |
| vllm_bedrock_sonnet45-0015 | trae | claude-sonnet-45 | vllm | `3092375e` |
| vllm_bedrock_sonnet45-0005 | trae | claude-sonnet-45 | vllm | `21d93c14` |
| vllm_bedrock_sonnet45-0024 | trae | claude-sonnet-45 | vllm | `4fb56914` |
| vllm_bedrock_sonnet45-0001 | trae | claude-sonnet-45 | vllm | `015069b0` |
| vllm_core-0004 | trae | gpt-4o | vllm | `21d93c14` |
| vllm_core-0014 | trae | gpt-5 | vllm | `3092375e` |
| vllm_core-0014 | trae | gpt-5 | vllm | `3092375e` |
| vllm_core-0065 | trae | gpt-5 | vllm | `b6d10354` |
| ... | | | | |
| *(4 more runs)* | | | | |

### Missing: decord (6 runs)

**Root Cause:** Test requires module 'decord' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_043_93470a14 | trae | claude-sonnet-45 | sglang | `93470a14` |
| sglang_009_23c764b1 | trae | claude-sonnet-45 | sglang | `23c764b1` |
| sglang_036_86a876d8 | trae | claude-sonnet-45 | sglang | `86a876d8` |
| sglang_043_93470a14 | trae | gpt-5 | sglang | `93470a14` |
| sglang_036_86a876d8 | trae | gpt-5 | sglang | `86a876d8` |
| sglang_009_23c764b1 | trae | gpt-5 | sglang | `23c764b1` |

### Missing: outlines (6 runs)

**Root Cause:** Test requires module 'outlines' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_026_6a2941f4 | trae | claude-sonnet-45 | sglang | `6a2941f4` |
| sglang_051_ab4a83b2 | trae | claude-sonnet-45 | sglang | `ab4a83b2` |
| sglang_013_2854a5ea | trae | claude-sonnet-45 | sglang | `2854a5ea` |
| sglang_051_ab4a83b2 | trae | gpt-5 | sglang | `ab4a83b2` |
| sglang_026_6a2941f4 | trae | gpt-5 | sglang | `6a2941f4` |
| sglang_013_2854a5ea | trae | gpt-5 | sglang | `2854a5ea` |

### Missing: librosa (6 runs)

**Root Cause:** Test requires module 'librosa' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0078 | trae | claude-sonnet-45 | vllm | `ce6bf3a2` |
| vllm_bedrock_sonnet45-0012 | trae | claude-sonnet-45 | vllm | `2deb029d` |
| vllm_core-0011 | trae | gpt-5 | vllm | `2deb029d` |
| vllm_core-0011 | trae | gpt-5 | vllm | `2deb029d` |
| vllm_core-0074 | trae | gpt-5 | vllm | `ce6bf3a2` |
| vllm_core-0012 | codex | gpt-5 | vllm | `2deb029d` |

### Missing: pybase64 (4 runs)

**Root Cause:** Test requires module 'pybase64' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_050_a99801e0 | trae | claude-sonnet-45 | sglang | `a99801e0` |
| sglang_020_3212c2ad | trae | claude-sonnet-45 | sglang | `3212c2ad` |
| sglang_050_a99801e0 | trae | gpt-5 | sglang | `a99801e0` |
| sglang_020_3212c2ad | trae | gpt-5 | sglang | `3212c2ad` |

### Missing: transformers_neuronx (4 runs)

**Root Cause:** Test requires module 'transformers_neuronx' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0031 | trae | claude-sonnet-45 | vllm | `6a417b86` |
| vllm_core-0029 | trae | gpt-5 | vllm | `6a417b86` |
| vllm_core-0029 | trae | gpt-5 | vllm | `6a417b86` |
| vllm_core-0031 | codex | gpt-5 | vllm | `6a417b86` |

### Missing: vllm (3 runs)

**Root Cause:** Test requires module 'vllm' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_004_148254d4 | trae | claude-sonnet-45 | sglang | `148254d4` |
| sglang_015_2a754e57 | trae | gpt-5 | sglang | `2a754e57` |
| sglang_004_148254d4 | trae | gpt-5 | sglang | `148254d4` |

### Missing: deep_ep (3 runs)

**Root Cause:** Test requires module 'deep_ep' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0061 | trae | claude-sonnet-45 | vllm | `ac45c44d` |
| vllm_core-0058 | trae | gpt-5 | vllm | `ac45c44d` |
| vllm_core-0061 | codex | gpt-5 | vllm | `ac45c44d` |

### Missing: benchmark.kernels.minmax_text_01_lighting_attention (2 runs)

**Root Cause:** Test requires module 'benchmark.kernels.minmax_text_01_lighting_attention' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_061_c2f212d6 | trae | claude-sonnet-45 | sglang | `c2f212d6` |
| sglang_061_c2f212d6 | trae | gpt-5 | sglang | `c2f212d6` |

### Missing: einops (2 runs)

**Root Cause:** Test requires module 'einops' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_018_2f427491 | trae | claude-sonnet-45 | sglang | `2f427491` |
| sglang_018_2f427491 | trae | gpt-5 | sglang | `2f427491` |

### Missing: outlines_core.fsm (1 runs)

**Root Cause:** Test requires module 'outlines_core.fsm' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0027 | trae | gpt-5 | vllm | `660470e5` |

### Missing: vllm.core.block (1 runs)

**Root Cause:** Test requires module 'vllm.core.block' which is not installed

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0075 | trae | gpt-5 | vllm | `cf2f084d` |

---

## TARGET_NOT_RESOLVED (55 runs, 6.6%)

**Description:** Cannot find optimization target

### cannot import name 'default_cache_dir' from 'triton.runtime.cache' (/home/ubuntu (22 runs)

**Root Cause:** Cannot resolve optimization target: cannot import name 'default_cache_dir' from 'triton.runtime.cache' (/home/ubuntu/OmniPerf-Bench/bench-env/lib/python3.12

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_069_ddcf9fe3 | trae | claude-sonnet-45 | sglang | `ddcf9fe3` |
| sglang_005_187b85b7 | trae | claude-sonnet-45 | sglang | `187b85b7` |
| sglang_002_10189d08 | trae | claude-sonnet-45 | sglang | `10189d08` |
| sglang_060_c2bd094d | trae | claude-sonnet-45 | sglang | `c2bd094d` |
| sglang_059_c087ddd6 | trae | claude-sonnet-45 | sglang | `c087ddd6` |
| sglang_047_a191a0e4 | trae | claude-sonnet-45 | sglang | `a191a0e4` |
| sglang_041_9183c23e | trae | claude-sonnet-45 | sglang | `9183c23e` |
| sglang_003_132dad87 | trae | claude-sonnet-45 | sglang | `132dad87` |
| sglang_055_b1e5a33a | trae | claude-sonnet-45 | sglang | `b1e5a33a` |
| sglang_019_31589e17 | trae | claude-sonnet-45 | sglang | `31589e17` |
| ... | | | | |
| *(12 more runs)* | | | | |

### Failed to import vLLM components: cannot import name 'PromptAdapterConfig' from  (15 runs)

**Root Cause:** Cannot resolve optimization target: Failed to import vLLM components: cannot import name 'PromptAdapterConfig' from 'vllm.config' (/home/ubuntu/OmniPerf-Ben

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| chunked_local_attn_opt-0000 | openhands | gpt-5 | vllm | `8aa1485f` |
| vllm_core-0000 | trae | gpt-4o | vllm | `8aa1485f` |
| vllm_core-0000 | trae | gpt-4o | vllm | `8aa1485f` |
| vllm_core-0000 | trae | gpt-4o | vllm | `8aa1485f` |
| vllm_core-0043 | trae | gpt-5 | vllm | `8aa1485f` |
| vllm_core-0000 | trae | gpt-5 | vllm | `8aa1485f` |
| vllm_core-0000 | trae | gpt-5 | vllm | `8aa1485f` |
| vllm_core-0000 | trae | gpt-5 | vllm | `8aa1485f` |
| vllm_core-0000 | trae | gpt-5 | vllm | `8aa1485f` |
| vllm_core-0043 | trae | gpt-5 | vllm | `8aa1485f` |
| ... | | | | |
| *(5 more runs)* | | | | |

### type object 'P2pNcclEngine' has no attribute 'extract_kv_from_layer' (5 runs)

**Root Cause:** Cannot resolve optimization target: type object 'P2pNcclEngine' has no attribute 'extract_kv_from_layer'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0045 | trae | claude-sonnet-45 | vllm | `8a4e5c5f` |
| vllm_core-0042 | trae | gpt-5 | vllm | `8a4e5c5f` |
| vllm_core-0042 | trae | gpt-5 | vllm | `8a4e5c5f` |
| vllm_core-0042 | trae | gpt-5 | vllm | `8a4e5c5f` |
| vllm_core-0045 | codex | gpt-5 | vllm | `8a4e5c5f` |

### module 'vllm.model_executor.models.utils' has no attribute 'fast_topk' (3 runs)

**Root Cause:** Cannot resolve optimization target: module 'vllm.model_executor.models.utils' has no attribute 'fast_topk'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0072 | trae | claude-sonnet-45 | vllm | `bd6028d6` |
| vllm_core-0068 | trae | gpt-5 | vllm | `bd6028d6` |
| vllm_core-0072 | codex | gpt-5 | vllm | `bd6028d6` |

### module 'vllm.model_executor.layers.quantization.utils.fp8_utils' has no attribut (2 runs)

**Root Cause:** Cannot resolve optimization target: module 'vllm.model_executor.layers.quantization.utils.fp8_utils' has no attribute 'apply_fp8_linear_generic'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0070 | trae | claude-sonnet-45 | vllm | `baeded25` |
| vllm_core-0066 | trae | gpt-5 | vllm | `baeded25` |

### cannot import name 'build_regex_from_schema' from 'outlines.fsm.json_schema' (/h (2 runs)

**Root Cause:** Cannot resolve optimization target: cannot import name 'build_regex_from_schema' from 'outlines.fsm.json_schema' (/home/ubuntu/OmniPerf-Bench/bench-env/lib/

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0029 | trae | claude-sonnet-45 | vllm | `660470e5` |
| vllm_core-0027 | trae | gpt-5 | vllm | `660470e5` |

### module 'vllm.core.block_manager' has no attribute 'UncachedBlockAllocator' (2 runs)

**Root Cause:** Cannot resolve optimization target: module 'vllm.core.block_manager' has no attribute 'UncachedBlockAllocator'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0052 | trae | claude-sonnet-45 | vllm | `9474e89b` |
| vllm_core-0049 | trae | gpt-5 | vllm | `9474e89b` |

### module 'vllm.v1.engine.output_processor' has no attribute 'RequestOutputCollecto (2 runs)

**Root Cause:** Cannot resolve optimization target: module 'vllm.v1.engine.output_processor' has no attribute 'RequestOutputCollector'

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0054 | trae | gpt-5 | vllm | `9d72daf4` |
| vllm_core-0057 | codex | gpt-5 | vllm | `9d72daf4` |

### Failed to import required classes: cannot import name 'XGrammarConfig' from 'vll (1 runs)

**Root Cause:** Cannot resolve optimization target: Failed to import required classes: cannot import name 'XGrammarConfig' from 'vllm.model_executor.guided_decoding.xgramma

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0081 | trae | gpt-5 | vllm | `e206b543` |

### cannot import name 'cuda_utils' from partially initialized module 'vllm' (most l (1 runs)

**Root Cause:** Cannot resolve optimization target: cannot import name 'cuda_utils' from partially initialized module 'vllm' (most likely due to a circular import) (/tmp/ev

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0086 | trae | gpt-5 | vllm | `ec3b5ce9` |

---

## OPT_PATH_NOT_HIT (2 runs, 0.2%)

**Description:** Optimization path not triggered

### Custom allreduce not initialized (2 runs)

**Root Cause:** Optimization path not triggered: Custom allreduce not initialized

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_011_25e1816e | trae | claude-sonnet-45 | sglang | `25e1816e` |
| sglang_011_25e1816e | trae | gpt-5 | sglang | `25e1816e` |

---

## GIT_WORKTREE_FAILED (33 runs, 4.0%)

**Description:** Git commit not in local repo

### Commit 067c34a1 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 067c34a1 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0026 | codex | gpt-5 | vllm | `58eee5f2` |

### Commit 005ae9be not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 005ae9be - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0003 | codex | gpt-5 | vllm | `0ec82edd` |

### Commit 25373b6c not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 25373b6c - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0067 | codex | gpt-5 | vllm | `b690e348` |

### Commit 0df4d9b0 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 0df4d9b0 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0024 | codex | gpt-5 | vllm | `4fb56914` |

### Commit 89ac266b not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 89ac266b - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0046 | codex | gpt-5 | vllm | `8aa1485f` |

### Commit a869baca not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit a869baca - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0088 | codex | gpt-5 | vllm | `e7523c2e` |

### Commit 526078a9 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 526078a9 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0028 | codex | gpt-5 | vllm | `61b8cea3` |

### Commit 1d35662e not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 1d35662e - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0085 | codex | gpt-5 | vllm | `e206b543` |

### Commit 3e1c76cf not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 3e1c76cf - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0070 | codex | gpt-5 | vllm | `baeded25` |

### Commit 733e7c9e not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 733e7c9e - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0019 | codex | gpt-5 | vllm | `35fad35a` |

### Commit f728ab8e not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit f728ab8e - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0010 | codex | gpt-5 | vllm | `299ebb62` |

### Commit 2b04c209 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 2b04c209 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0004 | codex | gpt-5 | vllm | `19d98e0c` |

### Commit 3a1e6481 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 3a1e6481 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0054 | codex | gpt-5 | vllm | `99abb8b6` |

### Commit f168b857 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit f168b857 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0038 | codex | gpt-5 | vllm | `7661e92e` |

### Commit 70363bcc not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 70363bcc - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0051 | codex | gpt-5 | vllm | `93e5f3c5` |

### Commit 2a0309a6 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 2a0309a6 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0095 | codex | gpt-5 | vllm | `fa63e710` |

### Commit edc4fa31 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit edc4fa31 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0022 | codex | gpt-5 | vllm | `3b61cb45` |

### Commit 92b0ce2a not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 92b0ce2a - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0096 | codex | gpt-5 | vllm | `fb0acb6c` |

### Commit 3cd91dc9 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 3cd91dc9 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0015 | codex | gpt-5 | vllm | `3092375e` |

### Commit e642ec96 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit e642ec96 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0059 | codex | gpt-5 | vllm | `9f1710f1` |

### Commit 3014c920 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 3014c920 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0055 | codex | gpt-5 | vllm | `9a3b8832` |

### Commit 88f6ba32 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 88f6ba32 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0002 | codex | gpt-5 | vllm | `0d243f2a` |

### Commit dd572c0a not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit dd572c0a - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0084 | codex | gpt-5 | vllm | `dcc6cfb9` |

### Commit 0032903a not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 0032903a - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0009 | codex | gpt-5 | vllm | `296f927f` |

### Commit c8d70e24 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit c8d70e24 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0023 | codex | gpt-5 | vllm | `4c822298` |

### Commit 4ce64e2d not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 4ce64e2d - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0087 | codex | gpt-5 | vllm | `e493e485` |

### Commit 1da8f0e1 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 1da8f0e1 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0093 | codex | gpt-5 | vllm | `f092153f` |

### Commit a6d795d5 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit a6d795d5 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0007 | codex | gpt-5 | vllm | `22dd9c27` |

### Commit 90f1e554 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 90f1e554 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0089 | codex | gpt-5 | vllm | `e7b20426` |

### Commit f508e03e not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit f508e03e - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0086 | codex | gpt-5 | vllm | `e3580537` |

### Commit 33368140 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 33368140 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0076 | codex | gpt-5 | vllm | `ca7a2d5f` |

### Commit 7d945771 not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 7d945771 - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0033 | codex | gpt-5 | vllm | `6d0734c5` |

### Commit 5b8a1fde not in repo (1 runs)

**Root Cause:** Cannot checkout pre-commit 5b8a1fde - commit does not exist in local repository clone

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0042 | codex | gpt-5 | vllm | `83450458` |

---

## NO_TEST_SCRIPT (13 runs, 1.6%)

**Description:** No test script for this commit

### No test for f06e90c2 (2 runs)

**Root Cause:** HuggingFace dataset has no test script for human commit f06e90c2

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_076_f06e90c2 | trae | claude-sonnet-45 | sglang | `f06e90c2` |
| sglang_076_f06e90c2 | trae | gpt-5 | sglang | `f06e90c2` |

### No test for fbcbb263 (2 runs)

**Root Cause:** HuggingFace dataset has no test script for human commit fbcbb263

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_078_fbcbb263 | trae | claude-sonnet-45 | sglang | `fbcbb263` |
| sglang_078_fbcbb263 | trae | gpt-5 | sglang | `fbcbb263` |

### No test for e88dd482 (2 runs)

**Root Cause:** HuggingFace dataset has no test script for human commit e88dd482

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_074_e88dd482 | trae | claude-sonnet-45 | sglang | `e88dd482` |
| sglang_074_e88dd482 | trae | gpt-5 | sglang | `e88dd482` |

### No test for f0653886 (2 runs)

**Root Cause:** HuggingFace dataset has no test script for human commit f0653886

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_075_f0653886 | trae | claude-sonnet-45 | sglang | `f0653886` |
| sglang_075_f0653886 | trae | gpt-5 | sglang | `f0653886` |

### No test for 5e5c8e09 (2 runs)

**Root Cause:** HuggingFace dataset has no test script for human commit 5e5c8e09

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0027 | trae | claude-sonnet-45 | vllm | `5e5c8e09` |
| vllm_core-0027 | codex | gpt-5 | vllm | `5e5c8e09` |

### No test for 81ede99c (2 runs)

**Root Cause:** HuggingFace dataset has no test script for human commit 81ede99c

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0041 | trae | claude-sonnet-45 | vllm | `81ede99c` |
| vllm_core-0041 | codex | gpt-5 | vllm | `81ede99c` |

### No test for b9986454 (1 runs)

**Root Cause:** HuggingFace dataset has no test script for human commit b9986454

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0069 | codex | gpt-5 | vllm | `b9986454` |

---

## PATCH_INVALID (31 runs, 3.7%)

**Description:** Patch marked as invalid

### 8 files, +397/-95 (3 runs)

**Root Cause:** Patch has 8 files changed (+397/-95 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0000 | trae | gpt-5 | vllm | `015069b0` |
| vllm_core-0000 | trae | gpt-5 | vllm | `2deb029d` |
| vllm_core-0000 | trae | gpt-5 | vllm | `015069b0` |

### 3 files, +5/-5 (2 runs)

**Root Cause:** Patch has 3 files changed (+5/-5 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0002 | trae | claude-sonnet-45 | vllm | `0d243f2a` |
| vllm_core-0001 | trae | gpt-5 | vllm | `0d243f2a` |

### 7 files, +30/-20 (1 runs)

**Root Cause:** Patch has 7 files changed (+30/-20 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_015_2a754e57 | trae | claude-sonnet-45 | sglang | `2a754e57` |

### 1 files, +1/-1 (1 runs)

**Root Cause:** Patch has 1 files changed (+1/-1 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0048 | trae | claude-sonnet-45 | vllm | `8c1e77fb` |

### 2 files, +12/-1 (1 runs)

**Root Cause:** Patch has 2 files changed (+12/-1 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_bedrock_sonnet45-0017 | trae | claude-sonnet-45 | vllm | `3127e975` |

### 2 files, +21/-3 (1 runs)

**Root Cause:** Patch has 2 files changed (+21/-3 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0016 | trae | gpt-5 | vllm | `3127e975` |

### 2 files, +8/-2 (1 runs)

**Root Cause:** Patch has 2 files changed (+8/-2 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0016 | trae | gpt-5 | vllm | `3127e975` |

### 1 files, +25/-1 (1 runs)

**Root Cause:** Patch has 1 files changed (+25/-1 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0045 | trae | gpt-5 | vllm | `8c1e77fb` |

### 2 files, +145/-26 (1 runs)

**Root Cause:** Patch has 2 files changed (+145/-26 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0088 | trae | gpt-5 | vllm | `eefbf4a6` |

### 2 files, +15/-0 (1 runs)

**Root Cause:** Patch has 2 files changed (+15/-0 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0017 | codex | gpt-5 | vllm | `3127e975` |

### 13 files, +258/-18 (1 runs)

**Root Cause:** Patch has 13 files changed (+258/-18 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0071 | codex | gpt-5 | vllm | `bc7c4d20` |

### 11 files, +239/-17 (1 runs)

**Root Cause:** Patch has 11 files changed (+239/-17 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0078 | codex | gpt-5 | vllm | `ce6bf3a2` |

### 10 files, +168/-55 (1 runs)

**Root Cause:** Patch has 10 files changed (+168/-55 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0047 | codex | gpt-5 | vllm | `8bc68e19` |

### 3 files, +34/-19 (1 runs)

**Root Cause:** Patch has 3 files changed (+34/-19 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0081 | codex | gpt-5 | vllm | `d55e446d` |

### 10 files, +192/-15 (1 runs)

**Root Cause:** Patch has 10 files changed (+192/-15 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0068 | codex | gpt-5 | vllm | `b6d10354` |

### 6 files, +72/-17 (1 runs)

**Root Cause:** Patch has 6 files changed (+72/-17 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0005 | codex | gpt-5 | vllm | `21d93c14` |

### 2 files, +17/-6 (1 runs)

**Root Cause:** Patch has 2 files changed (+17/-6 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0083 | codex | gpt-5 | vllm | `dae68969` |

### 5 files, +33/-24 (1 runs)

**Root Cause:** Patch has 5 files changed (+33/-24 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0077 | codex | gpt-5 | vllm | `ccf02fcb` |

### 3 files, +74/-42 (1 runs)

**Root Cause:** Patch has 3 files changed (+74/-42 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0090 | codex | gpt-5 | vllm | `ec3b5ce9` |

### 6 files, +57/-36 (1 runs)

**Root Cause:** Patch has 6 files changed (+57/-36 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0064 | codex | gpt-5 | vllm | `b10e5198` |

### 2 files, +208/-26 (1 runs)

**Root Cause:** Patch has 2 files changed (+208/-26 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0092 | codex | gpt-5 | vllm | `eefbf4a6` |

### 11 files, +161/-19 (1 runs)

**Root Cause:** Patch has 11 files changed (+161/-19 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0074 | codex | gpt-5 | vllm | `c0569dbc` |

### 3 files, +25/-15 (1 runs)

**Root Cause:** Patch has 3 files changed (+25/-15 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0008 | codex | gpt-5 | vllm | `25ebed2f` |

### 5 files, +141/-5 (1 runs)

**Root Cause:** Patch has 5 files changed (+141/-5 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0001 | codex | gpt-5 | vllm | `015069b0` |

### 3 files, +35/-15 (1 runs)

**Root Cause:** Patch has 3 files changed (+35/-15 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0048 | codex | gpt-5 | vllm | `8c1e77fb` |

### 8 files, +64/-24 (1 runs)

**Root Cause:** Patch has 8 files changed (+64/-24 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0044 | codex | gpt-5 | vllm | `89a84b0b` |

### 4 files, +40/-24 (1 runs)

**Root Cause:** Patch has 4 files changed (+40/-24 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0079 | codex | gpt-5 | vllm | `cf2f084d` |

### 5 files, +52/-34 (1 runs)

**Root Cause:** Patch has 5 files changed (+52/-34 lines) but was marked invalid/empty by evaluator

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0052 | codex | gpt-5 | vllm | `9474e89b` |

---

## PATCH_APPLY_FAILED (2 runs, 0.2%)

**Description:** Patch failed to apply

### Patched test failed to produce output (2 runs)

**Root Cause:** Patch could not be applied: Patched test failed to produce output

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| vllm_core-0016 | codex | gpt-5 | vllm | `310aca88` |
| vllm_core-0029 | codex | gpt-5 | vllm | `660470e5` |

---

## UNKNOWN (3 runs, 0.4%)

**Description:** Uncategorized

### status=success (3 runs)

**Root Cause:** Uncategorized: eval_status=success, error=none

| Item ID | Agent | Model | Repo | Commit |
|---------|-------|-------|------|--------|
| sglang_048_a37e1247 | trae | claude-sonnet-45 | sglang | `a37e1247` |
| vllm_bedrock_sonnet45-0047 | trae | claude-sonnet-45 | vllm | `8bc68e19` |
| vllm_core-0044 | trae | gpt-5 | vllm | `8bc68e19` |

---
