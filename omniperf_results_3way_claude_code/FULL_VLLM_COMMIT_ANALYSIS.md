# Full vLLM Commit Analysis: GPU Requirements & Fixability

**Generated**: 2026-01-05
**Total Commits**: 94
**Successfully Benchmarked**: 24 (26%)
**With Full 3-Way Agent Metrics**: 20 (21%)

---

## Executive Summary

**Success Criteria**: A commit is "successful" when it completes a 3-way benchmark comparing:
- **Baseline**: Parent commit (before optimization)
- **Human**: The actual PR commit (ground truth optimization)
- **Agent**: Claude Code's attempted optimization

**Critical Finding**: Only 26% of commits ran benchmarks successfully. Of those, **20 commits (83%) have complete 3-way metrics including agent results**. The remaining 74% failed due to:

| Root Cause | Count | Fixable? | Action Required |
|------------|-------|----------|-----------------|
| Missing baseline wheels | 24 | YES | Build wheels on CPU first |
| No metrics parsed | 15 | PARTIAL | Improve output parsing |
| Modal infrastructure | 10 | YES | Retry (transient errors) |
| Server startup failed | 8 | PARTIAL | Some version-specific |
| vLLM version bugs | 5 | NO | Skip these versions |
| Model too large | 3 | NO | Need more GPUs or skip |
| Wheel download failed | 2 | YES | Build from source |
| Other issues | 2 | PARTIAL | Case-by-case |
| No perf command | 1 | NO | Dataset issue |

**Bottom Line**: ~40 commits (43%) are potentially fixable with infrastructure improvements. ~30 commits (32%) have fundamental issues that cannot be fixed without dataset changes.

---

## GPU Configuration Breakdown

### H100:1 (Single GPU) - 79 commits

| Status | Count | Percentage |
|--------|-------|------------|
| SUCCESS | 22 | 28% |
| FAILED | 57 | 72% |

**Assessment**: Best success rate. Most commits target single GPU workloads.

### H100:2 (2 GPUs) - 1 commit

| Status | Count |
|--------|-------|
| SUCCESS | 0 |
| FAILED | 1 |

**Assessment**: Single commit failed with no metrics. Likely MoE model issue.

### H100:4 (4 GPUs) - 4 commits

| Status | Count |
|--------|-------|
| SUCCESS | 2 |
| FAILED | 2 |

**Assessment**: 50% success rate. Failures are wheel download and server startup issues.

### H100:8 (8 GPUs) - 4 commits

| Status | Count |
|--------|-------|
| SUCCESS | 0 |
| FAILED | 4 |

**Assessment**: 0% success rate. These are large model benchmarks (Mixtral, Nemotron, DeepSeek) that have infrastructure issues. Very expensive to retry.

### Unknown GPU - 6 commits

All failed - these have incomplete benchmark configurations.

---

## Error Category Deep Dive

### 1. SUCCESS (24 commits) - DONE

These benchmarks completed successfully. No action needed.

### 2. MISSING_WHEEL (24 commits) - FIXABLE

**Problem**: Baseline vLLM commit doesn't have a pre-built wheel in S3 or Modal volume.

**Fix**: Run CPU wheel build before GPU benchmark. The runner should:
1. Check if wheel exists in Modal volume
2. If not, build on CPU-only instance first
3. Then run GPU benchmark

**Estimated Cost**: ~$0.10-0.20 per wheel build (CPU instance for ~10-30 min)

### 3. NO_METRICS (15 commits) - PARTIALLY FIXABLE

**Problem**: Benchmark ran but output wasn't parsed correctly.

**Sub-categories**:
- Different output formats across vLLM versions
- Benchmark script errors (missing args, config issues)
- Actual benchmark failures producing no output

**Fix**:
- Improve regex patterns for different vLLM output formats
- Add output capture for debugging
- Skip commits with known bad configs

### 4. MODAL_INFRA (10 commits) - FIXABLE

**Problem**: Modal infrastructure errors ("Broken pipe", sandbox failures)

**Fix**: Simple retry. These are transient errors that succeed on retry.

**Cost**: Just retry cost (~$1-5 per retry depending on GPU config)

### 5. SERVER_FAILED (8 commits) - PARTIALLY FIXABLE

**Problem**: vLLM server failed to start.

**Sub-categories**:
- Model architecture not supported in baseline vLLM
- Memory issues (model too large for GPU config)
- vLLM startup errors (config incompatibility)

**Fixable cases**:
- Memory issues → Use larger GPU config
- Config issues → Fix benchmark command

**Not fixable cases**:
- Model architecture not in old vLLM (e.g., Qwen3, Mamba2)

### 6. VERSION_BUG (5 commits) - NOT FIXABLE

**Problem**: vLLM 0.6.3-0.6.4 has a known port binding bug (issue #8791).

**Reality**: These vLLM versions have a regression that breaks serving benchmarks. Cannot fix without patching vLLM.

**Action**: Skip these commits or use throughput benchmarks instead of serving benchmarks.

### 7. MODEL_TOO_LARGE (3 commits) - NOT FIXABLE

**Models affected**:
- DeepSeek-V3 (671B params) - Won't fit on 8xH100
- DeepSeek-V2 (236B params) - May require special quantization

**Action**: Skip these commits. Would need 16+ H100s or different infrastructure.

### 8. WHEEL_DOWNLOAD_FAILED (2 commits) - FIXABLE

**Problem**: S3 wheel URL exists in config but file is missing/moved.

**Fix**: Build from source instead of downloading.

### 9. OTHER (2 commits) - CASE-BY-CASE

- `526de822`: Config has placeholders (MODEL, BS not replaced) - Dataset issue
- `eefbf4a6`: Git timeout - Infrastructure retry needed

### 10. NO_PERF_COMMAND (1 commit) - NOT FIXABLE

**Problem**: Benchmark config doesn't define a performance command.

**Action**: Skip - dataset quality issue.

---

## Detailed Commit List by Status

### SUCCESSFUL (24 commits)

| Commit | GPU | Model | Notes |
|--------|-----|-------|-------|
| 299ebb62 | H100:1 | unknown | |
| 30172b49 | H100:1 | unknown | |
| 310aca88 | H100:4 | N/A | |
| 3b61cb45 | H100:1 | N/A | |
| 4c822298 | H100:1 | unknown | |
| 58eee5f2 | H100:1 | N/A | |
| 61b8cea3 | H100:1 | Meta-Llama-3-8B-Instruct | |
| 6a417b86 | H100:1 | unknown | |
| 6d0734c5 | H100:1 | Mistral-7B-Instruct-v0.3 | |
| 6dd94dbe | H100:1 | Meta-Llama-3-8B | |
| 70b808fe | H100:1 | unknown | |
| 8a4e5c5f | H100:1 | N/A | |
| 8c1e77fb | H100:1 | N/A | |
| 98f47f2a | H100:1 | N/A | |
| a3223766 | H100:1 | N/A | |
| b55ed6ef | H100:1 | N/A | |
| b690e348 | H100:1 | Bamba-9B-v2 | |
| bc7c4d20 | H100:1 | unknown | |
| ce6bf3a2 | H100:1 | N/A | |
| ed250545 | H100:1 | N/A | |
| f26c4aee | H100:4 | N/A | |
| fa63e710 | H100:1 | N/A | |
| fc542144 | H100:1 | Llama-3.1-8B-Instruct | |
| fe66b347 | H100:1 | unknown | |

---

### MISSING_WHEEL - Build Required (24 commits)

| Commit | GPU | Model | Missing Baseline Wheel |
|--------|-----|-------|----------------------|
| 2a052011 | H100:1 | Mixtral-8x7B-Instruct-v0.1 | 36fb68f94792 |
| 2f192835 | H100:1 | N/A | 95baec828f3e |
| 3476ed08 | H100:1 | Llama-3.1-8B-Instruct | 54600709b6d4 |
| 3a243095 | H100:1 | N/A | 64172a976c8d |
| 660470e5 | H100:1 | Llama-3.1-8B-Instruct | 8d59dbb00044 |
| 6ce01f30 | H100:1 | N/A | 6a11fdfbb8d6 |
| 6d646d08 | H100:1 | Llama-3-8B-Instruct | 95a178f86120 |
| 6e36f4fa | H100:1 | N/A | dd2a6a82e3f4 |
| 7c01f706 | H100:1 | Llama-3.1-8B-Instruct | 51e971d39e12 |
| 80aa7e91 | H100:1 | Llama-3.1-8B-Instruct | bd43973522ea |
| 89a84b0b | H100:1 | N/A | 084a01fd3544 |
| 8bc68e19 | H100:1 | N/A | 0fca3cdcf265 |
| 9474e89b | H100:1 | llama-7b | 20478c4d3abc |
| 9ed82e70 | H100:1 | N/A | 51f8aa90ad40 |
| ad8d696a | H100:1 | N/A | 3d925165f2b1 |
| b6d10354 | H100:1 | N/A | 51c31bc10ca7 |
| bfdb1ba5 | H100:1 | N/A | cf2f084d56a1 |
| c45f3c3a | H100:1 | N/A | 7a7929abe8e2 |
| cf2f084d | H100:1 | N/A | f721096d48a7 |
| d4bc1a4d | H100:1 | opt-125m | b56b6ca0d650 |
| d7740ea4 | H100:1 | N/A | cc466a32903d |
| e3580537 | H100:1 | N/A | f508e03e7f2d |
| ec3b5ce9 | H100:1 | N/A | 6368e777a8ea |
| fc7b8d1e | H100:1 | N/A | 67abdbb42fdb |

**Fix**: Pre-build these 24 baseline wheels on CPU instances before running benchmarks.

---

### NO_METRICS - Need Investigation (15 commits)

| Commit | GPU | Model | Fixable? | Reason |
|--------|-----|-------|----------|--------|
| 22d33bac | H100:1 | unknown | MAYBE | Output parsing issue |
| 22dd9c27 | H100:1 | Llama-3.1-8B-Instruct | MAYBE | VLLM_USE_V1 in old version |
| 2deb029d | unknown | N/A | NO | Missing GPU config |
| 3092375e | H100:1 | unknown | MAYBE | V1 serialization version |
| 35fad35a | H100:1 | unknown | MAYBE | V1 Sampler version |
| 83450458 | H100:1 | N/A | NO | ngram_prompt_lookup_max=None |
| 8d75fe48 | H100:1 | Llama-3-8B-FP8 | MAYBE | FP8 output parsing |
| 93e5f3c5 | H100:1 | unknown | MAYBE | Server startup issue |
| 9badee53 | H100:1 | unknown | NO | Missing ShareGPT dataset |
| 9d72daf4 | H100:1 | unknown | MAYBE | Unknown |
| aea94362 | H100:1 | Llama-3.1-8B-Instruct | MAYBE | V1 serving version |
| b10e5198 | H100:1 | unknown | MAYBE | Unknown |
| bd6028d6 | H100:2 | unknown | MAYBE | MoE benchmark issue |
| e206b543 | H100:1 | unknown | MAYBE | Unknown |
| fb0acb6c | H100:8 | unknown | MAYBE | Large model benchmark |

---

### MODAL_INFRA - Retry Required (10 commits)

| Commit | GPU | Model | Error |
|--------|-----|-------|-------|
| 015069b0 | H100:1 | unknown | Broken pipe |
| 296f927f | H100:1 | unknown | Broken pipe |
| 67da5720 | H100:1 | unknown | Broken pipe |
| 7661e92e | H100:8 | Nemotron-4-340B | Broken pipe |
| 99abb8b6 | H100:1 | unknown | Broken pipe |
| c0569dbc | H100:1 | Qwen3-30B-A3B-FP8 | Broken pipe |
| ca7a2d5f | H100:1 | unknown | Broken pipe |
| ccf02fcb | H100:1 | unknown | Broken pipe |
| dcc6cfb9 | H100:1 | Qwen3-30B-A3B-FP8 | Broken pipe |
| e7b20426 | H100:1 | Yi-1.5-9B-Chat | Broken pipe |

**Fix**: Simple retry - these are transient Modal infrastructure errors.

---

### SERVER_FAILED - Various Issues (8 commits)

| Commit | GPU | Model | Root Cause | Fixable? |
|--------|-----|-------|------------|----------|
| 0d243f2a | H100:1 | Mixtral-8x7B | ROCm-specific commit | NO - AMD code |
| 0ec82edd | H100:1 | Qwen3-30B-A3B | CUDA graph timeout | MAYBE - memory |
| 8aa1485f | H100:4 | Llama-4-Scout-17B | New model arch | NO - not in baseline |
| 9a3b8832 | H100:1 | Qwen2.5-VL-3B | VL model support | NO - not in baseline |
| d55e446d | H100:1 | unknown | Platform plugin error | MAYBE |
| dae68969 | H100:8 | unknown | 8-GPU coordination | MAYBE - retry |
| e493e485 | H100:1 | unknown | Platform plugin error | MAYBE |
| e7523c2e | H100:1 | gemma-3-12b | New model arch | NO - not in baseline |

---

### VERSION_BUG - Skip These (5 commits)

| Commit | GPU | vLLM Version | Bug |
|--------|-----|--------------|-----|
| 25ebed2f | H100:1 | 0.6.4.post2.dev375 | Port binding #8791 |
| 88693683 | H100:1 | 0.6.4.post2.dev368 | Port binding #8791 |
| 9323a315 | H100:1 | 0.6.4.post2.dev218 | Port binding #8791 |
| b2e0ad3b | H100:1 | 0.6.3.post2.dev398 | Port binding #8791 |
| f092153f | H100:1 | 0.6.4.post2.dev330 | Port binding #8791 |

**Action**: Skip - these vLLM versions have known bugs that prevent serving benchmarks.

---

### MODEL_TOO_LARGE - Skip (3 commits)

| Commit | GPU | Model | Size |
|--------|-----|-------|------|
| 4fb56914 | unknown | DeepSeek-V3-0324 | 671B |
| ac45c44d | unknown | DeepSeek-V2 | 236B |
| baeded25 | unknown | DeepSeek-V3 | 671B |

**Action**: Skip - would require 16+ H100s.

---

### WHEEL_DOWNLOAD_FAILED - Build from Source (2 commits)

| Commit | GPU | Model | Missing S3 Wheel |
|--------|-----|-------|------------------|
| 21d93c14 | H100:8 | Mixtral-8x7B-v0.1 | f1c85201... |
| 379da6dc | H100:4 | Meta-Llama-3-70B | ebce310b... |

**Fix**: Build these wheels from source instead of downloading.

---

### OTHER - Case by Case (2 commits)

| Commit | GPU | Model | Issue | Fix |
|--------|-----|-------|-------|-----|
| 526de822 | unknown | Qwen2-7B | Placeholder config | Dataset fix needed |
| eefbf4a6 | H100:1 | Qwen3-30B-A3B-FP8 | Git timeout | Retry |

---

### NO_PERF_COMMAND - Skip (1 commit)

| Commit | Issue |
|--------|-------|
| 3127e975 | No performance command defined in config |

---

## Recommendations

### Immediate Actions (High ROI)

1. **Retry MODAL_INFRA commits** (10 commits)
   - Cost: ~$10-30 for retries
   - Expected success: 80-90%

2. **Pre-build missing baseline wheels** (24 commits)
   - Cost: ~$5-10 for CPU builds
   - Expected success: 90%+ after wheel fix

3. **Retry WHEEL_DOWNLOAD_FAILED** (2 commits)
   - Build from source instead of S3 download

### Medium-term Fixes

4. **Improve NO_METRICS parsing** (15 commits)
   - Add more output format patterns
   - Some may still fail (bad configs)

5. **Investigate SERVER_FAILED** (8 commits)
   - ~3-4 may be fixable with retries
   - ~4-5 have fundamental architecture issues

### Skip Permanently

6. **VERSION_BUG commits** (5 commits) - vLLM has known regression
7. **MODEL_TOO_LARGE** (3 commits) - Need different infrastructure
8. **NO_PERF_COMMAND** (1 commit) - Dataset issue

---

## Cost Estimate for Full Fix

| Action | Commits | Est. Cost |
|--------|---------|-----------|
| Retry MODAL_INFRA | 10 | $20-50 |
| Build missing wheels | 24 | $5-10 |
| Retry with wheel fix | 24 | $50-100 |
| Investigate NO_METRICS | 15 | $30-60 |
| Total | ~70 | **$100-200** |

Expected additional successes: **+30-40 commits** (from 24 to 54-64)

---

## Final Verifiability Assessment

| Category | Count | Status |
|----------|-------|--------|
| Already verified | 24 | DONE |
| Likely verifiable (fix infra) | 34 | FIXABLE |
| Possibly verifiable (needs investigation) | 15 | MAYBE |
| Not verifiable (fundamental issues) | 21 | SKIP |
| **Total** | **94** | |

**Realistic Success Rate After Fixes**: 55-65% (vs current 26%)
