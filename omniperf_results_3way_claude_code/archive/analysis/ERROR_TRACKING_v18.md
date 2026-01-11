# Benchmark v18 Error Tracking

**Run Start**: 2026-01-03 21:03:20 UTC
**Total Commits**: 59
**Log File**: `rerun_log_v18_server_log_fix.txt`

## Fixes Applied in This Run

- **v34 Server Log Capture**: `log_file="/tmp/vllm_server.log"` added to all `start_server()` calls ✅ PARTIALLY WORKING
- **v34 No Metrics Debug**: Output tail included in NO_METRICS error messages ✅ WORKING
- **v33 Transformers Fix**: Per-commit transformers version selection (aimv2 fix) ⚠️ NOT IN PARALLEL PATH

## Summary (Updated: 2026-01-03 22:10 UTC)

| Metric | Count |
|--------|-------|
| Completed | 14/59 |
| Success | 0 |
| Errors | 14 |
| Active Modal Tasks | 27 |

## Error Classification

| Category | Description | Count | Fixable? |
|----------|-------------|-------|----------|
| MODAL_INFRA | Modal infrastructure issues | 1 | YES - retry |
| EXPECTED | Intentionally blocked models | 1 | NO |
| BAD_CONFIG | Unresolved placeholders / missing args | 3 | NO - dataset issue |
| AIMV2 | transformers config conflict | **1** | ⚠️ FIX NOT IN PARALLEL PATH |
| SCRIPT_MISSING | Benchmark script not found | **1** | BUG - script path issue |
| SERVER_FAILED | vLLM server failed to start | **7** | Some have logs now! |

## v34 Fix Validation

### ✅ NO_METRICS output tail - WORKING
We can now see actual error messages in the output tail.

### ⚠️ SERVER_FAILED log capture - PARTIALLY WORKING

| Error # | Baseline Logs | Human Logs |
|---------|---------------|------------|
| 7 | "No server logs available" | ✅ Has config + traceback |
| 8 | ✅ Has torch bindings info | |
| 9 | ✅ Has config | |
| 10 | "No server logs available" | ✅ Has config |
| 11 | ✅ Has partial error | |
| 12 | "No server logs available" | ✅ Has config |
| 13 | "No server logs available" | ✅ Has config |
| 14 | ✅ Has config + error | |

**Observation**: Human server logs are captured but baseline often says "No server logs available". This suggests the log capture is working in the human code path but not baseline.

## Detailed Error Log (First 14)

| # | Commit | Model | GPU | Status | Error Category | Details |
|---|--------|-------|-----|--------|----------------|---------|
| 1 | 21d93c14 | Mixtral-8x7B-v0.1 | H100:8 | error | MODAL_INFRA | Docker fallback failed |
| 2 | 4fb56914 | DeepSeek-V3-0324 | H100:8 | blocked_model | EXPECTED | Model too large |
| 3 | 22dd9c27 | Llama-3.1-8B | H100:1 | baseline_failed | AIMV2 | aimv2 conflict |
| 4 | 526de822 | MODEL placeholder | H100:1 | baseline_failed | BAD_CONFIG | `--batch-size: 'BS'` |
| 5 | 2a052011 | Mixtral-8x7B-FP8 | H100:2 | baseline_failed | BAD_CONFIG | Missing --input-len |
| 6 | 2deb029d | Llama-3-8B-FP8 | H100:1 | baseline_failed | SCRIPT_MISSING | Script path wrong |
| 7 | 015069b0 | Qwen3-7B | H100:1 | baseline_failed | SERVER_FAILED | V1 multiprocessing error |
| 8 | 0d243f2a | Mixtral-8x7B | H100:2 | human_failed | SERVER_FAILED | ROCm-specific commit |
| 9 | 22d33bac | Llama-3.1-8B | H100:1 | baseline_failed | SERVER_FAILED | Server startup failed |
| 10 | 35fad35a | Llama-3.1-8B | H100:1 | baseline_failed | SERVER_FAILED | Server startup failed |
| 11 | 0ec82edd | Qwen3-30B-A3B | H100:2 | baseline_failed | SERVER_FAILED | Architecture error |
| 12 | 296f927f | Bamba-9B-v2 | H100:1 | baseline_failed | SERVER_FAILED | Mamba2 not supported |
| 13 | 3092375e | Llama-3.1-8B | H100:1 | baseline_failed | SERVER_FAILED | Server startup failed |
| 14 | 83450458 | Llama-3.1-8B | H100:1 | baseline_failed | BAD_CONFIG | ngram_prompt_lookup_max=None |

## Key Error Insights from v34 Logs

### Error #7 (015069b0) - Full traceback captured:
```
WARNING [api_server.py:171] V1 is enabled, but got --disable-frontend-multiprocessing.
To disable frontend multiprocessing, set VLLM_USE_V1=0.
```
**Root cause**: V1 engine incompatibility with frontend flags

### Error #14 (83450458):
```
ValueError: ngram_prompt_lookup_max=None must be > 0
```
**Root cause**: Speculative decoding config requires ngram_prompt_lookup_max but none provided

## Required Fixes for v35

### Fix 1: Add transformers version to parallel path (CRITICAL)
### Fix 2: Fix benchmark script path (/tmp → /opt/vllm-commit)
### Fix 3: Investigate why baseline logs sometimes missing
### Fix 4: Skip commits with unresolved placeholders

## Active Monitoring

**Progress**: 14/59 completed (~1 hour elapsed), 27 Modal tasks running

Last update: 2026-01-03 22:10 UTC

