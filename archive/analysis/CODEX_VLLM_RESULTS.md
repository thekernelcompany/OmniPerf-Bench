# Codex vLLM Bench Run Results

## Summary

**Status: ✅ COMPLETED**

- **Total commits processed**: 99/99 (100%)
- **Success rate**: 99/99 (100%)
- **Error rate**: 0/99 (0%)
- **Patches generated**: 99 patches found
- **Run ID**: `vllm_core-90a1c13f`

## Results Breakdown

### Latest Run: `vllm_core-90a1c13f`

All 99 commits were successfully processed by Codex CLI:

- ✅ **99 successful** optimizations
- ❌ **0 errors**
- 📝 **99 patches** generated (`model_patch.diff` files)

### Execution Details

- **Agent**: Codex CLI with `kernel-bot` profile
- **Processing mode**: Sequential (1 worker)
- **All items completed**: Yes
- **Average duration per commit**: ~2 minutes
- **Total execution time**: ~3.4 hours for all 99 commits
- **Time to first edit**: ~0.016 seconds average
- **Average patch size**: ~146 lines of code per commit

## Output Location

Results are stored in:
```
perf-agents-bench/state/runs/vllm_core-90a1c13f/
```

Each commit has its own directory:
- `vllm_core-0001/` through `vllm_core-0099/`
- Each contains:
  - `journal.json` - Execution metadata and status
  - `model_patch.diff` - Generated optimization patch
  - `task.txt` - Original task description
  - Other execution artifacts

## Viewing Results

### Check individual commit results:
```bash
cd perf-agents-bench
cat state/runs/vllm_core-90a1c13f/vllm_core-0001/journal.json | jq .
```

### View generated patch:
```bash
cat state/runs/vllm_core-90a1c13f/vllm_core-0001/model_patch.diff
```

### Count patches:
```bash
find state/runs/vllm_core-90a1c13f -name "model_patch.diff" | wc -l
```

### Generate summary report:
```bash
cd perf-agents-bench
source ../bench-env/bin/activate
python -m bench.cli report state/runs/vllm_core-90a1c13f
```

## Log Files

Main execution log:
- `perf-agents-bench/codex_vllm_run_20251120_053326.log` (580,295 lines)

The log shows:
- 99 "Task status determined as: success" entries
- 0 errors
- All commits processed successfully

## Next Steps

1. **Review patches**: Examine generated optimizations
2. **Generate report**: Use `bench.cli report` for detailed analysis
3. **Compare results**: Compare Codex optimizations against human commits
4. **Evaluate quality**: Assess patch quality and correctness

