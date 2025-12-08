# Codex SGLang Bench Run Results

## Summary

**Status: ✅ COMPLETED**

- **Total commits processed**: 80/80 (100%)
- **Success rate**: 80/80 (100%)
- **Error rate**: 0/80 (0%)
- **Patches generated**: 80 patches found
- **Run ID**: `sglang_core-389be848`

## Results Breakdown

### Latest Run: `sglang_core-389be848`

All 80 commits were successfully processed by Codex CLI:

- ✅ **80 successful** optimizations
- ❌ **0 errors**
- 📝 **80 patches** generated (`model_patch.diff` files)

### Execution Details

- **Agent**: Codex CLI with `kernel-bot` profile
- **Processing mode**: Sequential (1 worker)
- **All items completed**: Yes
- **Average duration per commit**: ~3.1 minutes (185.7 seconds)
- **Total execution time**: ~4.1 hours (247.6 minutes)
- **Time to first edit**: Average 167.4s (median 159.5s)
- **Average patch size**: 94 LOC per commit
- **Total lines of code**: 7,560 LOC across all patches

## Output Location

Results are stored in:
```
perf-agents-bench/state/runs/sglang_core-389be848/
```

Each commit has its own directory:
- `sglang_core-0001/` through `sglang_core-0080/`
- Each contains:
  - `journal.json` - Execution metadata and status
  - `model_patch.diff` - Generated optimization patch
  - `task.txt` - Original task description
  - Other execution artifacts

## Execution Timeline

- **Run started**: 2025-11-21 03:21:16 (tmux session created)
- **Run completed**: 2025-11-21 07:29:05 (last journal written)
- **Total duration**: ~4.1 hours
- **Last commit processed**: `sglang_core-0080`

## Viewing Results

### Check individual commit results:
```bash
cd perf-agents-bench
cat state/runs/sglang_core-389be848/sglang_core-0001/journal.json | jq .
```

### View generated patch:
```bash
cat state/runs/sglang_core-389be848/sglang_core-0001/model_patch.diff
```

### Count patches:
```bash
find state/runs/sglang_core-389be848 -name "model_patch.diff" | wc -l
```

### Generate summary report:
```bash
cd perf-agents-bench
source ../bench-env/bin/activate
python -m bench.cli report state/runs/sglang_core-389be848
```

## Log Files

Main execution log:
- `perf-agents-bench/codex_sglang_run_20251121_032116.log`

The log shows:
- 80 "Task status determined as: success" entries
- 0 errors
- All commits processed successfully
- Final message: "✓ Prepare completed: state/runs/sglang_core-389be848"

## Comparison with vLLM Results

| Metric | vLLM | SGLang |
|--------|------|--------|
| Commits | 99 | 80 |
| Success rate | 100% | 100% |
| Avg duration | 3.1 min | 3.1 min |
| Total time | 5.2 hours | 4.1 hours |
| Avg patch size | 99 LOC | 94 LOC |
| Total LOC | 9,763 | 7,560 |

## Next Steps

1. **Review patches**: Examine generated optimizations
2. **Generate report**: Use `bench.cli report` for detailed analysis
3. **Compare results**: Compare Codex optimizations against human commits
4. **Evaluate quality**: Assess patch quality and correctness

