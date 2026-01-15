# Evaluation Results v2

## Directory Structure

```
eval_results_v2/
├── evaluation_report.json    # Aggregated evaluation metrics
├── vllm/
│   ├── codex/
│   │   └── gpt-5/           (1 run)
│   └── trae/
│       ├── claude-sonnet-45/ (2 runs)
│       └── gpt-5/           (64 runs)
└── sglang/
    ├── codex/
    │   └── gpt-5/           (1 run)
    └── trae/
        ├── claude-sonnet-45/ (1 run)
        └── gpt-5/           (2 runs)
```

## Summary

| Repo | Agent | Model | Runs |
|------|-------|-------|------|
| vllm | trae | gpt-5 | 64 |
| vllm | trae | claude-sonnet-45 | 2 |
| vllm | codex | gpt-5 | 1 |
| sglang | trae | gpt-5 | 2 |
| sglang | trae | claude-sonnet-45 | 1 |
| sglang | codex | gpt-5 | 1 |

**Total: 71 evaluation runs**

## Structure per Run

Each run folder (e.g., `vllm/trae/gpt-5/04ff5d51/`) contains:
- `{task_id}/test_results.json` - Evaluation results with speedup metrics
- `{task_id}/test_stdout.txt` - Test execution stdout
- `{task_id}/test_stderr.txt` - Test execution stderr
