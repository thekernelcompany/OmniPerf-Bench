# Agent Benchmark Runs

Raw agent run data from benchmark evaluations.

## Directory Structure

```
runs/
└── vllm/
    ├── claude_code/sonnet-4.5/    # 96 runs
    ├── codex/gpt-5/               # 99 runs
    ├── trae/
    │   ├── gpt-5/                 # 70 runs
    │   └── sonnet-4.5/            # 91 runs
    └── _misc/                     # Legacy/experimental
```

## Naming Convention

All items use standardized commit-based naming:

| Repository | Format | Example |
|------------|--------|---------|
| vLLM | `vllm_<commit>` | `vllm_015069b0` |
| SGLang | `sglang_<commit>` | `sglang_021f76e4` |

The 8-character commit hash uniquely identifies each optimization task.

## Benchmark Run Dates

| Agent | Model | Runs | Run Date |
|-------|-------|------|----------|
| Claude Code | Sonnet 4.5 | 96 | 2025-12-22 |
| Codex | GPT-5 | 99 | 2025-11-20 |
| TRAE | GPT-5 | 70 | 2025-12-26 |
| TRAE | Sonnet 4.5 | 91 | 2025-12-23 |

## Per-Run Files

| File | Description |
|------|-------------|
| `run_summary.json` | Execution metadata, status |
| `trajectory.json` | Full agent interaction history |
| `model_patch.diff` | Generated code changes |
| `journal.json` | Task config, commit hashes |
| `task.txt` | Original task prompt |

## See Also

- [SOFT_METRICS_ANALYSIS.md](../../SOFT_METRICS_ANALYSIS.md) - Benchmark results
- `../analysis/` - Soft metrics analysis outputs
