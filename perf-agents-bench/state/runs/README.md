# Benchmark Runs Summary

## Directory Structure

```
runs/
├── vllm/
│   ├── trae/
│   │   ├── gpt-5/            (303 runs)
│   │   ├── gpt-4o/           (18 runs)
│   │   ├── o4-mini/          (1 run)
│   │   └── claude-sonnet-45/ (181 runs)
│   ├── openhands/
│   │   └── gpt-5/            (7 runs)
│   └── codex/
│       ├── gpt-5/            (103 runs)
│       └── gpt-4o/           (9 runs)
└── sglang/
    └── trae/
        ├── gpt-5/            (131 runs)
        └── claude-sonnet-45/ (80 runs)
```

## Model Breakdown by Repository

### vllm (622 total runs)

| Agent | Model | Runs |
|-------|-------|------|
| trae | gpt-5 | 303 |
| trae | gpt-4o | 18 |
| trae | o4-mini | 1 |
| trae | claude-sonnet-45 | 181 |
| openhands | gpt-5 | 7 |
| codex | gpt-5 | 103 |
| codex | gpt-4o | 9 |

### sglang (211 total runs)

| Agent | Model | Runs |
|-------|-------|------|
| trae | gpt-5 | 131 |
| trae | claude-sonnet-45 | 80 |

## Model Details

| Model ID | Provider | Full Model Name |
|----------|----------|-----------------|
| gpt-5 | openai | gpt-5-2025-08-07 |
| gpt-4o | openai | gpt-4o |
| o4-mini | openai | o4-mini |
| claude-sonnet-45 | bedrock | us.anthropic.claude-sonnet-4-5-20250929-v1:0 |

## Notes

- Model information extracted from `trajectory.json` (field: `model`) or `prediction.jsonl` (field: `model_name_or_path`)
- Incomplete runs (missing journal.json or model_patch.diff) are stored in `../incomplete_runs/`
- Directory structure: `{repo}/{agent}/{model}/{timestamp}/{task_id}/`
- Total runs: 833 (622 vllm + 211 sglang)
