# Soft Metrics Analysis Output

LLM-as-a-Judge evaluation results for agent benchmark runs.

## Directory Structure

```
state/analysis/
├── aggregate_report.json              # Cross-run statistics
├── vllm/
│   ├── claude_code/sonnet-4.5/        # 96 analyses
│   ├── codex/gpt-5/                   # 99 analyses
│   └── trae/
│       ├── gpt-5/                     # 70 analyses
│       └── sonnet-4.5/                # 91 analyses
└── sglang/
    └── claude_code/sonnet-4.5/        # 80 analyses
```

## Naming Convention

All items use standardized commit-based naming:

| Repository | Format | Example |
|------------|--------|---------|
| vLLM | `vllm_<commit>` | `vllm_015069b0` |
| SGLang | `sglang_<commit>` | `sglang_021f76e4` |

The 8-character commit hash uniquely identifies each optimization task.

## Per-Item Files

Each `{repo}_{commit}/` directory contains:

| File | Description |
|------|-------------|
| `metrics_summary.json` | Quick-reference scores (0-10) |
| `analysis.json` | Complete Pydantic-validated analysis |
| `patch_similarity.json` | Agent vs human patch comparison |
| `llm_prompt.txt` | Exact prompt sent to LLM judge |
| `llm_response.json` | Full LLM response |
| `llm_analysis.json` | Parsed structured scores |

## Quick Start

```bash
cd perf-agents-bench

# Analyze all runs (requires OPENROUTER_API_KEY)
python -m bench.cli analyze \
  --state-root ./state \
  --data-dir ../data \
  --repo vllm

# Analyze a single run
python -m bench.cli analyze \
  --run-dir ./state/runs/vllm/claude_code/sonnet-4.5/vllm_015069b0 \
  --data-dir ../data
```

## Data Directory

The `--data-dir` option points to the benchmark datasets (usually `../data` from perf-agents-bench). Required for patch comparison against human reference solutions.

## Key Output Files

### `patch_similarity.json`

Compares the agent's patch against the human developer's reference:

```json
{
  "human_patch_available": true,
  "human_patch_source": "vllm_final_dataset.jsonl:line71",
  "file_overlap_pct": 100.0,
  "line_overlap_pct": 56.6,
  "approach_similarity_score": 6.53
}
```

> **Note**: If `human_patch_available` is `false`, check that `--data-dir` points to the correct location containing `data/final/*.jsonl` datasets.

### `metrics_summary.json`

Quick-reference metrics for dashboards:

```json
{
  "scores": {
    "code_understanding": 8.5,
    "task_alignment": 9.0,
    "approach_quality": 7.5,
    "execution_quality": 8.0,
    "overall": 8.25
  },
  "patch_similarity": {
    "human_patch_available": true,
    "file_overlap_pct": 100.0
  }
}
```

## CLI Options Reference

| Option | Short | Description |
|--------|-------|-------------|
| `--state-root` | `-s` | Path to state directory containing `runs/` |
| `--output-dir` | `-o` | Output directory for analysis results |
| `--data-dir` | `-D` | Path to `data/` directory for benchmark datasets |
| `--run-dir` | `-d` | Analyze a single run directory |
| `--repo` | `-r` | Filter by repo (vllm, sglang) |
| `--agent` | `-a` | Filter by agent (trae, codex, openhands, claude_code) |
| `--model` | `-m` | Filter by model name |
| `--skip-llm` | | Skip LLM analysis (quantitative only) |
| `--dry-run` | | Discover runs without analyzing |
| `--max-concurrent` | | Maximum concurrent analyses (default: 3) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Required for LLM analysis (unless `--skip-llm`) |

## Troubleshooting

### `human_patch_available: false`

The analyzer couldn't find the human reference patch. Check:

1. **`--data-dir` path is correct**: Should point to the `data/` folder containing `final/vllm_final_dataset.jsonl`
2. **Dataset contains the commit**: The run's `commits.human` hash must exist in the dataset
3. **Dataset has `diff_text` field**: The JSONL entry must have the patch content

### No LLM scores generated

Ensure `OPENROUTER_API_KEY` is set or passed via `--api-key`. Use `--skip-llm` for quantitative-only analysis.
