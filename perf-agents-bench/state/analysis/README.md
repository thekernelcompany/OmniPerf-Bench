# Soft Metrics Analysis Output

This directory contains the output of the **Soft Metrics Analyzer** - a system for extracting qualitative and quantitative metrics from agent benchmark runs using LLM-as-a-Judge.

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
  --run-dir ./state/runs/vllm/claude_code/default/2025-12-16_12-59-57/vllm_core-0000 \
  --data-dir ../data

# Skip LLM analysis (quantitative metrics only)
python -m bench.cli analyze --state-root ./state --data-dir ../data --skip-llm
```

## Important: `--data-dir` Option

The `--data-dir` option specifies the path to the benchmark datasets directory (usually `data/` at the repo root). This is **required** for accurate patch comparison against human reference patches.

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` / `-D` | Auto-discovery | Path to `data/` directory containing `final/vllm_final_dataset.jsonl` etc. |

**If not specified**, the analyzer will try to auto-discover the `data/` folder by traversing parent directories. For reliable results, always specify `--data-dir ../data` when running from `perf-agents-bench/`.

## Directory Structure

```
state/analysis/
├── aggregate_report.json                    # Cross-run statistics
└── {repo}/
    └── {agent}/
        └── {model}/
            └── {timestamp}/
                └── {item_id}/
                    ├── analysis.json           # Complete Pydantic-validated analysis
                    ├── metrics_summary.json    # Quick-reference compact summary
                    ├── patch_similarity.json   # Agent vs human patch comparison
                    ├── trajectory_metrics.json # Per-step token/timing metrics
                    ├── llm_prompt.txt          # Exact prompt sent to LLM
                    ├── llm_response.json       # Full LLM response
                    └── llm_analysis.json       # Parsed structured JSON from LLM
```

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
