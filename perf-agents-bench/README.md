# perf-agents-bench

Agent benchmarking harness for evaluating AI coding agents on performance optimization tasks.

## Quick Links

| Document | Description |
|----------|-------------|
| [SOFT_METRICS_ANALYSIS.md](SOFT_METRICS_ANALYSIS.md) | **Full benchmark results and methodology** |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and design |
| [SUCCESS_FAILURE_ANALYSIS.md](SUCCESS_FAILURE_ANALYSIS.md) | Detailed failure analysis |

## Benchmark Results Summary

| Agent | Model | Runs | Avg Score | Success |
|-------|-------|------|-----------|---------|
| Claude Code | Sonnet 4.5 | 192 | 7.64 | 98% |
| Codex | GPT-5 | 99 | 7.05 | 99% |
| TRAE | GPT-5 | 70 | 7.57 | 70% |
| TRAE | Sonnet 4.5 | 91 | 8.12 | 51% |

*See [SOFT_METRICS_ANALYSIS.md](SOFT_METRICS_ANALYSIS.md) for full details.*

---

## Setup

### Requirements
- Python 3.12+
- Docker (for evaluation)
- OpenRouter API key (for soft metrics analysis)

### Installation

```bash
cd perf-agents-bench

# Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API key for analysis
export OPENROUTER_API_KEY="sk-or-v1-..."
```

---

## Directory Structure

```
perf-agents-bench/
├── bench/                    # CLI and analysis code
│   ├── cli.py               # Main entry point
│   ├── analysis/            # Soft metrics analyzer
│   └── pipeline.py          # Agent execution pipeline
│
├── state/
│   ├── runs/                # Raw agent run data
│   │   └── vllm/{agent}/{model}/{timestamp}/{item}/
│   │       ├── trajectory.json
│   │       ├── model_patch.diff
│   │       └── run_summary.json
│   │
│   └── analysis/            # Soft metrics results
│       └── vllm/{agent}/{model}/{timestamp}/{item}/
│           └── metrics_summary.json
│
├── tasks/                   # Task configurations
│   └── vllm.yaml
│
└── *.md                     # Documentation
```

---

## CLI Commands

### Run Soft Metrics Analysis

```bash
# Analyze all runs for an agent/model
python -m bench.cli analyze \
    --state-root ./state \
    --data-dir ../data \
    --repo vllm \
    --agent trae \
    --model gpt-5

# Analyze single run
python -m bench.cli analyze \
    --run-dir state/runs/vllm/trae/gpt-5/2025-12-26_15-06-42/vllm_gpt5_rerun_0d243f2a \
    --data-dir ../data
```

### Generate Reports

```bash
# View aggregate report
cat state/analysis/aggregate_report.json | jq '.summary'

# Generate run report
python -m bench.cli report state/runs/<run_id>
```

### Plan and Execute (OpenHands)

```bash
# 1. Plan commits
python -m bench.cli plan tasks/vllm.yaml \
    --commits .work/vllm_commits.txt \
    --out ./state/plan.json

# 2. Execute
python -m bench.cli prepare tasks/vllm.yaml \
    --from-plan ./state/plan.json \
    --bench-cfg bench.yaml \
    --max-workers 1
```

---

## Data Sources

### Canonical Datasets

| Dataset | Location | Runs |
|---------|----------|------|
| Claude Code | `state/runs/vllm/claude_code/sonnet-4.5/` | 192 |
| Codex GPT-5 | `state/runs/vllm/codex/gpt-5/` | 99 |
| TRAE GPT-5 | `state/runs/vllm/trae/gpt-5/` | 70 |
| TRAE Sonnet 4.5 | `state/runs/vllm/trae/sonnet-4.5/` | 91 |

**Note:** Legacy/experimental runs are in `_misc/`. See [SOFT_METRICS_ANALYSIS.md](SOFT_METRICS_ANALYSIS.md#trae-gpt-5-dataset-merge) for details.

### HuggingFace

```python
from datasets import load_dataset

# TRAE GPT-5 trajectories
ds = load_dataset("Inferencebench/trae-gpt5-trajectories")
```

---

## Troubleshooting

### API Key Issues
```bash
# Check key is set
echo $OPENROUTER_API_KEY | head -c 20

# Key should NOT have quotes
export OPENROUTER_API_KEY=sk-or-v1-abc123...  # correct
export OPENROUTER_API_KEY="sk-or-v1-abc123..."  # wrong
```

### Analysis Failures
```bash
# Re-run analysis for specific item
python -m bench.cli analyze --run-dir <path> --data-dir ../data

# Check for zero scores with patches (analysis bug)
find state/analysis -name "metrics_summary.json" -exec sh -c \
  'score=$(jq -r ".scores.overall" "$1"); lines=$(jq -r ".patch.lines_added" "$1"); \
   if [ "$score" = "0" ] && [ "$lines" != "0" ]; then echo "$1"; fi' _ {} \;
```

---

## License

See repository root for license information.
