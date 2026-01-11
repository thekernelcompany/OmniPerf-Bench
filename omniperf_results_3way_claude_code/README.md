# Claude Code vLLM Benchmark Results

Benchmark results comparing **Claude Code** (AI agent) vs **Human developers** on vLLM performance optimization tasks.

## Quick Start

```bash
# View results summary
python upload_to_hf.py --dry-run

# Upload to HuggingFace
python upload_to_hf.py --repo-id "Inferencebench/claude-code-vllm-benchmarks"
```

## Results Summary

| Metric | Value |
|--------|-------|
| Total commits | 94 |
| Evaluable for Agent vs Human | 49 |
| Agent matches or beats human | 65% |
| Agent failure rate | 22% |

See `COMPREHENSIVE_BENCHMARK_ANALYSIS.md` for detailed analysis.

---

## Directory Structure

```
omniperf_results_3way_claude_code/
│
├── README.md                           # This file
├── upload_to_hf.py                     # HuggingFace upload script
├── COMPREHENSIVE_BENCHMARK_ANALYSIS.md # Main analysis document (START HERE)
├── METRIC_ANALYSIS.md                  # Schema documentation
│
├── results/                            # All benchmark result data
│   ├── modal/                          # Primary: Modal H100 pipeline
│   │   └── <commit>/                   # Per-commit folders
│   │       └── benchmark_result.json   # 3-way comparison result
│   ├── separate_baseline/              # Separate pipeline: baseline runs
│   ├── separate_agent/                 # Separate pipeline: human + agent runs
│   └── docker/                         # Docker verification runs
│
├── exports/                            # Exported datasets
│   ├── schema_v2_results.json          # Full results as JSON (v2 schema)
│   ├── claude_code_vllm_benchmarks.jsonl
│   ├── full_results.jsonl
│   ├── merged_benchmarks.jsonl
│   └── data/                           # Parquet exports
│
├── logs/                               # Execution logs
│   ├── modal/                          # Modal run progress logs
│   └── reruns/                         # Pipeline rerun logs (v1-v18)
│
└── archive/                            # Historical files (superseded)
    ├── analysis/                       # Old analysis documents
    └── operational/                    # Dated operational files
```

---

## Key Files

| File | Purpose |
|------|---------|
| `COMPREHENSIVE_BENCHMARK_ANALYSIS.md` | **Start here.** Complete analysis of all 96 commits with methodology, results, and limitations. |
| `METRIC_ANALYSIS.md` | HuggingFace schema documentation. Explains serving vs standalone metrics. |
| `upload_to_hf.py` | Script to upload results to HuggingFace. Supports dry-run mode. |

---

## Benchmark Design

Each commit is benchmarked in a **3-way comparison**:

| Version | Description |
|---------|-------------|
| **Baseline** | Performance before optimization (parent commit) |
| **Human** | Performance with human's optimization (ground truth) |
| **Agent** | Performance with Claude Code's optimization attempt |

### Benchmark Modes

| Mode | Metrics | Unit | Better |
|------|---------|------|--------|
| `serving` | TTFT, TPOT, ITL | milliseconds | Lower |
| `standalone` | latency_avg, throughput | ms / tok/s | Lower / Higher |

A commit has **one mode**, never both.

---

## Result Format

Each `results/modal/<commit>/benchmark_result.json` contains:

```json
{
  "instance": {
    "commit_hash": "abc123...",
    "commit_subject": "[Perf] Optimization description",
    "perf_command": "vllm bench serve ..."
  },
  "result": {
    "status": "success",
    "benchmark_mode": "serving",
    "baseline_metrics": {"ttft_mean": 100.0, "tpot_mean": 10.0},
    "human_metrics": {"ttft_mean": 90.0, "tpot_mean": 9.0},
    "agent_metrics": {"ttft_mean": 92.0, "tpot_mean": 9.2},
    "human_improvement": {"ttft_mean": 10.0},
    "agent_improvement": {"ttft_mean": 8.0},
    "agent_vs_human": {"ttft_mean": -2.0}
  }
}
```

### Improvement Calculation

```
improvement = (baseline - optimized) / baseline * 100
```

Positive = improvement (lower latency or higher throughput).

---

## Data Sources

### Modal Pipeline (Primary)

- Location: `results/modal/<commit>/benchmark_result.json`
- Infrastructure: Modal H100 GPUs
- Reliability: **Gold standard** - same config for baseline/human/agent
- Status: 94 commits processed

### Separate Pipeline (Secondary)

- Location: `results/separate_baseline/`, `results/separate_agent/`
- Infrastructure: Local Docker on H100
- Reliability: **Variable** - config may differ between runs
- Issue: Some commits have model mismatch (Qwen baseline vs Llama human)

---

## HuggingFace Dataset

Published at: **[Inferencebench/claude-code-vllm-benchmarks](https://huggingface.co/datasets/Inferencebench/claude-code-vllm-benchmarks)**

### Schema v3 (Current)

75 columns covering:
- Commit metadata (hash, subject, PR URL)
- Raw metrics (baseline/human/agent)
- Improvement calculations
- Agent metadata (name, model, date)

See `METRIC_ANALYSIS.md` for full schema documentation.

---

## Reproducing Results

### Prerequisites

1. Modal account with H100 access
2. HuggingFace token for model downloads
3. Python environment with dependencies

### Running Benchmarks

```bash
# From OmniPerf-Bench root
source bench-env/bin/activate

# Deploy Modal infrastructure
modal deploy src/eval/modal_benchmark.py

# Run benchmarks (see main repo README)
```

---

## Limitations

1. **Sample size**: 19 gold-standard 3-way comparisons, 49 total evaluable
2. **Infrastructure failures**: 41% of commits failed due to vLLM version bugs, model issues, or timeouts
3. **Model mismatch**: 8 commits from separate pipeline used different baseline models
4. **Agent failures**: 22% of evaluable commits had agent patches that crashed or produced no output

See `COMPREHENSIVE_BENCHMARK_ANALYSIS.md` Section "Data Quality Notes" for details.
