# Soft Metrics Analysis - Agent Benchmark Results

> **LLM-as-a-Judge evaluation of AI agent performance on vLLM optimization tasks**

This document provides comprehensive analysis results for multiple AI agents benchmarked on performance optimization tasks in the vLLM codebase.

---

## Executive Summary

| Agent | Model | Runs | Scored | Avg Score | Success Rate |
|-------|-------|------|--------|-----------|--------------|
| **Claude Code** | default (Sonnet 4.5) | 192 | 189 | 7.64 | 98.4% |
| **Codex** | GPT-5 | 99 | 98 | 7.05 | 99.0% |
| **TRAE** | GPT-5 (merged) | 70 | 49 | 7.57 | 70.0% |
| **TRAE** | Claude Sonnet 4.5 | 91 | 46 | 8.12 | 50.5% |

**Scoring Scale:** 0-10 (LLM-as-a-Judge using Gemini 3 Flash)

---

## Table of Contents

1. [Methodology](#methodology)
2. [Detailed Results](#detailed-results)
   - [Claude Code](#claude-code-default)
   - [Codex GPT-5](#codex-gpt-5)
   - [TRAE GPT-5](#trae-gpt-5)
   - [TRAE Sonnet 4.5](#trae-sonnet-45)
3. [Data Structure](#data-structure)
4. [Reproducing the Analysis](#reproducing-the-analysis)
5. [Known Issues](#known-issues)
6. [TRAE GPT-5 Dataset Merge](#trae-gpt-5-dataset-merge)

---

## Methodology

### LLM-as-a-Judge Soft Metrics

Each agent run is evaluated by an LLM (Gemini 3 Flash via OpenRouter) on four dimensions:

| Metric | Description |
|--------|-------------|
| **Code Understanding** | How well the agent understood the codebase and optimization target |
| **Task Alignment** | Whether the agent addressed the actual performance issue |
| **Approach Quality** | Quality of the optimization strategy chosen |
| **Execution Quality** | How well the agent executed the optimization (tool use, iteration) |

**Overall Score** = weighted average of the four metrics (0-10 scale)

### What Gets Analyzed

For each run, the analyzer examines:
- `trajectory.json` - Full agent interaction history
- `model_patch.diff` - Generated code changes
- `run_summary.json` - Execution metadata
- Human patch (from dataset) - For similarity comparison

### Patch Similarity Metrics

| Metric | Description |
|--------|-------------|
| `file_overlap_pct` | % of files in agent patch that match human patch |
| `line_overlap_pct` | % of lines that are similar to human solution |
| `approach_similarity` | Semantic similarity of optimization approach (0-10) |

---

## Detailed Results

### Claude Code (default)

**Model:** Claude Sonnet 4.5 (Anthropic)
**Total Runs:** 192 | **Scored:** 189 | **Zero Scores:** 3

| Metric | Value |
|--------|-------|
| Average Score (all) | 7.52 |
| Average Score (scored only) | 7.64 |
| Success Rate | 98.4% |

**Score Distribution:**
```
9.0+  ████████████ 24
8.0-9 ████████████████████████████████████████ 78
7.0-8 ████████████████████████████████ 62
6.0-7 ████████████ 18
5.0-6 ████ 5
<5.0  ██ 2
0.0   █ 3
```

**Strengths:**
- Highest success rate among all agents
- Consistent quality across runs
- Excellent tool usage patterns

**Weaknesses:**
- 3 runs with infrastructure failures (zero scores)

---

### Codex GPT-5

**Model:** GPT-5-2025-08-07 (OpenAI)
**Total Runs:** 99 | **Scored:** 98 | **Zero Scores:** 1

| Metric | Value |
|--------|-------|
| Average Score (all) | 6.98 |
| Average Score (scored only) | 7.05 |
| Success Rate | 99.0% |

**Score Distribution:**
```
9.0+  ████████ 15
8.0-9 ████████████████████████ 45
7.0-8 ████████████████ 28
6.0-7 ████ 6
5.0-6 ██ 3
<5.0  █ 1
0.0   ~ 1
```

**Strengths:**
- Near-perfect completion rate
- Strong systematic approach to optimization
- Good code understanding

**Weaknesses:**
- Slightly lower average than Claude Code
- Occasionally over-engineers solutions

---

### TRAE GPT-5 (merged)

**Model:** GPT-5-2025-08-07 (via TRAE agent framework)
**Total Runs:** 70 | **Scored:** 49 | **Zero Scores:** 21

| Metric | Value |
|--------|-------|
| Average Score (all) | 5.30 |
| Average Score (scored only) | 7.57 |
| Success Rate | 70.0% |

**Score Distribution:**
```
9.0+  ████ 5
8.0-9 ████████████████████████████ 31
7.0-8 ████████ 10
6.0-7 ██ 2
5.0-6 ~ 1
<5.0  ~ 0
0.0   ████████████████████ 21
```

**Zero Score Breakdown (21 runs):**
| Reason | Count | Description |
|--------|-------|-------------|
| API 429 Quota | 18 | GPT-5 rate limits during December benchmark |
| Empty Patch | 2 | Agent claimed success but patch was empty |
| API Error | 1 | Infrastructure failure |

**Strengths:**
- When successful, quality matches other agents (7.57 avg)
- Good optimization strategies

**Weaknesses:**
- 30% failure rate due to infrastructure issues
- API quota management problems

---

### TRAE Sonnet 4.5

**Model:** Claude Sonnet 4.5 (via AWS Bedrock)
**Total Runs:** 91 | **Scored:** 46 | **Zero Scores:** 45

| Metric | Value |
|--------|-------|
| Average Score (all) | 4.10 |
| Average Score (scored only) | 8.12 |
| Success Rate | 50.5% |

**Score Distribution:**
```
9.0+  ████████ 12
8.0-9 ████████████████████████ 28
7.0-8 ████ 5
6.0-7 ~ 1
5.0-6 ~ 0
<5.0  ~ 0
0.0   █████████████████████████████████████████████ 45
```

**Zero Score Breakdown (45 runs):**
| Reason | Count | Description |
|--------|-------|-------------|
| AWS Token Expiry | ~40 | SSO token expired mid-run |
| No Optimization Found | ~5 | Agent couldn't identify improvements |

**Strengths:**
- Highest quality when successful (8.12 avg)
- Deep code understanding

**Weaknesses:**
- 50% failure rate due to AWS credential issues
- Long runtime per task

---

## Data Structure

### Directory Layout

```
perf-agents-bench/state/
├── runs/                              # Raw agent run data
│   └── vllm/{agent}/{model}/          # Per-run directories
│
└── analysis/                          # Soft metrics results
    ├── vllm/
    │   ├── claude_code/sonnet-4.5/    # 96 analyses
    │   ├── codex/gpt-5/               # 99 analyses
    │   └── trae/
    │       ├── gpt-5/                 # 70 analyses
    │       └── sonnet-4.5/            # 91 analyses
    └── sglang/
        └── claude_code/sonnet-4.5/    # 80 analyses
```

### Benchmark Run Dates

| Agent | Model | Run Date | Items |
|-------|-------|----------|-------|
| Claude Code | Sonnet 4.5 | 2025-12-22 | 96 |
| Codex | GPT-5 | 2025-11-20 | 99 |
| TRAE | GPT-5 | 2025-12-26 | 70 |
| TRAE | Sonnet 4.5 | 2025-12-23 | 91 |
| Claude Code (SGLang) | Sonnet 4.5 | 2025-12-23 | 80 |

### Per-Run Files

Each run directory contains:

| File | Description |
|------|-------------|
| `run_summary.json` | Execution metadata, status, patch stats |
| `trajectory.json` | Full agent interaction history |
| `model_patch.diff` | Generated code changes |
| `journal.json` | Task configuration, commit hashes |
| `task.txt` | Original task prompt |
| `prediction.jsonl` | Model predictions (if applicable) |

### Analysis Output Schema

`metrics_summary.json`:
```json
{
  "run": {
    "repo": "vllm",
    "agent": "trae",
    "model": "gpt-5",
    "item_id": "vllm_core-0039",
    "task_id": "vllm_core"
  },
  "execution": {
    "status": "success|error|max_steps_exceeded",
    "duration_s": 1791.6,
    "steps": 41,
    "tool_calls": 41,
    "errors": 0
  },
  "tool_breakdown": {
    "bash": 10,
    "editor": 25,
    "read": 0,
    "search": 6,
    "other": 0
  },
  "patch": {
    "generated": true,
    "lines_added": 28,
    "lines_removed": 9,
    "files_changed": 1,
    "compliance_ok": true
  },
  "scores": {
    "code_understanding": 8.7,
    "task_alignment": 9.0,
    "approach_quality": 8.3,
    "execution_quality": 8.5,
    "overall": 8.62
  },
  "classification": {
    "approach": "systematic|minimal|exploratory",
    "tool_pattern": "balanced|editor_focused|bash_heavy",
    "failure_reason": "none|api_error|timeout|no_patch"
  },
  "insights": {
    "summary": "Human-readable analysis summary",
    "strengths": ["List of strengths"],
    "weaknesses": ["List of weaknesses"]
  },
  "patch_similarity": {
    "human_patch_available": true,
    "file_overlap_pct": 100.0,
    "line_overlap_pct": 10.2,
    "approach_similarity": 4.7
  }
}
```

---

## Reproducing the Analysis

### Prerequisites

```bash
# Clone the repository
git clone --recursive <repo-url>
cd OmniPerf-Bench

# Set up virtual environment
cd perf-agents-bench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API key (for LLM analysis)
export OPENROUTER_API_KEY="sk-or-v1-..."
# Or add to .env file in repo root
```

### Running Analysis

```bash
cd perf-agents-bench
source .venv/bin/activate

# Analyze all runs for an agent
python -m bench.cli analyze \
    --state-root ./state \
    --data-dir ../data \
    --repo vllm \
    --agent trae \
    --model gpt-5 \
    --max-concurrent 5

# Analyze a single run
python -m bench.cli analyze \
    --run-dir state/runs/vllm/trae/gpt-5/2025-12-26_15-06-42/vllm_gpt5_rerun_0d243f2a \
    --data-dir ../data
```

### Viewing Results

```bash
# Check aggregate report
cat state/analysis/aggregate_report.json | jq '.summary'

# View individual analysis
cat state/analysis/vllm/trae/gpt-5/2025-12-26_15-06-42/vllm_gpt5_rerun_0d243f2a/metrics_summary.json | jq '.scores'

# List all scores
find state/analysis -name "metrics_summary.json" -exec sh -c 'echo $(jq -r ".scores.overall" "$1") $(dirname "$1" | xargs basename)' _ {} \; | sort -rn | head -20
```

---

## Known Issues

### 1. API 429 Quota Errors (TRAE GPT-5)

**Symptom:** 18 runs with `status: error`, `steps: 1`, `tool_calls: 0`

**Cause:** OpenAI GPT-5 API rate limits during December 2025 benchmark run

**Impact:** Runs crashed immediately, no work done

**Identification:**
```bash
# Check trajectory for 429 error
jq '.agent_steps[0].error' trajectory.json
# Returns: "Error code: 429 - {'error': {'message': 'You exceeded your current quota...'"
```

### 2. Empty Patches Despite "Success" Status

**Symptom:** `status: success` but `patch.lines_added: 0`

**Cause:** Agent made tool calls but changes weren't captured in git diff

**Affected Runs:** 2 in TRAE GPT-5 (58eee5f2, 61b8cea3)

**Identification:**
```bash
# Trajectory shows tool calls, but patch is empty
jq '.agent_steps | length' trajectory.json  # Shows 34+ steps
cat model_patch.diff  # Shows empty or just file creation
```

### 3. LLM Analysis Failures

**Symptom:** `scores.overall: 0.0` but `patch.generated: true` with actual lines

**Cause:** Gemini 3 Flash returned empty response during analysis

**Fix:** Re-run analysis for affected items:
```bash
python -m bench.cli analyze --run-dir <path_to_run> --data-dir ../data
```

### 4. max_steps_exceeded Status

**Symptom:** Analyzer crashes with `'>' not supported between instances of 'NoneType' and 'int'`

**Cause:** Bug in analyzer when handling `status: max_steps_exceeded`

**Workaround:** Manual scoring based on patch review

---

## TRAE GPT-5 Dataset Merge

### Background

Two TRAE GPT-5 datasets existed with different quality:

| Source | Date | Runs | Issue |
|--------|------|------|-------|
| Local (Git) | November 2025 | 60 | "No tool output" bug in 50/60 |
| HuggingFace | December 2025 | 53 | Clean, post-bug-fix |

### The Bug

November runs had a TRAE agent bug where tool outputs weren't properly captured:
- 50/60 trajectories show "No tool output found"
- Inflated success rate (false positives)
- Unreliable patch quality

### Merge Strategy

1. **HuggingFace runs (53)**: Always use these (reliable)
2. **Local-only clean runs (17)**: Include (verified no bug pattern)
3. **Local-only buggy runs (26)**: Exclude

**Final merged dataset:** 70 runs in `gpt-5/`

### Downloading HuggingFace Data

```python
from huggingface_hub import hf_hub_download

dataset_id = "Inferencebench/trae-gpt5-trajectories"
for commit in commits:
    for file in ["trajectory.json", "run_summary.json", "model_patch.diff"]:
        hf_hub_download(dataset_id, f"vllm/{commit}/{file}", repo_type="dataset")
```

### Clean Local-Only Commits (17)

```
015069b0, 2a052011, 2deb029d, 2f192835, 3476ed08, 526de822, 660470e5,
6d0734c5, 6dd94dbe, 7661e92e, 7c01f706, 80aa7e91, 83450458, 88693683,
89a84b0b, e7b20426, ec3b5ce9
```

---

## Appendix: Score Interpretation Guide

| Score Range | Interpretation |
|-------------|----------------|
| **9.0-10.0** | Exceptional - matches or exceeds human solution |
| **8.0-8.9** | Excellent - high-quality optimization, minor gaps |
| **7.0-7.9** | Good - solid optimization with some weaknesses |
| **6.0-6.9** | Adequate - functional but suboptimal approach |
| **5.0-5.9** | Marginal - minimal optimization, significant gaps |
| **< 5.0** | Poor - incorrect approach or incomplete execution |
| **0.0** | Failed - no patch generated or analysis failed |

---

## References

- [OmniPerf-Bench Repository](https://github.com/...)
- [HuggingFace Dataset: trae-gpt5-trajectories](https://huggingface.co/datasets/Inferencebench/trae-gpt5-trajectories)
- [vLLM Repository](https://github.com/vllm-project/vllm)

---

*Last updated: January 2026*
*Analysis performed with: Gemini 3 Flash (google/gemini-3-flash-preview) via OpenRouter*
