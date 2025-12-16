# Soft Metrics Analysis Output

This directory contains the output of the **V3 Academic-Grade Soft Metrics Analyzer** - a system for extracting qualitative and quantitative metrics from agent benchmark runs.

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
                    ├── llm_prompt.txt          # Exact prompt sent to Gemini
                    ├── llm_response.json       # Full LLM response with thinking
                    ├── llm_analysis.json       # Raw structured JSON from LLM
                    ├── llm_raw_scores.json     # Extracted scores only
                    ├── metrics_summary.json    # Quick-reference compact summary
                    ├── trajectory_metrics.json # Per-step token/timing metrics
                    └── patch_similarity.json   # Agent vs human patch comparison
```

### Path Components

| Component | Description | Examples |
|-----------|-------------|----------|
| `{repo}` | Target repository | `vllm`, `sglang` |
| `{agent}` | Agent framework | `trae`, `codex`, `openhands`, `claude_code` |
| `{model}` | LLM model used | `claude-sonnet-45`, `gpt-5`, `o4-mini` |
| `{timestamp}` | Run timestamp | `2025-11-27_15-42-14` |
| `{item_id}` | Unique run identifier | `vllm_core-0001`, `moe_align_opt-0000` |

---

## Output Files

### 1. `analysis.json` - Complete Analysis

The full Pydantic-validated analysis containing all metrics, scores, and metadata.

```json
{
  "schema_version": "3.0",
  "meta": {
    "repo": "vllm",
    "agent": "trae",
    "model": "claude-sonnet-45",
    "item_id": "vllm_bedrock_sonnet45-0001",
    "task_id": "vllm_core",
    "commits": {"pre": "abc123", "human": "def456"}
  },
  "quantitative": { ... },
  "qualitative": { ... },
  "categorical": { ... },
  "analysis": { ... },
  "trajectory": { ... },
  "patch_similarity": { ... },
  "llm_raw": { ... }
}
```

### 2. `metrics_summary.json` - Quick Reference

Compact summary for quick lookups and aggregation. Best file for building dashboards.

```json
{
  "run": {
    "repo": "vllm",
    "agent": "trae",
    "model": "claude-sonnet-45",
    "item_id": "vllm_bedrock_sonnet45-0001"
  },
  "execution": {
    "status": "success",
    "duration_s": 567.1,
    "steps": 46,
    "tool_calls": 32,
    "errors": 3
  },
  "scores": {
    "code_understanding": 9.0,
    "task_alignment": 9.0,
    "approach_quality": 10.0,
    "execution_quality": 9.0,
    "overall": 9.25
  },
  "classification": {
    "approach": "systematic",
    "tool_pattern": "balanced",
    "failure_reason": "none"
  },
  "insights": {
    "summary": "...",
    "strengths": ["...", "..."],
    "weaknesses": ["..."]
  },
  "trajectory": {
    "total_input_tokens": 1196740,
    "total_output_tokens": 16486,
    "avg_step_duration_s": 8.28,
    "tool_success_rate": 1.0
  },
  "patch_similarity": {
    "human_patch_available": false,
    "file_overlap_pct": 0.0,
    "line_overlap_pct": 0.0
  }
}
```

### 3. `trajectory_metrics.json` - Per-Step Metrics

Detailed per-interaction metrics extracted from `trajectory.json`.

```json
{
  "start_time": "2025-11-27T10:02:49.605179",
  "end_time": "2025-11-27T10:12:14.397912",
  "total_duration_s": 564.79,
  "step_count": 45,
  "step_durations": [5.27, 2.92, 2.56, ...],
  "avg_step_duration_s": 8.28,
  "max_step_duration_s": 33.10,
  "min_step_duration_s": 2.56,

  "total_input_tokens": 1196740,
  "total_output_tokens": 16486,
  "total_cache_read_tokens": 0,
  "total_reasoning_tokens": 0,

  "total_tool_calls": 45,
  "successful_tool_calls": 45,
  "failed_tool_calls": 0,
  "tool_success_rate": 1.0,
  "tool_call_counts": {
    "bash": 22,
    "str_replace_based_edit_tool": 17,
    "sequentialthinking": 5
  },

  "interactions": [
    {
      "step_index": 0,
      "timestamp": "2025-11-27T10:02:54.879031",
      "elapsed_s": 5.27,
      "input_tokens": 6074,
      "output_tokens": 121,
      "tool_calls": ["bash"],
      "tool_success": 1,
      "tool_failure": 0
    },
    ...
  ]
}
```

### 4. `patch_similarity.json` - Agent vs Human Comparison

Compares the agent's generated patch against the human developer's original optimization.

```json
{
  "human_patch_available": true,
  "human_patch_source": "vllm_final_dataset.jsonl:line42",

  "agent_files": ["vllm/model.py", "vllm/utils.py"],
  "human_files": ["vllm/model.py"],
  "common_files": ["vllm/model.py"],
  "agent_only_files": ["vllm/utils.py"],
  "human_only_files": [],

  "file_overlap_pct": 100.0,
  "agent_lines_added": 15,
  "agent_lines_removed": 8,
  "human_lines_added": 12,
  "human_lines_removed": 6,
  "matching_additions": 10,
  "matching_removals": 5,
  "line_overlap_pct": 83.3,

  "approach_similarity_score": 7.5
}
```

### 5. `llm_analysis.json` - Raw LLM Output

The complete structured JSON returned by the analysis LLM (Gemini 3 Pro).

```json
{
  "quality_scores": {
    "code_understanding": {
      "score": 9.0,
      "repository_navigation": 9.0,
      "function_identification": 9.0,
      "dependency_awareness": 8.0,
      "justification": "..."
    },
    ...
  },
  "categorical": {
    "approach_category": "systematic",
    "tool_usage_pattern": "balanced",
    "failure_category": "none"
  },
  "quantitative_assessment": { ... },
  "detailed_analysis": { ... },
  "recommendations": { ... }
}
```

### 6. `llm_raw_scores.json` - Scores Only

Just the numeric scores and categories for quick analysis.

```json
{
  "scores": {
    "code_understanding": 9.0,
    "task_alignment": 9.0,
    "approach_quality": 10.0,
    "execution_quality": 9.0,
    "overall": 9.25
  },
  "sub_scores": {
    "code_understanding": {
      "repository_navigation": 9.0,
      "function_identification": 9.0,
      "dependency_awareness": 8.0
    },
    ...
  },
  "categorical": {
    "approach_category": "systematic",
    "tool_usage_pattern": "balanced",
    "failure_category": "none"
  }
}
```

### 7. `llm_prompt.txt` - Analysis Prompt

The exact prompt sent to the LLM for analysis. Useful for reproducibility and debugging.

### 8. `llm_response.json` - Full LLM Response

Complete LLM response including thinking traces and token usage.

```json
{
  "model": "google/gemini-3-pro-preview",
  "provider": "openrouter",
  "thinking_enabled": true,
  "timestamp": "2025-12-16T21:48:00.123456",
  "duration_s": 45.2,
  "usage": {
    "prompt_tokens": 52000,
    "completion_tokens": 3500,
    "total_tokens": 55500
  },
  "thinking": "...",
  "response": "..."
}
```

---

## Aggregate Report

`aggregate_report.json` provides cross-run statistics grouped by:
- **Overall**: All runs combined
- **By Repo**: Per repository (vllm, sglang)
- **By Agent**: Per agent framework (trae, codex, openhands)
- **By Model**: Per LLM model (claude-sonnet-45, gpt-5)
- **By Agent+Model**: Combined grouping

```json
{
  "generated_at": "2025-12-16T21:49:57.064119",
  "total_runs": 42,
  "overall": {
    "count": 42,
    "success_rate": 0.76,
    "patch_rate": 0.81,
    "scores": {
      "avg_code_understanding": 7.2,
      "avg_overall": 6.8
    },
    "distributions": {
      "approach_categories": {"systematic": 25, "direct_edit": 10, ...},
      "failure_categories": {"execution": 5, "understanding": 3, ...}
    }
  },
  "by_agent": { ... },
  "by_model": { ... }
}
```

---

## Metrics Reference

### Quality Scores (0-10 scale, LLM-assessed)

| Score | Description | Calibration |
|-------|-------------|-------------|
| `code_understanding` | Repository navigation, function identification, dependency awareness | 0-2: Failure, 3-4: Poor, 5-6: Average, 7-8: Good, 9: Excellent, 10: Exceptional |
| `task_alignment` | Goal comprehension, constraint adherence, output relevance | Same scale |
| `approach_quality` | Strategy coherence, exploration efficiency, decision quality | Same scale |
| `execution_quality` | Edit correctness, test verification, error recovery | Same scale |
| `overall` | Weighted average of above scores | Same scale |

### Categorical Classifications

| Category | Options |
|----------|---------|
| `approach_category` | `systematic`, `direct_edit`, `exploration_heavy`, `trial_error`, `minimal` |
| `tool_usage_pattern` | `bash_heavy`, `editor_focused`, `read_heavy`, `balanced` |
| `failure_category` | `understanding`, `execution`, `timeout`, `api_error`, `no_patch`, `none` |

### Trajectory Metrics

| Metric | Description |
|--------|-------------|
| `total_input_tokens` | Sum of input tokens across all LLM calls |
| `total_output_tokens` | Sum of output tokens |
| `total_reasoning_tokens` | Tokens used for chain-of-thought reasoning |
| `avg_step_duration_s` | Average time per agent step |
| `tool_success_rate` | Successful tool calls / total tool calls |
| `tool_call_counts` | Count per tool type (bash, editor, etc.) |

### Patch Similarity Metrics

| Metric | Description |
|--------|-------------|
| `file_overlap_pct` | % of human-modified files also modified by agent |
| `line_overlap_pct` | % of human line changes matched by agent |
| `approach_similarity_score` | Semantic similarity (0-10) based on code patterns |
| `human_patch_available` | Whether human reference patch was found |

---

## Usage

### Running Analysis

```bash
cd perf-agents-bench

# Analyze all runs
python -m bench.cli analyze state/runs --output state/analysis

# Analyze specific repo/agent
python -m bench.cli analyze state/runs --repo vllm --agent trae

# Skip LLM (quantitative only)
python -m bench.cli analyze state/runs --skip-llm
```

### Loading Results Programmatically

```python
from pathlib import Path
import json

# Load a single analysis
analysis_path = Path("state/analysis/vllm/trae/claude-sonnet-45/2025-11-27_15-42-14/vllm_bedrock_sonnet45-0001")
summary = json.loads((analysis_path / "metrics_summary.json").read_text())

print(f"Overall score: {summary['scores']['overall']}")
print(f"Token usage: {summary['trajectory']['total_input_tokens']:,} input")

# Load aggregate report
report = json.loads(Path("state/analysis/aggregate_report.json").read_text())
print(f"Success rate: {report['overall']['success_rate']:.1%}")
```

---

## Schema Version History

| Version | Changes |
|---------|---------|
| 3.0 | Added trajectory_metrics.json, patch_similarity.json, llm_analysis.json, llm_raw_scores.json |
| 2.1 | Added score calibration guidelines |
| 2.0 | Initial Pydantic-based schema |
