# Soft Metrics Analyzer V3 — LLM-as-a-Judge Evaluation System

An academic-grade evaluation system that uses an LLM judge to assess how well AI coding agents perform on performance optimization tasks. Given an agent's benchmark run artifacts and a human expert's reference solution, it produces structured quality scores, categorical assessments, and detailed analysis suitable for research papers.

**Branch:** `feature/soft-metrics-analyzer-v3`

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Steps](#pipeline-steps)
- [Model Configuration](#model-configuration)
- [Prompt 1: Main Analysis](#prompt-1-main-analysis)
  - [Inputs](#main-analysis-inputs)
  - [Outputs](#main-analysis-outputs)
  - [Score Calibration](#score-calibration)
- [Prompt 2: Patch Quality (V4)](#prompt-2-patch-quality-v4)
  - [Inputs](#patch-quality-inputs)
  - [Outputs](#patch-quality-outputs)
  - [Category Taxonomy](#category-taxonomy)
- [Prompt 3: Aggregation](#prompt-3-aggregation)
- [Rule-Based Patch Comparison](#rule-based-patch-comparison)
- [Run Artifact Loading](#run-artifact-loading)
- [Output Files](#output-files)
- [Data Models](#data-models)
- [Key Findings](#key-findings)
- [Benchmark Results](#benchmark-results)
- [Source Files](#source-files)

---

## Overview

The system answers two questions for each agent benchmark run:

1. **How capable was the agent?** Scored on four dimensions (0-10) with sub-scores, plus categorical classification of approach, tool usage, and failure mode.

2. **How does the agent's patch compare to the human expert's?** Assessed categorically across bottleneck targeting, optimization techniques, approach similarity, speedup likelihood, and failure mode — with mandatory written discussion for every judgment.

The judge model is **Gemini 3 Flash** via OpenRouter, called at temperature 0.0 for deterministic outputs. Two separate LLM calls are made per run — one for the main analysis, one for the patch quality comparison.

---

## Architecture

```
Run Artifacts
  │
  │  ┌──────────────────────────────────────────────────────┐
  │  │ RunLoader                                            │
  │  │   Reads: journal.json, run_summary.json, prompt.json │
  │  │          stdout, stderr, patch, trajectory.json      │
  │  └──────────────┬───────────────────────────────────────┘
  │                 │
  ▼                 ▼
  ┌─────────────────────────────┐    ┌──────────────────────────────┐
  │ Quantitative Extraction     │    │ PatchComparator              │
  │ (no LLM)                    │    │ (no LLM)                     │
  │                             │    │                              │
  │ - Tool call counts          │    │ - File overlap (Jaccard)     │
  │ - Timing (duration, TTFE)   │    │ - Line overlap (fuzzy 80%)   │
  │ - Patch stats (LOC, files)  │    │ - Semantic similarity (0-10) │
  │ - Error counts              │    │ - Pattern matching           │
  └─────────────┬───────────────┘    └──────────────┬───────────────┘
                │                                   │
                ▼                                   ▼
  ┌─────────────────────────────┐    ┌──────────────────────────────┐
  │ LLM Call #1                 │    │ LLM Call #2                  │
  │ Main Analysis Prompt        │    │ Patch Quality Prompt (V4)    │
  │                             │    │                              │
  │ Input: full agent output,   │    │ Input: human patch, agent    │
  │   task context, metadata    │    │   patch, rule-based metrics  │
  │                             │    │                              │
  │ Output:                     │    │ Output:                      │
  │ - 4 quality scores (0-10)  │    │ - Categorical assessments    │
  │ - 12 sub-scores             │    │ - Mandatory discussions      │
  │ - 3 categorical labels      │    │ - NO numeric scores          │
  │ - Detailed analysis text    │    │                              │
  └─────────────┬───────────────┘    └──────────────┬───────────────┘
                │                                   │
                ▼                                   ▼
  ┌─────────────────────────────┐    ┌──────────────────────────────┐
  │ Trajectory Metrics          │    │ OutputWriter                 │
  │ (no LLM)                    │    │                              │
  │                             │    │ Writes 9 files per run:      │
  │ - Per-step token usage      │    │ analysis.json, llm_prompt,   │
  │ - Tool success rates        │    │ llm_response, raw_scores,    │
  │ - Step timing analysis      │    │ metrics_summary, trajectory, │
  │                             │    │ patch_similarity,            │
  └─────────────┬───────────────┘    │ patch_quality, llm_analysis  │
                │                    └──────────────────────────────┘
                └────────► Combined into RunAnalysis (Pydantic) ──►
```

---

## Pipeline Steps

### Step 1: Load Artifacts

`RunLoader` reads everything from the run directory:

```
state/runs/{repo}/{agent}/{model}/{timestamp}/{item_id}/
├── journal.json              # Agent metadata, status, metrics
├── run_summary.json          # Execution summary, patch stats
├── prompt.json               # Task prompt, constraints, target files
├── diff_targets.json         # File compliance rules
├── task.txt                  # Task description text
├── {agent}_stdout.txt        # Agent execution stdout
├── {agent}_stderr.txt        # Agent execution stderr
├── model_patch.diff          # Generated unified diff
├── trajectory.json           # Full agent step-by-step history
└── prediction.jsonl          # Model predictions (optional)
```

Agent log file names vary by agent:

| Agent | stdout | stderr |
|-------|--------|--------|
| trae | `trae_stdout.txt` | `trae_stderr.txt` |
| openhands | `openhands_stdout.txt` | `openhands_stderr.txt` |
| codex | `codex_stdout.txt` | `codex_stderr.txt` |
| claude_code | `claude_code_stdout.json` | `claude_code_stderr.txt` |

Metadata is extracted from the directory path structure: `state/runs/{repo}/{agent}/{model}/{timestamp}/{item_id}`.

### Step 2: Extract Quantitative Metrics (No LLM)

Parsed directly from the artifact files:
- **Timing**: total duration, time to first edit
- **Tool calls**: bash, editor, read, search, web_search, other (with breakdown)
- **Tokens**: input tokens, output tokens consumed
- **Patch**: lines added/removed, files changed, compliance with allowed files
- **Errors**: count, exception types, warnings
- **Status**: final run status, return code

### Step 3: Compare Patches Rule-Based (No LLM)

`PatchComparator` parses both the agent's and human's unified diffs, then computes:

| Metric | Method |
|--------|--------|
| `file_overlap_pct` | Jaccard similarity of modified file sets |
| `line_overlap_pct` | Jaccard similarity of modified lines, with 80% fuzzy threshold via `difflib.SequenceMatcher` |
| `approach_similarity_score` | Weighted combination (0-10): 30% file overlap + 40% line content similarity + 30% code pattern overlap |

Code pattern overlap uses regex extraction:
- `def\s+(\w+)` — function names
- `class\s+(\w+)` — class names
- `import\s+(\w+)` and `from\s+(\w+)` — import names
- `(\w+)\s*=` — assignment targets

Human patches are loaded from dataset files in priority order:
1. `data/final/{repo}_final_dataset.jsonl`
2. `data/{repo}_dataset.jsonl`
3. `data/{repo}_dataset_with_test.jsonl`

Matching uses commit hash prefix (first 7 characters).

### Step 4: LLM Call #1 — Main Analysis

See [Prompt 1: Main Analysis](#prompt-1-main-analysis) below.

### Step 5: LLM Call #2 — Patch Quality

See [Prompt 2: Patch Quality (V4)](#prompt-2-patch-quality-v4) below.

### Step 6: Extract Trajectory Metrics (No LLM)

Parses `trajectory.json` to extract per-interaction metrics:
- Token consumption per step (input, output, cache, reasoning)
- Tool calls per step with success/failure counts
- Step timing (duration, elapsed since start)
- Aggregates: total tokens, tool success rate, average step duration

### Step 7: Write Outputs

`OutputWriter` saves results to `state/analysis/{repo}/{agent}/{model}/{timestamp}/{item_id}/`. See [Output Files](#output-files).

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | `google/gemini-3-flash-preview` |
| **Provider** | OpenRouter (`https://openrouter.ai/api/v1/chat/completions`) |
| **Temperature** | 0.0 (deterministic) |
| **Max tokens** | 8,192 |
| **Timeout** | 7,200 seconds (2 hours, for massive patches) |
| **Retries** | 3 with exponential backoff (5s base delay) |
| **Thinking mode** | Enabled, 10,000 token budget |
| **Response caching** | SHA256 hash of prompt, stored on disk |
| **Authentication** | `OPENROUTER_API_KEY` environment variable |

The client extracts thinking content from `<thinking>` tags in the response if present, and strips it from the main content. JSON is extracted from markdown code blocks or raw response text, with multiple fallback strategies.

---

## Prompt 1: Main Analysis

### Main Analysis Inputs

The prompt template (`ANALYSIS_PROMPT_TEMPLATE`) is populated with:

| Field | Source File | Truncation |
|-------|------------|-----------|
| `task_content` | `task.txt` or `prompt.json` description | None |
| `agent_type` | `journal.json` | None |
| `model_full` | `run_summary.json` meta | None |
| `repo` | Directory path | None |
| `time_budget_minutes` | `journal.json` agent info | None |
| `task_id` | `journal.json` or `run_summary.json` | None |
| `commit_pre` | `run_summary.json` commits | None |
| `commit_human` | `run_summary.json` commits | None |
| `constraints` | `prompt.json` constraints list | None |
| `target_files` | `prompt.json` target_files list | None |
| `stdout_content` | `{agent}_stdout.txt` | **500 KB** (first half + last half) |
| `stderr_content` | `{agent}_stderr.txt` | **50 KB** (first 50KB) |
| `patch_content` | `model_patch.diff` | **100 KB** (first 100KB) |
| `diff_targets` | `diff_targets.json` (JSON string) | None |
| `status` | `journal.json` | None |
| `duration_s` | `journal.json` agent info | None |
| `time_to_first_edit_s` | `journal.json` metrics | None |
| `patch_generated` | Computed from patch size | None |
| `files_changed_count` | `journal.json` or `run_summary.json` | None |
| `lines_added` / `lines_removed` | `run_summary.json` patch_stats | None |
| `returncode` | `journal.json` agent info | None |

### Main Analysis Outputs

The LLM returns a JSON object with five sections:

#### 1. Quantitative Assessment (LLM-counted from logs)

```json
{
  "tool_calls": {"bash": 0, "editor": 0, "read": 0, "other": 0},
  "error_counts": {"syntax": 0, "runtime": 0, "api": 0, "timeout": 0},
  "iteration_count": 0,
  "exploration_steps": 0,
  "execution_steps": 0
}
```

#### 2. Tool Usage Analysis

```json
{
  "bash": 0, "read": 0, "editor": 0, "web_search": 0, "search": 0, "other": 0,
  "other_details": {"tool_name": 0},
  "discussion": "Verification that counts match logs"
}
```

#### 3. Quality Scores — four dimensions, each with three sub-scores

```json
{
  "code_understanding": {
    "score": 8.0,
    "repository_navigation": 8.0,
    "function_identification": 9.0,
    "dependency_awareness": 7.0,
    "justification": "Agent navigated the repo effectively..."
  },
  "task_alignment": {
    "score": 9.0,
    "goal_comprehension": 10.0,
    "constraint_adherence": 9.0,
    "output_relevance": 9.0,
    "justification": "..."
  },
  "approach_quality": {
    "score": 8.0,
    "strategy_coherence": 9.0,
    "exploration_efficiency": 9.0,
    "decision_quality": 7.0,
    "justification": "..."
  },
  "execution_quality": {
    "score": 8.0,
    "edit_correctness": 9.0,
    "test_verification": 6.0,
    "error_recovery": 5.0,
    "justification": "..."
  }
}
```

Overall score is the unweighted average (0.25 per dimension).

#### 4. Categorical Classification

```json
{
  "approach_category": "direct_edit",
  "tool_usage_pattern": "balanced",
  "failure_category": "none"
}
```

**Approach categories:**
- `systematic` — methodical, step-by-step, clear plan
- `direct_edit` — quick targeted changes, minimal exploration
- `exploration_heavy` — extensive codebase exploration before acting
- `trial_error` — iterative attempts without clear strategy
- `minimal` — minimal viable changes, possibly incomplete

**Tool usage patterns:**
- `bash_heavy` — primarily shell commands
- `editor_focused` — primarily file editing
- `read_heavy` — primarily reading files
- `balanced` — mixed

**Failure categories:**
- `understanding` — misunderstood task or codebase
- `execution` — understood but failed to execute
- `timeout` — ran out of time or steps
- `api_error` — technical failures
- `no_patch` — completed but no meaningful changes
- `none` — successful

#### 5. Detailed Analysis and Recommendations

```json
{
  "detailed_analysis": {
    "key_decisions": ["Chose to modify X before Y", "..."],
    "optimization_techniques": ["memory_optimization", "..."],
    "missed_opportunities": ["Could have used pinned memory", "..."],
    "error_recovery_strategy": "Reverted changes after syntax error...",
    "summary": "Efficient run with good constraint adherence...",
    "strengths": ["Fast execution", "..."],
    "weaknesses": ["No performance verification", "..."]
  },
  "recommendations": {
    "prompting_improvements": ["Include profiling data in prompt", "..."],
    "capability_additions": ["GPU profiling tool access", "..."],
    "patterns_to_encourage": ["Verify changes with tests", "..."],
    "patterns_to_discourage": ["Blind pattern application", "..."]
  }
}
```

### Score Calibration

The prompt enforces strict calibration to prevent score inflation:

| Range | Meaning | Expected frequency |
|-------|---------|-------------------|
| 0-2 | Complete failure, catastrophic errors | Broken runs |
| 3-4 | Poor, significant gaps, major mistakes | Failed attempts |
| 5-6 | Average, partial success, notable weaknesses | Mediocre runs |
| 7-8 | Good, mostly correct, minor issues | **Most successful runs** |
| 9 | Excellent, near-optimal | Top performers |
| 10 | Exceptional, exceeds expectations | **<5% of runs** |

The prompt explicitly states: *"Most successful runs should score 6-8, not 9-10. Be critical and specific about deficiencies even for good runs."*

---

## Prompt 2: Patch Quality (V4)

This is the categorical assessment prompt, inspired by the **GSO Benchmark paper** (arXiv:2505.23671v3). It compares the agent's patch against the human expert's reference and returns **categories with mandatory discussion fields only — no numeric scores**.

### Patch Quality Inputs

| Field | Source | Truncation |
|-------|--------|-----------|
| `task_description` | `task.txt` | None |
| `human_patch` | Dataset JSONL (matched by commit hash) | **50 KB** |
| `agent_patch` | `model_patch.diff` | **50 KB** |
| `rule_based_metrics` | PatchComparator output (formatted as key-value lines) | None |

### Patch Quality Outputs

```json
{
  "task_analysis": {
    "domain": "memory",
    "complexity": "high",
    "description": "Optimizing LoRA management path..."
  },
  "bottleneck_target": {
    "category": "same_target",
    "human_target": "Optimizes LoRA metadata via pinned memory...",
    "agent_target": "Switches torch.zeros to torch.empty...",
    "discussion": "Both target LoRA manager. Human identifies deeper bottleneck..."
  },
  "optimization_techniques": {
    "human_techniques": ["memory_optimization", "parallelization", "lazy_computation"],
    "agent_techniques": ["memory_optimization"],
    "technique_overlap": true,
    "discussion": "Human uses pinned memory and non-blocking copies..."
  },
  "approach_comparison": {
    "category": "partial_solution",
    "discussion": "Agent's approach is a subset of human's..."
  },
  "speedup_likelihood": {
    "category": "likely_ineffective",
    "discussion": "Changes provide negligible speedup compared to..."
  },
  "failure_mode": {
    "category": "complexity_avoidance",
    "discussion": "Agent followed example pattern too literally..."
  },
  "observations": {
    "key_differences": ["Human implemented async transfer via pinned memory", "..."],
    "agent_strengths": ["Correctly identified all torch.zeros locations", "..."],
    "agent_weaknesses": ["Over-reliance on example pattern", "..."],
    "benchmark_needed": "LoRA forward pass latency with many adapters..."
  },
  "library_failure": {
    "responsible_libraries": ["pytorch"],
    "failure_reason": "Failed to use pin_memory=True and non_blocking=True..."
  }
}
```

Every `discussion` field is **mandatory**. Selecting `other` for any category requires an explanation.

### Category Taxonomy

#### Task Domain
`compute` | `memory` | `io` | `concurrency` | `other`

#### Task Complexity
`low` | `medium` | `high` | `extreme`

#### Bottleneck Target — does the agent optimize the same thing?

| Category | Meaning |
|----------|---------|
| `same_target` | Optimizes the exact same bottleneck/function as human |
| `related_target` | Optimizes related code affecting the same performance path |
| `different_target` | Optimizes something else entirely |
| `no_optimization` | No meaningful optimization attempted |
| `other` | Must explain in discussion |

#### Optimization Techniques — what methods did each use?

| Technique | Meaning |
|-----------|---------|
| `algorithmic` | Better algorithm or data structure |
| `memory_optimization` | Reduced allocations, better layout, caching |
| `parallelization` | Threading, vectorization, GPU offload |
| `api_library` | Faster API calls or optimized libraries |
| `lazy_computation` | Deferred or avoided unnecessary computation |
| `compiler_optimization` | torch.compile, JIT, fusion |
| `batching` | Combined operations to reduce overhead |
| `low_level` | Assembly, CUDA kernels, intrinsics |
| `other` | Must explain in discussion |

#### Approach Comparison — how similar is the solution?

| Category | Meaning |
|----------|---------|
| `same_approach` | Essentially the same solution |
| `similar_approach` | Same strategy, different implementation details |
| `valid_alternative` | Different but potentially valid optimization |
| `partial_solution` | Addresses part of the optimization |
| `ineffective` | Changes unlikely to improve performance |
| `harmful` | Changes likely to hurt performance or correctness |
| `other` | Must explain in discussion |

#### Speedup Likelihood — will it actually work?

| Category | Meaning |
|----------|---------|
| `likely_similar` | Probably achieves similar speedup to human |
| `likely_partial` | Probably achieves some but not full speedup |
| `uncertain` | Cannot determine without benchmarking |
| `likely_ineffective` | Probably no meaningful speedup |
| `likely_regression` | May cause performance regression |
| `other` | Must explain in discussion |

#### Failure Mode — what went wrong?

| Category | Meaning |
|----------|---------|
| `localization_failure` | Misidentified the bottleneck or wrong abstraction level |
| `technique_mismatch` | Right target, wrong optimization technique |
| `incomplete_implementation` | Right idea, incomplete execution |
| `complexity_avoidance` | Avoided necessary low-level optimizations |
| `overcomplicated` | Added unnecessary complexity |
| `library_misuse` | Incorrect usage of a specific library |
| `not_applicable` | Agent solution is valid/successful |
| `other` | Must explain in discussion |

---

## Prompt 3: Aggregation

`build_summary_prompt()` takes a list of completed `RunAnalysis` results and asks the LLM to produce an aggregate report:

**Inputs per run:** item_id, agent, model, status, duration, the four quality scores.

**Requested output:**
```json
{
  "summary": "...",
  "success_rate": 0.0,
  "avg_scores": {},
  "common_failures": [],
  "agent_comparison": {},
  "model_comparison": {},
  "key_insights": []
}
```

---

## Rule-Based Patch Comparison

`PatchComparator` runs before the LLM calls and feeds its results into both prompts as context. It works in three stages:

### 1. Parse Diffs
`PatchParser` splits a unified diff into:
- Files changed (from `---`/`+++` headers)
- Hunks (from `@@` headers with line ranges)
- Added lines (lines starting with `+`)
- Removed lines (lines starting with `-`)

### 2. Compute Overlap
- **File overlap**: Jaccard = `|agent_files ∩ human_files| / |agent_files ∪ human_files|` (as percentage)
- **Line overlap**: Jaccard on the set of changed lines, where matching uses:
  - Exact match after normalization (strip + lowercase)
  - Fuzzy match at 80% threshold via `difflib.SequenceMatcher` for remaining lines
  - Union = `agent_changes + human_changes - matching_lines`

### 3. Semantic Similarity (0-10)
Weighted combination:

| Component | Weight | Method |
|-----------|--------|--------|
| File similarity | 30% | `|common_files| / |human_files|` |
| Content similarity | 40% | `SequenceMatcher(all_agent_lines, all_human_lines).ratio()` |
| Pattern similarity | 30% | Jaccard overlap of extracted identifiers (function/class/import/assignment names via regex) |

Final score = `(file_sim * 0.3 + content_sim * 0.4 + pattern_sim * 0.3) * 10`, clamped to [0, 10].

---

## Run Artifact Loading

### Directory Discovery

`SoftMetricsAnalyzer.discover_runs()` scans `state/runs/` and finds all item directories matching optional filters:
- `repo_filter` — e.g., "vllm" or "sglang"
- `agent_filter` — e.g., "trae" or "claude_code"
- `model_filter` — e.g., "gpt-5"
- `timestamp_filter` — specific run timestamp

### Human Patch Resolution

The system searches for the human reference patch in dataset files:

**Search priority for directories:**
1. `data/final/` (curated final datasets)
2. `data/` (general data directory)

**Search priority for files:**
1. `{repo}_final_dataset.jsonl`
2. `{repo}_dataset.jsonl`
3. `{repo}_dataset_with_test.jsonl`
4. `{repo}.jsonl`

Commit matching uses the first 7 characters of the hash. The patch is read from either the `patch` or `diff_text` field in the JSONL record.

---

## Output Files

Written to `state/analysis/{repo}/{agent}/{model}/{timestamp}/{item_id}/`:

| File | Contents | Always written |
|------|----------|---------------|
| `analysis.json` | Complete `RunAnalysis` Pydantic model dump | Yes |
| `llm_prompt.txt` | Exact prompt text sent to Gemini | Yes |
| `llm_response.json` | Full LLM response: model, thinking content, response text, token usage | Yes |
| `metrics_summary.json` | Compact summary of key metrics for quick reference | Yes |
| `llm_analysis.json` | Structured JSON from LLM: quality scores, categories, detailed analysis | If LLM succeeded |
| `llm_raw_scores.json` | Just the scores, sub-scores, and categorical labels | If LLM succeeded |
| `trajectory_metrics.json` | Per-step token/tool/timing breakdown | If trajectory.json exists |
| `patch_similarity.json` | Rule-based agent vs human patch comparison | If human patch found |
| `patch_quality.json` | V4 categorical patch assessment | If human patch found |

---

## Data Models

All models are Pydantic `BaseModel` subclasses with validation, defined in `schemas.py`.

### Top-Level: `RunAnalysis`

```
RunAnalysis (schema_version: "3.0")
│
├── meta: RunMetadata
│     repo, agent, model, model_full, timestamp, item_id,
│     task_id, commits: {pre, human}, source_files
│
├── quantitative: QuantitativeMetrics
│     ├── duration_s, time_to_first_edit_s, step_count, iteration_count
│     ├── tool_calls: ToolDistribution
│     │     bash, editor, read, search, web_search, other,
│     │     other_details: {name: count}, total (computed)
│     ├── input_tokens, output_tokens
│     ├── patch: PatchMetrics
│     │     generated, lines_added, lines_removed, hunks,
│     │     files_changed, files_allowed, files_disallowed,
│     │     compliance_ok, total_lines (computed)
│     ├── error_count, warning_count, exception_types
│     └── status, returncode
│
├── qualitative: QualitativeScores
│     ├── code_understanding: QualityScore {score, sub_scores, justification}
│     ├── task_alignment: QualityScore
│     ├── approach_quality: QualityScore
│     ├── execution_quality: QualityScore
│     └── overall_score (computed: mean of four scores)
│
├── categorical: CategoricalMetrics
│     approach_category, tool_usage_pattern, failure_category
│
├── analysis: FreeFormAnalysis
│     summary, key_decisions, optimization_techniques,
│     missed_opportunities, error_recovery_strategy,
│     strengths, weaknesses, recommendations
│
├── llm: LLMAnalysisResult
│     model, provider, thinking_enabled,
│     prompt_tokens, completion_tokens, total_tokens,
│     timestamp, duration_s
│
├── trajectory: TrajectoryMetrics (optional)
│     ├── Timing: start_time, end_time, total_duration_s,
│     │          step_count, step_durations, avg/max/min_step_duration_s
│     ├── Tokens: total_input/output/cache_read/cache_creation/reasoning_tokens,
│     │          total_tokens (computed)
│     ├── Tools: total_tool_calls, successful/failed_tool_calls,
│     │         tool_success_rate, tool_call_counts: {name: count}
│     └── interactions: [InteractionMetrics]
│           step_index, timestamp, elapsed_s, tokens, tool_calls
│
├── patch_similarity: PatchSimilarityMetrics (optional)
│     human_patch_available, human_patch_source,
│     agent/human/common/agent_only/human_only_files,
│     file_overlap_pct, line_overlap_pct,
│     agent/human_lines_added/removed,
│     matching_additions/removals,
│     approach_similarity_score
│
├── llm_raw: LLMRawScores (optional)
│     quality_scores, categorical, quantitative_assessment,
│     tool_usage_analysis, detailed_analysis, recommendations
│
├── patch_quality: PatchQualityAnalysis (optional)
│     ├── human_patch_available, analysis_model, analysis_tokens
│     ├── task_analysis: TaskAnalysis {domain, complexity, description}
│     ├── bottleneck_target: {category, human_target, agent_target, discussion}
│     ├── optimization_techniques: {human_techniques, agent_techniques, overlap, discussion}
│     ├── approach_comparison: {category, discussion}
│     ├── speedup_likelihood: {category, discussion}
│     ├── failure_mode: {category, discussion}
│     ├── observations: {key_differences, strengths, weaknesses, benchmark_needed}
│     └── library_failure: {responsible_libraries, failure_reason}
│
└── analyzed_at, analysis_duration_s
```

---

## Key Findings

### Soft vs Hard Metrics Correlation

When comparing the LLM judge's predictions against actual runtime benchmarks (31 matched commits across vLLM and SGLang):

| Prediction | Accuracy | Notes |
|-----------|----------|-------|
| `likely_ineffective` | **77.8%** | Most reliable — the judge correctly identifies non-improvements |
| `likely_regression` | **100%** | High confidence, but few samples |
| `likely_similar` | 60% | Good for vLLM (100%), poor for SGLang (0%) |
| `likely_partial` | **40%** | Least reliable — high false positive rate |

**Overall accuracy:** 58% (vLLM: 64%, SGLang: 53%).

### "Match to Human" Does Not Equal Good Performance

| Approach category | Actual success rate | Explanation |
|-------------------|---------------------|-------------|
| `partial_solution` | **76%** | Focused, clean changes tend to work |
| `valid_alternative` | ~65% | Different approach can still succeed |
| `similar_approach` | **45%** | Trying to replicate the human exactly introduces implementation bugs |
| `same_approach` | ~51% | Same strategy but details differ enough to break things |

### Anti-Patterns Kill Performance

Runs with identified anti-patterns (manual Python loops instead of builtins, wrong API usage like `any()` vs `torch.any()`) have a **75% failure rate**.

---

## Benchmark Results

Results from the V3 analysis across 452 agent runs:

| Agent | Runs | Scored | Avg Score | Success Rate |
|-------|------|--------|-----------|-------------|
| Claude Code (Sonnet 4.5) | 192 | 189 | 7.64 | 98.4% |
| Codex (GPT-5) | 99 | 98 | 7.05 | 99.0% |
| TRAE (GPT-5) | 70 | 49 | 7.57 scored / 5.30 all | 70.0% |
| TRAE (Sonnet 4.5) | 91 | 46 | 8.12 scored / 4.10 all | 50.5% |

TRAE's lower success rates are due to infrastructure issues (API 429 errors, AWS token expiry), not agent capability. When TRAE runs succeed, its scores are competitive.

---

## Source Files

All files live on the `feature/soft-metrics-analyzer-v3` branch:

| File | Lines | Purpose |
|------|-------|---------|
| `perf-agents-bench/bench/analysis/analyzer.py` | ~1,189 | Main orchestrator class `SoftMetricsAnalyzer` |
| `perf-agents-bench/bench/analysis/prompts.py` | ~450 | All three prompt templates and builder functions |
| `perf-agents-bench/bench/analysis/openrouter_client.py` | ~250 | OpenRouter API client with caching, retry, thinking mode |
| `perf-agents-bench/bench/analysis/schemas.py` | ~641 | All Pydantic data models and enums |
| `perf-agents-bench/bench/analysis/patch_comparator.py` | ~350 | Rule-based patch parsing and comparison |
| `perf-agents-bench/bench/analysis/run_loader.py` | ~250 | Artifact loading from run directories |
| `perf-agents-bench/bench/analysis/output_writer.py` | ~200 | Hierarchical result writing |
| `scripts/analyze_soft_metrics.py` | — | Aggregation across all runs |
| `scripts/compare_soft_hard_metrics.py` | — | Soft vs hard metrics correlation analysis |

---

## Appendix A: Verbatim Main Analysis Prompt

The following is the exact `ANALYSIS_PROMPT_TEMPLATE` from `prompts.py`. Placeholder variables (e.g., `{task_content}`) are populated at runtime by `build_analysis_prompt()`.

```text
# Agent Performance Analysis for Academic Research

You are analyzing a software agent's attempt to optimize code performance. This analysis will be used for academic research on AI agent capabilities.

## Task Context

The agent was given the following task:

<task_description>
{task_content}
</task_description>

## Agent Configuration

- **Agent Framework**: {agent_type}
- **LLM Model**: {model_full}
- **Repository**: {repo}
- **Time Budget**: {time_budget_minutes} minutes
- **Task ID**: {task_id}

## Commits

- **Base Commit (pre-optimization)**: {commit_pre}
- **Human Reference Commit**: {commit_human}

## Constraints

{constraints}

## Target Files (Files the agent should modify)

{target_files}

---

## Agent Execution Data

### Full Agent Output (stdout)

<stdout>
{stdout_content}
</stdout>

### Errors and Warnings (stderr)

<stderr>
{stderr_content}
</stderr>

### Generated Patch

```diff
{patch_content}
```

### File Compliance Check

<diff_targets>
{diff_targets}
</diff_targets>

### Run Status Summary

- **Final Status**: {status}
- **Total Duration**: {duration_s:.1f} seconds
- **Time to First Edit**: {time_to_first_edit_s:.1f} seconds
- **Patch Generated**: {patch_generated}
- **Files Changed**: {files_changed_count}
- **Lines Added**: {lines_added}
- **Lines Removed**: {lines_removed}
- **Return Code**: {returncode}

---

## Analysis Instructions

Please provide a comprehensive analysis of this agent run. Your analysis should be rigorous and suitable for inclusion in an academic paper.

### 1. Quantitative Assessment

Count and categorize the following from the agent output:

- **Tool Calls**: Count each type of tool/action used (bash commands, file edits, file reads, etc.)
- **Errors**: Count errors by category (syntax, runtime, API, timeout)
- **Iterations**: Count edit-test cycles (how many times the agent modified code and tested)
- **Exploration Steps**: Count how many steps were spent exploring vs. executing

### 2. Tool Usage Analysis

Analyze the logs to provide exact counts of tool usage. Group tools as follows:
- **bash**: Shell commands (ls, cd, python, etc.)
- **read**: File reading operations (Read, etc.)
- **editor**: File modification operations (Write, Edit, NotebookEdit, TodoWrite, etc.)
- **web_search**: Web search operations (WebSearch, BraveSearch, etc.)
- **search**: Codebase search operations (Grep, Glob, Find, etc.)
- **other**: Any other tools not matching above
- **other_details**: A dictionary mapping specific "other" tool names to their counts.
- **discussion**: Briefly explain if the tool usage seems matched to the counts, especially for "other".

### 3. Quality Scores (0-10 scale with detailed justification)

**IMPORTANT CALIBRATION GUIDELINES:**
- **0-2**: Complete failure - no meaningful attempt or catastrophic errors
- **3-4**: Poor - significant gaps, major mistakes, largely ineffective
- **5-6**: Average - partial success, some correct decisions but notable weaknesses
- **7-8**: Good - mostly correct approach with minor issues, solid execution
- **9**: Excellent - nearly optimal, only minor imperfections
- **10**: Exceptional - reserve ONLY for truly outstanding performance that exceeds expectations; should be rare (<5% of runs)

**Scoring discipline:**
- A score of 10/10 requires flawless execution AND going beyond requirements
- Most successful runs should score 6-8, not 9-10
- Be critical and specific about deficiencies even for good runs
- Consider what an ideal agent would do, not just whether the task was completed

For each score, provide:
- The numeric score (0-10, calibrated per above guidelines)
- A detailed justification explaining why you gave this score
- Specific evidence from the agent output

#### Code Understanding Score
Sub-components:
- Repository Navigation (0-10): How well did the agent explore and understand the codebase structure?
- Function Identification (0-10): Did the agent correctly identify the relevant functions to modify?
- Dependency Awareness (0-10): Did the agent understand how different parts of the code interact?

#### Task Alignment Score
Sub-components:
- Goal Comprehension (0-10): Did the agent correctly understand what optimization was needed?
- Constraint Adherence (0-10): Did the agent respect all given constraints?
- Output Relevance (0-10): Were the changes relevant to the performance goal?

#### Approach Quality Score
Sub-components:
- Strategy Coherence (0-10): Was there a clear, logical strategy?
- Exploration Efficiency (0-10): Was exploration focused or scattered?
- Decision Quality (0-10): Were key decisions well-reasoned?

#### Execution Quality Score
Sub-components:
- Edit Correctness (0-10): Were edits syntactically and semantically correct?
- Test Verification (0-10): Did the agent verify that changes worked?
- Error Recovery (0-10): How well did the agent handle failures?

### 3. Categorical Classification

Classify the agent's behavior:

#### Approach Category
Choose ONE:
- `systematic`: Methodical, step-by-step approach with clear plan
- `direct_edit`: Quick targeted changes with minimal exploration
- `exploration_heavy`: Extensive codebase exploration before acting
- `trial_error`: Iterative attempts without clear strategy
- `minimal`: Minimal viable changes, possibly incomplete

#### Tool Usage Pattern
Choose ONE:
- `bash_heavy`: Primarily used bash/shell commands
- `editor_focused`: Primarily used file editing tools
- `read_heavy`: Primarily read files without much editing
- `balanced`: Balanced mix of different tools

#### Failure Category (if the run failed)
Choose ONE or `none` if successful:
- `understanding`: Agent misunderstood the task or codebase
- `execution`: Agent understood but failed to execute correctly
- `timeout`: Agent ran out of time or steps
- `api_error`: Technical/API failures interrupted the agent
- `no_patch`: Agent completed but produced no meaningful changes
- `none`: The run was successful

### 4. Detailed Analysis

#### Key Decisions (list 3-5)
Identify the most important decisions the agent made and explain their impact.

#### Optimization Techniques Used
List any performance optimization techniques the agent attempted (e.g., algorithm changes, memory optimization, parallelization).

#### Missed Opportunities
What could the agent have done better? What obvious optimizations did it miss?

#### Error Recovery Strategy
How did the agent respond to errors? Was the recovery effective?

### 5. Recommendations

- What changes to the agent's prompting might improve performance?
- What additional capabilities would help this type of task?
- What patterns should be encouraged or discouraged?

---

## Output Format

Return your analysis as a JSON object with the following structure:

{{
  "quantitative_assessment": {{
    "tool_calls": {{"bash": 0, "editor": 0, "read": 0, "other": 0}},
    "error_counts": {{"syntax": 0, "runtime": 0, "api": 0, "timeout": 0}},
    "iteration_count": 0,
    "exploration_steps": 0,
    "execution_steps": 0
  }},
  "technical_summary": "Expanded technical explanation of the changes...",
  "tool_usage_analysis": {{
      "bash": 0,
      "read": 0,
      "editor": 0,
      "web_search": 0,
      "search": 0,
      "other": 0,
      "other_details": {{
          "tool_name": 0
      }},
      "discussion": "Brief verification of tool usage vs logs"
  }},
  "quality_scores": {{
    "code_understanding": {{
      "score": 0.0,
      "repository_navigation": 0.0,
      "function_identification": 0.0,
      "dependency_awareness": 0.0,
      "justification": "..."
    }},
    "task_alignment": {{
      "score": 0.0,
      "goal_comprehension": 0.0,
      "constraint_adherence": 0.0,
      "output_relevance": 0.0,
      "justification": "..."
    }},
    "approach_quality": {{
      "score": 0.0,
      "strategy_coherence": 0.0,
      "exploration_efficiency": 0.0,
      "decision_quality": 0.0,
      "justification": "..."
    }},
    "execution_quality": {{
      "score": 0.0,
      "edit_correctness": 0.0,
      "test_verification": 0.0,
      "error_recovery": 0.0,
      "justification": "..."
    }}
  }},
  "categorical": {{
    "approach_category": "systematic|direct_edit|exploration_heavy|trial_error|minimal",
    "tool_usage_pattern": "bash_heavy|editor_focused|read_heavy|balanced",
    "failure_category": "understanding|execution|timeout|api_error|no_patch|none"
  }},
  "detailed_analysis": {{
    "key_decisions": ["decision 1", "decision 2", "..."],
    "optimization_techniques": ["technique 1", "..."],
    "missed_opportunities": ["opportunity 1", "..."],
    "error_recovery_strategy": "Description of how errors were handled",
    "summary": "Brief overall summary of the run",
    "strengths": ["strength 1", "..."],
    "weaknesses": ["weakness 1", "..."]
  }},
  "recommendations": {{
    "prompting_improvements": ["improvement 1", "..."],
    "capability_additions": ["capability 1", "..."],
    "patterns_to_encourage": ["pattern 1", "..."],
    "patterns_to_discourage": ["pattern 1", "..."]
  }}
}}

Ensure all scores are numbers between 0 and 10. Be specific and evidence-based in your justifications.
```

---

## Appendix B: Verbatim Patch Quality Prompt (V4)

The following is the exact `PATCH_QUALITY_PROMPT_TEMPLATE` from `prompts.py`. Placeholder variables are populated at runtime by `build_patch_quality_prompt()`.

```text
# Patch Quality Analysis: Agent vs Human Reference

You are analyzing how an AI agent's optimization patch compares to a human expert's reference implementation. This is for academic research on AI agent capabilities in performance optimization tasks.

## Task Description

<task_description>
{task_description}
</task_description>

## Human Reference Patch (Ground Truth)

This is the human expert's solution that achieved verified performance improvement:

```diff
{human_patch}
```

## Agent's Patch

This is what the AI agent produced:

```diff
{agent_patch}
```

## Rule-Based Metrics (Pre-computed)

These metrics were calculated automatically without LLM:

<rule_based_metrics>
{rule_based_metrics}
</rule_based_metrics>

---

## Analysis Instructions

Analyze how the agent's patch compares to the human reference. Focus on categorical assessment and detailed discussion - **DO NOT provide numeric scores**.

**IMPORTANT**: If you select any "other" category, you MUST explain exactly what it is in the discussion field. "Other" without explanation is invalid.

### 1. Task Analysis

Analyze the nature of the optimization task itself.

- What is the primary domain? (e.g. compute, memory, io, concurrency, algorithmic complication)
- What is the task about? Like a short description of the task.
- How complex is the task? (low, medium, high, extreme)
- Provide a brief description of the task's technical challenges.

### 2. Bottleneck Target

Does the agent target the same performance bottleneck as the human?

Categories:
- `same_target`: Optimizes the exact same bottleneck/function
- `related_target`: Optimizes related code that affects same performance path
- `different_target`: Optimizes something else entirely
- `no_optimization`: No meaningful optimization attempted
- `other`: None of the above - explain in discussion

### 4. Optimization Techniques

What optimization technique(s) did each use? (Select all that apply for each)

Techniques:
- `algorithmic`: Better algorithm/data structure (O(n²) → O(n log n))
- `memory_optimization`: Reduced allocations, better memory layout, caching
- `parallelization`: Threading, vectorization, GPU offload
- `api_library`: Using faster API calls or optimized libraries
- `lazy_computation`: Deferred/avoided unnecessary computation (lazy loading, generator usage, etc.)
- `compiler_optimization`: Torch.compile, JIT, fusion, etc.
- `batching`: Combined operations to reduce overhead
- `low_level`: Assembly, CUDA kernels, intrinsics
- `other`: Describe in discussion

### 5. Approach Comparison

How does the agent's approach compare to the human's?

Categories:
- `same_approach`: Essentially the same solution
- `similar_approach`: Same strategy, different implementation details
- `valid_alternative`: Different but potentially valid optimization
- `partial_solution`: Addresses part of the optimization
- `ineffective`: Changes unlikely to improve performance
- `harmful`: Changes likely to hurt performance or correctness
- `other`: None of the above - explain in discussion

### 6. Speedup Likelihood

Based on the patch analysis, what is the likely performance impact compared to human's verified speedup?

Categories:
- `likely_similar`: Probably achieves similar speedup to human
- `likely_partial`: Probably achieves some but not full speedup
- `uncertain`: Cannot determine without benchmarking
- `likely_ineffective`: Probably no meaningful speedup
- `likely_regression`: May cause performance regression
- `other`: None of the above - explain in discussion

### 7. Failure Mode

If the agent's solution differs from human's, what went wrong?

Categories:
- `localization_failure`: Misidentified the bottleneck or wrong abstraction level
- `technique_mismatch`: Right target, wrong optimization technique
- `incomplete_implementation`: Right idea, incomplete execution
- `complexity_avoidance`: Avoided necessary low-level optimizations
- `overcomplicated`: Added unnecessary complexity
- `library_misuse`: Incorrect usage of a specific library (e.g. torch, cuda)
- `not_applicable`: Agent solution is valid/successful
- `other`: None of the above - explain in discussion

### 8. Library Responsibility (If Failure)

If the agent failed or produced a failing patch, identify if a specific library was responsible or misused (e.g. "pytorch", "cuda", "triton").
Provide the library name and the specific reason.

---

## Output Format

Return your analysis as a JSON object. Every field with "discussion" is REQUIRED and must contain meaningful explanation.

{{
  "task_analysis": {{
    "domain": "compute|memory|io|concurrency|other",
    "complexity": "low|medium|high|extreme",
    "description": "Brief description of task challenges"
  }},
  "bottleneck_target": {{
    "category": "same_target|related_target|different_target|no_optimization|other",
    "human_target": "Brief description of what the human patch optimizes",
    "agent_target": "Brief description of what the agent patch optimizes",
    "discussion": "REQUIRED: Detailed explanation of bottleneck comparison"
  }},
  "optimization_techniques": {{
    "human_techniques": ["technique1", "technique2"],
    "agent_techniques": ["technique1"],
    "technique_overlap": true,
    "discussion": "REQUIRED: Explanation of techniques used and comparison"
  }},
  "approach_comparison": {{
    "category": "same_approach|similar_approach|valid_alternative|partial_solution|ineffective|harmful|other",
    "discussion": "REQUIRED: Why this category, key differences between approaches"
  }},
  "speedup_likelihood": {{
    "category": "likely_similar|likely_partial|uncertain|likely_ineffective|likely_regression|other",
    "discussion": "REQUIRED: Reasoning for this assessment, what would need benchmarking to verify"
  }},
  "failure_mode": {{
    "category": "localization_failure|technique_mismatch|incomplete_implementation|complexity_avoidance|overcomplicated|not_applicable|other",
    "discussion": "REQUIRED: What went wrong (if applicable), or why the solution is valid"
  }},
  "observations": {{
    "key_differences": ["Specific difference 1", "Specific difference 2"],
    "agent_strengths": ["Strength 1", "Strength 2"],
    "agent_weaknesses": ["Weakness 1", "Weakness 2"],
    "benchmark_needed": "What specific benchmark would verify performance claims"
  }},
  "library_failure": {{
    "responsible_libraries": ["lib1", "lib2"],
    "failure_reason": "Explanation of how the library was involved in the failure"
  }}
}}

Be specific and evidence-based. Reference actual code changes from the patches in your discussion.
```

---

## Appendix C: Verbatim Aggregation Prompt

Built dynamically by `build_summary_prompt()` in `prompts.py`. The template:

```text
# Aggregate Analysis of Agent Benchmark Runs

You are summarizing the results of multiple agent benchmark runs for academic research.

## Run Analyses

### Run {i}: {item_id}
- Agent: {agent} / Model: {model}
- Status: {status}
- Duration: {duration:.1f}s
- Scores: Understanding={code_understanding:.1f}, Alignment={task_alignment:.1f}, Approach={approach_quality:.1f}, Execution={execution_quality:.1f}

[repeated for each run]

## Instructions

Provide an aggregate analysis including:
1. Overall success rate and patterns
2. Common failure modes
3. Agent/model comparison
4. Key insights for paper

Return as JSON with structure:
{
  "summary": "...",
  "success_rate": 0.0,
  "avg_scores": {...},
  "common_failures": [...],
  "agent_comparison": {...},
  "model_comparison": {...},
  "key_insights": [...]
}
```

---

## Appendix D: Real Example Output — `sglang_021f76e4` (Claude Code / Sonnet 4.5)

### metrics_summary.json

```json
{
  "run": {
    "repo": "sglang",
    "agent": "claude_code",
    "model": "default",
    "item_id": "sglang_000_021f76e4",
    "task_id": "sglang_core"
  },
  "execution": {
    "status": "success",
    "duration_s": 167.8,
    "steps": 0,
    "tool_calls": 9,
    "errors": 0
  },
  "tool_breakdown": {
    "bash": 4,
    "editor": 2,
    "read": 2,
    "search": 1,
    "other": 0
  },
  "patch": {
    "generated": true,
    "lines_added": 11,
    "lines_removed": 9,
    "files_changed": 2,
    "compliance_ok": true
  },
  "scores": {
    "code_understanding": 8.0,
    "task_alignment": 9.0,
    "approach_quality": 8.0,
    "execution_quality": 8.0,
    "overall": 8.25
  },
  "classification": {
    "approach": "direct_edit",
    "tool_pattern": "balanced",
    "failure_reason": "none"
  },
  "insights": {
    "summary": "A very efficient run where the agent followed instructions to the letter. It successfully applied the requested optimization patterns to the target files, demonstrating good understanding of PyTorch memory allocation nuances.",
    "strengths": [
      "High speed of execution",
      "Strict adherence to constraints",
      "Correct handling of edge cases when switching to uninitialized memory (seg_indptr[0])"
    ],
    "weaknesses": [
      "Lack of empirical performance verification in the logs",
      "Relatively narrow focus on the provided example pattern"
    ]
  },
  "patch_similarity": {
    "human_patch_available": true,
    "file_overlap_pct": 100.0,
    "line_overlap_pct": 5.2,
    "approach_similarity": 4.2
  }
}
```

### llm_raw_scores.json

```json
{
  "scores": {
    "code_understanding": 8.0,
    "task_alignment": 9.0,
    "approach_quality": 8.0,
    "execution_quality": 8.0,
    "overall": 8.25
  },
  "sub_scores": {
    "code_understanding": {
      "repository_navigation": 8.0,
      "function_identification": 9.0,
      "dependency_awareness": 7.0
    },
    "task_alignment": {
      "goal_comprehension": 10.0,
      "constraint_adherence": 9.0,
      "output_relevance": 9.0
    },
    "approach_quality": {
      "strategy_coherence": 9.0,
      "exploration_efficiency": 9.0,
      "decision_quality": 7.0
    },
    "execution_quality": {
      "edit_correctness": 9.0,
      "test_verification": 6.0,
      "error_recovery": 5.0
    }
  },
  "categorical": {
    "approach_category": "direct_edit",
    "tool_usage_pattern": "balanced",
    "failure_category": "none"
  }
}
```

### patch_quality.json (V4 categorical assessment)

```json
{
  "human_patch_available": true,
  "analysis_model": "google/gemini-3-flash-preview",
  "analysis_tokens": 7865,
  "task_analysis": {
    "domain": "memory",
    "complexity": "high",
    "description": "The task involves optimizing the LoRA (Low-Rank Adaptation) management path in a high-performance LLM serving engine (sglang). The primary challenges involve reducing host-to-device (H2D) synchronization overhead, optimizing tensor allocation patterns, and managing memory pool eviction logic to minimize latency during the forward pass."
  },
  "library_failure": {
    "responsible_libraries": ["pytorch"],
    "failure_reason": "The agent failed to utilize advanced PyTorch features like pinned memory (pin_memory=True) and non-blocking transfers (non_blocking=True) which are standard for optimizing host-to-device bottlenecks."
  },
  "bottleneck_target": {
    "category": "same_target",
    "human_target": "Optimizes the LoRA metadata preparation path by using pinned memory for asynchronous H2D transfers and avoiding redundant tensor operations in the CUDA graph path.",
    "agent_target": "Optimizes tensor initialization in the LoRA manager and memory pool by switching from zeros to empty and fixing minor initialization logic.",
    "discussion": "Both patches target the LoRA manager and memory pool. However, the human patch identifies a deeper bottleneck: the synchronous nature of metadata transfers and redundant calculations of sequence lengths. The agent focuses on the surface-level suggestion provided in the prompt (torch.zeros to torch.empty)."
  },
  "optimization_techniques": {
    "human_techniques": ["memory_optimization", "parallelization", "lazy_computation"],
    "agent_techniques": ["memory_optimization"],
    "technique_overlap": true,
    "discussion": "The human uses pinned memory and non-blocking copies to overlap compute with data transfer. The agent strictly follows the 'torch.zeros -> torch.empty' pattern suggested in the example optimization diff, which is a micro-optimization compared to the human's architectural changes."
  },
  "approach_comparison": {
    "category": "partial_solution",
    "discussion": "The agent's approach is a subset of the human's. While the agent correctly identifies that `torch.empty` is faster than `torch.zeros` for buffers that are immediately overwritten, it misses the critical optimization of using pinned memory for the metadata transfer. The agent also fails to optimize the `max_len` calculation which the human moved to CPU to avoid a device-to-host sync."
  },
  "speedup_likelihood": {
    "category": "likely_ineffective",
    "discussion": "The agent's changes (zeros to empty) provide negligible speedup in the context of Python-based model management. The human's use of `non_blocking=True` and pinned memory addresses the actual overhead of GPU synchronization, which is orders of magnitude more significant than the difference between `zeros` and `empty` for small metadata tensors."
  },
  "failure_mode": {
    "category": "complexity_avoidance",
    "discussion": "The agent followed the provided example optimization pattern too literally. It successfully applied the 'zeros to empty' transformation but failed to perform the deeper analysis required to implement the asynchronous transfer logic (pinned memory) that the human expert identified as the primary bottleneck."
  },
  "observations": {
    "key_differences": [
      "Human implemented an asynchronous transfer mechanism using pinned memory.",
      "Human optimized the memory pool eviction logic to be more efficient.",
      "Agent focused almost exclusively on replacing torch.zeros with torch.empty.",
      "Agent added a fill_(0) in the memory pool which might actually be slower than the original assignment."
    ],
    "agent_strengths": [
      "Correctly identified all locations where torch.zeros was used for temporary buffers.",
      "Maintained functional correctness."
    ],
    "agent_weaknesses": [
      "Over-reliance on the provided example optimization pattern.",
      "Missed the synchronization bottleneck in the LoRA metadata path.",
      "Did not implement the pinned memory optimization."
    ],
    "benchmark_needed": "A benchmark measuring the latency of the LoRA forward pass with a high number of unique adapters per batch to highlight H2D transfer overhead."
  }
}
```

---

## Appendix E: Prompt Construction Code

### `build_analysis_prompt()` — populates the main analysis template

```python
def build_analysis_prompt(data: Dict[str, Any]) -> str:
    metadata = data.get("metadata", {})
    journal = data.get("journal", {})
    prompt_data = data.get("prompt", {})
    diff_targets = data.get("diff_targets", {})
    run_summary = data.get("run_summary", {})

    agent_type = metadata.get("agent", "unknown")
    agent_info = journal.get(agent_type, {})

    # Format constraints as bullet list
    constraints = prompt_data.get("constraints", [])
    constraints_str = "\n".join(f"- {c}" for c in constraints) if constraints else "No specific constraints provided."

    # Format target files as bullet list
    target_files = prompt_data.get("target_files", [])
    target_files_str = "\n".join(f"- `{f}`" for f in target_files) if target_files else "No specific target files provided."

    diff_targets_str = json.dumps(diff_targets, indent=2) if diff_targets else "{}"

    metrics = journal.get("metrics", {})
    agent_metrics = run_summary.get("agent", {})
    patch_stats = agent_metrics.get("patch_stats", {})

    # Truncation logic
    stdout = data.get("stdout", "")
    if len(stdout) > 500000:  # 500KB
        half = 250000
        stdout = stdout[:half] + "\n\n... [TRUNCATED] ...\n\n" + stdout[-half:]

    stderr = data.get("stderr", "")
    if len(stderr) > 50000:  # 50KB
        stderr = stderr[:50000] + "\n\n... [TRUNCATED] ..."

    patch_content = data.get("patch", "No patch generated.")
    if len(patch_content) > 100000:  # 100KB
        patch_content = patch_content[:100000] + "\n\n... [TRUNCATED DUE TO SIZE] ..."

    # Format with all fields
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        task_content=data.get("task", prompt_data.get("description", "No task description available.")),
        agent_type=agent_type,
        model_full=metadata.get("model_full", metadata.get("model", "unknown")),
        repo=metadata.get("repo", "unknown"),
        time_budget_minutes=agent_info.get("time_budget_minutes", 120),
        task_id=metadata.get("task_id", journal.get("task_id", "unknown")),
        commit_pre=metadata.get("commits", {}).get("pre", "unknown"),
        commit_human=metadata.get("commits", {}).get("human", "unknown"),
        constraints=constraints_str,
        target_files=target_files_str,
        stdout_content=stdout or "No stdout captured.",
        stderr_content=stderr or "No stderr captured.",
        patch_content=patch_content,
        diff_targets=diff_targets_str,
        status=journal.get("status", "unknown"),
        duration_s=agent_info.get("duration_s", 0) or 0,
        time_to_first_edit_s=metrics.get("time_to_first_edit_s", 0) or 0,
        patch_generated=metrics.get("patch_size_loc", 0) > 0 or agent_metrics.get("patch_generated", False),
        files_changed_count=metrics.get("changed_files_count", 0) or patch_stats.get("files_changed", 0),
        lines_added=patch_stats.get("lines_added", 0),
        lines_removed=patch_stats.get("lines_removed", 0),
        returncode=agent_info.get("returncode", 0),
    )
    return prompt
```

### `build_patch_quality_prompt()` — populates the V4 patch quality template

```python
def build_patch_quality_prompt(
    task_description: str,
    human_patch: str,
    agent_patch: str,
    rule_based_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    # Format rule-based metrics as key-value lines
    if rule_based_metrics:
        metrics_str = "\n".join(
            f"- {key}: {value}"
            for key, value in rule_based_metrics.items()
            if key not in ["human_patch_available", "human_patch_source"]
        )
    else:
        metrics_str = "No pre-computed metrics available."

    # Truncate patches to 50KB each
    max_patch_len = 50000
    if len(human_patch) > max_patch_len:
        human_patch = human_patch[:max_patch_len] + "\n\n... [TRUNCATED - patch too long] ..."
    if len(agent_patch) > max_patch_len:
        agent_patch = agent_patch[:max_patch_len] + "\n\n... [TRUNCATED - patch too long] ..."

    prompt = PATCH_QUALITY_PROMPT_TEMPLATE.format(
        task_description=task_description or "No task description provided.",
        human_patch=human_patch or "No human patch available.",
        agent_patch=agent_patch or "No agent patch generated.",
        rule_based_metrics=metrics_str,
    )
    return prompt
```
