# Plan: Redesign Qualitative Metrics with Patch-Centric Analysis

## Problem Statement

The current qualitative metrics system has several issues:
1. **Wrong field name**: `load_human_patch_from_dataset` looks for `patch` but dataset uses `diff_text`
2. **Vague metrics**: Abstract questions like "code understanding" are unreliable
3. **Massive context**: 500KB stdout sent to LLM (wasteful, noisy)
4. **No ground truth**: Quality scores aren't compared to human reference
5. **Expensive**: ~50K tokens per analysis

## Proposed Solution: Patch-Centric Quality Metrics

### Core Insight
**We have ground truth** - the human patch. All quality assessment should be relative to this reference.

---

## New Architecture

### Tier 1: Rule-Based Metrics (No LLM, 100% Reproducible)

Already exists in `patch_similarity.json`, keep as-is but fix bugs:
- `file_overlap_pct`: % of human files also modified by agent
- `line_overlap_pct`: % of human line changes matched by agent
- `approach_similarity_score`: Code pattern similarity (0-10)
- Diff size metrics: lines added/removed comparison

### Tier 2: LLM-Based Patch Analysis (NO SCORING - Categories & Discussion Only)

**Inspired by GSO Benchmark (arxiv:2505.23671v3)**

Since we don't run benchmarks, we cannot measure actual speedup. We focus on **categorical assessment** and **expert discussion** rather than fake scores.

**Minimal context sent to LLM (~2-5K tokens):**
```
<task_description>
Brief optimization task (50-100 words)
</task_description>

<human_patch>
The reference solution (diff)
</human_patch>

<agent_patch>
The agent's solution (diff)
</agent_patch>

<rule_based_metrics>
- file_overlap_pct: X%
- line_overlap_pct: Y%
- agent_extra_files: [list]
- agent_extra_lines: N
</rule_based_metrics>
```

---

#### 1. Bottleneck Target (categorical)

Does the agent target the same performance bottleneck as the human?

| Category | Description |
|----------|-------------|
| `same_target` | Optimizes the exact same bottleneck/function |
| `related_target` | Optimizes related code that affects same performance path |
| `different_target` | Optimizes something else entirely |
| `no_optimization` | No meaningful optimization attempted |
| `other` | See discussion |

**+ Discussion field**: Free-form explanation of what bottleneck each targets.

---

#### 2. Optimization Technique (categorical, multi-select)

What optimization technique(s) did each use?

| Technique | Description |
|-----------|-------------|
| `algorithmic` | Better algorithm/data structure (O(n²) → O(n log n)) |
| `memory_optimization` | Reduced allocations, better memory layout, caching |
| `parallelization` | Threading, vectorization, GPU offload |
| `api_library` | Using faster API calls or optimized libraries |
| `lazy_computation` | Deferred/avoided unnecessary computation |
| `batching` | Combined operations to reduce overhead |
| `low_level` | Assembly, CUDA kernels, intrinsics |
| `other` | Describe in discussion |

Output:
```json
{
  "human_techniques": ["memory_optimization", "batching"],
  "agent_techniques": ["memory_optimization"],
  "technique_overlap": true,
  "discussion": "Both focus on reducing tensor allocations, but human also batches..."
}
```

---

#### 3. Approach Comparison (categorical)

How does the agent's approach compare to human's?

| Category | Description |
|----------|-------------|
| `same_approach` | Essentially the same solution |
| `similar_approach` | Same strategy, different implementation details |
| `valid_alternative` | Different but potentially valid optimization |
| `partial_solution` | Addresses part of the optimization |
| `ineffective` | Changes unlikely to improve performance |
| `harmful` | Changes likely to hurt performance or correctness |
| `other` | See discussion |

**+ Discussion field**: Explain the key differences and why categorized this way.

---

#### 4. Speedup Likelihood (opinion, NOT a score)

Based on the patch analysis, what is the likely performance impact?

| Category | Description |
|----------|-------------|
| `likely_similar` | Probably achieves similar speedup to human |
| `likely_partial` | Probably achieves some but not full speedup |
| `uncertain` | Cannot determine without benchmarking |
| `likely_ineffective` | Probably no meaningful speedup |
| `likely_regression` | May cause performance regression |
| `other` | See discussion |

**+ Discussion field**: Explain reasoning, what would need to be measured.

---

#### 5. Failure Mode (if agent didn't match human, categorical)

Based on GSO paper's failure taxonomy:

| Category | Description |
|----------|-------------|
| `localization_failure` | Misidentified the bottleneck or wrong abstraction level |
| `technique_mismatch` | Right target, wrong optimization technique |
| `incomplete_implementation` | Right idea, incomplete execution |
| `complexity_avoidance` | Avoided necessary low-level optimizations |
| `overcomplicated` | Added unnecessary complexity |
| `not_applicable` | Agent solution is valid/successful |
| `other` | See discussion |

**+ Discussion field**: Detailed analysis of what went wrong.

---

#### 6. Key Observations (free-form)

Structured discussion fields:

```json
{
  "key_differences": [
    "Human uses torch.empty() while agent uses torch.zeros()",
    "Human pre-computes seg_lens, agent computes per-batch"
  ],
  "agent_strengths": [
    "Cleaner code structure",
    "Better error handling"
  ],
  "agent_weaknesses": [
    "Missed the pinned memory optimization",
    "Extra synchronization points"
  ],
  "would_need_benchmark": "To verify if the agent's approach achieves similar speedup, need to run LoRA batch initialization benchmarks with varying batch sizes."
}
```

---

## Output Schema

### New file: `patch_quality.json`

```json
{
  "human_patch_available": true,
  "analysis_model": "google/gemini-3-pro-preview",
  "analysis_tokens": 2340,

  "bottleneck_target": {
    "category": "same_target",
    "human_target": "LoRA batch initialization in lora_manager.py",
    "agent_target": "LoRA batch initialization in lora_manager.py",
    "discussion": "Both patches target the same bottleneck: reducing CUDA stream synchronizations during LoRA batch setup."
  },

  "optimization_techniques": {
    "human_techniques": ["memory_optimization", "lazy_computation", "batching"],
    "agent_techniques": ["memory_optimization", "lazy_computation"],
    "technique_overlap": true,
    "discussion": "Both use torch.empty() instead of torch.zeros() to avoid initialization overhead. Human additionally pre-computes seg_lens/seg_indptr outside the hot path and adds pinned memory transfers."
  },

  "approach_comparison": {
    "category": "partial_solution",
    "discussion": "Agent correctly identified the torch.zeros() → torch.empty() optimization but missed the transfer_adapter_info helper that enables async host-to-device transfers using pinned memory."
  },

  "speedup_likelihood": {
    "category": "likely_partial",
    "discussion": "The empty() optimization will help, but without the pinned memory async transfers, the agent's solution likely achieves 30-50% of the human's TTFT improvement. Would need to benchmark with varying LoRA batch sizes to confirm."
  },

  "failure_mode": {
    "category": "incomplete_implementation",
    "discussion": "Agent understood the core issue (unnecessary initializations) but stopped short of the full solution. Didn't implement the async transfer pattern that eliminates remaining sync points."
  },

  "observations": {
    "key_differences": [
      "Human adds transfer_adapter_info() helper for async pinned memory transfers",
      "Human pre-computes seg_lens/seg_indptr in init_cuda_graph_batch_info()",
      "Agent only does the torch.zeros() → torch.empty() conversion"
    ],
    "agent_strengths": [
      "Correctly identified the initialization overhead issue",
      "Clean, minimal changes"
    ],
    "agent_weaknesses": [
      "Missed the async transfer optimization (main speedup source)",
      "Didn't recognize the CUDA sync elimination opportunity"
    ],
    "benchmark_needed": "Run sglang bench_serving with --lora-name to measure TTFT/ITL improvement"
  }
}
```

### Key Design Principles

1. **No numeric scores** - Categories + discussion only
2. **Every category has "other" option** - With required discussion when selected
3. **Discussion fields are mandatory** - Forces explanation, not just labels
4. **References rule-based metrics** - Grounds the analysis in measurable data
5. **Acknowledges uncertainty** - "likely_partial", "would need benchmark"

---

## Implementation Tasks

### Phase 1: Fix Data Loading Bugs

**File: `patch_comparator.py`**

1. Fix `load_human_patch_from_dataset()` to check `diff_text` field (not just `patch`)
   ```python
   # Current (broken):
   patch = item.get("patch", "")

   # Fixed:
   patch = item.get("patch", "") or item.get("diff_text", "")
   ```

2. Fix commit hash lookup in `analyzer.py`:
   ```python
   # Current (problematic):
   human_commit = commits.get("human", "") or commits.get("head", "")

   # Fixed:
   human_commit = commits.get("human", "")
   ```

### Phase 2: New Prompt Template

**File: `prompts.py`** - Add new function:

```python
def build_patch_quality_prompt(
    task_description: str,
    human_patch: str,
    agent_patch: str,
    context: Optional[str] = None,
) -> str:
    """Build minimal prompt for patch quality assessment."""
```

~100 lines vs current 260 lines. Focus on concrete comparison questions.

### Phase 3: New Schema

**File: `schemas.py`** - Add new models:

```python
# Categorical enums - all include "other" option
class BottleneckTargetCategory(str, Enum):
    SAME_TARGET = "same_target"
    RELATED_TARGET = "related_target"
    DIFFERENT_TARGET = "different_target"
    NO_OPTIMIZATION = "no_optimization"
    OTHER = "other"

class OptimizationTechnique(str, Enum):
    ALGORITHMIC = "algorithmic"
    MEMORY_OPTIMIZATION = "memory_optimization"
    PARALLELIZATION = "parallelization"
    API_LIBRARY = "api_library"
    LAZY_COMPUTATION = "lazy_computation"
    BATCHING = "batching"
    LOW_LEVEL = "low_level"
    OTHER = "other"

class ApproachCategory(str, Enum):
    SAME_APPROACH = "same_approach"
    SIMILAR_APPROACH = "similar_approach"
    VALID_ALTERNATIVE = "valid_alternative"
    PARTIAL_SOLUTION = "partial_solution"
    INEFFECTIVE = "ineffective"
    HARMFUL = "harmful"
    OTHER = "other"

class SpeedupLikelihood(str, Enum):
    LIKELY_SIMILAR = "likely_similar"
    LIKELY_PARTIAL = "likely_partial"
    UNCERTAIN = "uncertain"
    LIKELY_INEFFECTIVE = "likely_ineffective"
    LIKELY_REGRESSION = "likely_regression"
    OTHER = "other"

class FailureMode(str, Enum):
    LOCALIZATION_FAILURE = "localization_failure"
    TECHNIQUE_MISMATCH = "technique_mismatch"
    INCOMPLETE_IMPLEMENTATION = "incomplete_implementation"
    COMPLEXITY_AVOIDANCE = "complexity_avoidance"
    OVERCOMPLICATED = "overcomplicated"
    NOT_APPLICABLE = "not_applicable"
    OTHER = "other"


# Analysis components - category + discussion (NO SCORES)
class BottleneckTargetAnalysis(BaseModel):
    category: BottleneckTargetCategory
    human_target: str = Field(description="What bottleneck the human patch targets")
    agent_target: str = Field(description="What bottleneck the agent patch targets")
    discussion: str = Field(description="Explanation of bottleneck comparison")

class OptimizationTechniqueAnalysis(BaseModel):
    human_techniques: List[OptimizationTechnique]
    agent_techniques: List[OptimizationTechnique]
    technique_overlap: bool
    discussion: str

class ApproachComparisonAnalysis(BaseModel):
    category: ApproachCategory
    discussion: str = Field(description="Why this category, key differences")

class SpeedupLikelihoodAnalysis(BaseModel):
    category: SpeedupLikelihood
    discussion: str = Field(description="Reasoning, what would need benchmarking")

class FailureModeAnalysis(BaseModel):
    category: FailureMode
    discussion: str = Field(description="What went wrong, if applicable")

class PatchObservations(BaseModel):
    key_differences: List[str]
    agent_strengths: List[str]
    agent_weaknesses: List[str]
    benchmark_needed: str = Field(description="What benchmark would verify performance")


# Main output schema
class PatchQualityAnalysis(BaseModel):
    """LLM-based patch quality analysis - categories and discussion only, no scores."""

    human_patch_available: bool
    analysis_model: str
    analysis_tokens: int

    bottleneck_target: BottleneckTargetAnalysis
    optimization_techniques: OptimizationTechniqueAnalysis
    approach_comparison: ApproachComparisonAnalysis
    speedup_likelihood: SpeedupLikelihoodAnalysis
    failure_mode: FailureModeAnalysis
    observations: PatchObservations
```

### Phase 4: Update Analyzer

**File: `analyzer.py`**

1. Add new method `_analyze_patch_quality()`:
   - Takes human_patch, agent_patch, task_description
   - Optional: include_context flag for original code
   - Builds minimal prompt
   - Calls LLM
   - Parses into `PatchQualityMetrics`

2. Call from `analyze_run()` after `_compare_patches()`

3. Save output to `patch_quality.json`

**File: CLI (if exists) or analyzer.py args**

Add configurable options:
```python
# In analyze_run() or CLI
include_code_context: bool = False  # Default: patches only
```

### Phase 5: Integration

The new `patch_quality.json` is an **additional** output that complements existing analysis:

```
state/analysis/{repo}/{agent}/{model}/{timestamp}/{item_id}/
├── analysis.json           # Complete analysis (existing)
├── llm_analysis.json       # Existing LLM analysis (keep as-is)
├── llm_raw_scores.json     # Existing scores (keep as-is)
├── patch_similarity.json   # Rule-based comparison (fix bugs)
├── patch_quality.json      # NEW: LLM-based patch quality assessment
└── metrics_summary.json    # Updated to include new metrics
```

- Keep all existing files unchanged
- Add `patch_quality.json` as new output
- Update `metrics_summary.json` to include patch quality scores
- Update `RunAnalysis` schema to include optional `patch_quality` field

---

## Files to Modify

| File | Changes |
|------|---------|
| `patch_comparator.py` | Fix `diff_text` field lookup |
| `analyzer.py` | Fix commit lookup, add `_analyze_patch_quality()` |
| `prompts.py` | Add `build_patch_quality_prompt()` |
| `schemas.py` | Add `PatchQualityAnalysis` and related enums/models |
| `README.md` | Document new output file |

---

## Benefits

| Metric | Old System | New System |
|--------|------------|------------|
| Context size | ~50K tokens | ~2-5K tokens |
| Cost per analysis | High | ~10-25x cheaper |
| Reproducibility | Low (vague questions) | High (specific comparisons) |
| Ground truth | None | Human patch reference |
| Interpretability | Abstract scores | Concrete comparisons |

---

## User Decisions

1. **Backwards compatibility**: Keep all existing files, add `patch_quality.json` as new output
2. **No human patch**: Keep as-is (skip LLM quality analysis for those runs)
3. **Code context**: Make it configurable via CLI flag (default: patches only)
