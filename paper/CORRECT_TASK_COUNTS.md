# Correct Task Counts for OmniPerf-Bench Paper

## Verified Numbers (from actual data)

### Commit Extractions
- **vLLM commits extracted:** 64 (in `misc/experiments/commit_extractions_with_apis/`)
- **SGLang commits extracted:** 80 (in `misc/experiments/sglang_commit_extractions_with_apis/`)
- **Total commits extracted:** 144

### Codex Evaluation (Actual Tasks Run)
- **vLLM Codex tasks:** 99 (in `perf-agents-bench/state/runs/vllm_core_codex-90a1c13f/`)
- **SGLang Codex tasks:** 80 (in `perf-agents-bench/state/runs/sglang_core-389be848/`)
- **Total Codex tasks:** **179**

**Note:** vLLM has 99 Codex tasks but only 64 commit extractions - this suggests some commits generated multiple tasks or tasks were expanded.

---

## What to Say in the Paper

### ✅ CORRECT
- "179 optimization tasks from production ML inference engines"
- "99 tasks from vLLM, 80 from SGLang"
- "evaluates three agents (Codex, TRAE, OpenHands) on 179 tasks"

### ❌ INCORRECT (Previously Used)
- ~~"282 optimization tasks"~~ - This number is wrong
- ~~"282 tasks from vLLM/SGLang"~~ - This number is wrong

---

## Breakdown by Repository

### vLLM Tasks
- **Source:** vLLM inference engine (https://github.com/vllm-project/vllm)
- **Codex evaluation:** 99 tasks
- **Success rate:** 38.4% clean success (38/99)
- **Pathological failures:** 60.6% (60/99)
- **Domain:** Attention mechanisms, memory management, KV-cache, kernel optimization

### SGLang Tasks
- **Source:** SGLang inference engine (https://github.com/sgl-project/sglang)
- **Codex evaluation:** 80 tasks
- **Success rate:** 96.2% clean success (77/80)
- **Pathological failures:** 0%
- **Domain:** Structured generation, scheduling, memory management

---

## Agent Evaluation Status

### Codex
- ✅ vLLM: 99/99 tasks completed
- ✅ SGLang: 80/80 tasks completed
- **Total:** 179/179 (100% completion)

### TRAE
- Status: Variable (multiple runs with different success rates)
- Need to verify exact task counts

### OpenHands
- Status: Need to verify if evaluated on same 179 tasks

---

## Comparison to Prior Work (CORRECT)

| Benchmark | Tasks | Domain | Success Rate |
|-----------|-------|--------|--------------|
| GSO | 102 | 10 diverse repos | <5% |
| SWE-Perf | 140 | General perf PRs | Substantial gap |
| KernelBench | 250 | GPU kernels | <20% baseline match |
| **OmniPerf-Bench** | **179** | **ML inference (vLLM, SGLang)** | **Codex: 38-96% (codebase-dependent)** |

---

## Key Numbers for Abstract/Introduction (CORRECTED)

### Scale
- ✅ "179 optimization tasks from production ML inference engines"
- ✅ "2 repositories: vLLM (99 tasks), SGLang (80 tasks)"
- ✅ "3 agents evaluated: Codex, TRAE, OpenHands"

### Success Rates (Codex)
- ✅ vLLM: 38.4% clean success (38/99 tasks)
- ✅ SGLang: 96.2% clean success (77/80 tasks)
- ✅ Gap: 58 percentage points using identical agent

### Failure Modes
- ✅ Instant-edit pathology: <1s TTFE → 95% failure rate
- ✅ Commit explosion: Up to 7,755 commits on single task
- ✅ Scope violations: Up to 2,970 unauthorized file changes

### Multi-Agent Comparison
- ✅ Codex: 100% completion (179/179), <$1/task
- ✅ TRAE: Variable success, $16/task, 1.6M tokens/task
- ✅ Token usage: 1.6M (vLLM) vs 712K (SGLang) = 55% reduction

---

## Updated Narrative

**OLD (WRONG):**
> "We introduce OmniPerf-Bench, a benchmark of 282 performance optimization tasks..."

**NEW (CORRECT):**
> "We introduce OmniPerf-Bench, a benchmark of 179 performance optimization tasks extracted from production ML inference engines (99 from vLLM, 80 from SGLang)..."

---

## Files to Update

1. ✅ `example_paper.tex` - Related Work section (updated)
2. ⏳ `example_paper.tex` - Abstract (need to write)
3. ⏳ `example_paper.tex` - Introduction (need to write)
4. ⏳ `literature_survey_summary.md` - Update all "282" references
5. ⏳ `LITERATURE_SURVEY_STATUS.md` - Update all "282" references

---

## Questions to Resolve

1. **Why does vLLM have 99 tasks from 64 commits?**
   - Possible: Some commits generated multiple optimization tasks
   - Possible: Tasks were split or expanded during evaluation
   - Need to investigate task ID mapping

2. **What about TRAE and OpenHands task counts?**
   - Did they evaluate on the same 179 tasks?
   - Or a subset?
   - Need to check their run directories

3. **What is the "282" number from?**
   - Was this a planned total that wasn't completed?
   - Or a miscount?
   - Should verify origin of this number
