# Soft + Hard Metrics: Complete Analysis Findings

## Core Insight: Both Metrics Are Necessary

Neither soft metrics nor hard metrics alone provide a complete picture of agent performance. They are **complementary** - each captures a different dimension of success.

| Metric Type | Measures | Limitation | Use For |
|-------------|----------|------------|---------|
| **Hard** | Does it work? | Can't tell if RIGHT thing was optimized | Validation |
| **Soft** | Did agent understand? | Can't tell if implementation is correct | Understanding |
| **Both** | TRUE SUCCESS | - | Complete evaluation |

---

## The Four Quadrants Framework

```
                        HARD METRIC
                   Beats    |    Worse
                -------------------------
    same/       |  TRUE     |  Understood but
    related     |  SUCCESS  |  buggy impl
    target      |           |
SOFT -----------|-----------|-------------
METRIC          |           |
                |  Lucky -  |  Complete
  ineffective/  |  didn't   |  Failure
  no_opt        |  solve    |
                |  task     |
```

### Quadrant 1: TRUE SUCCESS (Correct Target + Beats)
- Agent identified the bottleneck ✓
- Agent's fix actually works ✓

### Quadrant 2: Good Understanding, Bad Execution (Correct Target + Worse)
- Agent understood the problem ✓
- Implementation has bugs/anti-patterns ✗

### Quadrant 3: Lucky Win (Wrong Target + Beats)
- Agent missed the actual bottleneck ✗
- Random changes happened to help ✓

### Quadrant 4: Complete Failure (Wrong Target + Worse)
- Agent missed the bottleneck ✗
- Changes didn't help or made things worse ✗

---

## Detailed Case Studies

### Case Study 1: Quadrant 3 (Lucky Win)
**Commit:** e3580537 | **Hard Outcome:** BEATS

**Task:** Handle chunked prefill with prefix caching efficiently

**Human:** Modified scheduler and model runner to skip computation

**Agent:** Avoided list re-allocation, added safety checks

**Soft Metric Assessment:**
> "The agent's approach is largely ineffective for the specific performance goal. While it makes minor Python-level improvements (like avoiding list re-allocation), it fails to implement the logic necessary to handle chunked prefill with prefix caching."

**Agent Strengths:** "Identified some unnecessary list allocations", "Added safety checks for empty input lists"

**Why It Beat:** Benchmark measures total execution time. Micro-optimizations reduced overhead in hot paths without solving the actual problem.

**Verdict:** Task NOT solved. Benchmark improvement is coincidental.

---

### Case Study 2: Quadrant 3 (Lucky Win)
**Commit:** 6a417b86 | **Hard Outcome:** BEATS

**Task:** Fix Neuron backend cache efficiency

**Human:** Added +1 buffer to GPU blocks (critical architectural fix)

**Agent:** Reduced property lookups, combined assertions, cached integers

**Soft Metric Assessment:**
> "The agent's approach focuses on micro-optimizations like reducing property lookups and combining assertions. The human patch makes a functional change to the cache size (+1), which is likely critical for the Neuron backend's efficiency."

**Why It Beat:** Property caching in hot paths helped. Benchmark didn't specifically test where +1 buffer matters.

**Verdict:** Task NOT solved. The +1 buffer fix was completely missed.

---

### Case Study 3: Quadrant 2 (Good Understanding, Bad Execution)
**Commit:** fc7b8d1e | **Hard Outcome:** WORSE

**Task:** Optimize sequence block management

**Human:** Added fast-path using `is_single_seq` abstraction, used high-level Python constructs

**Agent:** Identified same bottleneck BUT replaced list concatenation with manual Python loop

**Soft Metric Assessment:**
> "The core of both solutions is identical... However, the agent replaced a high-level list concatenation with a manual Python loop, which is often slower in CPython."

**Agent Weakness:** "The manual loop is a classic 'anti-optimization' in Python"

**Why It's Worse:** Manual Python loops are slower than built-in list operations due to interpreter overhead.

**Verdict:** Correct understanding, flawed execution.

---

### Case Study 4: Quadrant 2 (Good Understanding, Bad Execution)
**Commit:** fe66b347 | **Hard Outcome:** WORSE

**Task:** Vectorize state management in Mamba model

**Human:** Used `torch.vmap` to vectorize state updates

**Agent:** Used standard PyTorch advanced indexing (valid approach) BUT used Python `any()` instead of `torch.any()`

**Soft Metric Assessment:**
> "The core strategy for both was replacing `for idx in tensor: state[idx].op()` with a single batched operation."

**Agent Weakness:** "The use of `any(has_initial_states)` is a Python built-in which can be slower than `torch.any()` on large tensors"

**Why It's Worse:** Python's `any()` forces GPU→CPU sync, negating vectorization gains.

**Verdict:** Correct strategy, wrong API choice.

---

### Case Study 5: Quadrant 1 (TRUE SUCCESS)
**Commit:** 015069b0 | **Hard Outcome:** SIMILAR

**Task:** Optimize reasoning content extraction by removing regex overhead

**Human:** Replaced regex with `str.partition`

**Agent:** Also replaced regex with string operations, additionally optimized streaming method, added token caching

**Soft Metric Assessment:**
> "Both patches correctly identify that the regex-based extraction is a bottleneck. The agent goes further by also attempting to optimize the streaming version."

**Agent Strengths:** "Comprehensive coverage", "Proactive caching of constants"

**Verdict:** TRUE SUCCESS - Agent understood and executed correctly.

---

## Aggregated Correlation Data

### Bottleneck Target vs Hard Outcome (n=101)

| Category | Total | Success Rate |
|----------|-------|--------------|
| same_target | 37 | **51.4%** |
| related_target | 41 | **63.4%** |
| different_target | 8 | **62.5%** |
| no_optimization | 15 | **60.0%** |

**Finding:** `same_target` has LOWEST success (51.4%). Agents targeting exact same bottleneck try to replicate human solution → implementation bugs.

---

### Approach Comparison vs Hard Outcome (n=101)

| Category | Total | Success Rate |
|----------|-------|--------------|
| similar_approach | 20 | **45.0%** |
| valid_alternative | 22 | **45.5%** |
| partial_solution | 29 | **75.9%** |
| ineffective | 24 | **62.5%** |

**Finding:** `partial_solution` wins (75.9%). Focused optimizations of real bottlenecks > attempting complete solutions with implementation risks.

---

### Weakness Patterns That Predict Failure

| Weakness Type | Occurrence | Worse Rate |
|---------------|------------|------------|
| **anti_pattern** | 4 | **75%** |
| architecture | 8 | 50% |
| over_engineering | 17 | 47% |
| wrong_api | 14 | 43% |
| missed_optimization | 108 | 35% |

**Finding:** Anti-patterns are the KILLER. Soft metrics identify them, hard metrics confirm 75% worse rate.

---

## Agent-Specific Analysis

| Agent | same_target | technique_overlap | Hard Success | TRUE SUCCESS |
|-------|-------------|-------------------|--------------|--------------|
| Claude Code | 34% | 76.7% | 57.1% | ~55% |
| Codex | 56% (best) | 94.5% (best) | 50.0% | ~57% |
| TRAE (Sonnet) | 17% | 40.5% | 54.3% | ~22% |
| TRAE (GPT) | 34% | 52.2% | 44.4% | ~38% |

**Codex:** Best understanding (56% same_target, 94.5% technique_overlap) but implementation bugs hurt hard performance.

**Claude Code:** Balanced - not best at identifying target, but fewer bugs → highest hard success.

**TRAE Sonnet:** 46% no_optimization (fails to even attempt), but competitive when it works.

---

## Conclusion

> Neither metric alone is sufficient.
>
> Hard metrics validate that performance improved.
>
> Soft metrics validate that the agent understood the task.
>
> TRUE SUCCESS requires both.

**Key Findings:**

1. **Soft metrics correctly identify task understanding** - "ineffective" means agent missed the real bottleneck

2. **Hard metrics validate but don't explain** - Benchmark improvement without correct targeting is coincidental (Lucky Win)

3. **Anti-patterns bridge both metrics** - Soft identifies them, hard confirms 75% worse rate

4. **partial_solution > similar_approach** - Focused optimizations (75.9%) beat replicating human (45%)

5. **Understanding ≠ Execution** - Codex has best understanding, Claude Code has best hard performance

---

*Analysis based on 101 matched commits across 4 agents using patch_quality.json, patch_similarity.json (soft) and HuggingFace benchmark (hard).*
