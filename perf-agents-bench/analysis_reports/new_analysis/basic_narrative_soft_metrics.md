# Soft + Hard Metrics: A Complementary Evaluation Framework

## Core Insight: Both Metrics Are Necessary

Neither soft metrics nor hard metrics alone provide a complete picture of agent performance. They are **complementary** - each captures a different dimension of success.

---

## What Each Metric Type Measures

### Hard Metrics
**What they CAN tell:** Did performance actually improve?
**What they CAN'T tell:** Did agent solve the RIGHT problem?

**Limitation:** Agent can beat benchmark via random micro-optimizations without understanding the actual task.

### Soft Metrics
**What they CAN tell:** Did agent target the correct bottleneck? Did agent understand the optimization goal?
**What they CAN'T tell:** Does the approach actually work in practice?

**Limitation:** Agent can have correct understanding but buggy implementation.

---

## Why BOTH Are Necessary

| Scenario | Soft Metric | Hard Metric | Interpretation |
|----------|-------------|-------------|----------------|
| **TRUE SUCCESS** | same_target / partial_solution | beats / similar | Agent understood AND executed correctly |
| **Lucky Win** | ineffective | beats | Agent got lucky with micro-opts, didn't solve actual task |
| **Good Intent, Bad Execution** | similar_approach | worse | Agent understood but introduced bugs |
| **Complete Failure** | no_optimization | worse | Agent failed entirely |

---

## The Four Quadrants

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

---

## What Each Quadrant Tells Us

### Quadrant 1: TRUE SUCCESS (Correct Target + Beats)
- Agent identified the bottleneck ✓
- Agent's fix actually works ✓
- **This is what we want**

### Quadrant 2: Good Understanding, Bad Execution (Correct Target + Worse)
- Agent understood the problem ✓
- Implementation has bugs/anti-patterns ✗
- **Problem:** Execution quality, not understanding

### Quadrant 3: Lucky Win (Wrong Target + Beats)
- Agent missed the actual bottleneck ✗
- Random changes happened to help ✓
- **Problem:** Task not actually solved, benchmark gamed

### Quadrant 4: Complete Failure (Wrong Target + Worse)
- Agent missed the bottleneck ✗
- Changes didn't help or made things worse ✗
- **Problem:** Everything

---

## Case Studies

### Case 1: Commit e3580537 - "BEATS" but in Quadrant 3 (Lucky Win)

**Task:** Handle chunked prefill with prefix caching

**Human Solution:** Modified scheduler and model runner to skip computation

**Agent Solution:** Avoided list re-allocation, added safety checks

**Soft Metric (Correct Assessment):**
> "The agent's approach is largely ineffective... it fails to implement the logic necessary to handle chunked prefill with prefix caching."

**Hard Metric:** BEATS

**Reality:** Agent didn't solve the chunked prefill problem. Micro-optimizations coincidentally improved total execution time. The actual task was NOT completed.

---

### Case 2: Commit 6a417b86 - "BEATS" but in Quadrant 3 (Lucky Win)

**Task:** Fix Neuron backend cache efficiency

**Human Solution:** Added +1 buffer to GPU blocks (critical architectural fix)

**Agent Solution:** Reduced property lookups, combined assertions, cached integers

**Soft Metric (Correct Assessment):**
> "The agent's approach focuses on micro-optimizations... The human patch makes a functional change to the cache size (+1), which is likely critical for the Neuron backend's efficiency."

**Hard Metric:** BEATS

**Reality:** Agent completely missed the +1 buffer fix. Micro-optimizations are noise that happened to improve the benchmark.

---

### Case 3: Commit fc7b8d1e - "similar_approach" but WORSE (Quadrant 2)

**Soft Metric:**
> "The core of both solutions is identical... However, the agent replaced list concatenation with a manual Python loop, which is often slower in CPython."

**Agent Weakness:** "Manual loop is a classic 'anti-optimization' in Python"

**Reality:** Same approach but agent introduced anti-pattern → worse performance. Understanding was correct, execution was flawed.

---

## Weakness Patterns That Predict Hard Metric Failure

| Weakness Type | Occurrence | Worse Rate | Impact |
|---------------|------------|------------|--------|
| **anti_pattern** | 4 | **75%** | KILLER |
| architecture | 8 | 50% | Significant |
| over_engineering | 17 | 47% | Moderate |
| wrong_api | 14 | 43% | Moderate |
| missed_optimization | 108 | 35% | Common but recoverable |

**Key Insight:** Anti-patterns (manual Python loops instead of builtins, wrong APIs like `any()` vs `torch.any()`) are the bridge between soft and hard metrics. Soft metrics identify them, hard metrics confirm their impact.

---

## Agent Analysis Using Both Metrics

### Claude Code
- **Soft Profile:** 34% same_target, 54% related_target (good understanding)
- **Hard Profile:** 57% success rate
- **TRUE SUCCESS:** ~55%
- **Interpretation:** Good at understanding, decent execution

### Codex
- **Soft Profile:** 56% same_target (best at identifying bottleneck), 94.5% technique overlap
- **Hard Profile:** 50% success rate
- **TRUE SUCCESS:** ~57%
- **Interpretation:** Best understanding, but implementation bugs hurt hard performance

### TRAE (Sonnet)
- **Soft Profile:** 46% no_optimization (high failure to even attempt), 17% same_target
- **Hard Profile:** 54% success rate (when it produces a patch)
- **TRUE SUCCESS:** ~22%
- **Interpretation:** Often fails to produce patch, but when it does, it can work

### TRAE (GPT)
- **Soft Profile:** 34% same_target, 28% no_optimization
- **Hard Profile:** 44% success rate
- **TRUE SUCCESS:** ~38%
- **Interpretation:** Middle ground, lower coverage

---

## Key Insights

### 1. Codex has best UNDERSTANDING but not best HARD success
- 56% same_target (highest among all agents)
- 94.5% technique overlap (highest)
- But only 50% hard success
- **Gap = Implementation quality issues**

### 2. Claude Code has balanced profile
- Not the best at identifying target
- But fewer implementation bugs
- Results in highest hard success (57%)

### 3. TRAE Sonnet has bimodal behavior
- 46% complete failures (no_optimization)
- But when it works, 54% hard success
- **Problem:** Reliability, not capability

### 4. Anti-patterns bridge soft and hard metrics
- Soft metrics identify: "agent used wrong API" or "introduced anti-pattern"
- Hard metrics confirm: 75% worse rate for anti-patterns
- **This is where soft metrics PREDICT hard outcomes**

---

## Correct Evaluation Framework

```
COMPLETE EVALUATION:

1. Did agent produce a valid patch? (Basic)

2. Did agent target correct bottleneck? (Soft)
   - same_target: Full credit
   - related_target: Partial credit
   - different_target/no_opt: No credit

3. Did performance improve? (Hard)
   - beats: Full credit
   - similar: Partial credit
   - worse: No credit

4. Final Score = Weighted combination of Soft × Hard
```

---

## Summary Table

| Metric Type | Measures | Limitation | Use For |
|-------------|----------|------------|---------|
| **Hard** | Does it work? | Can't tell if RIGHT thing was optimized | Validation |
| **Soft** | Did agent understand? | Can't tell if implementation is correct | Understanding |
| **Both** | TRUE SUCCESS | - | Complete evaluation |

---

## Conclusion

> Neither metric alone is sufficient.
>
> Hard metrics validate that performance improved.
>
> Soft metrics validate that the agent understood the task.
>
> TRUE SUCCESS requires both: correct understanding AND working implementation.

---

*Analysis based on 101 matched commits across 4 agents (Claude Code, Codex, TRAE Sonnet, TRAE GPT) with data from patch_quality.json (soft) and HuggingFace benchmark dataset (hard).*
