# Findings: Claude Code (Sonnet 4.5) on vLLM Optimization Tasks

---

## Part A: Hard Metrics Story

### A.1 Overall Performance

| Outcome | Count | % |
|---------|-------|---|
| Agent beats Human (>2%) | 9 | 20.5% |
| Similar performance (±2%) | 12 | 27.3% |
| Agent worse (<-2%) | 11 | 25.0% |
| Agent patch failed to run | 11 | 25.0% |
| Human patch failed | 1 | 2.3% |

**47.7% achieve equal-or-better performance.** Wide variance (+24% to -19%) indicates task-dependent success.

---

## Part B: Soft Metrics Story

### B.1 The Precision Gap

| Stage | Success Rate |
|-------|--------------|
| Find correct file | 69% |
| Find correct bottleneck | 37.5% |
| Gap | 31.5 pp |

Agents can find the right files but struggle to identify the specific bottleneck within.

### B.2 Agent Approach Patterns

| Approach | Count | % |
|----------|-------|---|
| valid_alternative | 17 | 38.6% |
| partial_solution | 13 | 29.5% |
| ineffective | 9 | 20.5% |
| similar_approach | 5 | 11.4% |

**Only 11% match human approach. 39% find different but valid alternatives.**

### B.3 Optimization Technique Preferences

| Technique | % |
|-----------|---|
| memory_optimization | 47.4% |
| lazy_computation | 18.4% |
| api_library | 18.4% |
| parallelization | 2.6% |
| algorithmic | 2.6% |

Agents strongly prefer simple memory optimizations over complex architectural changes.

### B.4 Failure Mode Distribution

| Failure Mode | % |
|--------------|---|
| localization_failure | 34.1% |
| complexity_avoidance | 31.8% |
| not_applicable | 25.0% |
| incomplete_implementation | 9.1% |

---

## Part C: Why Both Metrics Matter - The Valid Alternative Story

### C.1 The Key Insight

When soft metrics say "valid_alternative" + hard metrics show improvement, what does it mean?

**Answer:** The benchmark may not precisely capture the intended bottleneck.

### C.2 Evidence from "valid_alternative" Cases

| Commit | Human Target | Agent Target |
|--------|--------------|--------------|
| 6ce01f30 | SequenceGroup iteration caching | General Python list operations |
| a3223766 | Tensor update trigger logic | Memory allocation patterns |
| b55ed6ef | Data compaction/shifting | Tensor initialization (zeros→empty) |
| e206b543 | deepcopy overhead in clone() | Vocabulary processing allocations |

**Pattern:** Human and agent target DIFFERENT specific bottlenecks. Both are valid optimizations, but they address different performance issues.

### C.3 What This Reveals About Benchmarks

**Benchmark limitation:** Benchmarks measure OVERALL performance, not isolated bottlenecks.

When agent uses "valid_alternative" approach and benchmark improves:
- Agent didn't solve the human's specific bottleneck
- Agent found a DIFFERENT valid optimization
- Benchmark improved, but possibly for different reasons than intended

### C.4 The a3223766 Example

**Human target:** Optimizes trigger logic for when tensors need updating
**Agent target:** Optimizes memory allocation (zeros→empty) and boolean indexing
**Result:** Agent +8.17%

**What happened:**
- Human identified a logical bottleneck (redundant tensor rebuilds)
- Agent applied generic memory optimization
- Benchmark improved because BOTH optimizations help
- But agent didn't solve the specific problem human identified

### C.5 Why You Need BOTH Metrics

| Metric Type | What It Tells You | What It Misses |
|-------------|-------------------|----------------|
| Hard metrics only | Did performance improve? | Did agent solve the INTENDED problem? |
| Soft metrics only | What did agent target? | Did it actually help performance? |
| Both together | Full picture: different approach that still worked |

**Without soft metrics:** You might think agent "solved" the problem when it actually found a different optimization

**Without hard metrics:** You might dismiss a "wrong" approach that actually improved performance

---

## Part D: The Disconnect Between Approaches and Outcomes

### D.1 Same Bottleneck Target ≠ Better Outcome

| Bottleneck Target | Beats % | Worse % |
|-------------------|---------|---------|
| same_target | 25% | 42% |
| related_target | 35% | 30% |
| different_target | 0% | 33% |

**Surprising:** Targeting SAME bottleneck has LOWER beats rate (25%) than related_target (35%).

**Why?** When agent targets same bottleneck as human:
- Agent may execute the approach poorly
- Human's solution is already optimized for that specific issue
- Agent's "same target" is often a weaker version

When agent targets related area:
- Agent may find optimizations human missed
- Alternative approach may be better suited to benchmark

### D.2 The Case Study Contrast

**e3580537 (+24.43%):** related_target, ineffective approach
- Human: Chunked prefill integration
- Agent: Low-level Python cleanup
- Benchmark improved from agent's unrelated optimizations

**2deb029d (-19.21%):** different_target, ineffective approach
- Human: mark_blocks_as_computed call
- Agent: Dict lookup caching
- Agent missed functional requirement, broke the feature

**What soft metrics captured:** Both used ineffective approaches
**What hard metrics captured:** One improved, one regressed
**Together:** Agent 1's unrelated opts accidentally helped; Agent 2's broke functionality

---

## Part E: Key Findings

### Finding 1: The Precision Gap
- 69% file-level accuracy
- 37.5% bottleneck-level accuracy
- Agents find files but miss specific bottlenecks

### Finding 2: Valid Alternative Approaches Reveal Benchmark Limitations
- 39% find valid alternatives targeting different bottlenecks
- Benchmark can improve from optimizations that don't address intended problem
- Shows benchmarks measure general performance, not isolated bottlenecks

### Finding 3: Lazy Optimization Bias
- 47% use only memory_optimization
- Complex techniques rarely used
- Agents default to pattern-based changes

### Finding 4: Both Metrics Are Necessary
- Hard metrics alone: Can't tell if agent solved intended problem
- Soft metrics alone: Can't tell if approach actually helped
- Together: Reveal whether improvement came from solving the problem or a valid alternative

### Finding 5: different_target Is Only Reliable Negative Signal
- 0% beats rate when agent completely misses bottleneck area
- same_target and related_target don't predict outcomes
