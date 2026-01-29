# Verified Analysis Report (v8) - Full Data Integrity Audit

## Executive Summary

This report is a **corrected version** of `all_analysis_incorrect_soft.md` after thorough verification against:
1. HuggingFace dataset (`Inferencebench/claude-code-vllm-benchmarks`)
2. Dataset README caveats documenting known data quality issues
3. Correlation analysis to detect benchmark configuration mismatches

**Critical Findings:**

| Finding | Impact |
|---------|--------|
| Codex/TRAE have benchmark config mismatch | **All comparisons INVALID** (r < 0.7) |
| 3 Claude Code commits have README caveats | Should be excluded from analysis |
| Only 20 clean full comparisons remain | Best improvement: +17.24% |
| e3580537 (+22.26%) is flagged | "Universal failure" in README |

---

## Validity Methodology

### 1. Correlation Analysis (Config Mismatch Detection)

| Agent | Comparisons | Correlation (r) | Status |
|-------|-------------|-----------------|--------|
| **Claude Code** | 25 | **r = 0.998** | VALID |
| Codex | 23 | r = 0.507 | **CONFIG MISMATCH** |
| TRAE | 37 | r = 0.561 | **CONFIG MISMATCH** |

**Why this matters:** Low correlation means agent TTFT values don't track human values, indicating different benchmark configurations were used.

### 2. README Caveats (Known Data Quality Issues)

The dataset README documents several problem commits:

| Category | Commits | Issue |
|----------|---------|-------|
| Infrastructure Failures | `6ce01f30`, `e7523c2e` | Corrupted Docker, memory limits |
| Partial Data Issues | `83450458`, `19d98e0c` | Missing human metrics |
| Universal Failures | 12 commits including `e3580537`, `fc7b8d1e` | All agents failed |
| Unbenchmarkable | 7 commits | Old vLLM versions |
| Non-Standard | `ccf02fcb`, `ce6bf3a2` | TPU-only, accuracy tests |

---

## Claude Code Results (After Applying All Filters)

### Commits Flagged by README Caveats

| Commit | Improvement | Caveat | Action |
|--------|-------------|--------|--------|
| **e3580537** | +22.26% | **universal_failure** | EXCLUDE |
| fc7b8d1e | -9.69% | universal_failure | EXCLUDE |
| 19d98e0c | -59.43% | partial_data_issue | EXCLUDE |

**Critical: e3580537 was the "top performer" claim but is flagged as a universal failure!**

### Clean Comparisons (No Caveats)

After removing flagged commits: **22 clean comparisons**
- 20 full (B/H/A all exist)
- 2 partial (no baseline): `6e36f4fa`, `89a84b0b`

### Full Comparisons (B/H/A All Exist) - Sorted by Improvement

| Rank | Commit | Baseline | Human | Agent | Improvement |
|------|--------|----------|-------|-------|-------------|
| 1 | **22d33bac** | 596.3ms | 786.8ms | 651.1ms | **+17.24%** |
| 2 | a3223766 | 35.8ms | 33.5ms | 30.8ms | +8.17% |
| 3 | 6a417b86 | 1762.0ms | 1160.4ms | 1097.6ms | +5.41% |
| 4 | bc7c4d20 | 2435.9ms | 2520.7ms | 2454.7ms | +2.62% |
| 5 | 30172b49 | 1115.7ms | 1103.5ms | 1074.9ms | +2.59% |
| 6 | 8a4e5c5f | 898.6ms | 924.7ms | 908.6ms | +1.74% |
| 7 | 015069b0 | 10.5ms | 13.7ms | 13.5ms | +1.53% |
| 8 | 70b808fe | 59.8ms | 58.7ms | 58.2ms | +0.89% |
| 9 | 9f1710f1 | 382.8ms | 387.4ms | 385.7ms | +0.44% |
| 10 | fc542144 | 32.0ms | 34.5ms | 34.6ms | -0.32% |
| 11 | ed250545 | 818.3ms | 799.2ms | 807.9ms | -1.08% |
| 12 | 6d0734c5 | 2194.9ms | 2167.0ms | 2194.8ms | -1.28% |
| 13 | 299ebb62 | 25.7ms | 22.6ms | 22.9ms | -1.51% |
| 14 | b55ed6ef | 1145.2ms | 1031.6ms | 1056.1ms | -2.38% |
| 15 | fe66b347 | 6225.6ms | 5722.9ms | 5874.6ms | -2.65% |
| 16 | 58eee5f2 | 838.1ms | 811.1ms | 835.5ms | -3.01% |
| 17 | 9badee53 | 2894.7ms | 168.6ms | 174.7ms | -3.59% |
| 18 | 296f927f | 1404.2ms | 1355.8ms | 1406.1ms | -3.71% |
| 19 | b690e348 | 37803.3ms | 9130.9ms | 9640.5ms | -5.58% |
| 20 | **e206b543** | 576.8ms | 574.2ms | 669.9ms | **-16.67%** |

### Partial Comparisons (No Baseline - Use With Caution)

| Commit | Human | Agent | Improvement | Note |
|--------|-------|-------|-------------|------|
| 6e36f4fa | 858.5ms | 1011.5ms | -17.82% | No caveat but no baseline |
| 89a84b0b | 205.7ms | 356.1ms | **-73.13%** | No caveat but no baseline |

---

## Summary Statistics (Clean Full Comparisons Only)

| Metric | Value |
|--------|-------|
| Total clean full comparisons | 20 |
| Beats human (>1%) | 7 (35%) |
| Similar (±1%) | 3 (15%) |
| Worse (<-1%) | 10 (50%) |
| **Best improvement** | **+17.24% (22d33bac)** |
| Worst regression | -16.67% (e206b543) |
| **Average improvement** | **-0.06%** |

---

## Original Report Claims: Final Verification

| Commit | Agent | Claimed | Actual | Status |
|--------|-------|---------|--------|--------|
| e3580537 | claude-code | +22.3% | +22.26% | **INVALID** (README: universal_failure) |
| 22d33bac | claude-code | +17.2% | +17.24% | **VERIFIED** (best clean result) |
| a3223766 | claude-code | +8.2% | +8.17% | **VERIFIED** |
| 89a84b0b | claude-code | -73.1% | -73.13% | **CAUTION** (no baseline) |
| 30172b49 | codex | +46.9% | — | **INVALID** (config mismatch r=0.51) |
| b55ed6ef | codex | +42.8% | — | **INVALID** (config mismatch) |
| 58eee5f2 | codex | +25.8% | — | **INVALID** (config mismatch) |
| 30172b49 | trae | +45.7% | — | **INVALID** (config mismatch r=0.56) |
| b55ed6ef | trae | +43.7% | — | **INVALID** (config mismatch) |
| 58eee5f2 | trae | +25.4% | — | **INVALID** (config mismatch) |

**Final tally: 2 VERIFIED, 1 CAUTION, 7 INVALID**

---

## Case Study: 89a84b0b (-73.13% Regression)

This commit illustrates why agents often fail to replicate human optimizations.

### Commit Info

| Field | Value |
|-------|-------|
| **Commit** | 89a84b0bb7b30706a02836234a94493ea8f780bf |
| **Title** | [Core] Use array to speedup padding (#6779) |
| **PR** | https://github.com/vllm-project/vllm/pull/6779 |
| **Files** | sampler.py, sampling_metadata.py, sequence.py |
| **Date** | July 26, 2024 |

### Benchmark Results

| Metric | Human | Agent | Change |
|--------|-------|-------|--------|
| TTFT Mean | 205.66ms | 356.05ms | **-73.13%** (regression) |
| TTFT Median | 196.92ms | 349.72ms | -77.6% |
| TPOT Mean | 24.53ms | 23.73ms | +3.3% |
| **Baseline** | **None** | — | **No verification possible** |

### Human's Optimization (Ground Truth)

The human changed Python `list` to `array('l')` for token storage:

```python
# Before (list)
self._prompt_token_ids: List[int] = list(prompt_token_ids)
self._output_token_ids: List[int] = list(output_token_ids)

# After (array)
from array import array
self._prompt_token_ids = array('l', prompt_token_ids)
self._output_token_ids = array('l', output_token_ids)
```

**Why this is faster:** Python `array` is a compact, homogeneous storage that's much more memory-efficient and faster for numeric operations than Python lists. It avoids Python object overhead per element and has better cache locality.

### Claude Code's Optimization (Agent Patch)

The agent made **completely different** changes:

**1. Changed `torch.zeros` to `torch.empty` + `fill_(0)`:**
```python
# Agent's change
- bin_counts = torch.zeros((num_seqs, vocab_size + 1), ...)
+ bin_counts = torch.empty((num_seqs, vocab_size + 1), ...)
+ bin_counts.fill_(0)
```

**2. Changed list `+=` to `.extend()`:**
```python
# Agent's change
- temperatures += [temperature] * prefill_len
+ temperatures.extend([temperature] * prefill_len)
```

**3. Changed list concatenation to slice + extend:**
```python
# Agent's change
- self._cached_all_token_ids = self._prompt_token_ids + self._output_token_ids
+ self._cached_all_token_ids = self._prompt_token_ids[:]
+ self._cached_all_token_ids.extend(self._output_token_ids)
```

### Why The Regression Occurred

**The agent's "optimizations" are actually slower because:**

1. **`torch.empty` + `fill_(0)` ≠ `torch.zeros`** — Two operations instead of one optimized call
2. **`.extend()` with a new list ≠ faster than `+=`** — Still creates intermediate list `[temperature] * prefill_len`
3. **Slice copy + extend ≠ faster than concatenation** — `list[:]` creates a full copy first, then extends

**The human's optimization (using `array` type) is fundamentally different:**
- Uses compact memory representation (no Python object per element)
- Better cache locality for sequential access
- Native C-level operations instead of Python object manipulation

### Key Insight

The agent completely missed the actual optimization pattern. Instead of recognizing that the human used Python's `array` module for efficient numeric storage, the agent applied generic micro-optimizations (`extend` vs `+=`, `torch.empty` vs `torch.zeros`) that either had no effect or made performance worse.

**This case demonstrates that "agent beats human" or "agent loses to human" doesn't mean the agent found the same optimization — they're often solving different problems.**

---

## Key Conclusions

1. **Only Claude Code has valid data** (r=0.998 correlation)

2. **Codex/TRAE data is completely unusable** (r=0.51, r=0.56 - config mismatch)

3. **Top performer claim is invalid**: e3580537 (+22.26%) is flagged as "universal_failure" in README

4. **Best verified improvement is +17.24%** (22d33bac)

5. **Overall performance is neutral**: Average improvement is -0.06% across 20 clean full comparisons

6. **35% beats human, 50% worse**: When clean data is used, agent performance is modest

7. **Extreme regressions lack baseline verification**: -73% (89a84b0b) and -17.82% (6e36f4fa) have no baseline data

---

## Data Quality Filters Applied

```python
# Commits excluded due to README caveats
readme_caveats = ["e3580537", "fc7b8d1e", "19d98e0c"]

# Agents excluded due to config mismatch (r < 0.7)
invalid_agents = ["codex", "trae"]

# Clean data = Claude Code only, excluding README caveats, full B/H/A
clean_full_comparisons = 20
```

---

*Generated: 2026-01-20*
*Methodology: Correlation-based validity (r > 0.7) + README caveats*
*Verified against: HuggingFace `Inferencebench/claude-code-vllm-benchmarks`*
*Verification script: `scripts/verify_benchmark_data.py`*
