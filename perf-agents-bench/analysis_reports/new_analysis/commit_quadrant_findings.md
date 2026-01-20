# Commit-by-Commit Quadrant Analysis

## Overview

Analysis of 39 vLLM performance optimization commits across 4 AI agents using the Four Quadrants Framework.

**TRUE SUCCESS** = (same_target OR related_target) AND (beats OR similar)

---

## The Four Quadrants

```
                        HARD METRIC
                   Beats    |    Worse
                -------------------------
    same/       |  Q1 TRUE  |  Q2 Good
    related     |  SUCCESS  |  Intent
    target      |           |  Bad Exec
SOFT -----------|-----------|-------------
METRIC          |           |
                |  Q3 Lucky |  Q4 Complete
  other         |  Win      |  Failure
                |           |
```

---

## Claude Code (30 VALID commits)

| Commit | Hard | Target | Approach | Quadrant |
|--------|------|--------|----------|----------|
| 89a84b0b | worse | same_target | partial_solution | Q2 |
| 3476ed08 | worse | related_target | partial_solution | Q2 |
| 19d98e0c | worse | related_target | partial_solution | Q2 |
| **fa63e710** | similar | **different_target** | ineffective | **Q3 Lucky** |
| b690e348 | worse | related_target | ineffective | Q2 |
| 6e36f4fa | worse | related_target | partial_solution | Q2 |
| fc7b8d1e | worse | same_target | similar_approach | Q2 |
| 30172b49 | **beats** | related_target | partial_solution | **Q1** |
| 2deb029d | NO_DATA | different_target | ineffective | Q4 |
| 4c822298 | similar | same_target | partial_solution | **Q1** |
| b55ed6ef | worse | related_target | valid_alternative | Q2 |
| 015069b0 | similar | same_target | similar_approach | **Q1** |
| a3223766 | **beats** | related_target | valid_alternative | **Q1** |
| 99abb8b6 | worse | same_target | ineffective | Q2 |
| 310aca88 | similar | related_target | ineffective | **Q1** |
| 58eee5f2 | worse | same_target | valid_alternative | Q2 |
| 9f1710f1 | similar | related_target | valid_alternative | **Q1** |
| bc7c4d20 | **beats** | same_target | partial_solution | **Q1** |
| e3580537 | **beats** | related_target | ineffective | **Q1** |
| 9474e89b | worse | same_target | similar_approach | Q2 |
| 98f47f2a | **beats** | same_target | partial_solution | **Q1** |
| fc542144 | similar | same_target | valid_alternative | **Q1** |
| **22d33bac** | **beats** | **different_target** | valid_alternative | **Q3 Lucky** |
| 299ebb62 | similar | same_target | similar_approach | **Q1** |
| 9badee53 | worse | related_target | valid_alternative | Q2 |
| 6a417b86 | **beats** | related_target | ineffective | **Q1** |
| 296f927f | worse | same_target | partial_solution | Q2 |
| e206b543 | worse | related_target | valid_alternative | Q2 |
| 70b808fe | similar | related_target | valid_alternative | **Q1** |
| fe66b347 | worse | same_target | similar_approach | Q2 |

### Claude Code Summary
| Quadrant | Count | Percentage |
|----------|-------|------------|
| Q1 TRUE SUCCESS | 13 | 43.3% |
| Q2 Good Intent, Bad Exec | 14 | 46.7% |
| Q3 Lucky Win | 2 | 6.7% |
| Q4 Complete Failure | 1 | 3.3% |

**Hard Success:** 15/30 (50.0%)
**TRUE SUCCESS:** 13/30 (43.3%)
**Lucky Wins:** fa63e710, 22d33bac

---

## Codex (17 VALID commits)

| Commit | Hard | Target | Approach | Quadrant |
|--------|------|--------|----------|----------|
| 19d98e0c | NO_DATA | different_target | ineffective | Q4 |
| fa63e710 | similar | same_target | partial_solution | **Q1** |
| 30172b49 | **beats** | related_target | partial_solution | **Q1** |
| 4c822298 | worse | same_target | similar_approach | Q2 |
| b55ed6ef | **beats** | related_target | partial_solution | **Q1** |
| e7b20426 | worse | related_target | partial_solution | Q2 |
| 99abb8b6 | worse | related_target | partial_solution | Q2 |
| 58eee5f2 | **beats** | same_target | similar_approach | **Q1** |
| 9f1710f1 | worse | related_target | harmful | Q2 |
| 8c1e77fb | worse | different_target | valid_alternative | Q4 |
| 98f47f2a | **beats** | same_target | valid_alternative | **Q1** |
| fc542144 | worse | same_target | valid_alternative | Q2 |
| 3b61cb45 | worse | same_target | similar_approach | Q2 |
| 22d33bac | **beats** | same_target | similar_approach | **Q1** |
| 299ebb62 | worse | same_target | similar_approach | Q2 |
| 9badee53 | similar | related_target | valid_alternative | **Q1** |
| e206b543 | worse | same_target | similar_approach | Q2 |

### Codex Summary
| Quadrant | Count | Percentage |
|----------|-------|------------|
| Q1 TRUE SUCCESS | 7 | 41.2% |
| Q2 Good Intent, Bad Exec | 8 | 47.1% |
| Q3 Lucky Win | **0** | **0%** |
| Q4 Complete Failure | 2 | 11.8% |

**Hard Success:** 7/17 (41.2%)
**TRUE SUCCESS:** 7/17 (41.2%)
**Lucky Wins:** None

---

## TRAE Sonnet (17 VALID commits)

| Commit | Hard | Target | Approach | Quadrant |
|--------|------|--------|----------|----------|
| 19d98e0c | NO_DATA | no_optimization | other | Q4 |
| **fa63e710** | similar | **no_optimization** | other | **Q3 Lucky** |
| 30172b49 | **beats** | related_target | partial_solution | **Q1** |
| 4c822298 | worse | NO_FILE | NO_FILE | Q4 |
| b55ed6ef | **beats** | related_target | partial_solution | **Q1** |
| e7b20426 | worse | no_optimization | other | Q4 |
| 99abb8b6 | worse | same_target | ineffective | Q2 |
| **58eee5f2** | **beats** | **NO_FILE** | NO_FILE | **Q3 Lucky** |
| 9f1710f1 | worse | related_target | valid_alternative | Q2 |
| 8c1e77fb | worse | related_target | valid_alternative | Q2 |
| 98f47f2a | **beats** | same_target | partial_solution | **Q1** |
| fc542144 | worse | no_optimization | ineffective | Q4 |
| 3b61cb45 | worse | no_optimization | other | Q4 |
| **22d33bac** | **beats** | **different_target** | ineffective | **Q3 Lucky** |
| 299ebb62 | worse | same_target | similar_approach | Q2 |
| **9badee53** | **beats** | **NO_FILE** | NO_FILE | **Q3 Lucky** |
| e206b543 | worse | no_optimization | ineffective | Q4 |

### TRAE Sonnet Summary
| Quadrant | Count | Percentage |
|----------|-------|------------|
| Q1 TRUE SUCCESS | 3 | 17.6% |
| Q2 Good Intent, Bad Exec | 4 | 23.5% |
| Q3 Lucky Win | **4** | **23.5%** |
| Q4 Complete Failure | 6 | 35.3% |

**Hard Success:** 7/17 (41.2%)
**TRUE SUCCESS:** 3/17 (17.6%)
**Lucky Wins:** fa63e710, 58eee5f2, 22d33bac, 9badee53

---

## TRAE GPT (12 VALID commits)

| Commit | Hard | Target | Approach | Quadrant |
|--------|------|--------|----------|----------|
| **fa63e710** | similar | **NO_FILE** | NO_FILE | **Q3 Lucky** |
| 4c822298 | worse | NO_FILE | NO_FILE | Q4 |
| b55ed6ef | **beats** | related_target | valid_alternative | **Q1** |
| 99abb8b6 | worse | related_target | ineffective | Q2 |
| **58eee5f2** | **beats** | **no_optimization** | ineffective | **Q3 Lucky** |
| 8c1e77fb | worse | related_target | valid_alternative | Q2 |
| 98f47f2a | **beats** | same_target | similar_approach | **Q1** |
| 3b61cb45 | worse | NO_FILE | NO_FILE | Q4 |
| **22d33bac** | **beats** | **different_target** | valid_alternative | **Q3 Lucky** |
| 9badee53 | worse | same_target | similar_approach | Q2 |
| 296f927f | worse | same_target | similar_approach | Q2 |
| 70b808fe | worse | different_target | ineffective | Q4 |

### TRAE GPT Summary
| Quadrant | Count | Percentage |
|----------|-------|------------|
| Q1 TRUE SUCCESS | 2 | 16.7% |
| Q2 Good Intent, Bad Exec | 4 | 33.3% |
| Q3 Lucky Win | **3** | **25.0%** |
| Q4 Complete Failure | 3 | 25.0% |

**Hard Success:** 5/12 (41.7%)
**TRUE SUCCESS:** 2/12 (16.7%)
**Lucky Wins:** fa63e710, 58eee5f2, 22d33bac

---

## Final Summary Table

| Agent | Valid/39 | Patch% | Hard Success | TRUE SUCCESS | Lucky Wins |
|-------|----------|--------|--------------|--------------|------------|
| Claude Code | 30 | 76.9% | 15 (50.0%) | 13 (43.3%) | 2 (6.7%) |
| Codex | 17 | 43.6% | 7 (41.2%) | 7 (41.2%) | **0 (0%)** |
| TRAE Sonnet | 17 | 43.6% | 7 (41.2%) | 3 (17.6%) | **4 (23.5%)** |
| TRAE GPT | 12 | 30.8% | 5 (41.7%) | 2 (16.7%) | **3 (25.0%)** |

---

## Key Findings

### 1. Codex has ZERO Lucky Wins
All 7 of Codex's hard successes came from commits where it correctly targeted the bottleneck (same_target or related_target). TRUE SUCCESS = Hard Success.

### 2. TRAE Agents Have High Lucky Win Rate
- TRAE Sonnet: 4/7 hard successes (57%) were lucky wins
- TRAE GPT: 3/5 hard successes (60%) were lucky wins

This means majority of TRAE's "successes" came from micro-optimizations, not solving the actual task.

### 3. Claude Code is Balanced
- Highest patch success (76.9%)
- Only 2 lucky wins out of 15 hard successes (13%)
- Most commits are Q1 or Q2 (correct targeting)

### 4. Common Lucky Win Commits
Three commits appear as lucky wins across multiple agents:
- **fa63e710** - Lucky for Claude Code, TRAE Sonnet, TRAE GPT
- **22d33bac** - Lucky for Claude Code, TRAE Sonnet, TRAE GPT
- **58eee5f2** - Lucky for TRAE Sonnet, TRAE GPT

These commits may have benchmarks that are easily gamed by micro-optimizations.

---

## Quadrant Distribution Comparison

| Agent | Q1 TRUE | Q2 Good/Bad | Q3 Lucky | Q4 Fail |
|-------|---------|-------------|----------|---------|
| Claude Code | 43.3% | 46.7% | 6.7% | 3.3% |
| Codex | 41.2% | 47.1% | 0% | 11.8% |
| TRAE Sonnet | 17.6% | 23.5% | 23.5% | 35.3% |
| TRAE GPT | 16.7% | 33.3% | 25.0% | 25.0% |

---

*Analysis based on 39 commits from commits_v5.md with soft metrics from patch_quality.json and hard metrics from HuggingFace benchmark dataset.*
