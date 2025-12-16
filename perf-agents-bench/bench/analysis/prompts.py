"""LLM analysis prompt templates for soft metrics extraction.

Contains the academic-grade prompt template for analyzing agent benchmark runs
using Gemini 3 Pro via OpenRouter.
"""

from __future__ import annotations

import json
from typing import Dict, Any, List, Optional


ANALYSIS_PROMPT_TEMPLATE = '''# Agent Performance Analysis for Academic Research

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

### 2. Quality Scores (0-10 scale with detailed justification)

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

```json
{{
  "quantitative_assessment": {{
    "tool_calls": {{"bash": 0, "editor": 0, "read": 0, "other": 0}},
    "error_counts": {{"syntax": 0, "runtime": 0, "api": 0, "timeout": 0}},
    "iteration_count": 0,
    "exploration_steps": 0,
    "execution_steps": 0
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
```

Ensure all scores are numbers between 0 and 10. Be specific and evidence-based in your justifications.
'''


def build_analysis_prompt(data: Dict[str, Any]) -> str:
    """Build the analysis prompt from loaded run data.

    Args:
        data: Dict from RunLoader.load_all() containing all run artifacts

    Returns:
        Formatted prompt string
    """
    metadata = data.get("metadata", {})
    journal = data.get("journal", {})
    prompt_data = data.get("prompt", {})
    diff_targets = data.get("diff_targets", {})
    run_summary = data.get("run_summary", {})

    # Get agent info from journal
    agent_type = metadata.get("agent", "unknown")
    agent_info = journal.get(agent_type, {})

    # Format constraints
    constraints = prompt_data.get("constraints", [])
    constraints_str = "\n".join(f"- {c}" for c in constraints) if constraints else "No specific constraints provided."

    # Format target files
    target_files = prompt_data.get("target_files", [])
    target_files_str = "\n".join(f"- `{f}`" for f in target_files) if target_files else "No specific target files provided."

    # Format diff targets
    diff_targets_str = json.dumps(diff_targets, indent=2) if diff_targets else "{}"

    # Get metrics
    metrics = journal.get("metrics", {})
    agent_metrics = run_summary.get("agent", {})
    patch_stats = agent_metrics.get("patch_stats", {})

    # Truncate stdout/stderr if too long (keep first and last parts)
    stdout = data.get("stdout", "")
    stderr = data.get("stderr", "")

    max_stdout_len = 500000  # ~500KB
    if len(stdout) > max_stdout_len:
        half = max_stdout_len // 2
        stdout = stdout[:half] + "\n\n... [TRUNCATED] ...\n\n" + stdout[-half:]

    max_stderr_len = 50000  # ~50KB
    if len(stderr) > max_stderr_len:
        stderr = stderr[:max_stderr_len] + "\n\n... [TRUNCATED] ..."

    # Build the prompt
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        # Task context
        task_content=data.get("task", prompt_data.get("description", "No task description available.")),
        agent_type=agent_type,
        model_full=metadata.get("model_full", metadata.get("model", "unknown")),
        repo=metadata.get("repo", "unknown"),
        time_budget_minutes=agent_info.get("time_budget_minutes", 120),
        task_id=metadata.get("task_id", journal.get("task_id", "unknown")),

        # Commits
        commit_pre=metadata.get("commits", {}).get("pre", "unknown"),
        commit_human=metadata.get("commits", {}).get("human", "unknown"),

        # Constraints and targets
        constraints=constraints_str,
        target_files=target_files_str,

        # Agent output
        stdout_content=stdout or "No stdout captured.",
        stderr_content=stderr or "No stderr captured.",
        patch_content=data.get("patch", "No patch generated."),
        diff_targets=diff_targets_str,

        # Status
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


def build_summary_prompt(analyses: List[Dict[str, Any]]) -> str:
    """Build a prompt for summarizing multiple run analyses.

    Args:
        analyses: List of RunAnalysis.to_dict() results

    Returns:
        Prompt for aggregate analysis
    """
    prompt = """# Aggregate Analysis of Agent Benchmark Runs

You are summarizing the results of multiple agent benchmark runs for academic research.

## Run Analyses

"""
    for i, analysis in enumerate(analyses, 1):
        meta = analysis.get("meta", {})
        quant = analysis.get("quantitative_metrics", {})
        qual = analysis.get("qualitative_metrics", {}).get("scores", {})

        prompt += f"""
### Run {i}: {meta.get('item_id', 'unknown')}
- Agent: {meta.get('agent', 'unknown')} / Model: {meta.get('model', 'unknown')}
- Status: {quant.get('status', 'unknown')}
- Duration: {quant.get('total_duration_s', 0):.1f}s
- Scores: Understanding={qual.get('code_understanding', 0):.1f}, Alignment={qual.get('task_alignment', 0):.1f}, Approach={qual.get('approach_quality', 0):.1f}, Execution={qual.get('execution_quality', 0):.1f}

"""

    prompt += """
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
"""

    return prompt
