# OpenHands Integration Guide

Complete guide for resolving the "Agent reached maximum iterations without completing task" issue and successfully running OpenHands with the perf-agents-bench system.

## Problem Summary

When running `python test_integration.py --real --here`, the OpenHands agent was failing with:
```
Error: OpenHands task failed: Agent reached maximum iterations without completing task
```

This guide documents the root causes and provides a complete solution.

## Root Causes Identified

### 1. Missing OpenHands Installation
The primary issue was that OpenHands wasn't installed in the active Python environment.

**Error seen:**
```
Test failed with exception: OpenHands Python package not installed
```

### 2. Insufficient Iteration Limits
The test was using only 10 iterations, which wasn't enough for the agent to complete even simple tasks.

### 3. Vague Task Descriptions
The task descriptions lacked clear completion criteria and specific instructions on how the agent should finish the task.

### 4. Missing Target Files
The agent was looking for files that didn't exist or weren't clearly specified in the workspace.

## Complete Solution

### Step 1: Use the Correct Virtual Environment

The `bench-env` virtual environment already has OpenHands installed. Always activate it before running tests:

```bash
# Activate the virtual environment
source /root/OmniPerf-Bench/bench-env/bin/activate

# Verify OpenHands is installed
pip list | grep openhands
# Should show:
# openhands-aci                            0.3.1
# openhands-ai                             0.53.0
```

### Step 2: Create Target Files for Testing

Create a simple Python file for the agent to work on:

```python
# simple.py
"""
Simple Python file for testing OpenHands integration.
"""

def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b

def calculate_area(length, width):
    return length * width

result = add_numbers(5, 3)
print(f"Result: {result}")
```

### Step 3: Improve Task Description Format

Updated the `create_opensource_task` function in `bench/agents/openhands_cli.py` to include explicit completion criteria:

```python
def create_opensource_task(
    task_name: str,
    description: str,
    target_files: list,
    constraints: list = None,
    reference_commit: str = None,
    primary_metric: str = None
) -> str:
    # ... existing code ...
    
    if primary_metric:
        sections.extend([
            "",
            "## Success Criteria",
            f"- Primary metric: {primary_metric}",
            "- All existing tests must pass",
            "- No regression in functionality",
            "",
            "## Instructions",
            "1. Analyze the target files for performance bottlenecks",
            "2. Implement optimizations while respecting constraints",
            "3. Test your changes to ensure correctness",
            "4. Commit your optimizations",
            "",
            "## Task Completion",
            "When you have successfully completed the task:",
            "1. Save all changes to the target files",
            "2. Run `git add .` to stage changes",
            "3. Run `git commit -m 'Add descriptive comments to functions'` to commit",
            "4. Use the `finish` command to indicate task completion",
            "",
            "The task is complete when you have added the requested comments and committed the changes."
        ])
    
    return "\n".join(sections)
```

### Step 4: Increase Iteration Limits

Updated the test integration file to use more reasonable iteration limits:

```python
# In test_integration.py
result = orchestrator.optimize_repository(
    repo_path=workspace,
    task_name="Simple Code Enhancement",
    description="Add a descriptive comment to the add_numbers function in simple.py explaining what it does",
    target_files=task_target_files,
    constraints=["Only add comments, do not change functionality", "Do not modify test files"],
    max_iterations=20,  # Increased from 10
    timeout_minutes=5
)
```

### Step 5: Make Task Descriptions More Specific

Changed vague descriptions to specific, actionable tasks:

**Before:**
```python
description="Add a simple comment to the add_numbers function"
```

**After:**
```python
description="Add a descriptive comment to the add_numbers function in simple.py explaining what it does"
```

## How to Use This Solution

### For Testing OpenHands Integration

1. **Activate the correct environment:**
   ```bash
   cd /path/to/OmniPerf-Bench/perf-agents-bench
   source ../bench-env/bin/activate
   ```

2. **Ensure you have the required files:**
   ```bash
   # Check that simple.py exists
   ls simple.py
   
   # If not, create it with the content shown above
   ```

3. **Run the integration test:**
   ```bash
   # Safe test (no API costs)
   python test_integration.py
   
   # Real API test (costs money - be careful!)
   python test_integration.py --real --here
   ```

### For Custom OpenHands Tasks

1. **Always use the bench-env environment:**
   ```bash
   source /path/to/bench-env/bin/activate
   ```

2. **Create specific, actionable task descriptions:**
   ```python
   task = create_opensource_task(
       task_name="Your Task Name",
       description="Specific description of what to do and which files to modify",
       target_files=["specific_file.py"],  # List actual files
       constraints=[
           "Only add comments, do not change functionality",
           "Do not modify test files"
       ],
       primary_metric="code_quality"
   )
   ```

3. **Use reasonable iteration limits:**
   - Simple tasks (comments, small changes): 15-25 iterations
   - Medium tasks (refactoring, optimizations): 30-50 iterations
   - Complex tasks: 50+ iterations

4. **Include explicit completion criteria in task descriptions**

### Environment Setup for New Users

If you're setting up a new environment:

1. **Install OpenHands:**
   ```bash
   pip install openhands-ai
   # or
   uv pip install openhands-ai
   ```

2. **Configure OpenHands:**
   ```bash
   # Set up your API keys in .env file
   echo "OPENAI_API_KEY=your_key_here" >> .env
   ```

3. **Verify installation:**
   ```python
   python -c "import openhands; print('OpenHands installed successfully')"
   ```

## Success Metrics

After implementing these fixes, you should see:

### Successful Test Output
```
============================================================
TEST RESULTS
============================================================
Duration: 178.7 seconds
Success: True
Cost Estimate: $0.0700
Agent Branch: agent/simple_code_enhancement/1756733369
SUCCESS: OpenHands completed the task!

First commit detected (current HEAD):
commit 3e80afb1da69a733d421f8be980501fbba6b5be2
Author: openhands <openhands@all-hands.dev>
Date:   Mon Sep 1 13:32:13 2025 +0000

    Add descriptive comment to add_numbers in simple.py
```

### Agent State Progression
```
AgentState.LOADING → AgentState.RUNNING → AgentState.FINISHED
```

**NOT:**
```
AgentState.LOADING → AgentState.RUNNING → AgentState.ERROR
```

## Troubleshooting

### Common Issues and Solutions

1. **"OpenHands Python package not installed"**
   - Solution: Activate the bench-env virtual environment

2. **"Agent reached maximum iterations"**
   - Solution: Increase max_iterations parameter (try 20-30 for simple tasks)
   - Solution: Make task description more specific and actionable

3. **Agent gets confused about target files**
   - Solution: Ensure target files exist in the workspace
   - Solution: Be specific about file names in the task description

4. **Agent doesn't know when to finish**
   - Solution: Include explicit completion criteria in task description
   - Solution: Mention the `finish` command in instructions

### Debug Commands

```bash
# Check OpenHands installation
pip list | grep openhands

# Test basic import
python -c "import openhands; print('OK')"

# Check available Python files in workspace
ls *.py

# View git status after test
git status
git log -1 --stat
```

## Cost Management

- Simple comment addition tasks: ~$0.05-0.10
- Medium optimization tasks: ~$0.50-2.00
- Complex refactoring: ~$2.00-10.00

Always test with `--real` flag carefully as it incurs actual API costs.

## Files Modified

This solution required changes to:

1. **`test_integration.py`**: Updated iteration limits and task descriptions
2. **`bench/agents/openhands_cli.py`**: Enhanced task completion criteria
3. **`simple.py`**: Created target file for testing
4. **Environment**: Used bench-env with pre-installed OpenHands

## Contributing

When working on OpenHands integration:

1. Always test with the safe mode first: `python test_integration.py`
2. Use specific, actionable task descriptions
3. Include clear completion criteria
4. Test with realistic iteration limits
5. Document any new patterns or issues discovered

## Support

If you encounter issues:

1. Check that you're using the bench-env virtual environment
2. Verify OpenHands installation with `pip list | grep openhands`
3. Review the task description for clarity and specificity
4. Check iteration limits are reasonable for task complexity
5. Ensure target files exist and are accessible

The key insight is that OpenHands needs explicit guidance on when and how to complete tasks, combined with a proper execution environment.
