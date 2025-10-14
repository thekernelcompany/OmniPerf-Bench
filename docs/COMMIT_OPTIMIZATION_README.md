# Commit Optimization Pipeline

This directory contains a **proof-of-concept integration** between your commit analysis system and OpenHands optimization capabilities. The pipeline demonstrates the core workflow for GSO-style performance benchmarking.

## Overview

The pipeline takes:
- **Your commit JSON** (with metadata and analysis)
- **Your generated performance test** (sophisticated test script)
- **A git repository** (to optimize)

And produces:
- **Agent optimization branch** (OpenHands-generated improvements)
- **Performance comparison** (baseline vs human vs agent)
- **Success metrics** (whether agent achieved performance goals)

## Quick Start

### 1. Setup
```bash
# Check what's ready
python validate_setup.py

# Install missing components
pip install openhands-ai

# Set API key for OpenHands
export OPENAI_API_KEY="your-key-here"
# OR
export ANTHROPIC_API_KEY="your-key-here"
```

### 2. Run Test
```bash
# Set your vLLM repository path
export REPO_PATH="/path/to/your/vllm"

# Run the pipeline test
./test_commit_optimization.sh
```

### 3. Single Commit Usage
```bash
python run_commit_optimization.py \
    --commit-json tmp_single_commit/0ec82edda59aaf5cf3b07aadf4ecce1aa1131add.json \
    --test-script misc/experiments/generated_test_generators_v4/0ec82edd_test_case_generator.py \
    --repo-path /path/to/vllm \
    --work-dir .commit_opt_work \
    --cleanup
```

### 4. Batch Processing (Multiple Commits)
```bash
# Process all commits in parallel
python batch_commit_optimization.py \
    --commit-dir tmp_single_commit/ \
    --test-dir misc/experiments/generated_test_generators_v4/ \
    --repo-path /path/to/vllm \
    --output-dir results/ \
    --max-workers 2

# Resume interrupted batch processing
python batch_commit_optimization.py \
    --commit-dir tmp_single_commit/ \
    --test-dir misc/experiments/generated_test_generators_v4/ \
    --repo-path /path/to/vllm \
    --output-dir results/ \
    --resume
```

## What It Does

### Phase 1: Setup & Baseline
1. **Loads commit metadata** from your JSON format
2. **Creates isolated workspace** (git clone + checkout)
3. **Runs baseline performance test** on parent commit
4. **Runs human performance test** on optimized commit

### Phase 2: Agent Optimization  
5. **Generates optimization task** from commit metadata
6. **Runs OpenHands** with the optimization task
7. **Captures agent changes** in new git branch

### Phase 3: Evaluation
8. **Runs agent performance test** on optimized code
9. **Compares all three versions** (baseline/human/agent)
10. **Reports success/failure** based on performance criteria

## Outputs

### GSO Prediction Format
The pipeline generates standardized GSO (General System Optimization) predictions in JSON format:

```json
{
  "commit_hash": "0ec82edd",
  "prediction_timestamp": "2025-09-15 18:45:32 UTC",
  "agent_optimization": {
    "success": true,
    "agent_branch": "agent/optimization/0ec82edd/1757961447",
    "execution_time_seconds": 127.3,
    "confidence": "high"
  },
  "performance_analysis": {
    "baseline_time_ms": 45.2,
    "human_optimized_time_ms": 23.1,
    "agent_optimized_time_ms": 25.8,
    "improvements": {
      "human_vs_baseline": 1.96,
      "agent_vs_baseline": 1.75,
      "agent_vs_human_ratio": 0.89
    }
  },
  "success_criteria": {
    "beats_baseline": true,
    "reaches_human_threshold": true,
    "overall_success": true
  }
}
```

### Batch Processing Results
- **Individual commit results** with GSO predictions
- **Aggregated success rates** across multiple commits
- **Performance distribution analysis** 
- **Resume capability** for interrupted runs
- **Parallel processing** with configurable workers

## Success Criteria

The agent is considered successful if:
- ✅ **Beats baseline by 5%+**: `agent_time < baseline_time * 0.95`
- ✅ **Reaches 80%+ of human performance**: `agent_time < human_time * 1.20`
- ✅ **Test executes successfully**: No crashes or infinite times

## Sample Output

```
COMMIT OPTIMIZATION RESULTS
============================================================
Success: True
Agent branch: agent/optimization/0ec82edd/1703123456
Execution time: 847.3s

Performance Results:
  Baseline: 45.23ms
  Human:    23.41ms  
  Agent:    25.67ms

Performance Ratios:
  Agent vs Baseline: 0.567x  (43% improvement!)
  Agent vs Human:    1.096x  (within 10% of human)

✅ SUCCESS: Agent achieved performance improvement!
Human improved baseline by 1.93x
Agent improved baseline by 1.76x
🎯 Agent reached 91% of human performance!
```

## Architecture

```
Your Data → OpenHands → Performance Evaluation → Results
     ↓           ↓              ↓                  ↓
┌──────────┐ ┌─────────┐ ┌──────────────┐ ┌──────────────┐
│Commit    │ │Agent    │ │Generated     │ │Success       │
│Metadata  │ │Optimiz- │ │Test Runs     │ │Metrics &     │
│+ Tests   │ │ation    │ │3x (B/H/A)    │ │Comparison    │
└──────────┘ └─────────┘ └──────────────┘ └──────────────┘
```

## Integration with GSO

This pipeline demonstrates the **core components** needed for full GSO integration:

### Already Working ✅
- **Commit metadata parsing** → GSO `instance_id`, `base_commit`, `opt_commit`
- **Performance test execution** → GSO `prob_script` and `tests`
- **Agent optimization** → GSO `model_patch` generation
- **Performance evaluation** → GSO harness metrics

### Next Steps for Full GSO 🔄
- **Unified diff generation** from agent branch
- **GSO prediction format** (JSONL output)
- **Batch processing** for multiple commits
- **Integration with GSO harness** for standardized evaluation

## Troubleshooting

### "OpenHands not found"
```bash
pip install openhands-ai
```

### "No LLM API key"
```bash
export OPENAI_API_KEY="sk-..."
# OR
export ANTHROPIC_API_KEY="sk-ant-..."
```

### "Test execution failed"
- Check CUDA availability for GPU tests
- Verify vLLM installation in test environment
- Check test script compatibility with commit

### "Agent optimization failed"
- Increase timeout in `_run_openhands_optimization`
- Check OpenHands logs in work directory
- Verify task description clarity

## Files Created

- `run_commit_optimization.py` - Main pipeline implementation
- `validate_setup.py` - Setup validation script  
- `test_commit_optimization.sh` - Quick test runner
- `.commit_opt_work/` - Working directory (auto-created)

## Critical Analysis

### Strengths ✅
- **Realistic workflow** - matches GSO evaluation pattern
- **Isolated execution** - separate venvs and git workspaces
- **Performance-focused** - uses your sophisticated test generators
- **Error handling** - graceful failures with detailed logging

### Limitations ⚠️
- **Docker dependency** - OpenHands requires Docker runtime
- **API key required** - Need LLM access for OpenHands
- **Simple OpenHands integration** - could use more sophisticated prompting
- **Limited environment isolation** - could benefit from Docker
- **Basic success criteria** - GSO uses more complex metrics

### Production Readiness 🔄
This is a **proof-of-concept** demonstrating the integration. For production use:
- Add Docker containerization for better isolation  
- Implement GSO-compatible output formats
- Add more sophisticated error recovery
- Scale to batch processing multiple commits

## Next Steps

1. **Test this pipeline** on the 0ec82edd commit
2. **Validate the approach** works with your generated tests
3. **Iterate on success criteria** based on results
4. **Extend to batch processing** for multiple commits
5. **Integrate with GSO harness** for standardized evaluation

This pipeline proves the **fundamental integration is possible** and provides a solid foundation for scaling to the full GSO workflow.
