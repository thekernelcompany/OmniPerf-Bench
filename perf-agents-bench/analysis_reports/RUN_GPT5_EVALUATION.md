# Running GPT-5 Evaluation on All Patches

This document explains how to run the comprehensive GPT-5 evaluation on all 300 attempts.

## Prerequisites

1. **OpenAI API Key**: Set the `OPENAI_API_KEY` environment variable
2. **Python Dependencies**: Ensure `openai` package is installed
3. **Data Files**: Ensure `all_attempts_comprehensive.json` exists

## Quick Start

```bash
cd /home/raven/coding-mess/kernel-corp/OmniPerf-Bench/perf-agents-bench

# Set API key
export OPENAI_API_KEY="your-api-key-here"

# Run evaluation (starts with 10 commits, first 3 attempts each)
python3 evaluate_patches_with_gpt5.py
```

## What It Does

The script:

1. **Loads Commit Data**: Reads human commit optimizations from `tmp_single_commit/*.json`
2. **Loads Attempt Data**: Reads all 300 attempts from `all_attempts_comprehensive.json`
3. **For Each Commit**:
   - Finds all attempts for that commit
   - For each attempt (up to first 3):
     - Loads generated patch
     - Loads task prompt
     - Sends to GPT-5 for evaluation
     - GPT-5 compares patch to human commit
     - Returns evaluation with:
       - Syntax correctness
       - Files match
       - Logic match
       - Completeness
       - Template code detection
       - Quality score
       - Issues list
       - Reasoning
4. **Saves Results**: Writes to `gpt5_evaluations.json`

## Evaluation Criteria

GPT-5 evaluates each patch on:

1. **Syntax Correctness**: No Python syntax errors
2. **Files Match**: Modifies same files as human commit
3. **Optimization Logic Match**: Implements same optimization
4. **Completeness**: Addresses full optimization (not partial)
5. **Code Quality**: Actual code (not template/example)
6. **True Success**: All criteria met

## Output Format

Results are saved as JSON:

```json
[
  {
    "commit": "8aa1485fcff7be3e42300c0615ee0f3f3cbce9a8",
    "subject": "[Perf] Disable chunked local attention...",
    "human_files": ["vllm/config.py", "vllm/envs.py"],
    "total_attempts": 21,
    "evaluated_attempts": [
      {
        "run_id": "vllm_core-3368ff88",
        "item_id": "vllm_core-0000",
        "status": "success",
        "patch_size": 8411,
        "evaluation": {
          "syntax_correct": false,
          "files_match": true,
          "optimization_logic_match": true,
          "is_complete": false,
          "is_template_code": false,
          "quality_score": 0.6,
          "issues": ["Syntax error: ...", "Over-optimization: ..."],
          "reasoning": "Detailed explanation...",
          "true_success": false
        }
      }
    ],
    "summary": {
      "marked_successful": 12,
      "true_successful": 0
    }
  }
]
```

## Cost Estimation

**Per Evaluation:**
- Input tokens: ~2000-3000 (human commit + patch + prompt)
- Output tokens: ~500-1000 (evaluation JSON)
- Cost per evaluation: ~$0.01-0.02

**Full Evaluation (300 attempts):**
- Estimated cost: $3-6
- Time: ~1-2 hours (with rate limiting)

## Rate Limiting

The script includes a 1-second delay between evaluations to avoid rate limits. Adjust if needed:

```python
time.sleep(1)  # In evaluate_patch_with_gpt5()
```

## Extending the Evaluation

To evaluate more commits/attempts:

```python
# In main(), change:
for commit_hash, attempts in list(attempts_by_commit.items())[:10]:  # Change 10 to more
    for i, attempt in enumerate(attempts[:3], 1):  # Change 3 to more
```

## Using Results

After evaluation, use `gpt5_evaluations.json` to:

1. **Calculate True Success Rate**:
   ```python
   total_true = sum(r["summary"]["true_successful"] for r in results)
   total_marked = sum(r["summary"]["marked_successful"] for r in results)
   true_rate = total_true / total_marked
   ```

2. **Identify Common Issues**:
   - Syntax errors
   - Template code
   - Incomplete patches
   - Wrong files

3. **Generate Report**: Use results to create comprehensive analysis document

## Troubleshooting

**Error: OPENAI_API_KEY not set**
- Set environment variable: `export OPENAI_API_KEY="your-key"`

**Error: File not found**
- Ensure `all_attempts_comprehensive.json` exists
- Run data extraction scripts first

**Rate Limit Errors**
- Increase `time.sleep()` delay
- Use exponential backoff

**JSON Parse Errors**
- GPT-5 sometimes returns markdown-wrapped JSON
- Script handles this, but may need adjustment

## Next Steps

After running evaluation:

1. **Analyze Results**: Review `gpt5_evaluations.json`
2. **Update Analysis Document**: Add GPT-5 evaluations to `COMPREHENSIVE_PER_COMMIT_ANALYSIS.md`
3. **Calculate Statistics**: True success rate, common issues, cost per success
4. **Generate Report**: Create final research-quality analysis document


