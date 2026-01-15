### alpha-vllm-99-commits

**Overview**: 99 vLLM commits merged and filled. Primary key: `commit_hash`.

- **Sources**:
  - `vllm_pr_timeline_data.jsonl` (timeline)
  - `vllm_pr_analysis.jsonl` (flags/tests)
  - `lm_eval_merged.jsonl` (lm-eval/perf)
  - `vllm_commits (1).json` (commit metadata/diffs)
  - Filter: `human_vllm_commit.txt` (approved 99)
- **Join policy**: Full outer on `commit_hash`. For overlapping fields, sensible precedence; otherwise `null`.
- **Null semantics**: Missing -> `null`. Non-model-specific commits have `models=['N/A']`.
- **LLM filling**: Claude Opus 4.1 via Bedrock filled `pr_date`, `models`, `lm_eval_commands`, `perf_command` using a rubric (Pydantic-validated, extended thinking; synthesizes commands when implied).
- **Verification**: `scripts/verify_non_nulls_with_opus.py` compares LLM inferences to existing non-nulls.

**Rows**: 99  
**Unique commit hashes**: 99

### Fields
- `affected_paths`
- `analysis_extracted_at`
- `apis`
- `commit_date`
- `commit_hash`
- `commit_message`
- `commit_subject`
- `diff_text`
- `files_changed`
- `functions_changed`
- `has_general_test`
- `has_lm_eval`
- `has_performance`
- `has_serving`
- `llm_api_reason`
- `llm_reason`
- `lm_eval_commands`
- `models`
- `perf_command`
- `pr_date`
- `pr_url`
- `repo_path`
- `stats`
- `test_details`
- `timeline_extracted_at`
- `timeline_text`

### Non-null counts
- **affected_paths**: 99
- **analysis_extracted_at**: 99
- **apis**: 99
- **commit_date**: 99
- **commit_hash**: 99
- **commit_message**: 99
- **commit_subject**: 99
- **diff_text**: 99
- **files_changed**: 99
- **functions_changed**: 99
- **has_general_test**: 99
- **has_lm_eval**: 99
- **has_performance**: 99
- **has_serving**: 99
- **llm_api_reason**: 99
- **llm_reason**: 99
- **lm_eval_commands**: 42
- **models**: 99
- **perf_command**: 61
- **pr_date**: 88
- **pr_url**: 99
- **repo_path**: 99
- **stats**: 99
- **test_details**: 99
- **timeline_extracted_at**: 98
- **timeline_text**: 99

### Load
```python
from datasets import load_dataset
ds = load_dataset("Inferencebench/alpha-vllm-99-commits", split="train")
```

### Files
- `combined_vllm_pr_dataset.filled.jsonl` (raw combined, filled)
- `combined_vllm_pr_dataset_filled_summary.json` (coverage summary)