# OmniPerf-Bench Dataset

Performance optimization benchmark for AI coding agents.

## Dataset Summary

| Config | Records | Description |
|--------|---------|-------------|
| `vllm` | 96 | vLLM performance optimization commits with tests |
| `sglang` | 74 | SGLang performance optimization commits with tests |
| `all` | 170 | Combined dataset (vLLM + SGLang) |

## Usage

```python
from datasets import load_dataset

# Load vLLM data only
ds_vllm = load_dataset("parquet", data_dir="path/to/omniperf_bench/vllm")

# Load SGLang data only
ds_sglang = load_dataset("parquet", data_dir="path/to/omniperf_bench/sglang")

# Load all data
ds_all = load_dataset("parquet", data_dir="path/to/omniperf_bench/all")

# Access the data
print(ds_vllm['train'][0])  # First vLLM record
print(ds_sglang['train'][0])  # First SGLang record
```

## Columns

| Column | Description |
|--------|-------------|
| `commit_hash` | Git commit hash |
| `commit_message` | Full commit message |
| `commit_subject` | Commit subject line |
| `commit_date` | Commit date |
| `diff_text` | Full diff of the commit |
| `files_changed` | List of changed files |
| `functions_changed` | List of changed functions |
| `apis` | Affected APIs |
| `test_script` | Python test script for measuring performance |
| `repo` | Repository name ("vllm" or "sglang") |
| `pr_url` | Pull request URL |
| `has_performance` | Whether commit has performance impact |
| `perf_command` | Performance testing command |
| `llm_reason` | LLM analysis of the commit |
| `llm_api_reason` | LLM API impact analysis |

## File Structure

```
omniperf_bench/
├── vllm/
│   └── train-00000-of-00001.parquet   (96 records)
├── sglang/
│   └── train-00000-of-00001.parquet   (74 records)
├── all/
│   └── train-00000-of-00001.parquet   (170 records)
├── dataset_info.json
└── README.md
```

## License

See repository for license information.
