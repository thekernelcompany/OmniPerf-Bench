---
configs:
  - config_name: default
    data_files:
      - split: vllm
        path: vllm.parquet
      - split: sglang
        path: sglang.parquet
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - code
  - performance
  - optimization
  - benchmark
  - llm-inference
  - vllm
  - sglang
size_categories:
  - n<1K
---

# OmniPerf-Bench

A benchmark dataset for evaluating AI agents on software performance optimization tasks.

## Dataset Description

OmniPerf-Bench contains **170 real-world performance optimization commits** from two major AI inference libraries:

| Split | Records | Description |
|-------|---------|-------------|
| `vllm` | 96 | High-throughput LLM serving engine |
| `sglang` | 74 | Fast serving framework for LLMs |

Each record includes the original code change (diff) and a test script to measure performance improvement.

## Usage

```python
from datasets import load_dataset

# Load the full dataset
ds = load_dataset("YOUR_USERNAME/omniperf-bench")
print(ds)
# DatasetDict({
#     vllm: Dataset({ num_rows: 96 })
#     sglang: Dataset({ num_rows: 74 })
# })

# Load vLLM data only
vllm_data = load_dataset("YOUR_USERNAME/omniperf-bench", split="vllm")

# Load SGLang data only
sglang_data = load_dataset("YOUR_USERNAME/omniperf-bench", split="sglang")

# Access samples
print(vllm_data[0]["commit_subject"])
print(sglang_data[0]["test_script"][:500])
```

## Dataset Fields

| Field | Description |
|-------|-------------|
| `commit_hash` | Git commit SHA |
| `commit_subject` | One-line commit summary |
| `commit_message` | Full commit message |
| `commit_date` | Date of the commit |
| `diff_text` | Full unified diff of code changes |
| `test_script` | Python script to measure performance |
| `repo` | Repository name (`vllm` or `sglang`) |
| `pr_url` | Pull request URL |
| `apis` | Affected APIs |
| `files_changed` | List of modified files |
| `functions_changed` | List of modified functions |
| `has_performance` | Whether commit impacts performance |
| `perf_command` | Command to run performance test |

## Purpose

This dataset enables:
- Benchmarking AI coding agents on performance optimization
- Studying how expert developers optimize ML inference code
- Training models to identify and implement performance improvements

## Citation

If you use this dataset, please cite:

```bibtex
@misc{omniperf-bench,
  title={OmniPerf-Bench: A Benchmark for AI-Driven Software Performance Optimization},
  year={2024},
  url={https://github.com/YOUR_USERNAME/OmniPerf-Bench}
}
```

## License

Apache 2.0
