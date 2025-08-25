---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- code
- optimization
- vllm
- performance
size_categories:
- n<1K
---

# vLLM GSO Performance Optimization Dataset

This dataset contains performance optimization tasks extracted from the [vLLM](https://github.com/vllm-project/vllm) project using the [GSO (Global Software Optimization)](https://github.com/gso-bench/gso) framework.

## Dataset Statistics

- **Total Problems**: 282
- **Performance Commits**: 129
- **Unique APIs**: 282

## Dataset Contents

### Files

1. **`vllm_problems.json`**: Main dataset file containing performance test problems
   - Each problem includes setup code, test code, and metadata
   - Tests are designed to measure performance improvements

2. **`vllm_commits.json`**: Performance-related commits extracted from vLLM
   - Filtered from the full commit history using LLM analysis
   - Contains commit messages, diffs, and statistics

3. **`vllm_ac_map.json`**: API-Commit mapping
   - Maps each commit to the high-level APIs it affects
   - Used to organize tests by API

4. **`vllm_experiment.yaml`**: Configuration used for dataset generation
   - Includes installation commands and API documentation

### Dataset Format

The main dataset is also available in Parquet format for easy loading:

```python
from datasets import load_dataset

dataset = load_dataset("onethree/vllm-gso-problems")
```

Each record contains:
- `problem_id`: Unique identifier
- `repo_name`: Repository name (vllm)
- `api`: The vLLM API being optimized
- `base_commit`: Starting commit for optimization
- `target_commit`: Commit with optimization
- `test_script`: Python script to measure performance
- `install_commands`: Commands to set up the environment

## Usage

### Loading the Dataset

```python
# Load as HuggingFace dataset
from datasets import load_dataset
dataset = load_dataset("onethree/vllm-gso-problems")

# Or load JSON directly
import json
with open("vllm_problems.json", "r") as f:
    problems = json.load(f)
```

### Running Performance Tests

Each test script is self-contained and measures the performance of a specific vLLM API:

```python
# Example test structure
def setup():
    # Prepare workload
    pass

def experiment():
    # Run vLLM operation
    pass

def run_test(eqcheck=False, reference=False, prefix=''):
    # Measure performance
    pass
```

## Generation Process

This dataset was created using the GSO collection framework:

1. **Commit Extraction**: Analyzed vLLM's git history to find performance-related commits
2. **API Identification**: Mapped commits to affected vLLM APIs using embeddings
3. **Test Generation**: Used GPT-4 to generate performance tests for each optimization
4. **Validation**: Tests include equivalence checking to ensure correctness

## License

This dataset is released under the MIT license, consistent with the vLLM project.

## Citation

If you use this dataset, please cite both GSO and vLLM:

```bibtex
@software{vllm2023,
  title = {vLLM: Easy, Fast, and Cheap LLM Serving},
  author = {vLLM Team},
  year = {2023},
  url = {https://github.com/vllm-project/vllm}
}
```
