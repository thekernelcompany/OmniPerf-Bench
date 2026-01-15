# InferenceBench Test Generation Scripts

This dataset contains automatically generated performance test scripts for the InferenceBench project. These scripts are designed to measure and benchmark the performance impact of various optimizations in machine learning inference systems.

## Dataset Overview

- **Total Scripts**: 178 Python test generators
- **Categories**: 2 main collections
- **Domain**: Machine Learning Performance Testing
- **Target Systems**: Inference optimization benchmarks

## Directory Structure

### `generated_test_generators_v4/` (82 files)
This directory contains the fourth version of automatically generated test scripts. These are the latest iteration of test generators with improved:
- Performance measurement accuracy
- Hardware detection capabilities
- Cross-platform compatibility
- Equivalence checking mechanisms

### `working_test_generators/` (96 files)
This directory contains verified and validated test generators that have been successfully tested in production environments. These scripts include:
- Comprehensive error handling
- Robust mock implementations
- Advanced timing mechanisms
- Multi-environment support

## Script Features

Each test generator script typically includes:

- **Deterministic Setup**: Ensures reproducible results across runs
- **Hardware Detection**: Automatically detects GPU/CPU capabilities
- **Target Resolution**: Dynamically imports and tests optimization targets
- **Performance Timing**: High-precision timing with statistical analysis
- **Equivalence Checking**: Validates functional correctness of optimizations
- **Result Storage**: Serializable output for comparative analysis

## Usage

Each script follows a standardized interface:

```python
python script_name.py [--eqcheck] [--reference] [--prefix PREFIX]
```

- `--eqcheck`: Enable equivalence checking against reference results
- `--reference`: Generate reference results for future comparisons
- `--prefix`: Add prefix to output files

## Output Format

Scripts output JSON with performance metrics:

```json
{
  "impl_tag": "child",
  "commit_hash": "abc123...",
  "device": "cuda",
  "dtype": "torch.float16",
  "avg_ms": 1.23,
  "p50_ms": 1.20,
  "p95_ms": 1.45,
  "eq_level": "numeric"
}
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Hardware-specific dependencies (CUDA for GPU tests)

## Contributing

These scripts are part of the InferenceBench project. For issues or contributions, please refer to the main InferenceBench repository.

## License

This dataset is released under the same license as the InferenceBench project.

## Citation

If you use these test generators in your research, please cite:

```bibtex
@dataset{inferencebench_test_generators,
  title={InferenceBench Test Generation Scripts},
  author={InferenceBench Team},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/Inferencebench/test-generation-scripts}
}
```

## Technical Details

### Test Categories

The scripts cover various optimization categories:
- Memory management optimizations
- Attention mechanism improvements
- Batching and scheduling optimizations
- Model serving enhancements
- Hardware-specific optimizations
- Communication and synchronization improvements

### Measurement Methodology

- **Warmup Iterations**: 3-10 iterations to stabilize performance
- **Measurement Iterations**: 10-100 iterations for statistical significance
- **Timing Methods**: CUDA events for GPU, high-resolution counters for CPU
- **Statistics**: Mean, median, 95th percentile, standard deviation

### Validation Approach

- **Functional Equivalence**: Numerical comparison of outputs
- **Performance Regression**: Comparison against baseline implementations
- **Cross-platform Testing**: Validation across different hardware configurations
