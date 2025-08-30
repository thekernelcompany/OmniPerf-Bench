# Performance Test Script Generator Prompt

You are an expert performance engineer. You will generate a SINGLE, FOCUSED performance test script (prob_script) that accurately measures the real-world performance impact of the optimization described in the commit.

## Instructions

1. **Analyze the Commit Thoroughly**: Parse the commit message and diff to understand:
   - What specific optimization is being implemented (e.g., algorithm improvement, kernel optimization, caching)
   - What performance improvement is claimed and what metric it's measured in
   - Which specific functions/methods/classes are being modified
   - What data types, shapes, and computational patterns are involved
   - What hardware requirements or dependencies exist
   - What the before/after code patterns look like

2. **Determine Correct Imports**: Based on commit diff analysis:
   - Identify the repository and specific modules being modified
   - Import the actual optimized functions/classes from the commit
   - Handle import errors gracefully with clear error messages
   - Example: `from repository.module import OptimizedFunction`

3. **Setup Realistic Workload**: Write a function `setup()` that creates appropriate test data:
   - Analyze the commit to determine realistic data dimensions and types
   - Use scales that match the optimization's target use case
   - Create data that will actually trigger the optimized code path
   - Set random seeds for reproducibility
   - Handle different data types (tensors, arrays, objects) appropriately
   - DO NOT CREATE WORKLOADS THAT CAN BE EASILY HACKED FOR PERFORMANCE

4. **Define Performance Experiment**: Write an `experiment()` function that:
   - Contains ONLY the performance-critical code path being optimized
   - Uses the exact function calls and patterns from the commit diff
   - Represents the specific optimization scenario being tested
   - Focuses on the actual computational kernel being improved
   - AVOIDS any setup/download code (that's in setup function)

5. **Implement Result Storage**: Write `store_result()` and `load_result()` functions:
   - Analyze the result type from experiment() to choose appropriate serialization
   - Use torch.save/load for PyTorch tensors, json for simple data structures
   - Store essential properties needed for equivalence checking
   - Handle repository-specific serialization APIs when available
   - Include metadata (shape, dtype, device) for tensor results

6. **Add Equivalence Checking**: Write `check_equivalence()` function:
   - Takes two results (reference and current) from experiment function
   - Performs comprehensive assertions on important properties
   - Uses appropriate tolerances for floating-point comparisons
   - Handles type conversions and serialization artifacts
   - Assumes reference result is always correct

7. **Create Main Test Function**: Write `run_test()` with exact signature:
   ```python
   def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
   ```
   - Calls setup() once at the beginning to prepare data
   - Detects hardware availability and chooses appropriate timing method
   - Uses CUDA events for GPU workloads, time.time for CPU workloads
   - Includes warmup iterations for stable measurements
   - Runs sufficient iterations for statistical stability
   - Handles equivalence checking when requested
   - Returns average execution time in milliseconds

8. **Hardware and API Integration**:
   - Import actual functions based on commit diff analysis
   - Check for required hardware/capabilities dynamically
   - Gracefully handle cases where optimization isn't supported
   - Use appropriate data types and computational resources

## Required Output Structure

Generate a SINGLE Python script that contains the following functions:

```python
#!/usr/bin/env python3
"""
Performance test script for commit: {commit_hash}
{commit_message}

This script measures the actual performance of the optimization described in the commit.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple

# Import the actual optimized modules from the commit
# CRITICALLY IMPORTANT: Analyze the commit diff to determine correct imports
# Look at the repository structure and modified files to import the right modules
# Example patterns based on common repositories:
# from transformers import AutoModelForCausalLM  # For HuggingFace models
# from torch.nn import functional as F            # For PyTorch operations
# from sklearn.ensemble import RandomForestClassifier  # For ML algorithms
# from numpy import linalg                          # For numerical computations
try:
    # Replace with actual imports based on commit analysis
    # These will vary for each commit - DO NOT copy this blindly
    import torch
    import numpy as np
except ImportError as e:
    print(f"Required modules not available: {e}")
    exit(1)

def setup() -> Dict[str, Any]:
    """Create realistic workload that exercises the optimization"""
    torch.manual_seed(42)
    np.random.seed(42)

    # Create workload based on the specific optimization being tested
    # Analyze commit to determine appropriate data dimensions and types
    # Example dimensions - adjust based on the actual optimization:
    batch_size = 32
    seq_len = 1024
    hidden_dim = 4096

    # Create input data appropriate for the optimization
    x = torch.randn((batch_size, hidden_dim), dtype=torch.float16)

    # Move to GPU if optimization requires it
    if torch.cuda.is_available():
        x = x.cuda()

    return {
        'x': x,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'hidden_dim': hidden_dim,
    }

def experiment(data: Dict[str, Any]) -> Any:
    """
    Execute the performance-critical code path being optimized.

    Args:
        data: Dictionary containing setup data from setup() function

    Returns:
        Result of the optimized computation
    """
    # CRITICALLY IMPORTANT: Replace this with the actual optimized code from the commit
    # Analyze the commit diff to identify the exact function calls and patterns
    #
    # Examples of what this might look like for different optimizations:
    #
    # For a matrix multiplication optimization:
    # result = torch.matmul(data['input_a'], data['input_b'])
    #
    # For a transformer attention optimization:
    # attention_output = F.scaled_dot_product_attention(data['query'], data['key'], data['value'])
    #
    # For a numerical algorithm optimization:
    # result = np.linalg.solve(data['matrix_a'], data['vector_b'])
    #
    # For a machine learning model optimization:
    # outputs = model(data['input_tensor'])
    #
    # Always use the EXACT function calls and patterns from the commit diff

    # Placeholder - replace with actual optimized code:
    with torch.no_grad():
        result = torch.matmul(data['x'], data['x'].t())

    return result

def store_result(result: Any, filepath: str) -> None:
    """Store result for future equivalence checking"""
    # Use torch.save for tensor results
    torch.save({
        'result': result,
        'shape': result.shape,
        'dtype': result.dtype,
        'device': result.device,
        # Store sample values for equivalence checking
        'sample_values': result.flatten()[:100].cpu().numpy()
    }, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result for equivalence checking"""
    data = torch.load(filepath)
    return data['result']

def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Check that current result is equivalent to reference"""
    # Check shapes
    assert current_result.shape == reference_result.shape, f"Shape mismatch: {current_result.shape} vs {reference_result.shape}"

    # Check dtypes
    assert current_result.dtype == reference_result.dtype, f"Dtype mismatch: {current_result.dtype} vs {reference_result.dtype}"

    # Check devices
    assert current_result.device == reference_result.device, f"Device mismatch: {current_result.device} vs {reference_result.device}"

    # Check numerical values with appropriate tolerance
    torch.testing.assert_close(
        current_result, reference_result,
        rtol=1e-3, atol=1e-6,
        msg="Results are not numerically equivalent"
    )

def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """
    Run the performance test and return execution time.

    Args:
        eqcheck: Whether to perform equivalence checking
        reference: Whether to store result as reference
        prefix: Prefix for reference files

    Returns:
        Average execution time in milliseconds
    """
    # Setup data once
    data = setup()

    # Check hardware availability (adapt based on optimization requirements)
    if torch.cuda.is_available():
        # Use CUDA events for precise GPU timing
        torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = experiment(data)

        # Measure performance using CUDA events
        num_iterations = 50
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()

            for _ in range(num_iterations):
                result = experiment(data)

            end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
    else:
        # Fallback to CPU timing if GPU not available
        import time
        num_iterations = 10  # Fewer iterations for CPU

        # Warmup
        for _ in range(3):
            _ = experiment(data)

        start_time = time.time()
        for _ in range(num_iterations):
            result = experiment(data)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Handle equivalence checking
    if eqcheck or reference:
        result = experiment(data)  # Get fresh result for checking

        if reference:
            store_result(result, f'{prefix}_reference.pt')
        elif eqcheck:
            reference_result = load_result(f'{prefix}_reference.pt')
            check_equivalence(result, reference_result)

    return elapsed_time / num_iterations

# The harness will be automatically appended here with argument parsing
```

## Critical Requirements

### Commit-Specific Adaptation Requirements
- **Import Analysis**: Every script must analyze the commit diff to determine correct imports
- **Function Call Matching**: Use EXACT function calls and patterns from the commit
- **Data Type Awareness**: Match data types and shapes used in the optimization
- **Hardware Requirements**: Identify and handle specific hardware dependencies
- **Optimization Path**: Ensure the test triggers the specific optimized code path

### Function Structure Requirements
- **`setup()`**: Must adapt data creation based on commit analysis
- **`experiment(data)`**: Contains ONLY the performance-critical code from the commit
- **`run_test()`**: Standard signature, adapts timing method based on hardware
- **`store_result()`/`load_result()`**: Choose serialization based on result type
- **`check_equivalence()`**: Comprehensive assertions adapted to result structure

### Performance Measurement Standards
- **Adaptive Timing**: Use CUDA events for GPU workloads, time.time for CPU
- **Hardware Detection**: Automatically detect available hardware and capabilities
- **Statistical Stability**: Sufficient iterations for reliable measurements
- **Warmup Handling**: Include appropriate warmup for the specific optimization
- **Unit Consistency**: Return time in milliseconds regardless of timing method

### Equivalence Checking Standards
- **Type-Aware Validation**: Adapt assertions based on result data types
- **Tolerance Selection**: Choose appropriate tolerances for numerical comparisons
- **Serialization Handling**: Handle conversions from different serialization formats
- **Property Validation**: Check relevant properties (shape, dtype, device, values)
- **Reference Integrity**: Assume reference result represents correct behavior

### Hardware Awareness Requirements
- **Dynamic Detection**: Check hardware availability at runtime
- **Capability Matching**: Ensure hardware meets optimization requirements
- **Graceful Fallbacks**: Handle cases where optimization isn't supported
- **Resource Optimization**: Use appropriate computational resources

### Data Quality Standards
- **Realistic Workloads**: Create data that matches the optimization's use case
- **Optimization Triggering**: Ensure data actually exercises the optimized path
- **Diversity**: Generate challenging, non-uniform data patterns
- **Reproducibility**: Use random seeds for consistent results

## Integration Notes

- **Harness Compatibility**: The script will be automatically appended with argument parsing harness
- **Import Handling**: Use try/except blocks for optional module imports
- **Error Handling**: Provide clear error messages for missing dependencies
- **File Paths**: Use provided `prefix` parameter for reference file storage

Generate a SINGLE, focused performance script that accurately measures the real-world impact of the optimization while maintaining full equivalence checking capabilities.
