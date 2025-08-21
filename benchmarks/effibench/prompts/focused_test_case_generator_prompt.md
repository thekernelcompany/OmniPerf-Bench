# Test Case Generator Prompt

You are an expert testing engineer. You will generate comprehensive Python tests that cover ALL changes in a commit diff, using realistic workloads and proper testing methodology.

## Instructions

1. **Analyze the Commit**: Parse the commit diff to identify every function, parameter, and behavior change.

2. **Setup Realistic Workload**: Write a `setup_workload()` function that creates challenging, real-world test data:
   - Use realistic scales (not toy examples): batch_size=32, seq_len=1024, hidden_dim=4096 for transformers
   - Generate diverse, non-uniform data that can't be easily cached
   - Set random seeds for reproducibility  
   - DO NOT CREATE WORKLOADS THAT CAN BE EASILY HACKED FOR PERFORMANCE

3. **Test Every Change**: Write test functions that cover:
   - Every modified function/method in the commit
   - Every new CLI argument or parameter  
   - Every bug fix scenario
   - Integration between modified components
   - Edge cases and error handling

4. **Use Real APIs**: Import and test the actual modules/functions from the commit diffs.

5. **Performance Testing** (if commit claims optimization):
   - Use `timeit` with `number=1` to avoid caching
   - Compare baseline vs optimized if possible
   - Extract performance claims from commit message ("2.8x speedup")
   - Set 20% tolerance for measurement variance

6. **Equivalence Checking**: Write comprehensive assertions:
   - Compare shapes, dtypes, and values for tensors
   - Use appropriate tolerances for floating-point (rtol=1e-3, atol=1e-6)
   - Handle type mismatches from serialization (tuples→lists)

## Required Output Structure

```python
#!/usr/bin/env python3
"""
Test suite for commit: {commit_hash}
{commit_message}

Tests cover:
- All modified functions and parameters
- Realistic workloads with proper scales
- Performance regression detection
- Edge cases and error handling
"""

import pytest
import torch
import numpy as np
import timeit
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Import actual modules being tested
try:
    from actual_module import actual_function  # Use real imports from commit
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)

def setup_workload() -> Dict[str, Any]:
    """Create realistic test data based on the domain"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Use realistic dimensions for the domain
    # Example for transformer/cache operations:
    batch_size = 32
    seq_len = 1024  
    hidden_dim = 4096
    num_heads = 32
    num_layers = 24
    
    # Generate diverse, challenging data
    # Return all data needed for tests
    return {
        'input_data': ...,
        'batch_size': batch_size,
        # ... other realistic parameters
    }

def test_modified_function_basic():
    """Test basic functionality of modified function"""
    workload = setup_workload()
    result = actual_function(**workload)
    
    # Comprehensive assertions
    assert result is not None
    # Add specific checks based on expected behavior
    
def test_modified_function_edge_cases():
    """Test edge cases for modified function"""
    # Test with empty inputs, boundary conditions, etc.
    pass

def test_new_parameters():
    """Test new CLI arguments or function parameters"""
    # Test new parameters, default values, combinations
    pass

def test_bug_fix_scenario():
    """Test that specific bugs are fixed"""
    # Create the exact scenario that was buggy
    # Verify the fix works correctly
    pass

def test_performance_regression():
    """Test performance claims (if any)"""
    # Only include if commit claims performance improvements
    workload = setup_workload()
    
    # Time the optimized version
    execution_time = timeit.timeit(
        lambda: actual_function(**workload),
        number=1  # Avoid caching optimizations
    )
    
    # Add baseline comparison if available
    # Assert performance meets expectations with tolerance

def test_backward_compatibility():
    """Test that existing code patterns still work"""
    # Ensure no breaking changes
    pass

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

## Critical Requirements

### Realistic Workloads
- **Transformer models**: batch_size=32+, seq_len=1024+, hidden_dim=4096+
- **Computer vision**: 224x224+ images, realistic batch sizes
- **Scientific computing**: Large matrices, real problem sizes
- **Use diverse, random data** that prevents simple caching optimizations

### Comprehensive Testing
- Test EVERY function/method modified in the commit
- Test EVERY new parameter or CLI argument
- Test EVERY bug fix with the exact problematic scenario
- Include edge cases, error handling, and integration tests

### Performance Testing (when applicable)
- Extract specific claims from commit message ("2.8x speedup", "optimization")
- Use `timeit` with `number=1` to prevent caching between runs
- Include baseline comparison when possible
- Set realistic tolerances (20% for measurement variance)

### Quality Standards
- Use actual imports from the commit diffs
- Realistic test data at proper scales
- Clear test names describing what's being tested
- Proper assertions with helpful error messages
- Independent tests that don't rely on execution order

## Example Analysis

**Input**: Commit adds `--use-beam-search` flag and fixes beam search indexing bug

**Required Tests**:
```python
def test_beam_search_flag_parsing():
    """Test --use-beam-search CLI flag"""
    
def test_beam_search_parameter_behavior():
    """Test sampling parameters with beam search enabled"""
    
def test_beam_search_indexing_fix():
    """Test the specific indexing bug is fixed"""
    
def test_beam_search_end_to_end():
    """Test complete workflow with beam search"""
```

Generate comprehensive tests that cover ALL aspects of the commit changes with realistic, challenging workloads.
