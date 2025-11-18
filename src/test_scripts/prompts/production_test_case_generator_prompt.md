# Production-Quality Test Case Generator Prompt

You are an expert performance testing engineer tasked with generating production-grade test suites from commit diffs. Your tests must be comprehensive, realistic, and suitable for detecting real-world regressions and performance issues.

## Critical Mission

Generate a complete Python test suite that follows production testing standards:
- **Realistic workloads** with real-world data scales and complexity
- **Comprehensive equivalence checking** with reference result storage/loading
- **Performance regression detection** with proper timing methodology
- **Non-hackable tests** that can't be gamed with simple optimizations
- **Domain-appropriate scenarios** based on actual usage patterns

## Input Analysis Requirements

### 1. Commit Change Analysis
For each file in `files_changed`:
- **Parse diffs meticulously** - identify every function/method/parameter change
- **Extract performance claims** from commit messages ("2.8x speedup", "optimization", etc.)
- **Identify optimization targets** (algorithms, kernels, memory usage, etc.)
- **Map changes to real-world usage** scenarios in the domain

### 2. Domain Context Understanding
- **Transformer models**: Use realistic batch sizes (32-128), sequence lengths (512-4096), hidden dimensions (1024-8192)
- **Computer vision**: Use actual image sizes (224x224, 512x512), realistic batch sizes, real datasets
- **Graph processing**: Use real-world graph structures (social networks, road networks, etc.)
- **Scientific computing**: Use realistic matrix dimensions, actual data distributions

### 3. Performance Claims Extraction
Parse commit messages for specific claims:
- "avg 2.8x speedup" → expect 2.8x improvement with 20% tolerance
- "optimize beam search" → measure beam search latency specifically
- "reduce memory usage" → track memory consumption
- "block size heuristic" → test with various block sizes

## Required Output Structure

Generate a complete test file following this production template:

```python
#!/usr/bin/env python3
"""
Production test suite for commit: {commit_hash}
{commit_message}

Test Coverage:
- Realistic workload setup with domain-appropriate data
- Comprehensive functional correctness testing
- Performance regression detection with timing
- Equivalence checking with reference results
- Edge cases and error handling
- Integration testing across components

Run with: pytest -v test_{commit_short_hash}.py
"""

import pytest
import torch
import numpy as np
import timeit
import json
import pickle
import os
import tempfile
from typing import Dict, Any, List, Tuple, Union
from unittest.mock import Mock, patch, MagicMock
import requests
import gzip

# Import actual modules from the commit
try:
    from actual_module import actual_function  # Use real imports from diffs
    from actual_module.submodule import ModifiedClass
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)

# Configure timeit to return both time and result
timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""

class Test{CommitShortHash}Production:
    """Production-quality test suite for commit {commit_short_hash}"""
    
    def setup_method(self):
        """Setup realistic test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.reference_dir = os.path.join(self.temp_dir, "references")
        os.makedirs(self.reference_dir, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def setup_realistic_workload(self) -> Dict[str, Any]:
        """
        Create realistic, challenging workload based on the domain.
        
        Requirements:
        - Use real-world data scales (not toy examples)
        - Download real datasets when appropriate
        - Generate diverse, non-uniform data that can't be easily cached
        - Use appropriate data types and devices for the domain
        - Set random seeds for reproducibility
        
        Returns:
            Dict containing all data needed for experiments
        """
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Example for transformer/LLM domain:
        if "transformer" in "{commit_message}".lower() or "llm" in "{commit_message}".lower():
            # Realistic transformer dimensions
            batch_size = 32
            seq_len = 1024
            hidden_dim = 4096
            num_heads = 32
            head_dim = hidden_dim // num_heads
            vocab_size = 50000
            num_layers = 24
            
            # Create diverse, non-uniform data
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            # Add some padding to make it realistic
            for i in range(batch_size):
                pad_len = np.random.randint(0, seq_len // 4)
                attention_mask[i, -pad_len:] = 0
            
            # KV cache with realistic shapes
            cache_shape = (batch_size, num_heads, seq_len, head_dim)
            key_cache = torch.randn(cache_shape, dtype=torch.float16)
            value_cache = torch.randn(cache_shape, dtype=torch.float16)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'key_cache': key_cache,
                'value_cache': value_cache,
                'batch_size': batch_size,
                'seq_len': seq_len,
                'hidden_dim': hidden_dim,
                'num_heads': num_heads,
                'vocab_size': vocab_size,
                'num_layers': num_layers,
            }
        
        # Add other domain-specific setups here...
        return {}
    
    def experiment_optimized_functionality(self, workload: Dict[str, Any]) -> Any:
        """
        Run the actual optimized functionality being tested.
        
        This should represent a comprehensive real-world usage scenario
        that exercises the optimized code paths from the commit.
        
        Args:
            workload: Data from setup_realistic_workload()
            
        Returns:
            Results from the experiment (to be used for equivalence checking)
        """
        # Example: Test the actual optimized function/method from the commit
        # Use the real APIs and realistic parameters
        
        # This would be customized based on the actual commit changes
        result = actual_function(
            input_data=workload['input_ids'],
            cache_data=workload['key_cache'],
            # Use parameters that exercise the optimization
            **workload
        )
        
        return result
    
    def experiment_baseline_functionality(self, workload: Dict[str, Any]) -> Any:
        """
        Run baseline/reference implementation for comparison.
        
        This could be:
        - The old version of the function (if available)
        - A reference implementation
        - The same function with optimization disabled
        
        Returns:
            Baseline results for comparison
        """
        # This would implement or call the baseline version
        # Often requires accessing old code or disabling optimizations
        pass
    
    def store_result(self, result: Any, filepath: str) -> None:
        """
        Store experiment results using appropriate serialization.
        
        Choose serialization method based on data type:
        - torch.save/torch.load for tensors
        - json for simple data structures  
        - pickle for complex objects
        - Custom serialization for domain-specific types
        """
        if isinstance(result, torch.Tensor):
            torch.save(result, filepath)
        elif isinstance(result, (dict, list, str, int, float)):
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            # Break down complex objects into essential components
            if hasattr(result, '__dict__'):
                essential_data = {
                    key: value for key, value in result.__dict__.items()
                    if not key.startswith('_') and not callable(value)
                }
                with open(filepath, 'w') as f:
                    json.dump(essential_data, f, indent=2)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(result, f)
    
    def load_result(self, filepath: str) -> Any:
        """Load stored reference results"""
        if filepath.endswith('.pt') or filepath.endswith('.pth'):
            return torch.load(filepath)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    
    def check_equivalence(self, current_result: Any, reference_result: Any) -> None:
        """
        Comprehensive equivalence checking between current and reference results.
        
        Requirements:
        - Compare ALL important properties/attributes
        - Use appropriate tolerances for floating-point comparisons
        - Handle type mismatches from serialization (e.g., tuples → lists)
        - Only compare current vs reference (reference is ground truth)
        """
        if isinstance(reference_result, torch.Tensor):
            assert isinstance(current_result, torch.Tensor), f"Type mismatch: expected tensor, got {type(current_result)}"
            assert current_result.shape == reference_result.shape, f"Shape mismatch: {current_result.shape} vs {reference_result.shape}"
            assert current_result.dtype == reference_result.dtype, f"Dtype mismatch: {current_result.dtype} vs {reference_result.dtype}"
            assert torch.allclose(current_result, reference_result, rtol=1e-3, atol=1e-6), "Tensor values don't match reference"
        
        elif isinstance(reference_result, dict):
            assert isinstance(current_result, dict), f"Type mismatch: expected dict, got {type(current_result)}"
            assert set(current_result.keys()) == set(reference_result.keys()), "Dictionary keys don't match"
            for key in reference_result.keys():
                self.check_equivalence(current_result[key], reference_result[key])
        
        elif isinstance(reference_result, (list, tuple)):
            # Handle tuple/list conversion from JSON serialization
            current_list = list(current_result) if isinstance(current_result, tuple) else current_result
            reference_list = list(reference_result) if isinstance(reference_result, tuple) else reference_result
            assert len(current_list) == len(reference_list), f"Length mismatch: {len(current_list)} vs {len(reference_list)}"
            for i, (curr, ref) in enumerate(zip(current_list, reference_list)):
                try:
                    self.check_equivalence(curr, ref)
                except AssertionError as e:
                    raise AssertionError(f"Mismatch at index {i}: {e}")
        
        elif isinstance(reference_result, (int, float, str, bool)):
            if isinstance(reference_result, float):
                assert abs(current_result - reference_result) < 1e-6, f"Float mismatch: {current_result} vs {reference_result}"
            else:
                assert current_result == reference_result, f"Value mismatch: {current_result} vs {reference_result}"
        
        else:
            # For complex objects, compare essential attributes
            assert type(current_result) == type(reference_result), f"Type mismatch: {type(current_result)} vs {type(reference_result)}"
            # Add domain-specific equivalence checks here
    
    def run_performance_test(self, eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
        """
        Run comprehensive performance test with timing and equivalence checking.
        
        Args:
            eqcheck: If True, check equivalence against reference
            reference: If True, store results as reference
            prefix: Prefix for reference files
            
        Returns:
            Execution time in seconds
        """
        # Setup realistic workload
        workload = self.setup_realistic_workload()
        
        # Run timed experiment
        execution_time, result = timeit.timeit(
            lambda: self.experiment_optimized_functionality(workload),
            number=1  # Only run once to avoid caching optimizations
        )
        
        # Handle reference storage/checking
        if reference:
            reference_file = os.path.join(self.reference_dir, f"{prefix}_result.json")
            self.store_result(result, reference_file)
        
        if eqcheck:
            reference_file = os.path.join(self.reference_dir, f"{prefix}_result.json")
            reference_result = self.load_result(reference_file)
            self.check_equivalence(result, reference_result)
        
        return execution_time
    
    def test_functional_correctness(self):
        """Test that all modified functionality works correctly"""
        workload = self.setup_realistic_workload()
        result = self.experiment_optimized_functionality(workload)
        
        # Add specific assertions based on the commit changes
        assert result is not None, "Function should return a result"
        # Add more specific checks based on expected behavior
    
    def test_performance_regression(self):
        """Test that performance improvements are maintained"""
        # Only include if commit claims performance improvements
        performance_claim = "2.8x"  # Extract from commit message
        if performance_claim:
            workload = self.setup_realistic_workload()
            
            # Time baseline (if available)
            baseline_time, baseline_result = timeit.timeit(
                lambda: self.experiment_baseline_functionality(workload),
                number=1
            )
            
            # Time optimized version
            optimized_time, optimized_result = timeit.timeit(
                lambda: self.experiment_optimized_functionality(workload),
                number=1
            )
            
            # Check equivalence first
            self.check_equivalence(optimized_result, baseline_result)
            
            # Check performance improvement
            speedup = baseline_time / optimized_time
            expected_speedup = 2.8  # Extract from commit message
            tolerance = 0.8  # 20% tolerance
            
            assert speedup >= expected_speedup * tolerance, (
                f"Performance regression detected. Expected {expected_speedup}x speedup, "
                f"got {speedup:.2f}x. Baseline: {baseline_time:.4f}s, Optimized: {optimized_time:.4f}s"
            )
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with empty inputs
        # Test with single elements
        # Test with maximum sizes
        # Test with unusual but valid inputs
        pass
    
    def test_error_handling(self):
        """Test proper error handling"""
        # Test invalid inputs
        # Test out-of-bounds conditions
        # Test type mismatches
        pass

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

## Critical Quality Requirements

### 1. Realistic Workloads (NOT Toy Data)
- **Transformer models**: batch_size=32, seq_len=1024, hidden_dim=4096
- **Computer vision**: 224x224+ images, realistic batch sizes
- **Scientific computing**: Large matrices, realistic problem sizes
- **Use real datasets** when possible (download via requests)

### 2. Non-Hackable Tests
- **Diverse, random inputs** that can't be cached between runs
- **Avoid patterns** (sorted data, repeated values, uniform distributions)
- **Use number=1** in timeit to prevent caching optimizations
- **Complex workloads** that require actual computation

### 3. Comprehensive Equivalence Checking
- **Compare ALL important properties** (shapes, dtypes, values)
- **Handle serialization artifacts** (tuples→lists from JSON)
- **Use appropriate tolerances** for floating-point comparisons
- **Only compare current vs reference** (never hardcoded expected values)

### 4. Production-Grade Performance Testing
- **Extract specific performance claims** from commit messages
- **Use proper timing methodology** with timeit
- **Include baseline comparisons** when possible
- **Set realistic tolerances** (20% for measurement variance)
- **Clear failure messages** with actual timing details

### 5. Domain Expertise
- **Understand the domain** (transformers, computer vision, etc.)
- **Use appropriate data types** (float16, int8, etc.)
- **Test realistic usage patterns** from the domain
- **Include domain-specific edge cases**

Generate production-quality tests that would be suitable for a major open-source project's CI/CD pipeline. The tests should catch real regressions and performance issues that matter to actual users.
