#!/usr/bin/env python3
"""
Performance test for commit: e493e48524e9e78ab33eafec6461b3940e361189
Message: [V0][Bugfix] Fix parallel sampling performance regression when guided decoding is enabled (#17731)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import copy
import importlib
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch

# =======================
# Determinism Setup
# =======================
def ensure_determinism():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Disable TF32 for reproducibility unless required
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

# =======================
# Hardware Detection
# =======================
def detect_hardware() -> Dict[str, Any]:
    hw_info = {}
    if torch.cuda.is_available():
        hw_info["device"] = "cuda"
        hw_info["device_name"] = torch.cuda.get_device_name()
        hw_info["capability"] = torch.cuda.get_device_capability()
        hw_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        hw_info["device"] = "cpu"
        hw_info["device_name"] = "CPU"
        hw_info["memory_gb"] = 0
    return hw_info

# =======================
# Import Resolution
# =======================
def resolve_target() -> Tuple[Any, str]:
    """Resolve the optimization target from environment or metadata."""
    
    # Priority 1: Environment variables
    module_path = os.getenv("PROB_MODULE", "")
    symbol_name = os.getenv("PROB_SYMBOL", "")
    
    # Priority 2: Parse from commit metadata - target is SamplingParams.clone
    if not (module_path and symbol_name):
        # Based on the APIs list and the key change in sequence.py
        module_path = "vllm.sequence"
        symbol_name = "SamplingParams"
    
    # Import with error handling
    try:
        module = importlib.import_module(module_path)
        target = module
        for attr in symbol_name.split("."):
            target = getattr(target, attr)
        
        fq_name = f"{module_path}.{symbol_name}"
        return target, fq_name
        
    except (ImportError, AttributeError) as e:
        error_data = {
            "target_resolved": False,
            "error": str(e),
            "attempted_module": module_path,
            "attempted_symbol": symbol_name
        }
        print(json.dumps(error_data))
        sys.exit(1)

# =======================
# Workload Setup
# =======================
def setup() -> Dict[str, Any]:
    """Create realistic workload for the optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    # Resolve the target class
    SamplingParams, _ = resolve_target()
    
    # Create a complex SamplingParams object with guided decoding settings
    # This mimics the scenario where parallel sampling with guided decoding is used
    params_list = []
    
    # Create multiple SamplingParams instances with various configurations
    # to simulate real-world usage in parallel sampling
    for i in range(100):  # Create 100 instances to amplify the performance difference
        # Create params with guided decoding configuration
        params = SamplingParams(
            n=4,  # Parallel samples
            temperature=0.7 + (i % 10) * 0.05,
            top_p=0.9,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.1,
            length_penalty=1.0,
            early_stopping=False,
            max_tokens=256 + i * 10,
            min_tokens=10,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            truncate_prompt_tokens=None,
            include_stop_str_in_output=False,
            seed=42 + i,
        )
        
        # Add guided decoding attributes if they exist
        # These would be present when guided decoding is enabled
        if hasattr(params, 'guided_options_request'):
            params.guided_options_request = {
                'guided_regex': r'[a-zA-Z0-9]+',
                'guided_json': None,
                'guided_choice': None,
                'guided_grammar': None,
            }
        
        params_list.append(params)
    
    data = {
        "device": hw_info["device"],
        "dtype": torch.float32,  # SamplingParams doesn't use tensors directly
        "hw_info": hw_info,
        "params_list": params_list,
        "SamplingParams": SamplingParams,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    params_list = data["params_list"]
    
    # Test the clone() method performance
    # This is what replaced copy.deepcopy in the optimization
    cloned_params = []
    
    for params in params_list:
        # Use the new clone() method if available, otherwise fall back to deepcopy
        if hasattr(params, 'clone'):
            cloned = params.clone()
        else:
            # Fallback for parent commit that doesn't have clone()
            cloned = copy.deepcopy(params)
        cloned_params.append(cloned)
    
    return cloned_params

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Store the relevant attributes of cloned params for comparison
    serializable_result = []
    for params in result:
        # Extract key attributes for comparison
        attrs = {}
        for attr in ['n', 'temperature', 'top_p', 'top_k', 'min_p', 
                    'repetition_penalty', 'length_penalty', 'max_tokens', 
                    'min_tokens', 'seed']:
            if hasattr(params, attr):
                attrs[attr] = getattr(params, attr)
        serializable_result.append(attrs)
    
    torch.save({"type": "params_list", "data": serializable_result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    # For SamplingParams, we check that the cloned objects have the same attributes
    if isinstance(current_result, list) and isinstance(reference_result, list):
        assert len(current_result) == len(reference_result), f"Length mismatch: {len(current_result)} vs {len(reference_result)}"
        
        # For the actual result objects
        for i, (current, ref) in enumerate(zip(current_result, reference_result)):
            # Check key attributes
            for attr in ['n', 'temperature', 'top_p', 'top_k', 'min_p', 
                        'repetition_penalty', 'length_penalty', 'max_tokens', 
                        'min_tokens', 'seed']:
                if hasattr(current, attr):
                    current_val = getattr(current, attr)
                    if hasattr(ref, attr):
                        ref_val = getattr(ref, attr)
                        if isinstance(current_val, float) and isinstance(ref_val, float):
                            assert abs(current_val - ref_val) < 1e-7, f"Mismatch at index {i}, attr {attr}: {current_val} vs {ref_val}"
                        else:
                            assert current_val == ref_val, f"Mismatch at index {i}, attr {attr}: {current_val} vs {ref_val}"
    else:
        # Fallback for stored results (dict format)
        assert len(current_result) == len(reference_result), f"Length mismatch"

# =======================
# Timing Implementation
# =======================
def time_cpu_operation(func, warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
    """Time CPU operations."""
    # Warmup
    for _ in range(warmup):
        _ = func()
    
    # Timing
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    
    # Statistics
    times_ms.sort()
    stats = {
        "avg_ms": sum(times_ms) / len(times_ms),
        "p50_ms": times_ms[len(times_ms) // 2],
        "p95_ms": times_ms[min(int(len(times_ms) * 0.95), len(times_ms) - 1)],
        "p99_ms": times_ms[min(int(len(times_ms) * 0.99), len(times_ms) - 1)],
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "std_ms": np.std(times_ms) if len(times_ms) > 1 else 0.0
    }
    
    return result, stats

# =======================
# Main Test Function
# =======================
def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """Main test entry point."""
    
    # Setup
    data = setup()
    hw_info = data["hw_info"]
    
    # This is a CPU-bound operation (object cloning)
    warmup = 5
    iters = 100  # More iterations for stable measurement of fast operation
    
    result, timing_stats = time_cpu_operation(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "e493e48524e9e78ab33eafec6461b3940e361189")
    impl_tag = os.getenv("IMPL_TAG", "child")
    ref_file = f"{prefix}_{impl_tag}_{commit_hash}_reference.pt"
    
    if reference:
        store_result(result, ref_file)
    
    if eqcheck and os.path.exists(ref_file):
        ref_result = load_result(ref_file)
        # For stored results, we only compare the extracted attributes
        current_attrs = []
        for params in result:
            attrs = {}
            for attr in ['n', 'temperature', 'top_p', 'top_k', 'min_p', 
                        'repetition_penalty', 'length_penalty', 'max_tokens', 
                        'min_tokens', 'seed']:
                if hasattr(params, attr):
                    attrs[attr] = getattr(params, attr)
            current_attrs.append(attrs)
        
        assert len(current_attrs) == len(ref_result), "Result count mismatch"
        for i, (current, ref) in enumerate(zip(current_attrs, ref_result)):
            for key in ref:
                assert key in current, f"Missing key {key} at index {i}"
                if isinstance(ref[key], float):
                    assert abs(current[key] - ref[key]) < 1e-7, f"Value mismatch for {key} at index {i}"
                else:
                    assert current[key] == ref[key], f"Value mismatch for {key} at index {i}"
    
    # Check if optimization path was hit
    # For child commit, clone() should be available
    # For parent commit, it falls back to deepcopy
    SamplingParams = data["SamplingParams"]
    opt_path_hit = hasattr(SamplingParams, 'clone') if impl_tag == "child" else True
    
    # Output compact JSON schema
    summary = {
        "impl_tag": impl_tag,
        "commit_hash": commit_hash,
        "device": "cpu",  # This is a CPU operation
        "dtype": "object",  # Working with Python objects
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "exact"),
        "opt_path_hit": opt_path_hit
    }
    print(json.dumps(summary))
    
    return avg_ms / 1000.0

# =======================
# Entry Point
# =======================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eqcheck", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--prefix", type=str, default="")
    args = parser.parse_args()
    
    run_test(args.eqcheck, args.reference, args.prefix)