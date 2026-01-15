#!/usr/bin/env python3
"""
Performance test for commit: 6e2da5156176ed2d7fe2445b7c7316bc1650b20a
Message: Replace time.time() to time.perf_counter() for benchmarking. (#6178)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import math
import importlib
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

# =======================
# Determinism Setup
# =======================
def ensure_determinism():
    np.random.seed(42)

# =======================
# Hardware Detection
# =======================
def detect_hardware() -> Dict[str, Any]:
    hw_info = {}
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
    module_path = os.getenv("PROB_MODULE", "time")
    symbol_name = os.getenv("PROB_SYMBOL", "perf_counter")
    
    # Import with error handling
    try:
        module = importlib.import_module(module_path)
        target = getattr(module, symbol_name)
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
    
    # Simulate a realistic benchmarking scenario with many timing calls
    # This represents the typical pattern in the benchmark files
    num_iterations = 1000  # Number of benchmark iterations
    num_timing_calls = 50  # Timing calls per iteration (start/end pairs)
    
    # Create mock work to time (simulating actual benchmark operations)
    # Use simple arithmetic to simulate work between timing calls
    work_data = np.random.randn(100, 100)
    
    data = {
        "device": "cpu",
        "dtype": "float64",
        "hw_info": hw_info,
        "num_iterations": num_iterations,
        "num_timing_calls": num_timing_calls,
        "work_data": work_data,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Simulate a benchmark loop with many timing calls
    # This matches the pattern seen in the diff
    num_iterations = data["num_iterations"]
    num_timing_calls = data["num_timing_calls"]
    work_data = data["work_data"]
    
    results = []
    total_timing_overhead = 0.0
    
    for i in range(num_iterations):
        iteration_times = []
        
        for j in range(num_timing_calls):
            # Start timing
            start = target()
            
            # Simulate some work (matrix operations like in benchmarks)
            _ = np.dot(work_data, work_data.T)
            _ = np.sum(work_data) * 0.1
            
            # End timing
            end = target()
            elapsed = end - start
            
            iteration_times.append(elapsed)
            
            # Measure the overhead of the timing call itself
            overhead_start = target()
            overhead_end = target()
            total_timing_overhead += (overhead_end - overhead_start)
        
        results.append({
            "iteration": i,
            "times": iteration_times,
            "mean_time": np.mean(iteration_times),
        })
    
    # Return aggregated results
    return {
        "num_iterations": num_iterations,
        "num_timing_calls": num_timing_calls,
        "total_timing_overhead": total_timing_overhead,
        "avg_overhead_per_call": total_timing_overhead / (num_iterations * num_timing_calls * 2),
        "results_sample": results[:10],  # Sample for equivalence checking
    }

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_result = {
        "num_iterations": result["num_iterations"],
        "num_timing_calls": result["num_timing_calls"],
        "total_timing_overhead": result["total_timing_overhead"],
        "avg_overhead_per_call": result["avg_overhead_per_call"],
        "results_sample": result["results_sample"],
    }
    
    with open(filepath, 'w') as f:
        json.dump(serializable_result, f)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    with open(filepath, 'r') as f:
        return json.load(f)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    # For timing functions, we check structural equivalence
    # The actual timing values will differ, but the structure should match
    assert current_result["num_iterations"] == reference_result["num_iterations"]
    assert current_result["num_timing_calls"] == reference_result["num_timing_calls"]
    
    # Check that timing overhead is reasonable (not orders of magnitude different)
    current_overhead = current_result["avg_overhead_per_call"]
    reference_overhead = reference_result["avg_overhead_per_call"]
    
    # Allow for variation in timing overhead but ensure same order of magnitude
    if reference_overhead > 0:
        ratio = current_overhead / reference_overhead
        assert 0.1 < ratio < 10.0, f"Timing overhead changed significantly: {ratio}x"
    
    # Verify the results structure matches
    assert len(current_result["results_sample"]) == len(reference_result["results_sample"])

# =======================
# Timing Implementation
# =======================
def time_cpu(func, warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
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
        "p95_ms": times_ms[int(len(times_ms) * 0.95) - 1] if len(times_ms) > 1 else times_ms[0],
        "p99_ms": times_ms[int(len(times_ms) * 0.99) - 1] if len(times_ms) > 1 else times_ms[0],
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
    
    # CPU timing
    warmup = 3
    iters = 10
    
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "6e2da5156176ed2d7fe2445b7c7316bc1650b20a")
    impl_tag = os.getenv("IMPL_TAG", "child")
    ref_file = f"{prefix}_{impl_tag}_{commit_hash}_reference.json"
    
    if reference:
        store_result(result, ref_file)
    
    if eqcheck and os.path.exists(ref_file):
        ref_result = load_result(ref_file)
        check_equivalence(result, ref_result)
    
    # Output compact JSON schema
    summary = {
        "impl_tag": impl_tag,
        "commit_hash": commit_hash,
        "device": str(hw_info["device"]),
        "dtype": "float64",
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "behavioral"),
        "opt_path_hit": True
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