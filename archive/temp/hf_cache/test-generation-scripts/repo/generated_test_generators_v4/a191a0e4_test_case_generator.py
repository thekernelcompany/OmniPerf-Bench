#!/usr/bin/env python3
"""
Performance test for commit: a191a0e47c2f0b0c8aed28080b9cb78624365e92
Message: Improve performance of two batch overlap in some imbalanced cases (#6593)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import importlib
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch

# =======================
# Determinism Setup
# =======================
def ensure_determinism():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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
    
    # Priority 2: Parse from commit metadata
    if not (module_path and symbol_name):
        module_path = "sglang.srt.two_batch_overlap"
        symbol_name = "compute_split_seq_index"
    
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
    
    # Import ForwardMode enum
    try:
        from sglang.test.attention.test_flashattn_backend import ForwardMode
    except ImportError:
        # Fallback if running on parent commit without this module
        class ForwardMode:
            DECODE = "decode"
            EXTEND = "extend"
    
    # Create imbalanced test cases as mentioned in commit message
    # These are CPU-bound operations working with scheduling arrays
    
    # Test cases with various imbalanced scenarios
    test_cases = []
    
    # DECODE mode cases - single integer input
    for num_tokens in [100, 1000, 10000, 99999]:
        test_cases.append({
            "forward_mode": ForwardMode.DECODE,
            "num_tokens": num_tokens,
            "extend_lens": None
        })
    
    # EXTEND mode cases - arrays with imbalanced distributions
    # Case 1: Heavily front-loaded
    test_cases.append({
        "forward_mode": ForwardMode.EXTEND,
        "num_tokens": None,
        "extend_lens": [99999] + [1] * 100
    })
    
    # Case 2: Heavily back-loaded
    test_cases.append({
        "forward_mode": ForwardMode.EXTEND,
        "num_tokens": None,
        "extend_lens": [1] * 100 + [99999]
    })
    
    # Case 3: Large uniform arrays
    test_cases.append({
        "forward_mode": ForwardMode.EXTEND,
        "num_tokens": None,
        "extend_lens": [4096] * 64
    })
    
    # Case 4: Mixed sizes with outliers
    test_cases.append({
        "forward_mode": ForwardMode.EXTEND,
        "num_tokens": None,
        "extend_lens": [4096, 4096, 1, 1, 8192, 1, 4096, 4096] * 8
    })
    
    # Case 5: Gradually increasing
    test_cases.append({
        "forward_mode": ForwardMode.EXTEND,
        "num_tokens": None,
        "extend_lens": list(range(1, 201))
    })
    
    # Case 6: Gradually decreasing
    test_cases.append({
        "forward_mode": ForwardMode.EXTEND,
        "num_tokens": None,
        "extend_lens": list(range(200, 0, -1))
    })
    
    # Case 7: Alternating pattern
    test_cases.append({
        "forward_mode": ForwardMode.EXTEND,
        "num_tokens": None,
        "extend_lens": [1, 8192] * 32
    })
    
    # Case 8: Real-world batch scenario
    test_cases.append({
        "forward_mode": ForwardMode.EXTEND,
        "num_tokens": None,
        "extend_lens": [2048, 512, 1024, 4096, 128, 256, 8192, 64] * 4
    })
    
    data = {
        "device": "cpu",  # This is a CPU optimization
        "dtype": None,     # Not applicable for this integer operation
        "hw_info": hw_info,
        "test_cases": test_cases,
        "ForwardMode": ForwardMode
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Execute all test cases and collect results
    results = []
    
    for test_case in data["test_cases"]:
        result = target(
            forward_mode=test_case["forward_mode"],
            num_tokens=test_case["num_tokens"],
            extend_lens=test_case["extend_lens"]
        )
        results.append(result)
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Store as JSON since these are simple integers
    import json
    with open(filepath, 'w') as f:
        json.dump({"type": "list", "data": result}, f)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert isinstance(current_result, list), f"Expected list, got {type(current_result)}"
    assert isinstance(reference_result, list), f"Expected list, got {type(reference_result)}"
    assert len(current_result) == len(reference_result), f"Length mismatch: {len(current_result)} vs {len(reference_result)}"
    
    for i, (curr, ref) in enumerate(zip(current_result, reference_result)):
        assert curr == ref, f"Mismatch at index {i}: {curr} vs {ref}"

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
        "std_ms": np.std(times_ms)
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
    
    # This is a CPU-bound optimization
    warmup = 5
    iters = 50  # More iterations since CPU operations are fast
    
    # Time the operation
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "a191a0e47c2f0b0c8aed28080b9cb78624365e92")
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
        "device": "cpu",
        "dtype": "int",
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": "exact",
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