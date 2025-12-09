#!/usr/bin/env python3
"""
Performance test for commit: 10189d08dde1096f5759316c0a6ff05962714c4b
Message: [Performance]: Process affinity to CPU cores with multiple sockets support (#2171)

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
import torch

try:
    import psutil
except ImportError:
    print(json.dumps({"error": "psutil not installed", "target_resolved": False}))
    sys.exit(1)

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
    
    # Priority 2: Parse from commit metadata
    if not (module_path and symbol_name):
        # Based on the commit, the new function is gpu_proc_affinity
        module_path = "sglang.srt.utils"
        symbol_name = "gpu_proc_affinity"
    
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
    
    # For testing CPU affinity, we'll create a memory-intensive workload
    # that benefits from proper NUMA affinity
    device = torch.device("cpu")  # This is a CPU affinity optimization
    dtype = torch.float32
    
    # Get system info
    total_cores = psutil.cpu_count(logical=False)
    total_threads = psutil.cpu_count(logical=True)
    
    # Simulate different TP sizes and node configurations
    # We'll test with a common configuration
    tp_size = min(4, total_cores)  # Common TP size
    nnodes = 1  # Single node for testing
    gpu_id = 0  # Test with first GPU ID
    
    # Create memory-intensive workload that benefits from affinity
    # Large matrices for memory bandwidth testing
    size = 4096
    num_matrices = 10
    
    matrices = []
    for _ in range(num_matrices):
        m1 = torch.randn(size, size, dtype=dtype, device=device)
        m2 = torch.randn(size, size, dtype=dtype, device=device)
        matrices.append((m1, m2))
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "tp_size": tp_size,
        "nnodes": nnodes,
        "gpu_id": gpu_id,
        "matrices": matrices,
        "total_cores": total_cores,
        "total_threads": total_threads,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Store original affinity
    pid = os.getpid()
    p = psutil.Process(pid)
    original_affinity = p.cpu_affinity()
    
    # Apply the optimization - set CPU affinity
    target(data["tp_size"], data["nnodes"], data["gpu_id"])
    
    # Get new affinity
    new_affinity = p.cpu_affinity()
    
    # Perform memory-intensive workload to measure impact
    # This workload benefits from proper NUMA affinity
    results = []
    with torch.no_grad():
        for m1, m2 in data["matrices"]:
            # Matrix multiplication is memory bandwidth limited for large matrices
            result = torch.mm(m1, m2)
            results.append(result.mean().item())
    
    # Return affinity info and workload results
    return {
        "original_affinity": original_affinity,
        "new_affinity": new_affinity,
        "workload_results": results,
        "total_cores": data["total_cores"],
        "total_threads": data["total_threads"],
        "tp_size": data["tp_size"],
        "nnodes": data["nnodes"],
        "gpu_id": data["gpu_id"],
    }

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, torch.Tensor):
        torch.save({"type": "tensor", "data": result.cpu()}, filepath)
    else:
        torch.save({"type": "generic", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    # For CPU affinity, we check that the affinity calculation is consistent
    assert isinstance(current_result, dict), "Result should be a dictionary"
    assert isinstance(reference_result, dict), "Reference should be a dictionary"
    
    # Check that affinity was set (not necessarily the same across runs)
    assert len(current_result["new_affinity"]) > 0, "Affinity should be set"
    
    # Check workload results are similar (these should be deterministic)
    current_workload = current_result["workload_results"]
    reference_workload = reference_result["workload_results"]
    
    assert len(current_workload) == len(reference_workload), "Workload result count mismatch"
    
    for i, (curr, ref) in enumerate(zip(current_workload, reference_workload)):
        # Use reasonable tolerance for floating point
        assert abs(curr - ref) < 1e-3, f"Workload result {i} mismatch: {curr} vs {ref}"

# =======================
# Timing Implementation
# =======================
def time_cpu_workload(func, warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
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
        "p95_ms": times_ms[int(len(times_ms) * 0.95)] if len(times_ms) > 1 else times_ms[0],
        "p99_ms": times_ms[int(len(times_ms) * 0.99)] if len(times_ms) > 1 else times_ms[0],
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
    
    # For CPU affinity, we measure the performance of memory-intensive workload
    warmup = 3
    iters = 10
    
    # Time the workload with affinity setting
    result, timing_stats = time_cpu_workload(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "10189d08dde1096f5759316c0a6ff05962714c4b")
    impl_tag = os.getenv("IMPL_TAG", "child")
    ref_file = f"{prefix}_{impl_tag}_{commit_hash}_reference.pt"
    
    if reference:
        store_result(result, ref_file)
    
    if eqcheck and os.path.exists(ref_file):
        ref_result = load_result(ref_file)
        check_equivalence(result, ref_result)
    
    # Output compact JSON schema
    summary = {
        "impl_tag": impl_tag,
        "commit_hash": commit_hash,
        "device": "cpu",  # This is a CPU affinity optimization
        "dtype": str(data["dtype"]),
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "behavioral"),  # Affinity is behavioral
        "opt_path_hit": True  # We successfully set affinity
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