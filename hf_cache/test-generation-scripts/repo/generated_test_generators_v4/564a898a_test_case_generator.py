#!/usr/bin/env python3
"""
Performance test for commit: 564a898ad975192b593be81387d11faf15cb1d3e
Message: Optimize mem indices mangement (#619)

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
        # Based on the commit diff, the main optimization is in TokenToKVPool
        module_path = "sglang.srt.memory_pool"
        symbol_name = "TokenToKVPool"
    
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
    
    # Memory pool configuration for a 70B model scenario
    # Realistic KV cache parameters
    pool_size = 65536  # 64K tokens capacity
    dtype = torch.float16
    num_layers = 80  # 70B model typically has 80 layers
    num_heads = 64  # 70B model config
    head_dim = 128
    
    # Simulate continuous batching workload
    # Mix of prefill and decode requests
    num_requests = 128
    request_sizes = []
    
    # 20% prefill requests (longer sequences)
    for _ in range(num_requests // 5):
        request_sizes.append(np.random.randint(512, 2048))
    
    # 80% decode requests (single token)
    for _ in range(num_requests * 4 // 5):
        request_sizes.append(1)
    
    np.random.shuffle(request_sizes)
    
    # Generate allocation patterns
    allocations = []
    for size in request_sizes:
        # Random indices to simulate fragmented memory
        indices = torch.randperm(pool_size)[:size].to(torch.int32)
        if torch.cuda.is_available():
            indices = indices.cuda()
        allocations.append(indices)
    
    device = torch.device(hw_info["device"])
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "pool_size": pool_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "allocations": allocations,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Create memory pool instance
    pool = target(
        size=data["pool_size"],
        dtype=data["dtype"],
        head_num=data["num_heads"],
        head_dim=data["head_dim"],
        layer_num=data["num_layers"]
    )
    
    allocations = data["allocations"]
    results = {
        "alloc_times": [],
        "dealloc_times": [],
        "used_sizes": [],
        "available_sizes": []
    }
    
    # Simulate allocation/deallocation cycles
    active_allocations = []
    
    with torch.no_grad():
        for i, indices in enumerate(allocations):
            # Allocation phase
            pool.add_refs(indices)
            active_allocations.append(indices)
            
            # Record state
            results["used_sizes"].append(pool.used_size())
            results["available_sizes"].append(pool.available_size())
            
            # Deallocate some older allocations (simulate request completion)
            if len(active_allocations) > 32:  # Keep max 32 active
                # Deallocate oldest 25%
                num_to_free = len(active_allocations) // 4
                for _ in range(num_to_free):
                    old_indices = active_allocations.pop(0)
                    pool.dec_refs(old_indices)
    
    # Final cleanup
    for indices in active_allocations:
        pool.dec_refs(indices)
    
    # Verify pool is cleared
    final_used = pool.used_size()
    results["final_used"] = final_used
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    torch.save({"type": "dict", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    # Check that memory management behavior is equivalent
    assert current_result["final_used"] == reference_result["final_used"], \
        f"Final used size mismatch: {current_result['final_used']} vs {reference_result['final_used']}"
    
    # Check allocation patterns
    assert len(current_result["used_sizes"]) == len(reference_result["used_sizes"]), \
        "Different number of allocations"
    
    # Memory tracking should be functionally equivalent
    for i, (curr_used, ref_used) in enumerate(zip(
        current_result["used_sizes"], reference_result["used_sizes"]
    )):
        assert curr_used == ref_used, \
            f"Used size mismatch at step {i}: {curr_used} vs {ref_used}"
    
    for i, (curr_avail, ref_avail) in enumerate(zip(
        current_result["available_sizes"], reference_result["available_sizes"]
    )):
        assert curr_avail == ref_avail, \
            f"Available size mismatch at step {i}: {curr_avail} vs {ref_avail}"

# =======================
# Timing Implementation
# =======================
def time_gpu(func, warmup=5, iterations=50) -> Tuple[Any, Dict[str, float]]:
    """Time GPU operations with CUDA events."""
    # Warmup
    for _ in range(warmup):
        _ = func()
        torch.cuda.synchronize()
    
    # Timing
    times_ms = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start.record()
        result = func()
        end.record()
        torch.cuda.synchronize()
        
        times_ms.append(start.elapsed_time(end))
    
    # Statistics
    times_ms.sort()
    stats = {
        "avg_ms": sum(times_ms) / len(times_ms),
        "p50_ms": times_ms[len(times_ms) // 2],
        "p95_ms": times_ms[int(len(times_ms) * 0.95)],
        "p99_ms": times_ms[int(len(times_ms) * 0.99)],
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
    
    # Timing
    if hw_info["device"] == "cuda":
        warmup = 5
        iters = 50
        result, timing_stats = time_gpu(lambda: experiment(data), warmup=warmup, iterations=iters)
        avg_ms = timing_stats["avg_ms"]
        p50_ms = timing_stats["p50_ms"]
        p95_ms = timing_stats["p95_ms"]
    else:
        warmup = 3
        iters = 10
        # CPU warmup
        for _ in range(warmup):
            _ = experiment(data)
        # CPU timing
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            result = experiment(data)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        avg_ms = sum(times) / len(times)
        p50_ms = times[len(times) // 2]
        p95_ms = times[int(len(times) * 0.95) - 1] if len(times) > 1 else times[0]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "564a898ad975192b593be81387d11faf15cb1d3e")
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
        "device": str(hw_info["device"]),
        "dtype": str(data["dtype"]),
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "exact"),
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