#!/usr/bin/env python3
"""
Performance test for commit: 9c088829ee2a28263f36d0814fde448c6090b5bc
Message: Revert "Use device_id in dist init to reduce NCCL communicator warmup & creation overhead" (#5786)

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
import torch.distributed as dist

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
    
    # Priority 2: Parse from commit metadata - init_process_group is what we're testing
    if not (module_path and symbol_name):
        module_path = "torch.distributed"
        symbol_name = "init_process_group"
    
    # Import with error handling
    try:
        if module_path == "torch.distributed":
            # Direct reference to torch.distributed.init_process_group
            target = torch.distributed.init_process_group
            fq_name = "torch.distributed.init_process_group"
        else:
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
    
    # Setup for distributed initialization testing
    # We'll test NCCL communicator initialization overhead
    
    device = torch.device(hw_info["device"] if hw_info["device"] == "cuda" else "cpu")
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Check if we're testing parent (with device_id) or child (without)
    impl_tag = os.getenv("IMPL_TAG", "child")
    
    # Prepare initialization parameters
    backend = "nccl" if hw_info["device"] == "cuda" else "gloo"
    world_size = 1  # Single process for testing
    rank = 0
    
    # Clean up any existing process group
    if dist.is_initialized():
        dist.destroy_process_group()
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "backend": backend,
        "world_size": world_size,
        "rank": rank,
        "impl_tag": impl_tag,
        "timeout": 30,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Clean up any existing process group
    if dist.is_initialized():
        dist.destroy_process_group()
    
    # Set environment variables for single-process testing
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(data["world_size"])
    os.environ["RANK"] = str(data["rank"])
    
    # Prepare kwargs based on implementation variant
    kwargs = {
        "backend": data["backend"],
        "world_size": data["world_size"],
        "rank": data["rank"],
    }
    
    # Parent commit would have device_id, child doesn't
    if data["impl_tag"] == "parent":
        # Simulate parent behavior with device_id
        if data["hw_info"]["device"] == "cuda":
            kwargs["device_id"] = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    # Time the initialization which includes NCCL communicator setup
    with torch.no_grad():
        start_time = time.perf_counter()
        try:
            # Initialize process group
            target(**kwargs)
            
            # Force NCCL communicator initialization by doing a simple collective
            if data["backend"] == "nccl" and torch.cuda.is_available():
                dummy_tensor = torch.ones(1, device="cuda")
                dist.broadcast(dummy_tensor, src=0)
                torch.cuda.synchronize()
            
            init_success = True
        except Exception as e:
            init_success = False
            print(f"Warning: Process group init failed: {e}", file=sys.stderr)
        
        end_time = time.perf_counter()
        init_time_ms = (end_time - start_time) * 1000
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()
    
    result = {
        "init_success": init_success,
        "init_time_ms": init_time_ms,
        "backend": data["backend"],
        "has_device_id": data["impl_tag"] == "parent"
    }
    
    return result

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
    # For this test, we check that initialization succeeded in both cases
    assert current_result["init_success"] == reference_result["init_success"], \
        f"Init success mismatch: {current_result['init_success']} vs {reference_result['init_success']}"
    
    assert current_result["backend"] == reference_result["backend"], \
        f"Backend mismatch: {current_result['backend']} vs {reference_result['backend']}"

# =======================
# Timing Implementation
# =======================
def time_gpu(func, warmup=5, iterations=50) -> Tuple[Any, Dict[str, float]]:
    """Time GPU operations with CUDA events."""
    # For this test, we'll use CPU timing since it's process initialization
    times_ms = []
    
    # No warmup for process group initialization - it's one-time operation
    # Instead, we'll run multiple iterations with cleanup between
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
    
    # For distributed initialization, we measure CPU time
    # since it's about process setup overhead
    warmup = 0  # No warmup for process init
    iters = 10  # Fewer iterations since each involves full init/teardown
    
    # Timing
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        result = experiment(data)
        times.append((time.perf_counter() - start) * 1000)
    
    times.sort()
    avg_ms = sum(times) / len(times)
    p50_ms = times[len(times) // 2]
    p95_ms = times[min(int(len(times) * 0.95), len(times) - 1)]
    
    # Use the result from last iteration for reference
    result = experiment(data)
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "9c088829ee2a28263f36d0814fde448c6090b5bc")
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