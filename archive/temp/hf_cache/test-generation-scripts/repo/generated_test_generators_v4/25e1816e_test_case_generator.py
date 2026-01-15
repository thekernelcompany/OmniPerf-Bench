#!/usr/bin/env python3
"""
Performance test for commit: 25e1816eff104da56f97ce494e255306603fe2f6
Message: fix custom allreduce performance/accuracy problem (#4477)

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
        # Based on commit, the Python API is sgl_kernel.allreduce
        module_path = "sgl_kernel.allreduce"
        symbol_name = "all_reduce"
    
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
    
    # Allreduce workload for custom kernel testing
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Typical sizes for LLM gradient allreduce
    # Test different sizes to cover various kernel paths
    sizes = [
        4096 * 4096,        # 16M elements - medium size
        8192 * 8192,        # 64M elements - large size
        1024 * 1024,        # 1M elements - small size
    ]
    
    # Use middle size for main testing
    num_elements = sizes[1] if hw_info.get("memory_gb", 0) >= 16 else sizes[2]
    
    # Create input tensor simulating gradient data
    input_tensor = torch.randn(num_elements, device=device, dtype=dtype)
    
    # Initialize custom allreduce if available
    try:
        from sgl_kernel.allreduce import init_custom_ar
        # Initialize with single GPU for testing (rank 0 of 1)
        init_custom_ar(
            tensor=input_tensor,
            rank=0,
            world_size=1,
            num_shm_blocks=1
        )
        custom_ar_available = True
    except (ImportError, RuntimeError) as e:
        # Custom AR may not be available in all environments
        custom_ar_available = False
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "input_tensor": input_tensor,
        "num_elements": num_elements,
        "custom_ar_available": custom_ar_available,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    
    # Get the all_reduce function
    try:
        from sgl_kernel.allreduce import all_reduce
    except ImportError:
        # Fallback to mock if module not available
        # In production, this would fail
        error_data = {
            "target_resolved": False,
            "error": "sgl_kernel.allreduce not available",
            "opt_path_hit": False
        }
        print(json.dumps(error_data))
        sys.exit(1)
    
    input_tensor = data["input_tensor"]
    
    # Clone input since allreduce modifies in-place
    work_tensor = input_tensor.clone()
    
    with torch.no_grad():
        # Execute the custom allreduce
        # For single GPU testing, this should be identity
        all_reduce(work_tensor)
    
    return work_tensor

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
    if isinstance(current_result, torch.Tensor):
        assert current_result.shape == reference_result.shape
        assert current_result.dtype == reference_result.dtype
        
        # Determine tolerances based on dtype
        if current_result.dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 1e-3, 1e-4
        else:
            rtol, atol = 1e-5, 1e-7
        
        torch.testing.assert_close(
            current_result.cpu(),
            reference_result.cpu(),
            rtol=rtol, atol=atol
        )

# =======================
# Timing Implementation
# =======================
def time_gpu(func, warmup=5, iterations=50) -> Tuple[Any, Dict[str, float]]:
    """Time GPU operations with CUDA events."""
    # Warmup
    for _ in range(warmup):
        _ = func()
        torch.cuda.synchronize()
    
    # Clear cache
    torch.cuda.empty_cache()
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
    
    # Check if we can run the test
    if hw_info["device"] != "cuda":
        error_data = {
            "error_code": 2,
            "error_name": "CAPABILITY_UNSUPPORTED",
            "error_message": "Custom allreduce requires CUDA device",
            "target_resolved": True,
            "opt_path_hit": False
        }
        print(json.dumps(error_data))
        sys.exit(2)
    
    if not data.get("custom_ar_available", False):
        # Try fallback to standard allreduce for testing
        # In production this would indicate optimization not available
        error_data = {
            "error_code": 3,
            "error_name": "OPT_PATH_NOT_TRIGGERED",
            "error_message": "Custom allreduce not initialized",
            "target_resolved": True,
            "opt_path_hit": False
        }
        print(json.dumps(error_data))
        sys.exit(3)
    
    # Timing
    warmup = 5
    iters = 50
    result, timing_stats = time_gpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "25e1816eff104da56f97ce494e255306603fe2f6")
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
        "eq_level": os.getenv("PROB_EQ_LEVEL", "numeric"),
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