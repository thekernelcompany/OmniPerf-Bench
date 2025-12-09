#!/usr/bin/env python3
"""
Performance test for commit: df7f61ee7d235936e6663f07813d7c03c4ec1603
Message: Speed up rebalancing when using non-static dispatch algorithms (#6812)

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
from unittest.mock import MagicMock

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
    
    # Priority 2: Parse from commit metadata - ExpertLocationMetadata is the main target
    if not (module_path and symbol_name):
        module_path = "sglang.srt.managers.expert_location"
        symbol_name = "ExpertLocationMetadata"
    
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
    
    # Setup for expert location metadata rebalancing test
    device = torch.device(hw_info["device"])
    dtype = torch.int64  # Expert indices are integers
    
    # Simulate MoE configuration
    num_layers = 32  # Typical for large models
    num_logical_experts = 8  # Common MoE setup
    num_physical_experts = 16  # Distributed across GPUs
    ep_size = 4  # Expert parallelism size
    max_physical_per_logical = 4  # Maximum mapping size
    
    # Create mock ServerArgs with ep_dispatch_algorithm
    mock_server_args = MagicMock()
    mock_server_args.ep_dispatch_algorithm = "random"  # Test non-static to trigger optimization
    mock_server_args.device = device
    
    # Create physical to logical mapping
    physical_to_logical_map = torch.randint(
        0, num_logical_experts, 
        (num_layers, num_physical_experts), 
        device=device, dtype=dtype
    )
    
    # Create logical to all physical mapping
    logical_to_all_physical_map = torch.randint(
        0, num_physical_experts,
        (num_layers, num_logical_experts, max_physical_per_logical),
        device=device, dtype=dtype
    )
    
    # Number of valid mappings per logical expert
    logical_to_all_physical_map_num_valid = torch.randint(
        1, max_physical_per_logical + 1,
        (num_layers, num_logical_experts),
        device=device, dtype=dtype
    )
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "server_args": mock_server_args,
        "ep_size": ep_size,
        "physical_to_logical_map": physical_to_logical_map,
        "logical_to_all_physical_map": logical_to_all_physical_map,
        "logical_to_all_physical_map_num_valid": logical_to_all_physical_map_num_valid,
        "num_layers": num_layers,
        "num_logical_experts": num_logical_experts,
        "num_physical_experts": num_physical_experts,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    ExpertLocationMetadata, fq_name = resolve_target()
    
    # The optimization is in _init_raw - create metadata object
    # This will skip computing logical_to_rank_dispatch_physical_map for non-static algorithms
    with torch.no_grad():
        metadata = ExpertLocationMetadata._init_raw(
            server_args=data["server_args"],
            ep_size=data["ep_size"],
            physical_to_logical_map=data["physical_to_logical_map"],
            logical_to_all_physical_map=data["logical_to_all_physical_map"],
            logical_to_all_physical_map_num_valid=data["logical_to_all_physical_map_num_valid"],
        )
        
        # Test the update operation (rebalancing) which is optimized
        # Create another metadata object to update from
        other_metadata = ExpertLocationMetadata._init_raw(
            server_args=data["server_args"],
            ep_size=data["ep_size"],
            physical_to_logical_map=torch.roll(data["physical_to_logical_map"], 1, dims=1),
            logical_to_all_physical_map=torch.roll(data["logical_to_all_physical_map"], 1, dims=2),
            logical_to_all_physical_map_num_valid=data["logical_to_all_physical_map_num_valid"],
        )
        
        # Perform the update (rebalancing) operation
        metadata.update(other_metadata)
    
    # Return metadata state for equivalence checking
    result = {
        "physical_to_logical_map": metadata.physical_to_logical_map.cpu(),
        "logical_to_all_physical_map": metadata.logical_to_all_physical_map.cpu(),
        "logical_to_all_physical_map_num_valid": metadata.logical_to_all_physical_map_num_valid.cpu(),
        "logical_to_rank_dispatch_physical_map_is_none": metadata.logical_to_rank_dispatch_physical_map is None,
    }
    
    return result

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
    assert isinstance(current_result, dict) and isinstance(reference_result, dict)
    assert current_result.keys() == reference_result.keys()
    
    for key in current_result:
        if key == "logical_to_rank_dispatch_physical_map_is_none":
            # Boolean comparison
            assert current_result[key] == reference_result[key], f"Mismatch in {key}"
        else:
            # Tensor comparison
            current_tensor = current_result[key]
            reference_tensor = reference_result[key]
            
            assert current_tensor.shape == reference_tensor.shape, f"Shape mismatch for {key}"
            assert current_tensor.dtype == reference_tensor.dtype, f"Dtype mismatch for {key}"
            
            # Exact equivalence for integer tensors
            torch.testing.assert_close(
                current_tensor,
                reference_tensor,
                rtol=0, atol=0
            )

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
    
    # Timing - this optimization affects both CPU and GPU paths
    if hw_info["device"] == "cuda":
        warmup = 5
        iters = 50
        result, timing_stats = time_gpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    else:
        warmup = 3
        iters = 20  # Increase iterations for CPU timing
        result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "df7f61ee7d235936e6663f07813d7c03c4ec1603")
    impl_tag = os.getenv("IMPL_TAG", "child")
    ref_file = f"{prefix}_{impl_tag}_{commit_hash}_reference.pt"
    
    if reference:
        store_result(result, ref_file)
    
    if eqcheck and os.path.exists(ref_file):
        ref_result = load_result(ref_file)
        check_equivalence(result, ref_result)
    
    # Verify optimization path is hit - for non-static algorithms, the dispatch map should be None
    opt_path_hit = result["logical_to_rank_dispatch_physical_map_is_none"] == True
    
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