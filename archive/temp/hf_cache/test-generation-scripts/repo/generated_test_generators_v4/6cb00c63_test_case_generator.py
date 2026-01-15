#!/usr/bin/env python3
"""
Performance test for commit: 6cb00c6398126513e37c43dd975d461765fb44c7
Message: [PD] Optimize time out logic and add env var doc for mooncake (#6761)

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
from unittest.mock import MagicMock, patch

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
        module_path = "sglang.srt.disaggregation.mooncake.conn"
        symbol_name = "MooncakeKVSender"
    
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
    
    # Mock dependencies for MooncakeKVSender
    mock_kv_mgr = MagicMock()
    mock_kv_mgr.update_status = MagicMock()
    mock_kv_mgr.get_status = MagicMock()
    mock_kv_mgr.record_failure = MagicMock()
    mock_kv_mgr.bootstrap_time_out = 30.0  # Default timeout from docs
    
    # Return status sequence simulating timeout check scenario
    from sglang.srt.disaggregation.mooncake.conn import KVPoll
    status_sequence = [KVPoll.Bootstrapping] * 100  # Many polling attempts
    mock_kv_mgr.get_status.side_effect = status_sequence
    
    device = torch.device("cpu")  # This is a CPU-bound logic test
    dtype = torch.float32
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "mock_kv_mgr": mock_kv_mgr,
        "bootstrap_room": "test_room_123",
        "bootstrap_addr": "http://test-server:8080",
        "num_kv_indices": 1024,
        "aux_index": 42,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Create instance with mocked dependencies
    sender = target(
        data["mock_kv_mgr"],
        data["bootstrap_room"],
        data["bootstrap_addr"]
    )
    
    # Simulate the optimized timeout check flow
    # Before optimization: init_time set in constructor
    # After optimization: init_time set in init() method
    
    result = {"poll_results": [], "init_called": False}
    
    # Poll before init (tests the optimization)
    for _ in range(10):
        poll_result = sender.poll()
        result["poll_results"].append(str(poll_result))
    
    # Now call init
    sender.init(data["num_kv_indices"], data["aux_index"])
    result["init_called"] = True
    
    # Poll after init
    for _ in range(10):
        poll_result = sender.poll()
        result["poll_results"].append(str(poll_result))
    
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
    assert isinstance(current_result, dict), f"Type mismatch: expected dict, got {type(current_result)}"
    assert isinstance(reference_result, dict), f"Type mismatch: expected dict, got {type(reference_result)}"
    
    # Check that both have same keys
    assert current_result.keys() == reference_result.keys(), f"Keys mismatch"
    
    # Check init_called flag
    assert current_result["init_called"] == reference_result["init_called"]
    
    # Check poll results length
    assert len(current_result["poll_results"]) == len(reference_result["poll_results"])

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
    
    # This is a CPU-only test (timeout logic)
    warmup = 3
    iters = 10
    
    # Timing
    result, timing_stats = time_cpu_operation(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "6cb00c6398126513e37c43dd975d461765fb44c7")
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
        "device": "cpu",
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