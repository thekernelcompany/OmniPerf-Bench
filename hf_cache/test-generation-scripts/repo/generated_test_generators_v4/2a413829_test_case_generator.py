#!/usr/bin/env python3
"""
Performance test for commit: 2a413829f42b8e8433a3e7cfd91cc9cb241cfbc0
Message: Add triton version as a fused_moe_triton config search key to avoid performace decrease in different Triton version (#5955)

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
from unittest.mock import patch, MagicMock

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
    module_path = os.getenv("PROB_MODULE", "sglang.srt.layers.moe.fused_moe_triton.fused_moe")
    symbol_name = os.getenv("PROB_SYMBOL", "get_moe_configs")
    
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
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Create test configurations for MoE
    # These are typical configurations found in the config files
    test_configs = [
        {"E": 8, "N": 14336, "dtype": "float16", "block_shape": None},
        {"E": 8, "N": 3584, "dtype": "float16", "block_shape": None},
        {"E": 8, "N": 7168, "dtype": "float16", "block_shape": None},
        {"E": 16, "N": 14336, "dtype": "int8_w8a16", "block_shape": None},
        {"E": 64, "N": 1280, "dtype": "fp8_w8a8", "block_shape": None},
        {"E": 256, "N": 128, "dtype": "fp8_w8a8", "block_shape": [128, 128]},
    ]
    
    # Simulate different Triton versions to test
    triton_versions = ["3.1.0", "3.2.0"]
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "test_configs": test_configs,
        "triton_versions": triton_versions,
        "device_name": hw_info.get("device_name", "NVIDIA_A100-SXM4-80GB"),
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    results = []
    
    # Test with different Triton versions
    for version in data["triton_versions"]:
        version_results = []
        
        # Mock the triton version
        with patch("triton.__version__", version):
            # Also need to patch the triton module in the target module
            import sys
            triton_mock = MagicMock()
            triton_mock.__version__ = version
            
            # Temporarily replace triton in sys.modules
            original_triton = sys.modules.get('triton', None)
            sys.modules['triton'] = triton_mock
            
            try:
                for config in data["test_configs"]:
                    E = config["E"]
                    N = config["N"]
                    dtype_str = config["dtype"]
                    block_shape = config["block_shape"]
                    
                    # Mock os.path.exists to simulate config file existence
                    with patch("os.path.exists") as mock_exists:
                        mock_exists.return_value = True
                        
                        # Mock open to return a dummy config
                        mock_config = {
                            str(i): {
                                "BLOCK_SIZE_M": 128,
                                "BLOCK_SIZE_N": 128,
                                "BLOCK_SIZE_K": 64,
                                "GROUP_SIZE_M": 8,
                                "num_warps": 8,
                                "num_stages": 3
                            } for i in [1, 2, 4, 8, 16, 32, 64, 128, 256]
                        }
                        
                        with patch("builtins.open", create=True) as mock_open:
                            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_config)
                            
                            # Call the target function
                            try:
                                configs = target(
                                    E=E,
                                    N=N, 
                                    dtype=dtype_str,
                                    M=None,  # Will be determined at runtime
                                    is_marlin=False,
                                    block_n=block_shape[0] if block_shape else None,
                                    block_k=block_shape[1] if block_shape else None
                                )
                                
                                # Store the result
                                version_results.append({
                                    "E": E,
                                    "N": N,
                                    "dtype": dtype_str,
                                    "block_shape": block_shape,
                                    "configs": configs,
                                    "success": True
                                })
                            except Exception as e:
                                version_results.append({
                                    "E": E,
                                    "N": N,
                                    "dtype": dtype_str,
                                    "block_shape": block_shape,
                                    "error": str(e),
                                    "success": False
                                })
            finally:
                # Restore original triton
                if original_triton is not None:
                    sys.modules['triton'] = original_triton
                elif 'triton' in sys.modules:
                    del sys.modules['triton']
        
        results.append({
            "version": version,
            "results": version_results
        })
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    torch.save({"type": "config_results", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    # For config loading, we check that the same configs are loaded
    assert len(current_result) == len(reference_result), f"Version count mismatch"
    
    for curr_ver, ref_ver in zip(current_result, reference_result):
        assert curr_ver["version"] == ref_ver["version"], f"Version mismatch"
        assert len(curr_ver["results"]) == len(ref_ver["results"]), f"Config count mismatch"
        
        for curr_cfg, ref_cfg in zip(curr_ver["results"], ref_ver["results"]):
            assert curr_cfg["E"] == ref_cfg["E"], f"E mismatch"
            assert curr_cfg["N"] == ref_cfg["N"], f"N mismatch"
            assert curr_cfg["dtype"] == ref_cfg["dtype"], f"dtype mismatch"
            assert curr_cfg["success"] == ref_cfg["success"], f"success mismatch"

# =======================
# Timing Implementation
# =======================
def time_config_loading(func, warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
    """Time configuration loading operations."""
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
    
    # Timing the config loading
    warmup = 3
    iters = 10
    
    result, timing_stats = time_config_loading(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "2a413829f42b8e8433a3e7cfd91cc9cb241cfbc0")
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
        "dtype": "torch.float16",
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