#!/usr/bin/env python3
"""
Performance test for commit: 9183c23eca51bf76159e81dfd6edf5770796c2d8
Message: Speed up `update_weights_from_tensor` (#2695)

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
import io
import pickle

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
    
    # Priority 2: Parse from commit metadata - use MultiprocessingSerializer
    if not (module_path and symbol_name):
        # Based on diff, the new serialization utility is the key optimization
        module_path = "sglang.srt.utils"
        symbol_name = "MultiprocessingSerializer"
    
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
    
    # Based on test file - simulate weight update workload
    # Multiple large MLP weight tensors like in actual LLMs
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Realistic LLM weight dimensions (e.g., Llama MLP layers)
    # up_proj: hidden_size -> intermediate_size
    # Based on test: 16384 x 2048 tensors
    num_layers = 10  # Update 10 layers like in test
    hidden_size = 2048
    intermediate_size = 16384
    
    # Create named tensor list as used by the new API
    named_tensors = []
    for i in range(6, 6 + num_layers):  # Layers 6-15 like in test
        param_name = f"model.layers.{i}.mlp.up_proj.weight"
        # Create realistic weight tensor with small random values
        tensor = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype) * 0.02
        named_tensors.append((param_name, tensor))
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "named_tensors": named_tensors,
        "num_params": len(named_tensors),
        "total_elements": sum(t[1].numel() for t in named_tensors),
        "total_bytes": sum(t[1].numel() * t[1].element_size() for t in named_tensors)
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # The optimization is in the serialization/deserialization process
    # Simulate the actual operation flow from the commit
    named_tensors = data["named_tensors"]
    
    # Test the serialization path (the actual optimization)
    serialized = target.serialize(named_tensors)
    
    # And deserialization to complete the round-trip
    deserialized = target.deserialize(serialized)
    
    # Return both serialized size and deserialized result
    return {
        "serialized_bytes": len(serialized),
        "deserialized": deserialized,
        "num_tensors": len(deserialized)
    }

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Store the deserialized tensors for equivalence checking
    if isinstance(result, dict) and "deserialized" in result:
        # Convert tensors to CPU for storage
        stored_data = {
            "serialized_bytes": result["serialized_bytes"],
            "num_tensors": result["num_tensors"],
            "tensors": [(name, tensor.cpu()) for name, tensor in result["deserialized"]]
        }
        torch.save({"type": "serialization_result", "data": stored_data}, filepath)
    else:
        torch.save({"type": "generic", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    if data.get("type") == "serialization_result":
        # Reconstruct the result structure
        stored = data["data"]
        return {
            "serialized_bytes": stored["serialized_bytes"],
            "num_tensors": stored["num_tensors"],
            "deserialized": stored["tensors"]
        }
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert isinstance(current_result, dict) and isinstance(reference_result, dict)
    
    # Check metadata
    assert current_result["num_tensors"] == reference_result["num_tensors"], \
        f"Tensor count mismatch: {current_result['num_tensors']} vs {reference_result['num_tensors']}"
    
    # Allow some variance in serialized size due to pickling differences
    size_ratio = current_result["serialized_bytes"] / reference_result["serialized_bytes"]
    assert 0.95 <= size_ratio <= 1.05, \
        f"Serialized size variance too large: {current_result['serialized_bytes']} vs {reference_result['serialized_bytes']}"
    
    # Check deserialized tensors
    current_tensors = current_result["deserialized"]
    reference_tensors = reference_result["deserialized"]
    
    assert len(current_tensors) == len(reference_tensors), "Tensor list length mismatch"
    
    for i, ((curr_name, curr_tensor), (ref_name, ref_tensor)) in enumerate(zip(current_tensors, reference_tensors)):
        assert curr_name == ref_name, f"Name mismatch at {i}: {curr_name} vs {ref_name}"
        
        # Move to CPU for comparison if needed
        if isinstance(curr_tensor, torch.Tensor):
            curr_tensor = curr_tensor.cpu()
        if isinstance(ref_tensor, torch.Tensor):
            ref_tensor = ref_tensor.cpu()
        
        assert curr_tensor.shape == ref_tensor.shape, f"Shape mismatch at {i}"
        assert curr_tensor.dtype == ref_tensor.dtype, f"Dtype mismatch at {i}"
        
        # Determine tolerances based on dtype
        if curr_tensor.dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 1e-3, 1e-4
        else:
            rtol, atol = 1e-5, 1e-7
        
        torch.testing.assert_close(
            curr_tensor,
            ref_tensor,
            rtol=rtol, atol=atol
        )

# =======================
# Timing Implementation
# =======================
def time_cpu_operation(func, warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
    """Time CPU operations with perf_counter."""
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
    
    # This is a CPU-bound serialization optimization
    # Time on CPU regardless of GPU availability
    warmup = 5
    iters = 20  # Moderate iterations for serialization
    
    result, timing_stats = time_cpu_operation(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "9183c23eca51bf76159e81dfd6edf5770796c2d8")
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
        "device": "cpu",  # Serialization is CPU-bound
        "dtype": str(data["dtype"]),
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "numeric"),
        "opt_path_hit": True,
        "workload_info": {
            "num_tensors": data["num_params"],
            "total_elements": data["total_elements"],
            "total_bytes": data["total_bytes"],
            "serialized_bytes": result["serialized_bytes"]
        }
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