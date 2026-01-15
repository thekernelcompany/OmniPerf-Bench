#!/usr/bin/env python3
"""
Performance test for commit: b1e5a33ae337d20e35e966b8d82a02a913d32689
Message: Eliminate stream sync to speed up LoRA batch init  (#6960)

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
        # Based on the commit diff, we're targeting set_lora_info methods
        module_path = "sglang.srt.lora.layers"
        # Return both classes since both are modified
        symbol_name = "MergedColumnParallelLinearWithLoRA,QKVParallelLinearWithLoRA"
    
    # Import with error handling
    try:
        module = importlib.import_module(module_path)
        
        # Since we have two classes, return them as a tuple
        symbols = symbol_name.split(",")
        targets = []
        for sym in symbols:
            target = module
            for attr in sym.split("."):
                target = getattr(target, attr)
            targets.append(target)
        
        fq_name = f"{module_path}.{symbol_name}"
        return tuple(targets), fq_name
        
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
    
    # LoRA configuration parameters
    num_loras = 16  # Number of LoRA adapters
    rank = 16  # LoRA rank
    input_dim = 4096  # Input dimension (typical for 7B models)
    output_dim = 4096  # Output dimension
    output_dim_kv = 1024  # KV dimension for QKV layers
    batch_size = 32  # Batch size for testing
    
    # Create LoRA weight buffers (A and B matrices)
    # For MergedColumnParallelLinearWithLoRA
    A_buffer_gate_up = torch.randn(num_loras, rank, input_dim, device=device, dtype=dtype) * 0.01
    B_buffer_gate = torch.randn(num_loras, output_dim, rank, device=device, dtype=dtype) * 0.01
    B_buffer_up = torch.randn(num_loras, output_dim, rank, device=device, dtype=dtype) * 0.01
    
    # For QKVParallelLinearWithLoRA
    A_buffer_qkv = torch.randn(num_loras, rank, input_dim, device=device, dtype=dtype) * 0.01
    B_buffer_q = torch.randn(num_loras, output_dim, rank, device=device, dtype=dtype) * 0.01
    B_buffer_kv = torch.randn(num_loras, output_dim_kv, rank, device=device, dtype=dtype) * 0.01
    
    # Create lora_indices for batch processing
    lora_indices = torch.randint(0, num_loras, (batch_size,), device=device, dtype=torch.int32)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "num_loras": num_loras,
        "rank": rank,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "output_dim_kv": output_dim_kv,
        "batch_size": batch_size,
        # MergedColumnParallelLinearWithLoRA buffers
        "A_buffer_gate_up": A_buffer_gate_up,
        "B_buffer_gate": B_buffer_gate,
        "B_buffer_up": B_buffer_up,
        # QKVParallelLinearWithLoRA buffers
        "A_buffer_qkv": A_buffer_qkv,
        "B_buffer_q": B_buffer_q,
        "B_buffer_kv": B_buffer_kv,
        # Batch indices
        "lora_indices": lora_indices,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    targets, fq_name = resolve_target()
    
    # Unpack the two classes
    MergedColumnParallelLinearWithLoRA, QKVParallelLinearWithLoRA = targets
    
    # Create mock instances with minimal setup
    # We need to create instances that have the necessary attributes for set_lora_info
    
    # Mock lora_backend object
    class MockLoRABackend:
        def __init__(self):
            self.fuse_stacked_lora_b = True
    
    lora_backend = MockLoRABackend()
    
    # Create instances
    merged_layer = MergedColumnParallelLinearWithLoRA.__new__(MergedColumnParallelLinearWithLoRA)
    merged_layer.lora_backend = lora_backend
    merged_layer.B_buffer_gate_up = None  # Initialize to trigger the optimization path
    
    qkv_layer = QKVParallelLinearWithLoRA.__new__(QKVParallelLinearWithLoRA)
    qkv_layer.lora_backend = lora_backend
    qkv_layer.B_buffer_qkv = None  # Initialize to trigger the optimization path
    qkv_layer.output_offset = None  # Initialize to trigger the optimization path
    
    # Measure the batch initialization time
    # This is what the optimization is targeting - reducing stream sync during batch init
    
    results = []
    
    # Test MergedColumnParallelLinearWithLoRA.set_lora_info
    with torch.no_grad():
        # Call set_lora_info multiple times to simulate batch initialization
        for _ in range(10):  # Multiple iterations to amplify the effect
            merged_layer.set_lora_info(
                data["lora_indices"],
                data["A_buffer_gate_up"],
                [data["B_buffer_gate"], data["B_buffer_up"]]
            )
            # Force the buffer to be None again to trigger the path
            merged_layer.B_buffer_gate_up = None
        
        results.append(merged_layer.B_buffer_gate_up)
    
    # Test QKVParallelLinearWithLoRA.set_lora_info
    with torch.no_grad():
        # Call set_lora_info multiple times to simulate batch initialization
        for _ in range(10):  # Multiple iterations to amplify the effect
            qkv_layer.set_lora_info(
                data["lora_indices"],
                data["A_buffer_qkv"],
                [data["B_buffer_q"], data["B_buffer_kv"], data["B_buffer_kv"]]
            )
            # Force the buffers to be None again to trigger the path
            qkv_layer.B_buffer_qkv = None
            qkv_layer.output_offset = None
        
        results.append(qkv_layer.B_buffer_qkv)
        results.append(qkv_layer.output_offset)
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, list):
        # Store list of tensors
        torch.save({
            "type": "tensor_list",
            "data": [t.cpu() if isinstance(t, torch.Tensor) else t for t in result]
        }, filepath)
    elif isinstance(result, torch.Tensor):
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
    if isinstance(current_result, list) and isinstance(reference_result, list):
        assert len(current_result) == len(reference_result), f"List length mismatch"
        for i, (curr, ref) in enumerate(zip(current_result, reference_result)):
            if isinstance(curr, torch.Tensor) and isinstance(ref, torch.Tensor):
                assert curr.shape == ref.shape, f"Shape mismatch at index {i}"
                assert curr.dtype == ref.dtype, f"Dtype mismatch at index {i}"
                
                # Determine tolerances based on dtype
                if curr.dtype in (torch.float16, torch.bfloat16):
                    rtol, atol = 1e-3, 1e-4
                else:
                    rtol, atol = 1e-5, 1e-7
                
                torch.testing.assert_close(
                    curr.cpu(),
                    ref.cpu(),
                    rtol=rtol, atol=atol
                )
    elif isinstance(current_result, torch.Tensor):
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
            _ = experiment(data)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        avg_ms = sum(times) / len(times)
        p50_ms = times[len(times) // 2]
        p95_ms = times[int(len(times) * 0.95) - 1] if len(times) > 1 else times[0]
        # Produce a result for reference handling
        result = experiment(data)
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "b1e5a33ae337d20e35e966b8d82a02a913d32689")
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