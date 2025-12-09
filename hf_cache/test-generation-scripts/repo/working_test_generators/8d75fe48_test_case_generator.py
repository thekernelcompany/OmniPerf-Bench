#!/usr/bin/env python3
"""
Performance test for commit: 8d75fe48ca5f46b7af0f5201d8500b9604eed769
Message: [Kernel] Switch fp8 layers to use the CUTLASS kernels (#5183)

This script measures the actual performance impact of switching from
torch._scaled_mm to vLLM's CUTLASS FP8 kernels.
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
        major, minor = hw_info["capability"]
        hw_info["supports_fp8"] = major >= 9  # Hopper+
    else:
        hw_info["device"] = "cpu"
        hw_info["device_name"] = "CPU"
        hw_info["memory_gb"] = 0
        hw_info["supports_fp8"] = False
    return hw_info

# =======================
# Import Resolution
# =======================
def resolve_target() -> Tuple[Any, str]:
    """Resolve the optimization target from environment or metadata."""
    
    # Priority 1: Environment variables
    module_path = os.getenv("PROB_MODULE", "")
    symbol_name = os.getenv("PROB_SYMBOL", "")
    
    # Priority 2: Parse from commit metadata - target is cutlass_scaled_mm_dq
    if not (module_path and symbol_name):
        module_path = "vllm._custom_ops"
        symbol_name = "cutlass_scaled_mm_dq"
    
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
    """Create realistic workload for FP8 GEMM optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    device = torch.device(hw_info["device"])
    
    # FP8 requires CUDA
    if hw_info["device"] != "cuda":
        error_data = {
            "target_resolved": True,
            "error": "FP8 operations require CUDA device",
            "opt_path_hit": False
        }
        print(json.dumps(error_data))
        sys.exit(2)
    
    # Typical LLM linear layer dimensions (7B model)
    batch_size = 8
    seq_len = 2048
    hidden_size = 4096
    intermediate_size = 11008
    
    # Total tokens
    m = batch_size * seq_len
    n = intermediate_size
    k = hidden_size
    
    # Create FP16 inputs for quantization
    input_fp16 = torch.randn(m, k, device=device, dtype=torch.float16)
    weight_fp16 = torch.randn(k, n, device=device, dtype=torch.float16)
    
    # Quantize to FP8
    # Input quantization
    input_scale = torch.tensor(input_fp16.abs().max() / 448.0, device=device, dtype=torch.float32)
    a = (input_fp16 / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    scale_a = input_scale
    
    # Weight quantization  
    weight_scale = torch.tensor(weight_fp16.abs().max() / 448.0, device=device, dtype=torch.float32)
    b = (weight_fp16 / weight_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    scale_b = weight_scale
    
    # Output dtype
    out_dtype = torch.float16
    
    data = {
        "device": device,
        "dtype": out_dtype,
        "hw_info": hw_info,
        "a": a,
        "b": b,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "out_dtype": out_dtype,
        "m": m,
        "n": n,
        "k": k
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized CUTLASS FP8 GEMM operation."""
    
    # Get the cutlass_scaled_mm_dq function
    target, fq_name = resolve_target()
    
    # Call the CUTLASS kernel
    with torch.no_grad():
        result = target(
            a=data["a"],
            b=data["b"],
            scale_a=data["scale_a"],
            scale_b=data["scale_b"],
            out_dtype=data["out_dtype"]
        )
    
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
    if isinstance(current_result, torch.Tensor):
        assert current_result.shape == reference_result.shape, f"Shape mismatch: {current_result.shape} vs {reference_result.shape}"
        assert current_result.dtype == reference_result.dtype, f"Dtype mismatch: {current_result.dtype} vs {reference_result.dtype}"
        
        # FP8 operations have higher tolerance
        rtol, atol = 5e-2, 1e-2
        
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
        "p99_ms": times_ms[int(len(times_ms) * 0.99) if len(times_ms) > 1 else -1],
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
    
    # Timing
    if hw_info["device"] == "cuda":
        warmup = 5
        iters = 50
        result, timing_stats = time_gpu(lambda: experiment(data), warmup=warmup, iterations=iters)
        avg_ms = timing_stats["avg_ms"]
        p50_ms = timing_stats["p50_ms"]
        p95_ms = timing_stats["p95_ms"]
    else:
        # Should not reach here for FP8 tests
        error_data = {
            "target_resolved": True,
            "error": "FP8 operations require CUDA",
            "opt_path_hit": False
        }
        print(json.dumps(error_data))
        sys.exit(2)
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "8d75fe48ca5f46b7af0f5201d8500b9604eed769")
    impl_tag = os.getenv("IMPL_TAG", "child")
    ref_file = f"{prefix}_{impl_tag}_{commit_hash}_reference.pt"
    
    if reference:
        store_result(result, ref_file)
    
    if eqcheck and os.path.exists(ref_file):
        ref_result = load_result(ref_file)
        check_equivalence(result, ref_result)
    
    # Check if CUTLASS is available
    opt_path_hit = True  # We're directly calling cutlass_scaled_mm_dq
    
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
        "eq_level": os.getenv("PROB_EQ_LEVEL", "numeric"),
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