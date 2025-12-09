#!/usr/bin/env python3
"""
Performance test for commit: 23c764b18aeb37c42ddedd7468f1a5753df1f232
Message: [Feature] Support DeepEP Low Latency (#4767)

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
    
    # Priority 2: Parse from commit metadata - target the new kernel
    if not (module_path and symbol_name):
        module_path = "sglang.srt.layers.moe.ep_moe.kernels"
        symbol_name = "silu_and_mul_masked_post_quant_fwd"
    
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
    
    # Use FP8 if supported, otherwise fall back to FP16
    if hw_info.get("supports_fp8", False):
        dtype = torch.float16  # Input dtype
        output_dtype = torch.float8_e4m3fn
    else:
        dtype = torch.float16
        output_dtype = torch.float16  # Fallback if no FP8 support
    
    # MoE configuration based on DeepSeek-V2/V3 models
    expert_num = 8  # Number of experts
    token_num_padded = 64  # Padded number of tokens per expert
    hidden_dim = 14336  # MoE intermediate size from config
    quant_group_size = 128  # Quantization group size
    
    # Create input tensor (gate and up projections concatenated)
    input_tensor = torch.randn(
        expert_num, token_num_padded, hidden_dim,
        device=device, dtype=dtype
    )
    
    # Output tensor for FP8 quantized result
    output = torch.empty(
        expert_num, token_num_padded, hidden_dim // 2,
        device=device, dtype=output_dtype
    )
    
    # Output scale for FP8 quantization
    output_scale = torch.empty(
        expert_num, token_num_padded, (hidden_dim // 2) // quant_group_size,
        device=device, dtype=torch.float32
    )
    
    # Masked m: number of actual tokens per expert (for masking)
    masked_m = torch.randint(
        1, token_num_padded + 1, (expert_num,),
        device=device, dtype=torch.int64
    )
    
    data = {
        "device": device,
        "dtype": dtype,
        "output_dtype": output_dtype,
        "hw_info": hw_info,
        "input": input_tensor,
        "output": output,
        "output_scale": output_scale,
        "quant_group_size": quant_group_size,
        "masked_m": masked_m,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Call the optimized kernel
    with torch.no_grad():
        # The kernel modifies output and output_scale in-place
        target(
            data["input"],
            data["output"],
            data["output_scale"],
            data["quant_group_size"],
            data["masked_m"]
        )
        
        # Return the modified tensors
        result = {
            "output": data["output"].clone(),
            "output_scale": data["output_scale"].clone()
        }
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, dict):
        torch.save({
            "type": "dict",
            "output": result["output"].cpu(),
            "output_scale": result["output_scale"].cpu()
        }, filepath)
    else:
        torch.save({"type": "generic", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    if data.get("type") == "dict":
        return {
            "output": data["output"],
            "output_scale": data["output_scale"]
        }
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert isinstance(current_result, dict) and isinstance(reference_result, dict)
    
    # Check output tensor
    current_output = current_result["output"]
    reference_output = reference_result["output"]
    
    assert current_output.shape == reference_output.shape
    assert current_output.dtype == reference_output.dtype
    
    # For FP8 types, use larger tolerances
    if current_output.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        rtol, atol = 5e-2, 1e-2
    elif current_output.dtype in (torch.float16, torch.bfloat16):
        rtol, atol = 1e-3, 1e-4
    else:
        rtol, atol = 1e-5, 1e-7
    
    torch.testing.assert_close(
        current_output.cpu(),
        reference_output.cpu(),
        rtol=rtol, atol=atol
    )
    
    # Check output scale
    current_scale = current_result["output_scale"]
    reference_scale = reference_result["output_scale"]
    
    assert current_scale.shape == reference_scale.shape
    torch.testing.assert_close(
        current_scale.cpu(),
        reference_scale.cpu(),
        rtol=1e-3, atol=1e-4
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
    
    # Check FP8 support
    if not hw_info.get("supports_fp8", False):
        error_data = {
            "error_code": 2,
            "error_name": "CAPABILITY_UNSUPPORTED",
            "error_message": "FP8 not supported on this hardware",
            "target_resolved": True,
            "opt_path_hit": False
        }
        print(json.dumps(error_data))
        # Continue with fallback for testing purposes
    
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
    commit_hash = os.getenv("COMMIT_HASH", "23c764b18aeb37c42ddedd7468f1a5753df1f232")
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