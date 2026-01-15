#!/usr/bin/env python3
"""
Performance test for commit: c087ddd6865a52634326a05af66429cb5531cd16
Message: Refine pre_reorder_triton_kernel slightly to improve performance (#6627)

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
        # Based on the commit diff, the main optimization is in pre_reorder_triton_kernel
        module_path = "sglang.srt.layers.moe.ep_moe.kernels"
        symbol_name = "pre_reorder_triton_kernel"
    
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
    
    if hw_info["device"] != "cuda":
        error_data = {
            "target_resolved": True,
            "error": "Triton kernels require CUDA device",
            "error_code": 2,
            "error_name": "CAPABILITY_UNSUPPORTED"
        }
        print(json.dumps(error_data))
        sys.exit(2)
    
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Model configuration for MoE layer testing
    batch_size = 256  # Moderate batch size for stable timing
    topk = 2  # Number of experts per token
    hidden_size = 4096  # Standard hidden size for 7B models
    block_size = 512  # Block size for Triton kernel
    expert_range = (0, 7)  # 8 experts total
    num_experts = expert_range[1] - expert_range[0] + 1
    
    # Create input tensors matching the kernel interface
    input_ptr = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    gateup_input_ptr = torch.zeros(batch_size * topk, hidden_size, dtype=dtype, device=device)
    src2dst_ptr = torch.randint(0, batch_size * topk, (batch_size, topk), dtype=torch.int32, device=device)
    topk_ids_ptr = torch.randint(expert_range[0], expert_range[1] + 1, (batch_size, topk), dtype=torch.int32, device=device)
    a1_scales_ptr = torch.rand(num_experts, dtype=torch.float32, device=device)
    
    # Flatten tensors as required by the kernel
    input_ptr_flat = input_ptr.view(-1)
    gateup_input_ptr_flat = gateup_input_ptr.view(-1)
    src2dst_ptr_flat = src2dst_ptr.view(-1)
    topk_ids_ptr_flat = topk_ids_ptr.view(-1)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "batch_size": batch_size,
        "topk": topk,
        "hidden_size": hidden_size,
        "block_size": block_size,
        "expert_range": expert_range,
        "input_ptr": input_ptr_flat,
        "gateup_input_ptr": gateup_input_ptr_flat,
        "src2dst_ptr": src2dst_ptr_flat,
        "topk_ids_ptr": topk_ids_ptr_flat,
        "a1_scales_ptr": a1_scales_ptr,
        # Keep original tensors for result storage
        "gateup_input_orig": gateup_input_ptr.clone()
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Clone output tensor to avoid overwriting across iterations
    gateup_input_ptr = data["gateup_input_orig"].clone().view(-1)
    
    # Execute the Triton kernel with grid size based on batch size
    with torch.no_grad():
        target[(data["batch_size"],)](
            data["input_ptr"],
            gateup_input_ptr,
            data["src2dst_ptr"],
            data["topk_ids_ptr"],
            data["a1_scales_ptr"],
            data["expert_range"][0],  # start_expert_id
            data["expert_range"][1],  # end_expert_id
            data["topk"],
            data["hidden_size"],
            data["block_size"]
        )
    
    # Return the output tensor reshaped to original dimensions
    result = gateup_input_ptr.view(data["batch_size"] * data["topk"], data["hidden_size"])
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
        
        # Determine tolerances based on dtype
        if current_result.dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 1e-3, 1e-4
        else:
            rtol, atol = 1e-5, 1e-7
        
        # Move to CPU for comparison
        current_cpu = current_result.cpu()
        reference_cpu = reference_result.cpu()
        
        # Handle special values
        if torch.isnan(current_cpu).any() or torch.isnan(reference_cpu).any():
            assert torch.isnan(current_cpu).equal(torch.isnan(reference_cpu)), "NaN mismatch"
            # Compare non-NaN values
            mask = ~torch.isnan(current_cpu)
            torch.testing.assert_close(
                current_cpu[mask],
                reference_cpu[mask],
                rtol=rtol, atol=atol
            )
        else:
            torch.testing.assert_close(
                current_cpu,
                reference_cpu,
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
    
    # Clear cache for cleaner timing
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
    
    # Timing - GPU only for Triton kernels
    warmup = 5
    iters = 50
    result, timing_stats = time_gpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "c087ddd6865a52634326a05af66429cb5531cd16")
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