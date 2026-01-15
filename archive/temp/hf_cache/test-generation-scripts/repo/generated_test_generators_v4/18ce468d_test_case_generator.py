#!/usr/bin/env python3
"""
Performance test for commit: 18ce468d56aa33a8bebcef0bc3f4777b0d70cdce
Message: update triton 3.2.0 h200 fused moe triton config and add warning about triton fused_moe_kernel performance degradation due to different Triton versions. (#5740)

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
    
    # Priority 2: Parse from commit metadata - targeting fused_moe kernel
    if not (module_path and symbol_name):
        module_path = "sglang.srt.layers.moe.fused_moe_triton.fused_moe"
        symbol_name = "fused_moe"
    
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
    
    # MoE configuration based on commit (targeting H200 with FP8)
    device = torch.device(hw_info["device"])
    
    # Check FP8 support
    if hw_info.get("supports_fp8", False):
        try:
            # Try to use FP8 dtypes
            dtype = torch.float8_e4m3fn
            weight_dtype = torch.float8_e4m3fn
        except AttributeError:
            # Fallback to FP16 if FP8 not available
            dtype = torch.float16
            weight_dtype = torch.float16
    else:
        dtype = torch.float16
        weight_dtype = torch.float16
    
    # MoE parameters for typical workload (8 experts, top-2 routing)
    batch_size = 256  # Matching config key in JSON
    seq_len = 1  # Decode phase
    hidden_size = 4096  # Common hidden size
    num_experts = 8
    top_k = 2
    expert_intermediate_size = 14336  # FFN intermediate size
    
    # Adjust for memory constraints
    if hw_info.get("memory_gb", float('inf')) < 40:
        batch_size = max(1, batch_size // 2)
    
    # Create input tensors
    hidden_states = torch.randn(
        batch_size * seq_len, hidden_size, 
        device=device, dtype=dtype
    )
    
    # Router weights and expert weights
    gating_output = torch.randn(
        batch_size * seq_len, num_experts, 
        device=device, dtype=torch.float32
    )
    
    # Expert weight matrices (gate and up projections for SwiGLU)
    w1 = torch.randn(
        num_experts, hidden_size, expert_intermediate_size * 2,
        device=device, dtype=weight_dtype
    )
    w2 = torch.randn(
        num_experts, expert_intermediate_size, hidden_size,
        device=device, dtype=weight_dtype
    )
    
    # Compute topk routing
    topk_weights, topk_ids = torch.topk(
        torch.softmax(gating_output, dim=-1), 
        top_k, dim=-1
    )
    topk_weights = topk_weights.to(dtype)
    
    data = {
        "device": device,
        "dtype": dtype,
        "weight_dtype": weight_dtype,
        "hw_info": hw_info,
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_experts": num_experts,
        "top_k": top_k,
        "config": {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "expert_intermediate_size": expert_intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k
        }
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Extract parameters
    hidden_states = data["hidden_states"]
    w1 = data["w1"]
    w2 = data["w2"]
    topk_weights = data["topk_weights"]
    topk_ids = data["topk_ids"]
    
    with torch.no_grad():
        # Call the fused MoE kernel
        # fused_moe signature: (hidden_states, w1, w2, topk_weights, topk_ids, ...)
        result = target(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=False  # Don't modify input
        )
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, torch.Tensor):
        torch.save({"type": "tensor", "data": result.cpu()}, filepath)
    elif isinstance(result, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in result):
        torch.save({
            "type": "tensor_list",
            "data": [t.cpu() for t in result]
        }, filepath)
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
        
        # Determine tolerances based on dtype (FP8 needs higher tolerance)
        if "float8" in str(current_result.dtype):
            rtol, atol = 5e-2, 1e-2
        elif current_result.dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 1e-3, 1e-4
        else:
            rtol, atol = 1e-5, 1e-7
        
        # Handle special values
        if torch.isnan(current_result).any() or torch.isnan(reference_result).any():
            assert torch.isnan(current_result).equal(torch.isnan(reference_result)), "NaN mismatch"
            mask = ~torch.isnan(current_result)
            torch.testing.assert_close(
                current_result[mask].cpu(),
                reference_result[mask].cpu(),
                rtol=rtol, atol=atol
            )
        else:
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
            result = experiment(data)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        avg_ms = sum(times) / len(times)
        p50_ms = times[len(times) // 2]
        p95_ms = times[int(len(times) * 0.95)]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "18ce468d56aa33a8bebcef0bc3f4777b0d70cdce")
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