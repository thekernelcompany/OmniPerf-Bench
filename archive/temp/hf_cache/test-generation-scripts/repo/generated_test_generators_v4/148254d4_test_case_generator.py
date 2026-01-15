#!/usr/bin/env python3
"""
Performance test for commit: 148254d4db8bf3bffee23710cd1acbd5711ebd1b
Message: Improve moe reduce sum kernel performance (#2705)

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
        # Check if we're on AMD GPU (HIP)
        hw_info["is_hip"] = hasattr(torch.version, 'hip') and torch.version.hip is not None
    else:
        hw_info["device"] = "cpu"
        hw_info["device_name"] = "CPU"
        hw_info["memory_gb"] = 0
        hw_info["is_hip"] = False
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
        # Based on the commit, targeting fused_experts_impl
        module_path = "sglang.srt.layers.moe.fused_moe_triton.fused_moe"
        symbol_name = "fused_experts_impl"
    
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
    
    # MoE configuration for Mixtral-style model
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Adjust sizes based on available memory
    if hw_info.get("memory_gb", 0) < 16:
        batch_size = 2
        seq_len = 1024
    else:
        batch_size = 4
        seq_len = 2048
    
    # MoE parameters (Mixtral 8x7B style)
    num_experts = 8
    top_k = 2
    hidden_size = 4096
    intermediate_size = 14336  # Typical for Mixtral
    
    # Total tokens
    num_tokens = batch_size * seq_len
    
    # Create input hidden states
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    
    # Create expert weights (w1, w2, w3 for SwiGLU)
    w1 = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=dtype)  
    w3 = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype)
    
    # Simulate router outputs - top-k selection
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
    topk_weights, topk_ids = torch.topk(router_logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    
    # Sort by expert ID for efficient batching
    topk_ids_flat = topk_ids.view(-1)
    topk_weights_flat = topk_weights.view(-1)
    
    sorted_indices = torch.argsort(topk_ids_flat)
    sorted_token_ids = torch.div(sorted_indices, top_k, rounding_mode='floor')
    sorted_expert_ids = topk_ids_flat[sorted_indices]
    sorted_weights = topk_weights_flat[sorted_indices]
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "sorted_token_ids": sorted_token_ids,
        "sorted_expert_ids": sorted_expert_ids,
        "sorted_weights": sorted_weights,
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "top_k": top_k,
        "inplace": False,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Call fused_experts_impl with the MoE parameters
    with torch.no_grad():
        result = target(
            hidden_states=data["hidden_states"],
            w1=data["w1"],
            w2=data["w2"],
            w3=data["w3"],
            topk_weights=data["topk_weights"],
            topk_ids=data["topk_ids"],
            inplace=data["inplace"],
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
        p95_ms = times[int(len(times) * 0.95) - 1] if len(times) > 1 else avg_ms
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "148254d4db8bf3bffee23710cd1acbd5711ebd1b")
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
        "opt_path_hit": True,
        "is_hip": hw_info.get("is_hip", False)
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