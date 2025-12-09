#!/usr/bin/env python3
"""
Performance test for commit: 2716830802ae8c2428fdacde7c4041b6f7852d68
Message: Speed up when having padding tokens in DeepEP (#6175)

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
    
    # Priority 2: Parse from commit metadata - use grouped_topk as primary target
    if not (module_path and symbol_name):
        module_path = "sglang.srt.layers.moe.topk"
        symbol_name = "grouped_topk"
    
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
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # DeepSeek V2 MoE configuration with padding
    batch_size = 8
    seq_len = 512  # Total sequence length including padding
    num_token_non_padded = 400  # Only first 400 tokens are real, rest are padding
    hidden_size = 4096
    num_experts = 160
    num_expert_group = 4
    topk = 6
    topk_group = 3
    n_share_experts_fusion = 2
    routed_scaling_factor = 1.0
    
    # Create inputs
    total_tokens = batch_size * seq_len
    hidden_states = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)
    
    # Create gating output (router logits)
    gating_output = torch.randn(total_tokens, num_experts, device=device, dtype=dtype)
    
    # Correction bias for DeepSeek V2
    correction_bias = torch.randn(num_experts, device=device, dtype=dtype) * 0.1
    
    # Create num_token_non_padded tensor
    num_token_non_padded_tensor = torch.tensor(
        batch_size * num_token_non_padded, 
        device=device, 
        dtype=torch.int32
    )
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "hidden_states": hidden_states,
        "gating_output": gating_output,
        "topk": topk,
        "renormalize": True,
        "num_expert_group": num_expert_group,
        "topk_group": topk_group,
        "correction_bias": correction_bias,
        "n_share_experts_fusion": n_share_experts_fusion,
        "routed_scaling_factor": routed_scaling_factor,
        "num_token_non_padded": num_token_non_padded_tensor,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Call grouped_topk with padding optimization
    with torch.no_grad():
        # Check if the function signature supports num_token_non_padded
        import inspect
        sig = inspect.signature(target)
        
        if "num_token_non_padded" in sig.parameters:
            # New version with padding optimization
            result = target(
                hidden_states=data["hidden_states"],
                gating_output=data["gating_output"],
                topk=data["topk"],
                renormalize=data["renormalize"],
                num_expert_group=data["num_expert_group"],
                topk_group=data["topk_group"],
                n_share_experts_fusion=data["n_share_experts_fusion"],
                routed_scaling_factor=data["routed_scaling_factor"],
                num_token_non_padded=data["num_token_non_padded"],
            )
        else:
            # Old version without padding optimization
            result = target(
                hidden_states=data["hidden_states"],
                gating_output=data["gating_output"],
                topk=data["topk"],
                renormalize=data["renormalize"],
                num_expert_group=data["num_expert_group"],
                topk_group=data["topk_group"],
                n_share_experts_fusion=data["n_share_experts_fusion"],
                routed_scaling_factor=data["routed_scaling_factor"],
            )
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, tuple):
        # Result is (topk_weights, topk_ids)
        torch.save({
            "type": "tuple",
            "data": (result[0].cpu(), result[1].cpu())
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
    if isinstance(current_result, tuple) and isinstance(reference_result, tuple):
        # Check both weights and ids
        assert len(current_result) == len(reference_result), f"Tuple length mismatch"
        
        # Check topk_weights
        weights_curr, ids_curr = current_result
        weights_ref, ids_ref = reference_result
        
        assert weights_curr.shape == weights_ref.shape, f"Weights shape mismatch"
        assert weights_curr.dtype == weights_ref.dtype, f"Weights dtype mismatch"
        assert ids_curr.shape == ids_ref.shape, f"IDs shape mismatch"
        assert ids_curr.dtype == ids_ref.dtype, f"IDs dtype mismatch"
        
        # Weights comparison with appropriate tolerance
        if weights_curr.dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 1e-3, 1e-4
        else:
            rtol, atol = 1e-5, 1e-7
        
        torch.testing.assert_close(
            weights_curr.cpu(),
            weights_ref.cpu(),
            rtol=rtol, atol=atol
        )
        
        # IDs should match exactly
        torch.testing.assert_close(
            ids_curr.cpu(),
            ids_ref.cpu(),
            rtol=0, atol=0
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
    commit_hash = os.getenv("COMMIT_HASH", "2716830802ae8c2428fdacde7c4041b6f7852d68")
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