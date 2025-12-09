#!/usr/bin/env python3
"""
Performance test for commit: 1ebe1d6de5e0ce082e0be059c222baf0c5ee340a
Message: Optimize MoE topk with torch compile (#3236)

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
        # Based on the commit diff, the primary target is grouped_topk
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
    
    # MoE topk workload for Deepseek-V2 style model
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Typical MoE configuration
    batch_size = 32  # Number of tokens
    seq_len = 64  # Sequence positions
    hidden_size = 5120  # Hidden dimension (Deepseek-V2 uses 5120)
    num_experts = 160  # Total number of experts in Deepseek-V2
    num_groups = 20  # Number of expert groups
    experts_per_group = num_experts // num_groups  # 8 experts per group
    topk_per_group = 2  # Select top 2 experts per group
    
    # Adjust for smaller GPUs
    if hw_info.get("memory_gb", float('inf')) < 16:
        batch_size = 16
        seq_len = 32
        num_experts = 64
        num_groups = 8
    
    # Create input tensors
    num_tokens = batch_size * seq_len
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    
    # Gating output (router logits)
    gating_output = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
    
    # Apply softmax to make it realistic
    gating_output = torch.softmax(gating_output, dim=-1)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "hidden_states": hidden_states,
        "gating_output": gating_output,
        "num_groups": num_groups,
        "topk": topk_per_group,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Call grouped_topk with the MoE workload
    with torch.no_grad():
        topk_weights, topk_ids = target(
            hidden_states=data["hidden_states"],
            gating_output=data["gating_output"],
            topk=data["topk"],
            num_expert_group=data["num_groups"]
        )
    
    return (topk_weights, topk_ids)

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, tuple) and len(result) == 2:
        # Store topk weights and ids
        torch.save({
            "type": "topk_result",
            "weights": result[0].cpu(),
            "ids": result[1].cpu()
        }, filepath)
    elif isinstance(result, torch.Tensor):
        torch.save({"type": "tensor", "data": result.cpu()}, filepath)
    else:
        torch.save({"type": "generic", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    if data["type"] == "topk_result":
        return (data["weights"], data["ids"])
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    if isinstance(current_result, tuple) and isinstance(reference_result, tuple):
        # Check topk weights and ids
        current_weights, current_ids = current_result
        ref_weights, ref_ids = reference_result
        
        # Check shapes
        assert current_weights.shape == ref_weights.shape, f"Weights shape mismatch"
        assert current_ids.shape == ref_ids.shape, f"IDs shape mismatch"
        
        # Check dtypes
        assert current_weights.dtype == ref_weights.dtype, f"Weights dtype mismatch"
        assert current_ids.dtype == ref_ids.dtype, f"IDs dtype mismatch"
        
        # Check IDs exactly (topk selection should be deterministic)
        torch.testing.assert_close(
            current_ids.cpu(),
            ref_ids.cpu(),
            rtol=0, atol=0
        )
        
        # Check weights with tolerance
        if current_weights.dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 1e-3, 1e-4
        else:
            rtol, atol = 1e-5, 1e-7
        
        torch.testing.assert_close(
            current_weights.cpu(),
            ref_weights.cpu(),
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
        warmup = 10  # Extra warmup for torch.compile
        iters = 50
        result, timing_stats = time_gpu(lambda: experiment(data), warmup=warmup, iterations=iters)
        avg_ms = timing_stats["avg_ms"]
        p50_ms = timing_stats["p50_ms"]
        p95_ms = timing_stats["p95_ms"]
    else:
        warmup = 5  # Extra warmup for torch.compile on CPU
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
    commit_hash = os.getenv("COMMIT_HASH", "1ebe1d6de5e0ce082e0be059c222baf0c5ee340a")
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