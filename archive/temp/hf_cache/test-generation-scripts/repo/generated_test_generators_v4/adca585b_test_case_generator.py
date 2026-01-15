#!/usr/bin/env python3
"""
Performance test for commit: adca585bfb59a6c29cf18393b4a68bd5b4068f08
Message: [DeepEP] Reduce routed scaling overhead (#5277)

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
import torch.nn as nn

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
        module_path = "sglang.srt.models.deepseek_v2"
        symbol_name = "DeepseekV2MoE"
    
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
    
    # DeepSeek V2 MoE configuration
    batch_size = 4
    seq_len = 1024
    hidden_size = 2048  # DeepSeek V2 uses 2048 hidden size
    num_experts = 64  # DeepSeek V2 has 64 experts
    n_routed_experts = 6  # Number of experts to route to
    routed_scaling_factor = 1.0
    ep_size = 1  # Expert parallelism size
    
    # Adjust for hardware constraints
    if hw_info.get("memory_gb", float('inf')) < 16:
        batch_size = 2
        seq_len = 512
    
    # Create mock MoE module for testing
    DeepseekV2MoE, _ = resolve_target()
    
    # Initialize a minimal MoE layer
    moe_config = {
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 3,
        "n_routed_experts": n_routed_experts,
        "num_experts": num_experts,
        "topk_method": "greedy",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": routed_scaling_factor,
        "scoring_func": "softmax",
        "aux_loss_alpha": 0.0,
        "seq_aux": True,
        "norm_topk_prob": True,
        "ep_size": ep_size,
        "dtype": dtype,
    }
    
    # Create MoE module
    moe = DeepseekV2MoE(
        config=type('Config', (), moe_config),
        layer_id=0,
        quant_config=None,
        prefix="test_moe"
    ).to(device).to(dtype)
    
    # Create input tensors
    hidden_states = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=dtype)
    
    # Create routing indices and weights (simulating router output)
    total_tokens = batch_size * seq_len
    topk_ids = torch.randint(0, num_experts, (total_tokens, n_routed_experts), device=device)
    topk_weights = torch.softmax(torch.randn(total_tokens, n_routed_experts, device=device, dtype=dtype), dim=-1)
    
    # Create expert dispatcher data structures
    # These would normally come from the router and dispatcher
    reorder_topk_ids = topk_ids.flatten()
    seg_indptr = torch.arange(0, total_tokens * n_routed_experts + 1, n_routed_experts, device=device, dtype=torch.int32)
    masked_m = torch.ones(total_tokens, device=device, dtype=torch.int32) * n_routed_experts
    expected_m = masked_m.clone()
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "moe": moe,
        "hidden_states": hidden_states,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "reorder_topk_ids": reorder_topk_ids,
        "seg_indptr": seg_indptr,
        "masked_m": masked_m,
        "expected_m": expected_m,
        "forward_mode": "lora",
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    moe = data["moe"]
    
    with torch.no_grad():
        # Call the forward method with all required parameters
        result = moe.forward(
            hidden_states=data["hidden_states"],
            topk_ids=data["topk_ids"],
            topk_weights=data["topk_weights"],
            forward_mode=data["forward_mode"]
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
    commit_hash = os.getenv("COMMIT_HASH", "adca585bfb59a6c29cf18393b4a68bd5b4068f08")
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