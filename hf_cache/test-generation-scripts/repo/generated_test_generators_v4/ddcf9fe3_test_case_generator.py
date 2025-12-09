#!/usr/bin/env python3
"""
Performance test for commit: ddcf9fe3beacd8aed573c711942194dd02350da4
Message: Optimize triton attention custom mask (#3731)

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
        # From the commit JSON, the API is extend_attention_fwd
        module_path = "sglang.srt.layers.attention.triton_ops.extend_attention"
        symbol_name = "extend_attention_fwd"
    
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
    
    # Triton attention workload with custom mask
    # Simulating a prefill scenario with custom attention mask
    device = torch.device(hw_info["device"] if hw_info["device"] == "cuda" else "cpu")
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Adjust batch size based on memory
    if hw_info.get("memory_gb", float('inf')) < 16:
        batch_size = 2
        seq_len = 1024
    else:
        batch_size = 4
        seq_len = 2048
    
    # Standard attention dimensions
    num_heads = 32
    head_dim = 128
    num_kv_heads = 32  # No GQA for simplicity
    
    # Attention inputs - extend phase (new tokens being processed)
    q_extend = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                           device=device, dtype=dtype)
    k_extend = torch.randn(batch_size, num_kv_heads, seq_len, head_dim,
                           device=device, dtype=dtype)
    v_extend = torch.randn(batch_size, num_kv_heads, seq_len, head_dim,
                           device=device, dtype=dtype)
    o_extend = torch.zeros(batch_size, num_heads, seq_len, head_dim,
                           device=device, dtype=dtype)
    
    # Attention inputs - prefix cache (already processed tokens)
    prefix_len = seq_len // 2  # Half the sequence is already cached
    k_prefix = torch.randn(batch_size, num_kv_heads, prefix_len, head_dim,
                           device=device, dtype=dtype)
    v_prefix = torch.randn(batch_size, num_kv_heads, prefix_len, head_dim,
                           device=device, dtype=dtype)
    
    # Per-request metadata
    start_extend_len = torch.zeros(batch_size, device=device, dtype=torch.int32)
    for i in range(batch_size):
        start_extend_len[i] = i * seq_len
    seq_lens_extend = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    seq_lens_prefix = torch.full((batch_size,), prefix_len, device=device, dtype=torch.int32)
    max_len_extend = seq_len
    
    # Custom mask - key optimization target
    # Create a realistic custom attention mask (e.g., for structured attention patterns)
    total_seq_len = prefix_len + seq_len
    custom_mask = torch.randn(batch_size, total_seq_len, total_seq_len, 
                              device=device, dtype=dtype)
    # Make it somewhat sparse/structured
    custom_mask = torch.where(custom_mask > 0, custom_mask, torch.zeros_like(custom_mask))
    
    # Scaling factor
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Logit cap (for stability)
    logit_cap = 0.0
    
    # Skip prefix custom mask - this is the optimization being tested
    skip_prefix_custom_mask = True  # The optimized behavior
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "q_extend": q_extend,
        "k_extend": k_extend,
        "v_extend": v_extend,
        "o_extend": o_extend,
        "k_prefix": k_prefix,
        "v_prefix": v_prefix,
        "custom_mask": custom_mask,
        "start_extend_len": start_extend_len,
        "seq_lens_extend": seq_lens_extend,
        "seq_lens_prefix": seq_lens_prefix,
        "max_len_extend": max_len_extend,
        "sm_scale": sm_scale,
        "logit_cap": logit_cap,
        "skip_prefix_custom_mask": skip_prefix_custom_mask,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Call the extend_attention_fwd function with the optimization flag
    with torch.no_grad():
        result = target(
            q_extend=data["q_extend"],
            k_extend=data["k_extend"],
            v_extend=data["v_extend"],
            o_extend=data["o_extend"],
            k_prefix=data["k_prefix"],
            v_prefix=data["v_prefix"],
            custom_mask=data["custom_mask"],
            start_extend_len=data["start_extend_len"],
            seq_lens_extend=data["seq_lens_extend"],
            seq_lens_prefix=data["seq_lens_prefix"],
            max_len_extend=data["max_len_extend"],
            sm_scale=data["sm_scale"],
            logit_cap=data["logit_cap"],
            skip_prefix_custom_mask=data["skip_prefix_custom_mask"],
        )
    
    return data["o_extend"]  # The output is written in-place

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
    
    # Check if we can run this test
    if hw_info["device"] != "cuda":
        error_data = {
            "error_code": 2,
            "error_name": "CAPABILITY_UNSUPPORTED",
            "error_message": "Triton kernels require CUDA device",
            "target_resolved": True,
            "opt_path_hit": False
        }
        print(json.dumps(error_data))
        sys.exit(2)
    
    # Timing
    warmup = 5
    iters = 50
    result, timing_stats = time_gpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "ddcf9fe3beacd8aed573c711942194dd02350da4")
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