#!/usr/bin/env python3
"""
Performance test for commit: c2f212d672ccaf8a1e5ef09099e981d943600b14
Message: optimize MiniMax-Text-01 lightning_attn_decode triton (#2966)

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
        # Based on the diff, the target is lightning_attn_decode in the benchmark file
        module_path = "benchmark.kernels.minmax-text-01-lighting_attention.benchmark_lighting_attention_decode"
        symbol_name = "lightning_attn_decode"
    
    # Import with error handling
    try:
        # Try to import from the actual source location
        # Since this is a benchmark file, we need to handle it specially
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Try direct import first
        try:
            from benchmark.kernels.minmax_text_01_lighting_attention.benchmark_lighting_attention_decode import lightning_attn_decode
            target = lightning_attn_decode
            fq_name = f"benchmark.kernels.minmax_text_01_lighting_attention.benchmark_lighting_attention_decode.lightning_attn_decode"
        except ImportError:
            # Fall back to dynamic import
            module_path = module_path.replace("-", "_")  # Handle dash in module name
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
    
    # Lightning attention decode workload
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Typical decode configuration
    batch_size = 8  # Moderate batch for decode
    num_heads = 32  # Typical for 7B models
    seq_len = 1  # Decode mode (single token)
    d = 128  # Head dimension
    e = 128  # Value dimension
    
    # Adjust for available memory
    if hw_info.get("memory_gb", float('inf')) < 16:
        batch_size = 4
    
    # Create tensors with realistic dimensions
    q = torch.randn(batch_size, num_heads, seq_len, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, e, device=device, dtype=dtype)
    kv = torch.randn(batch_size, num_heads, d, e, device=device, dtype=torch.float32)
    s = torch.ones(batch_size, num_heads, device=device, dtype=torch.float32)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "q": q,
        "k": k,
        "v": v,
        "kv": kv,
        "s": s,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "d": d,
        "e": e
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Call the lightning attention decode function
    with torch.no_grad():
        # The function returns (output, kv_out)
        o, kv_out = target(
            data["q"],
            data["k"],
            data["v"],
            data["kv"],
            data["s"]
        )
    
    return (o, kv_out)

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, tuple):
        # Store tuple of tensors
        torch.save({
            "type": "tuple",
            "data": [r.cpu() if isinstance(r, torch.Tensor) else r for r in result]
        }, filepath)
    elif isinstance(result, torch.Tensor):
        torch.save({"type": "tensor", "data": result.cpu()}, filepath)
    else:
        torch.save({"type": "generic", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    stored_data = data.get("data", data)
    
    if data.get("type") == "tuple":
        # Convert list back to tuple and move tensors to correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in stored_data)
    elif isinstance(stored_data, torch.Tensor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return stored_data.to(device)
    return stored_data

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    if isinstance(current_result, tuple) and isinstance(reference_result, tuple):
        assert len(current_result) == len(reference_result), f"Tuple length mismatch"
        for i, (curr, ref) in enumerate(zip(current_result, reference_result)):
            try:
                check_single_equivalence(curr, ref)
            except AssertionError as e:
                raise AssertionError(f"Mismatch at tuple index {i}: {e}")
    else:
        check_single_equivalence(current_result, reference_result)

def check_single_equivalence(current: Any, reference: Any) -> None:
    """Check equivalence for single values."""
    if isinstance(current, torch.Tensor):
        assert current.shape == reference.shape
        assert current.dtype == reference.dtype
        
        # Determine tolerances based on dtype
        if current.dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 1e-3, 1e-4
        elif current.dtype == torch.float32:
            rtol, atol = 1e-5, 1e-6
        else:
            rtol, atol = 1e-5, 1e-7
        
        torch.testing.assert_close(
            current.cpu(),
            reference.cpu(),
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
    
    # Clear cache before timing
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
            _ = experiment(data)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        avg_ms = sum(times) / len(times)
        p50_ms = times[len(times) // 2]
        p95_ms = times[int(len(times) * 0.95) - 1] if len(times) > 1 else avg_ms
        # Produce a result for reference handling
        result = experiment(data)
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "c2f212d672ccaf8a1e5ef09099e981d943600b14")
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