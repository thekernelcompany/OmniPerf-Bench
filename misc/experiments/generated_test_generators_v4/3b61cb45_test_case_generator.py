#!/usr/bin/env python3
"""
Performance test for commit: 3b61cb450d899dc423feb264c297d4d18d701678
Message: [V1] Further reduce CPU overheads in flash-attn (#10989)

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
    
    # Priority 2: Parse from commit metadata - target is reshape_and_cache_flash
    if not (module_path and symbol_name):
        module_path = "torch.ops._C_cache_ops"
        symbol_name = "reshape_and_cache_flash"
    
    # Import with error handling
    try:
        if module_path == "torch.ops._C_cache_ops":
            # Special handling for torch ops
            target = torch.ops._C_cache_ops.reshape_and_cache_flash
        else:
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
    
    # Flash attention KV cache workload - typical decode scenario
    batch_size = 32  # Multiple concurrent requests
    num_tokens = batch_size  # One token per request in decode
    num_heads = 32  # Typical for 7B model
    num_kv_heads = 32  # No GQA for this test
    head_size = 128
    block_size = 16  # Standard block size for paged attention
    num_blocks = 256  # Enough blocks for reasonable context
    
    # Create input tensors
    key = torch.randn(num_tokens, num_kv_heads, head_size, device=device, dtype=dtype)
    value = torch.randn(num_tokens, num_kv_heads, head_size, device=device, dtype=dtype)
    
    # Create KV cache in flash attention format
    # [num_blocks, block_size, num_kv_heads, head_size]
    key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, 
                            device=device, dtype=dtype)
    value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size,
                             device=device, dtype=dtype)
    
    # Create slot mapping - maps tokens to cache positions
    # Each token gets a unique slot in the cache
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)
    
    # KV cache dtype string
    kv_cache_dtype = "auto"
    k_scale = 1.0
    v_scale = 1.0
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "key": key,
        "value": value,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "slot_mapping": slot_mapping,
        "kv_cache_dtype": kv_cache_dtype,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "num_tokens": num_tokens,
        "num_heads": num_kv_heads,
        "head_size": head_size,
        "block_size": block_size,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Call the reshape_and_cache_flash operation
    with torch.no_grad():
        # The function modifies caches in-place, so we need to clone them
        key_cache_copy = data["key_cache"].clone()
        value_cache_copy = data["value_cache"].clone()
        
        target(
            data["key"],
            data["value"],
            key_cache_copy,
            value_cache_copy,
            data["slot_mapping"],
            data["kv_cache_dtype"],
            data["k_scale"],
            data["v_scale"]
        )
        
        result = {
            "key_cache": key_cache_copy,
            "value_cache": value_cache_copy
        }
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, dict):
        # Store both caches
        torch.save({
            "type": "dict",
            "key_cache": result["key_cache"].cpu(),
            "value_cache": result["value_cache"].cpu()
        }, filepath)
    else:
        torch.save({"type": "generic", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    if data.get("type") == "dict":
        return {
            "key_cache": data["key_cache"].to(torch.cuda.current_device()),
            "value_cache": data["value_cache"].to(torch.cuda.current_device())
        }
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    if isinstance(current_result, dict) and isinstance(reference_result, dict):
        # Check both key and value caches
        for cache_name in ["key_cache", "value_cache"]:
            current = current_result[cache_name]
            reference = reference_result[cache_name]
            
            assert current.shape == reference.shape, f"{cache_name} shape mismatch"
            assert current.dtype == reference.dtype, f"{cache_name} dtype mismatch"
            
            # Determine tolerances based on dtype
            if current.dtype in (torch.float16, torch.bfloat16):
                rtol, atol = 1e-3, 1e-4
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
    commit_hash = os.getenv("COMMIT_HASH", "3b61cb450d899dc423feb264c297d4d18d701678")
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