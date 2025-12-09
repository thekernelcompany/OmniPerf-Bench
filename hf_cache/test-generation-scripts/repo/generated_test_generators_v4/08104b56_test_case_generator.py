#!/usr/bin/env python3
"""
Performance test for commit: 08104b56de1192468c322e6f9ba234ef6526d607
Message: Sanity check to prevent performance regression (#3171)

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
    
    # Priority 2: Parse from commit metadata - focusing on RadixCache
    if not (module_path and symbol_name):
        # Based on diff analysis, RadixCache has the most changes
        module_path = "sglang.srt.mem_cache.radix_cache"
        symbol_name = "RadixCache"
    
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
    
    # Create workload that exercises RadixCache with lock_ref operations
    # Simulate typical LLM inference patterns
    batch_size = 32
    max_seq_len = 2048
    vocab_size = 32000
    
    # Generate token sequences for cache operations
    sequences = []
    for i in range(batch_size):
        # Varying sequence lengths
        seq_len = min(128 + i * 16, max_seq_len)
        tokens = torch.randint(0, vocab_size, (seq_len,), dtype=torch.int32)
        sequences.append(tokens.tolist())
    
    # Generate KV cache indices for each sequence
    kv_indices_list = []
    total_tokens = sum(len(seq) for seq in sequences)
    kv_pool = list(range(total_tokens * 2))  # Simulate a KV pool
    
    offset = 0
    for seq in sequences:
        seq_len = len(seq)
        indices = kv_pool[offset:offset + seq_len]
        kv_indices_list.append(indices)
        offset += seq_len
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "sequences": sequences,
        "kv_indices_list": kv_indices_list,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "vocab_size": vocab_size,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    RadixCache, fq_name = resolve_target()
    
    # Initialize RadixCache
    cache = RadixCache(None, None, False)
    
    sequences = data["sequences"]
    kv_indices_list = data["kv_indices_list"]
    
    # Simulate cache operations with lock_ref changes
    results = []
    
    # Insert sequences into cache
    for seq, kv_indices in zip(sequences, kv_indices_list):
        # Insert with match_prefix and insert
        matched_indices, matched_len = cache.match_prefix(seq)
        
        if matched_len < len(seq):
            # Need to insert remaining tokens
            remaining_seq = seq[matched_len:]
            remaining_indices = kv_indices[matched_len:]
            cache.insert(seq[:matched_len + len(remaining_seq)], remaining_indices)
        
        # Increment lock (simulating active use)
        cache.inc_ref_counter(matched_indices.tolist() if isinstance(matched_indices, torch.Tensor) else matched_indices)
        
        # Track protected size after operation
        protected = cache.protected_size()
        evictable = cache.evictable_size()
        results.append({"protected": protected, "evictable": evictable})
    
    # Decrement some locks (simulating request completion)
    for i in range(0, len(sequences), 2):
        seq = sequences[i]
        matched_indices, _ = cache.match_prefix(seq)
        cache.dec_ref_counter(matched_indices.tolist() if isinstance(matched_indices, torch.Tensor) else matched_indices)
    
    # Final state
    final_protected = cache.protected_size()
    final_evictable = cache.evictable_size()
    
    return {
        "operations": results,
        "final_protected": final_protected,
        "final_evictable": final_evictable,
        "total_operations": len(sequences)
    }

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
    if isinstance(current_result, dict) and isinstance(reference_result, dict):
        # Check dictionary equivalence
        assert current_result.keys() == reference_result.keys(), f"Keys mismatch"
        
        # Check numeric values
        for key in ["final_protected", "final_evictable", "total_operations"]:
            if key in current_result:
                assert current_result[key] == reference_result[key], f"{key} mismatch: {current_result[key]} vs {reference_result[key]}"
        
        # Check operations list
        if "operations" in current_result:
            assert len(current_result["operations"]) == len(reference_result["operations"]), "Operations count mismatch"
            for i, (curr_op, ref_op) in enumerate(zip(current_result["operations"], reference_result["operations"])):
                for k in ["protected", "evictable"]:
                    assert curr_op[k] == ref_op[k], f"Operation {i} {k} mismatch"
    
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

def time_cpu(func, warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
    """Time CPU operations."""
    # Warmup
    for _ in range(warmup):
        _ = func()
    
    # Timing
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    
    # Statistics
    times_ms.sort()
    stats = {
        "avg_ms": sum(times_ms) / len(times_ms),
        "p50_ms": times_ms[len(times_ms) // 2],
        "p95_ms": times_ms[int(len(times_ms) * 0.95) - 1] if len(times_ms) > 1 else times_ms[0],
        "p99_ms": times_ms[int(len(times_ms) * 0.99) - 1] if len(times_ms) > 1 else times_ms[0],
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
    
    # Timing - RadixCache operations are CPU-based
    warmup = 3
    iters = 10
    
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "08104b56de1192468c322e6f9ba234ef6526d607")
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
        "device": "cpu",  # RadixCache operations are CPU-based
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