#!/usr/bin/env python3
"""
Performance test for commit: 187b85b7f38496653948a2aba546d53c09ada0f3
Message: [PD] Optimize custom mem pool usage and bump mooncake version (#7393)

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
    
    # Priority 2: Parse from commit metadata - use MHATokenToKVPool as primary target
    if not module_path:
        module_path = "sglang.srt.mem_cache.memory_pool"
    if not symbol_name:
        symbol_name = "MHATokenToKVPool"
    
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
    """Create realistic workload for the memory pool optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    # Enable custom memory pool for testing optimization
    os.environ["SGLANG_MOONCAKE_CUSTOM_MEM_POOL"] = "true"
    
    device = torch.device(hw_info["device"] if hw_info["device"] == "cuda" else "cpu")
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Realistic KV cache dimensions for 7B model
    num_layers = 32
    num_heads = 32
    head_dim = 128
    kv_heads = 32  # No GQA for this size
    block_size = 16
    max_num_blocks = 4096  # Enough for several sequences
    
    # Adjust workload for available memory
    if hw_info.get("memory_gb", 0) < 16:
        max_num_blocks = 1024
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "num_layers": num_layers,
        "num_heads": num_heads, 
        "head_dim": head_dim,
        "block_size": block_size,
        "max_num_blocks": max_num_blocks,
        "kv_heads": kv_heads,
        # Simulated allocation pattern for continuous batching
        "batch_sizes": [8, 16, 32, 64],
        "seq_lens": [128, 256, 512, 1024],
        "alloc_iterations": 20  # Number of allocation/deallocation cycles
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the memory pool operations."""
    KVPoolClass, fq_name = resolve_target()
    
    # Initialize the KV pool with custom memory pool enabled
    kv_pool = KVPoolClass(
        size=data["max_num_blocks"],
        block_size=data["block_size"],
        layer_num=data["num_layers"],
        kv_head_num=data["kv_heads"],
        head_dim=data["head_dim"],
        device=data["device"],
        dtype=data["dtype"]
    )
    
    # Track allocations for cleanup
    allocated_buffers = []
    allocation_times = []
    deallocation_times = []
    
    # Simulate allocation pattern in continuous batching
    for iteration in range(data["alloc_iterations"]):
        batch_size = data["batch_sizes"][iteration % len(data["batch_sizes"])]
        seq_len = data["seq_lens"][iteration % len(data["seq_lens"])]
        num_blocks_needed = (seq_len + data["block_size"] - 1) // data["block_size"]
        
        # Allocate blocks for batch
        batch_allocations = []
        for _ in range(batch_size):
            if hasattr(kv_pool, 'allocate'):
                # Time allocation
                if data["hw_info"]["device"] == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                try:
                    blocks = kv_pool.allocate(num_blocks_needed)
                    batch_allocations.append(blocks)
                except:
                    # Pool exhausted, skip
                    pass
                
                if data["hw_info"]["device"] == "cuda":
                    torch.cuda.synchronize()
                allocation_times.append(time.perf_counter() - start_time)
        
        allocated_buffers.extend(batch_allocations)
        
        # Deallocate some buffers (simulating request completion)
        if len(allocated_buffers) > batch_size // 2 and hasattr(kv_pool, 'free'):
            to_free = allocated_buffers[:batch_size // 2]
            for blocks in to_free:
                if data["hw_info"]["device"] == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                kv_pool.free(blocks)
                
                if data["hw_info"]["device"] == "cuda":
                    torch.cuda.synchronize()
                deallocation_times.append(time.perf_counter() - start_time)
            
            allocated_buffers = allocated_buffers[batch_size // 2:]
    
    # Clean up remaining allocations
    if hasattr(kv_pool, 'free'):
        for blocks in allocated_buffers:
            kv_pool.free(blocks)
    
    # Return timing statistics
    result = {
        "num_allocations": len(allocation_times),
        "num_deallocations": len(deallocation_times),
        "avg_alloc_ms": np.mean(allocation_times) * 1000 if allocation_times else 0,
        "avg_dealloc_ms": np.mean(deallocation_times) * 1000 if deallocation_times else 0,
        "total_alloc_ms": sum(allocation_times) * 1000,
        "total_dealloc_ms": sum(deallocation_times) * 1000,
        "pool_initialized": True,
        "custom_pool_enabled": kv_pool.enable_custom_mem_pool if hasattr(kv_pool, 'enable_custom_mem_pool') else False
    }
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    torch.save({"type": "dict", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    # For memory pool operations, check that allocations/deallocations succeeded
    assert isinstance(current_result, dict), f"Result type mismatch: {type(current_result)}"
    assert isinstance(reference_result, dict), f"Reference type mismatch: {type(reference_result)}"
    
    # Check pool initialization
    assert current_result.get("pool_initialized") == reference_result.get("pool_initialized"), \
        "Pool initialization mismatch"
    
    # Check custom pool enablement
    assert current_result.get("custom_pool_enabled") == reference_result.get("custom_pool_enabled"), \
        "Custom pool enablement mismatch"
    
    # Verify allocation/deallocation counts are similar (within 10% tolerance due to pool state)
    current_allocs = current_result.get("num_allocations", 0)
    ref_allocs = reference_result.get("num_allocations", 0)
    if ref_allocs > 0:
        alloc_ratio = current_allocs / ref_allocs
        assert 0.9 <= alloc_ratio <= 1.1, f"Allocation count mismatch: {current_allocs} vs {ref_allocs}"

# =======================
# Timing Implementation
# =======================
def time_memory_ops(func, warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
    """Time memory pool operations."""
    # Warmup
    for _ in range(warmup):
        _ = func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Timing
    times_ms = []
    for _ in range(iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)
    
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
    
    # Timing - reduced iterations for memory operations
    warmup = 3
    iters = 10
    
    result, timing_stats = time_memory_ops(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "187b85b7f38496653948a2aba546d53c09ada0f3")
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
        "eq_level": os.getenv("PROB_EQ_LEVEL", "behavioral"),
        "opt_path_hit": result.get("custom_pool_enabled", False)
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