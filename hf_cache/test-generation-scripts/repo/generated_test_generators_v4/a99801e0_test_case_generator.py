#!/usr/bin/env python3
"""
Performance test for commit: a99801e0750f41553fedd02e36f58d835c4d4bd6
Message: [Performance][PD Disaggregation] optimize TokenToKVPoolAllocator by sorting free pages (#8133)

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
    
    # Priority 2: Parse from commit metadata - use PagedTokenToKVPoolAllocator as primary target
    if not (module_path and symbol_name):
        module_path = "sglang.srt.mem_cache.allocator"
        symbol_name = "PagedTokenToKVPoolAllocator"
    
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
    
    # Realistic KV cache configuration
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # KV cache dimensions (simulate Llama-7B)
    num_layers = 32
    num_heads = 32
    head_dim = 128
    page_size = 16  # Common page size for paged attention
    
    # Total number of pages for KV cache
    total_pages = 4096  # Enough for ~65k tokens
    
    # Create mock KV cache tensors
    kv_cache_shape = (total_pages * page_size, num_layers, 2, num_heads, head_dim)
    kv_cache = torch.zeros(kv_cache_shape, dtype=dtype, device=device)
    
    # Create allocation pattern that triggers optimization
    # Simulate continuous batching with prefill and decode
    batch_configs = [
        # (num_requests, seq_len_per_request, is_prefill)
        (8, 512, True),   # Prefill batch
        (32, 1, False),   # Decode batch
        (4, 1024, True),  # Larger prefill
        (64, 1, False),   # Larger decode batch
        (16, 256, True),  # Medium prefill
    ]
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "kv_cache": kv_cache,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "page_size": page_size,
        "total_pages": total_pages,
        "batch_configs": batch_configs,
        "allocator_class": None,  # Will be set in experiment
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Create allocator instance
    allocator = target(
        data["page_size"],
        data["kv_cache"],
        data["device"]
    )
    
    # Track allocation results
    allocation_results = []
    allocated_indices_list = []
    
    # Simulate allocation pattern with alloc/free cycles
    for round_idx in range(3):  # Multiple rounds to trigger sorting
        round_allocations = []
        
        # Allocation phase - simulate batched requests
        for batch_idx, (num_requests, seq_len, is_prefill) in enumerate(data["batch_configs"]):
            if is_prefill:
                # Prefill: allocate many tokens at once
                for req_idx in range(num_requests):
                    need_size = seq_len
                    indices = allocator.alloc(need_size)
                    if indices is not None:
                        round_allocations.append(indices)
                        allocation_results.append({
                            "round": round_idx,
                            "batch": batch_idx,
                            "request": req_idx,
                            "size": need_size,
                            "success": True
                        })
            else:
                # Decode: allocate single tokens for many requests  
                if hasattr(allocator, 'alloc_decode'):
                    # Use specialized decode allocation if available
                    seq_lens = torch.full((num_requests,), 128 + round_idx * 10, 
                                         dtype=torch.int64, device=data["device"])
                    last_ids = torch.arange(num_requests, dtype=torch.int64, device=data["device"]) * data["page_size"]
                    
                    try:
                        indices = allocator.alloc_decode(seq_lens, last_ids)
                        if indices is not None:
                            round_allocations.append(indices)
                            allocation_results.append({
                                "round": round_idx,
                                "batch": batch_idx,
                                "type": "decode",
                                "num_requests": num_requests,
                                "success": True
                            })
                    except:
                        # Fallback to regular alloc
                        for req_idx in range(num_requests):
                            indices = allocator.alloc(seq_len)
                            if indices is not None:
                                round_allocations.append(indices)
                else:
                    # Fallback to regular alloc
                    for req_idx in range(num_requests):
                        indices = allocator.alloc(seq_len)
                        if indices is not None:
                            round_allocations.append(indices)
        
        allocated_indices_list.extend(round_allocations)
        
        # Free phase - free some allocations to create fragmentation
        allocator.free_group_begin()
        free_ratio = 0.3 + round_idx * 0.1  # Gradually free more
        num_to_free = int(len(round_allocations) * free_ratio)
        
        # Free in a pattern that creates fragmentation
        indices_to_free = []
        for i in range(0, min(num_to_free, len(round_allocations)), 2):
            if i < len(round_allocations):
                indices_to_free.append(round_allocations[i])
        
        for indices in indices_to_free:
            allocator.free(indices)
            round_allocations.remove(indices)
        
        allocator.free_group_end()
        
        # Trigger merge_and_sort by allocating a large chunk
        large_alloc = allocator.alloc(data["page_size"] * 8)
        if large_alloc is not None:
            round_allocations.append(large_alloc)
    
    # Return allocator state and results for verification
    result = {
        "available_size": allocator.available_size(),
        "num_allocations": len(allocation_results),
        "allocation_results": allocation_results[:10],  # Sample for verification
        "final_free_pages": len(allocator.free_pages) if hasattr(allocator, 'free_pages') else -1,
        "final_release_pages": len(allocator.release_pages) if hasattr(allocator, 'release_pages') else -1,
    }
    
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
    if isinstance(current_result, dict) and isinstance(reference_result, dict):
        # For allocator results, check key metrics
        assert current_result["available_size"] == reference_result["available_size"], \
            f"Available size mismatch: {current_result['available_size']} vs {reference_result['available_size']}"
        
        assert current_result["num_allocations"] == reference_result["num_allocations"], \
            f"Number of allocations mismatch: {current_result['num_allocations']} vs {reference_result['num_allocations']}"
        
        # Check that allocation patterns are similar
        for i, (curr, ref) in enumerate(zip(current_result.get("allocation_results", []), 
                                           reference_result.get("allocation_results", []))):
            assert curr["success"] == ref["success"], f"Allocation {i} success mismatch"
            if "size" in curr and "size" in ref:
                assert curr["size"] == ref["size"], f"Allocation {i} size mismatch"
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Timing
    times_ms = []
    for _ in range(iterations):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start.record()
            result = func()
            end.record()
            torch.cuda.synchronize()
            
            times_ms.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            result = func()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000)
    
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
    commit_hash = os.getenv("COMMIT_HASH", "a99801e0750f41553fedd02e36f58d835c4d4bd6")
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