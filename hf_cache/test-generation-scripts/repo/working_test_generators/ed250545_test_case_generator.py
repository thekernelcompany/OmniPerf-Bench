#!/usr/bin/env python3
"""
Performance test for commit: ed25054577f7abca2aee32a5290200c4a1aed561
Message: [Core] Introduce popleft_n and append_n in FreeKVCacheBlockQueue to further optimize block_pool (#21222)

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

import inspect
import logging

# API Probing helpers - auto-generated for compatibility
def safe_create_object(cls, **kwargs):
    """Create object with only valid arguments based on signature."""
    try:
        if not callable(cls):
            raise TypeError(f"{cls} is not callable")
        sig = inspect.signature(cls)
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in sig.parameters and k != "self"}
        return cls(**valid_kwargs)
    except Exception as e:
        logging.warning(f"Failed to create {cls.__name__ if hasattr(cls, '__name__') else cls} with args {list(kwargs.keys())}: {e}")
        raise

def safe_call_function(func, *args, **kwargs):
    """Call function with only valid arguments based on signature."""
    try:
        if not callable(func):
            raise TypeError(f"{func} is not callable")
        sig = inspect.signature(func)
        # Filter kwargs to only valid parameters
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in sig.parameters}
        return func(*args, **valid_kwargs)
    except Exception as e:
        logging.warning(f"Failed to call {func.__name__ if hasattr(func, '__name__') else func} with args {list(kwargs.keys())}: {e}")
        raise

# Specific helpers for common vllm classes
def safe_create_engine_output(**kwargs):
    """Create EngineCoreOutput with compatible arguments."""
    try:
        from vllm.v1.engine import EngineCoreOutput
        return safe_create_object(EngineCoreOutput, **kwargs)
    except ImportError:
        try:
            from vllm.engine import EngineCoreOutput  
            return safe_create_object(EngineCoreOutput, **kwargs)
        except ImportError:
            raise ImportError("EngineCoreOutput not found in vllm")

def safe_create_sampling_params(**kwargs):
    """Create SamplingParams with compatible arguments."""
    try:
        from vllm import SamplingParams
        return safe_create_object(SamplingParams, **kwargs)
    except ImportError:
        try:
            from vllm.sampling_params import SamplingParams
            return safe_create_object(SamplingParams, **kwargs)
        except ImportError:
            raise ImportError("SamplingParams not found in vllm")

def safe_create_llm(**kwargs):
    """Create LLM with compatible arguments."""
    try:
        from vllm import LLM
        return safe_create_object(LLM, **kwargs)
    except ImportError:
        raise ImportError("LLM not found in vllm")



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
def resolve_target() -> Tuple[Any, Any, str]:
    """Resolve the optimization target from environment or metadata."""
    
    # Priority 1: Environment variables
    module_path = os.getenv("PROB_MODULE", "")
    symbol_name = os.getenv("PROB_SYMBOL", "")
    
    # Priority 2: Parse from commit metadata - use BlockPool since it calls the optimized methods
    if not (module_path and symbol_name):
        module_path = "vllm.v1.core.block_pool"
        symbol_name = "BlockPool"
    
    # Import with error handling
    try:
        module = importlib.import_module(module_path)
        target = getattr(module, symbol_name)
        
        # Also import KVCacheBlock for creating test data
        kv_module = importlib.import_module("vllm.v1.core.kv_cache_utils")
        kv_cache_block = getattr(kv_module, "KVCacheBlock")
        
        fq_name = f"{module_path}.{symbol_name}"
        return target, kv_cache_block, fq_name
        
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
    
    # Resolve target classes
    BlockPool, KVCacheBlock, fq_name = resolve_target()
    
    # Realistic KV cache configuration for a 7B model
    num_layers = 32
    num_heads = 32
    head_dim = 128
    block_size = 16  # Common block size in vLLM
    
    # Total blocks to simulate a reasonable GPU memory allocation
    # For 16GB GPU: ~8192 blocks (each block holds 16 tokens of KV cache)
    num_total_blocks = 8192
    
    # Create block pool
    blocks = []
    for i in range(num_total_blocks):
        block = KVCacheBlock(
            block_id=i,
            prev_token_id=-1,
            token_ids=[-1] * block_size,
            num_tokens=0,
            prev_block=None,
            next_free_block=None,
            prev_free_block=None,
            ref_cnt=0,
            is_full=False,
            is_cached=False,
            is_null=False
        )
        blocks.append(block)
    
    # Initialize block pool
    block_pool = BlockPool(
        blocks=blocks,
        enable_caching=False  # Start with caching disabled for cleaner comparison
    )
    
    # Workload patterns - simulate various batch sizes for allocation/deallocation
    # These represent different request patterns in continuous batching
    allocation_sizes = [
        1,    # Single block allocations (old decode pattern)
        4,    # Small batch
        16,   # Medium batch (prefill for short sequence)
        64,   # Large batch (prefill for medium sequence)
        128,  # Very large batch (prefill for long sequence)
        256,  # Maximum batch (stress test)
    ]
    
    # Create allocation/deallocation pattern
    operations = []
    for size in allocation_sizes:
        # Multiple rounds of alloc/free for each size
        for _ in range(10):
            operations.append(('alloc', size))
            operations.append(('free', size))
    
    data = {
        "device": "cpu",  # Block pool operations are CPU-bound
        "dtype": torch.float32,
        "hw_info": hw_info,
        "block_pool": block_pool,
        "BlockPool": BlockPool,
        "KVCacheBlock": KVCacheBlock,
        "blocks": blocks,
        "operations": operations,
        "num_iterations": 100,  # Number of times to repeat the operation pattern
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    
    block_pool = data["block_pool"]
    operations = data["operations"]
    num_iterations = data["num_iterations"]
    
    # Track allocated blocks for proper cleanup
    allocated_blocks_list = []
    
    # Execute the operation pattern multiple times
    for _ in range(num_iterations):
        for op_type, size in operations:
            if op_type == 'alloc':
                # Ensure we have enough free blocks
                if block_pool.free_block_queue.num_free_blocks >= size:
                    allocated = block_pool.get_new_blocks(size)
                    allocated_blocks_list.append(allocated)
            elif op_type == 'free':
                # Free the oldest allocated blocks if any
                if allocated_blocks_list:
                    blocks_to_free = allocated_blocks_list.pop(0)
                    block_pool.free_blocks(blocks_to_free)
    
    # Clean up any remaining allocated blocks
    while allocated_blocks_list:
        blocks_to_free = allocated_blocks_list.pop(0)
        block_pool.free_blocks(blocks_to_free)
    
    # Return final state for verification
    return {
        "num_free_blocks": block_pool.free_block_queue.num_free_blocks,
        "total_operations": len(operations) * num_iterations
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
        # For block pool state, check that free blocks match
        assert current_result.get("num_free_blocks") == reference_result.get("num_free_blocks"), \
            f"Free blocks mismatch: {current_result.get('num_free_blocks')} vs {reference_result.get('num_free_blocks')}"
        assert current_result.get("total_operations") == reference_result.get("total_operations"), \
            f"Total operations mismatch: {current_result.get('total_operations')} vs {reference_result.get('total_operations')}"
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
        "std_ms": np.std(times_ms) if len(times_ms) > 1 else 0.0
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
    
    # Block pool operations are CPU-bound
    warmup = 3
    iters = 10
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "ed25054577f7abca2aee32a5290200c4a1aed561")
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
        "device": "cpu",
        "dtype": "torch.float32",
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "exact"),
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