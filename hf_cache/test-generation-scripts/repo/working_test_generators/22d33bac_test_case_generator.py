#!/usr/bin/env python3
"""
Performance test for commit: 22d33baca2c0c639cfd45c48e99803e56c3efa74
Message: [FrontEnd][Perf] `merge_async_iterators` fast-path for single-prompt requests (#15150)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import asyncio
import importlib
from typing import Dict, Any, Tuple, Optional, List, AsyncIterator

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
def resolve_target() -> Tuple[Any, str]:
    """Resolve the optimization target from environment or metadata."""
    
    # Priority 1: Environment variables
    module_path = os.getenv("PROB_MODULE", "")
    symbol_name = os.getenv("PROB_SYMBOL", "")
    
    # Priority 2: Parse from commit metadata
    if not (module_path and symbol_name):
        # Based on the commit diff and apis field
        module_path = "vllm.utils"
        symbol_name = "merge_async_iterators"
    
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
# Async Iterator Creators
# =======================
async def create_single_async_iterator(num_items: int, delay: float = 0.0) -> AsyncIterator[Dict[str, Any]]:
    """Create a single async iterator simulating LLM token streaming."""
    for i in range(num_items):
        if delay > 0:
            await asyncio.sleep(delay)
        yield {
            "token_id": i,
            "text": f"token_{i}",
            "logprob": -0.5 * (i + 1),
            "finish_reason": "length" if i == num_items - 1 else None
        }

async def create_multiple_async_iterators(num_iterators: int, items_per_iterator: int) -> List[AsyncIterator[Dict[str, Any]]]:
    """Create multiple async iterators for testing the merge functionality."""
    iterators = []
    for j in range(num_iterators):
        async def make_iter(iter_id=j):
            for i in range(items_per_iterator):
                yield {
                    "iterator_id": iter_id,
                    "token_id": i,
                    "text": f"iter{iter_id}_token_{i}",
                    "logprob": -0.5 * (i + 1),
                    "finish_reason": "length" if i == items_per_iterator - 1 else None
                }
        iterators.append(make_iter())
    return iterators

# =======================
# Workload Setup
# =======================
def setup() -> Dict[str, Any]:
    """Create realistic workload for the optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    # Workload parameters for async iterator merging
    # Single iterator case (fast path)
    single_iterator_items = 100  # Simulating 100 token generation
    
    # Multiple iterator case (original path)
    num_multiple_iterators = 4  # Simulating 4 parallel requests
    items_per_iterator = 25  # Each generating 25 tokens
    
    data = {
        "device": "cpu",  # This is a CPU-bound operation
        "dtype": torch.float32,  # Not relevant for this test but included for consistency
        "hw_info": hw_info,
        "single_iterator_items": single_iterator_items,
        "num_multiple_iterators": num_multiple_iterators,
        "items_per_iterator": items_per_iterator,
        "test_single": True,  # Test the fast-path by default
    }
    
    return data

# =======================
# Experiment Execution
# =======================
async def async_experiment_runner(data: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    """Execute the optimized async operation."""
    target, fq_name = resolve_target()
    
    # Determine which path to test based on configuration
    if data.get("test_single", True):
        # Test the fast-path (single iterator)
        iterators = [create_single_async_iterator(data["single_iterator_items"])]
    else:
        # Test the original path (multiple iterators)
        iterators = await create_multiple_async_iterators(
            data["num_multiple_iterators"],
            data["items_per_iterator"]
        )
    
    # Collect results from merge_async_iterators
    results = []
    async for index, item in target(*iterators):
        results.append((index, item))
    
    return results

def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation (sync wrapper for async code)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(async_experiment_runner(data))
    finally:
        loop.close()
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Convert to JSON-serializable format
    serializable_result = []
    for index, item in result:
        serializable_result.append({
            "index": index,
            "item": item
        })
    
    with open(filepath, 'w') as f:
        json.dump(serializable_result, f)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    with open(filepath, 'r') as f:
        serializable_result = json.load(f)
    
    # Convert back to original format
    result = []
    for entry in serializable_result:
        result.append((entry["index"], entry["item"]))
    return result

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert len(current_result) == len(reference_result), f"Length mismatch: {len(current_result)} vs {len(reference_result)}"
    
    for i, ((curr_idx, curr_item), (ref_idx, ref_item)) in enumerate(zip(current_result, reference_result)):
        assert curr_idx == ref_idx, f"Index mismatch at position {i}: {curr_idx} vs {ref_idx}"
        assert curr_item == ref_item, f"Item mismatch at position {i}: {curr_item} vs {ref_item}"

# =======================
# Timing Implementation
# =======================
async def async_time_cpu(func, data, warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
    """Time async CPU operations."""
    # Warmup
    for _ in range(warmup):
        _ = await func(data)
    
    # Timing
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = await func(data)
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

def time_cpu(func, data, warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
    """Time sync CPU operations (wrapper for async timing)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result, stats = loop.run_until_complete(
            async_time_cpu(async_experiment_runner, data, warmup, iterations)
        )
    finally:
        loop.close()
    return result, stats

# =======================
# Main Test Function
# =======================
def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """Main test entry point."""
    
    # Setup
    data = setup()
    hw_info = data["hw_info"]
    
    # Test the fast-path (single iterator) by default
    data["test_single"] = True
    
    # Timing
    warmup = 3
    iters = 20  # More iterations since this is a fast operation
    
    result, timing_stats = time_cpu(None, data, warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "22d33baca2c0c639cfd45c48e99803e56c3efa74")
    impl_tag = os.getenv("IMPL_TAG", "child")
    ref_file = f"{prefix}_{impl_tag}_{commit_hash}_reference.json"
    
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
        "dtype": str(data["dtype"]),
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