#!/usr/bin/env python3
"""
Performance test for commit: 660470e5a36b8e52083615ad7c85e9b4fd4c72ce
Message: [Core] Optimize evictor-v2 performance (#7193)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import importlib
from typing import Dict, Any, Tuple, Optional, List
from collections import OrderedDict

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
            from vllm import SamplingParams
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
            from vllm import SamplingParams
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
        # Based on the commit diff, the optimization is in LRUEvictor
        module_path = "vllm.core.evictor_v2"
        symbol_name = "LRUEvictor"
    
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
    
    # LRU Evictor workload - simulate cache management scenario
    # The optimization improves evict() by early-breaking and update() by using move_to_end
    
    # Create test data for LRU cache eviction
    num_blocks = 10000  # Large number of blocks to stress the evictor
    num_operations = 50000  # Mix of adds, updates, and evictions
    
    # Generate block metadata
    blocks = []
    for i in range(num_blocks):
        block_id = i
        content_hash = hash(f"content_{i}") & 0x7FFFFFFF
        num_hashed_tokens = np.random.randint(1, 128)
        last_accessed = float(i) / 1000.0  # Monotonic timestamps initially
        blocks.append({
            "block_id": block_id,
            "content_hash": content_hash,
            "num_hashed_tokens": num_hashed_tokens,
            "last_accessed": last_accessed
        })
    
    # Generate operation sequence (realistic cache access pattern)
    operations = []
    current_time = float(num_blocks) / 1000.0
    
    # Initial population
    for block in blocks[:1000]:
        operations.append({
            "type": "add",
            "block_id": block["block_id"],
            "content_hash": block["content_hash"],
            "num_hashed_tokens": block["num_hashed_tokens"],
            "last_accessed": block["last_accessed"]
        })
    
    # Mix of operations
    np.random.seed(42)
    for _ in range(num_operations):
        op_type = np.random.choice(["update", "evict", "add", "remove"], p=[0.6, 0.2, 0.15, 0.05])
        
        if op_type == "update":
            # Update a random existing block
            block_id = np.random.randint(0, min(1000, len(blocks)))
            current_time += 0.001
            operations.append({
                "type": "update",
                "block_id": block_id,
                "last_accessed": current_time
            })
        elif op_type == "evict":
            operations.append({"type": "evict"})
        elif op_type == "add":
            # Add a new block if we have any left
            if len(operations) < len(blocks):
                idx = len([op for op in operations if op["type"] == "add"])
                if idx < len(blocks):
                    block = blocks[idx]
                    current_time += 0.001
                    operations.append({
                        "type": "add",
                        "block_id": block["block_id"],
                        "content_hash": block["content_hash"],
                        "num_hashed_tokens": block["num_hashed_tokens"],
                        "last_accessed": current_time
                    })
        elif op_type == "remove":
            # Remove a random block
            block_id = np.random.randint(0, min(1000, len(blocks)))
            operations.append({
                "type": "remove",
                "block_id": block_id
            })
    
    data = {
        "device": hw_info["device"],
        "dtype": torch.float32,
        "hw_info": hw_info,
        "blocks": blocks,
        "operations": operations
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    LRUEvictor, fq_name = resolve_target()
    
    # Create evictor instance
    evictor = LRUEvictor()
    
    # Track results for equivalence checking
    results = {
        "evicted_blocks": [],
        "final_state": {},
        "operation_count": 0
    }
    
    # Execute operations
    for op in data["operations"]:
        try:
            if op["type"] == "add":
                if op["block_id"] not in evictor.free_table:
                    evictor.add(
                        op["block_id"],
                        op["content_hash"],
                        op["num_hashed_tokens"],
                        op["last_accessed"]
                    )
            elif op["type"] == "update":
                if op["block_id"] in evictor.free_table:
                    evictor.update(op["block_id"], op["last_accessed"])
            elif op["type"] == "evict":
                if len(evictor.free_table) > 0:
                    evicted_id, evicted_hash = evictor.evict()
                    results["evicted_blocks"].append({
                        "block_id": evicted_id,
                        "content_hash": evicted_hash
                    })
            elif op["type"] == "remove":
                if op["block_id"] in evictor.free_table:
                    evictor.remove(op["block_id"])
            
            results["operation_count"] += 1
            
        except (ValueError, KeyError):
            # Handle expected errors gracefully
            pass
    
    # Capture final state
    results["final_state"] = {
        "num_blocks": evictor.num_blocks,
        "block_ids": list(evictor.free_table.keys())[:100]  # Sample for verification
    }
    
    return results

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
        # Check evicted blocks match
        assert len(current_result["evicted_blocks"]) == len(reference_result["evicted_blocks"]), \
            f"Evicted block count mismatch: {len(current_result['evicted_blocks'])} vs {len(reference_result['evicted_blocks'])}"
        
        for i, (curr, ref) in enumerate(zip(current_result["evicted_blocks"], reference_result["evicted_blocks"])):
            assert curr["block_id"] == ref["block_id"], \
                f"Evicted block ID mismatch at index {i}: {curr['block_id']} vs {ref['block_id']}"
            assert curr["content_hash"] == ref["content_hash"], \
                f"Evicted block hash mismatch at index {i}: {curr['content_hash']} vs {ref['content_hash']}"
        
        # Check final state
        assert current_result["final_state"]["num_blocks"] == reference_result["final_state"]["num_blocks"], \
            f"Final block count mismatch: {current_result['final_state']['num_blocks']} vs {reference_result['final_state']['num_blocks']}"
        
        assert current_result["final_state"]["block_ids"] == reference_result["final_state"]["block_ids"], \
            "Final block IDs mismatch"

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
    
    # This is a CPU-only operation (no GPU kernels involved)
    warmup = 3
    iters = 10
    
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "660470e5a36b8e52083615ad7c85e9b4fd4c72ce")
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
        "device": "cpu",  # Evictor is CPU-only
        "dtype": "none",  # No tensor operations
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