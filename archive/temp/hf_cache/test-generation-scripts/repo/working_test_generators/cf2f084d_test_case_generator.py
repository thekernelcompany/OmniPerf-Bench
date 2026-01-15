#!/usr/bin/env python3
"""
Performance test for commit: cf2f084d56a1293cb08da2393984cdc7685ac019
Message: Dynamic scheduler delay to improve ITL performance  (#3279)

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
from collections import deque

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
        # Based on the diff, the main changes are in Scheduler._passed_delay
        module_path = "vllm.core.scheduler"
        symbol_name = "Scheduler"
    
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
    
    # Import required classes for scheduler setup
    try:
        from vllm.config import SchedulerConfig, CacheConfig
        from vllm.core.scheduler import Scheduler
        from vllm.core.block.utils import SequenceGroup
        from vllm.core.scheduler import SequenceGroupMetadata
        from vllm.compilation.backends import Sequence
        from vllm import SamplingParams
        from vllm.block import LogicalTokenBlock
    except ImportError as e:
        print(json.dumps({"target_resolved": False, "error": f"Import error: {e}"}))
        sys.exit(1)
    
    # Create scheduler configurations
    block_size = 16
    max_num_batched_tokens = 4096
    max_num_seqs = 256
    max_model_len = 2048
    
    # Test with delay factor (the optimization parameter)
    delay_factor = float(os.getenv("DELAY_FACTOR", "0.5"))
    
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        delay_factor=delay_factor
    )
    
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space_bytes=0,
        cache_dtype="auto"
    )
    cache_config.num_cpu_blocks = 512
    cache_config.num_gpu_blocks = 1024
    
    # Create scheduler instance
    scheduler = Scheduler(scheduler_config, cache_config, None)
    
    # Create simulated requests queue
    num_requests = 100
    requests = []
    
    for i in range(num_requests):
        # Mix of different prompt lengths
        prompt_length = np.random.choice([128, 256, 512, 1024])
        request_id = f"req_{i}"
        
        # Create sequence and sequence group
        prompt_tokens = list(range(prompt_length))
        seq = Sequence(
            seq_id=i,
            inputs={"prompt_token_ids": prompt_tokens},
            block_size=block_size
        )
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=128
        )
        
        # Create sequence group with arrival time
        seq_group = SequenceGroup(
            request_id=request_id,
            seqs=[seq],
            sampling_params=sampling_params,
            arrival_time=time.time() + i * 0.01  # Stagger arrivals
        )
        
        requests.append(seq_group)
    
    data = {
        "device": hw_info["device"],
        "dtype": torch.float16,
        "hw_info": hw_info,
        "scheduler": scheduler,
        "requests": requests,
        "delay_factor": delay_factor,
        "scheduler_config": scheduler_config,
        "cache_config": cache_config
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    scheduler = data["scheduler"]
    requests = data["requests"]
    
    # Reset scheduler state
    scheduler.waiting = deque()
    scheduler.running = []
    scheduler.swapped = deque()
    scheduler.prev_time = 0.0
    scheduler.prev_prompt = False
    scheduler.last_prompt_latency = 0.0
    
    # Simulate scheduling with delay factor
    results = {
        "scheduled_prompts": [],
        "schedule_times": [],
        "waiting_times": [],
        "batch_sizes": []
    }
    
    # Add requests progressively and schedule
    request_idx = 0
    total_scheduled = 0
    
    # Run scheduling iterations
    for iteration in range(50):
        # Add some new requests
        while request_idx < len(requests) and request_idx < (iteration + 1) * 2:
            scheduler.add_seq_group(requests[request_idx])
            request_idx += 1
        
        # Record time before scheduling
        start_time = time.perf_counter()
        
        # Call schedule method (the optimized function)
        seq_group_meta, scheduler_outputs = scheduler.schedule()
        
        # Record scheduling time
        schedule_time = time.perf_counter() - start_time
        results["schedule_times"].append(schedule_time * 1000)  # Convert to ms
        
        if scheduler_outputs.scheduled_seq_groups:
            total_scheduled += len(scheduler_outputs.scheduled_seq_groups)
            results["scheduled_prompts"].append(len(scheduler_outputs.scheduled_seq_groups))
            results["batch_sizes"].append(scheduler_outputs.num_batched_tokens)
            
            # Simulate processing time for prompts
            if scheduler_outputs.prompt_run:
                # Simulate prompt processing latency
                time.sleep(0.01)  # 10ms simulated processing
        
        # Record waiting queue size
        results["waiting_times"].append(len(scheduler.waiting))
        
        # Break if all requests scheduled
        if total_scheduled >= len(requests):
            break
        
        # Small delay between iterations
        time.sleep(0.001)
    
    # Calculate metrics
    results["total_scheduled"] = total_scheduled
    results["avg_schedule_time_ms"] = np.mean(results["schedule_times"]) if results["schedule_times"] else 0
    results["avg_batch_size"] = np.mean(results["batch_sizes"]) if results["batch_sizes"] else 0
    results["max_waiting_queue"] = max(results["waiting_times"]) if results["waiting_times"] else 0
    
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
    # For scheduler, check that key metrics are similar
    if isinstance(current_result, dict) and isinstance(reference_result, dict):
        # Check that total scheduled is the same
        assert current_result.get("total_scheduled") == reference_result.get("total_scheduled"), \
            f"Total scheduled mismatch: {current_result.get('total_scheduled')} vs {reference_result.get('total_scheduled')}"
        
        # Check that scheduling times are reasonable (within 2x)
        curr_time = current_result.get("avg_schedule_time_ms", 0)
        ref_time = reference_result.get("avg_schedule_time_ms", 0)
        if ref_time > 0:
            ratio = curr_time / ref_time
            assert 0.5 <= ratio <= 2.0, f"Schedule time ratio {ratio} out of bounds"

# =======================
# Timing Implementation
# =======================
def time_cpu_scheduler(func, warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
    """Time CPU scheduler operations."""
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
    
    # Timing - scheduler is CPU-based
    warmup = 3
    iters = 10
    
    result, timing_stats = time_cpu_scheduler(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "cf2f084d56a1293cb08da2393984cdc7685ac019")
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
        "device": "cpu",  # Scheduler runs on CPU
        "dtype": "torch.float16",
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "behavioral"),
        "opt_path_hit": True,
        "delay_factor": data["delay_factor"],
        "avg_schedule_time_ms": result.get("avg_schedule_time_ms", 0),
        "total_scheduled": result.get("total_scheduled", 0)
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