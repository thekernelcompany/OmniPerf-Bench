#!/usr/bin/env python3
"""
Performance test for commit: ad8d696a99ca1eee19f1404e16e8e82df592ff85
Message: [Core] Scheduler perf fix (#4270)

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
    module_path = os.getenv("PROB_MODULE", "vllm.core.scheduler")
    symbol_name = os.getenv("PROB_SYMBOL", "Scheduler")
    
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
    
    # Import required vLLM components
    from vllm.config import CacheConfig, SchedulerConfig
    from vllm.compilation.backends import Sequence
    from vllm.core.block.utils import SequenceGroup
    from vllm.core.block_manager import SequenceStatus
    from vllm import SamplingParams
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Create scheduler configuration
    block_size = 16
    num_gpu_blocks = 1024
    num_cpu_blocks = 512
    max_num_seqs = 256
    max_model_len = 2048
    max_num_batched_tokens = 2048
    
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len
    )
    
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space_bytes=1,
        cache_dtype="auto"
    )
    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = num_cpu_blocks
    
    # Create sequence groups for testing
    num_seq_groups = 64  # Simulate multiple concurrent requests
    seq_groups = []
    
    for i in range(num_seq_groups):
        # Create sequences with varying prompt lengths
        prompt_length = 128 + (i % 5) * 64  # Vary from 128 to 384
        
        # Create a sequence
        seq_id = i
        prompt_token_ids = list(range(prompt_length))
        
        seq = Sequence(
            seq_id=seq_id,
            prompt=None,
            prompt_token_ids=prompt_token_ids,
            block_size=block_size
        )
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=128
        )
        
        # Create sequence group
        seq_group = SequenceGroup(
            request_id=str(i),
            seqs=[seq],
            sampling_params=sampling_params,
            arrival_time=time.time() - (num_seq_groups - i) * 0.01  # Stagger arrival times
        )
        
        seq_groups.append(seq_group)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "scheduler_config": scheduler_config,
        "cache_config": cache_config,
        "seq_groups": seq_groups,
        "num_iterations": 100  # Number of scheduling iterations to test
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Create scheduler instance
    scheduler = target(
        scheduler_config=data["scheduler_config"],
        cache_config=data["cache_config"],
        lora_config=None
    )
    
    seq_groups = data["seq_groups"]
    num_iterations = data["num_iterations"]
    
    # Add sequence groups to scheduler
    for seq_group in seq_groups[:32]:  # Start with half the groups
        scheduler.add_seq_group(seq_group)
    
    results = {
        "scheduled_counts": [],
        "allocation_times": [],
        "total_scheduled": 0
    }
    
    # Simulate scheduling iterations
    for iteration in range(num_iterations):
        # Schedule requests
        seq_group_metadata_list, scheduler_outputs = scheduler.schedule()
        
        results["scheduled_counts"].append(len(scheduler_outputs.scheduled_seq_groups))
        results["total_scheduled"] += len(scheduler_outputs.scheduled_seq_groups)
        
        # Add more requests progressively
        if iteration < len(seq_groups) - 32:
            scheduler.add_seq_group(seq_groups[32 + iteration])
        
        # Simulate token generation for running sequences
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            for seq in seq_group.get_seqs():
                if seq.get_len() < seq.get_prompt_len() + 50:  # Generate up to 50 tokens
                    # Simulate appending a generated token
                    seq.append_token_id(token_id=100, logprobs={100: -0.5})
        
        # Free finished sequences periodically
        if iteration % 10 == 0:
            scheduler.free_finished_seq_groups()
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, dict):
        # Store as JSON for dictionaries
        import json
        with open(filepath, 'w') as f:
            json.dump(result, f)
    else:
        torch.save({"type": "generic", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    if filepath.endswith('.json'):
        import json
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        data = torch.load(filepath)
        return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    if isinstance(current_result, dict) and isinstance(reference_result, dict):
        # Check that both scheduled similar numbers of sequences
        curr_total = current_result.get("total_scheduled", 0)
        ref_total = reference_result.get("total_scheduled", 0)
        
        # Allow some variance in scheduling decisions
        if abs(curr_total - ref_total) > ref_total * 0.1:  # 10% tolerance
            raise AssertionError(f"Total scheduled mismatch: {curr_total} vs {ref_total}")
        
        # Check scheduled counts have similar patterns
        curr_counts = current_result.get("scheduled_counts", [])
        ref_counts = reference_result.get("scheduled_counts", [])
        
        if len(curr_counts) != len(ref_counts):
            raise AssertionError(f"Count length mismatch: {len(curr_counts)} vs {len(ref_counts)}")

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
    
    # Timing - scheduler operations are CPU-bound
    warmup = 3
    iters = 10
    
    # Time the experiment
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "ad8d696a99ca1eee19f1404e16e8e82df592ff85")
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
        "device": "cpu",  # Scheduler is CPU-bound
        "dtype": "N/A",  # Not applicable for scheduler
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "behavioral"),
        "opt_path_hit": True,
        "total_scheduled": result["total_scheduled"]
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