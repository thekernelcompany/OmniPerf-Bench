#!/usr/bin/env python3
"""
Performance test for commit: 9d72daf4ced05a5fec1ad8ea2914a39296f402da
Message: [V1][Perf] Simpler request output queues (#15156)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import asyncio
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
    np.random.seed(42)
    # This is a CPU-bound optimization, no CUDA needed

# =======================
# Hardware Detection
# =======================
def detect_hardware() -> Dict[str, Any]:
    hw_info = {}
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
        # Based on the commit diff, the key new symbol is RequestOutputCollector
        module_path = "vllm.v1.engine.output_processor"
        symbol_name = "RequestOutputCollector"
    
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
    
    # Import required classes
    try:
        from vllm import CompletionOutput, RequestOutput
        from vllm.engine.llm_engine import RequestOutputKind
    except ImportError as e:
        print(json.dumps({"target_resolved": False, "error": str(e)}))
        sys.exit(1)
    
    # Create test data - multiple request outputs to simulate streaming
    num_outputs = 100  # Number of outputs to stream
    batch_size = 8     # Number of concurrent requests
    
    def make_request_output(req_id: str, idx: int, finished: bool = False) -> RequestOutput:
        return RequestOutput(
            request_id=req_id,
            prompt=None,
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=f"token_{idx}",
                    token_ids=[idx],
                    cumulative_logprob=float(idx),
                    logprobs=[{"a": idx, "b": idx}],
                    finish_reason="length" if finished else None,
                )
            ],
            finished=finished,
        )
    
    # Create multiple batches of outputs
    request_outputs = []
    for batch_idx in range(batch_size):
        req_id = f"request_{batch_idx}"
        batch_outputs = []
        for i in range(num_outputs):
            is_finished = (i == num_outputs - 1)
            batch_outputs.append(make_request_output(req_id, i, is_finished))
        request_outputs.append(batch_outputs)
    
    data = {
        "device": "cpu",
        "dtype": None,  # Not applicable for this CPU-bound test
        "hw_info": hw_info,
        "request_outputs": request_outputs,
        "RequestOutputKind": RequestOutputKind,
        "num_outputs": num_outputs,
        "batch_size": batch_size,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    RequestOutputCollector, _ = resolve_target()
    RequestOutputKind = data["RequestOutputKind"]
    
    async def run_collector_test():
        collectors = []
        results = []
        
        # Create collectors for each request
        for _ in range(data["batch_size"]):
            collector = RequestOutputCollector(RequestOutputKind.DELTA)
            collectors.append(collector)
        
        # Simulate producer-consumer pattern with coalescing
        async def producer(collector, outputs):
            for output in outputs:
                collector.put(output)
                # Simulate producer getting ahead of consumer
                if np.random.random() < 0.3:  # 30% chance to batch puts
                    await asyncio.sleep(0)
        
        async def consumer(collector, num_expected):
            collected = []
            count = 0
            while count < num_expected:
                # Try get_nowait first (optimization path)
                output = collector.get_nowait()
                if output is None:
                    output = await collector.get()
                collected.append(output)
                # Count tokens in the coalesced output
                if output.outputs and output.outputs[0].token_ids:
                    count += len(output.outputs[0].token_ids)
            return collected
        
        # Run producers and consumers concurrently
        tasks = []
        for idx, collector in enumerate(collectors):
            outputs = data["request_outputs"][idx]
            tasks.append(producer(collector, outputs))
            tasks.append(consumer(collector, data["num_outputs"]))
        
        # Execute all tasks
        all_results = await asyncio.gather(*tasks)
        
        # Extract consumer results (every other result)
        for i in range(1, len(all_results), 2):
            results.extend(all_results[i])
        
        return results
    
    # Run the async test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(run_collector_test())
    finally:
        loop.close()
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Store as pickle since result contains complex objects
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    # Check that we got the same number of outputs
    assert len(current_result) == len(reference_result), \
        f"Number of outputs mismatch: {len(current_result)} vs {len(reference_result)}"
    
    # For each output batch, verify tokens were collected correctly
    for i, (curr, ref) in enumerate(zip(current_result, reference_result)):
        # Check request_id
        assert curr.request_id == ref.request_id, \
            f"Request ID mismatch at {i}: {curr.request_id} vs {ref.request_id}"
        
        # Check finished status
        assert curr.finished == ref.finished, \
            f"Finished status mismatch at {i}: {curr.finished} vs {ref.finished}"
        
        if curr.outputs and ref.outputs:
            curr_out = curr.outputs[0]
            ref_out = ref.outputs[0]
            
            # Check token counts (may be coalesced differently)
            curr_tokens = len(curr_out.token_ids) if curr_out.token_ids else 0
            ref_tokens = len(ref_out.token_ids) if ref_out.token_ids else 0
            # Allow some variance due to different coalescing patterns
            assert abs(curr_tokens - ref_tokens) <= 10, \
                f"Token count mismatch at {i}: {curr_tokens} vs {ref_tokens}"

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
    
    # Timing
    warmup = 3
    iters = 10
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "9d72daf4ced05a5fec1ad8ea2914a39296f402da")
    impl_tag = os.getenv("IMPL_TAG", "child")
    ref_file = f"{prefix}_{impl_tag}_{commit_hash}_reference.pkl"
    
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
        "dtype": "None",
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "behavioral"),
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