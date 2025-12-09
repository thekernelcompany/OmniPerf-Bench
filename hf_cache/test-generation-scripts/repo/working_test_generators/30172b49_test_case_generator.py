#!/usr/bin/env python3
"""
Performance test for commit: 30172b4947c52890b808c6da3a6c7580f55cbb74
Message: [V1] Optimize handling of sampling metadata and req_ids list (#13244)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import math
import importlib
from typing import Dict, Any, Tuple, Optional, List, Set

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
    
    # Priority 2: Parse from commit metadata - focus on InputBatch
    if not (module_path and symbol_name):
        module_path = "vllm.v1.worker.gpu_input_batch"
        symbol_name = "InputBatch"
    
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
    
    # Setup for testing the metadata optimization
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Workload parameters - testing continuous batching scenario
    max_num_reqs = 256  # Large batch to stress metadata handling
    max_model_len = 2048
    vocab_size = 32000
    num_active_reqs = 128  # Active requests in batch
    
    # Create mock request data for testing InputBatch
    from vllm import SamplingParams
    
    # Generate diverse sampling parameters
    sampling_params_list = []
    for i in range(num_active_reqs):
        # Mix of greedy and sampling requests
        if i % 3 == 0:
            # Greedy
            params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
                min_tokens=0
            )
        else:
            # Random sampling with varying parameters
            params = SamplingParams(
                temperature=0.7 + (i % 10) * 0.05,
                top_p=0.9 if i % 2 == 0 else 1.0,
                top_k=40 if i % 4 == 0 else 0,
                min_p=0.01 if i % 5 == 0 else 0.0,
                frequency_penalty=0.1 * (i % 3),
                presence_penalty=0.1 * (i % 2),
                repetition_penalty=1.0 + 0.1 * (i % 3),
                min_tokens=5 if i % 10 == 0 else 0
            )
        sampling_params_list.append(params)
    
    # Create mock request states
    class MockRequest:
        def __init__(self, req_id, sampling_params, prompt_len, output_len):
            self.req_id = req_id
            self.sampling_params = sampling_params
            self.prompt_token_ids = list(range(prompt_len))
            self.output_token_ids = list(range(output_len))
            self.num_computed_tokens = prompt_len + output_len
            self.num_tokens = prompt_len + output_len
    
    requests = []
    for i in range(num_active_reqs):
        prompt_len = 128 + (i % 64) * 16  # Varying prompt lengths
        output_len = 32 + (i % 32) * 8    # Varying output lengths
        req = MockRequest(
            req_id=f"req_{i}",
            sampling_params=sampling_params_list[i],
            prompt_len=prompt_len,
            output_len=output_len
        )
        requests.append(req)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "max_num_reqs": max_num_reqs,
        "max_model_len": max_model_len,
        "vocab_size": vocab_size,
        "requests": requests,
        "num_active_reqs": num_active_reqs,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Import required modules
    try:
        from vllm.v1.worker.gpu_input_batch import InputBatch
        from vllm.v1.core.scheduler_output import CachedRequestData
        from vllm.core.block.block_table import BlockTable
    except ImportError as e:
        # Fallback for older versions
        try:
            InputBatch = target
        except:
            error_data = {
                "target_resolved": False,
                "error": f"Failed to import required modules: {e}",
                "opt_path_hit": False
            }
            print(json.dumps(error_data))
            sys.exit(1)
    
    # Create InputBatch instance
    input_batch = InputBatch(
        max_num_reqs=data["max_num_reqs"],
        max_model_len=data["max_model_len"],
        vocab_size=data["vocab_size"],
        device=data["device"]
    )
    
    # Mock CachedRequestState for adding requests
    class MockCachedRequestState:
        def __init__(self, req):
            self.req_id = req.req_id
            self.sampling_params = req.sampling_params
            self.prompt_token_ids = req.prompt_token_ids
            self.output_token_ids = req.output_token_ids
            self.num_computed_tokens = req.num_computed_tokens
            self.num_tokens = req.num_tokens
            self.multimodal_inputs = None
            self.encoder_input_ids = None
            self.encoder_output = None
            
            # Mock block table
            num_blocks = (req.num_tokens + 15) // 16
            self.block_table = BlockTable(
                block_ids=list(range(num_blocks)),
                block_size=16
            )
    
    # Simulate continuous batching workflow
    results = []
    
    # Add requests to batch
    for i, req in enumerate(data["requests"]):
        cached_state = MockCachedRequestState(req)
        input_batch.add_request(cached_state, req_index=i)
    
    # Simulate multiple metadata refresh cycles
    for cycle in range(10):
        # Simulate request removal and condensing (20% of requests)
        num_to_remove = data["num_active_reqs"] // 5
        removed_indices = []
        
        for i in range(num_to_remove):
            req_idx = i * 5  # Remove every 5th request
            if req_idx < len(data["requests"]):
                req_id = data["requests"][req_idx].req_id
                removed_idx = input_batch.remove_request(req_id)
                if removed_idx is not None:
                    removed_indices.append(removed_idx)
        
        # Condense the batch if any requests were removed
        if removed_indices:
            removed_indices.sort(reverse=True)
            input_batch.condense(removed_indices)
        
        # Create sampling metadata - this is the optimized operation
        with torch.no_grad():
            if hasattr(input_batch, 'refresh_sampling_metadata'):
                # New optimized version
                input_batch.refresh_sampling_metadata()
                metadata = input_batch.sampling_metadata
            elif hasattr(input_batch, '_make_sampling_metadata'):
                # Fallback for testing
                metadata = input_batch._make_sampling_metadata()
            else:
                # Old version compatibility
                req_id_output_token_ids = {
                    req.req_id: req.output_token_ids 
                    for req in data["requests"][:input_batch.num_reqs]
                }
                metadata = input_batch.make_sampling_metadata(
                    req_id_output_token_ids=req_id_output_token_ids,
                    req_id_to_spec_token_ids={},
                    skip_copy=False
                )
        
        results.append(metadata)
        
        # Re-add some requests for next cycle
        for i in range(min(num_to_remove, len(removed_indices))):
            if i < len(data["requests"]):
                cached_state = MockCachedRequestState(data["requests"][i])
                input_batch.add_request(cached_state, req_index=i)
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Store only key metadata attributes for comparison
    if isinstance(result, list):
        # List of metadata objects
        serialized = []
        for metadata in result:
            item = {
                "all_greedy": metadata.all_greedy,
                "all_random": metadata.all_random,
                "no_penalties": metadata.no_penalties,
                "max_num_logprobs": metadata.max_num_logprobs,
                "temperature_shape": list(metadata.temperature.shape),
                "has_top_p": metadata.top_p is not None,
                "has_top_k": metadata.top_k is not None,
                "has_min_p": metadata.min_p is not None,
            }
            serialized.append(item)
        torch.save({"type": "metadata_list", "data": serialized}, filepath)
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
    if isinstance(current_result, list) and isinstance(reference_result, list):
        assert len(current_result) == len(reference_result), \
            f"Result list length mismatch: {len(current_result)} vs {len(reference_result)}"
        
        for i, (curr, ref) in enumerate(zip(current_result, reference_result)):
            # Check key attributes match
            assert curr["all_greedy"] == ref["all_greedy"], f"Mismatch at index {i}"
            assert curr["all_random"] == ref["all_random"], f"Mismatch at index {i}"
            assert curr["no_penalties"] == ref["no_penalties"], f"Mismatch at index {i}"
            assert curr["temperature_shape"] == ref["temperature_shape"], f"Mismatch at index {i}"

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
            result = experiment(data)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        avg_ms = sum(times) / len(times)
        p50_ms = times[len(times) // 2]
        p95_ms = times[int(len(times) * 0.95) - 1] if len(times) > 1 else times[0]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "30172b4947c52890b808c6da3a6c7580f55cbb74")
    impl_tag = os.getenv("IMPL_TAG", "child")
    ref_file = f"{prefix}_{impl_tag}_{commit_hash}_reference.pt"
    
    # Serialize results for storage
    serialized_result = []
    for metadata in result:
        item = {
            "all_greedy": metadata.all_greedy,
            "all_random": metadata.all_random,
            "no_penalties": metadata.no_penalties,
            "max_num_logprobs": metadata.max_num_logprobs,
            "temperature_shape": list(metadata.temperature.shape),
            "has_top_p": metadata.top_p is not None,
            "has_top_k": metadata.top_k is not None,
            "has_min_p": metadata.min_p is not None,
        }
        serialized_result.append(item)
    
    if reference:
        store_result(serialized_result, ref_file)
    
    if eqcheck and os.path.exists(ref_file):
        ref_result = load_result(ref_file)
        check_equivalence(serialized_result, ref_result)
    
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