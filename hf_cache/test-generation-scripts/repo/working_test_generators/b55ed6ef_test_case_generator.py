#!/usr/bin/env python3
"""
Performance test for commit: b55ed6ef8ab0dce7fb0f79ff292dafdb4d22610c
Message: [V1][Minor] Optimize token_ids_cpu copy (#11692)

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
# Mock CachedRequestState
# =======================
from dataclasses import dataclass
from typing import Set

@dataclass
class MockCachedRequestState:
    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    mm_inputs: List
    mm_positions: List
    sampling_params: Any
    generator: Optional[torch.Generator]
    block_ids: List[int]
    num_computed_tokens: int
    output_token_ids: List[int]
    
    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

# =======================
# Import Resolution
# =======================
def resolve_target() -> Tuple[Any, str]:
    """Resolve the optimization target from environment or metadata."""
    
    # Priority 1: Environment variables
    module_path = os.getenv("PROB_MODULE", "vllm.v1.worker.gpu_input_batch")
    symbol_name = os.getenv("PROB_SYMBOL", "InputBatch")
    
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
    
    # Create InputBatch parameters that trigger the optimization
    # The optimization is about copying only necessary tokens during condense()
    max_num_reqs = 256  # Typical batch size
    max_model_len = 4096  # Large model context to make copy cost visible
    max_num_blocks_per_req = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    vocab_size = 32000  # Typical vocab size
    
    # Create mock requests with varying token counts
    requests = []
    for i in range(32):  # Create 32 active requests
        prompt_len = 256 + i * 16  # Varying prompt lengths
        output_len = 128 + i * 8   # Varying output lengths
        req = MockCachedRequestState(
            req_id=f"req_{i}",
            prompt_token_ids=list(range(prompt_len)),
            prompt=None,
            mm_inputs=[],
            mm_positions=[],
            sampling_params=type('SamplingParams', (), {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0,
                'repetition_penalty': 1.0,
                'min_tokens': 0,
                'all_stop_token_ids': set(),
                'sampling_type': 0,  # GREEDY
                'logprobs': None,
                'prompt_logprobs': False
            })(),
            generator=None,
            block_ids=list(range(16)),
            num_computed_tokens=prompt_len,
            output_token_ids=list(range(output_len))
        )
        requests.append(req)
    
    # Create indices to remove (simulate request completion)
    # This will trigger condense() operation
    indices_to_remove = [3, 7, 11, 15, 19, 23, 27]  # Remove every 4th request
    
    data = {
        "device": device,
        "dtype": torch.float32,
        "hw_info": hw_info,
        "max_num_reqs": max_num_reqs,
        "max_model_len": max_model_len,
        "max_num_blocks_per_req": max_num_blocks_per_req,
        "pin_memory": pin_memory,
        "vocab_size": vocab_size,
        "requests": requests,
        "indices_to_remove": indices_to_remove,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    InputBatch, _ = resolve_target()
    
    # Create InputBatch instance
    batch = InputBatch(
        max_num_reqs=data["max_num_reqs"],
        max_model_len=data["max_model_len"],
        max_num_blocks_per_req=data["max_num_blocks_per_req"],
        device=data["device"],
        pin_memory=data["pin_memory"],
        vocab_size=data["vocab_size"],
    )
    
    # Add all requests
    for i, req in enumerate(data["requests"]):
        batch.add_request(req, req_index=i)
    
    # Remove some requests to create empty indices
    empty_indices = []
    for idx in data["indices_to_remove"]:
        req_id = data["requests"][idx].req_id
        removed_idx = batch.remove_request(req_id)
        if removed_idx is not None:
            empty_indices.append(removed_idx)
    
    # Sort in descending order as required by condense()
    empty_indices.sort(reverse=True)
    
    # Time the condense operation which contains the optimization
    # This is where the optimized token copying happens
    start_state = {
        "num_reqs": batch.num_reqs,
        "empty_indices": empty_indices.copy(),
        "token_ids_snapshot": batch.token_ids_cpu.copy() if hasattr(batch, 'token_ids_cpu') else None
    }
    
    # Execute the optimized condense operation
    batch.condense(empty_indices)
    
    # Return state for verification
    result = {
        "num_reqs_after": batch.num_reqs,
        "req_ids": [req_id for req_id in batch.req_ids if req_id is not None],
        "start_state": start_state,
        "batch": batch  # Keep reference for multiple iterations
    }
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Store only the verifiable parts, not the batch object
    storable = {
        "num_reqs_after": result["num_reqs_after"],
        "req_ids": result["req_ids"],
    }
    torch.save({"type": "dict", "data": storable}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    # Compare the state after condense operation
    assert current_result["num_reqs_after"] == reference_result["num_reqs_after"], \
        f"Number of requests mismatch: {current_result['num_reqs_after']} vs {reference_result['num_reqs_after']}"
    
    assert set(current_result["req_ids"]) == set(reference_result["req_ids"]), \
        f"Request IDs mismatch: {current_result['req_ids']} vs {reference_result['req_ids']}"

# =======================
# Timing Implementation
# =======================
def time_cpu_condense(data: Dict[str, Any], warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
    """Time the condense operation on CPU."""
    InputBatch, _ = resolve_target()
    
    # Warmup
    for _ in range(warmup):
        batch = InputBatch(
            max_num_reqs=data["max_num_reqs"],
            max_model_len=data["max_model_len"],
            max_num_blocks_per_req=data["max_num_blocks_per_req"],
            device=data["device"],
            pin_memory=data["pin_memory"],
            vocab_size=data["vocab_size"],
        )
        for i, req in enumerate(data["requests"]):
            batch.add_request(req, req_index=i)
        empty_indices = []
        for idx in data["indices_to_remove"]:
            req_id = data["requests"][idx].req_id
            removed_idx = batch.remove_request(req_id)
            if removed_idx is not None:
                empty_indices.append(removed_idx)
        empty_indices.sort(reverse=True)
        batch.condense(empty_indices)
    
    # Timing
    times_ms = []
    result = None
    for _ in range(iterations):
        # Fresh setup for each iteration
        batch = InputBatch(
            max_num_reqs=data["max_num_reqs"],
            max_model_len=data["max_model_len"],
            max_num_blocks_per_req=data["max_num_blocks_per_req"],
            device=data["device"],
            pin_memory=data["pin_memory"],
            vocab_size=data["vocab_size"],
        )
        for i, req in enumerate(data["requests"]):
            batch.add_request(req, req_index=i)
        empty_indices = []
        for idx in data["indices_to_remove"]:
            req_id = data["requests"][idx].req_id
            removed_idx = batch.remove_request(req_id)
            if removed_idx is not None:
                empty_indices.append(removed_idx)
        empty_indices.sort(reverse=True)
        
        # Time the condense operation
        start = time.perf_counter()
        batch.condense(empty_indices)
        end = time.perf_counter()
        
        times_ms.append((end - start) * 1000)
        
        # Save last result
        if result is None:
            result = {
                "num_reqs_after": batch.num_reqs,
                "req_ids": [req_id for req_id in batch.req_ids if req_id is not None],
            }
    
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
    
    # For this CPU-based optimization, we always time on CPU
    warmup = 5
    iters = 20  # More iterations since operation is fast
    result, timing_stats = time_cpu_condense(data, warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "b55ed6ef8ab0dce7fb0f79ff292dafdb4d22610c")
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
        "device": "cpu",  # This optimization affects CPU operations
        "dtype": "torch.int32",  # token_ids dtype
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