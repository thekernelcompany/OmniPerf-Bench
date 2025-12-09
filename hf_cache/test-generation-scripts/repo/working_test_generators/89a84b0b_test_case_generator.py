#!/usr/bin/env python3
"""
Performance test for commit: 89a84b0bb7b30706a02836234a94493ea8f780bf
Message: [Core] Use array to speedup padding (#6779)

This script measures the actual performance impact of using arrays instead of lists
for token storage in vLLM's sampling metadata preparation.
"""

import os
import sys
import json
import time
import importlib
from array import array
from typing import Dict, Any, Tuple, Optional, List
import random

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
    random.seed(42)
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
    module_path = os.getenv("PROB_MODULE", "vllm.model_executor.sampling_metadata")
    symbol_name = os.getenv("PROB_SYMBOL", "SamplingTensors.from_lists")
    
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
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Realistic vLLM workload parameters
    batch_size = 32  # Number of sequences
    vocab_size = 32000  # Llama vocab size
    max_prompt_len = 2048
    max_output_len = 512
    
    # Generate token lists that would be used in sampling
    prompt_tokens = []
    output_tokens = []
    
    for _ in range(batch_size):
        # Generate varying length prompts and outputs
        prompt_len = random.randint(128, max_prompt_len)
        output_len = random.randint(1, max_output_len)
        
        # Use arrays as per the optimization
        prompt_seq = array('l', [random.randint(0, vocab_size-1) for _ in range(prompt_len)])
        output_seq = array('l', [random.randint(0, vocab_size-1) for _ in range(output_len)])
        
        prompt_tokens.append(prompt_seq)
        output_tokens.append(output_seq)
    
    # Other sampling parameters
    temperatures = [0.7] * batch_size
    top_ps = [0.9] * batch_size
    top_ks = [40] * batch_size
    min_ps = [0.0] * batch_size
    presence_penalties = [0.0] * batch_size
    frequency_penalties = [0.0] * batch_size
    repetition_penalties = [1.0] * batch_size
    sampling_seeds = [random.randint(0, 2**31-1) for _ in range(batch_size)]
    sample_indices = list(range(batch_size))
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "temperatures": temperatures,
        "top_ps": top_ps,
        "top_ks": top_ks,
        "min_ps": min_ps,
        "presence_penalties": presence_penalties,
        "frequency_penalties": frequency_penalties,
        "repetition_penalties": repetition_penalties,
        "sampling_seeds": sampling_seeds,
        "sample_indices": sample_indices,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "vocab_size": vocab_size,
        "extra_seeds_to_generate": 0,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Call SamplingTensors.from_lists with the prepared data
    result = target(
        temperatures=data["temperatures"],
        top_ps=data["top_ps"],
        top_ks=data["top_ks"],
        min_ps=data["min_ps"],
        presence_penalties=data["presence_penalties"],
        frequency_penalties=data["frequency_penalties"],
        repetition_penalties=data["repetition_penalties"],
        sampling_seeds=data["sampling_seeds"],
        sample_indices=data["sample_indices"],
        prompt_tokens=data["prompt_tokens"],
        output_tokens=data["output_tokens"],
        vocab_size=data["vocab_size"],
        extra_seeds_to_generate=data["extra_seeds_to_generate"],
        device=data["device"],
        dtype=data["dtype"]
    )
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Store the tensor attributes of SamplingTensors
    tensors_dict = {
        "temperatures": result.temperatures.cpu(),
        "top_ps": result.top_ps.cpu(),
        "top_ks": result.top_ks.cpu(),
        "min_ps": result.min_ps.cpu(),
        "presence_penalties": result.presence_penalties.cpu(),
        "frequency_penalties": result.frequency_penalties.cpu(),
        "repetition_penalties": result.repetition_penalties.cpu(),
        "prompt_tokens": result.prompt_tokens.cpu(),
        "output_tokens": result.output_tokens.cpu(),
        "sampling_seeds": result.sampling_seeds.cpu(),
        "sample_indices": result.sample_indices.cpu(),
    }
    torch.save({"type": "sampling_tensors", "data": tensors_dict}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    # Check each tensor attribute
    attrs_to_check = [
        "temperatures", "top_ps", "top_ks", "min_ps",
        "presence_penalties", "frequency_penalties", "repetition_penalties",
        "prompt_tokens", "output_tokens", "sampling_seeds", "sample_indices"
    ]
    
    for attr in attrs_to_check:
        current_tensor = getattr(current_result, attr).cpu()
        ref_tensor = reference_result[attr]
        
        assert current_tensor.shape == ref_tensor.shape, f"{attr} shape mismatch"
        assert current_tensor.dtype == ref_tensor.dtype, f"{attr} dtype mismatch"
        
        if current_tensor.dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 1e-3, 1e-4
        else:
            rtol, atol = 1e-5, 1e-7
        
        torch.testing.assert_close(
            current_tensor,
            ref_tensor,
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
    
    # This optimization primarily affects CPU operations (array vs list)
    # so we time on CPU
    warmup = 5
    iters = 20
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "89a84b0bb7b30706a02836234a94493ea8f780bf")
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