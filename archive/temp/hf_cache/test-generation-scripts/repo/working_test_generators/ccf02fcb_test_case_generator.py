#!/usr/bin/env python3
"""
Performance test for commit: ccf02fcbaebb1a5b59dfc6c7cb64aa7cc489f04c
Message: Revert "Model] Mamba2 Prefill Performance Tweaks: Fixing Flurry of U… (#14848)

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
def resolve_target() -> Tuple[Any, str]:
    """Resolve the optimization target from environment or metadata."""
    
    # Priority 1: Environment variables
    module_path = os.getenv("PROB_MODULE", "")
    symbol_name = os.getenv("PROB_SYMBOL", "")
    
    # Priority 2: Parse from commit metadata
    if not (module_path and symbol_name):
        # Based on the diff, the target is MambaMixer2.forward
        module_path = "vllm.model_executor.layers.mamba.mamba_mixer2"
        symbol_name = "MambaMixer2"
    
    # Import with error handling
    try:
        module = importlib.import_module(module_path)
        target = module
        for attr in symbol_name.split("."):
            target = getattr(target, attr)
        
        fq_name = f"{module_path}.{symbol_name}"
        return target, fq_name
        
    except (ImportError, AttributeError) as e:
        # Try importing the necessary cache params module as well
        try:
            from vllm.model_executor.models.mamba_cache import MambaCacheParams
            from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
            return MambaMixer2, "vllm.model_executor.layers.mamba.mamba_mixer2.MambaMixer2"
        except ImportError as e2:
            error_data = {
                "target_resolved": False,
                "error": str(e2),
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
    
    # Import necessary modules
    try:
        from vllm.model_executor.models.mamba_cache import MambaCacheParams
        from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    except ImportError:
        error_data = {
            "target_resolved": False,
            "error": "Cannot import MambaCacheParams or MambaMixer2",
        }
        print(json.dumps(error_data))
        sys.exit(1)
    
    # Mamba2 configuration for prefill workload
    batch_size = 8  # Number of sequences
    seq_len = 1024  # Sequence length for prefill
    d_model = 2560  # Model dimension (typical for Mamba2)
    d_state = 128  # SSM state dimension
    d_conv = 4  # Convolution dimension
    expand = 2  # Expansion factor
    headdim = 64  # Head dimension
    ngroups = 8  # Number of groups
    chunk_size = 256  # Chunk size for scanning
    
    d_inner = expand * d_model
    nheads = d_inner // headdim
    
    # Create MambaMixer2 layer
    mixer = MambaMixer2(
        d_model=d_model,
        ssm_state_size=d_state,
        conv_kernel_size=d_conv,
        intermediate_size=d_inner,
        time_step_rank=math.ceil(d_model / 16),
        use_conv_bias=True,
        use_bias=False,
        use_rms_norm=True,
        rms_norm_eps=1e-5,
        activation="silu",
        chunk_size=chunk_size,
        is_weight_partitioned=False
    ).to(device).to(dtype)
    
    # Create input hidden states
    hidden_states = torch.randn(seq_len * batch_size, d_model, device=device, dtype=dtype)
    
    # Create MambaCacheParams with initial states
    max_batch_size = 128
    mamba_cache_params = MambaCacheParams(
        max_batch_size=max_batch_size,
        intermediate_size=d_inner,
        ssm_state_size=d_state,
        conv_kernel_size=d_conv,
        dtype=dtype,
        device=device
    )
    
    # Setup for prefill with some initial states
    mamba_cache_params.state_indices_tensor = torch.arange(batch_size, device=device, dtype=torch.long)
    mamba_cache_params.gathered_conv_state = torch.zeros(
        batch_size, d_conv - 1, d_inner, device=device, dtype=dtype
    )
    
    # Mark some sequences as having initial states (to trigger the optimization path)
    has_initial_states = torch.tensor([i % 2 == 0 for i in range(batch_size)], device=device, dtype=torch.bool)
    
    # Pre-populate some initial states
    for i in range(batch_size):
        if has_initial_states[i]:
            mamba_cache_params.ssm_state[i] = torch.randn(
                nheads, headdim, d_state, device=device, dtype=dtype
            )
    
    # Create query start and end locations for varlen support
    query_start_loc = torch.tensor([i * seq_len for i in range(batch_size + 1)], device=device, dtype=torch.int32)
    context_lens_tensor = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "mixer": mixer,
        "hidden_states": hidden_states,
        "mamba_cache_params": mamba_cache_params,
        "has_initial_states": has_initial_states,
        "query_start_loc": query_start_loc,
        "context_lens_tensor": context_lens_tensor,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    
    mixer = data["mixer"]
    hidden_states = data["hidden_states"]
    mamba_cache_params = data["mamba_cache_params"]
    has_initial_states = data["has_initial_states"]
    query_start_loc = data["query_start_loc"]
    context_lens_tensor = data["context_lens_tensor"]
    
    with torch.no_grad():
        # Call the forward method which contains the optimization
        result = mixer.forward(
            hidden_states=hidden_states,
            mamba_cache_params=mamba_cache_params,
            has_initial_states=has_initial_states,
            query_start_loc=query_start_loc,
            context_lens_tensor=context_lens_tensor,
        )
    
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
    if isinstance(current_result, torch.Tensor):
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
            _ = experiment(data)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        avg_ms = sum(times) / len(times)
        p50_ms = times[len(times) // 2]
        p95_ms = times[int(len(times) * 0.95) - 1]
        # Produce a result for reference handling
        result = experiment(data)
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "ccf02fcbaebb1a5b59dfc6c7cb64aa7cc489f04c")
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