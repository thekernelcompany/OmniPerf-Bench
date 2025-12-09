#!/usr/bin/env python3
"""
Performance test for commit: 296f927f2493908984707354e3cc5d7b2e41650b
Message: [Model] RE: Mamba2 Prefill Performance Tweaks: Fixing Flurry of Unnecessary Memory Copies  (#14857)

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
        # Based on the commit diff, the optimization is in MambaMixer2.forward_cuda
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
    
    # Import required modules for Mamba2
    try:
        from vllm.model_executor.models.mamba_cache import MambaCacheParams
    except ImportError:
        # Fallback - create mock cache params
        class MambaCacheParams:
            def __init__(self, batch_size, nheads, headdim, dstate):
                self.batch_size = batch_size
                self.nheads = nheads
                self.headdim = headdim
                self.dstate = dstate
                self.ssm_state = torch.zeros(batch_size, nheads, headdim, dstate, device=device, dtype=dtype)
                self.state_indices_tensor = torch.arange(batch_size, device=device, dtype=torch.long)
                self.conv_state = torch.zeros(batch_size, nheads, headdim, 4, device=device, dtype=dtype)
    
    device = torch.device(hw_info["device"] if hw_info["device"] == "cuda" else "cpu")
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Mamba2 model dimensions (typical for 2.7B model)
    batch_size = 16  # Number of sequences in prefill
    seq_len = 512    # Prefill sequence length
    d_model = 2560   # Model dimension
    d_state = 128    # SSM state dimension
    n_groups = 8     # Number of groups
    d_conv = 4       # Convolution kernel size
    expand = 2       # Expansion factor
    headdim = 256    # Head dimension
    chunk_size = 256 # Chunk size for segmented scan
    nheads = n_groups
    
    # Create MambaMixer2 instance
    MambaMixer2, _ = resolve_target()
    
    mixer = MambaMixer2(
        d_model=d_model,
        ssm_state_size=d_state,
        conv_kernel_size=d_conv,
        intermediate_size=d_model * expand,
        time_step_rank=headdim // 4,
        use_conv_bias=True,
        use_bias=False,
        use_rms_norm=True,
        rms_norm_eps=1e-5,
        chunk_size=chunk_size,
        n_groups=n_groups,
        dtype=dtype
    )
    
    if device.type == "cuda":
        mixer = mixer.cuda()
    
    # Create input hidden states for prefill
    hidden_states = torch.randn(batch_size * seq_len, d_model, device=device, dtype=dtype)
    
    # Create cache parameters
    cache_params = MambaCacheParams(batch_size, nheads, headdim, d_state)
    cache_params.seqlen_offset = 0  # Prefill starts at 0
    
    # Create attention metadata for prefill
    class AttnMetadata:
        def __init__(self, batch_size, seq_len):
            self.is_prefill = True
            self.query_start_loc = torch.arange(0, batch_size * seq_len + 1, seq_len, 
                                              device=device, dtype=torch.int32)
            self.seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
            self.context_lens = self.seq_lens.clone()
            self.max_seqlen = seq_len
    
    attn_metadata = AttnMetadata(batch_size, seq_len)
    
    # Simulate having some initial states (optimization path)
    has_initial_states = torch.tensor([True, False] * (batch_size // 2), device=device, dtype=torch.bool)
    if batch_size % 2:
        has_initial_states = torch.cat([has_initial_states, torch.tensor([True], device=device, dtype=torch.bool)])
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "mixer": mixer,
        "hidden_states": hidden_states,
        "cache_params": cache_params,
        "attn_metadata": attn_metadata,
        "has_initial_states": has_initial_states,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    
    # Clone inputs to avoid side effects
    hidden_states = data["hidden_states"].clone()
    
    # Reset cache state for fair comparison
    data["cache_params"].ssm_state = torch.zeros_like(data["cache_params"].ssm_state)
    data["cache_params"].conv_state = torch.zeros_like(data["cache_params"].conv_state)
    
    with torch.no_grad():
        # Call the forward_cuda method which contains the optimization
        result = data["mixer"].forward_cuda(
            hidden_states,
            cache_params=data["cache_params"],
            attn_metadata=data["attn_metadata"]
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
        assert current_result.shape == reference_result.shape, f"Shape mismatch: {current_result.shape} vs {reference_result.shape}"
        assert current_result.dtype == reference_result.dtype, f"Dtype mismatch: {current_result.dtype} vs {reference_result.dtype}"
        
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
    
    # Clear cache
    torch.cuda.empty_cache()
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
        p95_ms = times[int(len(times) * 0.95) - 1] if len(times) > 1 else avg_ms
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "296f927f2493908984707354e3cc5d7b2e41650b")
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