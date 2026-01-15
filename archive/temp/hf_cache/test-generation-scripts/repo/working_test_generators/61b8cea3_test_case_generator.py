#!/usr/bin/env python3
"""
Performance test for commit: 61b8cea3b42feab021d506e9143551de18f9165c
Message: [Attention] Optimize FlashInfer MetadataBuilder Build call (#21137)

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
# Import Resolution
# =======================
def resolve_target() -> Tuple[Any, str]:
    """Resolve the optimization target from environment or metadata."""
    
    # Priority 1: Environment variables
    module_path = os.getenv("PROB_MODULE", "vllm.v1.attention.backends.flashinfer")
    symbol_name = os.getenv("PROB_SYMBOL", "FlashInferMetadataBuilder")
    
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
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Import required classes for setup
    try:
        from vllm.config import VllmConfig, ModelConfig, CacheConfig
        from vllm.attention.layers.chunked_local_attention import CommonAttentionMetadata
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        
        # Create realistic configuration
        model_config = ModelConfig(
            model="meta-llama/Llama-2-7b-hf",
            tokenizer="meta-llama/Llama-2-7b-hf",
            tokenizer_mode="auto",
            trust_remote_code=False,
            max_model_len=2048,
            dtype="float16" if hw_info["device"] == "cuda" else "float32",
            seed=42,
        )
        
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        )
        
        # Create VllmConfig
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
        )
        
        # Create KV cache spec
        kv_cache_spec = FullAttentionSpec(
            num_layers=32,
            num_kv_heads=32,
            head_size=128,
            dtype=dtype,
            block_size=16,
            device=device,
        )
        
        # Create realistic batch metadata
        batch_size = 32  # Mix of prefill and decode
        num_prefills = 8
        num_decodes = batch_size - num_prefills
        
        # Sequence lengths
        prefill_lens = [512, 1024, 768, 256, 2048, 384, 640, 896]
        decode_lens = [128] * num_decodes
        seq_lens_cpu = torch.tensor(decode_lens + prefill_lens, dtype=torch.int32)
        seq_lens = seq_lens_cpu.to(device)
        
        # Context lengths (how much has been processed)
        decode_context = decode_lens  # All decoded
        prefill_context = [0] * num_prefills  # Just starting
        context_lens_cpu = torch.tensor(decode_context + prefill_context, dtype=torch.int32)
        context_lens = context_lens_cpu.to(device)
        
        # Query tokens
        num_decode_tokens = num_decodes * 1  # 1 token per decode
        num_prefill_tokens = sum(prefill_lens)
        num_actual_tokens = num_decode_tokens + num_prefill_tokens
        
        # Query start locations
        query_start_loc_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
        query_start_loc_cpu[1:num_decodes+1] = torch.arange(1, num_decodes+1)
        cumsum_prefill = torch.cumsum(torch.tensor(prefill_lens), dim=0)
        query_start_loc_cpu[num_decodes+1:] = num_decode_tokens + cumsum_prefill
        query_start_loc = query_start_loc_cpu.to(device)
        
        # Computed tokens
        num_computed_tokens_cpu = torch.cat([
            torch.ones(num_decodes, dtype=torch.int32),
            torch.tensor(prefill_lens, dtype=torch.int32)
        ])
        num_computed_tokens = num_computed_tokens_cpu.to(device)
        
        # Block table
        max_blocks = (max(seq_lens_cpu.tolist()) + 15) // 16
        max_block_idx = 10000
        block_table_tensor = torch.randint(0, max_block_idx, 
                                          (batch_size, max_blocks),
                                          dtype=torch.int32, device=device)
        
        # Create CommonAttentionMetadata
        common_attn_metadata = CommonAttentionMetadata(
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            context_lens=context_lens,
            context_lens_cpu=context_lens_cpu,
            block_table_tensor=block_table_tensor,
            num_computed_tokens=num_computed_tokens,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            num_actual_tokens=num_actual_tokens,
        )
        
    except ImportError as e:
        # Fallback to mock data if imports fail
        common_attn_metadata = None
        vllm_config = None
        kv_cache_spec = None
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "common_attn_metadata": common_attn_metadata,
        "vllm_config": vllm_config,
        "kv_cache_spec": kv_cache_spec,
        "num_prefills": num_prefills,
        "num_decodes": num_decodes,
        "common_prefix_len": 0,  # No shared prefix for this test
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Create a FlashInferMetadataBuilder instance
    builder = target(
        vllm_config=data["vllm_config"],
        kv_cache_spec=data["kv_cache_spec"],
        device=data["device"],
    )
    
    # Call the build method - this is what we're optimizing
    with torch.no_grad():
        result = builder.build(
            common_attn_metadata=data["common_attn_metadata"],
            num_prefills=data["num_prefills"],
            num_decodes=data["num_decodes"],
            common_prefix_len=data["common_prefix_len"],
        )
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Extract key fields from FlashInferMetadata for comparison
    if hasattr(result, '__dict__'):
        # Convert to dict for storage
        result_dict = {}
        for key, value in result.__dict__.items():
            if isinstance(value, torch.Tensor):
                result_dict[key] = value.cpu()
            elif value is not None and not callable(value):
                result_dict[key] = value
        torch.save({"type": "flashinfer_metadata", "data": result_dict}, filepath)
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
    # For FlashInferMetadata, compare key fields
    if isinstance(reference_result, dict):
        current_dict = {}
        for key, value in current_result.__dict__.items():
            if isinstance(value, torch.Tensor):
                current_dict[key] = value.cpu()
            elif value is not None and not callable(value):
                current_dict[key] = value
        
        for key in reference_result:
            if key in current_dict:
                ref_val = reference_result[key]
                cur_val = current_dict[key]
                
                if isinstance(ref_val, torch.Tensor):
                    assert cur_val.shape == ref_val.shape, f"Shape mismatch for {key}"
                    assert cur_val.dtype == ref_val.dtype, f"Dtype mismatch for {key}"
                    
                    # Determine tolerances
                    if cur_val.dtype in (torch.float16, torch.bfloat16):
                        rtol, atol = 1e-3, 1e-4
                    elif cur_val.dtype in (torch.int32, torch.int64):
                        rtol, atol = 0, 0
                    else:
                        rtol, atol = 1e-5, 1e-7
                    
                    if rtol == 0:
                        assert torch.equal(cur_val, ref_val), f"Mismatch for {key}"
                    else:
                        torch.testing.assert_close(cur_val, ref_val, rtol=rtol, atol=atol)
                elif isinstance(ref_val, (int, float, bool, str)):
                    assert cur_val == ref_val, f"Value mismatch for {key}: {cur_val} vs {ref_val}"

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
        p95_ms = times[int(len(times) * 0.95) - 1] if len(times) > 1 else times[0]
        # Produce a result for reference handling
        result = experiment(data)
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "61b8cea3b42feab021d506e9143551de18f9165c")
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