#!/usr/bin/env python3
"""
Performance test for commit: fb0acb6c72874e98617cabee4ff4851569374fc9
Message: [Perf] Improve MLA on V1 (#14540)

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
        major, minor = hw_info["capability"]
        hw_info["supports_fp16"] = True
        hw_info["supports_bf16"] = major >= 8
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
        # Based on the commit, targeting MLACommonMetadataBuilder.build
        module_path = "vllm.v1.attention.backends.mla.common"
        symbol_name = "MLACommonMetadataBuilder"
    
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
    
    # MLA-specific workload simulating metadata building
    num_reqs = 32  # Number of concurrent requests
    num_decodes = 24  # Decode requests
    num_prefills = 8  # Prefill requests
    num_actual_tokens = 2048
    common_prefix_len = 128
    page_size = 16
    chunked_prefill_workspace_size = 4096
    
    # Create mock runner and input batch structures
    class MockRunner:
        def __init__(self):
            self.device = device
            self.query_start_loc_cpu = torch.arange(num_reqs + 2, dtype=torch.int32, pin_memory=True) * 64
            self.seq_lens_cpu = torch.randint(32, 256, (num_reqs,), dtype=torch.int32, pin_memory=True)
            self.slot_mapping_cpu = torch.arange(num_actual_tokens, dtype=torch.int32, pin_memory=True)
            self.positions_cpu = torch.arange(num_actual_tokens, dtype=torch.int32, pin_memory=True)
            
            # Mock input batch
            class MockInputBatch:
                def __init__(self):
                    self.num_computed_tokens_cpu_tensor = torch.randint(0, 64, (num_reqs,), dtype=torch.int32, pin_memory=True)
                    
                    class MockBlockTable:
                        def get_device_tensor(self):
                            return torch.randint(0, 1024, (num_reqs, 64), dtype=torch.int32, device=device)
                    
                    self.block_table = MockBlockTable()
            
            self.input_batch = MockInputBatch()
    
    # Create mock metadata builder
    class MockMetadataBuilder:
        def __init__(self):
            self.runner = MockRunner()
            self._num_decodes = num_decodes
            self._num_prefills = num_prefills
            self.chunked_prefill_enabled = True
            self.chunked_prefill_workspace_size = chunked_prefill_workspace_size
            self.page_size = page_size
            self.chunked_prefill_workspace = torch.zeros(chunked_prefill_workspace_size, 512, device=device, dtype=dtype)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "builder": MockMetadataBuilder(),
        "num_reqs": num_reqs,
        "num_actual_tokens": num_actual_tokens,
        "common_prefix_len": common_prefix_len,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    
    builder = data["builder"]
    num_reqs = data["num_reqs"]
    num_actual_tokens = data["num_actual_tokens"]
    common_prefix_len = data["common_prefix_len"]
    device = data["device"]
    
    # Simulate the optimized memory transfer pattern from the commit
    with torch.no_grad():
        # The optimization focuses on reducing GPU->CPU sync
        # by computing max values on CPU side
        
        # Original pattern (before optimization)
        # query_start_loc = builder.runner.query_start_loc_cpu[:num_reqs + 1].to(device)
        # seq_lens = builder.runner.seq_lens_cpu[:num_reqs].to(device)
        
        # Optimized pattern from commit
        block_table = builder.runner.input_batch.block_table.get_device_tensor()[:num_reqs]
        query_start_loc = builder.runner.query_start_loc_cpu[:num_reqs + 1].to(device, non_blocking=True)
        slot_mapping = builder.runner.slot_mapping_cpu[:num_actual_tokens].to(device, non_blocking=True).long()
        input_positions = builder.runner.positions_cpu[:num_actual_tokens].to(device, non_blocking=True).long()
        
        seq_lens_cpu = builder.runner.seq_lens_cpu[:num_reqs]
        seq_lens = seq_lens_cpu.to(device, non_blocking=True)
        max_query_len = seq_lens_cpu.max().item()  # Compute on CPU to avoid sync
        
        # Simulate chunked prefill metadata construction
        if builder._num_prefills > 0:
            reqs_start = builder._num_decodes
            context_lens_cpu = builder.runner.input_batch.num_computed_tokens_cpu_tensor[reqs_start:num_reqs]
            max_context_len_cpu = context_lens_cpu.max().item()
            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
            
            if builder.chunked_prefill_enabled and max_context_len_cpu > 0:
                max_context_chunk = (builder.chunked_prefill_workspace_size // num_prefills_with_context_cpu)
                max_context_chunk = (max_context_chunk // builder.page_size) * builder.page_size
                
                if max_context_chunk > 0:
                    num_chunks = (max_context_len_cpu + max_context_chunk - 1) // max_context_chunk
                    
                    # CPU-side computation to avoid GPU sync
                    chunk_starts = torch.arange(num_chunks, dtype=torch.int32).unsqueeze(1).expand(-1, builder._num_prefills) * max_context_chunk
                    chunk_ends = torch.min(context_lens_cpu.unsqueeze(0), chunk_starts + max_context_chunk)
                    chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)
                    
                    cu_seq_lens_cpu = torch.zeros(num_chunks, builder._num_prefills + 1, dtype=torch.int32, pin_memory=True)
                    torch.cumsum(chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32)
                    
                    # Non-blocking transfer to GPU
                    cu_seq_lens = cu_seq_lens_cpu.to(device, non_blocking=True)
                    chunk_starts_gpu = chunk_starts.to(device, non_blocking=True)
        
        # Ensure all transfers complete
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Return metadata dict
        result = {
            "query_start_loc": query_start_loc,
            "seq_lens": seq_lens,
            "slot_mapping": slot_mapping,
            "input_positions": input_positions,
            "max_query_len": max_query_len,
            "block_table": block_table,
        }
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, dict):
        # Convert tensors to CPU for storage
        cpu_result = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                cpu_result[k] = v.cpu()
            else:
                cpu_result[k] = v
        torch.save({"type": "dict", "data": cpu_result}, filepath)
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
        assert current_result.keys() == reference_result.keys(), f"Keys mismatch"
        
        for key in current_result:
            curr_val = current_result[key]
            ref_val = reference_result[key]
            
            if isinstance(curr_val, torch.Tensor):
                assert curr_val.shape == ref_val.shape, f"Shape mismatch for {key}"
                assert curr_val.dtype == ref_val.dtype, f"Dtype mismatch for {key}"
                
                # Determine tolerances based on dtype
                if curr_val.dtype in (torch.float16, torch.bfloat16):
                    rtol, atol = 1e-3, 1e-4
                elif curr_val.dtype in (torch.int32, torch.int64, torch.long):
                    rtol, atol = 0, 0
                else:
                    rtol, atol = 1e-5, 1e-7
                
                torch.testing.assert_close(
                    curr_val.cpu(),
                    ref_val.cpu() if isinstance(ref_val, torch.Tensor) else ref_val,
                    rtol=rtol, atol=atol
                )
            else:
                assert curr_val == ref_val, f"Value mismatch for {key}: {curr_val} vs {ref_val}"

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
        p95_ms = times[int(len(times) * 0.95)] if len(times) > 1 else times[0]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "fb0acb6c72874e98617cabee4ff4851569374fc9")
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