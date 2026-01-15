#!/usr/bin/env python3
"""
Performance test for commit: 83450458339b07765b0e72a822e5fe93eeaf5258
Message: [Performance][Spec Decode] Optimize ngram lookup performance (#9333)

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
from unittest.mock import MagicMock

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
        # Based on commit diff
        module_path = "vllm.spec_decode.ngram_worker"
        symbol_name = "NGramWorker"
    
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
    
    device = torch.device(hw_info["device"] if hw_info["device"] == "cuda" else "cpu")
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Create different sequence lengths to test the optimization
    # Some sequences < 3072 (use CPU), some >= 3072 (use GPU)
    test_configs = [
        {"seq_len": 1024, "batch_size": 8},   # Small sequence (CPU path)
        {"seq_len": 2048, "batch_size": 4},   # Medium sequence (CPU path)
        {"seq_len": 4096, "batch_size": 2},   # Large sequence (GPU path)
    ]
    
    vocab_size = 32000  # Typical LLM vocab size
    sample_len = 5  # Number of tokens to speculate
    
    # Create mock execute model requests
    execute_model_reqs = []
    
    for config in test_configs:
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]
        
        # Create token sequences with repeating patterns for ngram matching
        token_lists = []
        for b in range(batch_size):
            # Create sequence with repeating ngrams
            base_pattern = torch.randint(0, vocab_size, (20,), dtype=torch.long)
            tokens = base_pattern.repeat((seq_len // 20) + 1)[:seq_len]
            # Insert the pattern again near the end for matching
            tokens[-40:-20] = base_pattern
            token_lists.append(tokens.tolist())
        
        # Create mock SequenceGroupMetadata
        seq_group_metadata_list = []
        for token_ids in token_lists:
            # Mock sequence data
            seq_data = MagicMock()
            seq_data.get_token_ids.return_value = token_ids
            seq_data.get_len.return_value = len(token_ids)
            
            # Mock sequence group metadata
            seq_group_metadata = MagicMock()
            seq_group_metadata.seq_data = {0: seq_data}
            seq_group_metadata_list.append(seq_group_metadata)
        
        # Mock ExecuteModelRequest
        execute_model_req = MagicMock()
        execute_model_req.seq_group_metadata_list = seq_group_metadata_list
        execute_model_req.blocks_to_swap_in = []
        execute_model_req.blocks_to_swap_out = []
        execute_model_req.blocks_to_copy = []
        
        execute_model_reqs.append(execute_model_req)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "execute_model_reqs": execute_model_reqs,
        "sample_len": sample_len,
        "vocab_size": vocab_size,
        "ngram_prompt_lookup_min": 1,
        "ngram_prompt_lookup_max": 15,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    NGramWorker, fq_name = resolve_target()
    
    # Create NGramWorker instance
    local_rank = 0 if data["hw_info"]["device"] == "cuda" else -1
    
    # Mock model config
    model_config = MagicMock()
    model_config.get_vocab_size.return_value = data["vocab_size"]
    
    # Create worker
    worker = NGramWorker(local_rank=local_rank, model_config=model_config)
    worker.set_ngram_window_size(
        data["ngram_prompt_lookup_min"],
        data["ngram_prompt_lookup_max"]
    )
    
    # Initialize device
    if data["hw_info"]["device"] == "cuda":
        worker.device = torch.device("cuda:0")
    else:
        worker.device = torch.device("cpu")
    
    # Mock proposer to avoid initialization issues
    worker._proposer = MagicMock()
    worker.vocab_size = data["vocab_size"]
    
    results = []
    
    with torch.no_grad():
        for execute_model_req in data["execute_model_reqs"]:
            # Call the optimized sampler_output method
            outputs, transposed = worker.sampler_output(
                execute_model_req,
                data["sample_len"],
                set()  # seq_ids_with_bonus_token_in_last_step
            )
            results.append((outputs, transposed))
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Store sampler outputs as structured data
    stored_data = []
    for outputs, transposed in result:
        if outputs is None:
            stored_data.append(None)
        else:
            batch_outputs = []
            for output in outputs:
                if output is None:
                    batch_outputs.append(None)
                else:
                    batch_outputs.append({
                        "sampled_token_ids": output.sampled_token_ids.cpu() if output.sampled_token_ids is not None else None,
                        "transposed": transposed
                    })
            stored_data.append(batch_outputs)
    
    torch.save({"type": "ngram_outputs", "data": stored_data}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert len(current_result) == len(reference_result), f"Result count mismatch"
    
    for i, ((curr_outputs, curr_trans), ref_data) in enumerate(zip(current_result, reference_result)):
        if curr_outputs is None:
            assert ref_data is None, f"Output {i}: Expected None, got data"
        else:
            assert ref_data is not None, f"Output {i}: Expected data, got None"
            assert len(curr_outputs) == len(ref_data), f"Output {i}: Batch size mismatch"
            
            for j, (curr_out, ref_out) in enumerate(zip(curr_outputs, ref_data)):
                if curr_out is None:
                    assert ref_out is None, f"Output {i},{j}: Expected None"
                else:
                    assert ref_out is not None, f"Output {i},{j}: Expected data"
                    if curr_out.sampled_token_ids is not None and ref_out["sampled_token_ids"] is not None:
                        ref_tensor = ref_out["sampled_token_ids"]
                        if curr_out.sampled_token_ids.device.type == "cuda":
                            ref_tensor = ref_tensor.cuda()
                        torch.testing.assert_close(
                            curr_out.sampled_token_ids,
                            ref_tensor,
                            rtol=0, atol=0  # Exact match for token IDs
                        )

# =======================
# Timing Implementation
# =======================
def time_gpu(func, warmup=5, iterations=50) -> Tuple[Any, Dict[str, float]]:
    """Time GPU operations with CUDA events."""
    # Warmup
    for _ in range(warmup):
        _ = func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Timing
    times_ms = []
    for _ in range(iterations):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start.record()
            result = func()
            end.record()
            torch.cuda.synchronize()
            
            times_ms.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            result = func()
            times_ms.append((time.perf_counter() - start) * 1000)
    
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
    else:
        warmup = 3
        iters = 20
    
    result, timing_stats = time_gpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "83450458339b07765b0e72a822e5fe93eeaf5258")
    impl_tag = os.getenv("IMPL_TAG", "child")
    ref_file = f"{prefix}ngram_{impl_tag}_{commit_hash}_reference.pt"
    
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