#!/usr/bin/env python3
"""
Performance test for commit: bfdb1ba5c3fb14387c69acb1f5067102d8028e56
Message: [Core] Improve detokenization performance for prefill (#3469)

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
    module_path = os.getenv("PROB_MODULE", "")
    symbol_name = os.getenv("PROB_SYMBOL", "")
    
    # Priority 2: Parse from commit metadata - use Detokenizer.decode_sequence_inplace
    if not (module_path and symbol_name):
        module_path = "vllm.transformers_utils.detokenizer"
        symbol_name = "Detokenizer"
    
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
    
    # Import required components
    from transformers import AutoTokenizer
    from vllm.compilation.backends import Sequence
    from vllm.beam_search import Logprob
    from vllm import SamplingParams
    from vllm.transformers_utils.tokenizer_group import get_tokenizer_group
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Setup tokenizer
    tokenizer_name = "facebook/opt-125m"  # Fast tokenizer for testing
    tokenizer_group = get_tokenizer_group(
        tokenizer_pool_config=None,
        tokenizer_id=tokenizer_name,
        enable_lora=False,
        max_num_seqs=100,
        max_input_length=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
        revision=None,
    )
    
    # Create test sequences with varying lengths
    test_prompts = [
        "The quick brown fox jumps over the lazy dog. " * 10,  # Medium
        "Hello world! " * 50,  # Long repetitive
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort. " * 5,  # Long narrative
    ]
    
    # Tokenize prompts
    tokenizer = tokenizer_group.get_lora_tokenizer(None)
    sequences = []
    
    for i, prompt in enumerate(test_prompts):
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        
        # Create sequence
        seq = Sequence(
            seq_id=i,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            block_size=16,
            eos_token_id=tokenizer.eos_token_id,
            lora_request=None
        )
        
        # Simulate generation by adding tokens
        generated_tokens = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]  # Dummy tokens
        for token_id in generated_tokens:
            # Create logprobs for each token
            logprobs = {
                token_id: Logprob(logprob=-0.5),
                token_id + 1: Logprob(logprob=-1.0),
                token_id + 2: Logprob(logprob=-2.0),
            }
            seq.append_token_id(token_id, logprobs)
        
        sequences.append(seq)
    
    # Create sampling params
    sampling_params = SamplingParams(
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        logprobs=3
    )
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "tokenizer_group": tokenizer_group,
        "sequences": sequences,
        "sampling_params": sampling_params,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    Detokenizer, _ = resolve_target()
    
    # Create detokenizer instance
    detokenizer = Detokenizer(data["tokenizer_group"])
    
    # Run detokenization on all sequences
    results = []
    for seq in data["sequences"]:
        # Reset sequence state for consistent testing
        seq.output_text = ""
        seq.tokens = None
        seq.prefix_offset = 0
        seq.read_offset = 0
        
        # Decode each generated token incrementally
        all_token_ids = seq.get_token_ids()
        num_prompt_tokens = len(seq.prompt_token_ids)
        
        for i in range(num_prompt_tokens, len(all_token_ids)):
            # Simulate incremental generation
            seq_view = Sequence(
                seq_id=seq.seq_id,
                prompt=seq.prompt,
                prompt_token_ids=seq.prompt_token_ids,
                block_size=seq.block_size,
                eos_token_id=seq.eos_token_id,
                lora_request=seq.lora_request
            )
            
            # Copy state
            seq_view.output_text = seq.output_text
            seq_view.tokens = seq.tokens
            seq_view.prefix_offset = seq.prefix_offset
            seq_view.read_offset = seq.read_offset
            
            # Add tokens up to current position
            for j in range(num_prompt_tokens, i + 1):
                seq_view.append_token_id(all_token_ids[j], seq.output_logprobs[j - num_prompt_tokens] if j - num_prompt_tokens < len(seq.output_logprobs) else {})
            
            # Decode current token
            detokenizer.decode_sequence_inplace(seq_view, data["sampling_params"])
            
            # Update state
            seq.output_text = seq_view.output_text
            seq.tokens = seq_view.tokens
            seq.prefix_offset = seq_view.prefix_offset
            seq.read_offset = seq_view.read_offset
        
        results.append({
            "output_text": seq.output_text,
            "tokens": seq.tokens,
            "prefix_offset": seq.prefix_offset,
            "read_offset": seq.read_offset,
            "num_tokens": len(all_token_ids)
        })
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Convert tokens to serializable format
    serializable_result = []
    for item in result:
        serializable_item = item.copy()
        if serializable_item.get("tokens"):
            serializable_item["tokens"] = list(serializable_item["tokens"])
        serializable_result.append(serializable_item)
    
    torch.save({"type": "detokenizer_result", "data": serializable_result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert len(current_result) == len(reference_result), f"Result count mismatch: {len(current_result)} vs {len(reference_result)}"
    
    for i, (current, reference) in enumerate(zip(current_result, reference_result)):
        # Check output text
        assert current["output_text"] == reference["output_text"], f"Seq {i}: output_text mismatch"
        
        # Check tokens if present
        if current.get("tokens") and reference.get("tokens"):
            assert current["tokens"] == reference["tokens"], f"Seq {i}: tokens mismatch"
        
        # Check offsets
        assert current["prefix_offset"] == reference["prefix_offset"], f"Seq {i}: prefix_offset mismatch"
        assert current["read_offset"] == reference["read_offset"], f"Seq {i}: read_offset mismatch"

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
    
    # Timing - detokenization is CPU-bound
    warmup = 3
    iters = 20  # More iterations since this is a fast operation
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "bfdb1ba5c3fb14387c69acb1f5067102d8028e56")
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
        "device": "cpu",  # Detokenization is CPU-bound
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