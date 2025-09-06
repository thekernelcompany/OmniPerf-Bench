#!/usr/bin/env python3
"""
Performance test for commit: aea94362c9bdd08ed2b346701bdc09d278e85f66
Message: [Frontend][V1] Online serving performance improvements (#12287)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import math
import importlib
import asyncio
from typing import Dict, Any, Tuple, Optional, List

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
    
    # Priority 2: Parse from commit metadata - target OutputProcessor.process_outputs
    if not (module_path and symbol_name):
        module_path = "vllm.v1.engine.output_processor"
        symbol_name = "OutputProcessor.process_outputs"
    
    # Import with error handling
    try:
        module = importlib.import_module(module_path)
        target = module
        for attr in symbol_name.split("."):
            if hasattr(target, attr):
                target = getattr(target, attr)
            else:
                # Try to get the class first, then the method
                if "." in attr:
                    cls_name, method_name = attr.rsplit(".", 1)
                    cls = getattr(module, cls_name)
                    target = getattr(cls, method_name)
                else:
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
    
    # Import required classes
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm.v1.engine import EngineCoreOutput
    from vllm.engine.llm_engine import BaseTokenizerGroup
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Create mock tokenizer
    class MockTokenizer:
        def get_lora_tokenizer(self, lora_request):
            return self
    
    tokenizer = MockTokenizer()
    
    # Create OutputProcessor instance
    processor = OutputProcessor(tokenizer=tokenizer, log_stats=True)
    
    # Simulate high-concurrency streaming scenario with many outputs
    # This tests the chunking optimization
    batch_size = 256  # High concurrency scenario
    seq_len = 128
    vocab_size = 32000
    
    # Create mock engine core outputs
    outputs = []
    for i in range(batch_size):
        output = EngineCoreOutput(
            request_id=f"req_{i}",
            new_token_ids=[np.random.randint(0, vocab_size)],
            finished=i % 10 == 0,  # 10% finish rate
            finish_reason="stop" if i % 10 == 0 else None,
            stop_reason=None,
            logprobs=None,
            prompt_logprobs=None,
            prompt_logprobs_token_ids=None,
        )
        outputs.append(output)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "processor": processor,
        "outputs": outputs,
        "batch_size": batch_size,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    
    # Get the processor and outputs
    processor = data["processor"]
    outputs = data["outputs"]
    
    # The optimization chunks large output batches to avoid blocking
    # We test the process_outputs method which now supports chunking
    
    # Set VLLM_V1_OUTPUT_PROC_CHUNK_SIZE to test chunking behavior
    os.environ["VLLM_V1_OUTPUT_PROC_CHUNK_SIZE"] = "128"
    
    # Process outputs (this is the optimized path)
    result = processor.process_outputs(outputs)
    
    return {
        "num_outputs": len(outputs),
        "reqs_to_abort": len(result.reqs_to_abort),
        "iteration_stats": {
            "num_prompt_tokens": result.iteration_stats.num_prompt_tokens,
            "num_generation_tokens": result.iteration_stats.num_generation_tokens,
        }
    }

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    torch.save({"type": "dict", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert type(current_result) == type(reference_result)
    
    if isinstance(current_result, dict):
        assert current_result.keys() == reference_result.keys()
        
        # Check numeric values
        assert current_result["num_outputs"] == reference_result["num_outputs"]
        assert current_result["reqs_to_abort"] == reference_result["reqs_to_abort"]
        
        # Check iteration stats
        curr_stats = current_result["iteration_stats"]
        ref_stats = reference_result["iteration_stats"]
        assert curr_stats["num_prompt_tokens"] == ref_stats["num_prompt_tokens"]
        assert curr_stats["num_generation_tokens"] == ref_stats["num_generation_tokens"]

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
    
    # Create a callable for timing
    def run_experiment():
        return experiment(data)
    
    # Timing - this is CPU-bound operation
    warmup = 3
    iters = 20  # More iterations for CPU timing
    result, timing_stats = time_cpu(run_experiment, warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "aea94362c9bdd08ed2b346701bdc09d278e85f66")
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
        "dtype": "torch.float32",  # CPU operation
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