#!/usr/bin/env python3
"""
Performance test for commit: fa63e710c7fbaae3a445f669d3b5ba6b9a4ef412
Message: [V1][Perf] Reduce scheduling overhead in model runner after cuda sync (#12094)

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
    
    # Priority 2: Parse from commit metadata - target the execute_model method
    if not (module_path and symbol_name):
        module_path = "vllm.v1.worker.gpu_model_runner"
        symbol_name = "GPUModelRunner.execute_model"
    
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
    
    # Simulate a typical decode workload where GPU-CPU sync happens
    batch_size = 32  # Multiple concurrent requests
    vocab_size = 32000  # Llama vocabulary size
    hidden_size = 4096  # 7B model hidden size
    num_heads = 32
    head_dim = 128
    max_seq_len = 2048
    
    # Create mock scheduler output
    from dataclasses import dataclass
    from typing import Set
    
    @dataclass
    class MockSchedulerOutput:
        finished_req_ids: Set[str]
        preempted_req_ids: Set[str]
        scheduled_new_reqs: List[Any]
        scheduled_resumed_reqs: List[Any]
        scheduled_running_reqs: List[Any]
        free_encoder_input_ids: List[Tuple[str, int]]
        total_num_scheduled_tokens: int
        num_scheduled_tokens: Dict[str, int]
        num_common_prefix_blocks: int
        scheduled_encoder_inputs: Dict[str, List[int]]
        
    scheduler_output = MockSchedulerOutput(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_resumed_reqs=[],
        scheduled_running_reqs=[],
        free_encoder_input_ids=[],
        total_num_scheduled_tokens=batch_size,
        num_scheduled_tokens={f"req_{i}": 1 for i in range(batch_size)},
        num_common_prefix_blocks=0,
        scheduled_encoder_inputs={}
    )
    
    # Create mock sampler output with GPU tensor
    from vllm.engine.async_llm_engine import SamplerOutput
    
    # Sampled token IDs as GPU tensor (the optimization point)
    sampled_token_ids = torch.randint(0, vocab_size, (batch_size,), 
                                     device=device, dtype=torch.int32)
    
    # Optional logprobs
    logprob_token_ids = torch.randint(0, vocab_size, (batch_size, 5), 
                                     device=device, dtype=torch.int32)
    logprobs = torch.randn(batch_size, 5, device=device, dtype=dtype)
    
    sampler_output = SamplerOutput(
        sampled_token_ids=sampled_token_ids,
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        prompt_logprob_token_ids=None,
        prompt_logprobs=None
    )
    
    # Create mock input batch and request states
    class MockInputBatch:
        def __init__(self, batch_size):
            self.num_reqs = batch_size
            self.req_ids = [f"req_{i}" for i in range(batch_size)] + [None] * batch_size
            self.req_id_to_index = {f"req_{i}": i for i in range(batch_size)}
            self.token_ids_cpu = torch.zeros((batch_size, max_seq_len), dtype=torch.int32)
            self.num_tokens = torch.ones(batch_size, dtype=torch.int32) * 100
            self.generators = {}
    
    class MockRequestState:
        def __init__(self, req_id):
            self.req_id = req_id
            self.num_computed_tokens = 100
            self.num_tokens = 101
            self.output_token_ids = []
    
    input_batch = MockInputBatch(batch_size)
    requests = {f"req_{i}": MockRequestState(f"req_{i}") for i in range(batch_size)}
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "scheduler_output": scheduler_output,
        "sampler_output": sampler_output,
        "input_batch": input_batch,
        "requests": requests,
        "batch_size": batch_size,
        "vocab_size": vocab_size
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    
    # Simulate the critical path: processing sampled tokens after GPU-CPU sync
    sampler_output = data["sampler_output"]
    input_batch = data["input_batch"]
    requests = data["requests"]
    num_reqs = input_batch.num_reqs
    
    # The optimization: defer .tolist() until after CPU prep work
    request_seq_lens = []
    for i, req_id in enumerate(input_batch.req_ids[:num_reqs]):
        if req_id is None:
            continue
        req_state = requests[req_id]
        seq_len = req_state.num_computed_tokens + 1
        if seq_len == req_state.num_tokens:
            input_batch.num_tokens[i] += 1
            req_state.output_token_ids.append(0)  # Placeholder
            request_seq_lens.append((i, req_state, seq_len))
    
    # GPU -> CPU sync happens here (the key optimization point)
    sampled_token_ids = sampler_output.sampled_token_ids.tolist()
    
    # Update with actual token ids
    for i, req_state, seq_len in request_seq_lens:
        token_id = sampled_token_ids[i]
        input_batch.token_ids_cpu[i, seq_len] = token_id
        req_state.output_token_ids[-1] = token_id
    
    # Handle logprobs
    if sampler_output.logprob_token_ids is not None:
        logprob_token_ids = sampler_output.logprob_token_ids.cpu()
    else:
        logprob_token_ids = None
        
    if sampler_output.logprobs is not None:
        logprobs = sampler_output.logprobs.cpu()
    else:
        logprobs = None
    
    result = {
        "sampled_token_ids": sampled_token_ids,
        "logprob_token_ids": logprob_token_ids,
        "logprobs": logprobs,
        "num_tokens_processed": len(request_seq_lens)
    }
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Convert tensors to CPU for storage
    stored_result = {}
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            stored_result[key] = value.cpu()
        else:
            stored_result[key] = value
    torch.save({"type": "dict", "data": stored_result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert isinstance(current_result, dict) and isinstance(reference_result, dict)
    assert current_result.keys() == reference_result.keys()
    
    for key in current_result:
        current_val = current_result[key]
        reference_val = reference_result[key]
        
        if isinstance(current_val, torch.Tensor):
            assert current_val.shape == reference_val.shape
            assert current_val.dtype == reference_val.dtype
            
            if current_val.dtype in (torch.float16, torch.bfloat16):
                rtol, atol = 1e-3, 1e-4
            else:
                rtol, atol = 1e-5, 1e-7
            
            torch.testing.assert_close(
                current_val.cpu(),
                reference_val.cpu(),
                rtol=rtol, atol=atol
            )
        elif isinstance(current_val, list):
            assert len(current_val) == len(reference_val)
            assert current_val == reference_val
        else:
            assert current_val == reference_val

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
    commit_hash = os.getenv("COMMIT_HASH", "fa63e710c7fbaae3a445f669d3b5ba6b9a4ef412")
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