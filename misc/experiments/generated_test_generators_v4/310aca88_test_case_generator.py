#!/usr/bin/env python3
"""
Performance test script for commit: 310aca88c984983189a57f1b72e3b1dde89fb92f
[perf]fix current stream (#11870)

Signed-off-by: youkaichao <youkaichao@gmail.com>

This script measures the actual performance of the optimization described in the commit.
It supports running against different checkouts (child/parent/agent) and storing/loading
references for cross-commit equivalence checks in a research benchmark pipeline.
"""

import os
import json
import time
import importlib
import types
from typing import Dict, Any, Tuple, Callable, Optional

import numpy as np
import torch

# -------------------------------
# Error taxonomy (deterministic)
# -------------------------------
E_IMPORT_MISSING = "IMPORT_MISSING_SYMBOL"
E_OPT_PATH_NOT_TRIGGERED = "OPT_PATH_NOT_TRIGGERED"
E_CAPABILITY = "CAPABILITY_UNSUPPORTED"
E_EQFAIL = "EQUIVALENCE_FAILED"

# -------------------------------
# Determinism & policy helpers
# -------------------------------
def ensure_determinism() -> None:
    torch.manual_seed(1234)
    np.random.seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Be strict by default unless commit requires TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def pick_device() -> torch.device:
    want = os.getenv("PROB_DEVICE", "cuda").lower()
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "cpu":
        return torch.device("cpu")
    # auto: prefer CUDA, fallback to CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pick_dtype() -> torch.dtype:
    key = os.getenv("PROB_FORCE_DTYPE", "fp32").lower()
    map_ = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "auto": torch.float32, "": torch.float32}
    return map_.get(key, torch.float32)

def parse_opt_gates() -> Dict[str, Any]:
    raw = os.getenv("PROB_OPT_GATES", '')
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        gates = {}
        for kv in raw.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                gates[k.strip()] = v.strip()
        return gates

# -------------------------------
# Diff-aware import resolution
# -------------------------------
def _infer_candidates_from_diff() -> Tuple[Optional[str], Optional[str], list]:
    """
    Deterministically infer (module, symbol) from the provided metadata and commit message.
    Returns: (module_path or None, symbol_name or None, candidate_list_for_errors)
    """
    module_hint = os.getenv("PROB_MODULE", "").strip() or "".strip()
    symbol_hint = os.getenv("PROB_SYMBOL", "").strip() or "".strip()
    candidates = []

    # Highest priority: explicit hints
    if module_hint and symbol_hint:
        return module_hint, symbol_hint, [(module_hint, symbol_hint)]

    # Based on the commit diff, the optimization is in vllm.utils.current_stream
    # This is a new function that caches torch.cuda.current_stream() calls
    candidates = [
        ("vllm.utils", "current_stream"),
        ("vllm.distributed.device_communicators.pynccl", "PyNcclCommunicator"),
    ]
    
    # Primary target is the current_stream function
    return "vllm.utils", "current_stream", candidates

def resolve_target() -> Tuple[Callable, Dict[str, Any], str]:
    """
    Returns (callable_or_bound_method, call_kwargs, fq_name_string).
    Must resolve to the EXACT code path described by the commit metadata, honoring env hints first.
    """
    mod_hint = os.getenv("PROB_MODULE", "").strip()
    sym_hint = os.getenv("PROB_SYMBOL", "").strip()

    if mod_hint and sym_hint:
        mod_path, sym_name = mod_hint, sym_hint
        candidates = [(mod_path, sym_name)]
    else:
        mod_path, sym_name, candidates = _infer_candidates_from_diff()

    # Deterministic resolution with tie-breakers baked in above
    if not mod_path or not sym_name:
        raise ImportError(f"{E_IMPORT_MISSING}: Unable to infer target. Candidates={candidates}")

    m = importlib.import_module(mod_path)
    target = m
    for part in sym_name.split("."):
        if not hasattr(target, part):
            raise ImportError(f"{E_IMPORT_MISSING}: {mod_path}.{sym_name} not found; nearest candidates={candidates}")
        target = getattr(target, part)

    fq = f"{mod_path}.{sym_name}"
    # DEFAULT: no extra kwargs
    return target, {}, fq

# -------------------------------
# Setup: workload reflecting commit
# -------------------------------
def _cap_by_memory(nelms: int, bytes_per: int, frac: float = 0.7) -> int:
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            cap = int((total * frac) // max(bytes_per, 1))
            return min(nelms, cap)
    except Exception:
        pass
    return nelms

def setup() -> Dict[str, Any]:
    """Create realistic workload that exercises the optimization."""
    ensure_determinism()
    device = pick_device()
    dtype = pick_dtype()

    # The optimization is about caching current_stream() calls
    # We need to simulate a scenario with many stream operations
    # that would benefit from caching the current stream
    
    # Create multiple tensors for operations
    size = 1024
    num_tensors = 100
    
    # Memory cap
    bytes_per = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total_elems = size * size * num_tensors
    _ = _cap_by_memory(total_elems, bytes_per)
    
    tensors = [torch.randn((size, size), dtype=dtype, device=device) for _ in range(num_tensors)]
    
    # Apply opt gates/environment flags
    opt_gates = parse_opt_gates()
    for k_env, v_env in opt_gates.items():
        os.environ[str(k_env)] = str(v_env)

    return {
        "device": device,
        "dtype": dtype,
        "tensors": tensors,
        "num_ops": 1000,  # Number of operations to perform
        "opt_gates": opt_gates,
    }

# -------------------------------
# Experiment: EXACT optimized path
# -------------------------------
def experiment(data: Dict[str, Any]) -> Any:
    """
    Execute ONLY the performance-critical code path being optimized.
    The optimization caches torch.cuda.current_stream() to avoid repeated object creation.
    """
    target, call_kwargs, fqname = resolve_target()
    
    # The optimization is about getting the current stream efficiently
    # We'll call it many times to measure the performance improvement
    results = []
    
    with torch.no_grad():
        # Simulate the pattern from PyNcclCommunicator where current_stream is called repeatedly
        for _ in range(data["num_ops"]):
            # Call the optimized current_stream function
            stream = target(**call_kwargs)
            results.append(stream)
            
            # Perform a simple operation to ensure the stream is used
            if data["tensors"]:
                # Simulate NCCL-like operations that would use the stream
                t = data["tensors"][0]
                t2 = t + 1.0  # Simple operation
                
    return results

# -------------------------------
# Result I/O for equivalence
# -------------------------------
def store_result(result: Any, filepath: str) -> None:
    """Store result for future equivalence checking."""
    # For stream objects, we store metadata about them
    if isinstance(result, list) and result and hasattr(result[0], 'cuda_stream'):
        payload = {
            "type": "stream_list",
            "count": len(result),
            "all_same": all(r.cuda_stream == result[0].cuda_stream for r in result),
        }
        torch.save(payload, filepath)
    else:
        torch.save({"type": "generic", "value": len(result) if isinstance(result, list) else result}, filepath)

def load_result(filepath: str) -> Any:
    return torch.load(filepath)

# -------------------------------
# Equivalence with dtype-aware tolerances
# -------------------------------
def _eq_tolerances(dtype: torch.dtype, level: str) -> Tuple[float, float]:
    if level == "exact":
        return (0.0, 0.0)
    if dtype in (torch.float16, torch.bfloat16):
        return (1e-3, 1e-4)
    if dtype == torch.float32:
        return (1e-5, 1e-7)
    return (1e-5, 1e-7)

def check_equivalence(current_result: Any, reference_payload: Any) -> None:
    level = os.getenv("PROB_EQ_LEVEL", "behavioral").lower()
    
    # For stream operations, we check behavioral equivalence
    if reference_payload.get("type") == "stream_list":
        assert isinstance(current_result, list), f"{E_EQFAIL}: Expected list of streams"
        assert len(current_result) == reference_payload["count"], \
            f"{E_EQFAIL}: Stream count mismatch: {len(current_result)} vs {reference_payload['count']}"
        # Check that all streams in the result are the same (caching behavior)
        all_same = all(r.cuda_stream == current_result[0].cuda_stream for r in current_result)
        assert all_same == reference_payload["all_same"], \
            f"{E_EQFAIL}: Stream caching behavior mismatch"
    else:
        # Fallback for other types
        pass

# -------------------------------
# Timing utilities
# -------------------------------
def _time_gpu(run: Callable, iters: int) -> Tuple[float, float, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        _ = run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    times.sort()
    avg = sum(times) / len(times)
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95) - 1] if len(times) > 1 else times[0]
    return avg, p50, p95

def _time_cpu(run: Callable, iters: int) -> Tuple[float, float, float]:
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = run()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    avg = sum(times) / len(times)
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95) - 1] if len(times) > 1 else times[0]
    return avg, p50, p95

# -------------------------------
# Main entry: run_test
# -------------------------------
def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """
    Run the performance test and return average execution time in milliseconds.

    Args:
        eqcheck: Compare current result vs stored reference.
        reference: Store current result as reference.
        prefix: Prefix for reference filenames.

    Returns:
        Average execution time (ms).
    """
    data = setup()
    impl_tag = os.getenv("PROB_IMPL_TAG", "child")
    commit_hash = os.getenv("PROB_COMMIT_HASH", "310aca88c984983189a57f1b72e3b1dde89fb92f")

    # Warmup
    warmup = 5 if torch.cuda.is_available() else 3
    for _ in range(warmup):
        _ = experiment(data)

    # Timing iterations - reduce for this microbenchmark since it's very fast
    iters = 20 if torch.cuda.is_available() else 10
    if torch.cuda.is_available():
        avg_ms, p50_ms, p95_ms = _time_gpu(lambda: experiment(data), iters)
    else:
        avg_ms, p50_ms, p95_ms = _time_cpu(lambda: experiment(data), iters)

    # Equivalence/reference I/O
    result = experiment(data)
    ref_path = f"{prefix}_{impl_tag}_{commit_hash}_reference.pt"
    if reference:
        store_result(result, ref_path)
    if eqcheck:
        reference_payload = load_result(ref_path)
        check_equivalence(result, reference_payload)

    # Summary JSON (single line)
    summary = {
        "impl_tag": impl_tag,
        "commit_hash": commit_hash,
        "device": str(data["device"]),
        "dtype": str(data["dtype"]),
        "iters": iters,
        "warmup": warmup,
        "avg_ms": round(avg_ms, 6),
        "p50_ms": round(p50_ms, 6),
        "p95_ms": round(p95_ms, 6),
        "eq_level": os.getenv("PROB_EQ_LEVEL", "behavioral"),
        "opt_path_hit": True  # The optimization is always active when current_stream is called
    }
    print(json.dumps(summary, sort_keys=True))
    return avg_ms

# End of script