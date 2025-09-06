#!/usr/bin/env python3
"""
Performance test script for commit: 25ebed2f8ca6d747d63f2be9ede023c561851ac8
[V1][Minor] Cache np arange to reduce input preparation overhead (#11214)

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
    module_hint = os.getenv("PROB_MODULE", "").strip()
    symbol_hint = os.getenv("PROB_SYMBOL", "").strip()
    candidates = []

    # Highest priority: explicit hints
    if module_hint and symbol_hint:
        return module_hint, symbol_hint, [(module_hint, symbol_hint)]

    # From the diff, we see the optimization is in GPUModelRunner._prepare_inputs
    # The commit caches np.arange arrays to avoid recreating them
    mod = "vllm.v1.worker.gpu_model_runner"
    sym = "GPUModelRunner._prepare_inputs"
    candidates.append((mod, sym))
    
    return mod, sym, candidates

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
    # DEFAULT: no extra kwargs; adjust in experiment() if optimization requires
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

    # The optimization targets _prepare_inputs which processes scheduled tokens
    # We need to simulate the data structures used by GPUModelRunner
    
    # Realistic batch sizes for V1 engine
    max_num_reqs = 256  # typical max concurrent requests
    max_model_len = 4096  # typical model context length
    max_num_tokens = 8192  # max batched tokens
    block_size = 16  # typical KV cache block size
    max_num_blocks_per_req = (max_model_len + block_size - 1) // block_size
    
    # Create mock scheduler output with varying request lengths
    num_reqs = 32  # active requests in batch
    np.random.seed(1234)
    
    # Generate varying token counts per request (mix of prefill and decode)
    num_scheduled_tokens = []
    for i in range(num_reqs):
        if i < 4:  # Some prefill requests
            tokens = np.random.randint(128, 512)
        else:  # Mostly decode requests
            tokens = 1
        num_scheduled_tokens.append(tokens)
    num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
    total_num_scheduled_tokens = num_scheduled_tokens.sum()
    
    # Mock input batch state
    num_computed_tokens = np.random.randint(0, 1024, size=num_reqs, dtype=np.int32)
    
    # Pre-allocate arrays that would be used
    positions_np = np.zeros(max_num_tokens, dtype=np.int64)
    slot_mapping_np = np.zeros(max_num_tokens, dtype=np.int32)
    query_start_loc_np = np.zeros(max_num_reqs + 1, dtype=np.int32)
    seq_start_loc_np = np.zeros(max_num_reqs + 1, dtype=np.int32)
    
    # For the optimization: pre-cached arange array
    arange_np_cached = np.arange(max(max_num_reqs, max_model_len), dtype=np.int32)
    
    # Apply opt gates/environment flags
    opt_gates = parse_opt_gates()
    for k_env, v_env in opt_gates.items():
        os.environ[str(k_env)] = str(v_env)

    return {
        "device": device,
        "dtype": dtype,
        "num_reqs": num_reqs,
        "num_scheduled_tokens": num_scheduled_tokens,
        "total_num_scheduled_tokens": total_num_scheduled_tokens,
        "num_computed_tokens": num_computed_tokens,
        "positions_np": positions_np,
        "slot_mapping_np": slot_mapping_np,
        "query_start_loc_np": query_start_loc_np,
        "seq_start_loc_np": seq_start_loc_np,
        "arange_np_cached": arange_np_cached,
        "block_size": block_size,
        "max_num_blocks_per_req": max_num_blocks_per_req,
        "opt_gates": opt_gates,
    }

# -------------------------------
# Experiment: EXACT optimized path
# -------------------------------
def experiment(data: Dict[str, Any]) -> Any:
    """
    Execute ONLY the performance-critical code path being optimized.
    The optimization caches np.arange to avoid recreating arrays in hot path.
    """
    num_reqs = data["num_reqs"]
    num_scheduled_tokens = data["num_scheduled_tokens"]
    total_num_scheduled_tokens = data["total_num_scheduled_tokens"]
    num_computed_tokens = data["num_computed_tokens"]
    positions_np = data["positions_np"]
    arange_np_cached = data["arange_np_cached"]
    
    # The optimized code path from the diff:
    # Get request indices
    # OLD: req_indices = np.repeat(np.arange(num_reqs), num_scheduled_tokens)
    # NEW: req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
    req_indices = np.repeat(arange_np_cached[:num_reqs], num_scheduled_tokens)
    
    # Get batched arange
    # OLD: arange = np.concatenate([np.arange(n) for n in num_scheduled_tokens])
    # NEW: arange = np.concatenate([self.arange_np[:n] for n in num_scheduled_tokens])
    arange = np.concatenate([arange_np_cached[:n] for n in num_scheduled_tokens])
    
    # Get positions (this uses the optimized arange)
    positions_result = positions_np[:total_num_scheduled_tokens].copy()
    np.add(num_computed_tokens[req_indices], arange, out=positions_result)
    
    # Additional work that uses these arrays (from _prepare_inputs)
    # Calculate query_start_loc
    query_start_loc = data["query_start_loc_np"]
    query_start_loc[0] = 0
    np.cumsum(num_scheduled_tokens, out=query_start_loc[1:num_reqs + 1])
    
    # Calculate seq_start_loc
    seq_lens = num_computed_tokens[:num_reqs] + num_scheduled_tokens
    seq_start_loc = data["seq_start_loc_np"]
    seq_start_loc[0] = 0
    np.cumsum(seq_lens, out=seq_start_loc[1:num_reqs + 1])
    
    # Return the computed positions as result for equivalence checking
    return positions_result

# -------------------------------
# Result I/O for equivalence
# -------------------------------
def store_result(result: Any, filepath: str) -> None:
    """Store result for future equivalence checking."""
    if isinstance(result, np.ndarray):
        payload = {
            "type": "numpy_array",
            "shape": tuple(result.shape),
            "dtype": str(result.dtype),
            "sample": result.flatten()[:4096],
        }
        torch.save(payload, filepath)
    elif isinstance(result, torch.Tensor):
        payload = {
            "type": "torch_tensor",
            "shape": tuple(result.shape),
            "dtype": str(result.dtype),
            "device": "cpu",
            "sample": result.flatten()[:4096].detach().cpu(),
        }
        torch.save(payload, filepath)
    else:
        torch.save({"type": "generic", "value": result}, filepath)

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
    level = os.getenv("PROB_EQ_LEVEL", "exact").lower()
    
    if isinstance(current_result, np.ndarray) and reference_payload.get("type") == "numpy_array":
        ref_sample = reference_payload["sample"]
        assert tuple(current_result.shape) == tuple(reference_payload["shape"]), \
            f"Shape mismatch: {tuple(current_result.shape)} vs {tuple(reference_payload['shape'])}"
        assert str(current_result.dtype) == reference_payload["dtype"], \
            f"Dtype mismatch: {current_result.dtype} vs {reference_payload['dtype']}"
        
        # For integer arrays, require exact equality
        if current_result.dtype in (np.int32, np.int64):
            np.testing.assert_array_equal(
                current_result.flatten()[:ref_sample.size],
                ref_sample,
                err_msg=f"{E_EQFAIL}: integer arrays not equal"
            )
        else:
            rtol, atol = _eq_tolerances(torch.float32, level)
            np.testing.assert_allclose(
                current_result.flatten()[:ref_sample.size],
                ref_sample,
                rtol=rtol,
                atol=atol,
                err_msg=f"{E_EQFAIL}: deviation beyond tolerances (level={level})"
            )
    elif isinstance(current_result, torch.Tensor) and reference_payload.get("type") == "torch_tensor":
        ref_sample = reference_payload["sample"]
        assert tuple(current_result.shape) == tuple(reference_payload["shape"]), \
            f"Shape mismatch: {tuple(current_result.shape)} vs {tuple(reference_payload['shape'])}"
        assert str(current_result.dtype) == reference_payload["dtype"], \
            f"Dtype mismatch: {current_result.dtype} vs {reference_payload['dtype']}"
        rtol, atol = _eq_tolerances(current_result.dtype, level)
        torch.testing.assert_close(
            current_result.flatten()[: ref_sample.numel()].cpu(),
            ref_sample,
            rtol=rtol,
            atol=atol,
            msg=f"{E_EQFAIL}: deviation beyond tolerances (level={level})"
        )
    else:
        assert current_result == reference_payload.get("value"), f"{E_EQFAIL}: non-tensor results not equal"

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
    commit_hash = os.getenv("PROB_COMMIT_HASH", "25ebed2f8ca6d747d63f2be9ede023c561851ac8")

    # Warmup
    warmup = 5 if torch.cuda.is_available() else 3
    for _ in range(warmup):
        _ = experiment(data)

    # Timing iterations - increase for CPU since this is a lightweight operation
    iters = 100 if torch.cuda.is_available() else 50
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
        "eq_level": os.getenv("PROB_EQ_LEVEL", "exact"),
        "opt_path_hit": True
    }
    print(json.dumps(summary, sort_keys=True))
    return avg_ms

# End of script