import torch
import numpy as np
from typing import Dict, Any, Tuple

# Import the actual optimized modules from the commit
# Based on the diff, _get_ranks is defined in vllm.model_executor.layers.sampler
try:
    from vllm.model_executor.layers import sampler as vllm_sampler
    HAS_VLLM = True
except Exception as e:
    vllm_sampler = None
    HAS_VLLM = False
    IMPORT_ERROR_MSG = f"vLLM import unavailable, using local fallback for _get_ranks: {e}"

def _get_ranks_fallback(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Fallback implementation matching the optimized commit logic:

    vals = x[torch.arange(0, len(x), device=x.device, dtype=indices.dtype), indices]
    return (x > vals[:, None]).long().sum(1).add_(1)
    """
    # Ensure indices is a tensor on the same device as x
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, device=x.device, dtype=torch.long)
    if indices.device != x.device:
        indices = indices.to(device=x.device)
    if indices.dtype != torch.long and indices.dtype != torch.int64:
        indices = indices.to(dtype=torch.long)

    ar = torch.arange(0, x.size(0), device=x.device, dtype=indices.dtype)
    vals = x[ar, indices]
    return (x > vals[:, None]).long().sum(1).add_(1)

def _get_ranks_dispatch(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Dispatch to the repository's optimized _get_ranks if available.
    Falls back to a local implementation if the import fails or if older versions
    expect List[int] and cannot handle tensor indices on GPU.
    """
    # Ensure tensor indices for the optimized path
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, device=x.device, dtype=torch.long)
    else:
        if indices.device != x.device:
            indices = indices.to(x.device)
        if indices.dtype != torch.long and indices.dtype != torch.int64:
            indices = indices.to(dtype=torch.long)

    if HAS_VLLM and hasattr(vllm_sampler, "_get_ranks"):
        try:
            # New optimized path expects tensor indices on the same device as x
            return vllm_sampler._get_ranks(x, indices)
        except Exception:
            # Compatibility with older versions (pre-commit) that used Python ranges
            try:
                return vllm_sampler._get_ranks(x, indices.cpu().tolist())
            except Exception:
                # Final fallback to local optimized implementation
                return _get_ranks_fallback(x, indices)
    else:
        return _get_ranks_fallback(x, indices)

def setup() -> Dict[str, Any]:
    """Create realistic workload that exercises the optimization"""
    torch.manual_seed(42)
    np.random.seed(42)

    # Realistic logprobs shape: N = number of tokens in batch being scored,
    # M = vocab size (e.g., ~50k for GPT-2/GPT-3 style vocabularies).
    # Choose sizes that stress the GPU path but remain reasonable on CPU.
    if torch.cuda.is_available():
        N = 1024  # number of tokens
        M = 50257  # vocab size typical in LLMs
        device = torch.device("cuda")
    else:
        N = 256
        M = 32768
        device = torch.device("cpu")

    # Create logprobs tensor; values don't matter beyond relative ordering.
    # Use float32 to match the sampler's logprobs dtype.
    x = torch.randn((N, M), dtype=torch.float32, device=device)

    # Create chosen token indices (one per token row)
    indices = torch.randint(low=0, high=M, size=(N,), dtype=torch.long, device=device)

    meta = {
        'N': N,
        'M': M,
        'device': str(device),
        'has_vllm': HAS_VLLM,
    }

    return {
        'x': x,
        'indices': indices,
        'meta': meta,
    }

def experiment(data: Dict[str, Any]) -> Any:
    """
    Execute the performance-critical code path being optimized: _get_ranks.

    Args:
        data: Dictionary containing setup data from setup() function

    Returns:
        Result of the optimized computation (ranks tensor, shape (N,), dtype long)
    """
    x: torch.Tensor = data['x']
    indices: torch.Tensor = data['indices']

    with torch.no_grad():
        # Use the exact function as per commit: _get_ranks(x, indices_tensor)
        ranks = _get_ranks_dispatch(x, indices)

    return ranks

def store_result(result: Any, filepath: str) -> None:
    """Store result for future equivalence checking"""
    if isinstance(result, torch.Tensor):
        cpu_res = result.detach().cpu()
        torch.save({
            'result': cpu_res,
            'shape': tuple(cpu_res.shape),
            'dtype': str(cpu_res.dtype),
            'device': 'cpu',
            'sample_values': cpu_res.flatten()[:100].tolist(),
        }, filepath)
    else:
        # Fallback generic serializer
        import json
        with open(filepath, 'w') as f:
            json.dump({'result': result}, f)

def load_result(filepath: str) -> Any:
    """Load reference result for equivalence checking"""
    try:
        data = torch.load(filepath, map_location='cpu')
        return data['result']
    except Exception:
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['result']

def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Check that current result is equivalent to reference"""
    if isinstance(current_result, torch.Tensor) and isinstance(reference_result, torch.Tensor):
        # Move to CPU for comparison
        cur = current_result.detach().cpu()
        ref = reference_result.detach().cpu()

        assert cur.shape == ref.shape, f"Shape mismatch: {cur.shape} vs {ref.shape}"
        assert cur.dtype == ref.dtype, f"Dtype mismatch: {cur.dtype} vs {ref.dtype}"
        # Integer equality check
        if cur.numel() > 0:
            if not torch.equal(cur, ref):
                # Provide a small diff summary
                diff_count = (cur != ref).sum().item()
                raise AssertionError(f"Rank values differ in {diff_count} positions")
    else:
        # Generic equality
        if current_result != reference_result:
            raise AssertionError("Results differ (non-tensor comparison)")

def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """
    Run the performance test and return execution time.

    Args:
        eqcheck: Whether to perform equivalence checking
        reference: Whether to store result as reference
        prefix: Prefix for reference files

    Returns:
        Average execution time in milliseconds
    """
    # Setup data once
    data = setup()

    # If vLLM wasn't importable, announce once (graceful fallback)
    if not HAS_VLLM:
        print(IMPORT_ERROR_MSG)

    # Check hardware availability (adapt based on optimization requirements)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = experiment(data)

        # Measure performance using CUDA events
        num_iterations = 10
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()
            for _ in range(num_iterations):
                result = experiment(data)
            end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
    else:
        # Fallback to CPU timing if GPU not available
        import time
        num_iterations = 3

        # Warmup
        for _ in range(1):
            _ = experiment(data)

        start_time = time.perf_counter()
        for _ in range(num_iterations):
            result = experiment(data)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000.0  # Convert to milliseconds

    # Handle equivalence checking
    if eqcheck or reference:
        result = experiment(data)  # Get fresh result for checking

        if reference:
            store_result(result, f'{prefix}_reference.pt')
        elif eqcheck:
            reference_result = load_result(f'{prefix}_reference.pt')
            # Convert reference_result to tensor if needed
            if not isinstance(reference_result, torch.Tensor):
                reference_result = torch.tensor(reference_result)
            check_equivalence(result, reference_result)

    return elapsed_time / num_iterations

# The harness will be automatically appended here with argument parsing