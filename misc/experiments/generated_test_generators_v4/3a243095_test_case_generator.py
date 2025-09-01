import torch
import numpy as np
from typing import Dict, Any, Tuple

# Try to import the actual optimized function from vLLM
GET_RANKS_SRC = "local_fallback"
try:
    # The optimized function lives in vllm.model_executor.layers.sampler._get_ranks
    # We attempt a direct import of its symbol.
    from vllm.model_executor.layers.sampler import _get_ranks as get_ranks  # type: ignore
    GET_RANKS_SRC = "vllm.model_executor.layers.sampler._get_ranks"
except Exception as e:
    # Fallback to a local implementation matching the optimized commit
    # This ensures the script remains executable even if vllm isn't installed.
    def get_ranks(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        vals = x[torch.arange(0, len(x), device=x.device, dtype=indices.dtype), indices]
        return (x > vals[:, None]).long().sum(1).add_(1)
    print(f"Warning: Could not import optimized _get_ranks from vLLM ({e}). "
          f"Using local fallback implementation matching the commit.")


def setup() -> Dict[str, Any]:
    """Create realistic workload that exercises the optimization"""
    torch.manual_seed(42)
    np.random.seed(42)

    # Realistic tensor sizes for logprobs (N tokens, V vocab). Balance memory/time.
    # On GPU: larger workload to stress the kernel; on CPU: smaller to keep runtime reasonable.
    if torch.cuda.is_available():
        N = 1024   # number of rows (tokens)
        V = 16384  # vocab size
        device = torch.device("cuda")
    else:
        N = 256
        V = 8192
        device = torch.device("cpu")

    # Create a logprob-like tensor. Values don't need to sum to 1 for rank computation.
    # Use float32 to match typical logprobs dtype in vLLM and keep memory reasonable.
    x = torch.randn((N, V), dtype=torch.float32, device=device)
    # Optional: make it "logprob-like" by normalizing (no effect on ranks order, but realistic)
    x = torch.log_softmax(x, dim=-1)

    # Create selected token indices for each row
    indices = torch.randint(low=0, high=V, size=(N,), dtype=torch.long, device=device)

    return {
        'x': x,
        'indices': indices,
        'N': N,
        'V': V,
        'device': device,
        'get_ranks_src': GET_RANKS_SRC,
    }


def experiment(data: Dict[str, Any]) -> Any:
    """
    Execute the performance-critical code path being optimized.

    Args:
        data: Dictionary containing setup data from setup() function

    Returns:
        Result of the optimized computation (ranks tensor, shape [N], dtype long)
    """
    x = data['x']
    indices = data['indices']
    # Exact call matching the optimized function signature from the commit
    with torch.no_grad():
        result = get_ranks(x, indices)
    return result


def store_result(result: Any, filepath: str) -> None:
    """Store result for future equivalence checking"""
    if isinstance(result, torch.Tensor):
        torch.save({
            'type': 'tensor',
            'shape': tuple(result.shape),
            'dtype': str(result.dtype),
            'device': str(result.device),
            'result': result.detach().cpu(),  # store on CPU for portability
            'sample_values': result.detach().flatten()[:100].cpu().clone(),
        }, filepath)
    else:
        # Generic fallback (shouldn't happen here)
        import json
        with open(filepath, 'w') as f:
            json.dump({'result': result}, f)


def load_result(filepath: str) -> Any:
    """Load reference result for equivalence checking"""
    obj = torch.load(filepath, map_location='cpu')
    if isinstance(obj, dict) and obj.get('type') == 'tensor':
        return obj['result']
    return obj


def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Check that current result is equivalent to reference"""
    assert isinstance(current_result, torch.Tensor), "Current result must be a torch.Tensor"
    if isinstance(reference_result, dict) and 'result' in reference_result:
        reference_result = reference_result['result']
    assert isinstance(reference_result, torch.Tensor), "Reference result must be a torch.Tensor"

    # Compare on CPU to avoid device mismatches due to serialization choices
    cur = current_result.detach().cpu()
    ref = reference_result.detach().cpu()

    # Basic properties
    assert cur.shape == ref.shape, f"Shape mismatch: {cur.shape} vs {ref.shape}"
    assert cur.dtype == ref.dtype, f"Dtype mismatch: {cur.dtype} vs {ref.dtype}"

    # Exact equality for integer ranks
    torch.testing.assert_close(cur, ref, rtol=0, atol=0, msg="Rank results mismatch")


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
    data = setup()

    # Warmup and timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

        # Warmup: run the kernel a few times to stabilize any lazy init
        with torch.no_grad():
            for _ in range(5):
                _ = experiment(data)
        torch.cuda.synchronize()

        # Measure with CUDA events
        num_iterations = 50
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                result = experiment(data)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
    else:
        import time
        # Warmup
        for _ in range(3):
            _ = experiment(data)

        num_iterations = 20
        start_time = time.time()
        for _ in range(num_iterations):
            result = experiment(data)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000.0  # ms

    # Equivalence handling
    if eqcheck or reference:
        # Get a fresh result to store/check
        result = experiment(data)
        ref_path = f'{prefix}_reference.pt'
        if reference:
            store_result(result, ref_path)
        elif eqcheck:
            reference_result = load_result(ref_path)
            check_equivalence(result, reference_result)

    return elapsed_time / float(num_iterations)

# The harness will be automatically appended here with argument parsing