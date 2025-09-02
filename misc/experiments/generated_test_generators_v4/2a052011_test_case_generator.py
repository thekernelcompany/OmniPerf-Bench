import os
import json
import torch
import numpy as np
from typing import Dict, Any, Tuple

# Import the actual optimized modules from the commit
# We use the fused_moe kernel and the FP8 quantizer used in MixtralMoE.process_weights_after_loading.
try:
    from vllm.model_executor.layers.fused_moe import fused_moe
except Exception as e:
    raise ImportError(
        "Failed to import vllm.model_executor.layers.fused_moe.fused_moe. "
        "This performance test targets the fused MoE kernel from vLLM. "
        f"Original error: {e}"
    )

try:
    from vllm import _custom_ops as ops  # for ops.scaled_fp8_quant
except Exception as e:
    raise ImportError(
        "Failed to import vllm._custom_ops (needed for scaled_fp8_quant). "
        "This performance test requires the vLLM custom ops. "
        f"Original error: {e}"
    )

def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to run this FP8 fused MoE performance test. "
            "No CUDA device detected."
        )
    # Basic sanity check for FP8 dtype support in torch
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "Your PyTorch build does not expose torch.float8_e4m3fn. "
            "An FP8-capable build is required."
        )

def setup() -> Dict[str, Any]:
    """Create realistic workload that exercises the FP8-optimized fused MoE path."""
    torch.manual_seed(42)
    np.random.seed(42)
    _require_cuda()

    device = "cuda"

    # Dimensions chosen to reflect a realistic Mixtral-like workload while being memory-conscious:
    # - hidden_size (k): 4096 (Mixtral hidden dim)
    # - intermediate_size (n): 4096 (reduced from 14336 to keep memory in check)
    # - num_experts (e): 8 (Mixtral-8x)
    # - top_k: 2 (Mixtral default)
    # - m (tokens): 1024 (sufficient to drive the kernel)
    m = int(os.environ.get("VLLM_BENCH_M", 1024))
    k = int(os.environ.get("VLLM_BENCH_K", 4096))
    n = int(os.environ.get("VLLM_BENCH_N", 4096))
    e = int(os.environ.get("VLLM_BENCH_E", 8))
    topk = int(os.environ.get("VLLM_BENCH_TOPK", 2))

    # Activations in half precision (router is half/full in actual flow)
    a = (torch.randn((m, k), device=device, dtype=torch.float16) / 10)

    # Expert weights prior to quantization (fp16) - shapes follow fused_moe:
    # w1/w3 are concatenated as (E, 2N, K); w2 is (E, K, N)
    w13_fp16 = (torch.randn((e, 2 * n, k), device=device, dtype=torch.float16) / 10)
    w2_fp16 = (torch.randn((e, k, n), device=device, dtype=torch.float16) / 10)

    # Router logits (dtype aligned with activation dtype in upstream tests)
    score = torch.randn((m, e), device=device, dtype=torch.float16)

    # Quantize weights to FP8 with per-expert scales (as in MixtralMoE.process_weights_after_loading)
    w13_fp8 = torch.empty_like(w13_fp16, dtype=torch.float8_e4m3fn)
    w2_fp8 = torch.empty_like(w2_fp16, dtype=torch.float8_e4m3fn)
    w13_scales = torch.empty((e,), device=device, dtype=torch.float32)
    w2_scales = torch.empty((e,), device=device, dtype=torch.float32)

    # Quantize each expert tensor independently to simulate diverse per-expert scales
    for expert in range(e):
        q_w13, s13 = ops.scaled_fp8_quant(w13_fp16[expert])
        q_w2, s2 = ops.scaled_fp8_quant(w2_fp16[expert])
        w13_fp8[expert].copy_(q_w13)
        w2_fp8[expert].copy_(q_w2)
        w13_scales[expert] = s13
        w2_scales[expert] = s2

    # Activation scales: commit supports dynamic or static; here we use dynamic (None)
    a13_scale = None
    a2_scale = None

    return {
        "a": a,
        "w13": w13_fp8,
        "w2": w2_fp8,
        "score": score,
        "topk": topk,
        "w13_scale": w13_scales,
        "w2_scale": w2_scales,
        "a13_scale": a13_scale,
        "a2_scale": a2_scale,
    }

def experiment(data: Dict[str, Any]) -> Any:
    """
    Execute the performance-critical code path being optimized:
    fused_moe with FP8 expert weights and per-expert scales (use_fp8=True).
    """
    with torch.no_grad():
        out = fused_moe(
            data["a"],           # (m, k)
            data["w13"],         # (e, 2n, k) FP8
            data["w2"],          # (e, k, n)  FP8
            data["score"],       # (m, e)
            data["topk"],        # top-k experts
            renormalize=True,
            inplace=True,
            use_fp8=True,
            w1_scale=data["w13_scale"],  # per-expert scale
            w2_scale=data["w2_scale"],   # per-expert scale
            a1_scale=data["a13_scale"],  # None for dynamic activations
            a2_scale=data["a2_scale"],   # None for dynamic activations
        )
    return out

def store_result(result: Any, filepath: str) -> None:
    """Store result for future equivalence checking"""
    if isinstance(result, torch.Tensor):
        meta = {
            'shape': tuple(result.shape),
            'dtype': str(result.dtype),
            'device': str(result.device),
        }
        torch.save({
            'result': result.detach().cpu(),  # store on CPU for portability
            'meta': meta,
            'sample_values': result.flatten()[:100].detach().cpu().numpy().tolist(),
        }, filepath)
    else:
        # Fallback to JSON for non-tensors
        with open(filepath, 'w') as f:
            json.dump(result, f)

def load_result(filepath: str) -> Any:
    """Load reference result for equivalence checking"""
    obj = torch.load(filepath, map_location='cpu')
    return obj['result']

def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Check that current result is equivalent to reference"""
    if not isinstance(current_result, torch.Tensor) or not isinstance(reference_result, torch.Tensor):
        raise AssertionError("Results are expected to be torch.Tensors.")

    # Shape and dtype checks
    assert current_result.shape == reference_result.shape, f"Shape mismatch: {current_result.shape} vs {reference_result.shape}"
    # Tensors are stored as CPU in reference to be portable; move current to CPU for comparison
    current_cpu = current_result.detach().cpu()
    reference_cpu = reference_result.detach().cpu()
    assert current_cpu.dtype == reference_cpu.dtype, f"Dtype mismatch: {current_cpu.dtype} vs {reference_cpu.dtype}"

    # FP8 path determinism is generally stable; allow small tolerance
    torch.testing.assert_close(
        current_cpu, reference_cpu,
        rtol=1e-3, atol=1e-3,
        msg="Results are not numerically equivalent"
    )

def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """
    Run the performance test and return average execution time per iteration (milliseconds).

    Args:
        eqcheck: Whether to perform equivalence checking against a stored reference
        reference: Whether to store result as reference
        prefix: Prefix for reference files

    Returns:
        Average execution time in milliseconds
    """
    _require_cuda()
    data = setup()

    # Use CUDA events for precise GPU timing
    torch.cuda.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = experiment(data)
    torch.cuda.synchronize()

    # Measure performance
    num_iterations = 30
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        for _ in range(num_iterations):
            result = experiment(data)
        end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    # Equivalence handling
    if eqcheck or reference:
        # Fresh result for correctness snapshot
        result = experiment(data)
        ref_path = f'{prefix}_reference.pt'
        if reference:
            store_result(result, ref_path)
        elif eqcheck:
            reference_result = load_result(ref_path)
            check_equivalence(result, reference_result)

    return elapsed_time_ms / num_iterations

# The harness will be automatically appended here with argument parsing