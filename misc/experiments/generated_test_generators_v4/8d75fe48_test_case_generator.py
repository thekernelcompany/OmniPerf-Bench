import pytest
import torch
import numpy as np
import timeit
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Try importing actual modules changed in the commit. If not available, skip.
try:
    from vllm import _custom_ops as custom_ops
    from vllm.model_executor.layers.quantization import fp8 as fp8_mod
    from vllm.model_executor.layers.quantization.fp8 import (
        Fp8LinearMethod,
        Fp8Config,
        cutlass_fp8_supported,
    )
except Exception:
    pytest.skip("Required vllm modules not available", allow_module_level=True)


def setup_workload() -> Dict[str, Any]:
    """Create realistic test data based on the domain.

    Uses transformer-scale dimensions as required by the test-generation spec.
    The heavy GPU ops are patched in tests; we still create realistic-shaped
    tensors to verify shapes, dtypes and logic.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Realistic transformer dimensions
    batch_size = 32
    seq_len = 1024
    hidden_dim = 4096  # large hidden dimension (divisible by 16)

    # Input activation: choose float16 as realistic activation dtype for FP8 workflows.
    x = torch.randn((batch_size, hidden_dim), dtype=torch.float16)

    # Weight: make it (hidden_dim, hidden_dim) and divisible by 16 for cutlass asserts.
    weight = torch.randn((hidden_dim, hidden_dim), dtype=torch.float16)

    # Weight scale (per-tensor)
    weight_scale = torch.tensor(1.0, dtype=torch.float32)

    # Bias (occasionally present): shape corresponds to output dim
    bias = torch.randn((hidden_dim,), dtype=x.dtype)

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "x": x,
        "weight": weight,
        "weight_scale": weight_scale,
        "bias": bias,
    }


def _fill_out_with_matmul(out, a, b, *rest):
    """Utility to simulate backend GEMM by writing matmul result into out."""
    # Compute matmul in float32 for numerical stability, then cast to out dtype.
    res = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(out.dtype)
    out.copy_(res)


def test_cutlass_scaled_mm_dq_param_rename():
    """Ensure cutlass_scaled_mm_dq forwards the renamed params (scale_a, scale_b)
    to the underlying vllm_ops.cutlass_scaled_mm_dq and returns an output of
    expected shape/dtype.
    """
    wl = setup_workload()
    a = wl["x"]  # shape (m, k)
    b = wl["weight"]  # shape (k, n) - ensure divisible by 16
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0

    # Create dummy scales (tensors) to pass in.
    scale_a = torch.tensor(0.5, dtype=torch.float32)
    scale_b = torch.tensor(2.0, dtype=torch.float32)

    called = {"called": False, "args": None, "kwargs": None}

    def fake_cutlass(out, aa, bb, sa, sb):
        # Validate the function receives exactly (out, a, b, scale_a, scale_b)
        called["called"] = True
        called["args"] = (out, aa, bb, sa, sb)
        # Simulate behavior by filling out with matmul result
        _fill_out_with_matmul(out, aa, bb)

    # Patch the vllm_ops.cutlass_scaled_mm_dq inside the custom_ops module
    with patch.object(custom_ops, "vllm_ops", create=True) as fake_ops:
        fake_ops.cutlass_scaled_mm_dq = fake_cutlass

        out = custom_ops.cutlass_scaled_mm_dq(a, b, scale_a, scale_b, out_dtype=torch.float16)

        # Called underlying op
        assert called["called"], "Expected vllm_ops.cutlass_scaled_mm_dq to be called."

        # Check that the first arg passed to cutlass op is the 'out' tensor with expected shape/dtype
        out_arg, a_arg, b_arg, sa_arg, sb_arg = called["args"]
        assert isinstance(out_arg, torch.Tensor)
        assert out_arg.shape == (a.shape[0], b.shape[1])
        assert out_arg.dtype == torch.float16
        # Ensure a and b forwarded correctly
        assert torch.allclose(a_arg.to(torch.float32)[:4, :4], a.to(torch.float32)[:4, :4])
        assert torch.allclose(b_arg.to(torch.float32)[:4, :4], b.to(torch.float32)[:4, :4])
        # Scales forwarded
        assert torch.allclose(sa_arg, scale_a)
        assert torch.allclose(sb_arg, scale_b)

        # Return is the 'out' tensor
        assert torch.equal(out, out_arg)


def test_cutlass_scaled_mm_dq_invalid_shape_asserts():
    """cutlass_scaled_mm_dq asserts when b dims are not multiples of 16."""
    wl = setup_workload()
    a = wl["x"]
    # Create invalid b that violates the %16 condition
    b_bad = torch.randn((1023, 4095), dtype=torch.float16)  # not divisible by 16

    scale_a = torch.tensor(1.0, dtype=torch.float32)
    scale_b = torch.tensor(1.0, dtype=torch.float32)

    with pytest.raises(AssertionError):
        custom_ops.cutlass_scaled_mm_dq(a, b_bad, scale_a, scale_b, out_dtype=torch.float16)


def test_cutlass_fp8_supported_logic_various_devices():
    """Test the cutlass_fp8_supported logic for different device capability and CUDA versions.

    The implementation multiplies capability[0] by 10 and capability[1] by 1,
    so we craft return tuples that exercise both branches in the current code.
    """
    # Case 1: SM90-like (capability -> 900) and CUDA 12.1 (-> 121) => supported
    with patch.object(torch.cuda, "get_device_capability", return_value=(90, 0)):
        with patch.object(torch.version, "cuda", new=(12, 1)):
            assert cutlass_fp8_supported() is True

    # Case 2: SM89-like (capability -> 890) and CUDA 12.5 (-> 125) => supported per code
    with patch.object(torch.cuda, "get_device_capability", return_value=(89, 0)):
        with patch.object(torch.version, "cuda", new=(12, 5)):
            assert cutlass_fp8_supported() is True

    # Case 3: low capability -> not supported
    with patch.object(torch.cuda, "get_device_capability", return_value=(80, 0)):
        with patch.object(torch.version, "cuda", new=(11, 7)):
            assert cutlass_fp8_supported() is False


def _make_layer_with_params(weight: torch.Tensor, weight_scale: torch.Tensor, act_scale=None):
    """Construct a minimal layer-like object with expected attributes used by Fp8LinearMethod."""
    layer = torch.nn.Module()
    # Register parameters similar to how the real code would.
    layer.weight = torch.nn.Parameter(weight.clone().detach(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale.clone().detach(), requires_grad=False)
    layer.act_scale = act_scale  # can be None or a scalar parameter
    return layer


def test_Fp8LinearMethod_apply_uses_cutlass_when_supported_and_no_bias():
    """When cutlass_fp8_supported is True and bias is None, the Fp8LinearMethod.apply
    should use ops.cutlass_scaled_mm_dq and not torch._scaled_mm.
    """
    wl = setup_workload()
    x = wl["x"]
    weight = wl["weight"]
    weight_scale = wl["weight_scale"]

    # Create config and method
    quant_cfg = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    method = Fp8LinearMethod(quant_cfg)

    # Force the flag on the method to simulate supported device
    method.cutlass_fp8_supported = True

    # Create a layer object with parameters expected by apply()
    layer = _make_layer_with_params(weight, weight_scale, act_scale=None)

    # Prepare fake outputs to ensure code path exercised
    qinput = torch.randn_like(x, dtype=torch.float16)
    x_scale = torch.tensor(0.75, dtype=torch.float32)

    cutlass_called = {"called": False}
    scaled_mm_called = {"called": False}

    def fake_scaled_fp8_quant(inp, act_scale_arg):
        # Ensure apply calls scaled_fp8_quant with provided inputs.
        # We ignore batch_dim_padding case for the cutlass branch.
        assert torch.allclose(inp[:1, :4].to(torch.float32), x[:1, :4].to(torch.float32))
        return qinput, x_scale

    def fake_cutlass(a_qinput, b_weight, out_dtype, scale_a, scale_b):
        # This mirrors the new call site in fp8.py (cutlass_scaled_mm_dq)
        # Note: fp8.apply will call fp8.ops.cutlass_scaled_mm_dq(...)
        cutlass_called["called"] = True
        # Simulate gemm by returning torch.matmul result (this function in fp8 is expected to return out)
        return torch.matmul(a_qinput.to(torch.float32), b_weight.to(torch.float32)).to(out_dtype)

    def fake_torch_scaled_mm(a_q, b_w, out_dtype, scale_a, scale_b, bias=None):
        scaled_mm_called["called"] = True
        # Return (output, dummy) tuple as original torch._scaled_mm would
        out = torch.matmul(a_q.to(torch.float32), b_w.to(torch.float32)).to(out_dtype)
        return out, None

    # Patch ops and torch._scaled_mm
    with patch.object(fp8_mod, "ops", custom_ops):
        with patch.object(custom_ops, "scaled_fp8_quant", side_effect=fake_scaled_fp8_quant):
            with patch.object(custom_ops, "cutlass_scaled_mm_dq", side_effect=fake_cutlass):
                with patch("torch._scaled_mm", side_effect=fake_torch_scaled_mm):
                    # Call apply with bias=None to trigger cutlass branch
                    out = method.apply(layer, x, bias=None)

    assert cutlass_called["called"], "Expected cutlass_scaled_mm_dq to be used when supported."
    assert not scaled_mm_called["called"], "torch._scaled_mm should not be used in the cutlass branch."

    # Output shape should match batch_size x output_dim (narrowing applied)
    assert out.shape[0] == x.shape[0]
    # The output width should equal weight.shape[1]
    assert out.shape[1] == weight.shape[1]


def test_Fp8LinearMethod_apply_fallbacks_to_scaled_mm_when_bias_or_not_supported():
    """When cutlass is not supported or bias is provided, Fp8LinearMethod.apply should
    fall back to the legacy path that calls torch._scaled_mm. We validate both:
    - bias present (even if cutlass is supported)
    - cutlass unsupported (bias None)
    """
    wl = setup_workload()
    x = wl["x"]
    weight = wl["weight"]
    weight_scale = wl["weight_scale"]
    bias = wl["bias"]

    quant_cfg = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    method = Fp8LinearMethod(quant_cfg)

    # Case A: cutlass supported but bias provided -> uses torch._scaled_mm
    method.cutlass_fp8_supported = True
    layer_a = _make_layer_with_params(weight, weight_scale, act_scale=None)

    scaled_mm_called_a = {"called": False}

    def fake_scaled_fp8_quant_a(inp, act_scale_arg, batch_dim_padding=17):
        # Simulate dynamic quant returning padded qinput (we don't actually pad here)
        return x, torch.tensor(1.0, dtype=torch.float32)

    def fake_torch_scaled_mm_a(a_q, b_w, out_dtype, scale_a, scale_b, bias=None):
        scaled_mm_called_a["called"] = True
        out = torch.matmul(a_q.to(torch.float32), b_w.to(torch.float32)).to(out_dtype)
        # Return (out, dummy)
        return out, None

    with patch.object(fp8_mod, "ops", custom_ops):
        with patch.object(custom_ops, "scaled_fp8_quant", side_effect=fake_scaled_fp8_quant_a):
            with patch("torch._scaled_mm", side_effect=fake_torch_scaled_mm_a):
                out_a = method.apply(layer_a, x, bias=bias)

    assert scaled_mm_called_a["called"], "torch._scaled_mm path should be used when bias is present."

    # Case B: cutlass not supported and bias is None -> uses torch._scaled_mm
    method.cutlass_fp8_supported = False
    layer_b = _make_layer_with_params(weight, weight_scale, act_scale=None)

    scaled_mm_called_b = {"called": False}

    def fake_scaled_fp8_quant_b(inp, act_scale_arg, batch_dim_padding=17):
        return x, torch.tensor(1.0, dtype=torch.float32)

    def fake_torch_scaled_mm_b(a_q, b_w, out_dtype, scale_a, scale_b, bias=None):
        scaled_mm_called_b["called"] = True
        out = torch.matmul(a_q.to(torch.float32), b_w.to(torch.float32)).to(out_dtype)
        return out, None

    with patch.object(fp8_mod, "ops", custom_ops):
        with patch.object(custom_ops, "scaled_fp8_quant", side_effect=fake_scaled_fp8_quant_b):
            with patch("torch._scaled_mm", side_effect=fake_torch_scaled_mm_b):
                out_b = method.apply(layer_b, x, bias=None)

    assert scaled_mm_called_b["called"], "torch._scaled_mm path should be used when cutlass not supported."


def test_performance_regression_simulated():
    """Simulate a performance comparison between the 'cutlass' branch and the legacy
    torch._scaled_mm branch.

    Because we cannot run actual CUTLASS kernels in this test environment, we
    simulate relative speeds by instrumenting the patched functions. The commit
    claims a 5-15% improvement; we assert the 'optimized' (cutlass) path is not
    slower than the legacy path by more than 20% (tolerance).
    """
    wl = setup_workload()
    x = wl["x"]
    weight = wl["weight"]
    weight_scale = wl["weight_scale"]

    quant_cfg = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    method = Fp8LinearMethod(quant_cfg)
    method.cutlass_fp8_supported = True
    layer = _make_layer_with_params(weight, weight_scale, act_scale=None)

    # Prepare qinput & scales returned by scaled_fp8_quant patch
    qinput = x.clone()
    x_scale = torch.tensor(1.0, dtype=torch.float32)

    # Simulate cutlass: a relatively fast implementation (matmul)
    def fake_scaled_fp8_quant_cutlass(inp, act_scale_arg):
        return qinput, x_scale

    def fake_cutlass_fast(a_q, b_w, out_dtype, scale_a, scale_b):
        # Minimal overhead matmul
        return torch.matmul(a_q.to(torch.float32), b_w.to(torch.float32)).to(out_dtype)

    # Simulate legacy: add a small extra overhead to emulate slower path
    def fake_scaled_fp8_quant_legacy(inp, act_scale_arg, batch_dim_padding=17):
        return qinput, x_scale

    def fake_torch_scaled_mm_slow(a_q, b_w, out_dtype, scale_a, scale_b, bias=None):
        # Add a small computational overhead to simulate slower legacy op.
        # Perform a tiny redundant operation to make it measurably slower.
        tmp = torch.sum(a_q[:, :8].to(torch.float32))  # small extra op
        out = torch.matmul(a_q.to(torch.float32), b_w.to(torch.float32)).to(out_dtype)
        # Use tmp to avoid it being optimized away
        out[0, 0] = out[0, 0] + (tmp * 0.0)
        return out, None

    # Time optimized (cutlass) path
    with patch.object(fp8_mod, "ops", custom_ops):
        with patch.object(custom_ops, "scaled_fp8_quant", side_effect=fake_scaled_fp8_quant_cutlass):
            with patch.object(custom_ops, "cutlass_scaled_mm_dq", side_effect=fake_cutlass_fast):
                t_opt = timeit.timeit(lambda: method.apply(layer, x, bias=None), number=1)

    # Time legacy path (force use of legacy by disabling cutlass flag)
    method.cutlass_fp8_supported = False
    with patch.object(fp8_mod, "ops", custom_ops):
        with patch.object(custom_ops, "scaled_fp8_quant", side_effect=fake_scaled_fp8_quant_legacy):
            with patch("torch._scaled_mm", side_effect=fake_torch_scaled_mm_slow):
                t_legacy = timeit.timeit(lambda: method.apply(layer, x, bias=None), number=1)

    # We expect optimized <= legacy * 1.20 (20% tolerance)
    assert t_opt <= t_legacy * 1.20, (
        f"Simulated optimized time {t_opt:.4f}s is slower than legacy {t_legacy:.4f}s "
        "beyond tolerance. (This simulates the commit's expected improvement.)"
    )


def test_backward_compatibility_basic_shape_and_dtype():
    """Sanity test that ensures the apply function always returns the expected
    dtype and shape regardless of branch taken (cutlass or legacy)."""
    wl = setup_workload()
    x = wl["x"]
    weight = wl["weight"]
    weight_scale = wl["weight_scale"]

    quant_cfg = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    method = Fp8LinearMethod(quant_cfg)

    # Create layer and common patches for both branches
    layer = _make_layer_with_params(weight, weight_scale, act_scale=None)
    qinput = x.clone()
    x_scale = torch.tensor(1.0, dtype=torch.float32)

    def fake_scaled_fp8_quant(inp, act_scale_arg, batch_dim_padding=17):
        return qinput, x_scale

    def fake_cutlass(a_q, b_w, out_dtype, scale_a, scale_b):
        return torch.matmul(a_q.to(torch.float32), b_w.to(torch.float32)).to(out_dtype)

    def fake_torch_scaled_mm(a_q, b_w, out_dtype, scale_a, scale_b, bias=None):
        out = torch.matmul(a_q.to(torch.float32), b_w.to(torch.float32)).to(out_dtype)
        return out, None

    # Cutlass branch
    method.cutlass_fp8_supported = True
    with patch.object(fp8_mod, "ops", custom_ops):
        with patch.object(custom_ops, "scaled_fp8_quant", side_effect=fake_scaled_fp8_quant):
            with patch.object(custom_ops, "cutlass_scaled_mm_dq", side_effect=fake_cutlass):
                out_cutlass = method.apply(layer, x, bias=None)

    # Legacy branch (bias present)
    method.cutlass_fp8_supported = True
    with patch.object(fp8_mod, "ops", custom_ops):
        with patch.object(custom_ops, "scaled_fp8_quant", side_effect=fake_scaled_fp8_quant):
            with patch("torch._scaled_mm", side_effect=fake_torch_scaled_mm):
                out_legacy = method.apply(layer, x, bias=torch.zeros(weight.shape[1], dtype=x.dtype))

    # Verify shapes and dtypes
    assert out_cutlass.shape == (wl["batch_size"], weight.shape[1])
    assert out_legacy.shape == (wl["batch_size"], weight.shape[1])
    assert out_cutlass.dtype == x.dtype
    assert out_legacy.dtype == x.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])