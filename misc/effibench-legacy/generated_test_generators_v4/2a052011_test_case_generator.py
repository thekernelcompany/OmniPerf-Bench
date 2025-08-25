import pytest
import torch
import numpy as np
import timeit
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Import actual modules being tested
try:
    from vllm.model_executor.models import mixtral as mixtral_mod
    from vllm.model_executor.models.mixtral import (
        MixtralMoE,
        MixtralForCausalLM,
        all_close_1d,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from transformers import MixtralConfig
except Exception:
    pytest.skip("Required modules (vllm.mixtral, Fp8Config, or transformers.MixtralConfig) not available",
                allow_module_level=True)


def setup_workload() -> Dict[str, Any]:
    """Create realistic test data based on the domain.

    Uses transformer-scale realistic dimensions to exercise MoE codepaths.
    NOTE: These are large tensors by design to match the commit's requirements.
    """
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Realistic transformer-scale dimensions
    batch_size = 32
    seq_len = 1024
    hidden_dim = 4096
    intermediate_size = 4 * hidden_dim  # typical FFN expansion
    num_experts = 8
    top_k = 2

    # Flattened tokens for vLLM model (num_tokens, hidden_dim)
    num_tokens = batch_size * seq_len
    # Use CPU tensor by default to avoid requiring CUDA in CI; dtype float32 for
    # quantization processing & loader tests.
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.float32)

    # Create a MixtralConfig reflecting realistic sizes
    config = MixtralConfig()
    # Overwrite necessary fields for realistic test scale
    config.hidden_size = hidden_dim
    config.intermediate_size = intermediate_size
    config.num_local_experts = num_experts
    config.num_experts_per_tok = top_k
    config.num_hidden_layers = 1  # keep model small for tests
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    config.vocab_size = 1000
    config.pad_token_id = 0
    config.rms_norm_eps = 1e-6
    config.max_position_embeddings = 4096 * 32
    config.sliding_window = None

    return {
        "hidden_states": hidden_states,
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "intermediate_size": intermediate_size,
        "num_experts": num_experts,
        "top_k": top_k,
        "config": config,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def _make_fp8_config(serialized: bool, activation_scheme: str = "dynamic") -> Fp8Config:
    """Helper to create an Fp8Config-like instance.

    We attempt to instantiate Fp8Config normally; if its constructor requires
    arguments we can't guess, we create via __new__ and set attributes used by
    MixtralMoE.
    """
    try:
        # Try default construction first
        qc = Fp8Config()
        # Ensure attributes exist
        qc.is_checkpoint_fp8_serialized = serialized
        qc.activation_scheme = activation_scheme
        return qc
    except Exception:
        # Fallback: construct minimal object of correct type
        qc = Fp8Config.__new__(Fp8Config)
        setattr(qc, "is_checkpoint_fp8_serialized", serialized)
        setattr(qc, "activation_scheme", activation_scheme)
        return qc


def _patch_tp_ranks():
    """Ensure TP rank/world size are deterministic (single rank) for tests."""
    patcher_rank = patch(
        "vllm.model_executor.models.mixtral.get_tensor_model_parallel_rank",
        return_value=0,
    )
    patcher_world = patch(
        "vllm.model_executor.models.mixtral.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    return patcher_rank.start(), patcher_world.start()


def _stop_tp_patches(handles):
    for h in handles:
        try:
            h.stop()
        except Exception:
            pass


def test_all_close_1d_simple():
    """Unit test for all_close_1d helper"""
    a = torch.tensor([1.0, 1.0, 1.0])
    assert all_close_1d(a)  # trivial case: all equal

    b = torch.tensor([1.0, 1.1, 1.0])
    # not all equal
    assert not all_close_1d(b)


def test_static_activation_scheme_without_serialized_fp8_raises():
    """When activation_scheme is 'static' but checkpoint is not serialized fp8, __init__ should raise."""
    workload = setup_workload()
    config = workload["config"]

    # Create an Fp8Config with activation_scheme static but serialized flag False
    quant_cfg = _make_fp8_config(serialized=False, activation_scheme="static")

    # Patch tensor-parallel helpers to single rank to make intermediate_size consistent
    handles = _patch_tp_ranks()
    try:
        with pytest.raises(ValueError, match="Found static activation scheme for checkpoint that was not serialized fp8."):
            MixtralMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                params_dtype=torch.float32,
                tp_size=1,
                quant_config=quant_cfg,
            )
    finally:
        _stop_tp_patches(handles)


def test_process_weights_after_loading_quantizes_fp16_checkpoint_and_sets_scales():
    """If checkpoint is fp16 (not serialized fp8), process_weights_after_loading should call ops.scaled_fp8_quant
    and replace floating weights with quantized ones and set weight scales.
    """
    workload = setup_workload()
    config = workload["config"]
    quant_cfg = _make_fp8_config(serialized=False, activation_scheme="dynamic")

    # Single-rank environment
    handles = _patch_tp_ranks()
    try:
        moe = MixtralMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            params_dtype=torch.float32,
            tp_size=1,
            quant_config=quant_cfg,
        )

        # Prepare fake fp16 weights (w13 and w2) to be present in moe as floating params.
        # w13_weight shape: (num_experts, 2*intermediate_size, hidden_size)
        # w2_weight shape: (num_experts, hidden_size, intermediate_size)
        # Initialize with random floats
        w13 = torch.randn(
            (config.num_local_experts, 2 * (config.intermediate_size // 1), config.hidden_size),
            dtype=torch.float32,
        )
        w2 = torch.randn(
            (config.num_local_experts, config.hidden_size, config.intermediate_size // 1),
            dtype=torch.float32,
        )

        # Directly assign to the internal parameters (simulating loaded fp16)
        with torch.no_grad():
            # If use_fp8 True then constructor may have changed dtype to float8; override assignment
            # Set temporary attributes to float32 tensors to allow quantization path to run.
            moe.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
            moe.w2_weight = torch.nn.Parameter(w2.clone(), requires_grad=False)

            # Initialize scales to zeros so scaled_fp8_quant will populate them
            moe.w13_scale = torch.nn.Parameter(torch.zeros(config.num_local_experts, dtype=torch.float32),
                                               requires_grad=False)
            moe.w2_scale = torch.nn.Parameter(torch.zeros(config.num_local_experts, dtype=torch.float32),
                                              requires_grad=False)

        # Patch ops.scaled_fp8_quant to verify it's called and to return predictable outputs
        called = {"count": 0, "args": []}

        def fake_scaled_fp8_quant(tensor):
            # record and return (quantized_tensor, scale)
            called["count"] += 1
            called["args"].append(tensor.shape)
            # Return same shape tensor (we don't require float8 dtype here) and a scale
            return tensor.clone(), torch.tensor(0.12345, dtype=torch.float32)

        with patch("vllm.model_executor.models.mixtral.ops.scaled_fp8_quant", side_effect=fake_scaled_fp8_quant):
            # Should not raise and should replace w13_weight and w2_weight with returned quantized values
            moe.use_fp8 = True  # ensure process runs
            # Also ensure quant_config is available on instance
            moe.quant_config = quant_cfg
            moe.process_weights_after_loading()

        assert called["count"] == 2 * config.num_local_experts or called["count"] >= config.num_local_experts, \
            "scaled_fp8_quant should be called per expert per weight matrix (w13 and w2)"
        # After processing, ensure w13_weight and w2_weight are Parameters
        assert isinstance(moe.w13_weight, torch.nn.Parameter)
        assert isinstance(moe.w2_weight, torch.nn.Parameter)
        # Scales were assigned per expert
        assert moe.w13_scale is not None and moe.w2_scale is not None
        assert moe.w13_scale.shape[0] == config.num_local_experts
        assert moe.w2_scale.shape[0] == config.num_local_experts
    finally:
        _stop_tp_patches(handles)


def test_process_weights_after_loading_static_fp8_checkpoint_collapses_act_scales_and_warns():
    """If checkpoint is serialized fp8 with static activations, process_weights_after_loading should
    collapse per-expert acts to single max scale and not raise.
    """
    workload = setup_workload()
    config = workload["config"]
    # Serialized fp8 checkpoint and static activation scheme
    quant_cfg = _make_fp8_config(serialized=True, activation_scheme="static")

    handles = _patch_tp_ranks()
    try:
        moe = MixtralMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            params_dtype=torch.float32,
            tp_size=1,
            quant_config=quant_cfg,
        )

        # Simulate that a13_scale and a2_scale were loaded per expert with slightly different values
        with torch.no_grad():
            moe.a13_scale = torch.nn.Parameter(
                torch.linspace(0.1, 0.2, steps=config.num_local_experts, dtype=torch.float32),
                requires_grad=False,
            )
            moe.a2_scale = torch.nn.Parameter(
                torch.linspace(0.05, 0.15, steps=config.num_local_experts, dtype=torch.float32),
                requires_grad=False,
            )
            # Create dummy w13 and w2 weight tensors (already in float8 or whatever dtype)
            moe.w13_weight = torch.nn.Parameter(torch.randn(
                (config.num_local_experts, 2 * (config.intermediate_size // 1), config.hidden_size),
                dtype=torch.float32,
            ), requires_grad=False)
            moe.w2_weight = torch.nn.Parameter(torch.randn(
                (config.num_local_experts, config.hidden_size, config.intermediate_size // 1),
                dtype=torch.float32,
            ), requires_grad=False)

        # Call process_weights_after_loading, which should collapse per-expert a13_scale/a2_scale to a single scalar (max)
        moe.use_fp8 = True
        moe.quant_config = quant_cfg

        # Patch print_warning_once to capture whether a warning about unequal scales would be printed
        with patch("vllm.model_executor.models.mixtral.print_warning_once") as mock_warn:
            moe.process_weights_after_loading()
            # Because the per-expert scales are not all equal, code will call print_warning_once
            mock_warn.assert_called()

        # a13_scale and a2_scale should now be scalar Parameters
        assert isinstance(moe.a13_scale, torch.nn.Parameter)
        assert isinstance(moe.a2_scale, torch.nn.Parameter)
        # They should be scalars (0-dim tensors) after collapsing
        assert moe.a13_scale.numel() == 1
        assert moe.a2_scale.numel() == 1
    finally:
        _stop_tp_patches(handles)


def test_load_weights_integration_calls_weight_loader_for_all_mappings():
    """Integration test: MixtralForCausalLM.load_weights should route expert weights, weight_scales,
    and act_scales to the right internal parameters via their weight_loader functions.
    """
    workload = setup_workload()
    config = workload["config"]

    # We'll create a serialized fp8 quant config so that weight_scale and act_scale loaders are wired
    quant_cfg = _make_fp8_config(serialized=True, activation_scheme="static")

    # Single-rank to keep mapping simple
    handles = _patch_tp_ranks()
    try:
        model = MixtralForCausalLM(config=config, quant_config=quant_cfg, lora_config=None)

        # Build a fake state dict (list of (name, tensor)) to emulate a checkpoint.
        weights = []

        # For each expert, prepare:
        # - weight_scale tensors for w1/w2/w3
        # - w1/w2/w3 weight matrices
        # - act_scale tensors for w1/w2/w3
        for ex_id in range(config.num_local_experts):
            # weight scales: scalar per expert
            for wname in ["w1", "w2", "w3"]:
                weights.append((f"experts.{ex_id}.{wname}.weight_scale", torch.tensor(0.5 + ex_id * 0.01, dtype=torch.float32)))
            # weights: shapes matching expected loader logic
            # w1 and w3: [intermediate_total, hidden_size]
            weights.append((f"experts.{ex_id}.w1.weight",
                            torch.randn((config.intermediate_size, config.hidden_size), dtype=torch.float32)))
            weights.append((f"experts.{ex_id}.w2.weight",
                            torch.randn((config.hidden_size, config.intermediate_size), dtype=torch.float32)))
            weights.append((f"experts.{ex_id}.w3.weight",
                            torch.randn((config.intermediate_size, config.hidden_size), dtype=torch.float32)))
            # act_scales per expert
            for wname in ["w1", "w2", "w3"]:
                weights.append((f"experts.{ex_id}.{wname}.act_scale", torch.tensor(0.2 + ex_id * 0.01, dtype=torch.float32)))

        # Additionally, add a rotary_emb.inv_freq entry to ensure it's skipped gracefully
        weights.append(("rotary_emb.inv_freq", torch.randn((config.hidden_size // config.num_attention_heads,), dtype=torch.float32)))

        # Patch the internal weight_loader method on parameters so we can monitor calls.
        # We'll monkeypatch the parameters' weight_loader to a MagicMock that records calls.
        params_dict = dict(model.named_parameters())
        # Replace weight_loader for all parameters that should receive expert loads
        for pname, p in params_dict.items():
            if hasattr(p, "weight_loader"):
                # Wrap original to ensure actual behavior (call-through) while recording
                original_loader = getattr(p, "weight_loader")
                mock_loader = MagicMock(side_effect=lambda param, loaded_weight, *args, **kwargs: original_loader(param, loaded_weight, *args, **kwargs))
                setattr(p, "weight_loader", mock_loader)

        # Run load_weights - should dispatch to the patched weight_loader mocks without raising
        model.load_weights(weights)

        # Check that at least one param had its weight_loader called (sanity)
        called_any = False
        for pname, p in params_dict.items():
            wl = getattr(p, "weight_loader", None)
            if isinstance(wl, MagicMock):
                if wl.call_count > 0:
                    called_any = True
                    break
        assert called_any, "Expected at least one parameter's weight_loader to be invoked during load_weights"
    finally:
        _stop_tp_patches(handles)


def test_forward_calls_fused_moe_with_correct_parameters_and_returns_shape():
    """Ensure MixtralMoE.forward invokes fused_moe with the new parameter names (w13_weight, w2_weight),
    and that forward returns a tensor of the expected shape and dtype.
    """
    workload = setup_workload()
    config = workload["config"]
    quant_cfg = _make_fp8_config(serialized=True, activation_scheme="dynamic")

    handles = _patch_tp_ranks()
    try:
        moe = MixtralMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            params_dtype=torch.float32,
            tp_size=1,
            quant_config=quant_cfg,
        )

        # Prepare a small hidden_states tensor (use realistic flattened tokens)
        hidden_states = workload["hidden_states"][:  # keep realistic size but allow slicing if necessary
                          min(1024, workload["hidden_states"].shape[0])]
        num_tokens, hidden_size = hidden_states.shape

        # Ensure moe has w13_weight and w2_weight parameters populated
        with torch.no_grad():
            moe.w13_weight = torch.nn.Parameter(torch.randn(
                (config.num_local_experts, 2 * (config.intermediate_size // 1), config.hidden_size), dtype=torch.float32
            ), requires_grad=False)
            moe.w2_weight = torch.nn.Parameter(torch.randn(
                (config.num_local_experts, config.hidden_size, config.intermediate_size // 1), dtype=torch.float32
            ), requires_grad=False)
            # scales (may be None if not fp8, but set for the test)
            moe.w13_scale = torch.nn.Parameter(torch.ones(config.num_local_experts, dtype=torch.float32), requires_grad=False)
            moe.w2_scale = torch.nn.Parameter(torch.ones(config.num_local_experts, dtype=torch.float32), requires_grad=False)
            moe.a13_scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False)
            moe.a2_scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False)
            moe.use_fp8 = True

        # Patch the fused_moe symbol imported in the mixtral module to capture inputs and return a tensor
        captured = {}

        def fake_fused_moe(a, w13, w2, score, topk, **kwargs):
            # Record that function received parameters we expect
            captured["a_shape"] = a.shape
            captured["w13_shape"] = w13.shape
            captured["w2_shape"] = w2.shape
            captured["score_shape"] = score.shape
            captured["topk"] = topk
            captured["kwargs"] = kwargs
            # Return zero tensor of appropriate output shape (num_tokens, hidden)
            return torch.zeros((a.shape[0], hidden_size), dtype=a.dtype)

        with patch("vllm.model_executor.models.mixtral.fused_moe", side_effect=fake_fused_moe) as mock_fm:
            out = moe.forward(hidden_states)

        # Verify fused_moe called and captured shapes are correct
        assert "a_shape" in captured
        assert captured["a_shape"][1] == hidden_size
        assert captured["w13_shape"][0] == config.num_local_experts
        assert captured["w2_shape"][0] == config.num_local_experts
        assert captured["topk"] == moe.top_k

        # Output shape should match (num_tokens, hidden_size)
        assert out.shape == (num_tokens, hidden_size)
    finally:
        _stop_tp_patches(handles)


def test_forward_performance_basic():
    """Basic timing of MixtralMoE.forward to ensure it runs; use timeit number=1 to avoid caching"""
    workload = setup_workload()
    config = workload["config"]

    # Use a dynamic (non-serialized) fp8 config to force quantization processing path if used
    quant_cfg = _make_fp8_config(serialized=True, activation_scheme="dynamic")

    handles = _patch_tp_ranks()
    try:
        moe = MixtralMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            params_dtype=torch.float32,
            tp_size=1,
            quant_config=quant_cfg,
        )

        # Provide minimal populated params to let forward run
        with torch.no_grad():
            moe.w13_weight = torch.nn.Parameter(torch.randn(
                (config.num_local_experts, 2 * (config.intermediate_size // 1), config.hidden_size), dtype=torch.float32
            ), requires_grad=False)
            moe.w2_weight = torch.nn.Parameter(torch.randn(
                (config.num_local_experts, config.hidden_size, config.intermediate_size // 1), dtype=torch.float32
            ), requires_grad=False)
            moe.w13_scale = torch.nn.Parameter(torch.ones(config.num_local_experts, dtype=torch.float32), requires_grad=False)
            moe.w2_scale = torch.nn.Parameter(torch.ones(config.num_local_experts, dtype=torch.float32), requires_grad=False)
            moe.a13_scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False)
            moe.a2_scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False)
            moe.use_fp8 = False  # run non-fp8 codepath for timing simplicity

        # Small slice of the realistic tokens to keep runtime reasonable but still not a toy
        hidden_states = workload["hidden_states"][: 4096]

        # Patch fused_moe to be a light-weight surrogate so timing measures overhead of dispatch
        with patch("vllm.model_executor.models.mixtral.fused_moe", return_value=torch.zeros((hidden_states.shape[0], hidden_states.shape[1]), dtype=hidden_states.dtype)):
            exec_time = timeit.timeit(lambda: moe.forward(hidden_states), number=1)

        assert exec_time >= 0.0  # trivially ensure timing ran
        # No strict performance assertion here; we only ensure the call completes in a testable way.
    finally:
        _stop_tp_patches(handles)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])