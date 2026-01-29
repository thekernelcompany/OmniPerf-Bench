# SPDX-License-Identifier: Apache-2.0


# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for FlashInfer API changes.

Users of vLLM should always import **only** these wrappers.
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.util
from typing import Any, Callable, NoReturn

from vllm.logger import init_logger

logger = init_logger(__name__)


@functools.cache
def has_flashinfer() -> bool:
    """Return ``True`` if FlashInfer is available."""
    # Use find_spec to check if the module exists without importing it
    # This avoids potential CUDA initialization side effects
    return importlib.util.find_spec("flashinfer") is not None


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable FlashInfer backend."""
    raise RuntimeError(
        "FlashInfer backend is not available. Please install the package "
        "to enable FlashInfer kernels: "
        "https://github.com/flashinfer-ai/flashinfer")


def _get_submodule(module_name: str) -> Any | None:
    """Safely import a submodule and return it, or None if not available."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


# General lazy import wrapper
def _lazy_import_wrapper(module_name: str,
                         attr_name: str,
                         fallback_fn: Callable[..., Any] = _missing):
    """Create a lazy import wrapper for a specific function."""

    @functools.cache
    def _get_impl():
        if not has_flashinfer():
            return None
        mod = _get_submodule(module_name)
        return getattr(mod, attr_name, None) if mod else None

    def wrapper(*args, **kwargs):
        impl = _get_impl()
        if impl is None:
            return fallback_fn(*args, **kwargs)
        return impl(*args, **kwargs)

    return wrapper


# Create lazy wrappers for each function
flashinfer_cutlass_fused_moe = _lazy_import_wrapper("flashinfer.fused_moe",
                                                    "cutlass_fused_moe")
fp4_quantize = _lazy_import_wrapper("flashinfer", "fp4_quantize")
fp4_swizzle_blockscale = _lazy_import_wrapper("flashinfer",
                                              "fp4_swizzle_blockscale")

# Special case for autotune since it returns a context manager
autotune = _lazy_import_wrapper(
    "flashinfer.autotuner",
    "autotune",
    fallback_fn=lambda *args, **kwargs: contextlib.nullcontext())


# FP8 blockscale support wrappers
fp8_swizzle_blockscale = _lazy_import_wrapper(
    "flashinfer", "fp8_swizzle_blockscale")
flashinfer_fp8_blockscale_fused_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "fp8_blockscale_fused_moe")


@functools.cache
def has_flashinfer_cutlass_fused_moe() -> bool:
    """Return ``True`` if FlashInfer CUTLASS fused MoE is available."""
    if not has_flashinfer():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.fused_moe", "cutlass_fused_moe"),
        ("flashinfer", "fp4_quantize"),
        ("flashinfer", "fp4_swizzle_blockscale"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True

@functools.cache
def has_flashinfer_fp8_blockscale_fused_moe() -> bool:
    """Return ``True`` if FlashInfer FP8 blockscale fused MoE is available."""
    if not has_flashinfer():
        return False
    required_functions = [
        ("flashinfer.fused_moe", "fp8_blockscale_fused_moe"),
        ("flashinfer", "fp8_swizzle_blockscale"),
    ]
    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_moe() -> bool:
    """Return True if any FlashInfer MoE backend is available (CUTLASS FP4 or FP8 blockscale)."""
    if not has_flashinfer():
        return False
    # CUTLASS FP4 path
    if has_flashinfer_cutlass_fused_moe():
        return True
    # FP8 blockscale path (try multiple potential symbol names)
    mod = _get_submodule("flashinfer.fused_moe")
    if not mod:
        return False
    return any(
        hasattr(mod, name)
        for name in ("blockscale_fused_moe_fp8", "fused_moe_blockscale_fp8")
    )


def flashinfer_fused_moe_blockscale_fp8(*args, **kwargs):
    """Dispatch to FlashInfer FP8 blockscale fused MoE if available.

    Tries known symbol names to be resilient to upstream rename.
    """
    mod = _get_submodule("flashinfer.fused_moe") if has_flashinfer() else None
    if mod is not None:
        for name in ("blockscale_fused_moe_fp8", "fused_moe_blockscale_fp8"):
            fn = getattr(mod, name, None)
            if fn is not None:
                return fn(*args, **kwargs)
    return _missing(*args, **kwargs)



__all__ = [
    "has_flashinfer",
    "has_flashinfer_moe",
    "flashinfer_fused_moe_blockscale_fp8",

    "has_flashinfer_cutlass_fused_moe",
    "flashinfer_cutlass_fused_moe",
    "fp4_quantize",
    "fp4_swizzle_blockscale",
    "autotune",
]