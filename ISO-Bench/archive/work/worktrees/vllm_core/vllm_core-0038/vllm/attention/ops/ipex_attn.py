"""
Optional CPU-optimized attention ops.

This module provides a thin wrapper around torch.nn.functional.scaled_dot_product_attention
that ensures inputs are contiguous and leverages any available CPU optimizations
(e.g., IPEX) when installed. If IPEX is not available, this simply falls back to
PyTorch's implementation with minimal overhead.
"""
from typing import Optional

import torch
from torch.nn.functional import scaled_dot_product_attention as _sdpa

try:  # optional dependency
    import intel_extension_for_pytorch as ipex  # noqa: F401
    _HAS_IPEX = True
except Exception:  # pragma: no cover
    _HAS_IPEX = False


def _maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def sdpa_optimized(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    # Ensure layouts are friendly for kernels
    q = _maybe_contiguous(query)
    k = _maybe_contiguous(key)
    v = _maybe_contiguous(value)

    # If IPEX is present and running on CPU, the underlying kernels may be optimized.
    # We don't need special handling here; enabling IPEX generally patches PyTorch ops.
    return _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                 is_causal=is_causal, scale=scale)
