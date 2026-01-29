# SPDX-License-Identifier: Apache-2.0
import torch
from torch import Generator

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

DEVICE = "cpu"

BATCH_SIZE = 64
VOCAB_SIZE = 8192


def test_topk_impl_equivalance():
    with torch.device(DEVICE):
        generator = Generator(device=DEVICE).manual_seed(33)

        logits = torch.rand((BATCH_SIZE, VOCAB_SIZE), generator=generator)

        # Random top-k values between 1 and 64.
        k = torch.randint(1, 65, (BATCH_SIZE,), generator=generator)

        # Reference: mask all but top-k using topk + scatter
        topk_values, topk_indices = torch.topk(logits, k, dim=-1)
        ref = torch.empty_like(logits)
        ref.fill_(-float("inf"))
        ref.scatter_(-1, topk_indices, topk_values)

        # Under test: our optimized apply_top_k_top_p
        out = apply_top_k_top_p(logits, k, None)

        assert torch.equal(ref.isfinite(), out.isfinite())
        # For finite entries (top-k), values must match
        mask = ref.isfinite()
        assert torch.allclose(ref[mask], out[mask])
