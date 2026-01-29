# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    import flashinfer.sampling
    is_flashinfer_available = True
except ImportError:
    is_flashinfer_available = False


class TopKTopPSampler(nn.Module):

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda():
            if is_flashinfer_available:
                flashinfer_version = flashinfer.__version__
                if flashinfer_version >= "0.2.3":
                    # FIXME(DefTruth): Currently, we have errors when using
                    # FlashInfer>=v0.2.3 for top-p & top-k sampling. As a
                    # workaround, we disable FlashInfer for top-p & top-k
                    # sampling by default while FlashInfer>=v0.2.3.
                    # The sampling API removes the success return value
                    # of all sampling API, which is not compatible with
                    # earlier design.
                    # https://github.com/flashinfer-ai/flashinfer/releases/
                    # tag/v0.2.3
                    logger.info(
                        "Currently, FlashInfer top-p & top-k sampling sampler "
                        "is disabled because FlashInfer>=v0.2.3 is not "
                        "backward compatible. Falling back to the PyTorch-"
                        "native implementation of top-p & top-k sampling.")
                    self.forward = self.forward_native
                elif envs.VLLM_USE_FLASHINFER_SAMPLER is not False:
                    # NOTE(woosuk): The V0 sampler doesn't use FlashInfer for
                    # sampling unless VLLM_USE_FLASHINFER_SAMPLER=1 (i.e., by
                    # default it is unused). For backward compatibility, we set
                    # `VLLM_USE_FLASHINFER_SAMPLER` as None by default and
                    # interpret it differently in V0 and V1 samplers: In V0,
                    # None means False, while in V1, None means True. This is
                    # why we use the condition
                    # `envs.VLLM_USE_FLASHINFER_SAMPLER is not False` here.
                    logger.info("Using FlashInfer for top-p & top-k sampling.")
                    self.forward = self.forward_cuda
                else:
                    logger.warning(
                        "FlashInfer is available, but it is not enabled. "
                        "Falling back to the PyTorch-native implementation of "
                        "top-p & top-k sampling. For the best performance, "
                        "please set VLLM_USE_FLASHINFER_SAMPLER=1.")
                    self.forward = self.forward_native
            else:
                logger.warning(
                    "FlashInfer is not available. Falling back to the PyTorch-"
                    "native implementation of top-p & top-k sampling. For the "
                    "best performance, please install FlashInfer.")
                self.forward = self.forward_native
        elif current_platform.is_tpu():
            if envs.VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION:
                logger.warning(
                    "TPU-specific optimization for top-k & top-p sampling are "
                    "disabled, falling back to PyTorch-native implementation "
                    "which could be very slow.")
                self.forward = self.forward_native
            else:
                self.forward = self.forward_tpu
        else:
            self.forward = self.forward_native

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """PyTorch-native implementation of top-k and top-p sampling."""
        # Fast path: top-k only avoids full sort-based masking
        if k is not None and p is None:
            # Keep only top-k logits, mask others to -inf without building a bool mask
            if torch.is_tensor(k) and (k.dim() == 0 or k.numel() == 1):
                k_int = int(k.item()) if torch.is_tensor(k) else int(k)
                topk_values, topk_indices = torch.topk(logits, k_int, dim=-1)
                masked = torch.empty_like(logits)
                masked.fill_(-float("inf"))
                masked.scatter_(-1, topk_indices, topk_values)
            else:
                # Variable k per batch
                if not torch.is_tensor(k):
                    k = torch.tensor(k, device=logits.device)
                k_max = int(k.max().item())
                topk_values, topk_indices = torch.topk(logits, k_max, dim=-1)
                masked = torch.empty_like(logits)
                masked.fill_(-float("inf"))
                ar = torch.arange(k_max, device=logits.device)
                sel = ar.unsqueeze(0) < k.unsqueeze(1)
                rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
                rows = rows.expand_as(sel)
                masked[rows[sel], topk_indices[sel]] = topk_values[sel]
            probs = masked.softmax(dim=-1, dtype=torch.float32)
            return random_sample(probs, generators)
        logits = apply_top_k_top_p(logits, k, p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators)

    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """More optimized implementation for top-k and top-p sampling."""
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        if k is None and p is None:
            # We prefer `random_sample` over `flashinfer_sample` when sorting is
            # not needed. This is because `random_sample` does not require
            # CPU-GPU synchronization while `flashinfer_sample` does.
            return random_sample(probs, generators)
        return flashinfer_sample(probs, k, p, generators)

    def forward_tpu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # If only top-k is specified, use pytorch's builtin topk op. This leads
        # to significant speed up on TPU compared to using apply_top_k_top_p.
        if k is not None and p is None:
            if torch.is_tensor(k) and (k.dim() == 0 or k.numel() == 1):
                k_int = int(k.item()) if torch.is_tensor(k) else int(k)
                topk_values, topk_indices = torch.topk(logits, k_int, dim=-1)
                masked = torch.empty_like(logits)
                masked.fill_(-float('inf'))
                masked.scatter_(-1, topk_indices, topk_values)
                logits = masked
            else:
                # Variable k per batch element
                if not torch.is_tensor(k):
                    k = torch.tensor(k, device=logits.device)
                k_max = int(k.max().item())
                topk_values, topk_indices = torch.topk(logits, k_max, dim=-1)
                masked = torch.empty_like(logits)
                masked.fill_(-float('inf'))
                ar = torch.arange(k_max, device=logits.device)
                sel = ar.unsqueeze(0) < k.unsqueeze(1)
                rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
                rows = rows.expand_as(sel)
                masked[rows[sel], topk_indices[sel]] = topk_values[sel]
                logits = masked
        else:
            # TODO Placeholder for TPU optimized topp kernel
            # logits = apply_top_k_top_p(logits, k, p)
            pass

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    This function sorts the logits tensor, which can be slow for large batches.
    """
    if k is not None and p is None:
        # Fast path for top-k only: avoid global sort and boolean mask
        if torch.is_tensor(k):
            # Variable k per batch element
            if k.dim() == 0 or k.numel() == 1:
                k_int = int(k.item())
                topk_values, topk_indices = torch.topk(logits, k_int, dim=-1)
                out = torch.empty_like(logits)
                out.fill_(-float("inf"))
                out.scatter_(-1, topk_indices, topk_values)
                return out
            k_max = int(k.max().item())
            # Get top-k_max per row once
            topk_values, topk_indices = torch.topk(logits, k_max, dim=-1)
            out = torch.empty_like(logits)
            out.fill_(-float("inf"))
            # Build mask selecting first k[i] entries for each row
            ar = torch.arange(k_max, device=logits.device)
            sel = ar.unsqueeze(0) < k.unsqueeze(1)
            rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
            rows = rows.expand_as(sel)
            out[rows[sel], topk_indices[sel]] = topk_values[sel]
            return out
        else:
            # Scalar k provided
            topk_values, topk_indices = torch.topk(logits, k, dim=-1)
            out = torch.empty_like(logits)
            out.fill_(-float("inf"))
            out.scatter_(-1, topk_indices, topk_values)
            return out

    if k is None and p is None:
        return logits
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1, dtype=torch.float32)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def flashinfer_sample(
    probs: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Sample from the probabilities using FlashInfer.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.
    
    NOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.

    NOTE: This function includes CPU-GPU synchronization, while `random_sample`
    does not. Call this function at the end of the forward pass to minimize
    the synchronization overhead.
    """
    assert not (k is None and p is None)
    max_top_k_round = 32
    batch_size = probs.shape[0]
    uniform_samples = torch.empty((max_top_k_round, batch_size),
                                  device=probs.device)
    if len(generators) != batch_size:
        uniform_samples.uniform_()
    if generators:
        for i, generator in generators.items():
            uniform_samples[:, i].uniform_(generator=generator)

    if k is None:
        # Top-p only.
        next_token_ids, success = flashinfer.sampling.top_p_sampling_from_probs(
            probs, uniform_samples, p, deterministic=True)
    elif p is None:
        # Top-k only.
        next_token_ids, success = flashinfer.sampling.top_k_sampling_from_probs(
            probs, uniform_samples, k, deterministic=True)
    else:
        # Both top-k and top-p.
        next_token_ids, success = (
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, uniform_samples, k, p, deterministic=True))

    # NOTE: CPU-GPU synchronization happens here.
    if not success.all():
        if k is not None:
            probs = flashinfer.sampling.top_k_renorm_prob(probs, k)
        if p is not None:
            probs = flashinfer.sampling.top_p_renorm_prob(probs, p)
        next_token_ids = flashinfer.sampling.sampling_from_probs(
            probs, uniform_samples[0], deterministic=True)
    return next_token_ids.view(-1)
