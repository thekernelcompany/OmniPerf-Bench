# SPDX-License-Identifier: Apache-2.0
"""Utility methods for model layers."""
from typing import Tuple

import torch


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = tokens.new_zeros((num_seqs, vocab_size + 1))
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts.ne(0)

    return bin_counts, mask


def apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                    output_tokens_tensor: torch.Tensor,
                    presence_penalties: torch.Tensor,
                    frequency_penalties: torch.Tensor,
                    repetition_penalties: torch.Tensor) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts 
        are padded to the maximum prompt length within the batch using 
        `vocab_size` as the padding value. The value `vocab_size` is used 
        for padding because it does not correspond to any valid token ID 
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(prompt_tokens_tensor,
                                                   vocab_size, num_seqs)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)
    mask_any = prompt_mask | output_mask
    rep = repetition_penalties.unsqueeze(1)
    penalties = torch.where(mask_any, rep, 1.0)
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits *= scaling
    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(1).to(logits.dtype) * output_bin_counts.to(logits.dtype)
    logits -= presence_penalties.unsqueeze(1).to(logits.dtype) * output_mask.to(logits.dtype)
    return logits
