import os
import time
import math
import random
import os
import torch
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import importlib.util
from importlib.machinery import SourceFileLoader
utils_path = os.path.join(ROOT, 'vllm', 'model_executor', 'layers', 'utils.py')
utils_mod = SourceFileLoader('utils_local', utils_path).load_module()
apply_penalties = utils_mod.apply_penalties


def bench_once(num_seqs=64, vocab_size=32000, prompt_len=128, output_len=64, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logits = torch.randn(num_seqs, vocab_size, device=device, dtype=torch.float32)

    pad_id = vocab_size
    prompt = torch.randint(0, vocab_size, (num_seqs, prompt_len), device=device, dtype=torch.long)
    output = torch.randint(0, vocab_size, (num_seqs, output_len), device=device, dtype=torch.long)

    if prompt_len > 0:
        mask = torch.rand(num_seqs, prompt_len, device=device) < 0.1
        prompt[mask] = pad_id
    if output_len > 0:
        mask = torch.rand(num_seqs, output_len, device=device) < 0.1
        output[mask] = pad_id

    presence = torch.rand(num_seqs, device=device, dtype=torch.float32)
    frequency = torch.rand(num_seqs, device=device, dtype=torch.float32)
    repetition = 0.5 + torch.rand(num_seqs, device=device, dtype=torch.float32) * 1.5

    for _ in range(5):
        apply_penalties(logits.clone(), prompt, output, presence, frequency, repetition)

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    iters = 30
    for _ in range(iters):
        apply_penalties(logits.clone(), prompt, output, presence, frequency, repetition)
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return t1 - t0


def main():
    configs = [
        (32, 16384, 64, 32),
        (64, 32000, 128, 64),
        (8, 50000, 256, 64),
    ]
    for cfg in configs:
        dt = bench_once(*cfg)
        print(f"config={cfg} time_s={dt:.6f}")


if __name__ == '__main__':
    main()
