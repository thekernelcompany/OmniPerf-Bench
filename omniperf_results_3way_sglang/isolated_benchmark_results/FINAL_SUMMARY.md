# SGLang 3-Way Benchmark Results (H100)

## Successful Benchmarks

| Commit | PR | Baseline (tok/s) | Human (tok/s) | Speedup |
|--------|-----|------------------|---------------|---------|
| 187b85b7 | #7393 | 418 | 2055 | **4.9x** |
| 6b231325 | #6649 | 611 | 4276 | **7.0x** |
| c087ddd6 | #6627 | 2584 | 4292 | **1.7x** |
| dd1012fc | #6764 | 3098 | 4263 | **1.4x** |
| da47621c | #7058 | 3081 | 3108 | 1.0x |
| e3ec6bf4 | #6814 | 3063 | 3104 | 1.0x |
| df7f61ee | #6812 | 4233 | 3064 | 0.7x (regression) |
| 2a754e57 | #579 | (done separately) | - | - |

## Failed Benchmarks

- 148254d4: install_failed
- 2a413829: checkout failed
- 2bd18e2d: failed
- 4418f599: fp8_quant error
- 5e023301: install_failed
- 6cb00c63: install_failed
- 880221bd: git lock / failed
- b1e5a33a: install_failed
- ddcf9fe3: failed

## Configuration

- GPU: NVIDIA H100 PCIe (SM90)
- PyTorch: 2.5.1+cu124
- Triton: 3.3.0
- Model: TinyLlama-1.1B-Chat-v1.0
- Batch: 4, Input: 512, Output: 64

