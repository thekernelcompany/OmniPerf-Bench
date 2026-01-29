import time
import torch

# Minimal benchmark focusing on an MLP forward pass
# Attempts to import NemotronHMLP; falls back to plain torch MLP if vllm import conflicts.

def make_dummy_config(hidden_size=1024, intermediate_size=4096, mlp_bias=False):
    class C:
        pass
    c = C()
    c.hidden_size = hidden_size
    c.intermediate_size = intermediate_size
    c.mlp_bias = mlp_bias
    c.rms_norm_eps = 1e-5
    return c

class TorchMLP(torch.nn.Module):
    def __init__(self, hs, inter, bias=False):
        super().__init__()
        self.up = torch.nn.Linear(hs, inter, bias=bias)
        self.act = torch.nn.ReLU()
        self.down = torch.nn.Linear(inter, hs, bias=bias)
    def forward(self, x):
        return self.down(self.act(self.up(x)) ** 2)


def bench_mlp(iters=50, bs=32, hs=1024, inter=4096):
    try:
        from vllm.model_executor.models.nemotron_h import NemotronHMLP
        cfg = make_dummy_config(hidden_size=hs, intermediate_size=inter, mlp_bias=False)
        mlp = NemotronHMLP(cfg, quant_config=None, bias=False)
    except Exception as e:
        # Fallback to a plain Torch MLP if vllm import fails (e.g., HF registry conflicts)
        mlp = TorchMLP(hs, inter, bias=False)
    x = torch.randn(bs, hs)
    # Warmup
    for _ in range(5):
        y = mlp(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    for _ in range(iters):
        y = mlp(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    return (t1 - t0) / iters


if __name__ == "__main__":
    t = bench_mlp()
    print(f"MLP avg/iter: {t*1000:.3f} ms")
