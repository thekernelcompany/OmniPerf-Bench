import time
import random
import sys
import types
from pathlib import Path
from typing import List, Dict

import torch

# Create a stub 'vllm' package to avoid running vllm/__init__.py during imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_DIR = REPO_ROOT / "vllm"
if "vllm" not in sys.modules:
    vllm_stub = types.ModuleType("vllm")
    # Mark as a package
    vllm_stub.__path__ = [str(VLLM_DIR)]
    sys.modules["vllm"] = vllm_stub

# Monkey-patch importlib.metadata.version to avoid PackageNotFoundError in utils
try:
    import importlib.metadata as _ilm
    _orig_version = _ilm.version

    def _safe_version(name: str) -> str:  # type: ignore[override]
        try:
            return _orig_version(name)
        except Exception:
            # Return a CPU-tagged version string for vllm so utils.is_cpu()==True
            return "vllm-cpu" if name == "vllm" else ""

    _ilm.version = _safe_version  # type: ignore[assignment]
except Exception:
    pass

# Stub out Triton sampling module to avoid importing triton
ops_pkg_name = "vllm.model_executor.layers.ops"
if ops_pkg_name not in sys.modules:
    sys.modules[ops_pkg_name] = types.ModuleType(ops_pkg_name)

sample_mod_name = ops_pkg_name + ".sample"
if sample_mod_name not in sys.modules:
    sample_stub = types.ModuleType(sample_mod_name)

    def get_num_triton_sampler_splits(n_cols: int) -> int:
        return 1

    def sample(**kwargs):
        raise RuntimeError("triton sample stub called unexpectedly")

    sample_stub.get_num_triton_sampler_splits = get_num_triton_sampler_splits
    sample_stub.sample = sample
    sys.modules[sample_mod_name] = sample_stub

# Now we can safely import internal modules without triggering heavy deps
from vllm.sequence import SequenceData, SequenceGroupMetadata
from vllm.sampling_params import SamplingParams
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.sampler import Sampler

# Force-disable pinned memory to work in CPU-only envs
import importlib as _il
_vutils = _il.import_module("vllm.utils")
setattr(_vutils, "is_pin_memory_available", lambda: False)
# Also override the symbol imported into sampling_metadata at import-time
_vsm = _il.import_module("vllm.model_executor.sampling_metadata")
setattr(_vsm, "is_pin_memory_available", lambda: False)

# Micro-benchmark to exercise SamplingMetadata preparation, padding, and sampling

def make_seq_group(seq_id: int, prompt_len: int, out_len: int,
                   sp: SamplingParams, vocab_size: int) -> SequenceGroupMetadata:
    # Create token ids
    prompt = [random.randint(0, vocab_size - 1) for _ in range(prompt_len)]
    output = [random.randint(0, vocab_size - 1) for _ in range(out_len)]
    sd = SequenceData(prompt, output)
    seq_data: Dict[int, SequenceData] = {seq_id: sd}
    block_tables = {seq_id: []}
    # Decode stage only to avoid providing seq_lens/query_lens
    sgm = SequenceGroupMetadata(
        request_id=f"req-{seq_id}",
        is_prompt=False,
        seq_data=seq_data,
        sampling_params=sp,
        block_tables=block_tables,
        do_sample=True,
    )
    return sgm


def run_once(n_groups: int = 256,
             prompt_len: int = 200,
             out_len: int = 32,
             vocab_size: int = 32000,
             device: str = "cpu") -> float:
    # Set penalties/top-k/p to exercise padding + penalties path
    sp = SamplingParams(
        n=1,
        best_of=1,
        presence_penalty=0.5,
        frequency_penalty=0.2,
        repetition_penalty=1.1,
        temperature=0.8,
        top_p=0.95,
        top_k=64,
        min_p=0.05,
        seed=42,
        prompt_logprobs=None,
    )
    groups: List[SequenceGroupMetadata] = [
        make_seq_group(i, prompt_len, out_len, sp, vocab_size) for i in range(n_groups)
    ]

    sampling_metadata = SamplingMetadata.prepare(
        groups,
        seq_lens=[],
        query_lens=None,
        device=device,
        pin_memory=False,
    )
    # Keep CPU output minimal to avoid skewing results
    sampling_metadata.skip_sampler_cpu_output = True

    num_tokens = int(sampling_metadata.selected_token_indices.numel())
    logits = torch.randn(num_tokens, vocab_size, device=device)

    sampler = Sampler()

    t0 = time.time()
    _ = sampler.forward(logits, sampling_metadata)
    torch.cuda.synchronize() if device != "cpu" and torch.cuda.is_available() else None
    t1 = time.time()
    return t1 - t0


if __name__ == "__main__":
    # Warmup
    _ = run_once(n_groups=32)
    # Timed runs
    runs = 5
    times = [run_once() for _ in range(runs)]
    print({
        "runs": runs,
        "avg_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
        "times": times,
    })
