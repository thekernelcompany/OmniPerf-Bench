import time
import random
import torch
import types
import sys
import importlib.util
import os

# Stub minimal vllm modules to avoid heavy dependencies during benchmarking
vllm_mod = types.ModuleType('vllm')
sys.modules['vllm'] = vllm_mod

# Stub sampler output class
sampler_mod = types.ModuleType('vllm.model_executor.layers.sampler')
class SamplerOutput:
    def __init__(self, outputs=None, sampled_token_probs=None, logprobs=None, sampled_token_ids=None):
        self.outputs = outputs
        self.sampled_token_probs = sampled_token_probs
        self.logprobs = logprobs
        self.sampled_token_ids = sampled_token_ids
sampler_mod.SamplerOutput = SamplerOutput
sys.modules['vllm.model_executor'] = types.ModuleType('vllm.model_executor')
sys.modules['vllm.model_executor.layers'] = types.ModuleType('vllm.model_executor.layers')
sys.modules['vllm.model_executor.layers.sampler'] = sampler_mod

# Stub sequence types used only for type annotations in ngram_worker
sequence_mod = types.ModuleType('vllm.sequence')
class ExecuteModelRequest:
    def __init__(self, seq_group_metadata_list):
        self.seq_group_metadata_list = seq_group_metadata_list
        self.blocks_to_swap_in = []
        self.blocks_to_swap_out = []
        self.blocks_to_copy = []
sequence_mod.ExecuteModelRequest = ExecuteModelRequest
sys.modules['vllm.sequence'] = sequence_mod

# Stub proposer base and interfaces/top1
spec_decode_pkg = types.ModuleType('vllm.spec_decode')
sys.modules['vllm.spec_decode'] = spec_decode_pkg
interfaces_mod = types.ModuleType('vllm.spec_decode.interfaces')
class SpeculativeProposals: ...
interfaces_mod.SpeculativeProposals = SpeculativeProposals
sys.modules['vllm.spec_decode.interfaces'] = interfaces_mod

worker_base_mod = types.ModuleType('vllm.spec_decode.proposer_worker_base')
class NonLLMProposerWorkerBase:
    def set_include_gpu_probs_tensor(self):
        pass
    def set_should_modify_greedy_probs_inplace(self):
        pass
worker_base_mod.NonLLMProposerWorkerBase = NonLLMProposerWorkerBase
sys.modules['vllm.spec_decode.proposer_worker_base'] = worker_base_mod

top1_mod = types.ModuleType('vllm.spec_decode.top1_proposer')
class Top1Proposer:
    def __init__(self, *args, **kwargs):
        pass
    def get_spec_proposals(self, *args, **kwargs):
        return []
top1_mod.Top1Proposer = Top1Proposer
sys.modules['vllm.spec_decode.top1_proposer'] = top1_mod

# Now import NGramWorker directly from file path under the correct module name
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ngram_path = os.path.join(root, 'vllm', 'spec_decode', 'ngram_worker.py')
spec = importlib.util.spec_from_file_location('vllm.spec_decode.ngram_worker', ngram_path)
ngram_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ngram_module)  # type: ignore
NGramWorker = ngram_module.NGramWorker

# Minimal stubs for building requests compatible with NGramWorker usage
class SequenceData:
    @staticmethod
    def from_seqs(tokens):
        o = SequenceData()
        o._tokens = list(tokens)
        return o
    def get_len(self):
        return len(self._tokens)
    def get_token_ids(self):
        return self._tokens

class SequenceGroupMetadata:
    def __init__(self, request_id, is_prompt, seq_data, sampling_params, block_tables):
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables


class DummyModelConfig:
    def __init__(self, vocab_size: int = 32000):
        self._vocab_size = vocab_size
    def get_vocab_size(self) -> int:
        return self._vocab_size


def build_request(batch_sizes, token_range=(0, 1000)) -> ExecuteModelRequest:
    seq_group_metadata_list = []
    for i, L in enumerate(batch_sizes):
        # Build a token sequence with some repeated patterns to trigger n-gram matches
        pattern = list(range(64))
        tokens = (pattern * (L // len(pattern) + 1))[:L]
        # add some noise
        for j in range(0, L, 97):
            tokens[j] = random.randint(*token_range)
        sd = SequenceData.from_seqs(tokens)
        sgm = SequenceGroupMetadata(
            request_id=f"r{i}",
            is_prompt=True,
            seq_data={i: sd},
            sampling_params=None,
            block_tables={i: []},
        )
        seq_group_metadata_list.append(sgm)
    return ExecuteModelRequest(seq_group_metadata_list=seq_group_metadata_list)


def time_worker(worker: NGramWorker, req: ExecuteModelRequest, sample_len: int, iters: int = 50) -> float:
    start = time.perf_counter()
    empty = set()
    for _ in range(iters):
        worker.sampler_output(req, sample_len, empty)
    end = time.perf_counter()
    return (end - start) * 1000.0


def main():
    random.seed(0)
    torch.manual_seed(0)

    worker = NGramWorker(local_rank=0, model_config=DummyModelConfig())
    # Avoid requiring GPU for testing
    worker.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    worker.set_ngram_window_size(1, 8)

    # Build a mixed batch of sequence lengths
    batch_sizes = [512, 768, 1024, 1536, 2048, 256, 4096, 819, 1200, 1600]
    req = build_request(batch_sizes)

    # Warmup
    worker.sampler_output(req, 8, set())

    # Benchmark
    t_ms = time_worker(worker, req, sample_len=8, iters=100)
    print(f"NGramWorker.sampler_output: {t_ms:.2f} ms for 100 iters over batch={len(batch_sizes)}")


if __name__ == "__main__":
    main()
