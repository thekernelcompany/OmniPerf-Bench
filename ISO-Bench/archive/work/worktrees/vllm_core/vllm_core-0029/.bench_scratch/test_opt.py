import time
import types
import sys
from statistics import mean

# Stub out modules to avoid heavy/absent deps during testing
mod = types.ModuleType('transformers_neuronx')
mod_config = types.ModuleType('transformers_neuronx.config')
class GenerationConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
mod_config.GenerationConfig = GenerationConfig
sys.modules['transformers_neuronx'] = mod
sys.modules['transformers_neuronx.config'] = mod_config
# Stub gguf
mod_gguf = types.ModuleType('gguf')
import importlib.machinery as _machinery
mod_gguf.__spec__ = _machinery.ModuleSpec('gguf', loader=None)
sys.modules['gguf'] = mod_gguf

from vllm.worker.neuron_worker import NeuronWorker


# Bypass __init__ to avoid heavy imports/initialization
wrk = object.__new__(NeuronWorker)
# Inject minimal scheduler and cache configs
wrk.scheduler_config = types.SimpleNamespace(max_num_seqs=512)
wrk.cache_config = types.SimpleNamespace(num_gpu_blocks=None, num_cpu_blocks=None)

# Warmup
for _ in range(1000):
    ng, nc = wrk.determine_num_available_blocks()
    wrk.initialize_cache(ng, nc)

# Timed
iters = 200000
start = time.perf_counter()
for _ in range(iters):
    ng, nc = wrk.determine_num_available_blocks()
    wrk.initialize_cache(ng, nc)
end = time.perf_counter()

print({
    'configured_max_num_seqs': wrk.scheduler_config.max_num_seqs,

    'iters': iters,
    'elapsed_s': end - start,
    'ns_per_iter': (end - start) * 1e9 / iters,
    'num_gpu_blocks': wrk.cache_config.num_gpu_blocks,
    'num_cpu_blocks': wrk.cache_config.num_cpu_blocks,
})
