import time
import sys
import types
import importlib.util
from pathlib import Path

# Dynamically load only the needed vllm modules to avoid heavy top-level imports
ROOT = Path(__file__).resolve().parents[1]
VLLM_DIR = ROOT / "vllm"

# Create stub package hierarchy: vllm, vllm.core, vllm.core.block
if "vllm" not in sys.modules:
    sys.modules["vllm"] = types.ModuleType("vllm")
if "vllm.core" not in sys.modules:
    sys.modules["vllm.core"] = types.ModuleType("vllm.core")
if "vllm.core.block" not in sys.modules:
    sys.modules["vllm.core.block"] = types.ModuleType("vllm.core.block")

# Mark them as namespace packages
sys.modules["vllm"].__path__ = []  # type: ignore[attr-defined]
sys.modules["vllm.core"].__path__ = []  # type: ignore[attr-defined]
sys.modules["vllm.core.block"].__path__ = []  # type: ignore[attr-defined]


# Provide a minimal stub for vllm.utils to satisfy imports
from enum import Enum
utils_stub = types.ModuleType("vllm.utils")
class Device(Enum):
    CPU = 0
    GPU = 1

def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b

utils_stub.Device = Device
utils_stub.cdiv = cdiv
sys.modules["vllm.utils"] = utils_stub

# Helper to load a module from file into a specific name

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

# Load dependencies required by prefix_caching_block.py
_load("vllm.core.block.interfaces", VLLM_DIR / "core" / "block" / "interfaces.py")
_load("vllm.core.block.common", VLLM_DIR / "core" / "block" / "common.py")
_load("vllm.core.block.naive_block", VLLM_DIR / "core" / "block" / "naive_block.py")
_load("vllm.core.evictor_v2", VLLM_DIR / "core" / "evictor_v2.py")

# Finally load the target module
prefix_mod = _load(
    "vllm.core.block.prefix_caching_block",
    VLLM_DIR / "core" / "block" / "prefix_caching_block.py",
)
PrefixCachingBlockAllocator = prefix_mod.PrefixCachingBlockAllocator

# Benchmark prefix caching block allocation with common prefixes
block_size = 16
num_blocks = 256
num_sequences = 8
common_prefix_blocks = 4

# Create allocator
allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)

# Common token IDs for shared prefix
common_token_ids = list(range(block_size * common_prefix_blocks))

# Time the allocation and marking operation
start = time.time()

# Allocate blocks for multiple sequences with common prefixes
for seq_idx in range(num_sequences):
    prev_block = None
    for block_idx in range(common_prefix_blocks):
        start_idx = block_idx * block_size
        end_idx = start_idx + block_size
        token_ids = common_token_ids[start_idx:end_idx]

        block = allocator.allocate_immutable_block(
            prev_block=prev_block,
            token_ids=token_ids
        )
        prev_block = block

# Mark blocks as computed (optimized operation path should be fast/no-op)
allocator.mark_blocks_as_computed([])

duration = time.time() - start
print(f"Duration: {duration:.6f} seconds")
print(f"Cache hit rate: {allocator.get_prefix_cache_hit_rate():.3f}")
