import os
import sys
import json
import time
import importlib
from typing import Dict, Any, Tuple, Optional, List
from collections import deque
from unittest.mock import MagicMock

import numpy as np
import torch

# =======================
# Determinism Setup
# =======================
def ensure_determinism():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

# =======================
# Hardware Detection
# =======================
def detect_hardware() -> Dict[str, Any]:
    hw_info = {}
    if torch.cuda.is_available():
        hw_info["device"] = "cuda"
        hw_info["device_name"] = torch.cuda.get_device_name()
        hw_info["capability"] = torch.cuda.get_device_capability()
        hw_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        hw_info["device"] = "cpu"
        hw_info["device_name"] = "CPU"
        hw_info["memory_gb"] = 0
    return hw_info

# =======================
# Import Resolution
# =======================
def resolve_target() -> Tuple[Any, str]:
    """Resolve the optimization target from environment or metadata."""
    
    # Priority 1: Environment variables
    module_path = os.getenv("PROB_MODULE", "vllm.core.scheduler")
    symbol_name = os.getenv("PROB_SYMBOL", "Scheduler")
    
    # Import with error handling
    try:
        module = importlib.import_module(module_path)
        target = getattr(module, symbol_name)
        
        fq_name = f"{module_path}.{symbol_name}"
        return target, fq_name
        
    except (ImportError, AttributeError) as e:
        error_data = {
            "target_resolved": False,
            "error": str(e),
            "attempted_module": module_path,
            "attempted_symbol": symbol_name
        }
        print(json.dumps(error_data))
        sys.exit(1)

# =======================
# Workload Setup
# =======================
def setup() -> Dict[str, Any]:
    """Create realistic workload for the scheduler optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    # Import required vLLM components
    try:
        from vllm.config import CacheConfig, SchedulerConfig
        from vllm.core.scheduler import Scheduler
        from vllm.core.block_manager import SequenceGroup, Sequence, SequenceStatus
        from vllm.core.scheduler import SequenceData
        from vllm import SamplingParams
        from vllm.block import LogicalTokenBlock
    except ImportError as e:
        print(json.dumps({"target_resolved": False, "error": f"Failed to import vLLM components: {e}"}))
        sys.exit(1)
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Create scheduler configuration
    block_size = 16
    num_gpu_blocks = 1024
    num_cpu_blocks = 512
    max_num_seqs = 256
    max_model_len = 2048
    max_num_batched_tokens = 2048
    
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len
    )
    
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space_bytes=0,
        cache_dtype="auto"
    )
    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = num_cpu_blocks
    
    # Create scheduler instance
    scheduler = Scheduler(scheduler_config, cache_config, None)
    
    # Create sequence groups to simulate realistic workload
    seq_groups = []
    num_requests = 64  # Simulate many concurrent requests
    
    for i in range(num_requests):
        # Create sequence with varying prompt lengths
        prompt_length = 128 + (i % 8) * 64  # Vary from 128 to 640 tokens
        seq_id = i
        
        # Create sequence data
        prompt_token_ids = list(range(prompt_length))
        seq_data = SequenceData(prompt_token_ids)
        
        # Create sequence
        seq = Sequence(
            seq_id=seq_id,
            prompt=prompt_token_ids,
            prompt_token_ids=prompt_token_ids,
            block_size=block_size
        )
        seq.data = seq_data
        seq.status = SequenceStatus.WAITING
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=128
        )
        
        # Create sequence group
        seq_group = SequenceGroup(
            request_id=str(i),
            seqs=[seq],
            sampling_params=sampling_params,
            arrival_time=time.time() - (num_requests - i) * 0.01  # Stagger arrival times
        )
        
        seq_groups.append(seq_group)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "scheduler": scheduler,
        "seq_groups": seq_groups,
        "scheduler_config": scheduler_config,
        "cache_config": cache_config
    }
    
    return data