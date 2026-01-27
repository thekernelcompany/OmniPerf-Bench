#!/usr/bin/env python3
"""
SGLang Per-Commit Isolated Benchmark Runner

Each commit gets its own isolated venv. No shared dependencies.
Fresh venv per commit using `pip install -e "python"` from sglang-install.md.

Usage:
    python run_isolated_benchmarks.py --commit 148254d4 --dry-run
    python run_isolated_benchmarks.py --all
    python run_isolated_benchmarks.py --commit 187b85b7 --variants baseline,human
"""

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = Path("/home/ubuntu/OmniPerf-Bench")
RESULTS_DIR = BASE_DIR / "omniperf_results_3way_sglang/isolated_benchmark_results"
COMMIT_MAPPING_FILE = BASE_DIR / "src/benchmark/fixes/sglang_commit_mapping.json"
SGLANG_REPO_DIR = BASE_DIR / "omniperf_results_3way_sglang/sglang-repo"
SGLANG_REPO_BACKUP = BASE_DIR / "omniperf_results_3way_sglang/sglang-local"  # Fallback
VENV_BASE_DIR = Path("/tmp/sglang-venvs")
AGENT_BASE_DIR = BASE_DIR / "perf-agents-bench/state/runs/sglang"

# Agent patch directories (verified paths with patches for all 17 target commits)
# NOTE: trae_sonnet45 has patches in TWO locations:
#   1. sglang/trae/claude-sonnet-45/ - Has ALL 17 target commits (primary)
#   2. sglan/trae/us-anthropic-claude-sonnet-4-5-20250929-v1-0/ - Has 8 commits (backup)
SGLAN_BASE_DIR = BASE_DIR / "perf-agents-bench/state/runs/sglan"
SGLAN_SONNET45_BASE = SGLAN_BASE_DIR / "trae/us-anthropic-claude-sonnet-4-5-20250929-v1-0"

# Primary trae_sonnet45 directory (has all 17 target commits!)
SGLANG_SONNET45_PRIMARY = AGENT_BASE_DIR / "trae/claude-sonnet-45/2025-11-28_15-26-15"

AGENT_CONFIGS = {
    "claude_code": AGENT_BASE_DIR / "claude_code/default/2025-12-23_06-28-44",
    "codex": AGENT_BASE_DIR / "codex/gpt-5/389be848",
    # trae_gpt5 searches across multiple subdirectories
    "trae_gpt5": AGENT_BASE_DIR / "trae/gpt-5",  # Base dir, will search subdirs
    # trae_sonnet45 searches primary dir first, then sglan subdirectories
    "trae_sonnet45": SGLANG_SONNET45_PRIMARY,  # Primary dir with all 17 commits
    # MIMO agent (xiaomi-mimo-v2-flash) - searches sglang_core-XXXX dirs by journal.json
    "mimo": SGLAN_BASE_DIR / "trae/xiaomi-mimo-v2-flash/2026-01-27_10-49-46",
}

# Ordered list of trae-gpt5 subdirectories to search (2nd dir has more patches)
TRAE_GPT5_SUBDIRS = [
    "2025-11-16_09-27-51",  # Has patches for commits empty in the first dir
    "2025-11-14_21-05-32",  # Original directory
]

# Ordered list of sglan subdirectories to search for trae_sonnet45 patches
# (from most to least patches, includes ALL subdirs with patches)
SGLAN_SONNET45_SUBDIRS = [
    "2025-12-24_19-36-38",  # 34 patches - main batch including c087ddd6, df7f61ee, etc.
    "2025-12-24_14-38-37",  # 5 patches - 148254d4, 132dad87, 205d5cb4, 021f76e4, 23c764b1
    "2025-12-24_12-47-47",  # 2 patches
    "2025-12-24_10-23-32",  # 2 patches
    "2025-12-24_09-35-05",  # 2 patches
    "2025-12-24_19-11-55",  # 1 patch - 2a413829 (target commit!)
    "2025-12-24_18-19-35",  # 1 patch
    "2025-12-24_09-21-46",  # 1 patch
]

# PERF_COMMAND_CONFIG from HuggingFace dataset Ayushnangia/omniperf_v1
# Maps commit -> benchmark type and original command from dataset
PERF_COMMAND_CONFIG = {
    "c087ddd6": {
        "bench_type": "one_batch",
        "model": "deepseek-ai/DeepSeek-V3",  # Needs tp=8, too big for single H100
        "full_command": "python -m sglang.bench_one_batch --model deepseek-ai/DeepSeek-V3 --trust-remote-code --tp 8 --batch-size 1 --input 128 --output 256",
    },
    "6b231325": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
    "dd1012fc": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
    "df7f61ee": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
    "e3ec6bf4": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
    "da47621c": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
    "a191a0e4": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
    "31589e17": {
        "bench_type": "serving",
        "model": "deepseek-ai/DeepSeek-V3-0324",  # Needs tp=8, too big
        "full_command": "python3 -m sglang.bench_serving --backend sglang --model deepseek-ai/DeepSeek-V3-0324 --dataset-name random --random-range-ratio 1 --random-input-len 1000 --random-output-len 1000 --max-concurrency 1",
    },
    "6cb00c63": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
    "132dad87": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
    "b1e5a33a": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
    "021f76e4": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python3 -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompt 480 --request-rate 8 --lora-name lora",
    },
    "2ed68d7a": {
        "bench_type": "serving",
        "model": "deepseek-ai/DeepSeek-V3",  # Needs tp=8, too big
        "full_command": "python3 -m sglang.bench_serving --backend sglang --model deepseek-ai/DeepSeek-V3 --dataset-name random --num-prompt 512 --random-input 1000 --random-output 1000",
    },
    "73b13e69": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
    "187b85b7": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
    "205d5cb4": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  # Large model
        "full_command": "python3 -m sglang.bench_serving --backend sglang --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --dataset-name random --num-prompts 100 --random-input 1000 --random-output 1000 --random-range-ratio 1.0 --max-concurrency 64",
    },
    "1acca3a2": {
        "bench_type": "serving",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "full_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
    },
}

# Models that can run on a single H100 (80GB)
SINGLE_GPU_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Fallback model for large models that need tp>1
FALLBACK_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Commits that require multi-GPU (DeepSeek-V3, Llama-4-Maverick) - SKIP these
MULTI_GPU_COMMITS = {
    "c087ddd6",  # DeepSeek-V3 (671B) - needs H100-TP8
    "2ed68d7a",  # DeepSeek-V3 (671B) - needs H100-TP8
    "31589e17",  # DeepSeek-V3-0324 (671B) - needs H100-TP16-DP16
    "205d5cb4",  # Llama-4-Maverick-17B-128E-FP8 - needs H100-TP8
}

# Single-GPU compatible commits (use actual perf_command from dataset)
SINGLE_GPU_COMMITS = [
    "021f76e4",  # PR#6994 - bench_serving --num-prompt 480 --request-rate 8 --lora-name lora
    "132dad87",  # PR#6922 - bench_serving --num-prompts 100
    "187b85b7",  # PR#7393 - bench_serving --num-prompts 100
    "1acca3a2",  # PR#5969 - bench_serving --num-prompts 100
    "6b231325",  # PR#6649 - bench_serving --num-prompts 100
    "6cb00c63",  # PR#6761 - bench_serving --num-prompts 100
    "73b13e69",  # PR#7285 - bench_serving --num-prompts 100
    "a191a0e4",  # PR#6593 - bench_serving --num-prompts 100
    "b1e5a33a",  # PR#6960 - bench_serving --num-prompts 100
    "da47621c",  # PR#7058 - bench_serving --num-prompts 100
    "dd1012fc",  # PR#6764 - bench_serving --num-prompts 100
    "df7f61ee",  # PR#6812 - bench_serving --num-prompts 100
    "e3ec6bf4",  # PR#6814 - bench_serving --num-prompts 100
]

# All 17 Target commits (8-char short hashes)
# Updated: Now using SINGLE_GPU_COMMITS as default (13 commits compatible with single H100)
# Multi-GPU commits (c087ddd6, 2ed68d7a, 31589e17, 205d5cb4) moved to MULTI_GPU_COMMITS
TARGET_COMMITS = SINGLE_GPU_COMMITS.copy()

# Actually completed commits (benchmarked successfully)
COMPLETED_COMMITS = [
    "2a754e57",  # Done from before
    "187b85b7", "6b231325", "c087ddd6", "da47621c",
    "dd1012fc", "df7f61ee", "e3ec6bf4",
]

# Commits to skip due to unresolvable issues
SKIP_COMMITS = []  # Fresh start with new commit list

# Commit era -> PyTorch version mapping for H100 (SM90) compatibility
COMMIT_ERA_CONFIG = {
    # June 2024 era (PR < 1000, SGLang 0.1.x)
    "june_2024": {
        "torch": "torch==2.3.0+cu121",
        "torch_index": "https://download.pytorch.org/whl/cu121",
        "triton": "triton==2.3.0",
        "vllm": None,  # Use from pyproject.toml
        "commits": ["2a754e57", "09deb20d", "1bf1cf19", "564a898a",
                    "6a2941f4", "6f560c76", "9216b106", "ac971ff6",
                    "bb3a3b66", "e822e590"]
    },
    # Oct-Dec 2024 era (SGLang 0.3.x-0.4.x)
    # NOTE: Using torch 2.5.1 - sgl_kernel has ABI issues with 2.6.0
    "late_2024": {
        "torch": "torch==2.5.1+cu124",
        "torch_index": "https://download.pytorch.org/whl/cu124",
        "triton": "triton==3.3.0",  # 3.0-3.1 removed default_cache_dir, added back in 3.2+
        "vllm": "vllm>=0.6.0",
        "commits": ["187b85b7", "6b231325", "6cb00c63", "148254d4",
                    "2bd18e2d", "880221bd", "b1e5a33a", "c087ddd6",
                    "da47621c", "dd1012fc", "ddcf9fe3", "df7f61ee",
                    "e3ec6bf4", "4418f599", "2a413829", "5e023301",
                    # New from PR#6500-7500 range:
                    "a191a0e4", "31589e17", "132dad87", "021f76e4",
                    "2ed68d7a", "73b13e69", "205d5cb4", "1acca3a2"]
    }
}

# Per-commit specific dependency overrides
# Maps commit -> dict of dependencies to install
COMMIT_DEPS_MAP = {
    # Older commits (Dec 2024) - need sgl-kernel==0.2.9
    "4418f599": {
        "vllm": "vllm==0.6.6.post1",
        "sgl_kernel": "sgl-kernel==0.2.9",
    },
    "5e023301": {
        "vllm": "vllm==0.6.6.post1",
        "sgl_kernel": "sgl-kernel==0.2.9",
    },
    "2a413829": {
        "vllm": "vllm==0.6.6.post1",
        "sgl_kernel": "sgl-kernel==0.2.9",
    },
    "6cb00c63": {
        "vllm": "vllm==0.6.6.post1",
        "sgl_kernel": "sgl-kernel==0.2.9",
    },
    "880221bd": {
        "vllm": "vllm==0.6.6.post1",
        "sgl_kernel": "sgl-kernel==0.2.9",
    },
    "b1e5a33a": {
        "vllm": "vllm==0.6.6.post1",
        "sgl_kernel": "sgl-kernel==0.2.9",
    },
    # Earlier commits (Nov 2024) - older vllm, likely no sgl_kernel
    "148254d4": {
        "vllm": "vllm==0.6.4.post1",
        "sgl_kernel": None,  # Skip - causes issues
    },
    "2bd18e2d": {
        "vllm": "vllm==0.6.4.post1",
        "sgl_kernel": "sgl-kernel==0.2.9",
    },
    "ddcf9fe3": {
        "vllm": "vllm==0.7.2",
        "sgl_kernel": "sgl-kernel==0.2.9",
    },
    # Successful commits (for reference)
    "187b85b7": {"vllm": "vllm>=0.6.0"},  # Worked without sgl_kernel
    "6b231325": {"vllm": "vllm>=0.6.0"},
    "c087ddd6": {"vllm": "vllm>=0.6.0"},
    "da47621c": {"vllm": "vllm>=0.6.0"},
    "dd1012fc": {"vllm": "vllm>=0.6.0"},
    "df7f61ee": {"vllm": "vllm>=0.6.0"},
    "e3ec6bf4": {"vllm": "vllm>=0.6.0"},
}

# Legacy - for backwards compatibility
VLLM_VERSION_OVERRIDES = {k: v.get("vllm", "vllm>=0.6.0") for k, v in COMMIT_DEPS_MAP.items() if v.get("vllm")}

# Benchmark settings
BENCHMARK_TIMEOUT = 600  # 10 minutes
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCH_SIZE = 4
INPUT_LEN = 512
OUTPUT_LEN = 64


# ============================================================================
# Helper Functions
# ============================================================================

def get_commit_era(commit_short: str) -> str:
    """Determine which era a commit belongs to for PyTorch version selection."""
    for era, config in COMMIT_ERA_CONFIG.items():
        if commit_short in config["commits"]:
            return era
    # Default to late_2024 for unknown commits
    return "late_2024"


def check_gpu_compatibility() -> bool:
    """Verify H100 GPU is available and compatible."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            return False

        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)

        print(f"GPU: {gpu_name}")
        print(f"Compute Capability: SM{compute_cap[0]}{compute_cap[1]}")

        # H100 = SM90
        if compute_cap[0] < 9:
            print(f"WARNING: GPU is SM{compute_cap[0]}0, not H100 (SM90)")

        return True
    except Exception as e:
        print(f"ERROR checking GPU: {e}")
        return False


def load_commit_mapping() -> dict:
    """Load commit mapping from JSON file."""
    with open(COMMIT_MAPPING_FILE) as f:
        data = json.load(f)

    mapping = {}
    index_mapping = {}  # For codex index lookup

    for idx, commit in enumerate(data["commits"]):
        short = commit["human_commit"][:8]
        mapping[short] = {
            "human_commit": commit["human_commit"],
            "base_commit": commit["base_commit"],
            "pr_number": commit.get("pr_number"),
            "subject": commit.get("subject", "Unknown"),
            "index": idx
        }
        index_mapping[idx] = short

    return mapping, index_mapping


def run_command(cmd: list[str], cwd: Optional[Path] = None,
                timeout: int = 300, env: Optional[dict] = None) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -2, "", str(e)


def clone_sglang_repo():
    """Clone fresh SGLang repo if not exists."""
    global SGLANG_REPO_DIR

    if SGLANG_REPO_DIR.exists():
        print(f"SGLang repo already exists at {SGLANG_REPO_DIR}")
        return True

    # Check for backup repo
    if SGLANG_REPO_BACKUP.exists():
        print(f"Using backup repo at {SGLANG_REPO_BACKUP}")
        SGLANG_REPO_DIR = SGLANG_REPO_BACKUP
        return True

    print(f"Cloning SGLang repo to {SGLANG_REPO_DIR}...")
    SGLANG_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)

    returncode, stdout, stderr = run_command(
        ["git", "clone", "https://github.com/sgl-project/sglang.git", str(SGLANG_REPO_DIR)],
        timeout=600
    )

    if returncode != 0:
        print(f"ERROR: Failed to clone repo: {stderr}")
        return False

    print("Successfully cloned SGLang repo")
    return True


def patch_sgl_kernel_import():
    """Patch SGLang to make sgl_kernel import optional when it fails to load.

    This modifies awq.py to catch ImportError and provide a dummy implementation,
    allowing benchmarks to run on non-AWQ workloads.
    """
    awq_file = SGLANG_REPO_DIR / "python/sglang/srt/layers/quantization/awq.py"
    if not awq_file.exists():
        return

    content = awq_file.read_text()

    # Check if already patched
    if "# PATCHED: sgl_kernel optional" in content:
        return

    # Replace the sgl_kernel import with a try/except
    old_pattern = '''_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel import awq_dequantize'''

    new_pattern = '''# PATCHED: sgl_kernel optional (for benchmarking without AWQ)
_is_cuda = is_cuda()
awq_dequantize = None
if _is_cuda:
    try:
        from sgl_kernel import awq_dequantize
    except ImportError:
        import logging
        logging.getLogger(__name__).warning("sgl_kernel not available, AWQ quantization disabled")
        awq_dequantize = None'''

    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        awq_file.write_text(content)
        print("    Patched awq.py to make sgl_kernel optional")
    else:
        print("    Could not find expected pattern in awq.py, skipping patch")


def patch_sgl_kernel_quantization():
    """Patch SGLang quantization/__init__.py to skip sgl_kernel-dependent imports.

    This disables W8A8Int8Config and other sgl_kernel-dependent quantization methods
    when sgl_kernel fails to load, allowing benchmarks to run on non-quantized workloads.

    Files that import from sgl_kernel:
    - awq.py: awq_dequantize (already patched by patch_sgl_kernel_import)
    - w8a8_int8.py: int8_scaled_mm (causes failures)
    - fp8_utils.py: various FP8 ops
    """
    quant_init = SGLANG_REPO_DIR / "python/sglang/srt/layers/quantization/__init__.py"
    if not quant_init.exists():
        print("    quantization/__init__.py not found, skipping patch")
        return False

    content = quant_init.read_text()

    # Check if already patched
    if "# PATCHED: sgl_kernel quantization optional" in content:
        print("    quantization/__init__.py already patched")
        return True

    # Pattern 1: Wrap W8A8Int8Config import in try/except
    old_w8a8_import = "from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config"
    new_w8a8_import = """# PATCHED: sgl_kernel quantization optional
try:
    from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config
except ImportError:
    W8A8Int8Config = None"""

    patched = False
    if old_w8a8_import in content:
        content = content.replace(old_w8a8_import, new_w8a8_import)
        patched = True
        print("    Patched W8A8Int8Config import to be optional")

    # Pattern 2: Wrap W8A8Fp8Config import in try/except (if exists)
    old_fp8_import = "from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config"
    new_fp8_import = """try:
    from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config
except ImportError:
    W8A8Fp8Config = None"""

    if old_fp8_import in content and "try:\n    from sglang.srt.layers.quantization.w8a8_fp8" not in content:
        content = content.replace(old_fp8_import, new_fp8_import)
        patched = True
        print("    Patched W8A8Fp8Config import to be optional")

    # Pattern 3: Patch the w8a8_int8.py file directly if it exists
    w8a8_file = SGLANG_REPO_DIR / "python/sglang/srt/layers/quantization/w8a8_int8.py"
    if w8a8_file.exists():
        w8a8_content = w8a8_file.read_text()
        if "from sgl_kernel import" in w8a8_content and "# PATCHED: sgl_kernel optional" not in w8a8_content:
            # Wrap the sgl_kernel import in a try/except
            w8a8_content = w8a8_content.replace(
                "from sgl_kernel import int8_scaled_mm",
                """# PATCHED: sgl_kernel optional
try:
    from sgl_kernel import int8_scaled_mm
except ImportError:
    int8_scaled_mm = None"""
            )
            w8a8_file.write_text(w8a8_content)
            patched = True
            print("    Patched w8a8_int8.py to make sgl_kernel optional")

    # Pattern 4: Patch fp8_utils.py if it exists
    fp8_utils_file = SGLANG_REPO_DIR / "python/sglang/srt/layers/quantization/fp8_utils.py"
    if fp8_utils_file.exists():
        fp8_content = fp8_utils_file.read_text()
        if "from sgl_kernel import" in fp8_content and "# PATCHED: sgl_kernel optional" not in fp8_content:
            # Common sgl_kernel imports in fp8_utils.py
            for old_import in [
                "from sgl_kernel import fp8_scaled_mm",
                "from sgl_kernel import per_token_group_quant_fp8",
            ]:
                if old_import in fp8_content:
                    func_name = old_import.split(" import ")[1]
                    new_import = f"""# PATCHED: sgl_kernel optional
try:
    {old_import}
except ImportError:
    {func_name} = None"""
                    fp8_content = fp8_content.replace(old_import, new_import)
                    patched = True
                    print(f"    Patched fp8_utils.py ({func_name})")

            if patched:
                fp8_utils_file.write_text(fp8_content)

    if patched:
        quant_init.write_text(content)
        return True

    print("    No sgl_kernel patterns found to patch in quantization")
    return False


def patch_deep_gemm_import():
    """Patch deep_gemm.py to make deep_gemm imports optional.

    The deep_gemm API changed significantly in sgl-kernel 0.3.7 (switched to C++ JIT).
    Older commits expect the old Python JIT interface which no longer exists.
    This patch makes the imports optional for non-quantized workloads.

    Handles imports inside `if is_cuda():` blocks (indented with 4 spaces).
    Handles multi-line imports with parentheses.
    """
    deep_gemm_file = SGLANG_REPO_DIR / "python/sglang/srt/layers/quantization/deep_gemm.py"
    if not deep_gemm_file.exists():
        print("    deep_gemm.py not found, skipping patch")
        return False

    content = deep_gemm_file.read_text()

    # Check if already patched
    if "# PATCHED: deep_gemm optional" in content:
        print("    deep_gemm.py already patched")
        return True

    def extract_var_names(import_part):
        """Extract variable names from import statement, handling 'as' aliases.

        Examples:
          'get_num_sms' -> ['get_num_sms']
          'includes as deep_gemm_includes' -> ['deep_gemm_includes']
          'FP8GemmRuntime, GemmType' -> ['FP8GemmRuntime', 'GemmType']
          'template as deep_gemm_template, includes' -> ['deep_gemm_template', 'includes']
        """
        names = []
        # Handle parenthesized imports like "from X import (\n    A,\n    B\n)"
        import_part = import_part.replace('(', '').replace(')', '').strip()
        for item in import_part.split(','):
            item = item.strip()
            if not item:
                continue
            if ' as ' in item:
                # Use the alias name
                names.append(item.split(' as ')[1].strip())
            else:
                names.append(item)
        return names

    patched = False
    lines = content.split('\n')
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()

        # Check for deep_gemm imports (both top-level and indented in if blocks)
        # Look for lines like "    from deep_gemm..." or "from deep_gemm..."
        if ('from deep_gemm' in line or line_stripped == 'import deep_gemm') and 'import' in line:
            # Determine the indentation level
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent

            # Check for multi-line import (line ends with '(' or contains '(' without ')')
            full_import_lines = [line_stripped]
            full_import = line_stripped

            if '(' in line_stripped and ')' not in line_stripped:
                # Multi-line import - collect all lines until closing ')'
                i += 1
                while i < len(lines) and ')' not in lines[i]:
                    full_import_lines.append(lines[i])
                    full_import += '\n' + lines[i]
                    i += 1
                # Include the closing line
                if i < len(lines):
                    full_import_lines.append(lines[i])
                    full_import += '\n' + lines[i]

            # Extract the imported names for the except block
            if 'from deep_gemm' in line:
                import_part = full_import.split('import', 1)[1].strip()
                names = extract_var_names(import_part)
                none_assignments = f'\n{indent_str}    '.join([f"{n} = None" for n in names])
            else:
                # "import deep_gemm" case
                names = ['deep_gemm']
                none_assignments = "deep_gemm = None"

            # Create the patched version with proper indentation
            new_lines.append(f"{indent_str}# PATCHED: deep_gemm optional")
            new_lines.append(f"{indent_str}try:")
            # Add all lines of the import with extra indentation (indent + 4 spaces)
            for j, import_line in enumerate(full_import_lines):
                if import_line.strip():
                    if j == 0:
                        # First line - use base indent + 4
                        new_lines.append(f"{indent_str}    {import_line.strip()}")
                    else:
                        # Continuation lines - preserve their relative indentation + 4 extra
                        orig_line_indent = len(import_line) - len(import_line.lstrip())
                        new_lines.append(' ' * (orig_line_indent + 4) + import_line.strip())
                else:
                    new_lines.append('')
            new_lines.append(f"{indent_str}except (ImportError, ModuleNotFoundError):")
            new_lines.append(f"{indent_str}    {none_assignments}")

            patched = True
            print(f"    Patched: {line_stripped[:60]}...")
        else:
            new_lines.append(line)

        i += 1

    if patched:
        deep_gemm_file.write_text('\n'.join(new_lines))
        return True

    print("    No deep_gemm imports found to patch")
    return False


def patch_torchao_import():
    """Patch torchao_utils.py to check for None config BEFORE importing torchao.

    This allows the benchmark to run without torchao installed when torchao_config is None.
    """
    torchao_file = SGLANG_REPO_DIR / "python/sglang/srt/layers/torchao_utils.py"
    if not torchao_file.exists():
        return

    content = torchao_file.read_text()

    # Check if already patched
    if "# PATCHED: early return" in content:
        return

    # The key insight: we need to add an early return BEFORE the imports
    # Original:
    #   def apply_torchao_config_to_model(...):
    #       """docstring"""
    #       # Lazy import...
    #       from torchao.quantization import ...
    #       if torchao_config == "" or torchao_config is None:
    #           return model
    #       elif ...
    #
    # We want to add the None check BEFORE the imports, keeping the elif chain intact

    # Replace the lazy import comment with an early return + the imports
    old_import_section = '''    # Lazy import to suppress some warnings
    from torchao.quantization import ('''

    new_import_section = '''    # PATCHED: early return for None config (avoids importing torchao)
    if torchao_config == "" or torchao_config is None:
        return model

    # Lazy import to suppress some warnings
    from torchao.quantization import ('''

    if old_import_section in content:
        # Also need to remove the original None check since we're adding it earlier
        content = content.replace(old_import_section, new_import_section)
        # Remove the now-redundant if block (change "if torchao_config == ..." to a pass-through)
        content = content.replace(
            "    if torchao_config == \"\" or torchao_config is None:\n        return model\n    elif ",
            "    if "
        )
        torchao_file.write_text(content)
        print("    Patched torchao_utils.py to defer imports")
    else:
        print("    Could not find expected pattern in torchao_utils.py")


def is_commit_completed(commit_short: str, variants: list[str]) -> tuple[bool, list[str]]:
    """Check if a commit has been completed SUCCESSFULLY and return missing variants."""
    result_file = RESULTS_DIR / f"{commit_short}_isolated.json"
    if not result_file.exists():
        return False, variants

    with open(result_file) as f:
        data = json.load(f)

    # Only count variants with successful throughput metrics
    successful_variants = set()
    for variant, vdata in data.get("variants", {}).items():
        # Check for throughput (success indicator)
        metrics = vdata.get("metrics", {})
        if metrics.get("throughput_tokens_per_sec") or metrics.get("output_throughput") or metrics.get("total_throughput"):
            successful_variants.add(variant)
        elif vdata.get("status") == "success":
            successful_variants.add(variant)

    required_variants = set(variants)
    missing = list(required_variants - successful_variants)

    if not missing:
        return True, []

    return False, missing


def create_venv(commit: str) -> Optional[Path]:
    """Create fresh uv venv for a commit."""
    venv_path = VENV_BASE_DIR / commit

    # Remove existing venv
    if venv_path.exists():
        print(f"  Removing existing venv at {venv_path}")
        shutil.rmtree(venv_path)

    venv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Creating venv at {venv_path}")
    returncode, stdout, stderr = run_command(
        ["uv", "venv", str(venv_path)],
        timeout=60
    )

    if returncode != 0:
        print(f"  ERROR: Failed to create venv: {stderr}")
        return None

    return venv_path


def checkout_commit(commit_hash: str) -> bool:
    """Checkout a specific commit in the SGLang repo."""
    print(f"  Checking out commit {commit_hash[:12]}...")

    # Reset and clean
    run_command(["git", "reset", "--hard"], cwd=SGLANG_REPO_DIR)
    run_command(["git", "clean", "-fdx"], cwd=SGLANG_REPO_DIR)

    # Checkout
    returncode, stdout, stderr = run_command(
        ["git", "checkout", commit_hash],
        cwd=SGLANG_REPO_DIR
    )

    if returncode != 0:
        print(f"  ERROR: Failed to checkout: {stderr}")
        return False

    return True


def get_vllm_version_from_pyproject() -> str:
    """Read vllm version requirement from pyproject.toml at current commit."""
    pyproject = SGLANG_REPO_DIR / "python" / "pyproject.toml"
    if not pyproject.exists():
        return "vllm"  # Default to latest

    try:
        content = pyproject.read_text()
        # Look for vllm version spec in srt dependencies
        import re
        # Pattern: vllm>=0.6.3.post1,<=0.6.4.post1 or vllm==0.5.x etc
        match = re.search(r'"vllm([^"]+)"', content)
        if match:
            version_spec = match.group(1)
            # Extract the upper bound or specific version
            # Version pattern: 0.6.4.post1 or 0.6.4
            version_pattern = r'([0-9]+\.[0-9]+\.[0-9]+(?:\.post\d+)?)'
            if "<=" in version_spec:
                # Use the upper bound
                upper = re.search(r'<=' + version_pattern, version_spec)
                if upper:
                    return f"vllm=={upper.group(1)}"
            elif "==" in version_spec:
                eq = re.search(r'==' + version_pattern, version_spec)
                if eq:
                    return f"vllm=={eq.group(1)}"
            elif ">=" in version_spec:
                # Use minimum version
                lower = re.search(r'>=' + version_pattern, version_spec)
                if lower:
                    return f"vllm=={lower.group(1)}"
    except Exception as e:
        print(f"  WARNING: Could not parse vllm version: {e}")

    return "vllm"  # Default to latest


def check_and_install_sgl_kernel(pip_path: Path, python_path: Path) -> tuple[bool, str]:
    """Check if sgl_kernel is needed and install it."""
    # Check if SGLang needs sgl_kernel
    sgl_kernel_imports = SGLANG_REPO_DIR / "python" / "sglang" / "srt"
    needs_sgl_kernel = False

    if sgl_kernel_imports.exists():
        for py_file in sgl_kernel_imports.rglob("*.py"):
            try:
                if "sgl_kernel" in py_file.read_text():
                    needs_sgl_kernel = True
                    break
            except:
                pass

    if not needs_sgl_kernel:
        return True, "not required for this commit"

    # Try installing from PyPI
    print("  Installing sgl_kernel from PyPI...")
    returncode, stdout, stderr = run_command(
        [str(pip_path), "install", "sgl_kernel"],
        timeout=300
    )

    if returncode == 0:
        # Verify it loads
        test_code = 'import sgl_kernel; print("OK")'
        rc, out, err = run_command([str(python_path), "-c", test_code], timeout=30)
        if rc == 0 and "OK" in out:
            return True, "installed from PyPI"
        return False, f"import failed: {err[:200]}"

    return False, f"install failed: {stderr[-200:]}"


def install_sglang(venv_path: Path, commit_short: str) -> bool:
    """Install SGLang using pyproject.toml from the checked-out commit.

    Strategy:
    1. Install torch first (correct version for H100)
    2. Install SGLang base (editable)
    3. Install [runtime_common] deps from pyproject.toml (skip flashinfer)
    4. Install vllm version from pyproject.toml
    5. Force correct triton version
    """
    python_path = venv_path / "bin" / "python"
    python_dir = SGLANG_REPO_DIR / "python"
    pyproject_path = python_dir / "pyproject.toml"

    # Get era config for torch version
    era = get_commit_era(commit_short)
    config = COMMIT_ERA_CONFIG[era]

    print(f"  Commit era: {era}")

    def uv_pip(*args, timeout=300):
        """Run uv pip with the venv's python."""
        cmd = ["uv", "pip", *args, "--python", str(python_path)]
        return run_command(cmd, timeout=timeout)

    # 1. Install torch first (correct version for H100)
    print(f"  Installing torch ({config['torch']})...")
    returncode, _, stderr = uv_pip(
        "install", config["torch"],
        "--index-url", config["torch_index"],
        timeout=300
    )
    if returncode != 0:
        print(f"  ERROR: torch install failed: {stderr[-300:]}")
        return False

    # 2. Read pyproject.toml to extract deps
    runtime_deps = []
    vllm_spec = None
    if pyproject_path.exists():
        print(f"  Reading deps from pyproject.toml...")
        content = pyproject_path.read_text()

        # Extract runtime_common deps
        import re
        runtime_match = re.search(r'runtime_common\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if runtime_match:
            deps_str = runtime_match.group(1)
            # Parse each dependency
            for dep in re.findall(r'"([^"]+)"', deps_str):
                # Skip problematic deps that cause cascade failures
                # NOTE: torchao removed from skip list - now installed explicitly
                skip_patterns = ["flashinfer", "sgl-kernel", "outlines", "xgrammar"]
                if any(p in dep.lower() for p in skip_patterns):
                    print(f"    Skipping: {dep}")
                    continue
                runtime_deps.append(dep)

        # Extract vllm version from [srt] extra
        srt_match = re.search(r'srt\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if srt_match:
            srt_str = srt_match.group(1)
            vllm_match = re.search(r'"(vllm[^"]*)"', srt_str)
            if vllm_match:
                vllm_spec = vllm_match.group(1)

    # 3. Install SGLang base (editable)
    print("  Installing SGLang base...")
    returncode, _, stderr = uv_pip("install", "-e", str(python_dir), timeout=180)
    if returncode != 0:
        print(f"  WARNING: SGLang base install issue: {stderr[-200:]}")

    # 4. Install runtime deps from pyproject.toml
    if runtime_deps:
        print(f"  Installing {len(runtime_deps)} runtime deps from pyproject.toml...")
        rc, out, err = uv_pip("install", *runtime_deps, timeout=600)
        if rc != 0:
            print(f"  WARNING: Some deps failed, trying one by one...")
            for dep in runtime_deps:
                uv_pip("install", dep, timeout=120)

    # 5. Install vllm - prefer override, then pyproject, then era config
    if commit_short in VLLM_VERSION_OVERRIDES:
        vllm_to_install = VLLM_VERSION_OVERRIDES[commit_short]
        print(f"  Using vllm override: {vllm_to_install}")
    elif vllm_spec:
        # Handle dev versions and version ranges
        if "dev" in vllm_spec or "post" in vllm_spec:
            # Extract base version and use closest release
            base_match = re.search(r'(\d+\.\d+\.\d+)', vllm_spec)
            if base_match:
                base_ver = base_match.group(1)
                vllm_to_install = f"vllm>={base_ver}"
            else:
                vllm_to_install = "vllm>=0.6.0"
        else:
            vllm_to_install = vllm_spec
        print(f"  Using vllm from pyproject: {vllm_to_install}")
    else:
        vllm_to_install = config.get("vllm", "vllm>=0.6.0")
        print(f"  Using vllm from era config: {vllm_to_install}")

    uv_pip("install", vllm_to_install, timeout=600)

    # 5b. Fix pyarrow IMMEDIATELY after vllm (PyExtensionType removed in v17+)
    print("  Fixing pyarrow version (PyExtensionType removed in v17+)...")
    uv_pip("install", "pyarrow>=12.0.0,<17", "datasets>=2.18.0", timeout=120)

    # 5c. Install compatible outlines version with its dependencies
    print("  Installing outlines (compatible version)...")
    uv_pip("install", "outlines>=0.0.44,<0.1.0", "pyairports", "pycountry", timeout=120)

    # 6. Force correct triton AFTER vllm (critical - vllm can downgrade it)
    print(f"  Force installing triton ({config['triton']})...")
    uv_pip("install", "--reinstall", config["triton"], timeout=120)

    # 7. Install additional critical deps that may be missing
    print("  Installing additional critical deps...")
    extra_deps = ["transformers>=4.40.0", "numpy<2", "pillow", "requests", "tqdm", "rpyc", "setuptools"]
    uv_pip("install", *extra_deps, timeout=180)

    # 7b. Force reinstall transformers after vllm (vllm may have downgraded it)
    # Pin to <5.0.0 to avoid AutoImageProcessor.register() API breaking changes
    print("  Force reinstalling transformers (ensuring AutoProcessor is available)...")
    uv_pip("install", "--reinstall", "transformers>=4.44.0,<5.0.0", timeout=120)

    # 8. Handle sgl_kernel if needed (always try for late_2024 era)
    if era == "late_2024":
        # Try installing latest sgl-kernel (not pinned) - may have better compatibility
        print("  Installing sgl-kernel (latest)...")
        uv_pip("install", "sgl-kernel", timeout=180)
        # Verify it works
        rc, out, err = run_command([str(python_path), "-c",
            "import sgl_kernel; print('sgl_kernel: OK')"], timeout=30)
        if rc == 0:
            print(f"  {out.strip()}")
        else:
            print(f"  sgl_kernel: import failed - {err[:100]}")
            # Patch SGLang to make sgl_kernel optional (comprehensive patching)
            print("  Patching SGLang to make sgl_kernel imports optional...")
            patch_sgl_kernel_import()  # Patches awq.py
            patch_sgl_kernel_quantization()  # Patches w8a8_int8.py, fp8_utils.py, __init__.py

    # 8b. Patch torchao_utils.py to defer imports (avoids needing torchao when not used)
    patch_torchao_import()

    # 8c. Patch deep_gemm.py to make deep_gemm imports optional (API changed in sgl-kernel 0.3.7)
    patch_deep_gemm_import()

    # 9. Verify key packages
    rc, out, _ = run_command([str(python_path), "-c",
        "import triton; print('triton:', triton.__version__)"], timeout=30)
    print(f"  {out.strip()}")

    rc, out, _ = run_command([str(python_path), "-c",
        "import transformers; print('transformers:', transformers.__version__)"], timeout=30)
    print(f"  {out.strip()}")

    # Verify imports work
    rc, out, err = run_command([str(python_path), "-c",
        "from transformers import AutoProcessor; print('AutoProcessor: OK')"], timeout=30)
    if rc == 0:
        print(f"  {out.strip()}")
    else:
        print(f"  AutoProcessor: MISSING - {err[:100]}")

    return True


def check_and_install_sgl_kernel_uv(python_path: Path, commit_short: str) -> tuple[bool, str]:
    """Check if sgl_kernel is needed and install with uv using commit-specific version."""
    # Get commit-specific sgl_kernel version
    commit_deps = COMMIT_DEPS_MAP.get(commit_short, {})
    sgl_kernel_spec = commit_deps.get("sgl_kernel")

    # If explicitly set to None, skip installation
    if sgl_kernel_spec is None:
        return True, "skipped per commit config"

    # Check if SGLang needs sgl_kernel
    sgl_kernel_imports = SGLANG_REPO_DIR / "python" / "sglang" / "srt"
    needs_sgl_kernel = False

    if sgl_kernel_imports.exists():
        for py_file in sgl_kernel_imports.rglob("*.py"):
            try:
                if "sgl_kernel" in py_file.read_text():
                    needs_sgl_kernel = True
                    break
            except:
                pass

    if not needs_sgl_kernel:
        return True, "not required"

    # Use commit-specific version or default
    sgl_kernel_to_install = sgl_kernel_spec or "sgl-kernel==0.2.9"
    print(f"  Installing {sgl_kernel_to_install}...")

    returncode, stdout, stderr = run_command(
        ["uv", "pip", "install", sgl_kernel_to_install, "--python", str(python_path)],
        timeout=180
    )

    if returncode == 0:
        # Verify import
        rc, out, err = run_command(
            [str(python_path), "-c", "import sgl_kernel; print('OK')"], timeout=30)
        if rc == 0 and "OK" in out:
            return True, "installed"
        return False, f"import failed: {err[:100]}"

    return False, f"install failed: {stderr[-100:]}"


def discover_benchmark_script() -> str:
    """Discover which benchmark script exists at this commit."""
    bench_one_batch = SGLANG_REPO_DIR / "python/sglang/bench_one_batch.py"
    bench_latency = SGLANG_REPO_DIR / "python/sglang/bench_latency.py"

    if bench_one_batch.exists():
        return "sglang.bench_one_batch"
    elif bench_latency.exists():
        return "sglang.bench_latency"
    else:
        # Check in srt subdirectory
        bench_latency_srt = SGLANG_REPO_DIR / "python/sglang/srt/bench_latency.py"
        if bench_latency_srt.exists():
            return "sglang.srt.bench_latency"
        return "sglang.bench_latency"  # Default fallback


def find_agent_patch(commit_short: str, agent: str, index_mapping: dict, mapping: dict) -> Optional[Path]:
    """Find the patch file for a commit and agent variant."""
    agent_path = AGENT_CONFIGS.get(agent)
    if not agent_path or not agent_path.exists():
        print(f"  Agent path not found for {agent}: {agent_path}")
        return None

    # For codex, find folder by reading prompt.json files (codex uses sglang_core-XXXX naming)
    if agent == "codex":
        # Search for the commit in codex folders by reading prompt.json
        for folder in agent_path.iterdir():
            if folder.is_dir() and folder.name.startswith("sglang_core-"):
                prompt_file = folder / "prompt.json"
                if prompt_file.exists():
                    try:
                        with open(prompt_file) as f:
                            prompt_data = json.load(f)
                        codex_commit = prompt_data.get("commits", {}).get("human", "")[:8]
                        if codex_commit == commit_short:
                            patch_file = folder / "model_patch.diff"
                            if patch_file.exists() and patch_file.stat().st_size > 0:
                                return patch_file
                            print(f"  Codex patch empty for {commit_short}: {patch_file}")
                            return None
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"  Codex patch not found for commit {commit_short}")
        return None

    # For MIMO, search sglang_core-XXXX directories by reading journal.json
    if agent == "mimo":
        if not agent_path.exists():
            print(f"  MIMO agent path not found: {agent_path}")
            return None
        for folder in agent_path.iterdir():
            if folder.is_dir() and folder.name.startswith("sglang_core-"):
                journal_file = folder / "journal.json"
                if journal_file.exists():
                    try:
                        with open(journal_file) as f:
                            journal_data = json.load(f)
                        mimo_commit = journal_data.get("commits", {}).get("human", "")[:8]
                        if mimo_commit == commit_short:
                            patch_file = folder / "model_patch.diff"
                            if patch_file.exists() and patch_file.stat().st_size > 0:
                                # Verify patch_loc > 0 (use 0 as default if key missing)
                                patch_loc = journal_data.get("metrics", {}).get("patch_size_loc") or 0
                                if patch_loc > 0:
                                    return patch_file
                                print(f"  MIMO patch has patch_loc=0 for {commit_short}")
                                return None
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"  MIMO patch not found for commit {commit_short}")
        return None

    # For trae_sonnet45, search primary directory first (has all 17 commits),
    # then fall back to sglan subdirectories
    if agent == "trae_sonnet45":
        # 1. Check primary directory (sglang/trae/claude-sonnet-45/) - uses sglang_XXX_<commit> naming
        if agent_path.exists():
            for folder in agent_path.iterdir():
                if folder.is_dir() and commit_short in folder.name:
                    patch_file = folder / "model_patch.diff"
                    if patch_file.exists() and patch_file.stat().st_size > 0:
                        return patch_file

        # 2. Fall back to sglan subdirectories - uses sglang_sonnet45_rerun_<commit> naming
        for subdir in SGLAN_SONNET45_SUBDIRS:
            patch_folder = SGLAN_SONNET45_BASE / subdir / f"sglang_sonnet45_rerun_{commit_short}"
            patch_file = patch_folder / "model_patch.diff"
            if patch_file.exists() and patch_file.stat().st_size > 0:
                return patch_file

        print(f"  trae_sonnet45 patch not found for commit {commit_short}")
        return None

    # For trae_gpt5, search across multiple subdirectories
    if agent == "trae_gpt5":
        for subdir in TRAE_GPT5_SUBDIRS:
            subdir_path = agent_path / subdir
            if not subdir_path.exists():
                continue
            for folder in subdir_path.iterdir():
                if folder.is_dir() and commit_short in folder.name:
                    patch_file = folder / "model_patch.diff"
                    if patch_file.exists() and patch_file.stat().st_size > 0:
                        return patch_file
        print(f"  trae_gpt5 patch not found for commit {commit_short}")
        return None

    # For other agents (claude_code), search by commit hash in folder name
    # These use sglang_XXX_<commit> naming
    for folder in agent_path.iterdir():
        if folder.is_dir() and commit_short in folder.name:
            patch_file = folder / "model_patch.diff"
            if patch_file.exists() and patch_file.stat().st_size > 0:
                return patch_file

    print(f"  Patch not found for {agent}/{commit_short}")
    return None


def apply_patch(patch_path: Path) -> tuple[bool, str]:
    """Apply a git patch to the SGLang repo."""
    print(f"  Applying patch: {patch_path.name}")

    # First try normal apply
    returncode, stdout, stderr = run_command(
        ["git", "apply", str(patch_path)],
        cwd=SGLANG_REPO_DIR
    )

    if returncode == 0:
        return True, "Patch applied successfully"

    # Try with --3way for better merge handling
    print("  Patch failed, trying with --3way...")
    returncode, stdout, stderr = run_command(
        ["git", "apply", "--3way", str(patch_path)],
        cwd=SGLANG_REPO_DIR
    )

    if returncode == 0:
        return True, "Patch applied with 3way merge"

    # Try with ignore-whitespace for minor formatting differences
    print("  Trying with --ignore-whitespace...")
    returncode, stdout, stderr = run_command(
        ["git", "apply", "--ignore-whitespace", str(patch_path)],
        cwd=SGLANG_REPO_DIR
    )

    if returncode == 0:
        return True, "Patch applied (whitespace ignored)"

    # Try with --reject for partial success - at least get some changes in
    print("  Trying with --reject for partial apply...")
    returncode, stdout, stderr = run_command(
        ["git", "apply", "--reject", "--ignore-whitespace", str(patch_path)],
        cwd=SGLANG_REPO_DIR
    )

    if returncode == 0:
        return True, "Patch applied with some rejections"

    return False, f"Patch failed: {stderr[:200]}"


SERVER_PORT = 30000
SERVER_TIMEOUT = 300  # Wait up to 5 minutes for server to start (increased from 120s)


def get_benchmark_config(commit_short: str) -> dict:
    """Get benchmark configuration for a commit from PERF_COMMAND_CONFIG.

    For single-GPU commits: use the actual perf_command from dataset
    For multi-GPU commits: mark as 'skip' (requires DeepSeek-V3 or Llama-4-Maverick with TP>1)
    """
    # Check if this commit requires multi-GPU hardware (skip it)
    if commit_short in MULTI_GPU_COMMITS:
        config = PERF_COMMAND_CONFIG.get(commit_short, {}).copy()
        config["skip"] = True
        config["skip_reason"] = f"Requires multi-GPU (model: {config.get('model', 'unknown')})"
        return config

    if commit_short in PERF_COMMAND_CONFIG:
        config = PERF_COMMAND_CONFIG[commit_short].copy()
        # Verify single-GPU compatibility (safety check)
        if config["model"] not in SINGLE_GPU_MODELS:
            print(f"  WARNING: Model {config['model']} not in SINGLE_GPU_MODELS, marking as skip")
            config["skip"] = True
            config["skip_reason"] = f"Model {config['model']} requires multi-GPU"
        return config

    # Default config for unknown commits (use safe defaults)
    return {
        "bench_type": "serving",
        "model": FALLBACK_MODEL,
        "full_command": f"python -m sglang.bench_serving --backend sglang --model {FALLBACK_MODEL} --num-prompts 100",
    }


def start_sglang_server(venv_path: Path, model: str) -> Optional[subprocess.Popen]:
    """Start SGLang server in the background."""
    python_path = venv_path / "bin" / "python"

    # Create log file for server output (for debugging)
    log_file = venv_path / "server_startup.log"

    # Use sglang.launch_server module
    cmd = [
        str(python_path), "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(SERVER_PORT),
        "--trust-remote-code",
        "--attention-backend", "triton",
        "--log-level", "warning",
    ]

    print(f"  Starting server: {' '.join(cmd[:6])}...")
    print(f"  Server log: {log_file}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SGLANG_REPO_DIR / "python")
    env["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

    try:
        # Open log file for capturing stderr
        log_handle = open(log_file, 'w')

        proc = subprocess.Popen(
            cmd,
            cwd=SGLANG_REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=log_handle,  # Capture stderr to log file for debugging
            env=env,
            preexec_fn=os.setsid,  # Create new process group for clean kill
        )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < SERVER_TIMEOUT:
            # Check if process has died
            if proc.poll() is not None:
                log_handle.close()
                print(f"  Server process died with code {proc.returncode}")
                # Read and print log file for debugging
                if log_file.exists():
                    log_content = log_file.read_text()
                    print(f"  Server log (last 1000 chars):\n{log_content[-1000:]}")
                return None

            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', SERVER_PORT))
                sock.close()
                if result == 0:
                    print(f"  Server started on port {SERVER_PORT}")
                    return proc
            except:
                pass
            time.sleep(2)

        # Timeout - capture logs before killing
        log_handle.close()
        print(f"  Server failed to start within {SERVER_TIMEOUT}s")
        if log_file.exists():
            log_content = log_file.read_text()
            print(f"  Server log (last 1000 chars):\n{log_content[-1000:]}")
        stop_sglang_server(proc)
        return None

    except Exception as e:
        print(f"  Failed to start server: {e}")
        return None


def stop_sglang_server(proc: subprocess.Popen):
    """Stop the SGLang server process."""
    if proc is None:
        return

    try:
        # Kill entire process group
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except:
            pass


def run_serving_benchmark(venv_path: Path, commit_short: str, model: str, num_prompts: int = 50) -> dict:
    """Run bench_serving benchmark against a running server."""
    python_path = venv_path / "bin" / "python"

    # Build benchmark command
    cmd = [
        str(python_path), "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--host", "localhost",
        "--port", str(SERVER_PORT),
        "--model", model,
        "--num-prompts", str(num_prompts),
        "--dataset-name", "random",
        "--random-input", "256",
        "--random-output", "64",
    ]

    print(f"  Running serving benchmark: {' '.join(cmd[:8])}...")

    env = {
        "PYTHONPATH": str(SGLANG_REPO_DIR / "python"),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    }

    returncode, stdout, stderr = run_command(
        cmd,
        cwd=SGLANG_REPO_DIR,
        timeout=BENCHMARK_TIMEOUT,
        env=env
    )

    output = stdout + stderr

    # Keep both beginning (metrics) and end (progress) of output
    if len(output) > 6000:
        truncated_output = output[:3000] + "\n...[TRUNCATED]...\n" + output[-3000:]
    else:
        truncated_output = output

    # Get the perf_command from config for recording
    config = get_benchmark_config(commit_short)

    result = {
        "returncode": returncode,
        "raw_output": truncated_output,
        "bench_type": "serving",
        "perf_command": config.get("full_command", ""),
        "model": model,
    }

    if returncode != 0:
        result["status"] = "failed"
        result["error"] = stderr[-500:] if stderr else "Unknown error"
        return result

    result["status"] = "success"
    result["metrics"] = parse_benchmark_output(output)
    return result


def run_one_batch_benchmark(venv_path: Path, commit_short: str, model: str) -> dict:
    """Run bench_one_batch benchmark (standalone, no server needed)."""
    python_path = venv_path / "bin" / "python"

    # Discover the correct benchmark module
    benchmark_module = discover_benchmark_script()

    cmd = [
        str(python_path), "-m", benchmark_module,
        "--model-path", model,
        "--batch-size", str(BATCH_SIZE),
        "--input-len", str(INPUT_LEN),
        "--output-len", str(OUTPUT_LEN),
        "--load-format", "dummy",
        "--trust-remote-code",
        "--attention-backend", "triton",
        "--sampling-backend", "pytorch",
    ]

    print(f"  Running one_batch benchmark: {' '.join(cmd[:6])}...")

    env = {
        "PYTHONPATH": str(SGLANG_REPO_DIR / "python"),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    }

    returncode, stdout, stderr = run_command(
        cmd,
        cwd=SGLANG_REPO_DIR,
        timeout=BENCHMARK_TIMEOUT,
        env=env
    )

    output = stdout + stderr

    result = {
        "returncode": returncode,
        "raw_output": output[-3000:] if len(output) > 3000 else output,
        "bench_type": "one_batch",
        "benchmark_module": benchmark_module,
    }

    if returncode != 0:
        result["status"] = "failed"
        result["error"] = stderr[-500:] if stderr else "Unknown error"
        return result

    result["status"] = "success"
    result["metrics"] = parse_benchmark_output(output)
    return result


def run_benchmark(venv_path: Path, commit_short: str, use_dataset_config: bool = True) -> dict:
    """Run the appropriate benchmark based on commit config."""
    config = get_benchmark_config(commit_short)

    # Check if commit should be skipped (multi-GPU requirement)
    if config.get("skip"):
        print(f"  SKIPPING: {config.get('skip_reason', 'Unknown reason')}")
        return {
            "status": "skipped",
            "skip_reason": config.get("skip_reason"),
            "bench_type": config.get("bench_type", "unknown"),
            "model": config.get("model", "unknown"),
        }

    bench_type = config["bench_type"]
    model = config["model"]
    full_command = config.get("full_command", "")

    print(f"  Benchmark type: {bench_type}, Model: {model}")
    print(f"  Dataset perf_command: {full_command[:80]}...")

    if bench_type == "serving":
        # Start server, run benchmark, stop server
        server_proc = start_sglang_server(venv_path, model)
        if server_proc is None:
            # Try to read the server log for error details
            server_log = venv_path / "server_startup.log"
            server_error = "No log available"
            if server_log.exists():
                log_content = server_log.read_text()
                server_error = log_content[-1000:] if log_content else "Empty log"
            return {
                "status": "failed",
                "error": f"Failed to start SGLang server. Log: {server_error}",
                "bench_type": "serving",
            }

        try:
            result = run_serving_benchmark(venv_path, commit_short, model)
        finally:
            stop_sglang_server(server_proc)

        return result
    else:
        # one_batch benchmark (standalone)
        return run_one_batch_benchmark(venv_path, commit_short, model)


def parse_benchmark_output(output: str) -> dict:
    """Parse SGLang benchmark output to extract metrics."""
    metrics = {}

    # Pattern definitions for various SGLang benchmark output formats
    patterns = [
        # bench_latency.py / bench_one_batch.py format: "Total. ... throughput: 1234.56 token/s"
        (r"Total\.\s+.*?throughput:\s*([\d.]+)\s*token/s", "total_throughput"),
        (r"Decode.*?throughput:\s*([\d.]+)", "decode_throughput"),
        (r"Prefill.*?throughput:\s*([\d.]+)", "prefill_throughput"),

        # bench_serving.py format - exact patterns from typical output
        (r"Output token throughput \(tok/s\):\s*([\d.]+)", "output_token_throughput"),
        (r"Total token throughput \(tok/s\):\s*([\d.]+)", "total_token_throughput"),
        (r"Request throughput \(req/s\):\s*([\d.]+)", "request_throughput"),
        (r"Input token throughput \(tok/s\):\s*([\d.]+)", "input_token_throughput"),

        # bench_serving.py latency metrics
        (r"Mean TTFT \(ms\):\s*([\d.]+)", "mean_ttft_ms"),
        (r"Median TTFT \(ms\):\s*([\d.]+)", "median_ttft_ms"),
        (r"P99 TTFT \(ms\):\s*([\d.]+)", "p99_ttft_ms"),
        (r"Mean ITL \(ms\):\s*([\d.]+)", "mean_itl_ms"),
        (r"Median ITL \(ms\):\s*([\d.]+)", "median_itl_ms"),
        (r"P99 ITL \(ms\):\s*([\d.]+)", "p99_itl_ms"),
        (r"Mean E2E Latency \(ms\):\s*([\d.]+)", "mean_e2e_latency_ms"),
        (r"Median E2E Latency \(ms\):\s*([\d.]+)", "median_e2e_latency_ms"),

        # bench_serving.py request metrics
        (r"Successful requests:\s*([\d.]+)", "successful_requests"),
        (r"Benchmark duration \(s\):\s*([\d.]+)", "benchmark_duration_s"),

        # Legacy patterns (case-insensitive)
        (r"output_token_throughput:\s*([\d.]+)", "output_token_throughput"),
        (r"request_throughput:\s*([\d.]+)", "request_throughput"),
        (r"median_e2e_latency_ms:\s*([\d.]+)", "e2e_latency_ms"),
        (r"median_ttft_ms:\s*([\d.]+)", "ttft_ms"),
        (r"median_itl_ms:\s*([\d.]+)", "itl_ms"),

        # Fallback patterns
        (r"Throughput:\s+([\d.]+)\s*tokens?/s", "throughput_tokens_per_sec"),
        (r"throughput:\s+([\d.]+)", "throughput"),
        (r"Output throughput:\s+([\d.]+)", "output_throughput"),
        (r"Latency:\s+([\d.]+)\s*ms", "latency_ms"),
    ]

    for pattern, key in patterns:
        match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                pass

    # Add a primary throughput metric for summary display
    if "output_token_throughput" in metrics:
        metrics["throughput_tokens_per_sec"] = metrics["output_token_throughput"]
    elif "total_token_throughput" in metrics:
        metrics["throughput_tokens_per_sec"] = metrics["total_token_throughput"]
    elif "total_throughput" in metrics:
        metrics["throughput_tokens_per_sec"] = metrics["total_throughput"]

    # Return error indication if no metrics found
    if not metrics:
        # Check for common error indicators
        if "error" in output.lower() or "traceback" in output.lower():
            metrics["status"] = "error_detected"
        else:
            metrics["status"] = "no_metrics_found"

    return metrics


def run_variant(commit_short: str, variant: str, venv_path: Path,
                commit_info: dict, index_mapping: dict, mapping: dict) -> dict:
    """Run a single benchmark variant for a commit."""
    result = {
        "commit": commit_short,
        "variant": variant,
        "timestamp": datetime.now().isoformat(),
    }

    # Checkout the appropriate commit
    # For agent variants (claude_code, codex, etc.), checkout BASE commit since
    # the agent patches were generated comparing base->agent, not human->agent
    if variant == "baseline":
        target_commit = commit_info["base_commit"]
    elif variant == "human":
        target_commit = commit_info["human_commit"]
    else:
        # Agent variants: checkout base commit, then apply full agent patch
        target_commit = commit_info["base_commit"]

    if not checkout_commit(target_commit):
        result["status"] = "failed"
        result["error"] = f"Failed to checkout {target_commit[:12]}"
        return result

    # Apply agent patch if needed
    if variant not in ["baseline", "human"]:
        patch_path = find_agent_patch(commit_short, variant, index_mapping, mapping)
        if patch_path:
            success, msg = apply_patch(patch_path)
            result["patch_applied"] = success
            result["patch_message"] = msg
            if not success:
                result["status"] = "patch_crashed"
                result["error"] = msg
                return result
        else:
            result["status"] = "no_patch"
            result["error"] = f"No patch found for {variant}"
            return result

    # Install SGLang with era-appropriate dependencies
    if not install_sglang(venv_path, commit_short):
        result["status"] = "install_failed"
        result["error"] = "Failed to install SGLang"
        return result

    # Run benchmark using config from dataset
    bench_result = run_benchmark(venv_path, commit_short)
    result.update(bench_result)

    return result


def run_commit_benchmarks(commit_short: str, variants: list[str],
                          mapping: dict, index_mapping: dict,
                          dry_run: bool = False) -> dict:
    """Run all benchmark variants for a single commit."""
    if commit_short not in mapping:
        return {"error": f"Commit {commit_short} not found in mapping"}

    commit_info = mapping[commit_short]

    print(f"\n{'='*70}")
    print(f"COMMIT: {commit_short}")
    print(f"PR: #{commit_info.get('pr_number', '?')}")
    print(f"Subject: {commit_info.get('subject', 'Unknown')[:60]}...")
    print(f"Variants: {', '.join(variants)}")
    print(f"{'='*70}")

    if dry_run:
        print("[DRY RUN] Would run benchmarks for this commit")
        return {"status": "dry_run", "commit": commit_short}

    results = {
        "commit": commit_short,
        "full_commit": commit_info["human_commit"],
        "base_commit": commit_info["base_commit"],
        "pr_number": commit_info.get("pr_number"),
        "subject": commit_info.get("subject"),
        "timestamp": datetime.now().isoformat(),
        "variants": {}
    }

    # Create venv for this commit
    venv_path = create_venv(commit_short)
    if not venv_path:
        results["error"] = "Failed to create venv"
        return results

    try:
        for i, variant in enumerate(variants, 1):
            print(f"\n[{i}/{len(variants)}] Running {variant.upper()} variant...")

            variant_result = run_variant(
                commit_short, variant, venv_path,
                commit_info, index_mapping, mapping
            )
            results["variants"][variant] = variant_result

            # Save intermediate results
            save_results(results, commit_short)

            status = variant_result.get("status", "unknown")
            if status == "success":
                metrics = variant_result.get("metrics", {})
                throughput = metrics.get("throughput_tokens_per_sec") or metrics.get("output_throughput") or "N/A"
                print(f"  {variant.upper()}: SUCCESS - {throughput} tokens/s")
            else:
                print(f"  {variant.upper()}: {status.upper()} - {variant_result.get('error', 'Unknown')[:50]}")

    finally:
        # Cleanup venv to free space
        print(f"\nCleaning up venv at {venv_path}")
        if venv_path.exists():
            shutil.rmtree(venv_path)

    return results


def save_results(results: dict, commit_short: str):
    """Save results to JSON file, merging with existing results if present."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RESULTS_DIR / f"{commit_short}_isolated.json"

    # Merge with existing results if present
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
        # Merge variants
        for variant, vresult in results.get("variants", {}).items():
            existing.setdefault("variants", {})[variant] = vresult
        results = existing

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {output_file}")


def generate_summary_report(all_results: list[dict], dry_run: bool = False):
    """Generate a summary report of all benchmarks."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for results in all_results:
        commit = results.get("commit", "?")
        pr = results.get("pr_number", "?")

        if "error" in results:
            print(f"\n{commit} (PR#{pr}): ERROR - {results['error']}")
            continue

        if results.get("status") == "dry_run":
            print(f"\n{commit} (PR#{pr}): DRY RUN")
            continue

        print(f"\n{commit} (PR#{pr}):")
        for variant, vresult in results.get("variants", {}).items():
            status = vresult.get("status", "unknown")
            if status == "success":
                metrics = vresult.get("metrics", {})
                throughput = (
                    metrics.get("throughput_tokens_per_sec") or
                    metrics.get("output_throughput") or
                    metrics.get("throughput") or
                    "N/A"
                )
                print(f"  {variant:15s}: {throughput} tokens/s")
            else:
                print(f"  {variant:15s}: {status.upper()}")

    # Save summary (skip in dry run)
    if dry_run:
        print("\n[DRY RUN] Summary would be saved to: " + str(RESULTS_DIR / "summary.json"))
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_commits": len(all_results),
            "results": all_results
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SGLang Per-Commit Isolated Benchmark Runner"
    )
    parser.add_argument("--commit", type=str, help="Specific commit (8-char)")
    parser.add_argument("--all", action="store_true", help="Benchmark all target commits")
    parser.add_argument("--variants", type=str,
                        default="baseline,human,claude_code,codex,trae_gpt5,trae_sonnet45",
                        help="Comma-separated variants to run")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--list", action="store_true", help="List target commits")
    parser.add_argument("--skip-clone", action="store_true", help="Skip repo cloning")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip already completed commits")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint (skip completed variants)")

    args = parser.parse_args()

    # Load commit mapping
    mapping, index_mapping = load_commit_mapping()

    # Check GPU compatibility (unless just listing or dry run)
    if not args.list and not args.dry_run:
        print("Checking GPU compatibility...")
        if not check_gpu_compatibility():
            print("WARNING: GPU check failed, proceeding anyway...")

    if args.list:
        print("\nTarget commits:")
        for commit in TARGET_COMMITS:
            info = mapping.get(commit, {})
            pr = info.get("pr_number", "?")
            subject = info.get("subject", "Unknown")[:50]
            status = " [DONE]" if commit in COMPLETED_COMMITS else (" [SKIP]" if commit in SKIP_COMMITS else "")
            print(f"  {commit}: PR#{pr} - {subject}{status}")

        print(f"\nTotal: {len(TARGET_COMMITS)} commits")
        print(f"Completed: {len(COMPLETED_COMMITS)} commits")
        print(f"Skipped: {len(SKIP_COMMITS)} commits")
        return

    # Determine which commits to run
    commits_to_run = []
    if args.all:
        commits_to_run = TARGET_COMMITS.copy()
        if args.skip_completed:
            commits_to_run = [c for c in commits_to_run if c not in COMPLETED_COMMITS and c not in SKIP_COMMITS]
    elif args.commit:
        commits_to_run = [args.commit]
    else:
        parser.print_help()
        return

    # Parse variants
    variants = [v.strip() for v in args.variants.split(",")]

    print(f"\nCommits to benchmark: {len(commits_to_run)}")
    print(f"Variants per commit: {len(variants)}")
    print(f"Total benchmarks: {len(commits_to_run) * len(variants)}")

    # Clone repo if needed
    if not args.skip_clone and not args.dry_run:
        if not clone_sglang_repo():
            print("ERROR: Failed to clone SGLang repo")
            sys.exit(1)

    # Run benchmarks
    all_results = []
    for i, commit in enumerate(commits_to_run, 1):
        print(f"\n[PROGRESS: {i}/{len(commits_to_run)}]")

        # Check for resume
        if args.resume:
            completed, missing = is_commit_completed(commit, variants)
            if completed:
                print(f"Skipping {commit} - all variants completed")
                # Load existing results
                result_file = RESULTS_DIR / f"{commit}_isolated.json"
                with open(result_file) as f:
                    all_results.append(json.load(f))
                continue
            elif missing != variants:
                print(f"Resuming {commit} - {len(missing)} variants remaining: {missing}")
                variants_to_run = missing
            else:
                variants_to_run = variants
        else:
            variants_to_run = variants

        try:
            results = run_commit_benchmarks(
                commit, variants_to_run, mapping, index_mapping, args.dry_run
            )
            all_results.append(results)
        except Exception as e:
            print(f"ERROR processing {commit}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"commit": commit, "error": str(e)})

    # Generate summary
    generate_summary_report(all_results, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
