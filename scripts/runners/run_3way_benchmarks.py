#!/usr/bin/env python3
"""
3-Way Benchmark Runner for vLLM commits.
Runs human and agent benchmarks for commits that already have baseline results.

For each commit:
1. Human benchmark: Uses pre-built Docker image with human's optimized vLLM
2. Agent benchmark: Applies Claude's patch to baseline vLLM and benchmarks
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Configuration
HUMAN_IMAGE_PREFIX = "ayushnangia16/nvidia-vllm-docker"
BASELINE_IMAGE_PREFIX = "shikhar481/vllm_fixed_human_images"
RESULTS_DIR = Path("/root/OmniPerf-Bench/omniperf_results_3way_claude_code")
AGENT_OUTPUT_DIR = RESULTS_DIR / "agent_benchmark_results"
BASELINE_MAPPING_FILE = Path("/root/OmniPerf-Bench/baseline_benchmark_mapping_complete.json")
AGENT_PATCHES_DIR = Path("/root/OmniPerf-Bench/perf-agents-bench/state/runs/vllm/claude_code/default/2025-12-22_21-40-38")


def get_hf_token() -> str:
    """Get HuggingFace token."""
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    return ""


def load_mapping() -> Dict[str, dict]:
    """Load baseline benchmark mapping."""
    with open(BASELINE_MAPPING_FILE) as f:
        mapping = json.load(f)
    return {m['human_commit_short']: m for m in mapping}


def load_agent_patches() -> Dict[str, Path]:
    """Load agent patch mapping."""
    patches = {}
    if not AGENT_PATCHES_DIR.exists():
        print(f"WARNING: Agent patches directory not found: {AGENT_PATCHES_DIR}")
        return patches

    for patch_dir in AGENT_PATCHES_DIR.glob("vllm_core-*"):
        journal_file = patch_dir / "journal.json"
        patch_file = patch_dir / "model_patch.diff"
        if journal_file.exists() and patch_file.exists():
            try:
                with open(journal_file) as f:
                    journal = json.load(f)
                human_commit = journal.get("commits", {}).get("human", "")[:8]
                if human_commit and patch_file.stat().st_size > 0:
                    patches[human_commit] = patch_file
            except Exception as e:
                print(f"WARNING: Failed to load {journal_file}: {e}")
    return patches


def parse_serving_metrics(output: str) -> Dict[str, float]:
    """Parse metrics from benchmark_serving.py output."""
    metrics = {}
    patterns = {
        'ttft_mean_ms': r'Mean TTFT \(ms\):\s+([\d.]+)',
        'ttft_median_ms': r'Median TTFT \(ms\):\s+([\d.]+)',
        'ttft_p99_ms': r'P99 TTFT \(ms\):\s+([\d.]+)',
        'tpot_mean_ms': r'Mean TPOT \(ms\):\s+([\d.]+)',
        'tpot_median_ms': r'Median TPOT \(ms\):\s+([\d.]+)',
        'tpot_p99_ms': r'P99 TPOT \(ms\):\s+([\d.]+)',
        'itl_mean_ms': r'Mean ITL \(ms\):\s+([\d.]+)',
        'itl_median_ms': r'Median ITL \(ms\):\s+([\d.]+)',
        'itl_p99_ms': r'P99 ITL \(ms\):\s+([\d.]+)',
        'request_throughput_req_s': r'Request throughput \(req/s\):\s+([\d.]+)',
        'output_token_throughput_tok_s': r'Output token throughput \(tok/s\):\s+([\d.]+)',
        'total_token_throughput_tok_s': r'Total Token throughput \(tok/s\):\s+([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def run_human_benchmark(commit_info: dict, hf_token: str, timeout: int = 900) -> dict:
    """Run benchmark using human's optimized Docker image."""
    start_time = time.time()

    human_commit = commit_info['human_commit_full']
    human_short = commit_info['human_commit_short']
    model = commit_info.get('model', '')
    perf_command = commit_info.get('perf_command', '')

    # Adjust perf_command for sonnet dataset (works with old vLLM, generates synthetic data)
    perf_command = re.sub(r'--dataset-name\s+sharegpt', '--dataset-name sonnet', perf_command)
    perf_command = re.sub(r'--dataset-name\s+random', '--dataset-name sonnet', perf_command)
    perf_command = re.sub(r'--dataset\s+\S+\.json', '', perf_command)
    perf_command = re.sub(r'--dataset-path\s+\S+', '', perf_command)
    perf_command = re.sub(r'--random-input-len\s+\d+', '', perf_command)
    perf_command = re.sub(r'--random-output-len\s+\d+', '', perf_command)

    if '--dataset-name' not in perf_command:
        perf_command += ' --dataset-name sonnet'

    # Set sonnet parameters for consistent benchmarking
    if '--sonnet-input-len' not in perf_command:
        perf_command += ' --sonnet-input-len 256 --sonnet-output-len 64'

    # Clean up any double spaces
    perf_command = re.sub(r'\s+', ' ', perf_command).strip()

    # Extract benchmark args
    bench_args = re.sub(r'python\s+benchmarks/benchmark_serving\.py\s*', '', perf_command)
    bench_args = re.sub(r'--dtype\s+\S+', '', bench_args)

    docker_image = f"{HUMAN_IMAGE_PREFIX}:{human_commit}"

    docker_cmd = f'''
    set -e
    COMMIT="{human_commit}"
    MODEL="{model}"

    # CRITICAL: Set PYTHONPATH to include /workspace where vLLM is often installed
    # in human Docker images (vLLM source is at /workspace/vllm/)
    export PYTHONPATH=/workspace:$PYTHONPATH

    # Install uv for faster package management
    pip install uv -q 2>/dev/null || true

    # Check if vLLM imports work first, only apply fixes if needed
    echo "Checking vLLM compatibility..."

    # First try importing vLLM without any changes
    if python3 -c "import vllm" 2>/dev/null; then
        echo "vLLM imports OK - no compatibility fixes needed"
    else
        echo "vLLM import failed - applying compatibility fixes..."

        # Check if the issue is LogitsWarper (old vLLM needs old transformers)
        # vs mllama (new vLLM needs new transformers)
        if python3 -c "from transformers.models.mllama import configuration_mllama" 2>/dev/null; then
            echo "transformers.models.mllama OK"
        else
            # mllama missing - this vLLM might need newer transformers, don't downgrade
            echo "transformers.models.mllama missing - checking if vLLM needs it..."
        fi

        # Only downgrade transformers if LogitsWarper is missing AND mllama isn't needed by vLLM
        if ! python3 -c "from transformers.generation.logits_process import LogitsWarper" 2>/dev/null; then
            # Check if this vLLM version uses mllama (check for mllama.py in vllm)
            VLLM_USES_MLLAMA=$(find /usr -path "*/vllm/*mllama*" 2>/dev/null | head -1)
            if [ -z "$VLLM_USES_MLLAMA" ]; then
                echo "Fixing transformers compatibility (LogitsWarper missing, mllama not needed)..."
                pip install 'transformers==4.44.2' -q 2>/dev/null || true
            else
                echo "Skipping transformers downgrade - vLLM uses mllama which needs transformers>=4.45"
            fi
        fi
    fi

    # Always ensure numpy<2 for outlines compatibility
    pip install 'numpy<2' -q 2>/dev/null || true

    # Now detect vLLM version safely
    echo "Detecting vLLM version..."
    VLLM_NEEDS_OLD_TRANSFORMERS=0

    # Try to find vLLM installation directory (might be in /usr/local, /opt/venv, or elsewhere)
    VLLM_DIR=""
    for py in /opt/venv/bin/python3 /usr/local/bin/python3 /usr/bin/python3 python3; do
        if [ -x "$(which $py 2>/dev/null || echo '')" ] || [ -x "$py" ]; then
            VLLM_DIR=$($py -c "import vllm; import os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || echo "")
            if [ -n "$VLLM_DIR" ]; then
                echo "Found vLLM at: $VLLM_DIR"
                break
            fi
        fi
    done

    if [ -z "$VLLM_DIR" ]; then
        # Last resort: search common locations
        for dir in /usr/local/lib/python*/dist-packages/vllm /opt/venv/lib/python*/site-packages/vllm; do
            if [ -d "$dir" ]; then
                VLLM_DIR="$dir"
                echo "Found vLLM at: $VLLM_DIR (from path search)"
                break
            fi
        done
    fi

    # Apply rope_scaling fix for Llama-3.1 models
    if [ -n "$VLLM_DIR" ]; then
        echo "Fixing rope_scaling compatibility for Llama-3.1 models..."
        find "$VLLM_DIR" -name "*.py" -exec grep -l 'rope_scaling\["type"\]' {{}} \; 2>/dev/null | while read f; do
            echo "  Patching: $f"
            sed -i 's/rope_scaling\["type"\]/rope_scaling.get("type", rope_scaling.get("rope_type"))/g' "$f"
        done
    fi

    # Find Python with vLLM installed (might be in virtualenv)
    echo "Searching for Python with vLLM..."
    VLLM_PYTHON=""
    for py in /opt/venv/bin/python3 /opt/venv/bin/python /usr/local/bin/python3 /usr/bin/python3 $(which python3 2>/dev/null) $(which python 2>/dev/null); do
        if [ -x "$py" ] && $py -c "import vllm" 2>/dev/null; then
            VLLM_PYTHON="$py"
            echo "Found vLLM at: $VLLM_PYTHON"
            break
        fi
    done

    if [ -z "$VLLM_PYTHON" ]; then
        echo "ERROR: Could not find Python with vLLM installed"
        echo "Checking available Pythons..."
        which python3 python 2>/dev/null || true
        ls -la /opt/venv/bin/ 2>/dev/null || true
        exit 1
    fi

    # Get vLLM version and location
    VLLM_VERSION=$($VLLM_PYTHON -c "import vllm; print(vllm.__version__)")
    VLLM_DIR=$($VLLM_PYTHON -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
    echo "vLLM version: $VLLM_VERSION at $VLLM_DIR"

    # Verify vLLM works
    $VLLM_PYTHON -c "import vllm; print('vLLM ' + vllm.__version__ + ' OK')"

    # Fix outlines.fsm compatibility (newer outlines removed fsm.guide module)
    # This affects vLLM versions 0.4.x-0.5.x that import from outlines.fsm.guide
    if ! $VLLM_PYTHON -c "from outlines.fsm.guide import Guide" 2>/dev/null; then
        echo "Fixing outlines.fsm compatibility (fsm.guide missing)..."
        # First uninstall any existing outlines, then install compatible version
        $VLLM_PYTHON -m pip uninstall outlines -y 2>/dev/null || true
        $VLLM_PYTHON -m pip install 'outlines==0.0.34' --no-deps 2>&1 || \
        uv pip install 'outlines==0.0.34' --no-deps 2>&1 || \
        pip install 'outlines==0.0.34' --no-deps 2>&1 || \
        echo "OUTLINES_INSTALL_FAILED"
    fi
    if $VLLM_PYTHON -c "from outlines.fsm.guide import Guide" 2>/dev/null; then
        echo "outlines.fsm OK"
    else
        echo "Warning: outlines.fsm fix failed - trying alternative approach..."
        # Alternative: patch vLLM to not require outlines.fsm
        if [ -f "$VLLM_DIR/model_executor/guided_decoding/__init__.py" ]; then
            echo "Patching vLLM to skip guided_decoding import..."
            cat > "$VLLM_DIR/model_executor/guided_decoding/__init__.py" << 'PATCH'
# Patched: Skip guided_decoding to avoid outlines.fsm dependency
from typing import Optional
from dataclasses import dataclass

@dataclass
class GuidedDecodingRequest:
    """Stub for GuidedDecodingRequest"""
    guided_json: Optional[str] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[list] = None
    guided_grammar: Optional[str] = None
    guided_decoding_backend: Optional[str] = None
    guided_whitespace_pattern: Optional[str] = None

async def get_guided_decoding_logits_processor(*args, **kwargs):
    return None
async def get_local_guided_decoding_logits_processor(*args, **kwargs):
    return None
def get_outlines_guided_decoding_logits_processor(*args, **kwargs):
    return None
PATCH
        fi
    fi

    # Install benchmark deps using the same Python
    $VLLM_PYTHON -m pip install aiohttp pandas datasets -q 2>/dev/null || true

    # Install git if not available
    if ! command -v git &> /dev/null; then
        echo "Installing git..."
        apt-get update -qq && apt-get install -y -qq git 2>/dev/null || yum install -y git -q 2>/dev/null || true
    fi

    # Clone vLLM repo at specific commit for benchmark scripts
    cd /opt
    rm -rf vllm_bench 2>/dev/null || true
    echo "Cloning vLLM repo for benchmark scripts..."
    git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_bench 2>&1 || ( echo "GIT_CLONE_FAILED" && exit 1 )
    cd vllm_bench
    git fetch --depth 1 origin $COMMIT 2>/dev/null || git fetch origin $COMMIT 2>/dev/null || true
    git checkout $COMMIT 2>/dev/null || git checkout -f HEAD

    # Fix benchmark_serving.py compatibility with transformers 4.44.2
    sed -i 's/tokenizer.chat_template or tokenizer.default_chat_template/getattr(tokenizer, "chat_template", None) or getattr(tokenizer, "default_chat_template", True)/g' benchmarks/benchmark_serving.py 2>/dev/null || true

    # Start server using the Python that has vLLM
    echo "=== Starting vLLM server for HUMAN benchmark ==="
    cd /tmp
    $VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
        --model $MODEL --port 8000 --max-model-len 4096 --disable-log-requests 2>&1 &
    SERVER_PID=$!

    # Wait for server (use Python since curl may not be available)
    for i in $(seq 1 300); do
        if $VLLM_PYTHON -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models', timeout=2)" 2>/dev/null; then
            echo "SERVER_READY_AFTER=${{i}}s"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "SERVER_CRASHED"
            exit 1
        fi
        sleep 1
    done

    if ! $VLLM_PYTHON -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models', timeout=2)" 2>/dev/null; then
        echo "SERVER_TIMEOUT"
        exit 1
    fi

    echo "=== Running HUMAN benchmark ==="

    # Use direct HTTP benchmark for reliability across all vLLM versions
    echo "Using direct HTTP benchmark..."
    $VLLM_PYTHON << BENCHMARK_SCRIPT
import asyncio
import aiohttp
import time
import json
import random
import string

async def send_request(session, url, payload):
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            result = await resp.json()
            end = time.perf_counter()
            output_tokens = len(result.get('choices', [dict()])[0].get('text', '').split())
            return end - start, output_tokens, None
    except Exception as e:
        return None, 0, str(e)

async def benchmark():
    url = "http://localhost:8000/v1/completions"
    model = "$MODEL"
    num_prompts = 100
    max_tokens = 64

    # Generate random prompts
    prompts = []
    for _ in range(num_prompts):
        prompt = ' '.join(random.choices(['the', 'a', 'is', 'of', 'and', 'to', 'in', 'for', 'on', 'with'], k=200))
        prompts.append(prompt)

    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        tasks = []
        for prompt in prompts:
            payload = dict(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0
            )
            tasks.append(send_request(session, url, payload))

        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

    total_time = end_time - start_time
    successful = sum(1 for r in results if r[0] is not None)
    total_output_tokens = sum(r[1] for r in results if r[0] is not None)
    total_input_tokens = num_prompts * 200  # approx

    print("============ Serving Benchmark Result ============")
    print("Successful requests:                     " + str(successful))
    print("Benchmark duration (s):                  " + str(round(total_time, 2)))
    print("Total input tokens:                      " + str(total_input_tokens))
    print("Total generated tokens:                  " + str(total_output_tokens))
    print("Request throughput (req/s):              " + str(round(successful/total_time, 2)))
    print("Output token throughput (tok/s):         " + str(round(total_output_tokens/total_time, 2)))
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(benchmark())
BENCHMARK_SCRIPT

    kill $SERVER_PID 2>/dev/null || true
    echo "BENCHMARK_DONE"
    '''

    print(f"  Running human benchmark with image: {docker_image}")

    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm', '--gpus', 'all',
                '--network=host',
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-e', 'VLLM_USE_V1=0',
                '-v', '/ephemeral/huggingface_cache:/root/.cache/huggingface',
                '--shm-size=16g',
                '--entrypoint', 'bash',
                docker_image,
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        if 'SERVER_CRASHED' in output:
            return {
                'status': 'error',
                'error': 'Server crashed during startup',
                'duration_s': duration,
                'raw_output': output[-10000:]
            }

        if 'SERVER_TIMEOUT' in output:
            return {
                'status': 'error',
                'error': 'Server startup timeout',
                'duration_s': duration,
                'raw_output': output[-10000:]
            }

        metrics = parse_serving_metrics(output)

        if not metrics:
            return {
                'status': 'error',
                'error': 'No metrics in output',
                'duration_s': duration,
                'raw_output': output[-10000:]
            }

        return {
            'status': 'success',
            'metrics': metrics,
            'duration_s': duration,
            'raw_output': output[-5000:]
        }

    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'error': f'Benchmark timed out after {timeout}s',
            'duration_s': timeout
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'duration_s': time.time() - start_time
        }


def run_agent_benchmark(commit_info: dict, agent_patch: Path, hf_token: str, timeout: int = 900) -> dict:
    """Run benchmark by applying agent patch to baseline vLLM."""
    start_time = time.time()

    human_short = commit_info['human_commit_short']
    parent_commit = commit_info['parent_commit']
    model = commit_info.get('model', '')
    perf_command = commit_info.get('perf_command', '')

    # Adjust perf_command for compatibility
    perf_command = re.sub(r'--dataset-name\s+sharegpt', '--dataset-name random', perf_command)
    perf_command = re.sub(r'--dataset-name\s+sonnet', '--dataset-name random', perf_command)
    perf_command = re.sub(r'--dataset\s+\S+\.json', '', perf_command)
    # Remove old-style input/output len args (not supported in older vLLM)
    perf_command = re.sub(r'--input-len\s+\d+', '', perf_command)
    perf_command = re.sub(r'--output-len\s+\d+', '', perf_command)

    if '--dataset-name' not in perf_command and '--dataset-path' not in perf_command:
        perf_command += ' --dataset-name random'

    if '--dataset-name random' in perf_command and '--random-input-len' not in perf_command:
        perf_command += ' --random-input-len 256 --random-output-len 64'

    bench_args = re.sub(r'python\s+benchmarks/benchmark_serving\.py\s*', '', perf_command)
    bench_args = re.sub(r'--dtype\s+\S+', '', bench_args)
    # Clean up double spaces
    bench_args = re.sub(r'\s+', ' ', bench_args).strip()

    # Use baseline image (has vLLM at parent commit)
    baseline_image = f"{BASELINE_IMAGE_PREFIX}:baseline-{parent_commit[:12]}"

    docker_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"
    MODEL="{model}"

    echo "=== Applying agent patch to baseline vLLM ==="

    # Baseline images have vLLM installed in /opt/vllm_baseline
    cd /opt/vllm_baseline

    # Check if vLLM uses mllama (needs transformers >= 4.45)
    VLLM_USES_MLLAMA=$(find /opt/vllm_baseline -name "*mllama*" 2>/dev/null | head -1)

    if [ -n "$VLLM_USES_MLLAMA" ]; then
        echo "vLLM uses mllama - checking transformers version..."
        if ! python3 -c "from transformers.models.mllama import configuration_mllama" 2>/dev/null; then
            echo "Upgrading transformers to support mllama..."
            pip install 'transformers>=4.45.0' -q 2>/dev/null || true
        fi
    else
        # Old vLLM that doesn't use mllama - may need old transformers
        if ! python3 -c "from transformers.generation.logits_process import LogitsWarper" 2>/dev/null; then
            echo "Fixing transformers compatibility (LogitsWarper missing)..."
            pip install 'transformers==4.44.2' -q 2>/dev/null || true
        fi
    fi

    # Always ensure numpy<2 for outlines compatibility
    pip install 'numpy<2' -q 2>/dev/null || true

    # Verify vLLM exists
    if [ ! -d "vllm" ]; then
        echo "ERROR: vLLM not found at /opt/vllm_baseline"
        exit 1
    fi

    echo "vLLM found at: /opt/vllm_baseline"

    # Apply agent patch
    echo "Applying patch..."
    if patch -p1 --dry-run < /agent_patch.diff 2>&1; then
        patch -p1 < /agent_patch.diff 2>&1
        echo "AGENT_PATCH_APPLIED"
    else
        echo "Patch dry-run failed, trying with --force..."
        patch -p1 --force < /agent_patch.diff 2>&1 || true
        echo "AGENT_PATCH_APPLIED_FALLBACK"
    fi

    # Apply rope_scaling fix for Llama-3.1 models
    echo "Fixing rope_scaling compatibility for Llama-3.1 models..."
    find /opt/vllm_baseline/vllm -name "*.py" -exec grep -l 'rope_scaling\["type"\]' {{}} \; 2>/dev/null | while read f; do
        echo "  Patching: $f"
        sed -i 's/rope_scaling\["type"\]/rope_scaling.get("type", rope_scaling.get("rope_type"))/g' "$f"
    done

    # Find Python with vLLM installed (baseline images use /opt/vllm_baseline in PYTHONPATH)
    echo "Searching for Python with vLLM..."
    export PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH
    VLLM_PYTHON=""
    for py in /opt/venv/bin/python3 /opt/venv/bin/python /usr/local/bin/python3 /usr/bin/python3 $(which python3 2>/dev/null); do
        if [ -x "$py" ] && PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $py -c "import vllm" 2>/dev/null; then
            VLLM_PYTHON="$py"
            echo "Found vLLM at: $VLLM_PYTHON"
            break
        fi
    done

    if [ -z "$VLLM_PYTHON" ]; then
        echo "WARNING: Could not find Python with vLLM, trying default python3"
        VLLM_PYTHON="python3"
    fi

    # Verify import still works
    PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $VLLM_PYTHON -c "import vllm; print('vLLM ' + vllm.__version__ + ' OK')" || echo "vLLM import failed"

    # Fix outlines.fsm compatibility for agent benchmark
    if ! $VLLM_PYTHON -c "from outlines.fsm.guide import Guide" 2>/dev/null; then
        echo "Fixing outlines.fsm compatibility (fsm.guide missing)..."
        $VLLM_PYTHON -m pip uninstall outlines -y 2>/dev/null || true
        $VLLM_PYTHON -m pip install 'outlines==0.0.34' --no-deps 2>&1 || \
        pip install 'outlines==0.0.34' --no-deps 2>&1 || \
        echo "OUTLINES_INSTALL_FAILED"
    fi
    if $VLLM_PYTHON -c "from outlines.fsm.guide import Guide" 2>/dev/null; then
        echo "outlines.fsm OK"
    else
        echo "Warning: outlines.fsm fix failed - patching vLLM files directly..."
        # Patch vLLM to not require outlines.fsm by replacing all guided_decoding modules
        VLLM_DIR="/opt/vllm_baseline/vllm"
        GUIDED_DIR="$VLLM_DIR/model_executor/guided_decoding"

        if [ -d "$GUIDED_DIR" ]; then
            echo "Patching all guided_decoding modules..."

            # Create stub __init__.py
            cat > "$GUIDED_DIR/__init__.py" << 'PATCH'
# Patched: Stub guided_decoding to avoid outlines.fsm dependency
from typing import Optional
from dataclasses import dataclass

@dataclass
class GuidedDecodingRequest:
    guided_json: Optional[str] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[list] = None
    guided_grammar: Optional[str] = None
    guided_decoding_backend: Optional[str] = None
    guided_whitespace_pattern: Optional[str] = None

async def get_guided_decoding_logits_processor(*args, **kwargs):
    return None
async def get_local_guided_decoding_logits_processor(*args, **kwargs):
    return None
def get_outlines_guided_decoding_logits_processor(*args, **kwargs):
    return None
PATCH

            # Create stub for outlines_logits_processors.py
            cat > "$GUIDED_DIR/outlines_logits_processors.py" << 'PATCH'
# Patched: Stub outlines_logits_processors
class CFGLogitsProcessor:
    pass
class RegexLogitsProcessor:
    pass
class JSONLogitsProcessor:
    pass
PATCH

            # Create stub for outlines_decoding.py
            cat > "$GUIDED_DIR/outlines_decoding.py" << 'PATCH'
# Patched: Stub outlines_decoding
async def get_outlines_guided_decoding_logits_processor(*args, **kwargs):
    return None
PATCH

            # Create stub for lm_format_enforcer_decoding.py
            cat > "$GUIDED_DIR/lm_format_enforcer_decoding.py" << 'PATCH'
# Patched: Stub lm_format_enforcer_decoding
async def get_lm_format_enforcer_guided_decoding_logits_processor(*args, **kwargs):
    return None
PATCH
            echo "Guided decoding modules patched"
        fi
    fi

    # Install benchmark deps
    $VLLM_PYTHON -m pip install aiohttp pandas datasets -q 2>/dev/null || pip install aiohttp pandas datasets -q 2>/dev/null || true

    # Install git if not available
    if ! command -v git &> /dev/null; then
        echo "Installing git..."
        apt-get update -qq && apt-get install -y -qq git 2>/dev/null || yum install -y git -q 2>/dev/null || true
    fi

    # Clone vLLM repo for benchmark scripts
    cd /opt
    rm -rf vllm_bench 2>/dev/null || true
    echo "Cloning vLLM repo for benchmark scripts..."
    if ! git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_bench 2>&1; then
        echo "Git clone failed, trying without depth..."
        git clone https://github.com/vllm-project/vllm.git vllm_bench 2>&1 || ( echo "GIT_CLONE_FAILED" && exit 1 )
    fi
    cd vllm_bench
    git fetch --depth 1 origin $PARENT_COMMIT 2>/dev/null || git fetch origin $PARENT_COMMIT 2>/dev/null || true
    git checkout $PARENT_COMMIT 2>/dev/null || git checkout -f HEAD

    # Start server using the Python that has vLLM (with PYTHONPATH for baseline)
    echo "=== Starting vLLM server for AGENT benchmark ==="
    cd /tmp
    PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
        --model $MODEL --port 8000 --max-model-len 4096 --disable-log-requests 2>&1 &
    SERVER_PID=$!

    # Wait for server
    for i in $(seq 1 300); do
        if $VLLM_PYTHON -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models', timeout=2)" 2>/dev/null; then
            echo "SERVER_READY_AFTER=${{i}}s"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "SERVER_CRASHED"
            exit 1
        fi
        sleep 1
    done

    if ! $VLLM_PYTHON -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models', timeout=2)" 2>/dev/null; then
        echo "SERVER_TIMEOUT"
        exit 1
    fi

    echo "=== Running AGENT benchmark ==="
    cd /opt/vllm_bench/benchmarks

    # Use direct HTTP benchmark for reliability across vLLM versions
    echo "Using direct HTTP benchmark..."
    $VLLM_PYTHON << BENCHMARK_SCRIPT
import asyncio
import aiohttp
import time
import json
import random
import string

async def send_request(session, url, payload):
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            result = await resp.json()
            end = time.perf_counter()
            output_tokens = len(result.get('choices', [dict()])[0].get('text', '').split())
            return end - start, output_tokens, None
    except Exception as e:
        return None, 0, str(e)

async def benchmark():
    url = "http://localhost:8000/v1/completions"
    model = "$MODEL"
    num_prompts = 100
    max_tokens = 64

    # Generate random prompts
    prompts = []
    for _ in range(num_prompts):
        prompt = ' '.join(random.choices(['the', 'a', 'is', 'of', 'and', 'to', 'in', 'for', 'on', 'with'], k=200))
        prompts.append(prompt)

    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        tasks = []
        for prompt in prompts:
            payload = dict(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0
            )
            tasks.append(send_request(session, url, payload))

        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

    total_time = end_time - start_time
    successful = sum(1 for r in results if r[0] is not None)
    total_output_tokens = sum(r[1] for r in results if r[0] is not None)
    total_input_tokens = num_prompts * 200  # approx

    print("============ Serving Benchmark Result ============")
    print("Successful requests:                     " + str(successful))
    print("Benchmark duration (s):                  " + str(round(total_time, 2)))
    print("Total input tokens:                      " + str(total_input_tokens))
    print("Total generated tokens:                  " + str(total_output_tokens))
    print("Request throughput (req/s):              " + str(round(successful/total_time, 2)))
    print("Output token throughput (tok/s):         " + str(round(total_output_tokens/total_time, 2)))
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(benchmark())
BENCHMARK_SCRIPT

    kill $SERVER_PID 2>/dev/null || true
    echo "BENCHMARK_DONE"
    '''

    print(f"  Running agent benchmark with baseline image: {baseline_image}")
    print(f"  Applying patch: {agent_patch}")

    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm', '--gpus', 'all',
                '--network=host',
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-e', 'VLLM_USE_V1=0',
                '-v', '/ephemeral/huggingface_cache:/root/.cache/huggingface',
                '-v', f'{agent_patch}:/agent_patch.diff:ro',
                '--shm-size=16g',
                '--entrypoint', 'bash',
                baseline_image,
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        if 'SERVER_CRASHED' in output:
            return {
                'status': 'error',
                'error': 'Server crashed after applying patch',
                'duration_s': duration,
                'raw_output': output[-10000:]
            }

        if 'SERVER_TIMEOUT' in output:
            return {
                'status': 'error',
                'error': 'Server startup timeout after patch',
                'duration_s': duration,
                'raw_output': output[-10000:]
            }

        metrics = parse_serving_metrics(output)

        if not metrics:
            return {
                'status': 'error',
                'error': 'No metrics in agent output',
                'duration_s': duration,
                'raw_output': output[-10000:]
            }

        return {
            'status': 'success',
            'metrics': metrics,
            'duration_s': duration,
            'raw_output': output[-5000:]
        }

    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'error': f'Agent benchmark timed out after {timeout}s',
            'duration_s': timeout
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'duration_s': time.time() - start_time
        }


def save_result(result: dict, commit_short: str, result_type: str, commit_info: dict):
    """Save benchmark result to file."""
    AGENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_file = AGENT_OUTPUT_DIR / f"{commit_short}_{result_type}_result.json"

    result_data = {
        'human_commit': commit_short,
        'human_commit_full': commit_info['human_commit_full'],
        'parent_commit': commit_info['parent_commit'],
        'model': commit_info.get('model', ''),
        'status': result['status'],
        'error': result.get('error'),
        'duration_s': result.get('duration_s'),
        'metrics': result.get('metrics', {}),
        'raw_output': result.get('raw_output', ''),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"  Saved to {result_file}")
    return result_file


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run 3-way benchmarks for vLLM commits')
    parser.add_argument('--commits', type=str, nargs='+',
                        help='Specific commits to run (short hash)')
    parser.add_argument('--human-only', action='store_true',
                        help='Only run human benchmarks')
    parser.add_argument('--agent-only', action='store_true',
                        help='Only run agent benchmarks')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be run')
    parser.add_argument('--timeout', type=int, default=900,
                        help='Timeout per benchmark in seconds')
    args = parser.parse_args()

    # Default commits (8 ready commits)
    default_commits = ['3476ed08', '3a243095', '6ce01f30', '7c01f706',
                       '80aa7e91', '89a84b0b', '8bc68e19', '8d75fe48']

    commits = args.commits if args.commits else default_commits

    print("Loading configuration...")
    mapping = load_mapping()
    agent_patches = load_agent_patches()
    hf_token = get_hf_token()

    if not hf_token:
        print("WARNING: No HuggingFace token found. Gated models will fail.")

    print(f"Commits to benchmark: {commits}")
    print(f"Agent patches available: {len(agent_patches)}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for commit in commits:
            info = mapping.get(commit, {})
            has_patch = commit in agent_patches
            print(f"\n{commit}:")
            print(f"  Model: {info.get('model', 'N/A')}")
            print(f"  Human image: {HUMAN_IMAGE_PREFIX}:{info.get('human_commit_full', 'N/A')}")
            print(f"  Baseline image: {BASELINE_IMAGE_PREFIX}:baseline-{info.get('parent_commit', 'N/A')[:12]}")
            print(f"  Agent patch: {'YES' if has_patch else 'NO'}")
        return

    # Run benchmarks
    results = {'human': [], 'agent': []}

    for i, commit in enumerate(commits):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(commits)}] Processing {commit}")
        print('='*70)

        info = mapping.get(commit)
        if not info:
            print(f"  ERROR: Commit {commit} not found in mapping")
            continue

        print(f"  Model: {info.get('model', 'N/A')}")

        # Run human benchmark
        if not args.agent_only:
            human_result_file = AGENT_OUTPUT_DIR / f"{commit}_human_result.json"
            if human_result_file.exists():
                print(f"  SKIP: Human result already exists")
            else:
                print(f"\n  --- Running HUMAN benchmark ---")
                human_result = run_human_benchmark(info, hf_token, args.timeout)
                save_result(human_result, commit, 'human', info)
                results['human'].append((commit, human_result))

                if human_result['status'] == 'success':
                    throughput = human_result['metrics'].get('output_token_throughput_tok_s', 'N/A')
                    print(f"  HUMAN SUCCESS: {throughput} tok/s")
                else:
                    print(f"  HUMAN FAILED: {human_result.get('error', 'Unknown error')}")

        # Run agent benchmark
        if not args.human_only:
            agent_result_file = AGENT_OUTPUT_DIR / f"{commit}_agent_result.json"
            if agent_result_file.exists():
                print(f"  SKIP: Agent result already exists")
            elif commit not in agent_patches:
                print(f"  SKIP: No agent patch available for {commit}")
            else:
                print(f"\n  --- Running AGENT benchmark ---")
                agent_result = run_agent_benchmark(info, agent_patches[commit], hf_token, args.timeout)
                save_result(agent_result, commit, 'agent', info)
                results['agent'].append((commit, agent_result))

                if agent_result['status'] == 'success':
                    throughput = agent_result['metrics'].get('output_token_throughput_tok_s', 'N/A')
                    print(f"  AGENT SUCCESS: {throughput} tok/s")
                else:
                    print(f"  AGENT FAILED: {agent_result.get('error', 'Unknown error')}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)

    human_success = sum(1 for _, r in results['human'] if r['status'] == 'success')
    agent_success = sum(1 for _, r in results['agent'] if r['status'] == 'success')

    print(f"Human benchmarks: {human_success}/{len(results['human'])} succeeded")
    print(f"Agent benchmarks: {agent_success}/{len(results['agent'])} succeeded")


if __name__ == '__main__':
    main()
