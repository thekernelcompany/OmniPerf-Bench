#!/usr/bin/env python3
"""
Custom context-length benchmark runner for models with non-standard max_model_len.
Specifically designed to handle models like huggyllama/llama-7b (2048 context).
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Configuration - Compute project root dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # scripts/runners/ -> OmniPerf-Bench/

HUMAN_IMAGE_PREFIX = "ayushnangia16/nvidia-vllm-docker"
BASELINE_IMAGE_PREFIX = "shikhar481/vllm_fixed_human_images"
RESULTS_DIR = ROOT_DIR / "archive/results/2026-01/omniperf_results_3way_claude_code"
AGENT_OUTPUT_DIR = RESULTS_DIR / "agent_benchmark_results"
BASELINE_MAPPING_FILE = ROOT_DIR / "baseline_benchmark_mapping_complete.json"
AGENT_PATCHES_DIR = ROOT_DIR / "perf-agents-bench/state/runs/vllm/claude_code/default/2025-12-22_21-40-38"


def get_hf_token() -> str:
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    return ""


def load_mapping():
    with open(BASELINE_MAPPING_FILE) as f:
        mapping = json.load(f)
    return {m['human_commit_short']: m for m in mapping}


def load_agent_patches():
    patches = {}
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
            except:
                pass
    return patches


def parse_serving_metrics(output: str):
    metrics = {}
    patterns = {
        'request_throughput_req_s': r'Request throughput \(req/s\):\s+([\d.]+)',
        'output_token_throughput_tok_s': r'Output token throughput \(tok/s\):\s+([\d.]+)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def run_human_benchmark(commit_info: dict, hf_token: str, max_model_len: int, timeout: int = 900):
    """Run human benchmark with custom max_model_len."""
    start_time = time.time()

    human_commit = commit_info['human_commit_full']
    model = commit_info.get('model', '')

    docker_image = f"{HUMAN_IMAGE_PREFIX}:{human_commit}"

    # Adjusted input length for smaller context models
    input_len = min(100, max_model_len // 4)  # Use 1/4 of context for input

    docker_cmd = f'''
    set -e
    MODEL="{model}"
    MAX_MODEL_LEN={max_model_len}
    INPUT_LEN={input_len}

    export PYTHONPATH=/workspace:$PYTHONPATH

    # Compatibility fixes
    pip install 'numpy<2' -q 2>/dev/null || true

    if ! python3 -c "from transformers.generation.logits_process import LogitsWarper" 2>/dev/null; then
        pip install 'transformers==4.44.2' -q 2>/dev/null || true
    fi

    # Find vLLM Python
    VLLM_PYTHON=""
    for py in /opt/venv/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
        if [ -x "$py" ] && $py -c "import vllm" 2>/dev/null; then
            VLLM_PYTHON="$py"
            break
        fi
    done

    if [ -z "$VLLM_PYTHON" ]; then
        echo "ERROR: Could not find Python with vLLM"
        exit 1
    fi

    # Fix rope_scaling
    VLLM_DIR=$($VLLM_PYTHON -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || echo "")
    if [ -n "$VLLM_DIR" ]; then
        find "$VLLM_DIR" -name "*.py" -exec grep -l 'rope_scaling\["type"\]' {{}} \; 2>/dev/null | while read f; do
            sed -i 's/rope_scaling\["type"\]/rope_scaling.get("type", rope_scaling.get("rope_type"))/g' "$f"
        done
    fi

    $VLLM_PYTHON -c "import vllm; print('vLLM ' + vllm.__version__ + ' OK')"

    # Fix outlines
    if ! $VLLM_PYTHON -c "from outlines.fsm.guide import Guide" 2>/dev/null; then
        $VLLM_PYTHON -m pip uninstall outlines -y 2>/dev/null || true
        $VLLM_PYTHON -m pip install 'outlines==0.0.34' --no-deps 2>&1 || true
    fi

    $VLLM_PYTHON -m pip install aiohttp -q 2>/dev/null || true

    echo "=== Starting vLLM server with max_model_len=$MAX_MODEL_LEN ==="
    cd /tmp
    $VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
        --model $MODEL --port 8000 --max-model-len $MAX_MODEL_LEN --disable-log-requests 2>&1 &
    SERVER_PID=$!

    for i in $(seq 1 300); do
        if $VLLM_PYTHON -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models', timeout=2)" 2>/dev/null; then
            echo "SERVER_READY"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "SERVER_CRASHED"
            exit 1
        fi
        sleep 1
    done

    echo "=== Running benchmark ==="
    $VLLM_PYTHON << BENCHMARK
import asyncio
import aiohttp
import time
import random

async def send_request(session, url, payload):
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            result = await resp.json()
            output_tokens = len(result.get('choices', [dict()])[0].get('text', '').split())
            return output_tokens, None
    except Exception as e:
        return 0, str(e)

async def benchmark():
    url = "http://localhost:8000/v1/completions"
    model = "$MODEL"
    num_prompts = 100
    max_tokens = 64
    input_len = $INPUT_LEN

    prompts = []
    for _ in range(num_prompts):
        prompt = ' '.join(random.choices(['the', 'a', 'is', 'of', 'and', 'to', 'in', 'for', 'on', 'with'], k=input_len))
        prompts.append(prompt)

    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        tasks = [send_request(session, url, dict(model=model, prompt=p, max_tokens=max_tokens, temperature=0.0)) for p in prompts]
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

    total_time = end_time - start_time
    successful = sum(1 for r in results if r[1] is None)
    total_output_tokens = sum(r[0] for r in results if r[1] is None)

    print("============ Serving Benchmark Result ============")
    print("Successful requests:                     " + str(successful))
    print("Benchmark duration (s):                  " + str(round(total_time, 2)))
    print("Request throughput (req/s):              " + str(round(successful/total_time, 2)))
    print("Output token throughput (tok/s):         " + str(round(total_output_tokens/total_time, 2)))
    print("==================================================")

asyncio.run(benchmark())
BENCHMARK

    kill $SERVER_PID 2>/dev/null || true
    echo "BENCHMARK_DONE"
    '''

    print(f"  Running human benchmark with max_model_len={max_model_len}")

    try:
        result = subprocess.run(
            ['docker', 'run', '--rm', '--gpus', 'all', '--network=host',
             '-e', f'HF_TOKEN={hf_token}', '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
             '-e', 'VLLM_USE_V1=0',
             '-v', '/ephemeral/huggingface_cache:/root/.cache/huggingface',
             '--shm-size=16g', '--entrypoint', 'bash', docker_image, '-c', docker_cmd],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        if 'SERVER_CRASHED' in output:
            return {'status': 'error', 'error': 'Server crashed', 'duration_s': duration, 'raw_output': output[-10000:]}

        metrics = parse_serving_metrics(output)
        if not metrics:
            return {'status': 'error', 'error': 'No metrics', 'duration_s': duration, 'raw_output': output[-10000:]}

        return {'status': 'success', 'metrics': metrics, 'duration_s': duration, 'raw_output': output[-5000:]}

    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'error': f'Timeout after {timeout}s', 'duration_s': timeout}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'duration_s': time.time() - start_time}


def run_agent_benchmark(commit_info: dict, agent_patch: Path, hf_token: str, max_model_len: int, timeout: int = 900):
    """Run agent benchmark with custom max_model_len."""
    start_time = time.time()

    parent_commit = commit_info['parent_commit']
    model = commit_info.get('model', '')

    baseline_image = f"{BASELINE_IMAGE_PREFIX}:baseline-{parent_commit[:12]}"
    input_len = min(100, max_model_len // 4)

    docker_cmd = f'''
    set -e
    MODEL="{model}"
    MAX_MODEL_LEN={max_model_len}
    INPUT_LEN={input_len}

    cd /opt/vllm_baseline

    pip install 'numpy<2' -q 2>/dev/null || true

    if ! python3 -c "from transformers.generation.logits_process import LogitsWarper" 2>/dev/null; then
        pip install 'transformers==4.44.2' -q 2>/dev/null || true
    fi

    echo "Applying patch..."
    patch -p1 --force < /agent_patch.diff 2>&1 || true
    echo "AGENT_PATCH_APPLIED"

    # Fix rope_scaling
    find /opt/vllm_baseline/vllm -name "*.py" -exec grep -l 'rope_scaling\["type"\]' {{}} \; 2>/dev/null | while read f; do
        sed -i 's/rope_scaling\["type"\]/rope_scaling.get("type", rope_scaling.get("rope_type"))/g' "$f"
    done

    export PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH
    VLLM_PYTHON="python3"

    $VLLM_PYTHON -c "import vllm; print('vLLM ' + vllm.__version__ + ' OK')" || echo "vLLM import warning"

    # Fix outlines
    if ! $VLLM_PYTHON -c "from outlines.fsm.guide import Guide" 2>/dev/null; then
        $VLLM_PYTHON -m pip uninstall outlines -y 2>/dev/null || true
        $VLLM_PYTHON -m pip install 'outlines==0.0.34' --no-deps 2>&1 || true
    fi

    $VLLM_PYTHON -m pip install aiohttp -q 2>/dev/null || true

    echo "=== Starting vLLM server with max_model_len=$MAX_MODEL_LEN ==="
    cd /tmp
    PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
        --model $MODEL --port 8000 --max-model-len $MAX_MODEL_LEN --disable-log-requests 2>&1 &
    SERVER_PID=$!

    for i in $(seq 1 300); do
        if $VLLM_PYTHON -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models', timeout=2)" 2>/dev/null; then
            echo "SERVER_READY"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "SERVER_CRASHED"
            exit 1
        fi
        sleep 1
    done

    echo "=== Running benchmark ==="
    $VLLM_PYTHON << BENCHMARK
import asyncio
import aiohttp
import time
import random

async def send_request(session, url, payload):
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            result = await resp.json()
            output_tokens = len(result.get('choices', [dict()])[0].get('text', '').split())
            return output_tokens, None
    except Exception as e:
        return 0, str(e)

async def benchmark():
    url = "http://localhost:8000/v1/completions"
    model = "$MODEL"
    num_prompts = 100
    max_tokens = 64
    input_len = $INPUT_LEN

    prompts = []
    for _ in range(num_prompts):
        prompt = ' '.join(random.choices(['the', 'a', 'is', 'of', 'and', 'to', 'in', 'for', 'on', 'with'], k=input_len))
        prompts.append(prompt)

    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        tasks = [send_request(session, url, dict(model=model, prompt=p, max_tokens=max_tokens, temperature=0.0)) for p in prompts]
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

    total_time = end_time - start_time
    successful = sum(1 for r in results if r[1] is None)
    total_output_tokens = sum(r[0] for r in results if r[1] is None)

    print("============ Serving Benchmark Result ============")
    print("Successful requests:                     " + str(successful))
    print("Benchmark duration (s):                  " + str(round(total_time, 2)))
    print("Request throughput (req/s):              " + str(round(successful/total_time, 2)))
    print("Output token throughput (tok/s):         " + str(round(total_output_tokens/total_time, 2)))
    print("==================================================")

asyncio.run(benchmark())
BENCHMARK

    kill $SERVER_PID 2>/dev/null || true
    echo "BENCHMARK_DONE"
    '''

    print(f"  Running agent benchmark with max_model_len={max_model_len}")

    try:
        result = subprocess.run(
            ['docker', 'run', '--rm', '--gpus', 'all', '--network=host',
             '-e', f'HF_TOKEN={hf_token}', '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
             '-e', 'VLLM_USE_V1=0',
             '-v', '/ephemeral/huggingface_cache:/root/.cache/huggingface',
             '-v', f'{agent_patch}:/agent_patch.diff:ro',
             '--shm-size=16g', '--entrypoint', 'bash', baseline_image, '-c', docker_cmd],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        if 'SERVER_CRASHED' in output:
            return {'status': 'error', 'error': 'Server crashed', 'duration_s': duration, 'raw_output': output[-10000:]}

        metrics = parse_serving_metrics(output)
        if not metrics:
            return {'status': 'error', 'error': 'No metrics', 'duration_s': duration, 'raw_output': output[-10000:]}

        return {'status': 'success', 'metrics': metrics, 'duration_s': duration, 'raw_output': output[-5000:]}

    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'error': f'Timeout after {timeout}s', 'duration_s': timeout}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'duration_s': time.time() - start_time}


def save_result(result: dict, commit_short: str, result_type: str, commit_info: dict):
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
    parser = argparse.ArgumentParser(description='Run benchmark with custom max_model_len')
    parser.add_argument('--commit', type=str, required=True, help='Commit short hash')
    parser.add_argument('--max-model-len', type=int, required=True, help='Max model length')
    parser.add_argument('--human-only', action='store_true')
    parser.add_argument('--agent-only', action='store_true')
    parser.add_argument('--force', action='store_true', help='Delete existing results')
    args = parser.parse_args()

    mapping = load_mapping()
    agent_patches = load_agent_patches()
    hf_token = get_hf_token()

    commit = args.commit
    info = mapping.get(commit)
    if not info:
        print(f"ERROR: Commit {commit} not found in mapping")
        sys.exit(1)

    print(f"Processing {commit}: {info.get('model', 'N/A')}")
    print(f"Using max_model_len={args.max_model_len}")

    if args.force:
        for suffix in ['human', 'agent']:
            result_file = AGENT_OUTPUT_DIR / f"{commit}_{suffix}_result.json"
            if result_file.exists():
                result_file.unlink()
                print(f"  Deleted existing {suffix} result")

    # Human benchmark
    if not args.agent_only:
        human_file = AGENT_OUTPUT_DIR / f"{commit}_human_result.json"
        if human_file.exists():
            print("  SKIP: Human result exists")
        else:
            print("\n--- Running HUMAN benchmark ---")
            result = run_human_benchmark(info, hf_token, args.max_model_len)
            save_result(result, commit, 'human', info)
            if result['status'] == 'success':
                print(f"  HUMAN SUCCESS: {result['metrics'].get('output_token_throughput_tok_s', 'N/A')} tok/s")
            else:
                print(f"  HUMAN FAILED: {result.get('error')}")

    # Agent benchmark
    if not args.human_only:
        agent_file = AGENT_OUTPUT_DIR / f"{commit}_agent_result.json"
        if agent_file.exists():
            print("  SKIP: Agent result exists")
        elif commit not in agent_patches:
            print(f"  SKIP: No agent patch for {commit}")
        else:
            print("\n--- Running AGENT benchmark ---")
            result = run_agent_benchmark(info, agent_patches[commit], hf_token, args.max_model_len)
            save_result(result, commit, 'agent', info)
            if result['status'] == 'success':
                print(f"  AGENT SUCCESS: {result['metrics'].get('output_token_throughput_tok_s', 'N/A')} tok/s")
            else:
                print(f"  AGENT FAILED: {result.get('error')}")


if __name__ == '__main__':
    main()
