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

# Configuration - Compute project root dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # scripts/runners/ -> OmniPerf-Bench/

HUMAN_IMAGE_PREFIX = "ayushnangia16/nvidia-vllm-docker"
BASELINE_IMAGE_PREFIX = "shikhar481/vllm_fixed_human_images"
PERF_DATA_FILE = ROOT_DIR / "archive/results/2026-01/omniperf_results_3way_claude_code/exports/full_results.jsonl"

# Agent configurations - paths to agent patch directories
AGENT_CONFIGS = {
    "claude_code": "ISO-Bench/state/runs/vllm/claude_code/default/2025-12-22_21-40-38",
    "codex_gpt5": "ISO-Bench/state/runs/vllm/codex/gpt-5",
    "codex_cli": "ISO-Bench/state/runs/vllm/codex_cli/default",  # Codex CLI GPT-5 runs
    "trae_gpt5": "ISO-Bench/state/runs/vllm/trae/gpt-5",  # Local trajectories
    "trae_sonnet45": "ISO-Bench/state/runs/vllm/trae/claude-sonnet-45",
    # TRAE specific run paths:
    "trae_gpt5_0123": "ISO-Bench/state/runs/vllm/trae/gpt-5/2026-01-23_21-19-19",
    "trae_sonnet45_0123": "ISO-Bench/state/runs/vllm/trae/us-anthropic-claude-sonnet-4-5-20250929-v1-0/2026-01-23_16-40-44",
}

# Output directories per agent type (archived results)
AGENT_OUTPUT_DIRS = {
    "claude_code": ROOT_DIR / "archive/results/2026-01/omniperf_results_3way_claude_code",
    "codex_gpt5": ROOT_DIR / "archive/results/2026-01/omniperf_results_3way_codex",
    "codex_cli": ROOT_DIR / "archive/results/2026-01/omniperf_results_3way_codex_cli",
    "trae_gpt5": ROOT_DIR / "archive/results/2026-01/omniperf_results_3way_trae_gpt5",
    "trae_sonnet45": ROOT_DIR / "archive/results/2026-01/omniperf_results_3way_trae_sonnet45",
    # TRAE specific run output dirs:
    "trae_gpt5_0123": ROOT_DIR / "archive/results/2026-01/omniperf_results_3way_trae_gpt5_0123",
    "trae_sonnet45_0123": ROOT_DIR / "archive/results/2026-01/omniperf_results_3way_trae_sonnet45_0123",
}

# Model overrides for compatibility issues (e.g., RoPE scaling)
# Old vLLM versions don't support Llama-3.1's "llama3" RoPE scaling type
MODEL_OVERRIDES = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "ibm-ai-platform/Bamba-9B-v2": "meta-llama/Meta-Llama-3-8B-Instruct",
    "ibm-ai-platform/Bamba-9B": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# Default (for backward compatibility)
RESULTS_DIR = ROOT_DIR / "archive/results/2026-01/omniperf_results_3way_claude_code"
AGENT_OUTPUT_DIR = RESULTS_DIR / "agent_benchmark_results"
BASELINE_MAPPING_FILE = ROOT_DIR / "baseline_benchmark_mapping_complete.json"
AGENT_PATCHES_DIR = ROOT_DIR / "ISO-Bench/state/runs/vllm/claude_code/default/2025-12-22_21-40-38"


def get_hf_token() -> str:
    """Get HuggingFace token."""
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    return ""


BENCHMARK_MODE_MAPPING_FILE = ROOT_DIR / "data/mappings/benchmark_mode_mapping.json"


def load_benchmark_mode_mapping() -> Dict[str, dict]:
    """Load benchmark mode mapping from HuggingFace dataset export."""
    if BENCHMARK_MODE_MAPPING_FILE.exists():
        with open(BENCHMARK_MODE_MAPPING_FILE) as f:
            return json.load(f)
    return {}


def load_perf_data() -> Dict[str, dict]:
    """Load perf_command, model, gpu_config, benchmark_mode from various sources."""
    perf_data = {}

    # First load benchmark mode mapping (from HuggingFace dataset)
    mode_mapping = load_benchmark_mode_mapping()
    for commit, data in mode_mapping.items():
        perf_data[commit] = {
            'perf_command': data.get('perf_command', ''),
            'model': data.get('model', 'unknown'),
            'gpu_config': 'H100:1',
            'commit_hash_full': data.get('commit_full', ''),
            'benchmark_mode': data.get('benchmark_mode'),  # serving, standalone, prefix_caching, or null
            'parent_commit': data.get('parent_commit', ''),
        }

    # Then overlay from full_results.jsonl (for any additional data)
    if PERF_DATA_FILE.exists():
        with open(PERF_DATA_FILE) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    commit = data.get('commit_hash', '')[:8]
                    if commit and data.get('perf_command'):
                        if commit not in perf_data:
                            perf_data[commit] = {}
                        perf_data[commit].update({
                            'perf_command': data['perf_command'],
                            'model': data.get('model', perf_data.get(commit, {}).get('model', 'unknown')),
                            'gpu_config': data.get('gpu_config', 'H100:1'),
                            'commit_hash_full': data.get('commit_hash', ''),
                        })
                        # Preserve benchmark_mode from HuggingFace if not in jsonl
                        if 'benchmark_mode' not in perf_data[commit]:
                            perf_data[commit]['benchmark_mode'] = None
                except json.JSONDecodeError:
                    continue
    return perf_data


def build_benchmark_mapping(agent_patches_dir: Path, perf_data: Dict[str, dict]) -> Dict[str, dict]:
    """Build benchmark mapping by combining agent patches with perf data.

    Returns dict mapping human_commit_short -> {
        human_commit_short, human_commit_full, parent_commit, parent_short,
        perf_command, model, gpu_config, patch_path
    }
    """
    mapping = {}

    if not agent_patches_dir.exists():
        print(f"WARNING: Agent patches directory not found: {agent_patches_dir}")
        return mapping

    # Walk through all timestamp directories to find run_summary.json files
    for root, dirs, files in os.walk(agent_patches_dir):
        if 'run_summary.json' in files and 'model_patch.diff' in files:
            summary_path = Path(root) / 'run_summary.json'
            patch_path = Path(root) / 'model_patch.diff'

            try:
                with open(summary_path) as f:
                    summary = json.load(f)

                human_commit = summary.get('commits', {}).get('human', '')
                parent_commit = summary.get('commits', {}).get('pre', '')

                if not human_commit or not parent_commit:
                    continue

                human_short = human_commit[:8]

                # Check if we have perf data for this commit
                if human_short not in perf_data:
                    continue

                # Only keep first patch found for each commit (latest run dir is usually sorted last)
                if human_short in mapping:
                    continue

                # Get full commit hash from perf_data if available (benchmark_mode_mapping.json has commit_full)
                full_hash = perf_data[human_short].get('commit_hash_full', human_commit)
                if not full_hash or len(full_hash) < 40:
                    full_hash = human_commit  # Fallback to short hash if full not available

                mapping[human_short] = {
                    'human_commit_short': human_short,
                    'human_commit_full': full_hash,  # Use full hash from perf_data
                    'parent_commit': parent_commit,
                    'parent_short': parent_commit[:12],
                    'patch_path': str(patch_path),
                    **perf_data[human_short]
                }
            except (json.JSONDecodeError, KeyError) as e:
                print(f"WARNING: Failed to parse {summary_path}: {e}")
                continue

    return mapping


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


def get_benchmark_type(perf_command: str) -> str:
    """Determine benchmark type from perf_command."""
    if not perf_command:
        return 'serving'
    if 'benchmark_latency' in perf_command or 'bench latency' in perf_command:
        return 'latency'
    if 'benchmark_throughput' in perf_command or 'bench throughput' in perf_command:
        return 'throughput'
    if 'benchmark_prefix_caching' in perf_command:
        return 'prefix_caching'
    # Default to serving (benchmark_serving.py or bench serve)
    return 'serving'


def needs_server(perf_command: str) -> bool:
    """Determine if benchmark needs a running vLLM server or runs offline.

    - benchmark_serving.py / vllm bench serve: needs server
    - benchmark_throughput.py / vllm bench throughput: runs offline
    - benchmark_latency.py / vllm bench latency: runs offline
    - benchmark_prefix_caching.py: runs offline
    """
    if not perf_command:
        return True  # Default to server-based
    perf_lower = perf_command.lower()

    # Offline benchmarks
    if 'benchmark_throughput' in perf_lower or 'bench throughput' in perf_lower:
        return False
    if 'benchmark_latency' in perf_lower or 'bench latency' in perf_lower:
        return False
    if 'benchmark_prefix_caching' in perf_lower:
        return False

    # Server-based benchmarks
    return True


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


def parse_latency_metrics(output: str) -> Dict[str, float]:
    """Parse metrics from benchmark_latency.py output."""
    metrics = {}
    patterns = {
        'latency_avg_ms': r'(?:Avg latency|avg_latency):\s*([\d.]+)\s*(?:ms|seconds)?',
        'latency_p50_ms': r'(?:P50 latency|median_latency):\s*([\d.]+)\s*(?:ms|seconds)?',
        'latency_p99_ms': r'(?:P99 latency|p99_latency):\s*([\d.]+)\s*(?:ms|seconds)?',
        'throughput_tok_s': r'(?:Throughput|throughput):\s*([\d.]+)\s*(?:tokens?/s|tok/s)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            # Convert seconds to ms if needed
            if 'latency' in key and 'seconds' in output[max(0, match.start()-20):match.end()+20].lower():
                val *= 1000
            metrics[key] = val

    return metrics


def parse_throughput_metrics(output: str) -> Dict[str, float]:
    """Parse metrics from benchmark_throughput.py output."""
    metrics = {}
    patterns = {
        'throughput_tok_s': r'(?:Throughput|throughput):\s*([\d.]+)\s*(?:tokens?/s|tok/s)',
        'elapsed_time_s': r'(?:Elapsed time|elapsed_time):\s*([\d.]+)\s*(?:s|seconds)?',
        'total_tokens': r'(?:Total tokens|total_tokens):\s*([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))

    # Handle old vLLM format: "Throughput: 10.39 requests/s, 4566.62 tokens/s"
    if 'throughput_tok_s' not in metrics:
        old_format = re.search(r'Throughput:\s*[\d.]+\s*requests/s,\s*([\d.]+)\s*tokens/s', output)
        if old_format:
            metrics['throughput_tok_s'] = float(old_format.group(1))

    return metrics


def parse_prefix_caching_metrics(output: str) -> Dict[str, float]:
    """Parse metrics from benchmark_prefix_caching.py output."""
    metrics = {}

    # Find the "start generating" section for actual performance (not warmup)
    start_gen_idx = output.rfind('------start generating------')
    if start_gen_idx != -1:
        gen_output = output[start_gen_idx:]
    else:
        gen_output = output

    # Parse speed from the LAST progress line (100% completion)
    # The format is: est. speed input: X toks/s, output: Y toks/s
    speed_matches = list(re.finditer(r'est\. speed input:\s*([\d.]+)\s*toks/s,\s*output:\s*([\d.]+)\s*toks/s', gen_output))
    if speed_matches:
        # Get the last match (100% progress)
        last_match = speed_matches[-1]
        metrics['input_throughput_tok_s'] = float(last_match.group(1))
        metrics['throughput_tok_s'] = float(last_match.group(2))  # Output throughput

    # Parse cost time (seconds for total run) - use the one after "start generating"
    cost_matches = list(re.finditer(r'cost time\s+([\d.]+)', gen_output))
    if cost_matches:
        # Get the last cost time (after "start generating")
        metrics['elapsed_time_s'] = float(cost_matches[-1].group(1))

    return metrics


def parse_metrics_by_type(output: str, benchmark_type: str) -> Dict[str, float]:
    """Parse metrics based on benchmark type."""
    if benchmark_type == 'latency':
        return parse_latency_metrics(output)
    elif benchmark_type == 'throughput':
        return parse_throughput_metrics(output)
    elif benchmark_type == 'standalone':
        # Standalone can be either latency or throughput - try both
        metrics = parse_throughput_metrics(output)
        metrics.update(parse_latency_metrics(output))
        return metrics
    elif benchmark_type == 'prefix_caching':
        # Prefix caching has its own output format
        return parse_prefix_caching_metrics(output)
    else:
        return parse_serving_metrics(output)


def run_human_benchmark_offline(commit_info: dict, hf_token: str, timeout: int = 900, benchmark_mode: str = 'standalone') -> dict:
    """Run standalone benchmark (throughput/latency) without a server."""
    start_time = time.time()

    human_commit = commit_info['human_commit_full']
    human_short = commit_info['human_commit_short']
    model = commit_info.get('model', '')
    # Apply model override for compatibility (e.g., RoPE scaling issues)
    original_model = model
    model = MODEL_OVERRIDES.get(model, model)
    if model != original_model:
        print(f"  Model override: {original_model} -> {model}")
    perf_command = commit_info.get('perf_command', '')
    # Also replace model in perf_command if overridden
    if model != original_model and original_model in perf_command:
        perf_command = perf_command.replace(original_model, model)

    docker_image = f"{HUMAN_IMAGE_PREFIX}:{human_commit}"

    # Build the benchmark command - run perf_command directly
    docker_cmd = f'''
    set -e
    MODEL="{model}"
    COMMIT="{human_commit}"

    # FIRST: Apply aimv2 compatibility fix BEFORE trying to import vllm
    # (import crashes due to this issue)
    echo "Applying aimv2 compatibility fix (pre-import)..."
    for VLLM_DIR in /usr/local/lib/python*/dist-packages/vllm /opt/venv/lib/python*/site-packages/vllm /usr/lib/python*/site-packages/vllm; do
        for ovis_file in "$VLLM_DIR/transformers_utils/configs/ovis.py" "$VLLM_DIR/transformers_utils/configs/ovis2.py"; do
            if [ -f "$ovis_file" ] && grep -q 'AutoConfig.register("aimv2"' "$ovis_file" 2>/dev/null; then
                if ! grep -q 'exist_ok=True' "$ovis_file" 2>/dev/null; then
                    sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' "$ovis_file"
                    echo "Patched aimv2 registration in $ovis_file"
                fi
            fi
        done
    done

    # Install transformers if missing (some images lack it)
    pip install transformers -q 2>/dev/null || true

    # Fix lm-format-enforcer compatibility with newer transformers (LogitsWarper removed)
    if ! python3 -c "import vllm" 2>/dev/null; then
        echo "vLLM import failed - trying lm-format-enforcer upgrade..."
        pip install --upgrade lm-format-enforcer 2>/dev/null || true
    fi

    # Now find Python with vLLM
    VLLM_PYTHON=""
    for py in /opt/venv/bin/python3 /usr/local/bin/python3 /usr/bin/python3 python3; do
        if [ -x "$(which $py 2>/dev/null || echo '')" ] || [ -x "$py" ]; then
            if $py -c "import vllm" 2>/dev/null; then
                VLLM_PYTHON="$py"
                break
            fi
        fi
    done

    if [ -z "$VLLM_PYTHON" ]; then
        echo "ERROR: Could not find Python with vLLM"
        exit 1
    fi

    echo "Using Python: $VLLM_PYTHON"
    echo "Running benchmark mode: {benchmark_mode}"
    echo "Original perf command: {perf_command}"

    # Download benchmark scripts from the HUMAN commit for compatibility
    echo "Downloading benchmark scripts from commit {human_commit}..."
    mkdir -p /opt/vllm_bench/benchmarks
    cd /opt/vllm_bench/benchmarks

    # Download benchmark scripts from the human commit (matching the vLLM version in Docker image)
    for script in benchmark_latency.py benchmark_throughput.py benchmark_serving.py backend_request_func.py benchmark_prefix_caching.py; do
        if [ ! -f "$script" ]; then
            curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/{human_commit}/benchmarks/$script" -o "$script" 2>/dev/null || \
            wget -q "https://raw.githubusercontent.com/vllm-project/vllm/{human_commit}/benchmarks/$script" -O "$script" 2>/dev/null || \
            curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/$script" -o "$script" 2>/dev/null || true
        fi
    done

    echo "Benchmark scripts downloaded from human commit"
    ls -la /opt/vllm_bench/benchmarks/

    PERF_CMD="{perf_command}"

    # Handle vllm bench throughput -> use cloned benchmark script
    if echo "$PERF_CMD" | grep -q "vllm bench throughput"; then
        ARGS=$(echo "$PERF_CMD" | sed 's/vllm bench throughput//')
        PERF_CMD="$VLLM_PYTHON /opt/vllm_bench/benchmarks/benchmark_throughput.py $ARGS"
        echo "Converted vllm bench throughput to script: $PERF_CMD"
    # Handle vllm bench latency -> use cloned benchmark script
    elif echo "$PERF_CMD" | grep -q "vllm bench latency"; then
        ARGS=$(echo "$PERF_CMD" | sed 's/vllm bench latency//')
        PERF_CMD="$VLLM_PYTHON /opt/vllm_bench/benchmarks/benchmark_latency.py $ARGS"
        echo "Converted vllm bench latency to script: $PERF_CMD"
    # Handle python benchmarks/benchmark_latency.py -> use cloned script
    elif echo "$PERF_CMD" | grep -q "benchmark_latency"; then
        ARGS=$(echo "$PERF_CMD" | sed -E 's|.*benchmark_latency\.py\s*||')
        PERF_CMD="$VLLM_PYTHON /opt/vllm_bench/benchmarks/benchmark_latency.py $ARGS"
        echo "Using cloned benchmark_latency.py: $PERF_CMD"
    # Handle python benchmarks/benchmark_throughput.py -> use cloned script
    elif echo "$PERF_CMD" | grep -q "benchmark_throughput"; then
        ARGS=$(echo "$PERF_CMD" | sed -E 's|.*benchmark_throughput\.py\s*||')
        PERF_CMD="$VLLM_PYTHON /opt/vllm_bench/benchmarks/benchmark_throughput.py $ARGS"
        echo "Using cloned benchmark_throughput.py: $PERF_CMD"
    # Handle other python commands
    elif echo "$PERF_CMD" | grep -q "^python"; then
        PERF_CMD=$(echo "$PERF_CMD" | sed "s|^python3\\? |$VLLM_PYTHON |")
        echo "Using Python command: $PERF_CMD"
    fi

    echo "Final command: $PERF_CMD"

    cd /opt/vllm_bench/benchmarks

    # Run the benchmark (use installed vLLM, not cloned repo)
    echo "=== Running HUMAN {benchmark_mode} benchmark ==="
    $PERF_CMD 2>&1 | tee /tmp/benchmark_output.txt

    echo "BENCHMARK_DONE"
    cat /tmp/benchmark_output.txt
    '''

    print(f"  Running human {benchmark_mode} benchmark with image: {docker_image}")

    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm', '--gpus', 'all',
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
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

        metrics = parse_metrics_by_type(output, benchmark_mode)

        if not metrics:
            return {
                'status': 'error',
                'error': f'No {benchmark_mode} metrics in output',
                'duration_s': duration,
                'benchmark_mode': benchmark_mode,
                'raw_output': output[-10000:]
            }

        return {
            'status': 'success',
            'metrics': metrics,
            'duration_s': duration,
            'benchmark_mode': benchmark_mode,
            'raw_output': output[-10000:]
        }

    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'error': f'Benchmark exceeded {timeout}s timeout',
            'duration_s': timeout,
            'benchmark_mode': benchmark_mode,
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'duration_s': time.time() - start_time,
            'benchmark_mode': benchmark_mode,
        }


def run_baseline_benchmark_offline(commit_info: dict, hf_token: str, timeout: int = 900, benchmark_mode: str = 'standalone') -> dict:
    """Run standalone benchmark on baseline (parent commit) Docker image."""
    start_time = time.time()

    human_commit = commit_info['human_commit_full']
    human_short = commit_info['human_commit_short']
    parent_commit = commit_info.get('parent_commit', '')
    model = commit_info.get('model', '')
    # Apply model override for compatibility (e.g., RoPE scaling issues)
    original_model = model
    model = MODEL_OVERRIDES.get(model, model)
    if model != original_model:
        print(f"  Model override: {original_model} -> {model}")
    perf_command = commit_info.get('perf_command', '')
    # Also replace model in perf_command if overridden
    if model != original_model and original_model in perf_command:
        perf_command = perf_command.replace(original_model, model)

    if not parent_commit:
        return {
            'status': 'error',
            'error': 'Missing parent_commit - cannot determine baseline image',
            'duration_s': time.time() - start_time,
        }

    # Baseline image uses parent commit hash
    baseline_image = f"{BASELINE_IMAGE_PREFIX}:baseline-{parent_commit[:12]}"

    # Build the benchmark command - use /opt/vllm_baseline
    docker_cmd = f'''
    set -e
    MODEL="{model}"
    COMMIT="{parent_commit}"

    # Apply aimv2 compatibility fix
    echo "Applying aimv2 compatibility fix..."
    for VLLM_DIR in /opt/vllm_baseline/vllm /usr/local/lib/python*/dist-packages/vllm; do
        for ovis_file in "$VLLM_DIR/transformers_utils/configs/ovis.py" "$VLLM_DIR/transformers_utils/configs/ovis2.py"; do
            if [ -f "$ovis_file" ] && grep -q 'AutoConfig.register("aimv2"' "$ovis_file" 2>/dev/null; then
                if ! grep -q 'exist_ok=True' "$ovis_file" 2>/dev/null; then
                    sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' "$ovis_file"
                    echo "Patched aimv2 registration in $ovis_file"
                fi
            fi
        done
    done

    # Install transformers if missing
    pip install transformers -q 2>/dev/null || true

    # Set PYTHONPATH to use baseline vLLM
    export PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH

    # Find Python with vLLM
    VLLM_PYTHON=""
    for py in /opt/venv/bin/python3 /usr/local/bin/python3 /usr/bin/python3 python3; do
        if [ -x "$(which $py 2>/dev/null || echo '')" ] || [ -x "$py" ]; then
            if PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $py -c "import vllm" 2>/dev/null; then
                VLLM_PYTHON="$py"
                break
            fi
        fi
    done

    if [ -z "$VLLM_PYTHON" ]; then
        echo "ERROR: Could not find Python with vLLM"
        exit 1
    fi

    echo "Using Python: $VLLM_PYTHON"
    echo "Running BASELINE benchmark mode: {benchmark_mode}"
    echo "Original perf command: {perf_command}"

    PERF_CMD="{perf_command}"

    # Use baseline vLLM's benchmark scripts from /opt/vllm_baseline/benchmarks/
    if echo "$PERF_CMD" | grep -q "vllm bench throughput"; then
        ARGS=$(echo "$PERF_CMD" | sed 's/vllm bench throughput//')
        PERF_CMD="PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_throughput.py $ARGS"
        echo "Converted to baseline throughput: $PERF_CMD"
    elif echo "$PERF_CMD" | grep -q "vllm bench latency"; then
        ARGS=$(echo "$PERF_CMD" | sed 's/vllm bench latency//')
        PERF_CMD="PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_latency.py $ARGS"
        echo "Converted to baseline latency: $PERF_CMD"
    elif echo "$PERF_CMD" | grep -q "benchmark_serving"; then
        # Serving benchmark - need to start a server first
        ARGS=$(echo "$PERF_CMD" | sed -E 's|.*benchmark_serving\\.py\\s*||')
        MODEL=$(echo "$ARGS" | grep -oP '(?<=--model\\s)\\S+')
        echo "Running SERVING benchmark - starting vLLM server first..."
        echo "Model: $MODEL"

        # Download newer benchmark scripts that support synthetic data generation
        echo "Downloading v0.6.0 benchmark scripts..."
        mkdir -p /opt/vllm_bench/benchmarks
        cd /opt/vllm_bench/benchmarks
        for script in benchmark_serving.py backend_request_func.py; do
            if [ ! -f "$script" ]; then
                curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/v0.6.0/benchmarks/$script" -o "$script" 2>/dev/null || \\
                wget -q "https://raw.githubusercontent.com/vllm-project/vllm/v0.6.0/benchmarks/$script" -O "$script" 2>/dev/null || true
            fi
        done
        echo "Benchmark scripts downloaded"

        # Create sonnet.txt dataset if missing
        if [ ! -f /opt/vllm_bench/benchmarks/sonnet.txt ]; then
            echo "Creating sonnet dataset file..."
            $VLLM_PYTHON -c "
lines = ['Shall I compare thee to a summers day? ' * 10] * 500
with open('/opt/vllm_bench/benchmarks/sonnet.txt', 'w') as f:
    f.write('\\n'.join(lines))
"
            echo "Sonnet dataset created"
        fi

        # Convert to sonnet dataset and add dataset path
        ARGS=$(echo "$ARGS" | sed 's/--dataset-name sharegpt/--dataset-name sonnet/')
        ARGS=$(echo "$ARGS" | sed 's/--dataset-name random/--dataset-name sonnet/')
        ARGS=$(echo "$ARGS" | sed 's/--dataset [^ ]*//')
        ARGS=$(echo "$ARGS" | sed 's/--dataset-path [^ ]*//')
        if ! echo "$ARGS" | grep -q -- "--dataset-name"; then
            ARGS="$ARGS --dataset-name sonnet"
        fi
        if ! echo "$ARGS" | grep -q -- "--sonnet-input-len"; then
            ARGS="$ARGS --sonnet-input-len 256 --sonnet-output-len 64"
        fi
        # Add explicit dataset path for sonnet.txt
        ARGS="$ARGS --dataset-path /opt/vllm_bench/benchmarks/sonnet.txt"
        echo "Modified ARGS for sonnet dataset: $ARGS"

        # Filter out server-specific arguments that benchmark_serving.py doesn't accept
        echo "Filtering server-only arguments from benchmark args..."
        FILTERED_ARGS=""
        SKIP_NEXT=0
        for arg in $ARGS; do
            if [ $SKIP_NEXT -eq 1 ]; then
                SKIP_NEXT=0
                continue
            fi
            case "$arg" in
                --dtype|--guided-decoding-backend|--tensor-parallel-size|--enforce-eager|--gpu-memory-utilization|--max-model-len|--max-concurrency)
                    SKIP_NEXT=1
                    echo "Filtering server-only arg: $arg"
                    ;;
                *)
                    FILTERED_ARGS="$FILTERED_ARGS $arg"
                    ;;
            esac
        done
        ARGS="$FILTERED_ARGS"
        echo "Filtered ARGS: $ARGS"

        # Fix transformers compatibility for older baseline vLLM (conditional like human benchmark)
        # Check both LogitsWarper AND transformers.utils (both can cause import failures)
        if ! $VLLM_PYTHON -c "from transformers.generation.logits_process import LogitsWarper; from transformers.utils import versions" 2>/dev/null; then
            # Check if this vLLM version uses mllama (check for mllama.py in vllm)
            VLLM_USES_MLLAMA=$(find /opt/vllm_baseline -name "*mllama*" 2>/dev/null | head -1)
            if [ -z "$VLLM_USES_MLLAMA" ]; then
                echo "Fixing transformers compatibility (LogitsWarper or transformers.utils missing, mllama not needed)..."
                # Force uninstall first to handle corrupted installations, then reinstall
                $VLLM_PYTHON -m pip uninstall transformers -y --quiet 2>&1 || true
                # Clean up any corrupted package remnants
                rm -rf /usr/local/lib/python*/dist-packages/transformers* 2>/dev/null || true
                rm -rf /usr/local/lib/python*/dist-packages/*ransformers* 2>/dev/null || true
                # Fresh install
                $VLLM_PYTHON -m pip install 'transformers==4.44.2' --quiet 2>&1 || echo "Warning: transformers install may have failed"
            else
                echo "Skipping transformers downgrade - vLLM uses mllama which needs transformers>=4.45"
            fi
        fi
        echo "Installing numpy<2..."
        $VLLM_PYTHON -m pip install 'numpy<2' --quiet 2>&1 || true
        # Fix outlines dependencies that may be missing
        echo "Installing pyairports pycountry..."
        $VLLM_PYTHON -m pip install pyairports pycountry 2>&1
        echo "Done installing dependencies"

        # Start vLLM server in background (use baseline vLLM)
        PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $VLLM_PYTHON -m vllm.entrypoints.openai.api_server \\
            --model "$MODEL" \\
            --port 8000 \\
            --disable-log-requests \\
            > /tmp/server.log 2>&1 &
        SERVER_PID=$!
        echo "Started vLLM server with PID $SERVER_PID"

        # Wait for server to be ready
        MAX_WAIT=300
        WAITED=0
        while [ $WAITED -lt $MAX_WAIT ]; do
            if curl -s http://localhost:8000/health > /dev/null 2>&1; then
                echo "Server is ready after $WAITED seconds"
                break
            fi
            sleep 5
            WAITED=$((WAITED + 5))
            echo "Waiting for server... ($WAITED/$MAX_WAIT)"
        done

        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: Server failed to start within $MAX_WAIT seconds"
            cat /tmp/server.log
            kill $SERVER_PID 2>/dev/null || true
            echo "SERVER_START_FAILED"
            exit 1
        fi

        # Run benchmark using downloaded v0.6.0 script (supports synthetic data)
        PERF_CMD="$VLLM_PYTHON /opt/vllm_bench/benchmarks/benchmark_serving.py $ARGS --base-url http://localhost:8000"
        echo "Running serving benchmark: $PERF_CMD"
        eval $PERF_CMD 2>&1 | tee /tmp/benchmark_output.txt

        # Stop the server
        echo "Stopping vLLM server..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true

        echo "BENCHMARK_DONE"
        cat /tmp/benchmark_output.txt
        exit 0
    elif echo "$PERF_CMD" | grep -q "benchmark_latency"; then
        ARGS=$(echo "$PERF_CMD" | sed -E 's|.*benchmark_latency\\.py\\s*||')

        # Filter out speculative decoding args (human's optimization, not baseline)
        echo "Filtering speculative decoding args from latency benchmark..."
        FILTERED_ARGS=""
        SKIP_NEXT=0
        for arg in $ARGS; do
            if [ $SKIP_NEXT -eq 1 ]; then
                SKIP_NEXT=0
                continue
            fi
            case "$arg" in
                --speculative-model|--num-speculative-tokens|--speculative-draft-token-sampling-method|--ngram-prompt-lookup-max|--spec-decoding-acceptance-method)
                    SKIP_NEXT=1
                    echo "Filtering speculative arg: $arg"
                    ;;
                *)
                    FILTERED_ARGS="$FILTERED_ARGS $arg"
                    ;;
            esac
        done
        ARGS="$FILTERED_ARGS"
        echo "Filtered latency ARGS: $ARGS"

        PERF_CMD="PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_latency.py $ARGS"
        echo "Using baseline benchmark_latency.py: $PERF_CMD"
    elif echo "$PERF_CMD" | grep -q "benchmark_throughput"; then
        ARGS=$(echo "$PERF_CMD" | sed -E 's|.*benchmark_throughput\\.py\\s*||')
        PERF_CMD="PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_throughput.py $ARGS"
        echo "Using baseline benchmark_throughput.py: $PERF_CMD"
    elif echo "$PERF_CMD" | grep -q "^python"; then
        PERF_CMD=$(echo "$PERF_CMD" | sed "s|^python3\\? |PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $VLLM_PYTHON |")
        echo "Using Python command: $PERF_CMD"
    fi

    echo "Final command: $PERF_CMD"

    # Run the benchmark (for non-serving benchmarks)
    echo "=== Running BASELINE {benchmark_mode} benchmark ==="
    eval $PERF_CMD 2>&1 | tee /tmp/benchmark_output.txt

    echo "BENCHMARK_DONE"
    cat /tmp/benchmark_output.txt
    '''

    print(f"  Running baseline {benchmark_mode} benchmark with image: {baseline_image}")

    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm', '--gpus', 'all',
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-e', 'VLLM_USE_V1=0',
                '-v', '/ephemeral/huggingface_cache:/root/.cache/huggingface',
                '--shm-size=16g',
                '--entrypoint', 'bash',
                baseline_image,
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        # Parse metrics from output
        metrics = {}
        benchmark_type = get_benchmark_type(perf_command)

        if benchmark_type == 'latency':
            latency_match = re.search(r'Avg latency:\s*([\d.]+)\s*seconds', output)
            if latency_match:
                metrics['latency_avg_ms'] = float(latency_match.group(1)) * 1000
        else:
            throughput_match = re.search(r'Output token throughput \(tok/s\):\s*([\d.]+)', output) or \
                              re.search(r'Throughput:\s*([\d.]+)\s*requests/s', output) or \
                              re.search(r'throughput[:\s]*([\d.]+)', output, re.IGNORECASE)
            if throughput_match:
                metrics['output_token_throughput_tok_s'] = float(throughput_match.group(1))

            ttft_match = re.search(r'Mean TTFT \(ms\):\s*([\d.]+)', output)
            tpot_match = re.search(r'Mean TPOT \(ms\):\s*([\d.]+)', output)
            itl_match = re.search(r'Mean ITL \(ms\):\s*([\d.]+)', output)
            if ttft_match:
                metrics['ttft_mean_ms'] = float(ttft_match.group(1))
            if tpot_match:
                metrics['tpot_mean_ms'] = float(tpot_match.group(1))
            if itl_match:
                metrics['itl_mean_ms'] = float(itl_match.group(1))

        if metrics:
            return {
                'status': 'success',
                'metrics': metrics,
                'duration_s': duration,
                'raw_output': output[-5000:],
                'benchmark_mode': benchmark_mode,
            }
        else:
            return {
                'status': 'error',
                'error': f'No {benchmark_type} metrics in output',
                'duration_s': duration,
                'raw_output': output[-5000:],
                'benchmark_mode': benchmark_mode,
            }

    except subprocess.TimeoutExpired:
        return {
            'status': 'error',
            'error': f'Timeout after {timeout}s',
            'duration_s': timeout,
            'benchmark_mode': benchmark_mode,
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'duration_s': time.time() - start_time,
            'benchmark_mode': benchmark_mode,
        }


def run_human_benchmark(commit_info: dict, hf_token: str, timeout: int = 900) -> dict:
    """Run benchmark using human's optimized Docker image."""
    start_time = time.time()

    human_commit = commit_info['human_commit_full']
    human_short = commit_info['human_commit_short']
    model = commit_info.get('model', '')
    # Apply model override for compatibility (e.g., RoPE scaling issues)
    original_model = model
    model = MODEL_OVERRIDES.get(model, model)
    if model != original_model:
        print(f"  Model override: {original_model} -> {model}")
        commit_info = dict(commit_info)  # Copy to avoid modifying original
        commit_info['model'] = model
    perf_command = commit_info.get('perf_command', '')
    # Also replace model in perf_command if overridden
    if model != original_model and original_model in perf_command:
        perf_command = perf_command.replace(original_model, model)
        commit_info['perf_command'] = perf_command
    benchmark_type = get_benchmark_type(perf_command)

    # Route based on perf_command type - offline vs server-based
    if not needs_server(perf_command):
        # Throughput, latency, prefix_caching benchmarks run offline
        return run_human_benchmark_offline(commit_info, hf_token, timeout, benchmark_type)

    # Fall through to serving benchmark (needs server)

    # Handle None perf_command
    if not perf_command:
        perf_command = f'python benchmarks/benchmark_serving.py --model {model}'

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

    # FIRST: Apply aimv2 compatibility fix BEFORE trying to import vllm
    # (import crashes due to aimv2 config conflict)
    echo "Applying aimv2 compatibility fix (pre-import)..."
    for VLLM_DIR in /usr/local/lib/python*/dist-packages/vllm /opt/venv/lib/python*/site-packages/vllm /usr/lib/python*/site-packages/vllm; do
        for ovis_file in "$VLLM_DIR/transformers_utils/configs/ovis.py" "$VLLM_DIR/transformers_utils/configs/ovis2.py"; do
            if [ -f "$ovis_file" ] && grep -q 'AutoConfig.register("aimv2"' "$ovis_file" 2>/dev/null; then
                if ! grep -q 'exist_ok=True' "$ovis_file" 2>/dev/null; then
                    sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' "$ovis_file"
                    echo "Patched aimv2 registration in $ovis_file"
                fi
            fi
        done
    done

    # Install transformers if missing (some images lack it)
    pip install transformers -q 2>/dev/null || true

    # Install uv for faster package management
    pip install uv -q 2>/dev/null || true

    # Check if vLLM imports work first, only apply fixes if needed
    echo "Checking vLLM compatibility..."

    # First try importing vLLM api_server (not just vllm - api_server imports lm-format-enforcer)
    if python3 -c "from vllm.entrypoints.openai import api_server" 2>/dev/null; then
        echo "vLLM api_server imports OK - no compatibility fixes needed"
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
                # Instead, upgrade lm-format-enforcer to work with newer transformers
                echo "Upgrading lm-format-enforcer for newer transformers compatibility..."
                pip install --upgrade lm-format-enforcer 2>/dev/null || true
            fi
        fi
    fi

    # Final fix: if vLLM api_server still fails to import due to lm-format-enforcer, upgrade it
    if ! python3 -c "from vllm.entrypoints.openai import api_server" 2>/dev/null; then
        echo "vLLM api_server still failing - trying lm-format-enforcer upgrade..."
        pip install --upgrade lm-format-enforcer 2>/dev/null || true
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

    # Download benchmark scripts from the HUMAN commit for compatibility
    echo "Downloading benchmark scripts from commit $COMMIT..."
    mkdir -p /opt/vllm_bench/benchmarks
    cd /opt/vllm_bench/benchmarks

    # Download benchmark scripts from the human commit (matching the vLLM version in Docker image)
    for script in benchmark_latency.py benchmark_throughput.py benchmark_serving.py backend_request_func.py benchmark_prefix_caching.py; do
        if [ ! -f "$script" ]; then
            curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/$COMMIT/benchmarks/$script" -o "$script" 2>/dev/null || \
            wget -q "https://raw.githubusercontent.com/vllm-project/vllm/$COMMIT/benchmarks/$script" -O "$script" 2>/dev/null || \
            curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/$script" -o "$script" 2>/dev/null || true
        fi
    done

    echo "Benchmark scripts downloaded from human commit"
    ls -la /opt/vllm_bench/benchmarks/

    # Create sonnet.txt dataset if missing
    if [ ! -f /opt/vllm_bench/benchmarks/sonnet.txt ]; then
        echo "Creating sonnet dataset..."
        $VLLM_PYTHON -c "
import random
lines = ['Shall I compare thee to a summers day? ' * 10] * 500
with open('/opt/vllm_bench/benchmarks/sonnet.txt', 'w') as f:
    f.write('\\n'.join(lines))
"
    fi

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

    # Use cloned vLLM's benchmark_serving.py for proper TTFT/TPOT/ITL metrics
    cd /opt/vllm_bench/benchmarks

    echo "Running benchmark_serving.py for serving metrics..."
    $VLLM_PYTHON /opt/vllm_bench/benchmarks/benchmark_serving.py \
        --model $MODEL \
        --backend vllm \
        --port 8000 \
        --dataset-name sonnet \
        --dataset-path /opt/vllm_bench/benchmarks/sonnet.txt \
        --sonnet-input-len 256 \
        --sonnet-output-len 64 \
        --num-prompts 100 \
        --request-rate inf \
        2>&1 | tee /tmp/benchmark_output.txt

    # Show TTFT/TPOT/ITL metrics
    echo "============ Serving Benchmark Result ============"
    cat /tmp/benchmark_output.txt | grep -E "(TTFT|TPOT|ITL|throughput|Throughput|Mean|Median|P99)" || true
    echo "=================================================="

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


def run_agent_benchmark_offline(commit_info: dict, agent_patch: Path, hf_token: str, timeout: int = 900, benchmark_type: str = 'throughput') -> dict:
    """Run offline benchmark (throughput/latency) by applying agent patch to baseline vLLM."""
    start_time = time.time()

    human_short = commit_info['human_commit_short']
    parent_commit = commit_info.get('parent_commit')
    model = commit_info.get('model', '') or 'meta-llama/Meta-Llama-3-8B-Instruct'
    # Apply model override for compatibility (e.g., RoPE scaling issues)
    original_model = model
    model = MODEL_OVERRIDES.get(model, model)
    if model != original_model:
        print(f"  Model override: {original_model} -> {model}")
    perf_command = commit_info.get('perf_command', '')
    # Also replace model in perf_command if overridden
    if model != original_model and original_model in perf_command:
        perf_command = perf_command.replace(original_model, model)

    # Skip if parent_commit is missing
    if not parent_commit:
        return {
            'status': 'error',
            'error': 'Missing parent_commit - cannot determine baseline image',
            'duration_s': time.time() - start_time,
            'benchmark_type': benchmark_type
        }

    # Use baseline image (has vLLM at parent commit)
    baseline_image = f"{BASELINE_IMAGE_PREFIX}:baseline-{parent_commit[:12]}"

    docker_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"
    MODEL="{model}"

    echo "=== Applying agent patch to baseline vLLM ==="

    # Baseline images have vLLM installed in /opt/vllm_baseline
    cd /opt/vllm_baseline

    # Compatibility fixes (same as server-based benchmark)
    VLLM_USES_MLLAMA=$(find /opt/vllm_baseline -name "*mllama*" 2>/dev/null | head -1)

    if [ -n "$VLLM_USES_MLLAMA" ]; then
        if ! python3 -c "from transformers.models.mllama import configuration_mllama" 2>/dev/null; then
            pip install 'transformers>=4.45.0' -q 2>/dev/null || true
        fi
    else
        if ! python3 -c "from transformers.generation.logits_process import LogitsWarper" 2>/dev/null; then
            pip install 'transformers==4.44.2' -q 2>/dev/null || true
        fi
    fi

    pip install 'numpy<2' -q 2>/dev/null || true

    # Fix broken transformers installations (check for common import issues)
    if ! python3 -c "from transformers.utils import logging" 2>/dev/null; then
        echo "Fixing broken transformers installation (utils module missing)..."
        pip install --force-reinstall 'transformers>=4.44.0,<5' -q 2>/dev/null || true
    fi

    # Verify vLLM exists
    if [ ! -d "vllm" ]; then
        echo "ERROR: vLLM not found at /opt/vllm_baseline"
        exit 1
    fi

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
    find /opt/vllm_baseline/vllm -name "*.py" -exec grep -l 'rope_scaling\\["type"\\]' {{}} \\; 2>/dev/null | while read f; do
        sed -i 's/rope_scaling\\["type"\\]/rope_scaling.get("type", rope_scaling.get("rope_type"))/g' "$f"
    done

    # Find Python with vLLM
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
        VLLM_PYTHON="python3"
    fi

    echo "=== Running AGENT {benchmark_type} benchmark (offline) ==="
    echo "Original perf command: {perf_command}"

    # Baseline images have OLD vLLM - use benchmark scripts from /opt/vllm_baseline
    # NOT the new CLI (which doesn't exist in old versions)
    PERF_CMD="{perf_command}"

    # Convert vllm bench commands to use the OLD benchmark scripts
    if echo "$PERF_CMD" | grep -q "vllm bench throughput"; then
        ARGS=$(echo "$PERF_CMD" | sed 's/vllm bench throughput//')
        PERF_CMD="$VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_throughput.py $ARGS"
        echo "Converted vllm bench throughput to script: $PERF_CMD"
    elif echo "$PERF_CMD" | grep -q "vllm bench latency"; then
        ARGS=$(echo "$PERF_CMD" | sed 's/vllm bench latency//')
        PERF_CMD="$VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_latency.py $ARGS"
        echo "Converted vllm bench latency to script: $PERF_CMD"
    # Handle python benchmarks/benchmark_latency.py -> use script from baseline
    elif echo "$PERF_CMD" | grep -q "benchmark_latency"; then
        ARGS=$(echo "$PERF_CMD" | sed -E 's|.*benchmark_latency\.py\s*||')
        PERF_CMD="$VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_latency.py $ARGS"
        echo "Using baseline benchmark_latency.py: $PERF_CMD"
    # Handle python benchmarks/benchmark_throughput.py -> use script from baseline
    elif echo "$PERF_CMD" | grep -q "benchmark_throughput"; then
        ARGS=$(echo "$PERF_CMD" | sed -E 's|.*benchmark_throughput\.py\s*||')
        PERF_CMD="$VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_throughput.py $ARGS"
        echo "Using baseline benchmark_throughput.py: $PERF_CMD"
    # Handle other python commands
    elif echo "$PERF_CMD" | grep -q "^python"; then
        PERF_CMD=$(echo "$PERF_CMD" | sed "s|^python3\\? |$VLLM_PYTHON |")
        echo "Using Python command: $PERF_CMD"
    fi

    # Translate --dataset-name sharegpt to --dataset /data/sharegpt_dataset.json for old vLLM versions
    if echo "$PERF_CMD" | grep -q "benchmark_throughput"; then
        if echo "$PERF_CMD" | grep -q "\\-\\-dataset-name sharegpt"; then
            echo "Translating --dataset-name sharegpt to --dataset /data/sharegpt_dataset.json"
            PERF_CMD=$(echo "$PERF_CMD" | sed 's/--dataset-name sharegpt/--dataset \/data\/sharegpt_dataset.json/')
        fi
    fi

    # Add default input/output lengths for throughput benchmarks if missing
    # BUT skip if --dataset is used (old vLLM asserts input_len is None when using dataset file)
    if echo "$PERF_CMD" | grep -q "benchmark_throughput"; then
        if ! echo "$PERF_CMD" | grep -q "\\-\\-input-len"; then
            if ! echo "$PERF_CMD" | grep -q "\\-\\-dataset "; then
                PERF_CMD="$PERF_CMD --input-len 512 --output-len 128"
                echo "Added default input/output lengths for throughput benchmark"
            else
                echo "Skipping default input/output lengths - using dataset file"
            fi
        fi
    fi

    echo "Final command: $PERF_CMD"

    # Run from baseline vLLM directory
    cd /opt/vllm_baseline

    # Run the benchmark with proper PYTHONPATH
    PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $PERF_CMD 2>&1 | tee /tmp/benchmark_output.txt

    echo "BENCHMARK_DONE"
    cat /tmp/benchmark_output.txt
    '''

    print(f"  Running agent {benchmark_type} benchmark (offline) with baseline image: {baseline_image}")
    print(f"  Applying patch: {agent_patch}")

    # Mount ShareGPT dataset if perf_command uses it
    sharegpt_mount = []
    sharegpt_path = ROOT_DIR / 'data/sharegpt_dataset.json'
    if sharegpt_path.exists() and 'sharegpt' in perf_command.lower():
        sharegpt_mount = ['-v', f'{sharegpt_path}:/data/sharegpt_dataset.json:ro']

    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm', '--gpus', 'all',
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-e', 'VLLM_USE_V1=0',
                '-v', '/ephemeral/huggingface_cache:/root/.cache/huggingface',
                '-v', f'{agent_patch}:/agent_patch.diff:ro',
            ] + sharegpt_mount + [
                '--shm-size=16g',
                '--entrypoint', 'bash',
                baseline_image,
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        metrics = parse_metrics_by_type(output, benchmark_type)

        if not metrics:
            return {
                'status': 'error',
                'error': f'No {benchmark_type} metrics in agent output',
                'duration_s': duration,
                'benchmark_type': benchmark_type,
                'raw_output': output[-10000:]
            }

        return {
            'status': 'success',
            'metrics': metrics,
            'duration_s': duration,
            'benchmark_type': benchmark_type,
            'raw_output': output[-5000:]
        }

    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'error': f'Agent benchmark timed out after {timeout}s',
            'duration_s': timeout,
            'benchmark_type': benchmark_type
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'duration_s': time.time() - start_time,
            'benchmark_type': benchmark_type
        }


def check_docker_image_exists(image: str) -> bool:
    """Check if a Docker image exists locally or can be pulled."""
    try:
        result = subprocess.run(
            ['docker', 'manifest', 'inspect', image],
            capture_output=True, timeout=30
        )
        return result.returncode == 0
    except:
        return False


def run_agent_benchmark_from_wheel(commit_info: dict, agent_patch: Path, hf_token: str, timeout: int = 1200, benchmark_type: str = 'serving') -> dict:
    """Run benchmark using vLLM wheel from S3 (fallback when Docker image unavailable)."""
    start_time = time.time()

    human_commit = commit_info.get('human_commit_full', '')
    human_short = commit_info['human_commit_short']
    parent_commit = commit_info.get('parent_commit', '')
    model = commit_info.get('model', '') or 'meta-llama/Meta-Llama-3-8B-Instruct'
    # Apply model override for compatibility (e.g., RoPE scaling issues)
    original_model = model
    model = MODEL_OVERRIDES.get(model, model)
    if model != original_model:
        print(f"  Model override: {original_model} -> {model}")
    perf_command = commit_info.get('perf_command', '')
    # Also replace model in perf_command if overridden
    if model != original_model and original_model in perf_command:
        perf_command = perf_command.replace(original_model, model)

    # vLLM wheel URL
    wheel_url = f"https://vllm-wheels.s3.us-west-2.amazonaws.com/{parent_commit}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

    # Use a recent stable vLLM image as base (has all dependencies)
    base_image = "vllm/vllm-openai:v0.6.0"

    docker_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"
    HUMAN_COMMIT="{human_commit}"
    MODEL="{model}"

    echo "=== Installing vLLM from wheel (fallback mode) ==="
    echo "Wheel URL: {wheel_url}"

    # Uninstall existing vLLM and install from wheel
    pip uninstall vllm -y 2>/dev/null || true
    pip install {wheel_url} 2>&1

    # Verify installation and get vLLM package directory (suppress logging)
    export VLLM_LOGGING_LEVEL=ERROR
    VLLM_DIR=$(python3 -c "import logging; logging.disable(logging.CRITICAL); import vllm; import os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)
    echo "vLLM installed at: $VLLM_DIR"
    python3 -c "import logging; logging.disable(logging.CRITICAL); import vllm; print('vLLM version:', vllm.__version__)" 2>/dev/null || echo "vLLM version check skipped"

    # Download benchmark scripts from parent commit (without cloning full repo)
    mkdir -p /tmp/benchmarks
    cd /tmp/benchmarks
    for script in benchmark_latency.py benchmark_throughput.py benchmark_serving.py backend_request_func.py benchmark_utils.py benchmark_dataset.py sonnet.py; do
        curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/$PARENT_COMMIT/benchmarks/$script" -o "$script" 2>/dev/null || \\
        wget -q "https://raw.githubusercontent.com/vllm-project/vllm/$PARENT_COMMIT/benchmarks/$script" -O "$script" 2>/dev/null || \\
        curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/$script" -o "$script" 2>/dev/null || true
    done
    # Create stubs for modules that don't exist in older vLLM
    if [ ! -s /tmp/benchmarks/benchmark_utils.py ]; then
        echo "# Stub for older vLLM" > /tmp/benchmarks/benchmark_utils.py
        echo "def convert_to_pytorch_benchmark_format(*args, **kwargs): pass" >> /tmp/benchmarks/benchmark_utils.py
    fi
    if [ ! -s /tmp/benchmarks/benchmark_dataset.py ]; then
        echo "# Stub for older vLLM" > /tmp/benchmarks/benchmark_dataset.py
        echo "class AIMODataset: pass" >> /tmp/benchmarks/benchmark_dataset.py
        echo "class ASRDataset: pass" >> /tmp/benchmarks/benchmark_dataset.py
        echo "class BurstGPTDataset: pass" >> /tmp/benchmarks/benchmark_dataset.py
        echo "class ConversationDataset: pass" >> /tmp/benchmarks/benchmark_dataset.py
        echo "class HumanEvalDataset: pass" >> /tmp/benchmarks/benchmark_dataset.py
        echo "class InstructCoderDataset: pass" >> /tmp/benchmarks/benchmark_dataset.py
        echo "class LongContextDataset: pass" >> /tmp/benchmarks/benchmark_dataset.py
        echo "class ShareGPTDataset: pass" >> /tmp/benchmarks/benchmark_dataset.py
        echo "class SonnetDataset: pass" >> /tmp/benchmarks/benchmark_dataset.py
        echo "class SyntheticDataset: pass" >> /tmp/benchmarks/benchmark_dataset.py
        echo "class VisionArenaDataset: pass" >> /tmp/benchmarks/benchmark_dataset.py
    fi
    echo "Benchmark scripts downloaded"

    echo "=== Applying agent patch to installed vLLM ==="
    # Create a temp directory structure matching the patch paths
    mkdir -p /tmp/patch_work
    cd /tmp/patch_work

    # Copy installed vLLM to work directory
    cp -r $VLLM_DIR vllm

    # Apply patch - patches expect vllm/ prefix
    if patch -p0 --dry-run < /agent_patch.diff 2>&1; then
        patch -p0 < /agent_patch.diff 2>&1
        echo "AGENT_PATCH_APPLIED"
    else
        echo "Trying patch -p1..."
        if patch -p1 --dry-run < /agent_patch.diff 2>&1; then
            patch -p1 < /agent_patch.diff 2>&1
            echo "AGENT_PATCH_APPLIED_P1"
        else
            echo "Patch dry-run failed, trying with --force..."
            patch -p1 --force < /agent_patch.diff 2>&1 || patch -p0 --force < /agent_patch.diff 2>&1 || true
            echo "AGENT_PATCH_APPLIED_FALLBACK"
        fi
    fi

    # Copy patched files back to installed vLLM
    cp -r /tmp/patch_work/vllm/* $VLLM_DIR/ 2>/dev/null || true

    # Apply rope_scaling fix for Llama-3.1 models
    find "$VLLM_DIR" -name "*.py" -exec grep -l 'rope_scaling\\["type"\\]' {{}} \\; 2>/dev/null | while read f; do
        sed -i 's/rope_scaling\\["type"\\]/rope_scaling.get("type", rope_scaling.get("rope_type"))/g' "$f"
    done

    # Install benchmark deps
    pip install aiohttp pandas datasets -q 2>/dev/null || true

    # Run the benchmark based on type
    PERF_CMD="{perf_command}"
    cd /tmp/benchmarks

    if echo "$PERF_CMD" | grep -q "benchmark_serving"; then
        echo "=== Running SERVING benchmark ==="
        # Start vLLM server
        python3 -m vllm.entrypoints.openai.api_server \\
            --model $MODEL --port 8000 --max-model-len 4096 --disable-log-requests 2>&1 &
        SERVER_PID=$!

        # Wait for server
        for i in $(seq 1 300); do
            if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models', timeout=2)" 2>/dev/null; then
                echo "SERVER_READY_AFTER=${{i}}s"
                break
            fi
            if ! kill -0 $SERVER_PID 2>/dev/null; then
                echo "SERVER_CRASHED"
                exit 1
            fi
            sleep 1
        done

        # Run benchmark
        python3 /tmp/benchmarks/benchmark_serving.py \\
            --model $MODEL \\
            --backend vllm \\
            --port 8000 \\
            --dataset-name random \\
            --random-input-len 256 \\
            --random-output-len 64 \\
            --num-prompts 100 \\
            --request-rate inf \\
            2>&1 | tee /tmp/benchmark_output.txt

        kill $SERVER_PID 2>/dev/null || true
    elif echo "$PERF_CMD" | grep -q "benchmark_latency"; then
        echo "=== Running LATENCY benchmark ==="
        ARGS=$(echo "$PERF_CMD" | sed -E 's|.*benchmark_latency\\.py\\s*||')
        python3 /tmp/benchmarks/benchmark_latency.py $ARGS 2>&1 | tee /tmp/benchmark_output.txt
    elif echo "$PERF_CMD" | grep -q "benchmark_throughput"; then
        echo "=== Running THROUGHPUT benchmark ==="
        ARGS=$(echo "$PERF_CMD" | sed -E 's|.*benchmark_throughput\\.py\\s*||')
        # Handle dataset-name sharegpt
        if echo "$ARGS" | grep -q "\\-\\-dataset-name sharegpt"; then
            ARGS=$(echo "$ARGS" | sed 's/--dataset-name sharegpt/--dataset \/data\/sharegpt_dataset.json/')
        fi
        python3 /tmp/benchmarks/benchmark_throughput.py $ARGS 2>&1 | tee /tmp/benchmark_output.txt
    fi

    echo "BENCHMARK_DONE"
    cat /tmp/benchmark_output.txt 2>/dev/null || true
    '''

    print(f"  Running agent benchmark from WHEEL: {wheel_url[:60]}...")
    print(f"  Applying patch: {agent_patch}")

    # Mount ShareGPT dataset if needed
    sharegpt_mount = []
    sharegpt_path = ROOT_DIR / 'data/sharegpt_dataset.json'
    if sharegpt_path.exists() and 'sharegpt' in perf_command.lower():
        sharegpt_mount = ['-v', f'{sharegpt_path}:/data/sharegpt_dataset.json:ro']

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
            ] + sharegpt_mount + [
                '--shm-size=16g',
                '--entrypoint', 'bash',
                base_image,
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        if 'SERVER_CRASHED' in output:
            return {
                'status': 'error',
                'error': 'Server crashed after applying patch (wheel mode)',
                'duration_s': duration,
                'raw_output': output[-10000:]
            }

        metrics = parse_metrics_by_type(output, benchmark_type)

        if not metrics:
            return {
                'status': 'error',
                'error': f'No {benchmark_type} metrics in agent output (wheel mode)',
                'duration_s': duration,
                'raw_output': output[-10000:]
            }

        return {
            'status': 'success',
            'metrics': metrics,
            'duration_s': duration,
            'benchmark_type': benchmark_type,
            'raw_output': output[-5000:]
        }

    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'error': f'Agent benchmark (wheel mode) timed out after {timeout}s',
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
    parent_commit = commit_info.get('parent_commit')
    model = commit_info.get('model', '') or 'meta-llama/Meta-Llama-3-8B-Instruct'
    # Apply model override for compatibility (e.g., RoPE scaling issues)
    original_model = model
    model = MODEL_OVERRIDES.get(model, model)
    if model != original_model:
        print(f"  Model override: {original_model} -> {model}")
        commit_info = dict(commit_info)  # Copy to avoid modifying original
        commit_info['model'] = model
    perf_command = commit_info.get('perf_command', '')
    # Also replace model in perf_command if overridden
    if model != original_model and original_model in perf_command:
        perf_command = perf_command.replace(original_model, model)
        commit_info['perf_command'] = perf_command
    benchmark_type = get_benchmark_type(perf_command)

    # Skip if parent_commit is missing
    if not parent_commit:
        return {
            'status': 'error',
            'error': 'Missing parent_commit - cannot determine baseline image',
            'duration_s': time.time() - start_time
        }

    # Check if baseline Docker image exists
    baseline_image = f"{BASELINE_IMAGE_PREFIX}:baseline-{parent_commit[:12]}"
    if not check_docker_image_exists(baseline_image):
        print(f"  Baseline image {baseline_image} not found, trying wheel-based fallback...")
        return run_agent_benchmark_from_wheel(commit_info, agent_patch, hf_token, timeout, benchmark_type)

    # Route based on perf_command type - offline vs server-based
    if not needs_server(perf_command):
        # Throughput, latency, prefix_caching benchmarks run offline
        return run_agent_benchmark_offline(commit_info, agent_patch, hf_token, timeout, benchmark_type)

    # Fall through to serving benchmark (needs server)

    # Handle None perf_command
    if not perf_command:
        perf_command = f'python benchmarks/benchmark_serving.py --model {model}'

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

    # Fix broken transformers installations (check for common import issues)
    if ! python3 -c "from transformers.utils import logging" 2>/dev/null; then
        echo "Fixing broken transformers installation (utils module missing)..."
        pip install --force-reinstall 'transformers>=4.44.0,<5' -q 2>/dev/null || true
    fi

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

    # No need to clone vLLM - we use /opt/vllm_baseline/benchmarks/ which has the full scripts

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

    # Use baseline vLLM's benchmark_serving.py for proper TTFT/TPOT/ITL metrics
    # (Latest vLLM has deprecated stubs, baseline has the full implementation)
    cd /opt/vllm_baseline

    echo "Running benchmark_serving.py for serving metrics..."
    PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_serving.py \
        --model $MODEL \
        --backend vllm \
        --port 8000 \
        --dataset-name random \
        --random-input-len 256 \
        --random-output-len 64 \
        --num-prompts 100 \
        --request-rate inf \
        2>&1 | tee /tmp/benchmark_output.txt

    # Show TTFT/TPOT/ITL metrics
    echo "============ Serving Benchmark Result ============"
    cat /tmp/benchmark_output.txt | grep -E "(TTFT|TPOT|ITL|throughput|Throughput|Mean|Median|P99)" || true
    echo "=================================================="

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
    global AGENT_OUTPUT_DIR, AGENT_PATCHES_DIR

    import argparse
    parser = argparse.ArgumentParser(description='Run 3-way benchmarks for vLLM commits')
    parser.add_argument('--agent-type', type=str, choices=list(AGENT_CONFIGS.keys()),
                        default='claude_code',
                        help='Agent type to benchmark (default: claude_code)')
    parser.add_argument('--commits', type=str, nargs='+',
                        help='Specific commits to run (short hash)')
    parser.add_argument('--human-only', action='store_true',
                        help='Only run human benchmarks')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Only run baseline benchmarks')
    parser.add_argument('--agent-only', action='store_true',
                        help='Only run agent benchmarks')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be run')
    parser.add_argument('--timeout', type=int, default=900,
                        help='Timeout per benchmark in seconds')
    parser.add_argument('--use-legacy-mapping', action='store_true',
                        help='Use legacy baseline_benchmark_mapping_complete.json')
    args = parser.parse_args()

    # Set agent-specific paths
    agent_type = args.agent_type
    AGENT_PATCHES_DIR = ROOT_DIR / AGENT_CONFIGS[agent_type]
    AGENT_OUTPUT_DIR = AGENT_OUTPUT_DIRS[agent_type] / "results"

    print(f"=== Agent Type: {agent_type} ===")
    print(f"Agent patches dir: {AGENT_PATCHES_DIR}")
    print(f"Output dir: {AGENT_OUTPUT_DIR}")

    # Load configuration
    print("Loading configuration...")

    if args.use_legacy_mapping:
        # Legacy mode for backward compatibility
        mapping = load_mapping()
        agent_patches = load_agent_patches()
    else:
        # New mode: build mapping from perf_data + run_summary.json
        perf_data = load_perf_data()
        print(f"Loaded perf data for {len(perf_data)} commits")

        mapping = build_benchmark_mapping(AGENT_PATCHES_DIR, perf_data)
        print(f"Built mapping for {len(mapping)} commits")

        # Agent patches are now embedded in the mapping (patch_path field)
        agent_patches = {k: Path(v['patch_path']) for k, v in mapping.items()}

    hf_token = get_hf_token()

    # Determine commits to run
    if args.commits:
        commits = args.commits
    else:
        # Run all commits in mapping
        commits = list(mapping.keys())

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
            parent = info.get('parent_commit') or 'N/A'
            print(f"  Baseline image: {BASELINE_IMAGE_PREFIX}:baseline-{parent[:12]}")
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
        if not args.agent_only and not args.baseline_only:
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

        # Run baseline benchmark
        if args.baseline_only or (not args.human_only and not args.agent_only):
            baseline_result_file = AGENT_OUTPUT_DIR / f"{commit}_baseline_result.json"
            if baseline_result_file.exists():
                print(f"  SKIP: Baseline result already exists")
            else:
                print(f"\n  --- Running BASELINE benchmark ---")
                benchmark_mode = info.get('benchmark_mode', 'standalone')
                if benchmark_mode == 'serving':
                    # For now, use offline for baseline (serving mode can be added later)
                    baseline_result = run_baseline_benchmark_offline(info, hf_token, args.timeout, 'standalone')
                else:
                    baseline_result = run_baseline_benchmark_offline(info, hf_token, args.timeout, benchmark_mode or 'standalone')

                # Save baseline result
                baseline_data = {
                    'human_commit': commit,
                    'human_commit_full': info.get('human_commit_full', ''),
                    'parent_commit': info.get('parent_commit', ''),
                    'model': info.get('model', ''),
                    'status': baseline_result['status'],
                    'error': baseline_result.get('error'),
                    'duration_s': baseline_result.get('duration_s', 0),
                    'metrics': baseline_result.get('metrics', {}),
                    'raw_output': baseline_result.get('raw_output', ''),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                AGENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                with open(baseline_result_file, 'w') as f:
                    json.dump(baseline_data, f, indent=2)
                print(f"  Saved to {baseline_result_file}")

                results.setdefault('baseline', []).append((commit, baseline_result))

                if baseline_result['status'] == 'success':
                    metrics = baseline_result['metrics']
                    throughput = metrics.get('output_token_throughput_tok_s')
                    latency = metrics.get('latency_avg_ms')
                    if throughput:
                        print(f"  BASELINE SUCCESS: {throughput} tok/s")
                    elif latency:
                        print(f"  BASELINE SUCCESS: {latency} ms latency")
                    else:
                        print(f"  BASELINE SUCCESS: {metrics}")
                else:
                    print(f"  BASELINE FAILED: {baseline_result.get('error', 'Unknown error')}")

        # Run agent benchmark
        if not args.human_only and not args.baseline_only:
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

    human_success = sum(1 for _, r in results.get('human', []) if r['status'] == 'success')
    baseline_success = sum(1 for _, r in results.get('baseline', []) if r['status'] == 'success')
    agent_success = sum(1 for _, r in results.get('agent', []) if r['status'] == 'success')

    print(f"Human benchmarks: {human_success}/{len(results.get('human', []))} succeeded")
    print(f"Baseline benchmarks: {baseline_success}/{len(results.get('baseline', []))} succeeded")
    print(f"Agent benchmarks: {agent_success}/{len(results.get('agent', []))} succeeded")


if __name__ == '__main__':
    main()
