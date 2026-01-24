#!/usr/bin/env python3
"""
Run SGLang benchmarks using actual perf_command from HuggingFace dataset.
Maps commits to available Docker images and runs with proper models.
"""

import subprocess
import json
import time
import os
import re
import urllib.request
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from huggingface_hub import hf_hub_download

# Configuration
NUM_GPUS = 8
RESULTS_DIR = Path("/home/ubuntu/OmniPerf-Bench/omniperf_results_3way_sglang/dataset_benchmarks")
HF_TOKEN = open(Path.home() / ".cache/huggingface/token").read().strip() if (Path.home() / ".cache/huggingface/token").exists() else ""
IMAGE_STATUS_FILE = Path("/home/ubuntu/OmniPerf-Bench/src/benchmark/fixes/sglang_docker_image_status.json")

def load_image_status():
    """Load commit-specific image status mapping."""
    if IMAGE_STATUS_FILE.exists():
        with open(IMAGE_STATUS_FILE) as f:
            return json.load(f)
    return {"working_images": {}, "broken_images": {}, "special_perf_commands": {}}

def get_available_docker_images():
    """Get all available Docker images from both repos and local builds."""
    images = {}

    # PRIORITY 1: Local v05-hf- images (improved Dockerfile with all deps)
    try:
        proc = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True, timeout=30
        )
        for line in proc.stdout.strip().split('\n'):
            if 'v05-hf-' in line:
                # Extract commit from tag like "shikhar481/sglang-images:v05-hf-021f76e4f498"
                tag = line.split(':')[-1]
                commit = tag.replace('v05-hf-', '')[:8]
                if len(commit) == 8:
                    if commit not in images:
                        images[commit] = []
                    # Insert at beginning to prioritize local images
                    images[commit].insert(0, line.strip())
    except Exception as e:
        print(f"Error fetching local v05-hf images: {e}")

    # PRIORITY 2: shikhar481/sglang-images from Docker Hub
    try:
        url = "https://hub.docker.com/v2/repositories/shikhar481/sglang-images/tags?page_size=200"
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
            for t in data.get('results', []):
                tag = t['name']
                # Extract commit hash from tag (e.g., "09deb20d-src" -> "09deb20d")
                commit = tag.split('-')[0] if '-' in tag else tag
                if len(commit) == 8:
                    if commit not in images:
                        images[commit] = []
                    images[commit].append(f"shikhar481/sglang-images:{tag}")
    except Exception as e:
        print(f"Error fetching shikhar481 images: {e}")

    # PRIORITY 3: ayushnangia16/nvidia-sglang-docker
    try:
        url = "https://hub.docker.com/v2/repositories/ayushnangia16/nvidia-sglang-docker/tags?page_size=200"
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
            for t in data.get('results', []):
                tag = t['name']
                commit = tag[:8]
                if commit not in images:
                    images[commit] = []
                images[commit].append(f"ayushnangia16/nvidia-sglang-docker:{tag}")
    except Exception as e:
        print(f"Error fetching ayushnangia16 images: {e}")

    return images


def extract_tp_size(perf_command: str) -> int:
    """Extract TP size from command."""
    patterns = [r'--tp[=\s]+(\d+)', r'--tp-size[=\s]+(\d+)', r'-tp[=\s]+(\d+)']
    for p in patterns:
        m = re.search(p, perf_command)
        if m:
            return int(m.group(1))
    return 1


def extract_model_from_command(perf_command: str) -> str:
    """Extract model path from perf_command."""
    # Try --model pattern
    patterns = [
        r'--model[=\s]+([^\s]+)',
        r'--model-path[=\s]+([^\s]+)',
        r'-m[=\s]+([^\s]+)',
    ]
    for p in patterns:
        m = re.search(p, perf_command)
        if m:
            return m.group(1)
    return None


def run_benchmark(commit: str, image: str, perf_command: str, models: list, gpu_ids: list, image_status: dict) -> dict:
    """Run benchmark for a commit with its actual perf_command."""
    short_commit = commit[:8]
    result = {
        "commit": commit,
        "image": image,
        "perf_command": perf_command,
        "models": models,
        "gpu_ids": gpu_ids,
        "status": "error",
        "error": None,
        "metrics": {},
        "duration_s": 0,
    }

    # Check if this commit is known to be broken
    if short_commit in image_status.get("broken_images", {}):
        broken_info = image_status["broken_images"][short_commit]
        result["error"] = f"Known broken: {broken_info.get('reason', broken_info.get('error', 'unknown'))}"
        result["skipped"] = True
        return result

    start_time = time.time()
    container_name = f"sglang-{short_commit}"
    tp_size = extract_tp_size(perf_command)

    # Determine model from command or models list
    model = extract_model_from_command(perf_command)
    if not model:
        # Check special_perf_commands for default model
        if short_commit in image_status.get("special_perf_commands", {}):
            model = image_status["special_perf_commands"][short_commit].get("default_model")
    if not model:
        model = models[0] if models else None
    if not model:
        result["error"] = "No model found in perf_command"
        return result

    # Check if we need to fix python -> python3 for this commit
    fix_python_cmd = False
    if short_commit in image_status.get("working_images", {}):
        fix_python_cmd = image_status["working_images"][short_commit].get("fix_python_cmd", False)

    # Build GPU device string
    gpu_str = ','.join(str(g) for g in gpu_ids[:tp_size])

    # Apply commit-specific fixes to perf_command
    benchmark_cmd = perf_command
    if fix_python_cmd:
        # Replace 'python ' with 'python3 ' for commits that need it
        benchmark_cmd = re.sub(r'\bpython\s', 'python3 ', benchmark_cmd)
        benchmark_cmd = re.sub(r'\bpython$', 'python3', benchmark_cmd)

    # Prepare the benchmark command
    # Handle different benchmark types
    if 'bench_serving' in perf_command:
        # Server-based benchmark - need to start server first
        docker_script = f'''
set -e

# Set PYTHONPATH
for p in /sglang/python /sgl-workspace/sglang/python /opt/sglang/python; do
    [ -d "$p" ] && export PYTHONPATH="$p:$PYTHONPATH"
done

# Check SGLang
python3 -c "import sglang; print(f'SGLANG_VERSION={{sglang.__version__}}')" 2>&1 || echo "IMPORT_FAILED"

# Start server in background
echo "STARTING_SERVER"
python3 -m sglang.launch_server \\
    --model-path {model} \\
    --port 30000 \\
    --host 0.0.0.0 \\
    --tp-size {tp_size} \\
    --trust-remote-code \\
    2>&1 &
SERVER_PID=$!

# Wait for server
for i in $(seq 1 180); do
    if curl -s http://localhost:30000/health 2>/dev/null | grep -qiE "ok|healthy|true"; then
        echo "SERVER_READY"
        break
    fi
    if curl -s http://localhost:30000/v1/models 2>/dev/null | grep -qiE "model|data"; then
        echo "SERVER_READY"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "SERVER_CRASHED"
        exit 1
    fi
    sleep 1
done

# Run benchmark
echo "RUNNING_BENCHMARK"
{benchmark_cmd} --host 127.0.0.1 --port 30000 2>&1

echo "BENCHMARK_COMPLETE"
kill $SERVER_PID 2>/dev/null || true
'''
    elif 'bench_one_batch' in perf_command or 'bench_latency' in perf_command:
        # Direct benchmark - no server needed
        docker_script = f'''
set -e

for p in /sglang/python /sgl-workspace/sglang/python /opt/sglang/python; do
    [ -d "$p" ] && export PYTHONPATH="$p:$PYTHONPATH"
done

python3 -c "import sglang; print(f'SGLANG_VERSION={{sglang.__version__}}')" 2>&1 || echo "IMPORT_FAILED"

echo "RUNNING_BENCHMARK"
{benchmark_cmd} 2>&1

echo "BENCHMARK_COMPLETE"
'''
    else:
        # Unknown benchmark type - try running as-is
        docker_script = f'''
set -e

for p in /sglang/python /sgl-workspace/sglang/python /opt/sglang/python; do
    [ -d "$p" ] && export PYTHONPATH="$p:$PYTHONPATH"
done

python3 -c "import sglang; print(f'SGLANG_VERSION={{sglang.__version__}}')" 2>&1 || echo "IMPORT_FAILED"

echo "RUNNING_BENCHMARK"
{benchmark_cmd} 2>&1

echo "BENCHMARK_COMPLETE"
'''

    try:
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=10)

        proc = subprocess.run(
            [
                'docker', 'run', '--rm',
                '--name', container_name,
                '--gpus', f'device={gpu_str}',
                '-e', f'HF_TOKEN={HF_TOKEN}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={HF_TOKEN}',
                '-v', '/home/ubuntu/.cache/huggingface:/root/.cache/huggingface',
                '--shm-size=32g',
                '--entrypoint', 'bash',
                image,
                '-c', docker_script
            ],
            capture_output=True, text=True, timeout=1800  # 30 min timeout
        )

        output = proc.stdout + proc.stderr
        result["raw_output"] = output[-10000:]

        # Parse version
        match = re.search(r'SGLANG_VERSION=(\S+)', output)
        if match:
            result["sglang_version"] = match.group(1)

        # Check status
        if "BENCHMARK_COMPLETE" in output:
            result["status"] = "success"

            # Parse metrics
            patterns = {
                'request_throughput': r'Request throughput[^:]*:\s*([\d.]+)',
                'output_throughput': r'Output token throughput[^:]*:\s*([\d.]+)',
                'input_throughput': r'Input token throughput[^:]*:\s*([\d.]+)',
                'ttft_mean': r'Mean TTFT[^:]*:\s*([\d.]+)',
                'tpot_mean': r'Mean TPOT[^:]*:\s*([\d.]+)',
                'latency': r'latency[^:]*:\s*([\d.]+)',
            }
            for key, pattern in patterns.items():
                m = re.search(pattern, output, re.IGNORECASE)
                if m:
                    result["metrics"][key] = float(m.group(1))
        elif "SERVER_CRASHED" in output:
            result["error"] = "Server crashed"
        elif "IMPORT_FAILED" in output:
            result["error"] = "SGLang import failed"
        else:
            result["error"] = f"Benchmark did not complete. Exit code: {proc.returncode}"

    except subprocess.TimeoutExpired:
        result["error"] = "Timeout (30 min)"
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=10)
    except Exception as e:
        result["error"] = str(e)

    result["duration_s"] = time.time() - start_time
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load commit-specific image status mapping
    image_status = load_image_status()
    broken_commits = set(image_status.get("broken_images", {}).keys())
    print(f"Loaded image status: {len(broken_commits)} known broken commits")

    # Load dataset
    print("Loading dataset...")
    file_path = hf_hub_download(
        repo_id="Ayushnangia/omniperf_v1",
        filename="sglang/train-00000-of-00001.parquet",
        repo_type="dataset"
    )
    df = pd.read_parquet(file_path)
    df_with_cmd = df[df['perf_command'].notna() & (df['perf_command'] != '')]
    print(f"Commits with perf_command: {len(df_with_cmd)}")

    # Get available Docker images
    print("\nFetching available Docker images...")
    available_images = get_available_docker_images()
    print(f"Found images for {len(available_images)} commits")

    # Match commits to images
    commits_to_run = []
    for _, row in df_with_cmd.iterrows():
        commit = row['commit_hash'][:8]
        if commit in available_images:
            # Prefer shikhar481 images with priority: -src > -v3 > -v2 > bare > ayushnangia16
            images = available_images[commit]
            shikhar_images = [i for i in images if 'shikhar481' in i]
            best_image = None
            for suffix in ['-src', '-v3', '-v2', '']:
                match = next((i for i in shikhar_images if i.endswith(suffix) or (suffix == '' and '-' not in i.split(':')[-1])), None)
                if match:
                    best_image = match
                    break
            if not best_image:
                # Fall back to ayushnangia16 if no shikhar481 image
                best_image = next((i for i in images if 'ayushnangia16' in i), images[0])

            tp_size = extract_tp_size(row['perf_command'])

            commits_to_run.append({
                'commit': commit,
                'full_commit': row['commit_hash'],
                'image': best_image,
                'perf_command': row['perf_command'],
                'models': row['models'] if isinstance(row['models'], list) else [],
                'tp_size': tp_size,
                'subject': row.get('commit_subject', '')[:60],
            })

    print(f"\nCommits with available images: {len(commits_to_run)}")

    # Sort by TP size (run single-GPU first)
    commits_to_run.sort(key=lambda x: x['tp_size'])

    print("\n=== Commits to run ===")
    for c in commits_to_run[:10]:
        print(f"  {c['commit']} | TP={c['tp_size']} | {c['models'][0] if c['models'] else 'N/A'}")
    if len(commits_to_run) > 10:
        print(f"  ... and {len(commits_to_run) - 10} more")

    # Run benchmarks
    print(f"\n{'='*60}")
    print("STARTING BENCHMARKS")
    print(f"{'='*60}")

    all_results = []
    success_count = 0
    start_time = time.time()

    # Group by TP size for efficient GPU allocation
    gpu_pool = list(range(NUM_GPUS))

    for commit_info in commits_to_run:
        tp_size = commit_info['tp_size']

        # Skip if not enough GPUs
        if tp_size > NUM_GPUS:
            print(f"[SKIP] {commit_info['commit']} - needs TP={tp_size}, only have {NUM_GPUS} GPUs")
            continue

        # Allocate GPUs
        gpu_ids = list(range(tp_size))

        print(f"\n[{len(all_results)+1}] {commit_info['commit']} | TP={tp_size} | {commit_info['subject'][:40]}...")

        result = run_benchmark(
            commit=commit_info['full_commit'],
            image=commit_info['image'],
            perf_command=commit_info['perf_command'],
            models=commit_info['models'],
            gpu_ids=gpu_ids,
            image_status=image_status
        )

        all_results.append(result)

        if result['status'] == 'success':
            success_count += 1
            print(f"    ✓ Success | {result.get('metrics', {})}")
        elif result.get('skipped'):
            print(f"    ⊘ Skipped: {result.get('error', 'Unknown')[:50]}")
        else:
            print(f"    ✗ Failed: {result.get('error', 'Unknown')[:50]}")

        # Save intermediate results
        if len(all_results) % 5 == 0:
            with open(RESULTS_DIR / "results_partial.json", 'w') as f:
                json.dump(all_results, f, indent=2)

    # Save final results
    total_time = time.time() - start_time
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_commits": len(commits_to_run),
        "ran": len(all_results),
        "success": success_count,
        "failed": len(all_results) - success_count,
        "duration_min": total_time / 60,
        "results": all_results
    }

    with open(RESULTS_DIR / "benchmark_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(all_results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(all_results) - success_count}")
    print(f"Duration: {total_time/60:.1f} minutes")
    print(f"Results: {RESULTS_DIR / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
