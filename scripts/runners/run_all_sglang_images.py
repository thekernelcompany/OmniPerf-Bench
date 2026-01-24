#!/usr/bin/env python3
"""
Run benchmarks on ALL SGLang Docker images from both repos.
Uses 8 GPUs in parallel for maximum throughput.
"""

import subprocess
import json
import time
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configuration
NUM_GPUS = 8
RESULTS_DIR = Path("/home/ubuntu/OmniPerf-Bench/omniperf_results_3way_sglang/all_images_benchmark")
HF_TOKEN = open(Path.home() / ".cache/huggingface/token").read().strip() if (Path.home() / ".cache/huggingface/token").exists() else ""

# Model to test - use a smaller one that's more likely to work with older transformers
TEST_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def run_benchmark_on_image(image: str, gpu_id: int) -> dict:
    """Run a quick benchmark on a single Docker image."""
    result = {
        "image": image,
        "gpu_id": gpu_id,
        "status": "error",
        "sglang_version": None,
        "server_started": False,
        "benchmark_completed": False,
        "metrics": {},
        "error": None,
        "duration_s": 0,
    }

    start_time = time.time()
    # Use image tag for unique container name to avoid conflicts
    image_tag = image.split(':')[-1][:12].replace('-', '')
    container_name = f"sglang-{image_tag}-gpu{gpu_id}"

    # Docker command to run benchmark
    docker_cmd = f'''
    set -e

    # Set PYTHONPATH for various image layouts
    for p in /sglang/python /sgl-workspace/sglang/python /opt/sglang/python; do
        [ -d "$p" ] && export PYTHONPATH="$p:$PYTHONPATH"
    done

    # Check SGLang version
    python3 -c "import sglang; print(f'SGLANG_VERSION={{sglang.__version__}}')" 2>&1 || echo "SGLANG_IMPORT_FAILED"

    # Fix missing dependencies
    echo "FIXING_DEPENDENCIES"

    # Install libnuma (for sgl_kernel)
    apt-get update -qq && apt-get install -y -qq libnuma-dev 2>&1 | tail -3 || true

    # Install missing Python packages
    pip install pyairports sentencepiece orjson outlines 2>&1 | tail -5 || true

    # Check current transformers version
    CURRENT_TF=$(pip show transformers 2>/dev/null | grep Version | cut -d' ' -f2)
    echo "CURRENT_TRANSFORMERS=$CURRENT_TF"

    # Only fix transformers if there's an import issue
    # Test if sglang imports successfully
    if ! python3 -c "import sglang" 2>/dev/null; then
        echo "SGLANG_IMPORT_ISSUE_FIXING"
        # Try downgrading transformers for torchao compatibility
        pip install "transformers==4.44.2" "tokenizers>=0.19" 2>&1 | tail -3 || true
        echo "FIXED_TRANSFORMERS=$(pip show transformers 2>/dev/null | grep Version | cut -d' ' -f2)"
    fi

    pip install huggingface_hub safetensors 2>&1 | tail -2 || true

    echo "DEPENDENCIES_FIXED"

    # Try to start server
    echo "STARTING_SERVER"
    python3 -m sglang.launch_server \\
        --model-path {TEST_MODEL} \\
        --port 30000 \\
        --host 0.0.0.0 \\
        --tp-size 1 \\
        --mem-fraction-static 0.8 \\
        2>&1 &
    SERVER_PID=$!

    # Wait for server (max 180s - longer for dependency installs)
    SERVER_READY=0
    for i in $(seq 1 180); do
        if curl -s http://localhost:30000/health 2>/dev/null | grep -qiE "ok|healthy|true"; then
            echo "SERVER_READY_AFTER=${{i}}s"
            SERVER_READY=1
            break
        fi
        if curl -s http://localhost:30000/v1/models 2>/dev/null | grep -qiE "model|data"; then
            echo "SERVER_READY_AFTER=${{i}}s"
            SERVER_READY=1
            break
        fi
        # Also check /get_model_info (older versions)
        if curl -s http://localhost:30000/get_model_info 2>/dev/null | grep -qiE "model"; then
            echo "SERVER_READY_AFTER=${{i}}s"
            SERVER_READY=1
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "SERVER_CRASHED"
            exit 1
        fi
        sleep 1
    done

    if [ $SERVER_READY -eq 0 ]; then
        echo "SERVER_STARTUP_TIMEOUT"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi

    # Quick benchmark (just 10 requests)
    echo "RUNNING_BENCHMARK"
    python3 -m sglang.bench_serving \\
        --backend sglang \\
        --host 127.0.0.1 \\
        --port 30000 \\
        --num-prompts 10 \\
        --random-input-len 128 \\
        --random-output-len 32 \\
        2>&1 || echo "BENCHMARK_FAILED"

    echo "BENCHMARK_DONE"
    kill $SERVER_PID 2>/dev/null || true
    '''

    try:
        # Kill any existing container with same name
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=10)

        # Run benchmark
        proc = subprocess.run(
            [
                'docker', 'run', '--rm',
                '--name', container_name,
                '--gpus', f'device={gpu_id}',
                '-e', f'HF_TOKEN={HF_TOKEN}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={HF_TOKEN}',
                '-v', '/home/ubuntu/.cache/huggingface:/root/.cache/huggingface',
                '--shm-size=16g',
                '--entrypoint', 'bash',
                image,
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=600  # 10 min timeout per image (includes dep install)
        )

        output = proc.stdout + proc.stderr
        result["raw_output"] = output[-5000:]  # Last 5000 chars

        # Parse results
        if "SGLANG_VERSION=" in output:
            match = re.search(r'SGLANG_VERSION=(\S+)', output)
            if match:
                result["sglang_version"] = match.group(1)

        if "SERVER_READY_AFTER=" in output:
            result["server_started"] = True

        if "BENCHMARK_DONE" in output and "BENCHMARK_FAILED" not in output:
            result["benchmark_completed"] = True
            result["status"] = "success"

            # Parse metrics
            patterns = {
                'request_throughput': r'Request throughput \(req/s\):\s+([\d.]+)',
                'output_throughput': r'Output token throughput \(tok/s\):\s+([\d.]+)',
            }
            for key, pattern in patterns.items():
                match = re.search(pattern, output)
                if match:
                    result["metrics"][key] = float(match.group(1))
        elif "SERVER_CRASHED" in output:
            result["error"] = "Server crashed during startup"
        elif "SERVER_STARTUP_TIMEOUT" in output:
            result["error"] = "Server startup timeout (180s)"
        elif "SGLANG_IMPORT_FAILED" in output:
            result["error"] = "SGLang import failed"
        else:
            # Try to detect specific errors
            if "cannot import name 'AutoProcessor'" in output:
                result["error"] = "AutoProcessor import (transformers)"
            elif "No module named 'pyairports'" in output:
                result["error"] = "Missing pyairports"
            elif "libnuma.so" in output:
                result["error"] = "Missing libnuma"
            elif "No module named 'sentencepiece'" in output:
                result["error"] = "Missing sentencepiece"
            elif "masking_utils" in output:
                result["error"] = "transformers.masking_utils"
            elif "sgl_kernel" in output and "undefined symbol" in output:
                result["error"] = "sgl_kernel ABI mismatch"
            elif "CUDA out of memory" in output:
                result["error"] = "CUDA OOM"
            elif "ModuleNotFoundError" in output:
                match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", output)
                if match:
                    result["error"] = f"Missing module: {match.group(1)}"
                else:
                    result["error"] = "ModuleNotFoundError"
            elif "ImportError" in output:
                match = re.search(r"ImportError: ([^\n]+)", output)
                if match:
                    result["error"] = f"ImportError: {match.group(1)[:50]}"
                else:
                    result["error"] = "ImportError"
            elif "Error" in output or "error" in output:
                # Try to find any error line
                match = re.search(r"(?:Error|error)[^\n]{0,80}", output)
                if match:
                    result["error"] = match.group(0)[:60]
                else:
                    result["error"] = "Unknown error"
            else:
                result["error"] = "Unknown error"

    except subprocess.TimeoutExpired:
        result["error"] = "Timeout (10 min)"
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=10)
    except Exception as e:
        result["error"] = str(e)

    result["duration_s"] = time.time() - start_time
    return result


def main():
    # Load all images
    with open('/tmp/all_sglang_images.txt') as f:
        all_images = [line.strip() for line in f if line.strip()]

    print(f"Total images to benchmark: {len(all_images)}")
    print(f"Using {NUM_GPUS} GPUs in parallel")
    print(f"Test model: {TEST_MODEL}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Results tracking
    all_results = []
    success_count = 0

    start_time = time.time()

    # Run in parallel batches
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = {}

        for i, image in enumerate(all_images):
            gpu_id = i % NUM_GPUS
            future = executor.submit(run_benchmark_on_image, image, gpu_id)
            futures[future] = image

        for future in as_completed(futures):
            image = futures[future]
            try:
                result = future.result()
                all_results.append(result)

                status_emoji = "✓" if result["status"] == "success" else "✗"
                if result["status"] == "success":
                    success_count += 1

                print(f"[{len(all_results)}/{len(all_images)}] {status_emoji} {image.split(':')[1][:12]} - "
                      f"v{result.get('sglang_version', '?')} - "
                      f"{result.get('error', 'OK')[:30]}")

                # Save intermediate results
                if len(all_results) % 10 == 0:
                    with open(RESULTS_DIR / "results_partial.json", 'w') as f:
                        json.dump(all_results, f, indent=2)

            except Exception as e:
                print(f"Error processing {image}: {e}")

    # Save final results
    total_time = time.time() - start_time

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_images": len(all_images),
        "success_count": success_count,
        "failure_count": len(all_images) - success_count,
        "success_rate": f"{success_count/len(all_images)*100:.1f}%",
        "total_duration_s": total_time,
        "test_model": TEST_MODEL,
        "results": all_results
    }

    with open(RESULTS_DIR / "all_images_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(all_images)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(all_images) - success_count}")
    print(f"Success rate: {success_count/len(all_images)*100:.1f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR / 'all_images_results.json'}")

    # List successful images
    successful = [r["image"] for r in all_results if r["status"] == "success"]
    if successful:
        print(f"\nWorking images ({len(successful)}):")
        for img in successful[:20]:
            print(f"  - {img}")
        if len(successful) > 20:
            print(f"  ... and {len(successful) - 20} more")


if __name__ == "__main__":
    main()
