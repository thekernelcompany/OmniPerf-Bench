#!/usr/bin/env python3
"""
Benchmark fixed SGLang Docker images in parallel.
Uses GPUs 4-7 for benchmarking while rebuild uses GPUs 0-3.
"""

import subprocess
import json
import time
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

RESULTS_DIR = Path("/home/ubuntu/OmniPerf-Bench/omniperf_results_3way_sglang/rebuilt_images")
HF_TOKEN = os.environ.get('HF_TOKEN', '')

def get_fixed_images():
    """Get list of fixed images from Docker."""
    proc = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True, timeout=30
    )
    images = [line.strip() for line in proc.stdout.split('\n') if line.strip() and ':fixed-' in line]
    return images


def run_benchmark(image: str, gpu_id: int) -> dict:
    """Run benchmark on a Docker image."""
    result = {
        'image': image,
        'gpu_id': gpu_id,
        'status': 'error',
        'sglang_version': None,
        'metrics': {},
        'error': None,
        'duration_s': 0,
    }

    start_time = time.time()
    container_name = f"bench-{image.split(':')[-1][:12]}-gpu{gpu_id}"

    docker_cmd = '''
    # Get SGLang version
    python3 -c "import sglang; print(f'SGLANG_VERSION={sglang.__version__}')" 2>&1 || echo "SGLANG_IMPORT_FAILED"

    # Start server (use --disable-cuda-graph to avoid OOM issues with some versions)
    echo "STARTING_SERVER"
    timeout 180 python3 -m sglang.launch_server \
        --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --port 30000 \
        --host 0.0.0.0 \
        --tp-size 1 \
        --mem-fraction-static 0.7 \
        --disable-cuda-graph \
        2>&1 &
    SERVER_PID=$!

    # Wait for server
    for i in $(seq 1 120); do
        if curl -s http://localhost:30000/health 2>/dev/null | grep -qiE "ok|healthy|true"; then
            echo "SERVER_READY_AFTER=${i}s"
            break
        fi
        if curl -s http://localhost:30000/get_model_info 2>/dev/null | grep -qiE "model"; then
            echo "SERVER_READY_AFTER=${i}s"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "SERVER_CRASHED"
            exit 1
        fi
        sleep 1
    done

    # Run benchmark (use --dataset-name random for random input/output lengths)
    echo "RUNNING_BENCHMARK"
    python3 -m sglang.bench_serving \
        --backend sglang \
        --host 127.0.0.1 \
        --port 30000 \
        --dataset-name random \
        --num-prompts 50 \
        --random-input-len 128 \
        --random-output-len 64 \
        2>&1 || echo "BENCHMARK_FAILED"

    echo "BENCHMARK_DONE"
    kill $SERVER_PID 2>/dev/null || true
    '''

    try:
        # Kill any existing container
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=10)

        proc = subprocess.run(
            [
                'docker', 'run', '--rm',
                '--name', container_name,
                '--gpus', f'device={gpu_id}',
                '-e', f'HF_TOKEN={HF_TOKEN}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={HF_TOKEN}',
                '-v', '/home/ubuntu/.cache/huggingface:/root/.cache/huggingface',
                '--shm-size=16g',
                image,
                'bash', '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=300
        )

        output = proc.stdout + proc.stderr
        result['raw_output'] = output[-3000:]

        # Parse version
        if "SGLANG_VERSION=" in output:
            match = re.search(r'SGLANG_VERSION=(\S+)', output)
            if match:
                result['sglang_version'] = match.group(1)

        # Parse results
        if "BENCHMARK_DONE" in output and "BENCHMARK_FAILED" not in output:
            result['status'] = 'success'
            patterns = {
                'request_throughput': r'Request throughput \(req/s\):\s+([\d.]+)',
                'output_throughput': r'Output token throughput \(tok/s\):\s+([\d.]+)',
            }
            for key, pattern in patterns.items():
                match = re.search(pattern, output)
                if match:
                    result['metrics'][key] = float(match.group(1))
        elif "SERVER_CRASHED" in output:
            result['error'] = "Server crashed"
        else:
            result['error'] = "Benchmark failed"

    except subprocess.TimeoutExpired:
        result['error'] = "Timeout"
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=10)
    except Exception as e:
        result['error'] = str(e)

    result['duration_s'] = time.time() - start_time
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="4,5,6,7", help="GPUs to use (comma-separated)")
    parser.add_argument("--continuous", action="store_true", help="Run continuously, checking for new images")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(',')]
    print(f"Using GPUs: {gpu_ids}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    benchmarked = set()

    # Load already benchmarked images
    results_file = RESULTS_DIR / "benchmark_results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            benchmarked = set(r['image'] for r in data.get('results', []))

    all_results = []

    while True:
        # Get fixed images
        images = get_fixed_images()
        new_images = [img for img in images if img not in benchmarked]

        if not new_images:
            if args.continuous:
                print(f"No new images to benchmark. Waiting...")
                time.sleep(30)
                continue
            else:
                print("No new images to benchmark.")
                break

        print(f"\n{'='*60}")
        print(f"Benchmarking {len(new_images)} new images")
        print(f"{'='*60}")

        # Run benchmarks in parallel
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = {}
            for i, image in enumerate(new_images):
                gpu_id = gpu_ids[i % len(gpu_ids)]
                future = executor.submit(run_benchmark, image, gpu_id)
                futures[future] = image

            for future in as_completed(futures):
                image = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    benchmarked.add(image)

                    status = "✓" if result['status'] == 'success' else "✗"
                    tok_s = result.get('metrics', {}).get('output_throughput', 0)
                    print(f"  {status} {image.split(':')[-1][:20]} | v{result.get('sglang_version', '?')} | {tok_s:.0f} tok/s")

                except Exception as e:
                    print(f"  Error: {image}: {e}")

        # Save results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total': len(all_results),
            'success': sum(1 for r in all_results if r['status'] == 'success'),
            'results': all_results
        }
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)

        if not args.continuous:
            break

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    success = [r for r in all_results if r['status'] == 'success']
    print(f"Total: {len(all_results)}")
    print(f"Success: {len(success)}")
    if success:
        print("\nSuccessful benchmarks:")
        for r in sorted(success, key=lambda x: x.get('metrics', {}).get('output_throughput', 0), reverse=True):
            tok_s = r.get('metrics', {}).get('output_throughput', 0)
            print(f"  {tok_s:6.0f} tok/s | {r['image'].split(':')[-1][:20]} | v{r.get('sglang_version', '?')}")


if __name__ == "__main__":
    main()
