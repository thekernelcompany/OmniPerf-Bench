import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cmd = [
    sys.executable,
    os.path.join(REPO_ROOT, 'benchmarks', 'benchmark_latency.py'),
    '--model', 'meta-llama/Meta-Llama-3-8B',
    '--load-format', 'dummy',
    '--device', 'cpu',
    '--worker-cls', 'vllm.worker.cpu_worker.CPUWorker',
    '--disable-async-output-proc',
    '--input-len', '8',
    '--output-len', '16',
    '--batch-size', '1',
    '--num-iters-warmup', '1',
    '--num-iters', '3',
]

def run_once():
    start = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    end = time.perf_counter()
    print(proc.stdout)
    if proc.returncode != 0:
        print('Command failed with return code', proc.returncode)
        sys.exit(proc.returncode)
    return end - start

if __name__ == '__main__':
    total = run_once()
    print(f'Elapsed: {total:.3f}s')
