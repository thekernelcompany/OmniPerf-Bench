"""
Modal-based benchmark runner for OmniPerf.

This module provides cloud GPU execution for benchmarks that require:
- Multi-GPU instances (models > 80GB VRAM)
- Specific GPU configurations (H100:4, H100:8)
- Clean, isolated environments

Usage:
    # Deploy the Modal app
    modal deploy src/eval/modal_benchmark.py

    # Or run directly
    modal run src/eval/modal_benchmark.py::run_benchmark_single_gpu --wheel-url "..." --perf-command "..."
"""

import modal
import subprocess
import time
import re
import os
import signal
from typing import Dict, Optional, Any, Tuple
from pathlib import Path

# Modal app configuration
app = modal.App("omniperf-benchmark")

# Base image with CUDA, Python, and benchmark dependencies
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "wget", "patch")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.40.0",
        "huggingface_hub>=0.23.0",
        "tokenizers>=0.19.0",
        "accelerate>=0.30.0",
        "numpy<2.0",
        "requests",
        "aiohttp",
        # Benchmark dependencies
        "datasets",
        "pandas",
        "tqdm",
    )
    .run_commands([
        # Clone vLLM repo for benchmark scripts
        "git clone --depth 1 https://github.com/vllm-project/vllm.git /opt/vllm-benchmarks",
    ])
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
    })
)

# Volume for caching models
model_cache = modal.Volume.from_name("omniperf-model-cache", create_if_missing=True)

# GPU configurations for different model sizes
GPU_CONFIGS = {
    "H100:1": {"gpu": "H100", "count": 1, "timeout": 3600},
    "H100:2": {"gpu": "H100", "count": 2, "timeout": 5400},
    "H100:4": {"gpu": "H100", "count": 4, "timeout": 7200},
    "H100:8": {"gpu": "H100", "count": 8, "timeout": 14400},
}

# Model size to GPU mapping
LARGE_MODEL_GPU_MAP = {
    "deepseek-ai/DeepSeek-V3": "H100:8",
    "deepseek-ai/DeepSeek-V3-0324": "H100:8",
    "deepseek-ai/DeepSeek-V2": "H100:8",
    "deepseek-ai/DeepSeek-V2-Lite": "H100:1",
    "nvidia/Nemotron-4-340B": "H100:8",
    "meta-llama/Llama-4-Scout-17B-16E": "H100:2",
    "meta-llama/Meta-Llama-3-70B": "H100:4",
    "meta-llama/Llama-3.1-70B": "H100:4",
    "meta-llama/Llama-3-70B": "H100:4",
    "70B": "H100:4",  # Catch-all for 70B models
    "70b": "H100:4",
}


def get_gpu_config(model: str, perf_command: str) -> str:
    """Determine GPU requirements based on model and command."""
    # Check for explicit tensor parallelism in command
    tp_match = re.search(r'(?:-tp|--tensor-parallel-size)\s+(\d+)', perf_command)
    if tp_match:
        tp_size = int(tp_match.group(1))
        if tp_size >= 8:
            return "H100:8"
        elif tp_size >= 4:
            return "H100:4"
        elif tp_size >= 2:
            return "H100:2"

    # Check known large models
    for pattern, config in LARGE_MODEL_GPU_MAP.items():
        if pattern.lower() in model.lower():
            return config

    return "H100:1"


def install_wheel(wheel_url: str) -> Tuple[bool, str]:
    """Install a vLLM wheel from URL."""
    try:
        # Uninstall existing vLLM
        subprocess.run(
            ["pip", "uninstall", "-y", "vllm"],
            capture_output=True,
            timeout=60,
        )

        # Install new wheel
        result = subprocess.run(
            ["pip", "install", wheel_url],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            return False, f"Failed to install wheel: {result.stderr}"

        # Verify installation
        version_result = subprocess.run(
            ["python", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
        )

        return True, version_result.stdout.strip()
    except Exception as e:
        return False, str(e)


def start_server(model: str, port: int = 8000, tensor_parallel: int = 1, extra_args: list = None) -> subprocess.Popen:
    """Start vLLM server as background process."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel),
    ]

    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tensor_parallel))
    # Enable unbuffered output for real-time logging
    env["PYTHONUNBUFFERED"] = "1"

    # Stream output directly to container logs (visible in Modal dashboard)
    process = subprocess.Popen(
        cmd,
        stdout=None,  # Inherit parent stdout - streams to Modal logs
        stderr=None,  # Inherit parent stderr - streams to Modal logs
        env=env,
        preexec_fn=os.setsid,  # Create new process group
    )

    return process


def wait_for_server(port: int = 8000, timeout: int = 600) -> bool:
    """Wait for server to be ready."""
    import urllib.request

    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return True
        except:
            pass
        time.sleep(5)

    return False


def stop_server(process: subprocess.Popen):
    """Stop server process and cleanup."""
    if process:
        try:
            # Kill entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
        except:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                pass


def parse_metrics(output: str) -> Dict[str, float]:
    """Parse benchmark output for metrics."""
    metrics = {}

    patterns = {
        r"Request throughput:\s*([\d.]+)\s*requests/s": "request_throughput",
        r"Output token throughput:\s*([\d.]+)\s*tokens/s": "output_throughput",
        r"Total Token throughput:\s*([\d.]+)\s*tokens/s": "total_throughput",
        r"Mean TTFT \(ms\):\s*([\d.]+)": "ttft_mean",
        r"Median TTFT \(ms\):\s*([\d.]+)": "ttft_median",
        r"P99 TTFT \(ms\):\s*([\d.]+)": "ttft_p99",
        r"Mean TPOT \(ms\):\s*([\d.]+)": "tpot_mean",
        r"Mean ITL \(ms\):\s*([\d.]+)": "itl_mean",
        r"Avg latency:\s*([\d.]+)\s*seconds": "latency_avg",
        r"Throughput:\s*([\d.]+)\s*requests/s": "throughput",
        r"throughput[=:]\s*([\d.]+)": "throughput",
    }

    for pattern, key in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def run_simple_benchmark(model: str, port: int = 8000, num_requests: int = 50) -> Dict[str, float]:
    """Run a simple benchmark against the vLLM server."""
    import urllib.request
    import json

    url = f"http://127.0.0.1:{port}/v1/completions"
    metrics = {}

    latencies = []
    total_tokens = 0

    for i in range(num_requests):
        data = json.dumps({
            "model": model,
            "prompt": "Write a short story about a robot learning to paint:",
            "max_tokens": 100,
            "temperature": 0.7
        }).encode()

        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode())
                latency = time.time() - start
                latencies.append(latency)
                if "usage" in result:
                    total_tokens += result["usage"].get("completion_tokens", 0)
        except Exception as e:
            print(f"Request {i} failed: {e}")
            continue

    if latencies:
        total_time = sum(latencies)
        metrics["request_throughput"] = len(latencies) / total_time
        metrics["latency_avg"] = sum(latencies) / len(latencies)
        metrics["latency_p50"] = sorted(latencies)[len(latencies) // 2]
        metrics["latency_p99"] = sorted(latencies)[int(len(latencies) * 0.99)]
        if total_tokens > 0:
            metrics["output_throughput"] = total_tokens / total_time
        metrics["successful_requests"] = len(latencies)
        metrics["total_requests"] = num_requests

    return metrics


def prepare_serving_command(perf_command: str, model: str, port: int = 8000) -> str:
    """Prepare benchmark client command - returns None to use built-in benchmark."""
    # We now use run_simple_benchmark instead of external commands
    return None


@app.function(
    image=base_image,
    gpu="H100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_benchmark_single_gpu(
    wheel_url: str,
    perf_command: str,
    model: str,
    benchmark_type: str = "serving",
    num_prompts: int = 100,
) -> Dict[str, Any]:
    """Run benchmark on single H100 GPU.

    Args:
        wheel_url: URL to vLLM wheel
        perf_command: Benchmark command to run
        model: Model name/path
        benchmark_type: One of "serving", "throughput", "latency"
        num_prompts: Number of prompts for serving benchmark

    Returns:
        Dict with metrics, status, and timing info
    """
    result = {
        "status": "error",
        "metrics": {},
        "vllm_version": None,
        "error": None,
        "duration_s": 0,
    }

    start_time = time.time()

    try:
        # Install wheel
        success, version = install_wheel(wheel_url)
        if not success:
            result["error"] = f"Wheel installation failed: {version}"
            return result

        result["vllm_version"] = version

        if benchmark_type == "serving":
            # Start server
            server = start_server(model, port=8000, tensor_parallel=1)

            try:
                if not wait_for_server(port=8000, timeout=3600):
                    result["error"] = "Server failed to start within timeout"
                    return result

                # Run built-in benchmark
                print(f"Server ready, running benchmark against {model}...")
                benchmark_metrics = run_simple_benchmark(model, port=8000, num_requests=50)

                result["metrics"] = benchmark_metrics
                result["raw_output"] = f"Benchmark completed: {benchmark_metrics}"

                if result["metrics"]:
                    result["status"] = "success"
                else:
                    result["error"] = "No metrics from benchmark"

            finally:
                stop_server(server)

        else:
            # Direct benchmark (throughput, latency)
            bench_result = subprocess.run(
                perf_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            output = bench_result.stdout + bench_result.stderr
            result["metrics"] = parse_metrics(output)
            result["raw_output"] = output[:5000]

            if result["metrics"]:
                result["status"] = "success"
            else:
                result["error"] = "No metrics parsed from output"

    except subprocess.TimeoutExpired:
        result["error"] = "Benchmark timed out"
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["duration_s"] = time.time() - start_time
        model_cache.commit()

    return result


@app.function(
    image=base_image,
    gpu="H100:4",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_benchmark_4gpu(
    wheel_url: str,
    perf_command: str,
    model: str,
    benchmark_type: str = "serving",
) -> Dict[str, Any]:
    """Run benchmark on 4x H100 GPUs for large models (70B-340B)."""
    result = {
        "status": "error",
        "metrics": {},
        "vllm_version": None,
        "error": None,
        "duration_s": 0,
        "gpu_config": "H100:4",
    }

    start_time = time.time()

    try:
        success, version = install_wheel(wheel_url)
        if not success:
            result["error"] = f"Wheel installation failed: {version}"
            return result

        result["vllm_version"] = version

        if benchmark_type == "serving":
            server = start_server(model, port=8000, tensor_parallel=4)

            try:
                if not wait_for_server(port=8000, timeout=900):
                    result["error"] = "Server failed to start within timeout"
                    return result

                # Run built-in benchmark
                print(f"Server ready, running benchmark against {model} with 4x H100...")
                benchmark_metrics = run_simple_benchmark(model, port=8000, num_requests=50)

                result["metrics"] = benchmark_metrics
                result["raw_output"] = f"Benchmark completed: {benchmark_metrics}"

                if result["metrics"]:
                    result["status"] = "success"
                else:
                    result["error"] = "No metrics from benchmark"

            finally:
                stop_server(server)
        else:
            # Update command with tensor parallelism
            cmd = re.sub(r'--tensor-parallel-size\s+\d+', '--tensor-parallel-size 4', perf_command)
            if '--tensor-parallel-size' not in cmd:
                cmd += ' --tensor-parallel-size 4'

            bench_result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=7200,
            )

            output = bench_result.stdout + bench_result.stderr
            result["metrics"] = parse_metrics(output)
            result["raw_output"] = output[:5000]

            if result["metrics"]:
                result["status"] = "success"
            else:
                result["error"] = "No metrics parsed from output"

    except subprocess.TimeoutExpired:
        result["error"] = "Benchmark timed out"
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["duration_s"] = time.time() - start_time
        model_cache.commit()

    return result


@app.function(
    image=base_image,
    gpu="H100:8",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_benchmark_8gpu(
    wheel_url: str,
    perf_command: str,
    model: str,
    benchmark_type: str = "serving",
) -> Dict[str, Any]:
    """Run benchmark on 8x H100 GPUs for massive models (340B-671B)."""
    result = {
        "status": "error",
        "metrics": {},
        "vllm_version": None,
        "error": None,
        "duration_s": 0,
        "gpu_config": "H100:8",
    }

    start_time = time.time()

    try:
        success, version = install_wheel(wheel_url)
        if not success:
            result["error"] = f"Wheel installation failed: {version}"
            return result

        result["vllm_version"] = version

        if benchmark_type == "serving":
            server = start_server(model, port=8000, tensor_parallel=8)

            try:
                if not wait_for_server(port=8000, timeout=3600):
                    result["error"] = "Server failed to start within timeout (check Modal dashboard logs for details)"
                    return result

                # Run built-in benchmark
                print(f"Server ready, running benchmark against {model} with 8x H100...")
                benchmark_metrics = run_simple_benchmark(model, port=8000, num_requests=50)

                result["metrics"] = benchmark_metrics
                result["raw_output"] = f"Benchmark completed: {benchmark_metrics}"

                if result["metrics"]:
                    result["status"] = "success"
                else:
                    result["error"] = "No metrics from benchmark"

            finally:
                stop_server(server)
        else:
            cmd = re.sub(r'--tensor-parallel-size\s+\d+', '--tensor-parallel-size 8', perf_command)
            if '--tensor-parallel-size' not in cmd:
                cmd += ' --tensor-parallel-size 8'

            bench_result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=14400,
            )

            output = bench_result.stdout + bench_result.stderr
            result["metrics"] = parse_metrics(output)
            result["raw_output"] = output[:5000]

            if result["metrics"]:
                result["status"] = "success"
            else:
                result["error"] = "No metrics parsed from output"

    except subprocess.TimeoutExpired:
        result["error"] = "Benchmark timed out"
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["duration_s"] = time.time() - start_time
        model_cache.commit()

    return result


# =====================================================================
# 3-WAY BENCHMARK FUNCTIONS
# Run baseline vs human vs agent comparison with actual perf_command
# =====================================================================

def run_perf_command(perf_command: str, model: str, port: int = 8000) -> Tuple[str, Dict[str, float]]:
    """
    Run the actual benchmark command from the dataset.

    Handles both old-style (python benchmarks/...) and new-style (vllm bench) commands.
    """
    # Translate old benchmark script paths to use /opt/vllm-benchmarks
    cmd = perf_command

    # Replace python benchmark scripts with the cloned repo path
    if "benchmarks/benchmark_serving.py" in cmd:
        cmd = cmd.replace("python benchmarks/benchmark_serving.py",
                         "python /opt/vllm-benchmarks/benchmarks/benchmark_serving.py")
        cmd = cmd.replace("python3 benchmarks/benchmark_serving.py",
                         "python /opt/vllm-benchmarks/benchmarks/benchmark_serving.py")
    elif "benchmarks/benchmark_throughput.py" in cmd:
        cmd = cmd.replace("python benchmarks/benchmark_throughput.py",
                         "python /opt/vllm-benchmarks/benchmarks/benchmark_throughput.py")
        cmd = cmd.replace("python3 benchmarks/benchmark_throughput.py",
                         "python /opt/vllm-benchmarks/benchmarks/benchmark_throughput.py")
    elif "benchmarks/benchmark_latency.py" in cmd:
        cmd = cmd.replace("python benchmarks/benchmark_latency.py",
                         "python /opt/vllm-benchmarks/benchmarks/benchmark_latency.py")
        cmd = cmd.replace("python3 benchmarks/benchmark_latency.py",
                         "python /opt/vllm-benchmarks/benchmarks/benchmark_latency.py")

    # For serving benchmarks, need to add host/port
    if "benchmark_serving" in cmd:
        # Remove server-only args that don't belong in benchmark client
        cmd = re.sub(r'--dtype\s+\S+', '', cmd)
        cmd = re.sub(r'--tensor-parallel-size\s+\d+', '', cmd)
        cmd = re.sub(r'-tp\s+\d+', '', cmd)
        cmd = re.sub(r'--trust-remote-code', '', cmd)
        cmd = re.sub(r'--max-model-len\s+\d+', '', cmd)

        # Add host/port
        if '--host' not in cmd and '--base-url' not in cmd:
            cmd += ' --host 127.0.0.1'
        if '--port' not in cmd and '--base-url' not in cmd:
            cmd += f' --port {port}'

        # Add dataset if not specified (required for benchmark_serving.py)
        if '--dataset-name' not in cmd and '--dataset-path' not in cmd:
            cmd += ' --dataset-name random --random-input-len 512 --random-output-len 128'

        # Add num-prompts if not specified
        if '--num-prompts' not in cmd:
            cmd += ' --num-prompts 100'

    # Clean up multiple spaces
    cmd = re.sub(r'\s+', ' ', cmd).strip()

    print(f"Running benchmark: {cmd[:150]}...")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout for benchmark
            cwd="/opt/vllm-benchmarks",
        )

        output = result.stdout + "\n" + result.stderr
        metrics = parse_metrics(output)

        return output, metrics

    except subprocess.TimeoutExpired:
        return "Benchmark timed out", {}
    except Exception as e:
        return f"Error: {str(e)}", {}


def apply_patch_to_vllm(patch_content: str) -> Tuple[bool, str]:
    """
    Apply a patch to the installed vLLM package.
    Only applies Python file changes, skips C/CUDA extensions.

    Returns:
        (success, message) tuple
    """
    import site
    import tempfile

    # Check if patch modifies C/CUDA files
    c_extensions = ['.c', '.cpp', '.cu', '.cuh', '.h', '.hpp']
    has_c_changes = any(ext in patch_content for ext in c_extensions)

    if has_c_changes:
        return False, "Patch modifies C/CUDA extensions - cannot apply without recompilation"

    # Find vLLM in site-packages
    vllm_path = None
    for sp in site.getsitepackages():
        candidate = Path(sp) / "vllm"
        if candidate.exists():
            vllm_path = Path(sp)
            break

    if not vllm_path:
        return False, "Could not find vLLM in site-packages"

    # Write patch to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
        f.write(patch_content)
        patch_file = f.name

    try:
        # Try to apply the patch
        # First rewrite paths: a/vllm/... -> site-packages/vllm/...
        modified_patch = patch_content

        # Find all vllm/ paths in the patch and make them absolute
        file_matches = re.findall(r'(?:a|b)/vllm/([^\s]+\.py)', patch_content)
        for rel_path in set(file_matches):
            old_a = f"a/vllm/{rel_path}"
            old_b = f"b/vllm/{rel_path}"
            new_path = str(vllm_path / "vllm" / rel_path)
            modified_patch = modified_patch.replace(old_a, new_path)
            modified_patch = modified_patch.replace(old_b, new_path)

        # Write modified patch
        with open(patch_file, 'w') as f:
            f.write(modified_patch)

        # Apply with patch command
        result = subprocess.run(
            ["patch", "-p0", "--forward", "--ignore-whitespace", "-i", patch_file],
            capture_output=True,
            text=True,
            cwd="/",
        )

        if result.returncode == 0:
            return True, f"Patch applied successfully to {vllm_path}"
        else:
            return False, f"Patch failed: {result.stderr}"

    finally:
        try:
            os.unlink(patch_file)
        except:
            pass


def compute_improvement(baseline: Dict[str, float], other: Dict[str, float]) -> Dict[str, float]:
    """Compute percentage improvement for each metric."""
    improvement = {}

    # Throughput metrics (higher is better)
    for key in ["request_throughput", "output_throughput", "total_throughput", "throughput"]:
        b = baseline.get(key)
        o = other.get(key)
        if b and o and b > 0:
            improvement[key] = ((o - b) / b) * 100

    # Latency metrics (lower is better, so invert)
    for key in ["ttft_mean", "ttft_median", "ttft_p99", "tpot_mean", "itl_mean", "latency_avg"]:
        b = baseline.get(key)
        o = other.get(key)
        if b and o and b > 0:
            improvement[key] = ((b - o) / b) * 100

    return improvement


@app.function(
    image=base_image,
    gpu="H100:4",
    timeout=10800,  # 3 hours for 3 benchmarks
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_3way_benchmark_4gpu(
    baseline_wheel_url: str,
    human_wheel_url: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
) -> Dict[str, Any]:
    """
    Run 3-way benchmark comparison on 4x H100 GPUs.

    1. Baseline wheel → metrics
    2. Human wheel → metrics
    3. Baseline wheel + agent patch → metrics (if Python-only)
    """
    result = {
        "status": "error",
        "gpu_config": "H100:4",
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": None,
        "human_improvement": {},
        "agent_improvement": None,
        "agent_vs_human": None,
        "error": None,
        "duration_s": 0,
        "perf_command": perf_command,
    }

    start_time = time.time()
    tensor_parallel = 4

    try:
        # ========== 1. BASELINE BENCHMARK ==========
        print(f"[1/3] Installing baseline wheel...")
        success, version = install_wheel(baseline_wheel_url)
        if not success:
            result["error"] = f"Baseline wheel install failed: {version}"
            return result

        result["baseline_version"] = version
        print(f"Baseline vLLM: {version}")

        print(f"[1/3] Starting server for BASELINE benchmark...")
        server = start_server(model, port=8000, tensor_parallel=tensor_parallel)

        try:
            if not wait_for_server(port=8000, timeout=3600):
                result["error"] = "Baseline server failed to start"
                return result

            print(f"[1/3] Running BASELINE benchmark...")
            baseline_output, baseline_metrics = run_perf_command(perf_command, model, port=8000)
            result["baseline_metrics"] = baseline_metrics
            result["baseline_raw"] = baseline_output[:3000]

            if not baseline_metrics:
                result["error"] = "Baseline benchmark produced no metrics"
                result["status"] = "baseline_failed"
                return result

            print(f"Baseline metrics: {baseline_metrics}")

        finally:
            stop_server(server)

        # ========== 2. HUMAN BENCHMARK ==========
        print(f"[2/3] Installing human wheel...")
        success, version = install_wheel(human_wheel_url)
        if not success:
            result["error"] = f"Human wheel install failed: {version}"
            result["status"] = "human_wheel_failed"
            return result

        result["human_version"] = version
        print(f"Human vLLM: {version}")

        print(f"[2/3] Starting server for HUMAN benchmark...")
        server = start_server(model, port=8000, tensor_parallel=tensor_parallel)

        try:
            if not wait_for_server(port=8000, timeout=3600):
                result["error"] = "Human server failed to start"
                result["status"] = "human_server_failed"
                return result

            print(f"[2/3] Running HUMAN benchmark...")
            human_output, human_metrics = run_perf_command(perf_command, model, port=8000)
            result["human_metrics"] = human_metrics
            result["human_raw"] = human_output[:3000]

            if not human_metrics:
                result["error"] = "Human benchmark produced no metrics"
                result["status"] = "human_failed"
                return result

            print(f"Human metrics: {human_metrics}")
            result["human_improvement"] = compute_improvement(baseline_metrics, human_metrics)
            print(f"Human improvement: {result['human_improvement']}")

        finally:
            stop_server(server)

        # ========== 3. AGENT BENCHMARK ==========
        if agent_patch:
            print(f"[3/3] Setting up AGENT benchmark...")

            # Reinstall baseline wheel
            success, version = install_wheel(baseline_wheel_url)
            if not success:
                print(f"Warning: Could not reinstall baseline for agent test")
                result["agent_metrics"] = None
                result["agent_error"] = "Could not reinstall baseline wheel"
            else:
                # Apply patch
                patch_success, patch_msg = apply_patch_to_vllm(agent_patch)

                if not patch_success:
                    print(f"Warning: {patch_msg}")
                    result["agent_metrics"] = None
                    result["agent_error"] = patch_msg
                else:
                    print(f"Patch applied: {patch_msg}")

                    print(f"[3/3] Starting server for AGENT benchmark...")
                    server = start_server(model, port=8000, tensor_parallel=tensor_parallel)

                    try:
                        if not wait_for_server(port=8000, timeout=3600):
                            result["agent_metrics"] = None
                            result["agent_error"] = "Agent server failed to start"
                        else:
                            print(f"[3/3] Running AGENT benchmark...")
                            agent_output, agent_metrics = run_perf_command(perf_command, model, port=8000)
                            result["agent_metrics"] = agent_metrics
                            result["agent_raw"] = agent_output[:3000]

                            if agent_metrics:
                                result["agent_improvement"] = compute_improvement(baseline_metrics, agent_metrics)
                                result["agent_vs_human"] = compute_improvement(human_metrics, agent_metrics)
                                print(f"Agent metrics: {agent_metrics}")
                                print(f"Agent improvement: {result['agent_improvement']}")
                            else:
                                result["agent_error"] = "Agent benchmark produced no metrics"
                    finally:
                        stop_server(server)
        else:
            print(f"[3/3] No agent patch provided, skipping")
            result["agent_metrics"] = None

        result["status"] = "success"

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error"
    finally:
        result["duration_s"] = time.time() - start_time
        model_cache.commit()

    return result


@app.function(
    image=base_image,
    gpu="H100:2",
    timeout=10800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_3way_benchmark_2gpu(
    baseline_wheel_url: str,
    human_wheel_url: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
) -> Dict[str, Any]:
    """Run 3-way benchmark on 2x H100 GPUs."""
    # Reuse 4GPU logic with different tensor_parallel
    result = {
        "status": "error",
        "gpu_config": "H100:2",
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": None,
        "human_improvement": {},
        "agent_improvement": None,
        "agent_vs_human": None,
        "error": None,
        "duration_s": 0,
        "perf_command": perf_command,
    }

    start_time = time.time()
    tensor_parallel = 2

    try:
        # Same logic as 4gpu but with tp=2
        # ... (abbreviated - same structure as run_3way_benchmark_4gpu)

        # BASELINE
        print(f"[1/3] Installing baseline wheel...")
        success, version = install_wheel(baseline_wheel_url)
        if not success:
            result["error"] = f"Baseline wheel install failed: {version}"
            return result

        result["baseline_version"] = version
        server = start_server(model, port=8000, tensor_parallel=tensor_parallel)

        try:
            if not wait_for_server(port=8000, timeout=3600):
                result["error"] = "Baseline server failed to start"
                return result

            baseline_output, baseline_metrics = run_perf_command(perf_command, model, port=8000)
            result["baseline_metrics"] = baseline_metrics

            if not baseline_metrics:
                result["error"] = "Baseline benchmark produced no metrics"
                result["status"] = "baseline_failed"
                return result
        finally:
            stop_server(server)

        # HUMAN
        print(f"[2/3] Installing human wheel...")
        success, version = install_wheel(human_wheel_url)
        if not success:
            result["error"] = f"Human wheel install failed: {version}"
            return result

        result["human_version"] = version
        server = start_server(model, port=8000, tensor_parallel=tensor_parallel)

        try:
            if not wait_for_server(port=8000, timeout=3600):
                result["error"] = "Human server failed to start"
                return result

            human_output, human_metrics = run_perf_command(perf_command, model, port=8000)
            result["human_metrics"] = human_metrics

            if not human_metrics:
                result["error"] = "Human benchmark produced no metrics"
                result["status"] = "human_failed"
                return result

            result["human_improvement"] = compute_improvement(baseline_metrics, human_metrics)
        finally:
            stop_server(server)

        # AGENT
        if agent_patch:
            success, version = install_wheel(baseline_wheel_url)
            if success:
                patch_success, patch_msg = apply_patch_to_vllm(agent_patch)
                if patch_success:
                    server = start_server(model, port=8000, tensor_parallel=tensor_parallel)
                    try:
                        if wait_for_server(port=8000, timeout=3600):
                            agent_output, agent_metrics = run_perf_command(perf_command, model, port=8000)
                            result["agent_metrics"] = agent_metrics
                            if agent_metrics:
                                result["agent_improvement"] = compute_improvement(baseline_metrics, agent_metrics)
                                result["agent_vs_human"] = compute_improvement(human_metrics, agent_metrics)
                    finally:
                        stop_server(server)
                else:
                    result["agent_error"] = patch_msg

        result["status"] = "success"

    except Exception as e:
        result["error"] = str(e)
    finally:
        result["duration_s"] = time.time() - start_time
        model_cache.commit()

    return result


@app.function(
    image=base_image,
    gpu="H100:8",
    timeout=14400,  # 4 hours
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_3way_benchmark_8gpu(
    baseline_wheel_url: str,
    human_wheel_url: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
) -> Dict[str, Any]:
    """Run 3-way benchmark on 8x H100 GPUs."""
    result = {
        "status": "error",
        "gpu_config": "H100:8",
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": None,
        "human_improvement": {},
        "agent_improvement": None,
        "agent_vs_human": None,
        "error": None,
        "duration_s": 0,
        "perf_command": perf_command,
    }

    start_time = time.time()
    tensor_parallel = 8

    try:
        # BASELINE
        print(f"[1/3] Installing baseline wheel...")
        success, version = install_wheel(baseline_wheel_url)
        if not success:
            result["error"] = f"Baseline wheel install failed: {version}"
            return result

        result["baseline_version"] = version
        server = start_server(model, port=8000, tensor_parallel=tensor_parallel)

        try:
            if not wait_for_server(port=8000, timeout=3600):
                result["error"] = "Baseline server failed to start"
                return result

            baseline_output, baseline_metrics = run_perf_command(perf_command, model, port=8000)
            result["baseline_metrics"] = baseline_metrics

            if not baseline_metrics:
                result["error"] = "Baseline benchmark produced no metrics"
                result["status"] = "baseline_failed"
                return result
        finally:
            stop_server(server)

        # HUMAN
        print(f"[2/3] Installing human wheel...")
        success, version = install_wheel(human_wheel_url)
        if not success:
            result["error"] = f"Human wheel install failed: {version}"
            return result

        result["human_version"] = version
        server = start_server(model, port=8000, tensor_parallel=tensor_parallel)

        try:
            if not wait_for_server(port=8000, timeout=3600):
                result["error"] = "Human server failed to start"
                return result

            human_output, human_metrics = run_perf_command(perf_command, model, port=8000)
            result["human_metrics"] = human_metrics

            if not human_metrics:
                result["error"] = "Human benchmark produced no metrics"
                result["status"] = "human_failed"
                return result

            result["human_improvement"] = compute_improvement(baseline_metrics, human_metrics)
        finally:
            stop_server(server)

        # AGENT
        if agent_patch:
            success, version = install_wheel(baseline_wheel_url)
            if success:
                patch_success, patch_msg = apply_patch_to_vllm(agent_patch)
                if patch_success:
                    server = start_server(model, port=8000, tensor_parallel=tensor_parallel)
                    try:
                        if wait_for_server(port=8000, timeout=3600):
                            agent_output, agent_metrics = run_perf_command(perf_command, model, port=8000)
                            result["agent_metrics"] = agent_metrics
                            if agent_metrics:
                                result["agent_improvement"] = compute_improvement(baseline_metrics, agent_metrics)
                                result["agent_vs_human"] = compute_improvement(human_metrics, agent_metrics)
                    finally:
                        stop_server(server)
                else:
                    result["agent_error"] = patch_msg

        result["status"] = "success"

    except Exception as e:
        result["error"] = str(e)
    finally:
        result["duration_s"] = time.time() - start_time
        model_cache.commit()

    return result


def run_3way_modal_benchmark(
    baseline_wheel_url: str,
    human_wheel_url: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
    gpu_config: str = None,
) -> Dict[str, Any]:
    """
    Run 3-way benchmark on Modal with automatic GPU selection.

    This is the main entry point for calling from native_benchmark_runner.py
    """
    if gpu_config is None:
        gpu_config = get_gpu_config(model, perf_command)

    print(f"Running 3-way benchmark on Modal with {gpu_config}...")

    if gpu_config == "H100:8":
        fn = modal.Function.from_name("omniperf-benchmark", "run_3way_benchmark_8gpu")
    elif gpu_config == "H100:4":
        fn = modal.Function.from_name("omniperf-benchmark", "run_3way_benchmark_4gpu")
    elif gpu_config == "H100:2":
        fn = modal.Function.from_name("omniperf-benchmark", "run_3way_benchmark_2gpu")
    else:
        # Fall back to 4GPU for unknown configs
        fn = modal.Function.from_name("omniperf-benchmark", "run_3way_benchmark_4gpu")

    return fn.remote(
        baseline_wheel_url=baseline_wheel_url,
        human_wheel_url=human_wheel_url,
        agent_patch=agent_patch,
        perf_command=perf_command,
        model=model,
    )


# Helper function for local orchestration
def run_modal_benchmark(
    wheel_url: str,
    perf_command: str,
    model: str,
    benchmark_type: str = "serving",
    gpu_config: str = None,
) -> Dict[str, Any]:
    """
    Run a benchmark on Modal with automatic GPU selection.

    This function should be called from the local runner to dispatch
    benchmarks to Modal cloud GPUs.

    Args:
        wheel_url: URL to vLLM wheel on wheels.vllm.ai
        perf_command: The benchmark command to execute
        model: Model name (used for GPU selection)
        benchmark_type: "serving", "throughput", or "latency"
        gpu_config: Override GPU config (e.g., "H100:4")

    Returns:
        Benchmark results dict with metrics and status
    """
    # Determine GPU configuration
    if gpu_config is None:
        gpu_config = get_gpu_config(model, perf_command)

    # Use Function.lookup to find deployed functions
    # This is required when calling from outside the Modal app context
    if gpu_config == "H100:8":
        fn = modal.Function.from_name("omniperf-benchmark", "run_benchmark_8gpu")
        return fn.remote(
            wheel_url=wheel_url,
            perf_command=perf_command,
            model=model,
            benchmark_type=benchmark_type,
        )
    elif gpu_config == "H100:4":
        fn = modal.Function.from_name("omniperf-benchmark", "run_benchmark_4gpu")
        return fn.remote(
            wheel_url=wheel_url,
            perf_command=perf_command,
            model=model,
            benchmark_type=benchmark_type,
        )
    else:
        fn = modal.Function.from_name("omniperf-benchmark", "run_benchmark_single_gpu")
        return fn.remote(
            wheel_url=wheel_url,
            perf_command=perf_command,
            model=model,
            benchmark_type=benchmark_type,
        )


@app.local_entrypoint()
def main(
    wheel_url: str = "",
    perf_command: str = "",
    model: str = "",
    benchmark_type: str = "serving",
):
    """CLI entrypoint for testing Modal deployment."""
    if not all([wheel_url, perf_command, model]):
        print("Usage: modal run modal_benchmark.py --wheel-url URL --perf-command CMD --model MODEL")
        return

    gpu_config = get_gpu_config(model, perf_command)
    print(f"Selected GPU config: {gpu_config}")

    result = run_modal_benchmark(
        wheel_url=wheel_url,
        perf_command=perf_command,
        model=model,
        benchmark_type=benchmark_type,
        gpu_config=gpu_config,
    )

    print(f"Status: {result['status']}")
    print(f"Metrics: {result.get('metrics', {})}")
    if result.get('error'):
        print(f"Error: {result['error']}")
    if result.get('raw_output'):
        print(f"Raw output:\n{result['raw_output']}")
