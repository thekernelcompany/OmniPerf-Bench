#!/usr/bin/env python3
"""
Local Docker 3-Way Benchmark Runner for SGLang.

Runs benchmarks inside pre-built Docker containers for SGLang commits.
Supports 3-way comparison: baseline (parent) vs human vs agent.

Uses pre-built Docker images from shikhar481/sglang-images.

Usage:
    # Run all 3 commit pairs
    python scripts/runners/local_docker_sglang_benchmark.py

    # Run specific commit
    python scripts/runners/local_docker_sglang_benchmark.py --commit d1112d85

    # Dry run (show what would be run)
    python scripts/runners/local_docker_sglang_benchmark.py --dry-run

    # Skip agent phase
    python scripts/runners/local_docker_sglang_benchmark.py --no-agent
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

# Configuration
DOCKER_REPO = "shikhar481/sglang-images"
# v2 results: re-run with correct models/perf_commands per HuggingFace dataset
RESULTS_DIR = Path("/root/sglang-images/OmniPerf-Bench/results/sglang_v2")
AGENT_PATCHES_BASE = Path("/root/sglang-images/OmniPerf-Bench/perf-agents-bench/state/runs/sglang/claude_code")

# Commit pairs to benchmark (human commit + parent commit)
# image_suffix: appended to short hash for Docker image tag (e.g., "-vllm-style")
COMMIT_PAIRS = [
    # Original commits (torch 2.5.1, triton 3.1.0)
    {
        "human_commit": "d1112d8548eb13c842900b3a8d622345f9737759",
        "parent_commit": "48efec7b052354865aa2f0605a5bf778721f3cbb",
        "model": "google/gemma-2-2b",
        "perf_command": "python -m sglang.bench_serving --backend sglang --model google/gemma-2-2b --num-prompts 100",
        "agent_patch_dir": "sglang_064_d1112d85",
        "image_suffix": "",  # Uses full commit hash
    },
    {
        "human_commit": "93470a14116a60fe5dd43f0599206e8ccabdc211",
        "parent_commit": "db452760e5b2378efd06b1ceb9385d2eeb6d217c",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "perf_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
        "agent_patch_dir": "sglang_043_93470a14",
        "image_suffix": "",
    },
    {
        "human_commit": "9c088829ee2a28263f36d0814fde448c6090b5bc",
        "parent_commit": "005aad32ad45ce27d73fd39aa1f7e9ba5d8ebb8f",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "perf_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
        "agent_patch_dir": "sglang_045_9c088829",
        "image_suffix": "",
    },
    # ===========================================================================
    # CORRECT COMMIT CONFIGS FROM HuggingFace dataset: Ayushnangia/omniperf_v1
    # Each commit needs its specific model and perf_command!
    # ===========================================================================

    # Pair 1: May 2025 - FA3 (FlashAttention 3) optimization
    # SGLang 0.4.6.post2, torch 2.6.0, triton 3.2.0
    # Optimization: FA3 kernel improvements for Llama models
    {
        "human_commit": "1acca3a2c685221cdb181c2abda4f635e1ead435",
        "parent_commit": "6ea1e6ac6e2fa949cebd1b4338f9bfb7036d14fe",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "perf_command": "python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100",
        "agent_patch_dir": "sglang_006_1acca3a2",
        "image_suffix": "-vllm-style",
        "use_short_hash": True,
        "use_flashinfer": True,  # FA3 optimization requires flashinfer
    },
    # Pair 2: Jun 2025 - LoRA optimization
    # SGLang 0.4.7, torch 2.7.1, triton 3.3.1
    # Optimization: LoRA adapter performance improvements
    # REQUIRES: LoRA adapter algoprog/fact-generation-llama-3.1-8b-instruct-lora
    {
        "human_commit": "021f76e4f49861b2e9ea9ccff06a46d577e3c548",
        "parent_commit": "777688b8929c877e4e28c2eac208d776abe4c3af",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "perf_command": "python3 -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompt 480 --request-rate 8 --lora-name lora",
        "agent_patch_dir": "sglang_000_021f76e4",
        "image_suffix": "",
        "use_short_hash": True,
        "use_flashinfer": True,
        "lora_adapter": "lora=algoprog/fact-generation-llama-3.1-8b-instruct-lora",  # Named LoRA adapter (name=path format)
    },
    # Pair 3: Jul 2025 - VLM pybase64 optimization
    # SGLang 0.4.9, torch 2.7.1, triton 3.3.1
    # Optimization: pybase64 image encoding for VLM models
    # REQUIRES: MMMU dataset for multimodal benchmark
    {
        "human_commit": "a37e1247c183cff86a18f2ed1a075e40704b1c5e",
        "parent_commit": "136c6e0431c2067c3a2a98ad2c77fc89a9cb98e7",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "perf_command": "python3 -m sglang.bench_serving --backend sglang --model Qwen/Qwen2.5-VL-7B-Instruct --dataset-name mmmu --request-rate 10 --num-prompts 100",
        "agent_patch_dir": "sglang_048_a37e1247",
        "image_suffix": "-vllm-style",
        "use_short_hash": True,
        "use_flashinfer": True,
    },
    # Pair 4: Jul 2025 - VLM tensor transport optimization
    # SGLang 0.4.9.post4, torch 2.7.1, triton 3.3.1
    # Optimization: Tensor transport for VLM embedding outputs
    {
        "human_commit": "3212c2ad3f7e4fb473dc807b4b176020a778ed5b",
        "parent_commit": "534756749ae4e664f762de2645a4f63ca2901bab",
        "model": "OpenGVLab/InternVL2_5-8B",
        "perf_command": "python3 -m sglang.bench_serving --backend sglang --model OpenGVLab/InternVL2_5-8B",
        "agent_patch_dir": "sglang_020_3212c2ad",
        "image_suffix": "",
        "use_short_hash": True,
        "use_flashinfer": True,  # VLM optimization requires flashinfer
    },
]


@dataclass
class BenchmarkResult:
    """Benchmark result data."""
    commit_hash: str
    phase: str  # baseline, human, agent
    status: str  # success, error, timeout
    model: str
    duration_s: float
    error: Optional[str] = None
    # Metrics
    request_throughput: Optional[float] = None
    output_throughput: Optional[float] = None
    input_throughput: Optional[float] = None
    ttft_mean: Optional[float] = None
    ttft_median: Optional[float] = None
    ttft_p99: Optional[float] = None
    tpot_mean: Optional[float] = None
    tpot_median: Optional[float] = None
    tpot_p99: Optional[float] = None
    itl_mean: Optional[float] = None
    itl_median: Optional[float] = None
    e2e_latency_mean: Optional[float] = None
    raw_output: Optional[str] = None


def get_hf_token() -> str:
    """Get HuggingFace token."""
    # Try reading from token file directly
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()

    # Try environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Fallback to huggingface_hub
    try:
        result = subprocess.run(
            ["python3", "-c", "from huggingface_hub import get_token; print(get_token() or '')"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return ""


def find_agent_patch(agent_patch_dir: str) -> Optional[Path]:
    """Find agent patch file for a commit."""
    # Search in agent patches directory
    for run_dir in AGENT_PATCHES_BASE.glob("*/*/"):
        patch_dir = run_dir / agent_patch_dir
        if patch_dir.exists():
            patch_file = patch_dir / "model_patch.diff"
            if patch_file.exists() and patch_file.stat().st_size > 0:
                return patch_file
    return None


def parse_sglang_metrics(output: str) -> Dict[str, float]:
    """Parse metrics from SGLang benchmark output."""
    metrics = {}

    patterns = {
        "request_throughput": r"Request throughput[:\s]+\(?req/s\)?[:\s]+([\d.]+)",
        "output_throughput": r"Output token throughput[:\s]+\(?tok/s\)?[:\s]+([\d.]+)",
        "input_throughput": r"Input token throughput[:\s]+\(?tok/s\)?[:\s]+([\d.]+)",
        "ttft_mean": r"Mean TTFT[:\s]+\(?ms\)?[:\s]+([\d.]+)",
        "ttft_median": r"Median TTFT[:\s]+\(?ms\)?[:\s]+([\d.]+)",
        "ttft_p99": r"P99 TTFT[:\s]+\(?ms\)?[:\s]+([\d.]+)",
        "tpot_mean": r"Mean TPOT[:\s]+\(?ms\)?[:\s]+([\d.]+)",
        "tpot_median": r"Median TPOT[:\s]+\(?ms\)?[:\s]+([\d.]+)",
        "tpot_p99": r"P99 TPOT[:\s]+\(?ms\)?[:\s]+([\d.]+)",
        "itl_mean": r"Mean ITL[:\s]+\(?ms\)?[:\s]+([\d.]+)",
        "itl_median": r"Median ITL[:\s]+\(?ms\)?[:\s]+([\d.]+)",
        "e2e_latency_mean": r"Mean E2E Latency[:\s]+\(?ms\)?[:\s]+([\d.]+)",
    }

    for name, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics[name] = float(match.group(1))

    return metrics


def create_benchmark_script(model: str, perf_command: str, port: int = 30000,
                            agent_patch: Optional[str] = None,
                            parent_commit: Optional[str] = None,
                            use_flashinfer: bool = False,
                            lora_adapter: Optional[str] = None) -> str:
    """Create the benchmark script to run inside Docker container."""

    # Escape special characters in perf_command
    perf_command_escaped = perf_command.replace('"', '\\"')

    # Add port to perf_command if not present
    if "--port" not in perf_command:
        perf_command_escaped += f" --port {port}"

    # Base benchmark script
    script = f'''
set -e

echo "=== SGLang Benchmark Script ==="
echo "Model: {model}"
echo "Port: {port}"

# Set environment
export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

# Ensure dependencies
pip install aiohttp requests numpy tqdm -q 2>/dev/null || true

# Install sgl-kernel if missing (fixes deep_gemm import error)
echo "Checking sgl-kernel..."
if ! python3 -c "import sgl_kernel" 2>/dev/null; then
    echo "Installing sgl-kernel..."
    pip install sgl-kernel -q 2>&1 || echo "sgl-kernel install warning"
fi

# Patch fp8_kernel.py to make deep_gemm import optional (it's not on PyPI)
# Only patch if deep_gemm is not already available
if ! python3 -c "import deep_gemm" 2>/dev/null; then
    echo "deep_gemm not found, patching fp8_kernel.py..."
    python3 << 'PYEOF'
import os
try:
    import sglang
    fp8_path = os.path.join(os.path.dirname(sglang.__file__), 'srt/layers/quantization/fp8_kernel.py')
    if os.path.exists(fp8_path):
        with open(fp8_path, 'r') as f:
            content = f.read()
        if 'import deep_gemm' in content and 'try:' not in content.split('import deep_gemm')[0][-20:]:
            patch_text = "try:\\n    import deep_gemm\\nexcept ImportError:\\n    deep_gemm = None"
            new_content = content.replace('import deep_gemm', patch_text.replace('\\n', chr(10)))
            with open(fp8_path, 'w') as f:
                f.write(new_content)
            print("Patched fp8_kernel.py")
        else:
            print("fp8_kernel.py already patched or different format")
except Exception as ex:
    print("Patch skipped: " + str(ex))
PYEOF
else
    echo "deep_gemm already available, skipping patch"
fi

'''

    # Add agent patch application if provided
    if agent_patch and parent_commit:
        script += f'''
# === Apply Agent Patch ===
echo "Applying agent patch..."

# Save patch content
cat > /tmp/agent.patch << 'PATCH_EOF'
{agent_patch}
PATCH_EOF

# Clone SGLang repo at parent commit
cd /tmp
rm -rf sglang_patch 2>/dev/null || true
git clone --depth 100 https://github.com/sgl-project/sglang.git sglang_patch 2>/dev/null
cd sglang_patch
git fetch origin {parent_commit} 2>/dev/null || true
git checkout {parent_commit} 2>/dev/null || git checkout HEAD

# Apply the patch
if git apply --verbose /tmp/agent.patch 2>&1; then
    echo "AGENT_PATCH_APPLIED"
else
    echo "Trying patch -p1..."
    if patch -p1 < /tmp/agent.patch 2>&1; then
        echo "AGENT_PATCH_APPLIED"
    else
        echo "AGENT_PATCH_FAILED"
        exit 1
    fi
fi

# Find installed SGLang path
SGLANG_PATH=$(python3 -c "import sglang; print(sglang.__path__[0])" 2>/dev/null || echo "")
if [ -z "$SGLANG_PATH" ]; then
    # Try common paths
    for p in /opt/sglang/python/sglang /usr/local/lib/python3.11/dist-packages/sglang /sgl-workspace/sglang/python/sglang; do
        if [ -d "$p" ]; then
            SGLANG_PATH=$p
            break
        fi
    done
fi

if [ -n "$SGLANG_PATH" ] && [ -d "$SGLANG_PATH" ]; then
    echo "Overlaying patched files to $SGLANG_PATH"
    cp -r /tmp/sglang_patch/python/sglang/* "$SGLANG_PATH/" 2>/dev/null || true
    echo "Patch overlay complete"
else
    echo "WARNING: Could not find SGLang installation path"
fi

cd /workspace
'''

    # Continue with benchmark execution - choose backend based on config
    if use_flashinfer:
        # Use default flashinfer/fa3 backend (works with triton 3.3.1 on H100)
        # Required for SGLang 0.4.9.post4 which has a bug with torch_native + Gemma
        lora_args = ""
        if lora_adapter:
            # LoRA requires --disable-radix-cache in SGLang
            lora_args = f"--lora-paths {lora_adapter} --disable-radix-cache"
        server_launch = f'''
# === Start SGLang Server ===
echo "Starting SGLang server (flashinfer/fa3 backend)..."
{f'echo "LoRA adapter: {lora_adapter}"' if lora_adapter else ''}

python3 -m sglang.launch_server \\
    --model-path {model} \\
    --port {port} \\
    --host 127.0.0.1 \\
    {lora_args} \\
    --log-level warning 2>&1 &
SERVER_PID=$!'''
    else:
        # Use torch_native backend to avoid triton JIT segfault on older versions
        server_launch = f'''
# === Start SGLang Server ===
echo "Starting SGLang server (torch_native backend)..."

# H100 workaround: Use torch_native backend to avoid triton JIT segfault
# See BUILD_STATUS.md for details
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

python3 -m sglang.launch_server \\
    --model-path {model} \\
    --port {port} \\
    --host 127.0.0.1 \\
    --dtype float16 \\
    --attention-backend torch_native \\
    --sampling-backend pytorch \\
    --disable-cuda-graph \\
    --disable-radix-cache \\
    --log-level warning 2>&1 &
SERVER_PID=$!'''

    script += server_launch

    # Continue with server wait and benchmark
    script += f'''

echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to start..."
SERVER_READY=false
for i in $(seq 1 300); do
    # Check if server process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "SERVER_CRASHED"
        exit 1
    fi

    # Try health endpoint
    if curl -s http://127.0.0.1:{port}/health 2>/dev/null | grep -q "ok\\|healthy\\|true"; then
        echo "SERVER_READY after ${{i}}s"
        SERVER_READY=true
        break
    fi

    # Try v1/models endpoint
    if curl -s http://127.0.0.1:{port}/v1/models 2>/dev/null | grep -q "model"; then
        echo "SERVER_READY after ${{i}}s (v1/models)"
        SERVER_READY=true
        break
    fi

    sleep 1
done

if [ "$SERVER_READY" != "true" ]; then
    echo "SERVER_TIMEOUT"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Give server a moment to stabilize
sleep 5

# === Run Benchmark ===
echo "Running benchmark..."
echo "Command: {perf_command_escaped}"

{perf_command_escaped} 2>&1

BENCH_EXIT=$?

# Cleanup
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
sleep 2

if [ $BENCH_EXIT -eq 0 ]; then
    echo "BENCHMARK_SUCCESS"
else
    echo "BENCHMARK_FAILED"
fi

exit $BENCH_EXIT
'''

    return script


def run_docker_benchmark(
    docker_image: str,
    phase: str,
    commit: str,
    model: str,
    perf_command: str,
    hf_token: str,
    agent_patch: Optional[str] = None,
    parent_commit: Optional[str] = None,
    timeout: int = 1800,  # 30 minutes
    use_flashinfer: bool = False,
    lora_adapter: Optional[str] = None,
) -> BenchmarkResult:
    """Run a benchmark inside a Docker container."""
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"[{phase.upper()}] Running benchmark for {commit[:8]}")
    print(f"  Image: {docker_image}")
    print(f"  Model: {model}")
    print(f"{'='*60}")

    # Create benchmark script
    script = create_benchmark_script(
        model=model,
        perf_command=perf_command,
        agent_patch=agent_patch,
        parent_commit=parent_commit,
        use_flashinfer=use_flashinfer,
        lora_adapter=lora_adapter,
    )

    # Docker command
    # H100 workaround: Set env vars to disable torch compile/dynamo (triton segfault)
    docker_cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-e", f"HF_TOKEN={hf_token}",
        "-e", f"HUGGING_FACE_HUB_TOKEN={hf_token}",
        "-e", "TORCH_COMPILE_DISABLE=1",
        "-e", "TORCHDYNAMO_DISABLE=1",
        "-v", "/ephemeral/huggingface:/root/.cache/huggingface",
        "--shm-size=16g",
        "--entrypoint", "bash",
        docker_image,
        "-c", script
    ]

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        output = result.stdout + "\n" + result.stderr
        duration = time.time() - start_time

        # Check for failures
        if "SERVER_CRASHED" in output:
            return BenchmarkResult(
                commit_hash=commit, phase=phase, status="error",
                model=model, duration_s=duration,
                error="Server crashed during startup",
                raw_output=output[-10000:]
            )

        if "SERVER_TIMEOUT" in output:
            return BenchmarkResult(
                commit_hash=commit, phase=phase, status="error",
                model=model, duration_s=duration,
                error="Server startup timeout",
                raw_output=output[-10000:]
            )

        if "AGENT_PATCH_FAILED" in output:
            return BenchmarkResult(
                commit_hash=commit, phase=phase, status="error",
                model=model, duration_s=duration,
                error="Failed to apply agent patch",
                raw_output=output[-10000:]
            )

        # Parse metrics
        metrics = parse_sglang_metrics(output)

        if not metrics:
            return BenchmarkResult(
                commit_hash=commit, phase=phase, status="error",
                model=model, duration_s=duration,
                error="No metrics found in output",
                raw_output=output[-10000:]
            )

        return BenchmarkResult(
            commit_hash=commit, phase=phase, status="success",
            model=model, duration_s=duration,
            raw_output=output[-10000:],
            **metrics
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=commit, phase=phase, status="timeout",
            model=model, duration_s=timeout,
            error=f"Benchmark timed out after {timeout}s"
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=commit, phase=phase, status="error",
            model=model, duration_s=time.time() - start_time,
            error=str(e)
        )


def save_result(result: BenchmarkResult, commit_pair: dict):
    """Save benchmark result to file."""
    human_short = commit_pair["human_commit"][:8]
    output_dir = RESULTS_DIR / human_short
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / f"{result.phase}_result.json"

    data = asdict(result)
    data["human_commit"] = commit_pair["human_commit"]
    data["parent_commit"] = commit_pair["parent_commit"]
    data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Don't save raw_output to keep file size manageable
    if "raw_output" in data:
        data["raw_output_length"] = len(data["raw_output"]) if data["raw_output"] else 0
        del data["raw_output"]

    with open(result_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Saved: {result_file}")


def get_docker_image_tag(commit: str, image_suffix: str, use_short_hash: bool = False) -> str:
    """Get Docker image tag for a commit.

    Args:
        commit: Full 40-char commit hash
        image_suffix: Optional suffix like "-vllm-style"
        use_short_hash: If True, use 8-char short hash; if False, use full hash

    For vLLM-style builds: use short hash (8 chars) + optional suffix
    For original builds: use full commit hash
    """
    if use_short_hash or image_suffix:
        # vLLM-style: short hash + optional suffix (e.g., "1acca3a2-vllm-style" or "021f76e4")
        return f"{commit[:8]}{image_suffix}"
    else:
        # Original: full commit hash
        return commit


def run_3way_benchmark(commit_pair: dict, hf_token: str, include_agent: bool = True) -> Dict[str, BenchmarkResult]:
    """Run full 3-way benchmark for a commit pair."""
    human_commit = commit_pair["human_commit"]
    parent_commit = commit_pair["parent_commit"]
    model = commit_pair["model"]
    perf_command = commit_pair["perf_command"]
    image_suffix = commit_pair.get("image_suffix", "")
    use_short_hash = commit_pair.get("use_short_hash", False)
    use_flashinfer = commit_pair.get("use_flashinfer", False)
    lora_adapter = commit_pair.get("lora_adapter", None)
    human_short = human_commit[:8]
    parent_short = parent_commit[:8]

    print(f"\n{'#'*60}")
    print(f"# 3-Way Benchmark: {human_short}")
    print(f"# Model: {model}")
    print(f"# Human commit: {human_short}")
    print(f"# Parent commit: {parent_short}")
    print(f"# Image suffix: {image_suffix or '(none)'}")
    print(f"# Use short hash: {use_short_hash}")
    print(f"{'#'*60}")

    results = {}

    # Phase 1: BASELINE (parent commit)
    print(f"\n[1/3] BASELINE phase (parent: {parent_short})")
    baseline_tag = get_docker_image_tag(parent_commit, image_suffix, use_short_hash)
    baseline_image = f"{DOCKER_REPO}:{baseline_tag}"
    print(f"  Image: {baseline_image}")
    results["baseline"] = run_docker_benchmark(
        docker_image=baseline_image,
        phase="baseline",
        commit=parent_commit,
        model=model,
        perf_command=perf_command,
        hf_token=hf_token,
        use_flashinfer=use_flashinfer,
        lora_adapter=lora_adapter,
    )
    save_result(results["baseline"], commit_pair)
    print_result_summary(results["baseline"])

    # Phase 2: HUMAN (human commit)
    print(f"\n[2/3] HUMAN phase (human: {human_short})")
    human_tag = get_docker_image_tag(human_commit, image_suffix, use_short_hash)
    human_image = f"{DOCKER_REPO}:{human_tag}"
    print(f"  Image: {human_image}")
    results["human"] = run_docker_benchmark(
        docker_image=human_image,
        phase="human",
        commit=human_commit,
        model=model,
        perf_command=perf_command,
        hf_token=hf_token,
        use_flashinfer=use_flashinfer,
        lora_adapter=lora_adapter,
    )
    save_result(results["human"], commit_pair)
    print_result_summary(results["human"])

    # Phase 3: AGENT (parent commit + agent patch)
    if include_agent:
        print(f"\n[3/3] AGENT phase (parent + patch)")
        agent_patch_path = find_agent_patch(commit_pair["agent_patch_dir"])

        if agent_patch_path:
            agent_patch_content = agent_patch_path.read_text()
            print(f"  Agent patch: {agent_patch_path}")

            agent_tag = get_docker_image_tag(parent_commit, image_suffix, use_short_hash)
            agent_image = f"{DOCKER_REPO}:{agent_tag}"
            print(f"  Image: {agent_image}")
            results["agent"] = run_docker_benchmark(
                docker_image=agent_image,
                phase="agent",
                commit=f"{human_short}_agent",
                model=model,
                perf_command=perf_command,
                hf_token=hf_token,
                agent_patch=agent_patch_content,
                parent_commit=parent_commit,
                use_flashinfer=use_flashinfer,
                lora_adapter=lora_adapter,
            )
            save_result(results["agent"], commit_pair)
            print_result_summary(results["agent"])
        else:
            print(f"  WARNING: Agent patch not found for {commit_pair['agent_patch_dir']}")
            results["agent"] = BenchmarkResult(
                commit_hash=f"{human_short}_agent",
                phase="agent",
                status="skipped",
                model=model,
                duration_s=0,
                error="Agent patch not found"
            )

    return results


def print_result_summary(result: BenchmarkResult):
    """Print a summary of the benchmark result."""
    if result.status == "success":
        print(f"  Status: SUCCESS ({result.duration_s:.1f}s)")
        if result.request_throughput:
            print(f"  Request throughput: {result.request_throughput:.2f} req/s")
        if result.output_throughput:
            print(f"  Output throughput: {result.output_throughput:.2f} tok/s")
        if result.ttft_mean:
            print(f"  TTFT mean: {result.ttft_mean:.2f} ms")
        if result.tpot_mean:
            print(f"  TPOT mean: {result.tpot_mean:.2f} ms")
    else:
        print(f"  Status: {result.status.upper()}")
        if result.error:
            print(f"  Error: {result.error}")


def print_3way_comparison(results: Dict[str, BenchmarkResult]):
    """Print comparison table for 3-way benchmark."""
    print(f"\n{'='*60}")
    print("3-WAY COMPARISON")
    print(f"{'='*60}")

    metrics_to_compare = [
        ("request_throughput", "Request throughput (req/s)", True),  # higher is better
        ("output_throughput", "Output throughput (tok/s)", True),
        ("ttft_mean", "TTFT mean (ms)", False),  # lower is better
        ("tpot_mean", "TPOT mean (ms)", False),
    ]

    print(f"\n{'Metric':<30} {'Baseline':>12} {'Human':>12} {'Agent':>12} {'Human vs Base':>15} {'Agent vs Base':>15}")
    print("-" * 100)

    for metric_key, metric_name, higher_is_better in metrics_to_compare:
        baseline_val = getattr(results.get("baseline"), metric_key, None)
        human_val = getattr(results.get("human"), metric_key, None)
        agent_val = getattr(results.get("agent"), metric_key, None) if "agent" in results else None

        baseline_str = f"{baseline_val:.2f}" if baseline_val else "N/A"
        human_str = f"{human_val:.2f}" if human_val else "N/A"
        agent_str = f"{agent_val:.2f}" if agent_val else "N/A"

        # Calculate improvements
        human_diff = ""
        agent_diff = ""
        if baseline_val and human_val:
            pct = ((human_val - baseline_val) / baseline_val) * 100
            sign = "+" if pct > 0 else ""
            better = (pct > 0) == higher_is_better
            human_diff = f"{sign}{pct:.1f}% {'*' if better else ''}"

        if baseline_val and agent_val:
            pct = ((agent_val - baseline_val) / baseline_val) * 100
            sign = "+" if pct > 0 else ""
            better = (pct > 0) == higher_is_better
            agent_diff = f"{sign}{pct:.1f}% {'*' if better else ''}"

        print(f"{metric_name:<30} {baseline_str:>12} {human_str:>12} {agent_str:>12} {human_diff:>15} {agent_diff:>15}")

    print("\n* indicates improvement over baseline")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Run local Docker 3-way benchmarks for SGLang")
    parser.add_argument("--commit", type=str, help="Run only for specific commit (short hash)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run")
    parser.add_argument("--no-agent", action="store_true", help="Skip agent phase")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per phase in seconds (default: 1800)")
    args = parser.parse_args()

    # Filter commits if specified
    commits_to_run = COMMIT_PAIRS
    if args.commit:
        commits_to_run = [
            c for c in COMMIT_PAIRS
            if c["human_commit"].startswith(args.commit) or c["parent_commit"].startswith(args.commit)
        ]
        if not commits_to_run:
            print(f"ERROR: No commits found matching '{args.commit}'")
            sys.exit(1)

    print(f"SGLang Local Docker 3-Way Benchmark")
    print(f"====================================")
    print(f"Commits to benchmark: {len(commits_to_run)}")
    print(f"Include agent phase: {not args.no_agent}")
    print(f"Timeout per phase: {args.timeout}s")
    print()

    # Dry run
    if args.dry_run:
        print("DRY RUN - Would run:")
        for i, cp in enumerate(commits_to_run, 1):
            image_suffix = cp.get("image_suffix", "")
            use_short_hash = cp.get("use_short_hash", False)
            human_tag = get_docker_image_tag(cp["human_commit"], image_suffix, use_short_hash)
            parent_tag = get_docker_image_tag(cp["parent_commit"], image_suffix, use_short_hash)

            print(f"\n[{i}/{len(commits_to_run)}] {cp['human_commit'][:8]}")
            print(f"  Human image: {DOCKER_REPO}:{human_tag}")
            print(f"  Parent image: {DOCKER_REPO}:{parent_tag}")
            print(f"  Model: {cp['model']}")
            print(f"  Image suffix: {image_suffix or '(none)'}")
            print(f"  Use short hash: {use_short_hash}")
            print(f"  Command: {cp['perf_command'][:80]}...")

            patch = find_agent_patch(cp["agent_patch_dir"])
            if patch:
                print(f"  Agent patch: {patch}")
            else:
                print(f"  Agent patch: NOT FOUND ({cp['agent_patch_dir']})")
        return

    # Get HF token
    hf_token = get_hf_token()
    if not hf_token:
        print("WARNING: No HuggingFace token found. Gated models may fail.")
    else:
        print(f"HuggingFace token: {'*' * 8}...{hf_token[-4:]}")

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    all_results = {}
    for i, commit_pair in enumerate(commits_to_run, 1):
        human_short = commit_pair["human_commit"][:8]
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(commits_to_run)}] Processing {human_short}")
        print(f"{'#'*60}")

        results = run_3way_benchmark(
            commit_pair=commit_pair,
            hf_token=hf_token,
            include_agent=not args.no_agent,
        )
        all_results[human_short] = results

        # Print comparison
        print_3way_comparison(results)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for human_short, results in all_results.items():
        print(f"\n{human_short}:")
        for phase, result in results.items():
            status_icon = "OK" if result.status == "success" else "FAIL"
            print(f"  {phase}: [{status_icon}] {result.duration_s:.1f}s", end="")
            if result.output_throughput:
                print(f" - {result.output_throughput:.1f} tok/s", end="")
            print()

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
