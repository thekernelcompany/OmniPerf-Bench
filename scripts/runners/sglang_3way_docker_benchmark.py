#!/usr/bin/env python3
"""
SGLang True 3-Way Benchmark Runner

Runs actual 3-way benchmarks (baseline vs human vs agent) for SGLang commits
using SEPARATE Docker images for baseline and human commits.

This solves the ABI compatibility issue by using pre-built images for each commit.

Docker Repository: shikhar481/sglang-images
- Baseline: shikhar481/sglang-images:{parent_commit_hash}
- Human: shikhar481/sglang-images:{human_commit_hash}
- Agent: Baseline image + agent patch overlay

Usage:
    # Dry run to see what would be benchmarked
    python sglang_3way_docker_benchmark.py --dry-run

    # Run all candidates
    python sglang_3way_docker_benchmark.py

    # Run specific commit
    python sglang_3way_docker_benchmark.py --commit 021f76e4

    # Run with parallel workers
    python sglang_3way_docker_benchmark.py --parallel 3
"""

import os
import sys
import json
import time
import logging
import argparse
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
DOCKER_REPO = "shikhar481/sglang-images"
RESULTS_DIR = Path("omniperf_results_3way_sglang_docker")
CANDIDATES_FILE = Path("/tmp/sglang_3way_candidates.json")

# GPU configurations
GPU_CONFIGS = {
    "H100:1": {"gpu": "H100", "count": 1, "timeout": 3600},
    "H100:2": {"gpu": "H100", "count": 2, "timeout": 5400},
    "H100:4": {"gpu": "H100", "count": 4, "timeout": 7200},
    "H100:8": {"gpu": "H100", "count": 8, "timeout": 14400},
}

# Large models that need multi-GPU
LARGE_MODEL_GPU_MAP = {
    "deepseek-v3": "H100:8",
    "deepseek-v2": "H100:4",
    "deepseek-r1": "H100:8",
    "llama-4": "H100:2",
    "llama-3-70b": "H100:4",
    "llama-3.1-70b": "H100:4",
    "mixtral": "H100:2",
    "qwen2-72b": "H100:4",
}


def check_docker_image_exists(tag: str) -> bool:
    """Check if Docker image exists on Docker Hub."""
    url = f"https://hub.docker.com/v2/repositories/{DOCKER_REPO}/tags/{tag}"
    try:
        req = urllib.request.Request(url, method='HEAD')
        urllib.request.urlopen(req, timeout=10)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        return False
    except Exception:
        return False


def get_gpu_config(model: str, perf_command: str) -> str:
    """Determine GPU config based on model and TP size."""
    import re

    # Check for TP size in command
    tp_patterns = [r'--tp[=\s]+(\d+)', r'--tp-size[=\s]+(\d+)']
    for pattern in tp_patterns:
        match = re.search(pattern, perf_command)
        if match:
            tp = int(match.group(1))
            if tp >= 8: return "H100:8"
            elif tp >= 4: return "H100:4"
            elif tp >= 2: return "H100:2"

    # Check for known large models
    model_lower = model.lower() if model else ""
    for pattern, config in LARGE_MODEL_GPU_MAP.items():
        if pattern in model_lower:
            return config

    return "H100:1"


def load_candidates() -> List[Dict]:
    """Load benchmark candidates from JSON file."""
    if not CANDIDATES_FILE.exists():
        logger.error(f"Candidates file not found: {CANDIDATES_FILE}")
        logger.info("Run the discovery script first to generate candidates")
        return []

    with open(CANDIDATES_FILE) as f:
        return json.load(f)


def create_3way_benchmark_script(
    baseline_commit: str,
    human_commit: str,
    agent_patch: str,
    perf_command: str,
    model: str,
) -> str:
    """
    Create the benchmark script that runs inside Modal sandbox.

    This script will be executed in BOTH baseline and human containers.
    The agent phase uses baseline container with patch overlay.
    """
    # Escape the agent patch for embedding in Python string
    escaped_patch = agent_patch.replace('\\', '\\\\').replace("'''", "\\'\\'\\'")

    script = f'''#!/usr/bin/env python3
"""
3-Way SGLang Benchmark Script
Runs inside Modal sandbox with pre-built Docker image.
"""

import subprocess
import sys
import os
import json
import time
import re
import signal
import tempfile

# Configuration
BASELINE_COMMIT = "{baseline_commit}"
HUMAN_COMMIT = "{human_commit}"
PERF_COMMAND = """{perf_command}"""
MODEL = "{model}"

AGENT_PATCH = \'\'\'
{escaped_patch}
\'\'\'

PORT = 30000
SERVER_TIMEOUT = 600
BENCHMARK_TIMEOUT = 900

results = {{
    "baseline_metrics": {{}},
    "human_metrics": {{}},
    "agent_metrics": {{}},
    "status": "error",
    "error": None,
    "baseline_raw": "",
    "human_raw": "",
    "agent_raw": "",
    "baseline_server_logs": "",
    "human_server_logs": "",
    "agent_server_logs": "",
}}


def parse_benchmark_output(output: str) -> dict:
    """Parse benchmark output to extract metrics."""
    metrics = {{}}

    patterns = {{
        "request_throughput": r"Request throughput.*?([\\d.]+)\\s*req/s",
        "output_token_throughput": r"Output token throughput.*?([\\d.]+)\\s*tok/s",
        "total_token_throughput": r"Total token throughput.*?([\\d.]+)\\s*tok/s",
        "mean_ttft": r"Mean TTFT.*?([\\d.]+)\\s*ms",
        "median_ttft": r"Median TTFT.*?([\\d.]+)\\s*ms",
        "mean_tpot": r"Mean TPOT.*?([\\d.]+)\\s*ms",
        "median_tpot": r"Median TPOT.*?([\\d.]+)\\s*ms",
        "mean_itl": r"Mean ITL.*?([\\d.]+)\\s*ms",
        "median_itl": r"Median ITL.*?([\\d.]+)\\s*ms",
    }}

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except:
                pass

    return metrics


def kill_port(port: int):
    """Kill any process using the given port."""
    try:
        subprocess.run(["fuser", "-k", f"{{port}}/tcp"], capture_output=True, timeout=10)
    except:
        pass
    time.sleep(1)


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Wait for server to be ready."""
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            url = f"http://127.0.0.1:{{port}}/health"
            req = urllib.request.Request(url, method='GET')
            urllib.request.urlopen(req, timeout=5)
            return True
        except:
            pass
        time.sleep(2)
    return False


def start_server(model: str, port: int) -> subprocess.Popen:
    """Start SGLang server."""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--host", "0.0.0.0",
    ]

    # Add extra args if model needs them
    if "deepseek" in model.lower() or "70b" in model.lower():
        cmd.extend(["--tp", "4"])

    print(f"Starting server: {{' '.join(cmd)}}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    return proc


def run_benchmark(perf_command: str, port: int, timeout: int = 600) -> tuple:
    """Run benchmark command and return (output, success)."""
    # Add port and host to command if not present
    cmd = perf_command
    if "--port" not in cmd:
        cmd += f" --port {{port}}"
    if "--host" not in cmd:
        cmd += " --host 127.0.0.1"

    print(f"Running benchmark: {{cmd}}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        return output, result.returncode == 0
    except subprocess.TimeoutExpired:
        return f"Benchmark timed out after {{timeout}}s", False
    except Exception as e:
        return str(e), False


def apply_patch(patch_content: str, sglang_path: str) -> bool:
    """Apply agent patch to SGLang installation."""
    if not patch_content.strip():
        print("No patch to apply")
        return True

    try:
        # Write patch to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(patch_content)
            patch_file = f.name

        # Apply patch
        result = subprocess.run(
            ["patch", "-p1", "-d", sglang_path, "-i", patch_file],
            capture_output=True,
            text=True,
            timeout=60,
        )

        os.unlink(patch_file)

        if result.returncode != 0:
            print(f"Patch failed: {{result.stderr}}")
            return False

        print("Patch applied successfully")
        return True
    except Exception as e:
        print(f"Patch error: {{e}}")
        return False


def find_sglang_path() -> str:
    """Find SGLang installation path."""
    import sglang
    sglang_file = sglang.__file__
    # Go up to find the sglang package root
    return str(Path(sglang_file).parent.parent)


def run_phase(phase: str, model: str, perf_command: str, port: int) -> dict:
    """Run a single benchmark phase (baseline/human/agent)."""
    print(f"\\n{'='*60}")
    print(f"PHASE: {{phase.upper()}}")
    print(f"{'='*60}")

    phase_result = {{
        "metrics": {{}},
        "raw_output": "",
        "server_logs": "",
        "success": False,
        "error": None,
    }}

    # Kill any existing server
    kill_port(port)

    # Start server
    server_proc = start_server(model, port)

    # Wait for server
    print(f"Waiting for server on port {{port}}...")
    if not wait_for_server(port, timeout=SERVER_TIMEOUT):
        # Collect server logs
        try:
            server_proc.terminate()
            server_proc.wait(timeout=10)
        except:
            server_proc.kill()

        logs = ""
        try:
            logs = server_proc.stdout.read() if server_proc.stdout else ""
        except:
            pass

        phase_result["error"] = "Server failed to start"
        phase_result["server_logs"] = logs[-5000:]
        return phase_result

    print(f"Server ready on port {{port}}")

    # Run benchmark
    output, success = run_benchmark(perf_command, port, timeout=BENCHMARK_TIMEOUT)
    phase_result["raw_output"] = output[-10000:]

    # Parse metrics
    if success:
        metrics = parse_benchmark_output(output)
        if metrics:
            phase_result["metrics"] = metrics
            phase_result["success"] = True
            print(f"Metrics: {{json.dumps(metrics, indent=2)}}")
        else:
            phase_result["error"] = "No metrics parsed from output"
    else:
        phase_result["error"] = f"Benchmark failed: {{output[-500:]}}"

    # Cleanup
    try:
        os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        server_proc.wait(timeout=10)
    except:
        try:
            server_proc.kill()
        except:
            pass

    kill_port(port)

    return phase_result


def main():
    print("=" * 60)
    print("SGLANG 3-WAY BENCHMARK")
    print("=" * 60)
    print(f"Baseline commit: {{BASELINE_COMMIT[:8]}}")
    print(f"Human commit: {{HUMAN_COMMIT[:8]}}")
    print(f"Model: {{MODEL}}")
    print(f"Command: {{PERF_COMMAND[:80]}}...")

    # This script runs in different containers for different phases
    # We detect which phase based on environment variable
    phase = os.environ.get("BENCHMARK_PHASE", "human")

    if phase == "baseline":
        result = run_phase("baseline", MODEL, PERF_COMMAND, PORT)
        results["baseline_metrics"] = result["metrics"]
        results["baseline_raw"] = result["raw_output"]
        results["baseline_server_logs"] = result.get("server_logs", "")

    elif phase == "human":
        result = run_phase("human", MODEL, PERF_COMMAND, PORT)
        results["human_metrics"] = result["metrics"]
        results["human_raw"] = result["raw_output"]
        results["human_server_logs"] = result.get("server_logs", "")

    elif phase == "agent":
        # Apply patch first
        sglang_path = find_sglang_path()
        print(f"SGLang path: {{sglang_path}}")

        if not apply_patch(AGENT_PATCH, sglang_path):
            results["error"] = "Failed to apply agent patch"
            print(f"\\n===BENCHMARK_RESULTS===")
            print(json.dumps(results))
            return

        result = run_phase("agent", MODEL, PERF_COMMAND, PORT)
        results["agent_metrics"] = result["metrics"]
        results["agent_raw"] = result["raw_output"]
        results["agent_server_logs"] = result.get("server_logs", "")

    if result["success"]:
        results["status"] = "success"
    else:
        results["error"] = result.get("error", "Unknown error")

    print(f"\\n===BENCHMARK_RESULTS===")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
'''
    return script


def run_3way_benchmark_modal(
    candidate: Dict,
    gpu_config: str = "H100:1",
    timeout: int = 3600,
) -> Dict[str, Any]:
    """
    Run true 3-way benchmark using Modal sandboxes with separate Docker images.

    Returns combined results from all three phases.
    """
    import modal

    human_short = candidate["human"]
    parent_short = candidate["parent"]
    model = candidate["model"]
    perf_command = candidate["perf_command"]

    # Read agent patch
    patch_path = candidate.get("patch_path")
    agent_patch = ""
    if patch_path and Path(patch_path).exists():
        agent_patch = Path(patch_path).read_text()

    result = {
        "commit": human_short,
        "human_commit": candidate["human_full"],
        "parent_commit": candidate["parent_full"],
        "model": model,
        "perf_command": perf_command,
        "gpu_config": gpu_config,
        "status": "error",
        "error": None,
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": {},
        "human_improvement": {},
        "agent_improvement": {},
        "agent_vs_human": {},
        "duration_s": 0,
        "benchmark_mode": "3way_docker",
    }

    start_time = time.time()

    # Get GPU config
    gpu_cfg = GPU_CONFIGS.get(gpu_config, GPU_CONFIGS["H100:1"])

    # Create benchmark script
    benchmark_script = create_3way_benchmark_script(
        baseline_commit=parent_short,
        human_commit=human_short,
        agent_patch=agent_patch,
        perf_command=perf_command,
        model=model,
    )

    # Model cache volume
    model_cache = modal.Volume.from_name("sglang-model-cache", create_if_missing=True)

    # Get app reference
    sandbox_app = modal.App.lookup("sglang-3way-benchmark", create_if_missing=True)

    def run_phase_in_sandbox(phase: str, docker_tag: str) -> Dict:
        """Run a single phase in Modal sandbox."""
        docker_image = f"{DOCKER_REPO}:{docker_tag}"
        print(f"\n[{phase.upper()}] Using image: {docker_image}")

        image = (
            modal.Image.from_registry(docker_image)
            .run_commands([
                "apt-get update && apt-get install -y git curl patch psmisc net-tools lsof || true",
                "pip install 'huggingface-hub>=0.35.0,<1.0' --force-reinstall || true",
                "pip install --no-deps datasets pandas tqdm aiohttp requests || true",
            ])
            .env({
                "HF_HOME": "/root/.cache/huggingface",
                "BENCHMARK_PHASE": phase,
            })
        )

        try:
            sandbox = modal.Sandbox.create(
                app=sandbox_app,
                image=image,
                gpu=f"{gpu_cfg['gpu']}:{gpu_cfg['count']}" if gpu_cfg['count'] > 1 else gpu_cfg['gpu'],
                timeout=timeout,
                volumes={"/root/.cache/huggingface": model_cache},
                secrets=[modal.Secret.from_name("huggingface-secret")],
            )

            # Write and run benchmark script
            script_path = "/tmp/benchmark.py"
            f = sandbox.open(script_path, "w")
            f.write(benchmark_script)
            f.close()

            proc = sandbox.exec("python3", "-u", script_path)
            stdout_lines = list(proc.stdout)
            proc.wait()

            sandbox.terminate()

            # Parse results from output
            full_output = "\n".join(line.rstrip('\n') for line in stdout_lines)

            if "===BENCHMARK_RESULTS===" in full_output:
                results_start = full_output.index("===BENCHMARK_RESULTS===") + len("===BENCHMARK_RESULTS===")
                results_json = full_output[results_start:].strip().split("\n")[0]
                return json.loads(results_json)

            return {"error": "No results marker found", "raw": full_output[-2000:]}

        except Exception as e:
            return {"error": str(e)}

    # Run all three phases
    phases = [
        ("baseline", parent_short),
        ("human", human_short),
        ("agent", parent_short),  # Agent runs on baseline image with patch
    ]

    all_phase_results = {}
    for phase, docker_tag in phases:
        print(f"\n{'='*60}")
        print(f"Running {phase.upper()} phase")
        print(f"{'='*60}")

        phase_result = run_phase_in_sandbox(phase, docker_tag)
        all_phase_results[phase] = phase_result

        if phase_result.get("error"):
            print(f"  [{phase}] Error: {phase_result['error']}")
        else:
            metrics = phase_result.get(f"{phase}_metrics", {})
            print(f"  [{phase}] Metrics: {metrics}")

    # Combine results
    result["baseline_metrics"] = all_phase_results.get("baseline", {}).get("baseline_metrics", {})
    result["human_metrics"] = all_phase_results.get("human", {}).get("human_metrics", {})
    result["agent_metrics"] = all_phase_results.get("agent", {}).get("agent_metrics", {})

    # Calculate improvements
    baseline = result["baseline_metrics"]
    human = result["human_metrics"]
    agent = result["agent_metrics"]

    if baseline and human:
        for key in baseline:
            if key in human and baseline[key] > 0:
                # For throughput metrics, higher is better
                if "throughput" in key.lower():
                    pct = ((human[key] - baseline[key]) / baseline[key]) * 100
                # For latency metrics, lower is better
                elif any(x in key.lower() for x in ["ttft", "tpot", "itl", "latency"]):
                    pct = ((baseline[key] - human[key]) / baseline[key]) * 100
                else:
                    pct = ((human[key] - baseline[key]) / baseline[key]) * 100
                result["human_improvement"][key] = round(pct, 2)

    if baseline and agent:
        for key in baseline:
            if key in agent and baseline[key] > 0:
                if "throughput" in key.lower():
                    pct = ((agent[key] - baseline[key]) / baseline[key]) * 100
                elif any(x in key.lower() for x in ["ttft", "tpot", "itl", "latency"]):
                    pct = ((baseline[key] - agent[key]) / baseline[key]) * 100
                else:
                    pct = ((agent[key] - baseline[key]) / baseline[key]) * 100
                result["agent_improvement"][key] = round(pct, 2)

    if human and agent:
        for key in human:
            if key in agent and human[key] > 0:
                pct = ((agent[key] - human[key]) / human[key]) * 100
                result["agent_vs_human"][key] = round(pct, 2)

    # Determine overall status
    if result["baseline_metrics"] and result["human_metrics"]:
        result["status"] = "success"
    elif result["human_metrics"]:
        result["status"] = "partial"
        result["error"] = "Baseline failed"
    else:
        result["status"] = "error"
        errors = [f"{p}: {r.get('error', 'unknown')}" for p, r in all_phase_results.items() if r.get("error")]
        result["error"] = "; ".join(errors) if errors else "All phases failed"

    result["duration_s"] = time.time() - start_time

    return result


def save_result(result: Dict, results_dir: Path):
    """Save benchmark result to JSON file."""
    commit = result.get("commit", "unknown")
    commit_dir = results_dir / "sglang" / commit
    commit_dir.mkdir(parents=True, exist_ok=True)

    result_file = commit_dir / "benchmark_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved result to {result_file}")


def main():
    parser = argparse.ArgumentParser(description="Run SGLang 3-way Docker benchmarks")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--commit", type=str, help="Run specific commit only")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel benchmarks")
    parser.add_argument("--skip-existing", action="store_true", help="Skip commits with existing results")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per phase in seconds")
    args = parser.parse_args()

    # Setup
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load candidates
    candidates = load_candidates()
    if not candidates:
        logger.error("No candidates to run")
        return

    logger.info(f"Loaded {len(candidates)} candidates")

    # Filter by commit if specified
    if args.commit:
        candidates = [c for c in candidates if c["human"].startswith(args.commit)]
        if not candidates:
            logger.error(f"No candidate matching commit {args.commit}")
            return

    # Check for existing results
    if args.skip_existing:
        existing = set()
        for rf in RESULTS_DIR.glob("sglang/*/benchmark_result.json"):
            try:
                r = json.loads(rf.read_text())
                if r.get("status") == "success":
                    existing.add(rf.parent.name)
            except:
                pass
        candidates = [c for c in candidates if c["human"] not in existing]
        logger.info(f"After skipping existing: {len(candidates)} candidates")

    if args.dry_run:
        print("\n=== DRY RUN - Would run these 3-way benchmarks ===")
        print(f"Docker repo: {DOCKER_REPO}")
        print()
        for i, c in enumerate(candidates, 1):
            gpu_config = get_gpu_config(c["model"], c["perf_command"])
            bench_type = "server" if "bench_serving" in c["perf_command"].lower() else "direct"
            print(f"{i}. {c['human']} (parent: {c['parent']})")
            print(f"   Subject: {c['subject']}")
            print(f"   Model: {c['model']}")
            print(f"   GPU: {gpu_config} | Type: {bench_type}")
            print(f"   Images: baseline={DOCKER_REPO}:{c['parent']}, human={DOCKER_REPO}:{c['human']}")
            print()
        return

    # Run benchmarks
    success_count = 0
    error_count = 0

    for i, candidate in enumerate(candidates, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(candidates)}] Running 3-way benchmark for {candidate['human']}")
        logger.info(f"Subject: {candidate['subject']}")
        logger.info(f"{'='*60}")

        gpu_config = get_gpu_config(candidate["model"], candidate["perf_command"])

        try:
            result = run_3way_benchmark_modal(
                candidate=candidate,
                gpu_config=gpu_config,
                timeout=args.timeout,
            )

            save_result(result, RESULTS_DIR)

            if result["status"] == "success":
                success_count += 1
                logger.info(f"SUCCESS ({result['duration_s']:.1f}s)")
                if result.get("human_improvement"):
                    logger.info(f"  Human improvement: {result['human_improvement']}")
                if result.get("agent_improvement"):
                    logger.info(f"  Agent improvement: {result['agent_improvement']}")
            else:
                error_count += 1
                logger.error(f"FAILED: {result.get('error', 'Unknown error')}")

        except Exception as e:
            error_count += 1
            logger.error(f"Exception: {e}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK RUN COMPLETE")
    logger.info(f"  Total: {len(candidates)}")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Results dir: {RESULTS_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
