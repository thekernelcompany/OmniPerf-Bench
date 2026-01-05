#!/usr/bin/env python3
"""
Smoke test for vLLM serving benchmark port binding issue.
Uses wheel-based installation matching our actual benchmark setup.
"""
import modal
import time

app = modal.App("smoke-test-wheel")

# Base image matching our benchmark setup
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["git", "curl", "wget", "lsof", "psmisc"])
    .pip_install([
        "uv",
        "torch==2.4.0",
        "transformers>=4.40.0",
        "huggingface_hub>=0.23.0",
        "tokenizers>=0.19.0",
        "accelerate>=0.30.0",
        "datasets",
        "aiohttp",
        "openai",
        "ray",
    ])
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTHONUNBUFFERED": "1",
    })
)

model_cache = modal.Volume.from_name("model-cache", create_if_missing=True)

BASELINE_WHEEL = "https://vllm-wheels.s3.us-west-2.amazonaws.com/4a18fd14ba4a349291c798a16bf62fa8a9af0b6b/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

@app.function(
    image=base_image,
    gpu="H100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def test_serving_approaches():
    """Test different approaches to start vLLM server."""
    import subprocess
    import os
    import time
    import socket
    import signal

    results = []
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Install vLLM from wheel
    print("Installing vLLM from wheel...")
    result = subprocess.run(
        ["uv", "pip", "install", BASELINE_WHEEL, "--no-deps", "-q"],
        capture_output=True, text=True, env={**os.environ, "UV_SKIP_WHEEL_FILENAME_CHECK": "1"}
    )
    if result.returncode != 0:
        print(f"Wheel install failed: {result.stderr}")
        # Try with pip
        subprocess.run(["pip", "install", BASELINE_WHEEL, "--no-deps", "-q"], check=True)

    # Install vLLM dependencies
    subprocess.run(["pip", "install", "vllm", "--no-deps", "-q"], capture_output=True)

    # Check vLLM version
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
    except Exception as e:
        print(f"Could not get vLLM version: {e}")

    def check_port(port):
        """Check if port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return True
            except OSError:
                return False

    def kill_all_servers():
        """Kill any running vLLM servers."""
        for cmd in [
            ["pkill", "-9", "-f", "vllm.entrypoints"],
            ["pkill", "-9", "-f", "uvicorn"],
            ["pkill", "-9", "-f", "ray::"],
            ["pkill", "-9", "-f", "multiprocessing"],
        ]:
            subprocess.run(cmd, capture_output=True)

        for port in [8000, 29000, 29001, 29002, 29500, 30000, 30010, 35000]:
            subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], capture_output=True)
        time.sleep(5)

    def try_start_server(approach_name, env_vars, extra_args, port, timeout=300):
        """Try to start server with given configuration."""
        print(f"\n{'='*60}")
        print(f"Testing: {approach_name}")
        print(f"Port: {port}")
        print(f"Extra env vars: {env_vars}")
        print(f"Extra args: {extra_args}")
        print(f"{'='*60}")

        kill_all_servers()
        time.sleep(2)

        # Check port is free
        port_free = check_port(port)
        print(f"Port {port} available: {port_free}")

        # Also check NCCL ports
        for p in [29500, 30000, 35000]:
            print(f"Port {p} available: {check_port(p)}")

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.85",
        ]
        cmd.extend(extra_args)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.update(env_vars)

        print(f"Command: {' '.join(cmd)}")
        print(f"Env: {env_vars}")

        server = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )

        # Wait for server to start (or fail)
        start_time = time.time()
        output_lines = []
        success = False
        error_msg = None

        while time.time() - start_time < timeout:
            line = server.stdout.readline()
            if line:
                line_str = line.decode('utf-8', errors='replace').strip()
                output_lines.append(line_str)

                # Only print key lines
                if any(x in line_str for x in ["error", "Error", "ERROR", "Route:", "Started server", "Application startup", "vLLM version"]):
                    print(f"  {line_str}")

                if "error while attempting to bind" in line_str.lower():
                    error_msg = "PORT_BINDING_ERROR"
                    break

                if "Application startup complete" in line_str:
                    # Wait to see if port error follows
                    time.sleep(5)
                    # Try health check
                    try:
                        import urllib.request
                        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=10) as r:
                            if r.status == 200:
                                print(f"SUCCESS: Health check passed!")
                                success = True
                                break
                    except Exception as e:
                        print(f"Health check failed: {e}")
                        # Check if server is still running
                        if server.poll() is not None:
                            error_msg = "SERVER_EXITED"
                            break
                        continue

            if server.poll() is not None:
                print(f"Server exited with code: {server.returncode}")
                error_msg = f"EXIT_CODE_{server.returncode}"
                break

        # Cleanup
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            server.wait(timeout=5)
        except:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except:
                pass

        kill_all_servers()

        return {
            "approach": approach_name,
            "success": success,
            "error": error_msg,
            "port": port,
            "output_tail": "\n".join(output_lines[-30:])
        }

    # Test approaches based on GitHub issue findings

    # Test 1: Default (what fails)
    results.append(try_start_server(
        "Default: spawn + disable-frontend-mp + port 29001",
        {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        ["--disable-frontend-multiprocessing"],
        29001
    ))

    # Test 2: Different port (8000)
    results.append(try_start_server(
        "Port 8000 instead of 29001",
        {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        ["--disable-frontend-multiprocessing"],
        8000
    ))

    # Test 3: With explicit MASTER_PORT far from HTTP port
    results.append(try_start_server(
        "Port 8000 + MASTER_PORT=35000",
        {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "MASTER_PORT": "35000",
            "MASTER_ADDR": "127.0.0.1",
        },
        ["--disable-frontend-multiprocessing"],
        8000
    ))

    # Test 4: Without --disable-frontend-multiprocessing
    results.append(try_start_server(
        "No disable-frontend-mp flag",
        {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        [],  # No extra args
        8000
    ))

    # Test 5: With Ray distributed backend (from GitHub workaround)
    results.append(try_start_server(
        "Ray distributed backend (GitHub workaround)",
        {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        ["--distributed-executor-backend", "ray"],
        8000
    ))

    # Test 6: fork method instead of spawn
    results.append(try_start_server(
        "Fork method instead of spawn",
        {"VLLM_WORKER_MULTIPROC_METHOD": "fork"},
        ["--disable-frontend-multiprocessing"],
        8000
    ))

    # Test 7: No VLLM_WORKER_MULTIPROC_METHOD at all
    results.append(try_start_server(
        "No VLLM_WORKER_MULTIPROC_METHOD (default)",
        {},
        ["--disable-frontend-multiprocessing"],
        8000
    ))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        status = "✅ SUCCESS" if r["success"] else f"❌ FAILED ({r.get('error', 'unknown')})"
        print(f"{status}: {r['approach']}")

    return results


@app.local_entrypoint()
def main():
    print("Starting smoke test...")
    results = test_serving_approaches.remote()

    print("\n\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    successes = []
    failures = []
    for r in results:
        if r['success']:
            successes.append(r['approach'])
            print(f"✅ {r['approach']}")
        else:
            failures.append(f"{r['approach']} ({r.get('error', 'unknown')})")
            print(f"❌ {r['approach']} - {r.get('error', 'unknown')}")

    print(f"\n{len(successes)} succeeded, {len(failures)} failed")

    if successes:
        print(f"\n>>> RECOMMENDED APPROACH: {successes[0]}")
