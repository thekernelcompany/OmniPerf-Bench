#!/usr/bin/env python3
"""
Smoke test for vLLM serving benchmark port binding issue.
Tests different workarounds for the port 29001 error.
"""
import modal
import time

app = modal.App("smoke-test-serving")

# Use the same image as our benchmark
base_image = modal.Image.from_registry(
    "ayushnangia16/nvidia-vllm-docker:b2e0ad3b598e",  # The failing commit
    add_python="3.12"
).pip_install(
    "huggingface_hub>=0.23.0",
).env({
    "HF_HOME": "/root/.cache/huggingface",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
})

model_cache = modal.Volume.from_name("model-cache", create_if_missing=True)

@app.function(
    image=base_image,
    gpu="H100",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def test_serving_approaches():
    """Test different approaches to start vLLM server."""
    import subprocess
    import os
    import time
    import socket

    results = []
    model = "meta-llama/Llama-3.1-8B-Instruct"

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
        subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "uvicorn"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "ray::"], capture_output=True)
        for port in [8000, 29000, 29001, 29002, 29500, 30000]:
            subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], capture_output=True)
        time.sleep(5)

    def try_start_server(approach_name, env_vars, extra_args, port):
        """Try to start server with given configuration."""
        print(f"\n{'='*60}")
        print(f"Testing: {approach_name}")
        print(f"Port: {port}")
        print(f"Extra env vars: {env_vars}")
        print(f"Extra args: {extra_args}")
        print(f"{'='*60}")

        kill_all_servers()

        # Check port is free
        if not check_port(port):
            print(f"WARNING: Port {port} not available before start!")
        else:
            print(f"Port {port} is available")

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--max-model-len", "2048",
        ]
        cmd.extend(extra_args)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.update(env_vars)

        print(f"Command: {' '.join(cmd)}")
        print(f"Environment additions: {env_vars}")

        server = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )

        # Wait for server to start (or fail)
        start_time = time.time()
        output_lines = []
        success = False

        while time.time() - start_time < 300:  # 5 min timeout
            line = server.stdout.readline()
            if line:
                line_str = line.decode('utf-8', errors='replace').strip()
                output_lines.append(line_str)
                print(f"  {line_str}")

                if "Application startup complete" in line_str:
                    # Wait a bit more to see if port error follows
                    time.sleep(3)
                    continue

                if "error while attempting to bind" in line_str.lower():
                    print(f"FAIL: Port binding error detected!")
                    success = False
                    break

                if "Uvicorn running on" in line_str or "Started server" in line_str:
                    # Try to hit health endpoint
                    time.sleep(2)
                    try:
                        import urllib.request
                        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as r:
                            if r.status == 200:
                                print(f"SUCCESS: Server health check passed!")
                                success = True
                                break
                    except Exception as e:
                        print(f"Health check failed: {e}")

            if server.poll() is not None:
                print(f"Server exited with code: {server.returncode}")
                break

        # Cleanup
        try:
            server.terminate()
            server.wait(timeout=5)
        except:
            server.kill()

        kill_all_servers()

        return {
            "approach": approach_name,
            "success": success,
            "port": port,
            "output_tail": "\n".join(output_lines[-20:])
        }

    # Test 1: Default approach (what we currently do)
    results.append(try_start_server(
        "Default (spawn method, port 29001)",
        {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        ["--disable-frontend-multiprocessing"],
        29001
    ))

    # Test 2: Standard port 8000
    results.append(try_start_server(
        "Standard port 8000",
        {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        ["--disable-frontend-multiprocessing"],
        8000
    ))

    # Test 3: With MASTER_PORT set
    results.append(try_start_server(
        "With explicit MASTER_PORT=35000",
        {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "MASTER_PORT": "35000",
        },
        ["--disable-frontend-multiprocessing"],
        8000
    ))

    # Test 4: Without --disable-frontend-multiprocessing
    results.append(try_start_server(
        "Without disable-frontend-multiprocessing flag",
        {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        [],
        8000
    ))

    # Test 5: Using Ray backend
    results.append(try_start_server(
        "With Ray distributed backend",
        {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        ["--distributed-executor-backend", "ray"],
        8000
    ))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        status = "✅ SUCCESS" if r["success"] else "❌ FAILED"
        print(f"{status}: {r['approach']}")

    return results


@app.local_entrypoint()
def main():
    results = test_serving_approaches.remote()
    print("\n\nFinal Results:")
    for r in results:
        print(f"  {r['approach']}: {'SUCCESS' if r['success'] else 'FAILED'}")
