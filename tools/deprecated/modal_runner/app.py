
import modal
import os

# Use Dockerfile to create python symlink before Modal's auto-pip-install
_dockerfile_path = os.path.join(os.path.dirname(__file__), "Dockerfile.modal")
IMG = modal.Image.from_dockerfile(_dockerfile_path)
app = modal.App("vllm-agent-tests-v5")

GENS = modal.Volume.from_name("opb-generators")
RESULTS = modal.Volume.from_name("opb-results")
SRC = modal.Volume.from_name("opb-source")


def _run_and_capture(script_rel: str, args: list[str]) -> str:
    import subprocess
    import os
    import shutil
    from pathlib import Path
    import time
    
    start_time = time.time()
    print("="*80)
    print("[STEP 1/6] Starting vLLM installation process...")
    print("="*80)
    
    # Copy source from read-only volume to writable /tmp location
    source_in_volume = "/opb-source/vllm_core-0000"
    writable_source = "/tmp/vllm_source"
    
    print(f"\n[STEP 2/6] Copying source from {source_in_volume} to {writable_source}...")
    copy_start = time.time()
    if os.path.exists(writable_source):
        print("  - Removing existing directory...")
        shutil.rmtree(writable_source)
    print("  - Copying directory tree...")
    shutil.copytree(source_in_volume, writable_source, symlinks=True)
    print(f"  ✓ Copy completed in {time.time() - copy_start:.1f}s")
    
    # Uninstall baked vllm
    print("\n[STEP 3/6] Uninstalling base vLLM...")
    uninstall_start = time.time()
    try:
        subprocess.run(["python3", "-m", "pip", "uninstall", "-y", "vllm"], 
                      check=False, capture_output=True)
        print(f"  ✓ Uninstall completed in {time.time() - uninstall_start:.1f}s")
    except Exception as e:
        print(f"  ! Uninstall failed (non-critical): {e}")
    
    print("\n[STEP 4/6] Building vLLM wheel (this will take 20-40 minutes)...")
    print(f"  - Working directory: {writable_source}")
    print(f"  - Build started at {time.strftime('%H:%M:%S')}")
    os.chdir(writable_source)
    
    # Build wheel with live output
    build_start = time.time()
    build_env = {**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0+agent"}
    
    # Run build WITHOUT capturing output so we can see progress
    print("  - Running: python3 setup.py bdist_wheel")
    print("  - (This builds CUDA extensions, please wait...)")
    build_result = subprocess.run(
        ["python3", "setup.py", "bdist_wheel"],
        check=False, env=build_env
    )
    
    build_time = time.time() - build_start
    if build_result.returncode != 0:
        print(f"\n  ✗ ERROR: Wheel build failed after {build_time:.1f}s")
        raise RuntimeError(f"Failed to build vLLM wheel with return code {build_result.returncode}")
    
    print(f"  ✓ Wheel build completed in {build_time/60:.1f} minutes ({build_time:.1f}s)")
    
    # Find the built wheel
    print("\n[STEP 5/6] Locating built wheel...")
    wheel_files = list(Path(writable_source).glob("dist/*.whl"))
    if not wheel_files:
        raise RuntimeError("No wheel file found after build")
    wheel_path = str(wheel_files[0])
    print(f"  ✓ Found wheel: {wheel_path}")
    
    # Install the wheel
    print("\n[STEP 6/6] Installing vLLM wheel...")
    install_start = time.time()
    result = subprocess.run(
        ["python3", "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel_path], 
        check=False
    )
    if result.returncode != 0:
        print("  ✗ ERROR: Wheel install failed")
        raise RuntimeError("Failed to install vLLM wheel")
    
    print(f"  ✓ Install completed in {time.time() - install_start:.1f}s")
    
    total_time = time.time() - start_time
    print("="*80)
    print(f"✓ vLLM installation complete! Total time: {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print("="*80)
    
    print("\n[STEP 7/7] Running test script...")
    print(f"  - Script: {script_rel}")
    print(f"  - Args: {args}")

    # Intercept optional JSON output path intended for Modal results volume
    json_out_path = None
    forwarded_args = list(args)
    if "--json-out" in forwarded_args:
        idx = forwarded_args.index("--json-out")
        if idx + 1 >= len(forwarded_args):
            raise ValueError("Missing path after --json-out")
        json_out_path = forwarded_args[idx + 1]
        # Strip the flag and its value before invoking the script
        del forwarded_args[idx:idx + 2]

    print(f"  - Forwarded Args: {forwarded_args}")

    script_path = os.path.join("/opb-generators", script_rel)
    cmd = ["python3", script_path, *forwarded_args]
    res = subprocess.run(cmd, check=False, text=True, capture_output=True)
    
    if res.returncode != 0:
        print("\n" + "="*80)
        print(f"ERROR: Test script failed with exit code {res.returncode}")
        print("Command: " + " ".join(cmd))
        print("="*80)
        print("\n--- STDOUT ---\n" + (res.stdout or ""))
        print("\n--- STDERR ---\n" + (res.stderr or ""))
        print("="*80 + "\n")
        raise subprocess.CalledProcessError(res.returncode, cmd, res.stdout, res.stderr)
    
    # If requested, persist stdout (prefer last JSON line if present) to results volume
    if json_out_path:
        from pathlib import Path
        import json as _json
        out_text = (res.stdout or "").strip()

        # Try to parse the last non-empty line as JSON; fall back to raw stdout
        try:
            non_empty_lines = [ln for ln in out_text.splitlines() if ln.strip()]
            if non_empty_lines:
                parsed = _json.loads(non_empty_lines[-1])
                out_text = _json.dumps(parsed)
        except Exception:
            pass

        parent_dir = os.path.dirname(json_out_path) or "."
        Path(parent_dir).mkdir(parents=True, exist_ok=True)
        with open(json_out_path, "w") as f:
            f.write(out_text)
    
    print("  ✓ Test script completed successfully")
    return res.stdout


@app.function(image=IMG, gpu="H100", timeout=7200, volumes={"/opb-generators": GENS, "/results": RESULTS, "/opb-source": SRC})
def test_h100(script_rel: str, args: list[str]) -> str:
    return _run_and_capture(script_rel, args)


@app.function(image=IMG, gpu="A100-40GB", timeout=7200, volumes={"/opb-generators": GENS, "/results": RESULTS, "/opb-source": SRC})
def test_a100(script_rel: str, args: list[str]) -> str:
    return _run_and_capture(script_rel, args)


@app.function(image=IMG, gpu="L40S", timeout=7200, volumes={"/opb-generators": GENS, "/results": RESULTS, "/opb-source": SRC})
def test_l40s(script_rel: str, args: list[str]) -> str:
    return _run_and_capture(script_rel, args)
