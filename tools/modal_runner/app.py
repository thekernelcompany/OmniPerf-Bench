
import modal

BASE = "ayushnangia16/nvidia-vllm-docker:b690e34824fd5a5c4054a0c0468ebfb6aa1dd215"
IMG = modal.Image.from_registry(BASE)
app = modal.App("vllm-agent-tests")

GENS = modal.Volume.from_name("opb-generators")
RESULTS = modal.Volume.from_name("opb-results")
SRC = modal.Volume.from_name("opb-source")


def _run_and_capture(script_rel: str, args: list[str]) -> str:
    import subprocess
    import os
    # Uninstall baked vllm and install agent source uploaded to /opb-source
    try:
        subprocess.run(["python3", "-m", "pip", "uninstall", "-y", "vllm"], check=False)
    except Exception:
        pass
    subprocess.run(["python3", "-m", "pip", "install", "--no-deps", "--force-reinstall", f"/opb-source/vllm_core-0000"], check=True)
    script_path = os.path.join("/opb-generators", script_rel)
    cmd = ["python", script_path, *args]
    res = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return res.stdout


@app.function(image=IMG, gpu=modal.gpu.H100(), timeout=3600, volumes={"/opb-generators": GENS, "/results": RESULTS, "/opb-source": SRC})
def test_h100(script_rel: str, args: list[str]) -> str:
    return _run_and_capture(script_rel, args)


@app.function(image=IMG, gpu=modal.gpu.A100(), timeout=3600, volumes={"/opb-generators": GENS, "/results": RESULTS, "/opb-source": SRC})
def test_a100(script_rel: str, args: list[str]) -> str:
    return _run_and_capture(script_rel, args)


@app.function(image=IMG, gpu=modal.gpu.L40S(), timeout=4800, volumes={"/opb-generators": GENS, "/results": RESULTS, "/opb-source": SRC})
def test_l40s(script_rel: str, args: list[str]) -> str:
    return _run_and_capture(script_rel, args)
