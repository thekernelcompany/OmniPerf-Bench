# sandbox_demo.py
import modal

APP_NAME = "simple-sandbox-demo"
VOLUME_NAME = "agent-volume"
WORKDIR_IN_VOLUME = "/vol/workspaces"

# CUDA base image with Python 3.11 baked in
CUDA_VERSION = "12.8.0"     # keep <= host CUDA version
FLAVOR = "devel"
OS = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{FLAVOR}-{OS}"

def main():
    # Image: CUDA + Python + a couple basics
    image = (
        modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
        .apt_install(["git", "build-essential"])
        .pip_install("wheel", "packaging", "ninja")
    )

    # App + Volume
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    vol = modal.Volume.from_name(VOLUME_NAME)

    # Create the sandbox with GPU and the volume mounted at /vol
    sandbox = modal.Sandbox.create(
        image=image,
        app=app,
        timeout=60 * 60,               # 1 hour
        volumes={"/vol": vol},
        workdir=WORKDIR_IN_VOLUME,     # working dir inside the volume
        cpu=8,
        memory=32768,                  # 16 GB
        gpu="A100-80GB",   # pick a GPU; adjust as needed
    )

    # Make sure /vol/workspaces exists
    try:
        sandbox.mkdir(WORKDIR_IN_VOLUME, parents=True)
    except Exception:
        # It may already exist; that's fine
        pass

    print(f"Sandbox ready. Working dir: {WORKDIR_IN_VOLUME}")

    # Optional: run a couple sanity checks inside the sandbox
    # 1) Check the GPU
    res = sandbox.exec("nvidia-smi")
    for line in res.stdout:
        print(line, end="")
    res.wait()
    print(f"\n[nvidia-smi exit code: {res.returncode}]")

    # 2) Check Python and CUDA toolkit presence
    res = sandbox.exec("python", "-V")
    for line in res.stdout:
        print(line, end="")
    res.wait()

    # 3) Create a file in /vol/workspace to prove write access
    f = sandbox.open(f"{WORKDIR_IN_VOLUME}/hello.txt", "w")
    f.write("hello from modal sandbox with GPU!\n")
    f.close()
    print("\nWrote /vol/workspace/hello.txt")

if __name__ == "__main__":
    main()
