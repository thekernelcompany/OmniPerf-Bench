#!/usr/bin/env python3
"""docker -> udocker shim for unprivileged container execution.

The Prime Intellect H100 instance is itself a non-privileged container
(no CAP_SYS_ADMIN, no usable user namespaces), so dockerd can't run.
udocker uses PRoot/ptrace and works fine.

This shim accepts the subset of `docker run` flags that
scripts/runners/run_3way_benchmarks.py emits:
    docker run --rm --gpus all -e VAR=val -v src:dst:ro? --shm-size=Xg
        --entrypoint bash <image> -c <cmd>

Anything else (build, push, login, pull as standalone subcommand) we
forward to udocker with --allow-root, or bail clearly.
"""
import os
import shlex
import subprocess
import sys
import uuid

UDOCKER = ["udocker", "--allow-root"]


def die(msg, code=2):
    print(f"docker-shim: {msg}", file=sys.stderr)
    sys.exit(code)


def parse_run(argv):
    """Parse `docker run <flags> <image> [-- cmd...]`.

    Returns dict with: rm, gpus, env (list), volumes (list), shm_size,
    entrypoint, image, cmd (argv list after image).
    """
    out = {
        "rm": False,
        "gpus": None,
        "env": [],
        "volumes": [],
        "shm_size": None,
        "entrypoint": None,
        "image": None,
        "cmd": [],
        "workdir": None,
        "name": None,
    }
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        # End of flags = first non-option that we recognize as image.
        if a == "--rm":
            out["rm"] = True
            i += 1
        elif a == "--gpus":
            out["gpus"] = argv[i + 1]
            i += 2
        elif a.startswith("--gpus="):
            out["gpus"] = a.split("=", 1)[1]
            i += 1
        elif a in ("-e", "--env"):
            out["env"].append(argv[i + 1])
            i += 2
        elif a.startswith("--env="):
            out["env"].append(a.split("=", 1)[1])
            i += 1
        elif a in ("-v", "--volume"):
            out["volumes"].append(argv[i + 1])
            i += 2
        elif a.startswith("--volume="):
            out["volumes"].append(a.split("=", 1)[1])
            i += 1
        elif a.startswith("--shm-size"):
            if "=" in a:
                out["shm_size"] = a.split("=", 1)[1]
                i += 1
            else:
                out["shm_size"] = argv[i + 1]
                i += 2
        elif a == "--entrypoint":
            out["entrypoint"] = argv[i + 1]
            i += 2
        elif a.startswith("--entrypoint="):
            out["entrypoint"] = a.split("=", 1)[1]
            i += 1
        elif a in ("-w", "--workdir"):
            out["workdir"] = argv[i + 1]
            i += 2
        elif a.startswith("--workdir="):
            out["workdir"] = a.split("=", 1)[1]
            i += 1
        elif a == "--name":
            out["name"] = argv[i + 1]
            i += 2
        elif a.startswith("--name="):
            out["name"] = a.split("=", 1)[1]
            i += 1
        elif a == "-i" or a == "-t" or a == "-it" or a == "--interactive" or a == "--tty":
            i += 1  # ignored; we always run non-interactive
        elif a == "--network" or a == "--net":
            i += 2  # we always use host network — udocker default
        elif a.startswith("--network=") or a.startswith("--net="):
            i += 1
        elif a == "--privileged":
            i += 1  # ignored, we can't be privileged anyway
        elif a.startswith("-"):
            die(f"unsupported `docker run` flag: {a}")
        else:
            # First positional is the image
            out["image"] = a
            out["cmd"] = argv[i + 1:]
            break
    if out["image"] is None:
        die("no image specified in `docker run`")
    return out


def ensure_image(image):
    """Pull (idempotently) and return udocker's canonical image name.

    `udocker images` lists partial pulls as present, so we always run pull
    rather than short-circuit. udocker's pull is idempotent on already-
    complete layers, so the cost is small for cache hits.
    """
    p = subprocess.run(UDOCKER + ["pull", image], stdout=sys.stderr, stderr=sys.stderr,
                       timeout=1800)
    if p.returncode != 0:
        die(f"udocker pull failed for {image}", p.returncode)
    return image


def run_subcommand(argv):
    parsed = parse_run(argv)
    image = ensure_image(parsed["image"])

    cname = parsed["name"] or f"shim-{uuid.uuid4().hex[:12]}"

    # Create container
    cr = subprocess.run(UDOCKER + ["create", f"--name={cname}", image],
                        capture_output=True, text=True)
    if cr.returncode != 0:
        # Container with that name might already exist; if so try to reuse.
        if "already exists" in (cr.stderr + cr.stdout):
            pass
        else:
            die(f"udocker create failed: {cr.stderr.strip() or cr.stdout.strip()}",
                cr.returncode)

    # Configure for GPU + ptrace mode
    if parsed["gpus"]:
        # --gpus implies "expose all NVIDIA GPUs". udocker setup --nvidia maps
        # host driver libs into the container; per-GPU isolation comes from
        # CUDA_VISIBLE_DEVICES.
        #
        # Use --force so host's libnvidia-ml.so.<driver> overwrites any 0-byte
        # placeholders the image may have shipped (vLLM benchmark images carry
        # stubs for an old driver version that pynvml will choke on).
        sn = subprocess.run(UDOCKER + ["setup", "--nvidia", "--force", cname],
                            capture_output=True, text=True)
        if sn.returncode != 0 and "already" not in (sn.stderr + sn.stdout).lower():
            print(sn.stderr, file=sys.stderr)
    # Default execmode P1 is fine; only switch if env override.
    execmode = os.environ.get("UDOCKER_EXECMODE", "P1")
    subprocess.run(UDOCKER + ["setup", f"--execmode={execmode}", cname],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Build udocker run command
    run_argv = list(UDOCKER) + ["run"]
    for v in parsed["volumes"]:
        # udocker's Uvolume.split does volume.split(":", 1) so host:cont:ro
        # parses as host="host" cont="cont:ro" (malformed). Strip the trailing
        # mode flag — udocker doesn't enforce ro/rw on bind mounts anyway.
        if v.endswith(":ro") or v.endswith(":rw"):
            v = v[:-3]
        run_argv += ["-v", v]
    # Auto-mount host CUDA dirs when --gpus is requested. Some baseline images
    # ship 0-byte placeholders for torchvision-bundled libcudart that need to
    # be patched at docker_cmd time; doing so requires a real libcudart inside
    # the container. Mounting host's /usr/local/cuda-* gives docker_cmd a
    # source it can copy from.
    if parsed["gpus"]:
        import glob as _glob
        for cudadir in _glob.glob("/usr/local/cuda-*"):
            run_argv += ["-v", f"{cudadir}:{cudadir}"]
        if os.path.isdir("/usr/local/cuda"):
            run_argv += ["-v", "/usr/local/cuda:/usr/local/cuda"]
    for e in parsed["env"]:
        run_argv += ["-e", e]
    # Auto-propagate CUDA_VISIBLE_DEVICES from host env if not already specified.
    # udocker --nvidia exposes ALL host GPUs by default; isolation across parallel
    # workers (one per GPU) relies on this env var.
    env_keys = {e.split("=", 1)[0] for e in parsed["env"]}
    if "CUDA_VISIBLE_DEVICES" not in env_keys and "CUDA_VISIBLE_DEVICES" in os.environ:
        run_argv += ["-e", f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"]
    if parsed["workdir"]:
        run_argv += ["-w", parsed["workdir"]]
    if parsed["entrypoint"]:
        run_argv += [f"--entrypoint={parsed['entrypoint']}"]
    run_argv.append(cname)
    run_argv += parsed["cmd"]

    # NOTE: --shm-size is dropped — udocker shares /dev/shm with host, which
    # on this 1TB-RAM box is functionally unlimited.

    rc = subprocess.run(run_argv).returncode

    if parsed["rm"]:
        subprocess.run(UDOCKER + ["rm", cname],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sys.exit(rc)


def main():
    argv = sys.argv[1:]
    if not argv:
        die("no subcommand")
    sub = argv[0]
    rest = argv[1:]
    if sub == "run":
        run_subcommand(rest)
    elif sub == "manifest":
        # Subset: `docker manifest inspect <image>` — runner uses this as a
        # cheap "does this image exist" probe before falling back to wheel mode.
        # Do NOT trust `udocker images` — it lists images with interrupted
        # pulls as "present". Always run an idempotent pull instead.
        if rest and rest[0] == "inspect" and len(rest) >= 2:
            image = rest[1]
            p = subprocess.run(UDOCKER + ["pull", image],
                               capture_output=True, text=True, timeout=1800)
            sys.exit(0 if p.returncode == 0 else 1)
        die(f"`docker manifest {' '.join(rest)}` not implemented in shim")
    elif sub in ("pull", "images", "rmi", "rm", "ps", "login", "logout"):
        # Passthrough subcommands udocker supports directly
        sys.exit(subprocess.run(UDOCKER + [sub] + rest).returncode)
    elif sub == "info":
        print("Server Version: udocker-shim 1.0")
        print("Storage Driver: udocker (PRoot)")
        sys.exit(0)
    elif sub == "version" or sub == "--version":
        print("Docker shim over udocker")
        sys.exit(0)
    else:
        die(f"subcommand `{sub}` not implemented in shim")


if __name__ == "__main__":
    main()
