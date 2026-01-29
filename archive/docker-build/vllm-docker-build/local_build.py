#!/usr/bin/env python3
"""
Local builder for commit-tagged Docker images (vLLM, SGLang).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
import threading
import re
import shutil
import collections

try:
    from tqdm import tqdm as _tqdm
    def _progress(total: int, desc: str):
        return _tqdm(total=total, desc=desc)
except Exception:
    def _progress(total: int, desc: str):
        class _Dummy:
            def update(self, n: int): pass
            def __enter__(self): return self
            def __exit__(self, *args): return False
        return _Dummy()

DEFAULT_IMAGE_NAME = "nvidia-vllm-docker"
DEFAULT_REPO_URL = "https://github.com/vllm-project/vllm.git"

def run_command(command: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    process = subprocess.Popen(
        command, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr

def run_command_stream(command: List[str], cwd: Optional[Path], line_prefix: str, log_file: Optional[Path] = None) -> Tuple[int, str]:
    process = subprocess.Popen(
        command, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, universal_newlines=True,
    )
    tail = collections.deque(maxlen=200)
    log_fp = None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_file.open("w", encoding="utf-8")
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        tail.append(line)
        formatted = f"{line_prefix} {line}"
        print(formatted)
        if log_fp: log_fp.write(formatted + "\n")
    rc = process.wait()
    if log_fp: log_fp.close()
    return rc, "\n".join(tail)

def read_blacklist(blacklist_path: Path) -> Set[str]:
    if not blacklist_path.exists(): return set()
    return {line.strip() for line in blacklist_path.read_text().splitlines() if line.strip()}

def fetch_existing_tags(dockerhub_username: str, image_name: str) -> Set[str]:
    import urllib.request
    tags: Set[str] = set()
    repo = f"{dockerhub_username}/{image_name}"
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100"
    while url and url != "null":
        try:
            with urllib.request.urlopen(url) as resp:
                payload = json.loads(resp.read())
            for item in payload.get("results", []):
                name = item.get("name")
                if name: tags.add(name)
            url = payload.get("next")
        except: break
    return tags

def iter_commits_with_dockerfile(dataset_path: Path) -> Iterable[Tuple[str, Optional[str]]]:
    with dataset_path.open("r") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            commit = obj.get("commit")
            if commit: yield commit, obj.get("Dockerfile")

def ensure_repo_cloned(repo_dir: Path, repo_url: str) -> None:
    if repo_dir.exists(): return
    rc, out, err = run_command(["git", "clone", repo_url, str(repo_dir)])
    if rc != 0: raise RuntimeError(f"git clone failed: {err or out}")

_repo_lock = threading.Lock()

def _materialize_commit_tree(repo_dir: Path, commit_sha: str, dest_dir: Path) -> Tuple[bool, str]:
    with _repo_lock:
        run_command(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo_dir)
    if dest_dir.exists(): shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = (dest_dir.parent / f"{dest_dir.name}.tar").resolve()
    rc, _, err = run_command(["git", "archive", "--format=tar", "-o", str(tar_path), commit_sha], cwd=repo_dir)
    if rc != 0: return False, err
    rc2, _, err2 = run_command(["tar", "-C", str(dest_dir), "-xf", str(tar_path)])
    tar_path.unlink(missing_ok=True)
    return (rc2 == 0), (err2 if rc2 != 0 else "ok")

def resolve_dockerfile(repo_dir: Path) -> Optional[Path]:
    candidates = [repo_dir / "docker" / "Dockerfile", repo_dir / "Dockerfile", repo_dir / "examples" / "usage" / "triton" / "Dockerfile"]
    for path in candidates:
        if path.exists(): return path
    return None

def _apply_generic_dockerfile_fixes(dockerfile_path: Path) -> None:
    try:
        df_text = dockerfile_path.read_text()
        new_text = df_text

        # Detect focal (Ubuntu 20.04)
        is_focal = "ubuntu20.04" in new_text.lower() or "ubuntu:20.04" in new_text.lower() or "focal" in new_text.lower()

        # 1) Uppercase AS
        new_text = re.sub(r"(?im)^(\s*FROM\b[^\n]*?)\s+as\s+(\w+)", lambda m: f"{m.group(1)} AS {m.group(2)}", new_text)

        # 2) Normalize ENV
        def _normalize_env(line: str) -> str:
            if not re.match(r"^\s*ENV\b", line): return line
            after = re.sub(r"^\s*ENV\s+", "", line)
            if "=" in after: return line
            parts = after.strip().split(None, 1)
            if len(parts) == 2: return f"ENV {parts[0]}={parts[1]}"
            return line
        new_text = "\n".join(_normalize_env(l) for l in new_text.splitlines())

        # 3) Remove .git bind mounts (handles multiple formats including multi-line)
        # Most aggressive pattern first - removes the entire --mount directive for .git
        # Handle various whitespace and continuation patterns
        new_text = re.sub(r"--mount=type=bind,source=\.git,target=\.git\s*\\?\s*\n?\s*", "", new_text)
        new_text = re.sub(r"--mount=type=bind,\s*source=\.git,\s*target=\.git\s*\\?\s*\n?\s*", "", new_text)
        # Pattern for entire line that's just the .git mount
        new_text = re.sub(r"^\s*--mount=type=bind,\s*source=\.git,\s*target=\.git\s*\\?\s*$\n?", "", new_text, flags=re.MULTILINE)
        # Clean up any leftover empty continuations (line with just backslash)
        new_text = re.sub(r"^\s*\\\s*$\n", "", new_text, flags=re.MULTILINE)
        # Clean up double spaces left over
        new_text = re.sub(r"  +", " ", new_text)

        # 4) Focal / Python 3.10 Fix - Simplified and robust
        if is_focal:
            # Inject Miniforge at the top (no ToS requirements unlike Miniconda)
            conda_install = textwrap.dedent("""
                RUN apt-get update && apt-get install -y wget bzip2 ca-certificates && \\
                    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh && \\
                    bash miniforge.sh -b -p /opt/conda && rm miniforge.sh && \\
                    /opt/conda/bin/conda install -y python=3.10 && \\
                    ln -sf /opt/conda/bin/python3 /usr/bin/python3 && \\
                    ln -sf /opt/conda/bin/python3 /usr/bin/python3.10 && \\
                    ln -sf /opt/conda/bin/pip /usr/bin/pip && \\
                    ln -sf /opt/conda/bin/pip /usr/bin/pip3
                ENV PATH=/opt/conda/bin:$PATH
            """).strip()
            new_text = re.sub(r"(^FROM\s+.*?\n)", "\\1" + conda_install + "\n", new_text, count=1, flags=re.MULTILINE)

            # Replace add-apt-repository ppa:deadsnakes/ppa with a no-op, preserving && on both sides
            # Match: && add-apt-repository [-y] ppa:deadsnakes/ppa [-y] [\ newline] &&
            new_text = re.sub(
                r"&&\s*add-apt-repository\s+(-y\s+)?ppa:deadsnakes/ppa(\s+-y)?\s*\\?\s*\n?\s*&&",
                "&& true &&",
                new_text
            )
            # Also handle case where it's at the start or end of a chain
            new_text = re.sub(
                r"add-apt-repository\s+(-y\s+)?ppa:deadsnakes/ppa(\s+-y)?",
                "true",
                new_text
            )

            # Replace python${PYTHON_VERSION} with python3 in paths (miniforge provides python3)
            # This preserves the path structure while using the miniforge python
            new_text = re.sub(r"python\$\{PYTHON_VERSION\}", "python3", new_text)

            # Replace commands that fail with "link and path can't be the same" with 'true' (no-op)
            # This is more robust than trying to surgically remove from && chains
            new_text = re.sub(
                r"update-alternatives\s+--install\s+/usr/bin/python3\s+python3\s+/usr/bin/python3\s+\d+",
                "true",
                new_text
            )
            new_text = re.sub(
                r"update-alternatives\s+--set\s+python3\s+/usr/bin/python3",
                "true",
                new_text
            )
            # Also handle ln -sf that creates same source and target
            new_text = re.sub(
                r"ln\s+-sf\s+/usr/bin/python3-config\s+/usr/bin/python3-config",
                "true",
                new_text
            )

            # Remove -dev/-venv/-distutils suffixes from apt-get install for python3
            # since miniforge already provides these capabilities
            new_text = re.sub(r"\s+python3-dev\b", "", new_text)
            new_text = re.sub(r"\s+python3-venv\b", "", new_text)
            new_text = re.sub(r"\s+python3-distutils\b", "", new_text)
            # Also handle python3.10-* and python3.12-* etc
            new_text = re.sub(r"\s+python3\.\d+-dev\b", "", new_text)
            new_text = re.sub(r"\s+python3\.\d+-venv\b", "", new_text)
            new_text = re.sub(r"\s+python3\.\d+-distutils\b", "", new_text)
            new_text = re.sub(r"\s+python3\.\d+\b(?![\w.-])", "", new_text)
        else:
            # For non-focal, still harden PPA
            ppa_fix = (
                "(DEBIAN_FRONTEND=noninteractive apt-get update && "
                "apt-get install -y gnupg2 ca-certificates curl lsb-release && "
                "CODENAME=$(lsb_release -sc 2>/dev/null || . /etc/os-release && echo $VERSION_CODENAME); "
                "for i in 1 2 3; do apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys BA6932366A755776 && break || sleep 2; done && "
                "echo \"deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu $CODENAME main\" > /etc/apt/sources.list.d/deadsnakes.list && "
                "apt-get update)"
            )
            new_text = re.sub(r"add-apt-repository\s+(?:-y\s+)?ppa:deadsnakes/ppa(?:\s+-y)?", ppa_fix, new_text)

        # 5) Fix stray RUN
        lines_fix = []
        lines_src = new_text.splitlines()
        i = 0
        while i < len(lines_src):
            cur = lines_src[i]
            if cur.strip().upper() == "RUN" and i+1 < len(lines_src):
                nxt = lines_src[i+1]
                if not re.match(r"^\s*(FROM|RUN|CMD|LABEL|MAINTAINER|EXPOSE|ENV|ADD|COPY|ENTRYPOINT|VOLUME|USER|WORKDIR|ARG|ONBUILD|STOPSIGNAL|HEALTHCHECK|SHELL)\b", nxt.strip(), re.IGNORECASE):
                    lines_fix.append("RUN " + nxt.lstrip())
                    i += 2; continue
            lines_fix.append(cur); i += 1
        new_text = "\n".join(lines_fix)

        # 6) Compilation and DeepEP/nvshmem fixes
        if "DeepEP" in new_text:
            # Ensure torch is present before cloning/building DeepEP
            new_text = new_text.replace(
                "git clone https://github.com/deepseek-ai/DeepEP.git",
                "python3 -m pip install torch && git clone https://github.com/deepseek-ai/DeepEP.git"
            )
            # Disable build isolation to use the 'torch' we just installed
            new_text = new_text.replace("pip install .", "python3 -m pip install --no-build-isolation .")
            new_text = new_text.replace("pip3 install .", "python3 -m pip install --no-build-isolation .")
            # Limit jobs to prevent OOM
            new_text = new_text.replace("python3 -m pip install --no-build-isolation .", "MAX_JOBS=4 python3 -m pip install --no-build-isolation .")

        new_text = new_text.replace("git apply /sgl-workspace/DeepEP/third-party/nvshmem.patch", "git apply /sgl-workspace/DeepEP/third-party/nvshmem.patch || true")
        new_text = new_text.replace("sed -i '1i#include <unistd.h>' examples/moe_shuffle.cu", "find . -name \"*.cu\" -o -name \"*.cpp\" -o -name \"*.h\" | xargs sed -i '1i#include <unistd.h>' || true")
        new_text = new_text.replace("cmake --build build --target install -j", "cmake --build build --target install -j4")

        # 7) Fix FlashInfer wheel installation - hardcoded wheels often have wrong Python version
        # Replace hardcoded wheel URLs with PyPI install (auto-selects compatible version)
        # Fall back to skipping if PyPI install fails (some older versions not on PyPI)
        new_text = re.sub(
            r"python3 -m pip install https://[^\s]+flashinfer[^\s]+\.whl",
            r"python3 -m pip install flashinfer || echo 'FlashInfer not available for this Python version'",
            new_text
        )

        if new_text != df_text: dockerfile_path.write_text(new_text)
    except Exception as e: print(f"Fix error: {e}")

def _apply_context_fixes(worktree_dir: Path, dockerfile_path: Optional[Path], project: str = "vllm", commit_sha: str = "") -> None:
    if dockerfile_path and dockerfile_path.exists():
        # Apply generic Dockerfile fixes (git mount removal, deadsnakes fix, etc.) for ALL projects
        _apply_generic_dockerfile_fixes(dockerfile_path)

        # Inject SETUPTOOLS_SCM_PRETEND_VERSION to fix version detection without .git
        # This is needed because we removed .git bind mounts
        # Version must be PEP 440 compliant: use "0.0.0+<commit_sha>" format
        if commit_sha:
            txt = dockerfile_path.read_text()
            # Add ENV after the first FROM instruction
            # Use "0.0.0.dev0+g<sha>" format which is PEP 440 compliant
            pep440_version = f"0.0.0.dev0+g{commit_sha[:8]}"
            scm_env = f"\nENV SETUPTOOLS_SCM_PRETEND_VERSION={pep440_version}\nENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM={pep440_version}\n"
            # Insert after first FROM line
            txt = re.sub(r"(^FROM\s+[^\n]+\n)", r"\1" + scm_env, txt, count=1, flags=re.MULTILINE)
            dockerfile_path.write_text(txt)

        # vLLM-specific fixes
        if project == "vllm":
            txt = dockerfile_path.read_text()
            txt = txt.replace("outlines", "outlines<0.0.43")
            # Disable wheel size check (wheel is ~468MB due to FA2+FA3 kernels, limit is 250MB)
            # Set RUN_WHEEL_CHECK=false after first FROM instruction
            if "RUN_WHEEL_CHECK" not in txt:
                txt = re.sub(r"(^FROM\s+[^\n]+\n)", r"\1ENV RUN_WHEEL_CHECK=false\n", txt, count=1, flags=re.MULTILINE)

            # H100 SM 9.0 ONLY - dramatically reduces compile time
            # Must REPLACE existing architecture settings, not just add new ones
            # vLLM Dockerfiles use: ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'

            # 1. Replace ARG torch_cuda_arch_list (lowercase, space-separated with +PTX)
            txt = re.sub(
                r"ARG\s+torch_cuda_arch_list\s*=\s*['\"]?[^'\"\n]+['\"]?",
                "ARG torch_cuda_arch_list='9.0'",
                txt,
                flags=re.IGNORECASE
            )

            # 2. Replace ARG TORCH_CUDA_ARCH_LIST (uppercase variant)
            txt = re.sub(
                r"ARG\s+TORCH_CUDA_ARCH_LIST\s*=\s*['\"]?[^'\"\n]+['\"]?",
                "ARG TORCH_CUDA_ARCH_LIST='9.0'",
                txt
            )

            # 3. Replace ENV TORCH_CUDA_ARCH_LIST with hardcoded value (not variable reference)
            txt = re.sub(
                r"ENV\s+TORCH_CUDA_ARCH_LIST\s*=\s*\$\{[^}]+\}",
                "ENV TORCH_CUDA_ARCH_LIST='9.0'",
                txt
            )
            txt = re.sub(
                r"ENV\s+TORCH_CUDA_ARCH_LIST\s*=\s*['\"]?[^'\"\n$]+['\"]?",
                "ENV TORCH_CUDA_ARCH_LIST='9.0'",
                txt
            )

            # 4. Replace VLLM_FA_CMAKE_GPU_ARCHES if present
            txt = re.sub(
                r"(ARG|ENV)\s+VLLM_FA_CMAKE_GPU_ARCHES\s*=\s*['\"]?[^'\"\n]+['\"]?",
                r"\1 VLLM_FA_CMAKE_GPU_ARCHES='90-real'",
                txt
            )

            # 5. Add comprehensive H100-only env vars after first FROM
            # Increased NVCC_THREADS for faster compilation (was 2, now 6)
            # MAX_JOBS controls parallel make jobs, NVCC_THREADS controls nvcc internal parallelism
            h100_env_vars = """
ARG torch_cuda_arch_list='9.0'
ARG TORCH_CUDA_ARCH_LIST='9.0'
ENV TORCH_CUDA_ARCH_LIST='9.0'
ARG VLLM_FA_CMAKE_GPU_ARCHES='90-real'
ENV VLLM_FA_CMAKE_GPU_ARCHES='90-real'
ENV MAX_JOBS=6
ENV NVCC_THREADS=6
ENV CCACHE_DIR=/root/.cache/ccache
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache
"""
            txt = re.sub(r"(^FROM\s+[^\n]+\n)", r"\1" + h100_env_vars, txt, count=1, flags=re.MULTILINE)
            dockerfile_path.write_text(txt)

def build_one_commit(repo_dir: Path, commit_sha: str, image_name: str, dockerhub_username: str, platform: str, push: bool, cache_from: str|None, cache_to: str|None, dataset_dockerfile: str|None, show_build_logs: bool, project: str) -> Tuple[str, bool, str]:
    repo_dir_abs = repo_dir.resolve()
    worktree_dir = repo_dir_abs.parent / ".build-contexts" / commit_sha
    ok_ctx, msg = _materialize_commit_tree(repo_dir, commit_sha, worktree_dir)
    if not ok_ctx: return commit_sha, False, f"Context fail: {msg}"
    try:
        dockerfile_path = resolve_dockerfile(worktree_dir)
        if not dockerfile_path and project == "sglang":
            dockerfile_path = worktree_dir / "Dockerfile"
            dockerfile_path.write_text("FROM nvcr.io/nvidia/tritonserver:24.04-py3-min\nENV DEBIAN_FRONTEND=noninteractive\nRUN apt update && apt install -y python3 python3-pip curl git sudo\nWORKDIR /sgl-workspace\nRUN git clone --depth=1 https://github.com/sgl-project/sglang.git && cd sglang && pip install -e \"python[all]\"")
        if not dockerfile_path: return commit_sha, False, "No Dockerfile"
        _apply_context_fixes(worktree_dir, dockerfile_path, project, commit_sha)
        tag = f"docker.io/{dockerhub_username}/{image_name}:{commit_sha}"
        cmd = ["docker", "buildx", "build", "--platform", platform, "--file", str(dockerfile_path), "--tag", tag, "--label", f"org.opencontainers.image.revision={commit_sha}"]
        if show_build_logs: cmd += ["--progress", "plain"]
        if cache_from: cmd += ["--cache-from", cache_from]
        if cache_to: cmd += ["--cache-to", cache_to]
        if push: cmd.append("--push")
        cmd.append(str(worktree_dir))
        
        status_dir = repo_dir_abs.parent / ".build-status"
        status_dir.mkdir(parents=True, exist_ok=True)
        (status_dir / f"{commit_sha}.building").write_text("building")
        
        if show_build_logs:
            rc, tail = run_command_stream(cmd, None, f"[{commit_sha[:7]}]", repo_dir_abs.parent / ".build-logs" / f"{commit_sha}.log")
            ok, msg = (rc == 0), ("ok" if rc == 0 else tail)
        else:
            rc, out, err = run_command(cmd)
            ok, msg = (rc == 0), ("ok" if rc == 0 else err or out)
            if not ok:
                log_dir = repo_dir_abs.parent / ".build-logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / f"{commit_sha}.log").write_text(out + "\n" + err)
        
        (status_dir / f"{commit_sha}.building").unlink(missing_ok=True)
        (status_dir / f"{commit_sha}.done").write_text("ok" if ok else "fail")

        # Cleanup local images after successful push to save disk space
        if ok and push:
            try:
                # Remove the specific image we just built
                subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, timeout=60)
                # Prune dangling images and build cache periodically
                subprocess.run(["docker", "image", "prune", "-f"], capture_output=True, timeout=60)
            except Exception:
                pass  # Don't fail the build if cleanup fails

        return commit_sha, ok, msg
    finally:
        shutil.rmtree(worktree_dir, ignore_errors=True)

def _active_buildx_driver() -> Optional[str]:
    rc, out, _ = run_command(["docker", "buildx", "ls"])
    if rc != 0: return None
    for line in out.splitlines():
        if "*" in line and "NAME/" not in line:
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) > 1: return parts[1].split()[0]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dockerhub-username", required=True)
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME)
    parser.add_argument("--dataset", default="nvidia-vllm-docker.jsonl")
    parser.add_argument("--blacklist", default="blacklist.txt")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--workdir", default="vllm")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--skip-pushed", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--cache-dir", default=".buildx-cache")
    parser.add_argument("--show-build-logs", action="store_true")
    parser.add_argument("--project", default="vllm", choices=["vllm", "sglang"])
    args = parser.parse_args()

    dataset_path, blacklist_path, repo_dir = Path(args.dataset), Path(args.blacklist), Path(args.workdir)
    ensure_repo_cloned(repo_dir, args.repo_url)
    blacklist = read_blacklist(blacklist_path)
    records = list(iter_commits_with_dockerfile(dataset_path))
    filtered = [(c, d) for (c, d) in records if c not in blacklist]
    if args.skip_pushed:
        existing = fetch_existing_tags(args.dockerhub_username, args.image_name)
        filtered = [(c, d) for (c, d) in filtered if c not in existing]
    
    to_build = filtered[:args.batch_size]
    if not to_build: print("Nothing to build"); return 0
    print(f"Building {len(to_build)} commits (parallel={args.max_parallel})...")

    driver = _active_buildx_driver() or ""
    cache_from, cache_to = None, None
    if driver != "docker":
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_from = f"type=local,src={cache_dir}"
        cache_to = f"type=local,dest={cache_dir},mode=max"

    def _worker(item):
        return build_one_commit(repo_dir, item[0], args.image_name, args.dockerhub_username, args.platform, not args.no_push, cache_from, cache_to, item[1], args.show_build_logs, args.project)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = {ex.submit(_worker, item): item[0] for item in to_build}
        with _progress(total=len(to_build), desc="Building") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                results.append(res)
                pbar.update(1)
                print(f"{res[0]} => {'OK' if res[1] else 'FAIL'}")

    return 1 if any(not r[1] for r in results) else 0

if __name__ == "__main__":
    raise SystemExit(main())
