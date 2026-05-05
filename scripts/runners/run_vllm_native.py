#!/usr/bin/env python3
"""Native vLLM benchmark runner — no docker, no udocker.

Per commit:
  1. uv venv at /tmp/native_venvs/<human_short>
  2. uv pip install https://wheels.vllm.ai/<parent>/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
     with --extra-index-url https://download.pytorch.org/whl/cu128 (pulls matching torch)
  3. uv pip install benchmark deps (aiohttp, pandas, datasets)
  4. Apply agent patch to the installed vllm package
  5. Start vllm.entrypoints.openai.api_server natively (no container)
  6. Run benchmark_serving.py (or offline benchmark)
  7. Parse metrics, write result JSON

Bypasses udocker entirely → no PRoot ↔ multiprocessing.spawn bug.
"""
import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Path resolution order (each can be overridden by CLI flag):
#   1. CLI flag (e.g. --root)
#   2. env var (e.g. OMNIPERF_ROOT)
#   3. derived from this script's location (parent.parent.parent of scripts/runners/)
#   4. legacy hardcode (/root/OmniPerf-Bench) — kept as last-resort default
_DEFAULT_ROOT = Path(
    os.environ.get("OMNIPERF_ROOT")
    or os.environ.get("ISOBENCH_ROOT")
    or str(Path(__file__).resolve().parent.parent.parent)
)

# These are populated in main() once flags are parsed; module-level placeholders
# preserve symbol names used throughout the script.
ROOT: Path = _DEFAULT_ROOT
MAPPING_FILE: Path = ROOT / "data/mappings/vllm_oh_mapping.json"
RESULTS_DIR: Path = ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45/results"
PATCHES_DIR: Path = ROOT / "ISO-Bench/state/runs/vllm/openhands_sonnet45/flat"

VENV_ROOT = Path(os.environ.get("NATIVE_VENV_ROOT", "/tmp/native_venvs"))

# HF_CACHE: prefer /ephemeral if it exists & is writable (Prime Intellect-style),
# else use a project-local dir, else fall back to ~/.cache/huggingface.
def _default_hf_cache() -> Path:
    eph = Path("/ephemeral/huggingface_cache")
    try:
        eph.mkdir(parents=True, exist_ok=True)
        return eph
    except (PermissionError, OSError):
        pass
    home_cache = Path.home() / ".cache/huggingface"
    return home_cache
HF_CACHE: Path = Path(os.environ.get("HF_HOME", str(_default_hf_cache())))

# uv binary: prefer $UV_BIN env var, else PATH, else /root/.local/bin/uv legacy.
def _default_uv_bin() -> Path:
    if env_uv := os.environ.get("UV_BIN"):
        return Path(env_uv)
    found = shutil.which("uv")
    if found:
        return Path(found)
    return Path("/root/.local/bin/uv")
UV_BIN: Path = _default_uv_bin()

WHEEL_URL_TEMPLATE = "https://wheels.vllm.ai/{commit}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

# benchmark_serving.py needs --dataset-path for sharegpt/sonnet. The local copy
# avoids re-downloading per-commit (and some old versions accept no --dataset-name
# default but still need a path).
SHAREGPT_PATH = str(ROOT / "data/archive/sharegpt_dataset.json")

# Agent-name → (patches_dir_relative, results_dir_relative). Add a new entry to
# benchmark a new agent without touching the rest of the file. `--agent-name`
# selects one of these at runtime.
AGENT_PROFILES = {
    "openhands_sonnet45": (
        "ISO-Bench/state/runs/vllm/openhands_sonnet45/flat",
        "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45/results",
    ),
    "openhands_gpt5": (
        "ISO-Bench/state/runs/vllm/openhands_gpt5/flat",
        "archive/results/2026-05/iso_bench_results_3way_openhands_gpt5/results",
    ),
}

MODEL_OVERRIDES = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "ibm-ai-platform/Bamba-9B-v2": "meta-llama/Meta-Llama-3-8B-Instruct",
    "ibm-ai-platform/Bamba-9B": "meta-llama/Meta-Llama-3-8B-Instruct",
    # ISO-Bench dataset has the (HF-nonexistent) typo `meta-llama/Llama-3-8B`;
    # canonical repo is `meta-llama/Meta-Llama-3-8B`. Keep both forms mapped.
    "meta-llama/Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    # ISO-Bench typo: Qwen3-7B-Instruct doesn't exist on HF; canonical is
    # Qwen2.5-7B-Instruct (which is what the dataset's `models` field says).
    "Qwen/Qwen3-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}


# ISO-Bench HF dataset (`ISO-Bench/ISO-Bench`) is the authoritative source of
# perf_commands and target models per the upstream repository. We load it once
# at first call and merge it over `vllm_oh_mapping.json` for these two fields.
# The local mapping retains authority for runtime/dep overrides
# (`transformers_pin`, `vllm_pypi_version`, `outlines_pin`, `tp`).
_ISO_BENCH_CACHE = None


def load_iso_bench_perf_overrides() -> dict:
    """Returns {commit_short_8: {'perf_command': str, 'model': str, ...}}.
    Falls back to empty dict if dataset is unavailable.
    """
    global _ISO_BENCH_CACHE
    if _ISO_BENCH_CACHE is not None:
        return _ISO_BENCH_CACHE
    candidates = [
        Path("/tmp/iso_bench_hf/data/vllm/train.parquet"),
        ROOT / "data/iso_bench/vllm/train.parquet",
    ]
    for path in candidates:
        if path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(path)
                cache = {}
                for _, row in df.iterrows():
                    h8 = str(row["commit_hash"])[:8]
                    pc_raw = row["perf_command"]
                    pc = str(pc_raw) if pc_raw is not None else ""
                    models_raw = row["models"] if "models" in df.columns else None
                    models = list(models_raw) if models_raw is not None else []
                    cache[h8] = {"perf_command": pc, "models": models}
                _ISO_BENCH_CACHE = cache
                return cache
            except Exception as e:
                print(f"  warn: ISO-Bench parquet at {path} unreadable: {e}")
    _ISO_BENCH_CACHE = {}
    return _ISO_BENCH_CACHE


def merge_with_iso_bench(commit_short: str, info: dict, log) -> dict:
    """Return a copy of `info` with ISO-Bench's perf_command + model spliced in.

    The model is extracted from `--model X` inside ISO's perf_command — that
    field is the ground truth, since ISO-Bench's separate `models` column is
    sometimes inconsistent with what perf_command actually passes (observed:
    perf_command says Llama-3.1, models[0] says Llama-2 for the same row).
    """
    iso = load_iso_bench_perf_overrides().get(commit_short, {})
    if not iso:
        return info
    merged = dict(info)
    iso_pc = (iso.get("perf_command") or "").strip()
    if iso_pc and iso_pc != (info.get("perf_command", "") or "").strip():
        log(f"  ISO-Bench perf_command override: {iso_pc[:140]}")
        merged["perf_command"] = iso_pc

    # Extract --model from ISO perf_command (handles `--model X` and
    # `--model-path X` for sglang-style commands too).
    if iso_pc:
        m = re.search(r'--model(?:-path)?(?:\s+|=)(\S+)', iso_pc)
        if m:
            iso_model = m.group(1)
            if iso_model and iso_model != info.get("model", ""):
                log(f"  ISO-Bench model override: {info.get('model', '')} -> {iso_model}")
                merged["model"] = iso_model
    return merged


# Orgs whose models are gated and require the dedicated gated-access token.
# Public/non-gated orgs (RedHatAI, neuralmagic, deepseek-ai, Qwen, ibm-ai-platform,
# huggyllama, RedHatAI mirrors) work fine with the default token.
GATED_ORGS = ("meta-llama/", "mistralai/")


def _read_token_file(path: Path) -> str:
    try:
        return path.read_text().strip() if path.exists() else ""
    except Exception:
        return ""


def get_hf_token(model: str = "") -> str:
    """Return an HF access token. Uses the gated-access token (if present)
    for models in GATED_ORGS, else the default cached HF token (shikhar's).
    """
    gated_path = Path.home() / ".config/omniperf/hf_token_gated"
    default_path = Path.home() / ".cache/huggingface/token"
    if any(model.startswith(org) for org in GATED_ORGS):
        tok = _read_token_file(gated_path)
        if tok:
            return tok
        # fallback if gated file missing — we still try the default
    return _read_token_file(default_path)


def wheel_exists(commit_full: str) -> bool:
    url = WHEEL_URL_TEMPLATE.format(commit=commit_full)
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def free_port(start: int = 30000) -> int:
    """Pick a port not currently bound. Per-worker pinning via CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return start + int(cvd) * 1000  # 30000, 31000, 32000, ...


def kill_gpu_orphans(log) -> None:
    """Kill stale python procs from previous /tmp/native_venvs runs that are
    holding GPU memory after their parent (api_server / launch_server) exited.

    vLLM uses multiprocessing.spawn for its tensor-parallel workers; if the
    parent dies via SIGKILL (which pkill does), the spawn-children become
    orphaned with PPID=1, keep their CUDA contexts open, and hold GiB of GPU
    memory until manually killed. Run this before every commit to guarantee
    a clean GPU state.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) != 2:
                continue
            pid, name = parts
            if "/tmp/native_venvs/" in name or "/tmp/native_sglang_venvs/" in name \
               or "/ephemeral/native_venvs/" in name or "/ephemeral/native_sglang_venvs/" in name:
                # Any /tmp/native_*venvs/<commit>/python compute app is stale by
                # definition at the start of a new commit's run_one. The PPID=1
                # check is too strict — some orphans are mid-reparent and still
                # show their dead parent's PID for a few seconds. Kill them all.
                log(f"  killing stale GPU compute app PID {pid} ({name})")
                try: os.kill(int(pid), 9)
                except ProcessLookupError: pass
        time.sleep(1)
    except Exception as e:
        log(f"  kill_gpu_orphans error: {e}")


def kill_port(port: int, log) -> None:
    """Kill anything currently listening on `port`. vLLM's multiprocessing
    spawn-children sometimes outlive the parent server even after terminate(),
    holding the bind port and breaking the next commit on this worker.
    """
    try:
        r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                           capture_output=True, text=True, timeout=10)
        pids = set(re.findall(r'pid=(\d+)', r.stdout))
        for pid in pids:
            try:
                os.kill(int(pid), 9)
                log(f"  Killed leftover PID {pid} on port {port}")
            except ProcessLookupError:
                pass
        if pids:
            time.sleep(2)
    except Exception as e:
        log(f"  kill_port({port}) error: {e}")


SOURCE_ROOT = Path("/tmp/vllm_src")


def setup_venv_overlay(commit_short: str, parent_full: str, log,
                       pypi_version: str = "",
                       outlines_pin: str = "==0.0.46") -> Path:
    """PyPI-vllm fallback for commits with no wheel at wheels.vllm.ai.

    Strategy: install vllm==<pypi_version> directly from PyPI. The PyPI
    release is tagged at (or very near) the parent commit, so the installed
    site-packages/vllm/ matches the baseline that the agent patch was
    generated against. Apply the agent patch on top.

    Uses Python 3.11 because old vllm (0.3-0.5) only ships cp38-cp311 wheels.
    """
    venv = VENV_ROOT / commit_short
    if venv.exists():
        shutil.rmtree(venv, ignore_errors=True)
    venv.parent.mkdir(parents=True, exist_ok=True)
    log(f"  Creating venv at {venv} (Python 3.11)")
    subprocess.run([str(UV_BIN), "venv", "--python", "3.11", str(venv)],
                   check=True, capture_output=True, text=True, timeout=120)
    py = venv / "bin/python"

    if not pypi_version:
        raise RuntimeError(f"no PyPI vllm version mapped for {commit_short} (parent {parent_full[:12]})")

    log(f"  Installing vllm=={pypi_version} from PyPI")
    env = os.environ.copy()
    env["UV_SKIP_WHEEL_FILENAME_CHECK"] = "1"
    r = subprocess.run(
        [str(UV_BIN), "pip", "install", "--python", str(py),
         f"vllm=={pypi_version}",
         "setuptools",  # triton/build deps need it
         "--index-strategy", "unsafe-best-match"],
        capture_output=True, text=True, timeout=900, env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(f"vllm=={pypi_version} install failed: {(r.stderr or r.stdout)[-500:]}")

    # Pin transformers + outlines to versions that match this older vllm slice.
    # PyPI vllm 0.5.x pulled torch 2.3.0; transformers 4.45+ wants torch>=2.4
    # so use 4.42-4.44. Outlines 1.0+ removed `outlines.fsm` which old vllm
    # imports — pin to <1.0. Also pin pydantic and lm-format-enforcer for
    # compatibility with the baseline's call sites.
    log(f"  Pinning baseline-compatible deps (transformers, outlines{outlines_pin}, lmformatenforcer)")
    subprocess.run(
        [str(UV_BIN), "pip", "install", "--python", str(py),
         "transformers>=4.42,<4.45",
         f"outlines{outlines_pin}",
         "lm-format-enforcer==0.10.1",
         "numpy<2",  # outlines 0.0.34 imports numpy.lib.function_base, removed in numpy 2.x
         "aiohttp", "pandas", "pillow"],
        capture_output=True, text=True, timeout=300, env=env,
    )
    # outlines 0.0.46 imports `from pyairports.airports import AIRPORT_LIST`
    # at startup but the pinned pyairports==0.0.1 wheel is empty (no module).
    # Stub it out — we don't use airport types in benchmarks.
    site_pkgs = next((venv / "lib").glob("python3.*/site-packages"))
    pa = site_pkgs / "pyairports"
    pa.mkdir(parents=True, exist_ok=True)
    (pa / "__init__.py").write_text("")
    (pa / "airports.py").write_text(
        "AIRPORT_LIST = []\n"
        "class Airport: pass\n"
        "class AirportNotFoundException(Exception): pass\n"
    )
    log(f"  Overlay (PyPI install) done")
    return venv


def setup_venv(commit_short: str, parent_full: str, log) -> Path:
    """Create a uv venv and install vllm wheel + deps. Returns venv path."""
    venv = VENV_ROOT / commit_short
    if venv.exists():
        shutil.rmtree(venv, ignore_errors=True)
    venv.parent.mkdir(parents=True, exist_ok=True)
    log(f"  Creating venv at {venv}")
    subprocess.run([str(UV_BIN), "venv", "--python", "3.12", str(venv)],
                   check=True, capture_output=True, text=True, timeout=120)
    py = venv / "bin/python"
    wheel_url = WHEEL_URL_TEMPLATE.format(commit=parent_full)
    log(f"  Installing vllm wheel from {wheel_url}")
    env = os.environ.copy()
    env["UV_SKIP_WHEEL_FILENAME_CHECK"] = "1"
    r = subprocess.run(
        [str(UV_BIN), "pip", "install",
         "--python", str(py),
         wheel_url,
         "--extra-index-url", "https://download.pytorch.org/whl/cu128",
         "--index-strategy", "unsafe-best-match"],
        capture_output=True, text=True, timeout=900, env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(f"vllm install failed: {(r.stderr or r.stdout)[-400:]}")
    return venv


def apply_patch(venv: Path, patch_path: Path, log) -> None:
    """Apply agent patch to the installed vllm package."""
    site_pkgs = next((venv / "lib").glob("python3.*/site-packages"))
    cwd = site_pkgs  # patch paths are like 'a/vllm/...' relative to package root
    log(f"  Applying patch at {site_pkgs}")
    r = subprocess.run(
        ["patch", "-p1", "--force"],
        cwd=str(cwd), input=patch_path.read_text(),
        capture_output=True, text=True, timeout=60,
    )
    log(f"  patch result rc={r.returncode}: {r.stdout[-300:].strip()}")


def start_server(venv: Path, model: str, port: int, log, max_model_len: int = 4096,
                 extra_server_args: list = None) -> subprocess.Popen:
    py = venv / "bin/python"
    log(f"  Starting vllm.entrypoints.openai.api_server --model {model} --port {port} --max-model-len {max_model_len}")
    env = os.environ.copy()
    env["HF_TOKEN"] = get_hf_token(model)
    env["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]
    env["VLLM_USE_V1"] = "0"
    env["HF_HOME"] = str(HF_CACHE)
    # Allow setting max-model-len > model's max_position_embeddings (e.g. opt-125m=2048).
    # vLLM warns but proceeds; rope-scaling extends context beyond the trained window.
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    # If multiple GPUs are visible, run with --tensor-parallel-size=N so MoE
    # / huge models can shard across them. Per-GPU runs default to tp=1.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    num_gpus = max(1, len([x for x in cvd.split(",") if x.strip()]))
    cmd = [str(py), "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--port", str(port),
           "--host", "127.0.0.1",
           "--max-model-len", str(max_model_len),
           "--tensor-parallel-size", str(num_gpus),
           "--disable-log-requests",
           "--enforce-eager",
           "--trust-remote-code"]
    if extra_server_args:
        cmd.extend(extra_server_args)
        log(f"  Extra server args: {extra_server_args}")
    # cwd=/tmp so an accidental `vllm/` directory at the project root (e.g. a
    # leftover submodule clone) doesn't shadow the venv's installed package
    # via namespace-package resolution (Python adds cwd to sys.path[0]).
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, cwd="/tmp")
    return proc


def extract_server_flags_from_perf(perf_command: str) -> list:
    """When the perf_command implies server-side flags (e.g. --enable-prefix-caching,
    --enable-chunked-prefill), pass those to start_server too. We strip them from
    the bench client invocation already, but the SERVER needs them to actually
    enable the feature being benchmarked.
    """
    args = []
    if '--enable-prefix-caching' in perf_command:
        args.append('--enable-prefix-caching')
    if '--enable-chunked-prefill' in perf_command:
        args.append('--enable-chunked-prefill')
    m = re.search(r'--max-num-batched-tokens\s+(\d+)', perf_command)
    if m: args += ['--max-num-batched-tokens', m.group(1)]
    m = re.search(r'--quantization\s+(\S+)', perf_command)
    if m: args += ['--quantization', m.group(1)]
    return args


def compute_max_model_len(perf_command: str) -> int:
    """Compute max_model_len from perf_command's input+output lengths.
    Defaults to 4096 if no length flags found.
    """
    in_len = 0
    out_len = 0
    for pat in [r'--random-input-len[\s=](\d+)', r'(?<!-random)--input-len[\s=](\d+)',
                r'--sharegpt-output-len[\s=](\d+)']:
        m = re.search(pat, perf_command)
        if m: in_len = max(in_len, int(m.group(1)))
    for pat in [r'--random-output-len[\s=](\d+)', r'(?<!-random)--output-len[\s=](\d+)']:
        m = re.search(pat, perf_command)
        if m: out_len = max(out_len, int(m.group(1)))
    needed = in_len + out_len + 256  # padding for prompt overhead
    return max(needed, 4096)


def wait_for_server(port: int, proc: subprocess.Popen, timeout: int = 600, log=print) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            log(f"  Server died early (exit {proc.returncode})")
            return False
        try:
            s = socket.socket()
            s.settimeout(2)
            s.connect(("127.0.0.1", port))
            s.close()
            log(f"  Server ready after {int(time.time() - start)}s")
            return True
        except Exception:
            pass
        time.sleep(2)
    log(f"  Server timeout after {timeout}s")
    return False


def fetch_benchmarks_dir(parent_full: str, log) -> Path:
    """Sparse-checkout vllm-project/vllm@<parent>:benchmarks/."""
    bench_dir = Path(f"/tmp/bench_repos/{parent_full[:12]}")
    if (bench_dir / "benchmarks/benchmark_serving.py").exists():
        return bench_dir
    bench_dir.parent.mkdir(parents=True, exist_ok=True)
    if bench_dir.exists():
        shutil.rmtree(bench_dir, ignore_errors=True)
    log(f"  cloning vllm @ {parent_full[:12]} for benchmarks/")
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                    "https://github.com/vllm-project/vllm.git", str(bench_dir)],
                   capture_output=True, text=True, timeout=120)
    subprocess.run(["git", "-C", str(bench_dir), "sparse-checkout", "init", "--cone"],
                   capture_output=True, text=True, timeout=30)
    subprocess.run(["git", "-C", str(bench_dir), "sparse-checkout", "set", "benchmarks"],
                   capture_output=True, text=True, timeout=30)
    subprocess.run(["git", "-C", str(bench_dir), "checkout", parent_full],
                   capture_output=True, text=True, timeout=120)
    return bench_dir


def run_benchmark(venv: Path, model: str, port: int, perf_command: str, parent_full: str, log) -> str:
    """Run the AUTHORITATIVE perf_command from Lossfunk/ISO-Bench, with --host/--port substituted.

    perf_command formats observed:
      `python benchmarks/benchmark_serving.py --model X --dataset-name sharegpt --num-prompts 100`
      `python benchmarks/benchmark_latency.py --model X --tensor-parallel-size 1 --input-len 1000 --batch-size 32`
      `python benchmarks/benchmark_throughput.py --model X ...`
      `python benchmarks/benchmark_prefix_caching.py --model X --output-len 200 --enable-prefix-caching`
      `vllm bench serve --model X ...` (newer CLI; rewrite to old benchmark_serving.py for these old vLLMs)
    """
    py = venv / "bin/python"
    bench_dir = fetch_benchmarks_dir(parent_full, log)

    pc = perf_command.strip()
    # Strip leading env-var prefixes (like VLLM_USE_V1=1) and capture them
    env_overrides = {}
    while True:
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(\S+)\s+(.*)$", pc)
        if not m: break
        env_overrides[m.group(1)] = m.group(2)
        pc = m.group(3)

    # Map `vllm bench serve` -> benchmark_serving.py (old CLI) for compatibility
    pc = re.sub(r"^vllm\s+bench\s+serve", "python benchmarks/benchmark_serving.py", pc)
    # Strip any `python(3)? -m? ` prefix and `benchmarks/...py` to leave only args
    m = re.match(r"^(?:python3?\s+)?(?:-m\s+)?(?:benchmarks/)?(\S+\.py)\s+(.*)$", pc)
    if m:
        script_name = m.group(1).rsplit('/', 1)[-1]  # bench_X.py
        bench_args = m.group(2)
    else:
        log(f"  WARN: could not parse perf_command: {pc[:120]}")
        script_name = "benchmark_serving.py"
        bench_args = pc

    bench_script = bench_dir / "benchmarks" / script_name
    if not bench_script.exists():
        # Try with full path matching
        cand = next(bench_dir.glob(f"benchmarks/{script_name}"), None)
        if cand: bench_script = cand
    if not bench_script.exists():
        log(f"  WARN: {script_name} not at this commit; falling back to benchmark_serving.py")
        bench_script = bench_dir / "benchmarks/benchmark_serving.py"

    # benchmark_prefix_caching.py is in-process (instantiates vllm.LLM directly,
    # no --host/--port). Don't classify it as serving.
    # Rewrite -tp N / --tensor-parallel-size N to match visible GPUs.
    # ISO-Bench perf_commands sometimes specify -tp 4 (Llama-3-70B etc) but
    # we may only have 2 GPUs; vllm.LLM would error with tp greater than
    # device count. Match it down. (-tp does NOT apply to benchmark_serving
    # bench client — that block strips it instead. For latency/throughput
    # benches, the bench script reads tp and passes it to LLM(), so we
    # have to keep a usable value.)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    visible_gpus = max(1, len([x for x in cvd.split(",") if x.strip()]))
    bench_args = re.sub(r'(-tp|--tensor-parallel-size)(\s+|=)\d+',
                        lambda m: f'{m.group(1)}{m.group(2)}{visible_gpus}',
                        bench_args)

    is_serving = "benchmark_serving" in bench_script.name
    # Override --model in bench_args to match what the server is actually serving
    # (after MODEL_OVERRIDES). For server-free benches (latency/throughput) the
    # bench script loads the model itself, so the override applies there too.
    if '--model' in bench_args:
        bench_args = re.sub(r'--model(?:\s+|=)\S+', f'--model {model}', bench_args)
    # Inject --host/--port for serving-style benchmarks
    if is_serving:
        if '--host' not in bench_args:
            bench_args += ' --host 127.0.0.1'
        else:
            bench_args = re.sub(r'--host\s+\S+', '--host 127.0.0.1', bench_args)
        if '--port' not in bench_args:
            bench_args += f' --port {port}'
        else:
            bench_args = re.sub(r'--port\s+\S+', f'--port {port}', bench_args)

    # benchmark_serving.py: inject --dataset-name/--dataset-path when missing.
    # Old vLLM versions only accept sharegpt/sonnet (no 'random'); newer ones
    # accept random. Strategy:
    #   - If --dataset-path is already provided, leave it alone.
    #   - If --random-input-len is present without --dataset-name, the user
    #     wants synthetic prompts — set `--dataset-name random` so the bench
    #     does not fall back to sharegpt and demand a path.
    #   - If --input-len (no --random- prefix) is present, the bench is using
    #     a different synthetic mode (very old style) — leave it alone.
    #   - If --dataset-name=sharegpt is set without path, inject local path.
    #   - Else if --dataset-name is missing entirely: inject sharegpt + path.
    # benchmark_throughput.py at older vLLMs uses `--dataset PATH` rather than
    # `--dataset-name sharegpt`. Rewrite when targeting old throughput bench.
    if "benchmark_throughput" in bench_script.name:
        if '--dataset-name sharegpt' in bench_args or '--dataset-name=sharegpt' in bench_args:
            bench_args = re.sub(r'--dataset-name(?:\s+|=)sharegpt',
                                f'--dataset {SHAREGPT_PATH}', bench_args)
        bench_args = re.sub(r'--dataset-name(?:\s+|=)random', '', bench_args)

    # Some Lossfunk perf_commands use the older `--input-len` / `--output-len`
    # flags. Modern benchmark_serving.py renamed these to `--random-input-len`
    # / `--random-output-len` (and requires `--dataset-name random`). Rewrite.
    if "benchmark_serving" in bench_script.name:
        # Match `--input-len N` but not `--random-input-len N`
        bench_args = re.sub(r'(?<!-random)--input-len(\s+|=)(\d+)',
                            r'--random-input-len\1\2', bench_args)
        bench_args = re.sub(r'(?<!-random)--output-len(\s+|=)(\d+)',
                            r'--random-output-len\1\2', bench_args)
        # Default --random-range-ratio is 1.0 which spreads prompt lengths
        # over [0, 2*N]. Force fixed length so max-model-len bound is tight.
        if '--random-input-len' in bench_args and '--random-range-ratio' not in bench_args:
            bench_args += ' --random-range-ratio 0'

    if "benchmark_serving" in bench_script.name:
        has_path = '--dataset-path' in bench_args
        has_synth_inlen = re.search(r'(?<!--random)--input-len\b', bench_args) is not None
        has_random_inlen = '--random-input-len' in bench_args
        has_ds_name = '--dataset-name' in bench_args
        if not has_path:
            if has_random_inlen and not has_ds_name:
                bench_args += ' --dataset-name random'
            elif not has_synth_inlen and not has_random_inlen:
                if not has_ds_name:
                    bench_args += f' --dataset-name sharegpt --dataset-path {SHAREGPT_PATH}'
                elif '--dataset-name sharegpt' in bench_args or '--dataset-name=sharegpt' in bench_args:
                    bench_args += f' --dataset-path {SHAREGPT_PATH}'
        else:
            # If a relative dataset path is given (e.g. `ShareGPT_V3...json`), rewrite to local
            m = re.search(r'--dataset-path\s+(\S+)', bench_args)
            if m and not os.path.isabs(m.group(1)):
                bench_args = re.sub(r'--dataset-path\s+\S+', f'--dataset-path {SHAREGPT_PATH}', bench_args)

    # Strip server-only flags that some Lossfunk perf_commands accidentally
    # include in the bench client invocation. These belong on the server CLI
    # (vllm.entrypoints.openai.api_server) not on benchmark_serving.py and
    # cause `unrecognized arguments` errors.
    if "benchmark_serving" in bench_script.name:
        bench_args = re.sub(r"--speculative-model\s+'?\S+'?", '', bench_args)
        bench_args = re.sub(r"--num-speculative-tokens\s+\d+", '', bench_args)
        bench_args = re.sub(r"--enable-prefix-caching\b", '', bench_args)
        bench_args = re.sub(r"--enable-chunked-prefill\b", '', bench_args)
        bench_args = re.sub(r"--enforce-eager\b", '', bench_args)
        bench_args = re.sub(r"--load-format\s+\S+", '', bench_args)
        bench_args = re.sub(r"--max-model-len\s+\d+", '', bench_args)
        bench_args = re.sub(r"--gpu-memory-utilization\s+[\d.]+", '', bench_args)
        bench_args = re.sub(r"--guided-decoding-ratio\s+[\d.]+", '', bench_args)
        bench_args = re.sub(r"--guided-decoding-backend\s+\S+", '', bench_args)
        bench_args = re.sub(r"--dtype\s+\S+", '', bench_args)
        bench_args = re.sub(r"--seed\s+\d+", '', bench_args)
        bench_args = re.sub(r"--quantization\s+\S+", '', bench_args)
        bench_args = re.sub(r"--served-model-name\s+\S+", '', bench_args)
        bench_args = re.sub(r"--tensor-parallel-size\s+\d+", '', bench_args)
        bench_args = re.sub(r"-tp\s+\d+", '', bench_args)
        bench_args = re.sub(r"--max-num-batched-tokens\s+\d+", '', bench_args)
        bench_args = re.sub(r"--max-num-seqs\s+\d+", '', bench_args)
        bench_args = re.sub(r"--block-size\s+\d+", '', bench_args)
        bench_args = re.sub(r"--swap-space\s+\d+", '', bench_args)
        bench_args = re.sub(r"--use-v2-block-manager\b", '', bench_args)
        bench_args = re.sub(r"--num-scheduler-steps\s+\d+", '', bench_args)
        bench_args = re.sub(r"\s+", ' ', bench_args).strip()

    # Cap --num-prompts to keep wall-clock manageable. Lossfunk specifies up to
    # 2048 for some commits and the bench script defaults to 1000. 100 is
    # enough for stable TTFT/TPOT distributions, and a busted agent patch
    # that hangs tokens stays bounded by 100*per-token-timeout instead of 1000.
    NUM_PROMPTS_CAP = 100
    if "benchmark_serving" in bench_script.name:
        m_np = re.search(r'--num-prompts\s+(\d+)', bench_args)
        if m_np:
            if int(m_np.group(1)) > NUM_PROMPTS_CAP:
                bench_args = re.sub(r'--num-prompts\s+\d+', f'--num-prompts {NUM_PROMPTS_CAP}', bench_args)
        else:
            bench_args += f' --num-prompts {NUM_PROMPTS_CAP}'

    cmd_str = f"{py} {bench_script} {bench_args}"
    log(f"  Running: {bench_script.name} {bench_args[:120]}")
    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE)
    # Propagate HF token to bench subprocess; HF_HOME redirects token lookup
    # away from ~/.cache/huggingface so we have to set HF_TOKEN explicitly for
    # server-free benches that load gated models (Llama-3-70B etc).
    tok = get_hf_token(model)
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    env.update(env_overrides)
    # cwd=/tmp to avoid project-root vllm/ directory shadowing the venv's vllm.
    r = subprocess.run(cmd_str, shell=True, capture_output=True, text=True,
                       timeout=1500, env=env, cwd="/tmp")
    return r.stdout + r.stderr


def parse_metrics(output: str) -> dict:
    metrics = {}
    # benchmark_serving.py format
    serving = {
        'ttft_mean_ms':              r'Mean TTFT[^:]*:\s*([\d.]+)',
        'ttft_median_ms':            r'Median TTFT[^:]*:\s*([\d.]+)',
        'ttft_p99_ms':               r'P99 TTFT[^:]*:\s*([\d.]+)',
        'tpot_mean_ms':              r'Mean TPOT[^:]*:\s*([\d.]+)',
        'tpot_median_ms':            r'Median TPOT[^:]*:\s*([\d.]+)',
        'tpot_p99_ms':               r'P99 TPOT[^:]*:\s*([\d.]+)',
        'itl_mean_ms':               r'Mean ITL[^:]*:\s*([\d.]+)',
        'itl_median_ms':             r'Median ITL[^:]*:\s*([\d.]+)',
        'itl_p99_ms':                r'P99 ITL[^:]*:\s*([\d.]+)',
        'output_token_throughput_tok_s': r'Output token throughput[^:]*:\s*([\d.]+)',
        'total_token_throughput_tok_s':  r'Total token throughput[^:]*:\s*([\d.]+)',
        'request_throughput_req_s':      r'Request throughput[^:]*:\s*([\d.]+)',
    }
    # benchmark_latency.py format: "Avg latency: X seconds", "10% percentile latency: X seconds"
    latency = {
        'avg_latency_s':             r'Avg latency:\s*([\d.]+)\s*seconds',
        'p10_latency_s':             r'10% percentile latency:\s*([\d.]+)',
        'p25_latency_s':             r'25% percentile latency:\s*([\d.]+)',
        'p50_latency_s':             r'50% percentile latency:\s*([\d.]+)',
        'p75_latency_s':             r'75% percentile latency:\s*([\d.]+)',
        'p90_latency_s':             r'90% percentile latency:\s*([\d.]+)',
        'p99_latency_s':             r'99% percentile latency:\s*([\d.]+)',
    }
    # benchmark_throughput.py: "Throughput: X requests/s, Y total tokens/s, Z output tokens/s"
    throughput = {
        'throughput_req_s':              r'Throughput:\s*([\d.]+)\s*requests/s',
        'throughput_total_tok_s':        r'requests/s,\s*([\d.]+)\s*total tokens/s',
        'throughput_output_tok_s':       r'total tokens/s,\s*([\d.]+)\s*output tokens/s',
    }
    # benchmark_prefix_caching.py: emits "cost time X" timings (older format)
    prefix = {
        'prefix_cache_cost_time_s':      r'cost time\s*([\d.]+)',
    }
    for group in (serving, latency, throughput, prefix):
        for k, pat in group.items():
            m = re.search(pat, output)
            if m:
                metrics[k] = float(m.group(1))
    return metrics


def run_one(commit_short: str, info: dict, timeout: int = 1800) -> dict:
    start = time.time()
    out_lines = []

    def log(msg):
        out_lines.append(msg)
        print(msg, flush=True)

    # ISO-Bench dataset is authoritative for perf_command + model. Merge before
    # downstream lookups so overrides are captured in raw_output for audit.
    info = merge_with_iso_bench(commit_short, info, log)
    parent_full = info.get("parent_commit") or info.get("human_commit_full", "")
    model = MODEL_OVERRIDES.get(info["model"], info["model"])
    if model != info["model"]:
        log(f"  Model override: {info['model']} -> {model}")
    perf = info.get("perf_command") or ""
    # Per-commit dep override applied via env so setup_venv picks it up
    if info.get("transformers_pin"):
        os.environ["TRANSFORMERS_PIN"] = info["transformers_pin"]
        log(f"  Per-commit transformers pin: {info['transformers_pin']}")
    if info.get("pyarrow_pin"):
        os.environ["PYARROW_PIN"] = info["pyarrow_pin"]
        log(f"  Per-commit pyarrow pin: {info['pyarrow_pin']}")
    else:
        os.environ.pop("PYARROW_PIN", None)
    if info.get("datasets_pin"):
        os.environ["DATASETS_PIN"] = info["datasets_pin"]
        log(f"  Per-commit datasets pin: {info['datasets_pin']}")
    else:
        os.environ.pop("DATASETS_PIN", None)
    # Always derive patch path from the active PATCHES_DIR. The `patch_path`
    # field in vllm_oh_mapping.json is hardcoded to the sonnet45 layout (legacy
    # convenience field) — using it would silently benchmark the wrong agent's
    # patch when --agent-name is anything else.
    task_dir = find_task_dir(commit_short)
    if not task_dir:
        return {"status": "error",
                "error": f"no task dir in {PATCHES_DIR} matches commit_short={commit_short}",
                "duration_s": 0, "metrics": {}, "raw_output": ""}
    patch_path = PATCHES_DIR / task_dir / "model_patch.diff"
    if not patch_path.exists():
        return {"status": "error", "error": f"patch not found at {patch_path}",
                "duration_s": 0, "metrics": {}, "raw_output": ""}
    if patch_path.stat().st_size == 0:
        return {"status": "error", "error": f"empty patch at {patch_path} (agent gave up)",
                "duration_s": 0, "metrics": {}, "raw_output": ""}

    use_source = not wheel_exists(parent_full)
    if use_source:
        log(f"  No wheel for {parent_full[:12]} — will build from source")

    # Decide if benchmark needs a server (serving / prefix_caching) or is server-free (latency / throughput).
    # prefix_caching is in-process (loads vllm.LLM directly); never start a
    # server for it even if the mapping's stale `has_serving: True` says so.
    # Only `benchmark_serving.py` and `vllm bench serve` need a server.
    if "prefix_caching" in perf or "benchmark_latency" in perf or "benchmark_throughput" in perf:
        needs_server = False
    else:
        needs_server = (info.get("has_serving") or
                        "benchmark_serving" in perf or
                        "vllm bench serve" in perf)
    log(f"  needs_server={needs_server} (perf_command first 80: {perf[:80]})")

    # Always sweep GPU orphans before starting a new commit. Without this,
    # the previous commit's spawn-children (now PPID=1) keep ~70 GiB of GPU
    # memory and the next commit OOMs at model-load time.
    kill_gpu_orphans(log)
    port = free_port()
    proc = None
    try:
        if use_source:
            pypi_v = info.get("vllm_pypi_version", "")
            outl = info.get("outlines_pin", "==0.0.46")
            venv = setup_venv_overlay(commit_short, parent_full, log,
                                      pypi_version=pypi_v, outlines_pin=outl)
            # The overlay path already installed pinned deps for the older
            # vllm; do NOT re-install transformers/datasets here (would
            # pull newer outlines that's incompatible with vllm 0.5.x).
        else:
            venv = setup_venv(commit_short, parent_full, log)
            # Install bench deps (always needed; not pulled by vllm wheel)
            env = os.environ.copy()
            pin = os.environ.get("TRANSFORMERS_PIN", ">=4.45,<4.47")
            pa_pin = os.environ.get("PYARROW_PIN", "")
            ds_pin = os.environ.get("DATASETS_PIN", "")
            datasets_spec = f"datasets{ds_pin}" if ds_pin else "datasets"
            pyarrow_spec = f"pyarrow{pa_pin}" if pa_pin else None
            log(f"  Installing benchmark deps + transformers{pin}"
                + (f" + pyarrow{pa_pin}" if pa_pin else "")
                + (f" + {datasets_spec}" if ds_pin else ""))
            install_cmd = [str(UV_BIN), "pip", "install", "--python", str(venv / "bin/python"),
                           "aiohttp", "pandas", datasets_spec, "pillow",
                           f"transformers{pin}"]
            if pyarrow_spec:
                install_cmd.append(pyarrow_spec)
            subprocess.run(install_cmd, capture_output=True, text=True, timeout=300, env=env)
        apply_patch(venv, patch_path, log)
        if needs_server:
            kill_port(port, log)  # ensure prior commit's spawn-children are gone
            mml = compute_max_model_len(perf)
            extra = extract_server_flags_from_perf(perf)
            proc = start_server(venv, model, port, log, max_model_len=mml,
                                extra_server_args=extra)
            if not wait_for_server(port, proc, timeout=600, log=log):
                stdout = ""
                try:
                    proc.terminate()
                    stdout = proc.communicate(timeout=10)[0] or ""
                except Exception:
                    pass
                return {"status": "error", "error": "Server startup timeout / crash",
                        "duration_s": time.time() - start,
                        "metrics": {},
                        "raw_output": "\n".join(out_lines[-50:]) + "\n--- server stdout ---\n" + stdout[-20000:]}
        bench_out = run_benchmark(venv, model, port, perf, parent_full, log)
        metrics = parse_metrics(bench_out)
        if not metrics:
            return {"status": "error", "error": "No metrics in benchmark output",
                    "duration_s": time.time() - start,
                    "metrics": {},
                    "raw_output": "\n".join(out_lines[-50:]) + "\n--- bench ---\n" + bench_out[-20000:]}
        return {"status": "success", "metrics": metrics,
                "duration_s": time.time() - start,
                "raw_output": "\n".join(out_lines[-30:]) + "\n--- bench tail ---\n" + bench_out[-10000:]}
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=10)
            except Exception: proc.kill()
        kill_port(port, log)  # final sweep for spawn children that outlive parent


def find_task_dir(commit_short: str) -> str:
    """Find the vllm_core-NNNN dir that matches a human commit short."""
    for d in sorted(PATCHES_DIR.iterdir()):
        rs = d / "run_summary.json"
        if not rs.exists(): continue
        j = json.loads(rs.read_text())
        if j["commits"]["human"][:8] == commit_short:
            return d.name
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commits", nargs="+", required=True)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--agent-name", default="openhands_sonnet45",
                   choices=sorted(AGENT_PROFILES.keys()),
                   help="Selects PATCHES_DIR and RESULTS_DIR (relative to project root).")
    p.add_argument("--root", type=Path, default=None,
                   help="Override project root (default: derived from script location).")
    p.add_argument("--patches-dir", type=Path, default=None,
                   help="Override patches dir (absolute). Overrides --agent-name's choice.")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Override results dir (absolute). Overrides --agent-name's choice.")
    p.add_argument("--mapping-file", type=Path, default=None,
                   help="Override path to vllm_oh_mapping.json.")
    args = p.parse_args()

    # Resolve paths and rebind module globals so the rest of the script (which
    # reads ROOT/PATCHES_DIR/RESULTS_DIR/MAPPING_FILE/SHAREGPT_PATH at the
    # module level via name lookup) sees the chosen values.
    global ROOT, MAPPING_FILE, RESULTS_DIR, PATCHES_DIR, SHAREGPT_PATH
    if args.root:
        ROOT = args.root.resolve()
        SHAREGPT_PATH = str(ROOT / "data/archive/sharegpt_dataset.json")
    patches_rel, results_rel = AGENT_PROFILES[args.agent_name]
    PATCHES_DIR = (args.patches_dir or (ROOT / patches_rel)).resolve()
    RESULTS_DIR = (args.results_dir or (ROOT / results_rel)).resolve()
    MAPPING_FILE = (args.mapping_file or (ROOT / "data/mappings/vllm_oh_mapping.json")).resolve()

    print(f"  ROOT          = {ROOT}")
    print(f"  PATCHES_DIR   = {PATCHES_DIR}")
    print(f"  RESULTS_DIR   = {RESULTS_DIR}")
    print(f"  MAPPING_FILE  = {MAPPING_FILE}")
    print(f"  HF_CACHE      = {HF_CACHE}")
    print(f"  UV_BIN        = {UV_BIN}")
    print(f"  SHAREGPT_PATH = {SHAREGPT_PATH}")

    if not MAPPING_FILE.exists():
        sys.exit(f"ERROR: mapping file not found at {MAPPING_FILE}")
    if not PATCHES_DIR.exists():
        sys.exit(f"ERROR: patches dir not found at {PATCHES_DIR}")
    if not UV_BIN.exists():
        sys.exit(f"ERROR: uv not found at {UV_BIN}; set UV_BIN env or install uv")

    mp = json.loads(MAPPING_FILE.read_text())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Native vLLM benchmark — {len(args.commits)} commits ===")
    for i, c in enumerate(args.commits, 1):
        print(f"\n{'='*70}\n[{i}/{len(args.commits)}] {c}\n{'='*70}")
        info = mp.get(c)
        if not info:
            print(f"  SKIP: {c} not in mapping")
            continue
        out_file = RESULTS_DIR / f"{c}_agent_result.json"
        if out_file.exists():
            print(f"  SKIP: result already exists")
            continue
        try:
            result = run_one(c, info, args.timeout)
        except Exception as e:
            import traceback
            result = {"status": "error",
                      "error": f"unhandled: {type(e).__name__}: {e}",
                      "duration_s": 0,
                      "metrics": {},
                      "raw_output": traceback.format_exc()}
        # Re-merge for record-keeping so model + perf_command in the JSON
        # match what was *actually* benchmarked (post-ISO override).
        info_for_record = merge_with_iso_bench(c, dict(info), lambda _msg: None)
        record = {
            "human_commit": c,
            "human_commit_full": info.get("human_commit_full", info.get("commit_full", "")),
            "parent_commit": info.get("parent_commit", ""),
            "model": MODEL_OVERRIDES.get(info_for_record["model"], info_for_record["model"]),
            "perf_command": info_for_record.get("perf_command", ""),
            **result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runner": "native",
            "iso_bench_override_applied": (info_for_record.get("perf_command") != info.get("perf_command")
                                            or info_for_record.get("model") != info.get("model")),
        }
        out_file.write_text(json.dumps(record, indent=2))
        s = result.get("status")
        if s == "success":
            tp = result["metrics"].get("output_token_throughput_tok_s", "?")
            print(f"  AGENT SUCCESS: {tp} tok/s")
        else:
            print(f"  AGENT FAILED: {result.get('error')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
