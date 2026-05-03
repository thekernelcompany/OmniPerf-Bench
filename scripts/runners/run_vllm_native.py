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

ROOT = Path("/root/OmniPerf-Bench")
MAPPING_FILE = ROOT / "data/mappings/vllm_oh_mapping.json"  # Lossfunk-derived authoritative
RESULTS_DIR = ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45/results"
PATCHES_DIR = ROOT / "ISO-Bench/state/runs/vllm/openhands_sonnet45/flat"
VENV_ROOT = Path("/tmp/native_venvs")
HF_CACHE = Path("/ephemeral/huggingface_cache")
UV_BIN = Path("/root/.local/bin/uv")

WHEEL_URL_TEMPLATE = "https://wheels.vllm.ai/{commit}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

# benchmark_serving.py needs --dataset-path for sharegpt/sonnet. The local copy
# avoids re-downloading per-commit (and some old versions accept no --dataset-name
# default but still need a path).
SHAREGPT_PATH = "/root/OmniPerf-Bench/data/archive/sharegpt_dataset.json"

MODEL_OVERRIDES = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "ibm-ai-platform/Bamba-9B-v2": "meta-llama/Meta-Llama-3-8B-Instruct",
    "ibm-ai-platform/Bamba-9B": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def get_hf_token() -> str:
    p = Path.home() / ".cache/huggingface/token"
    return p.read_text().strip() if p.exists() else ""


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
    env["HF_TOKEN"] = get_hf_token()
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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
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

    is_serving = "benchmark_serving" in bench_script.name or "prefix_caching" in bench_script.name
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
    tok = get_hf_token()
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    env.update(env_overrides)
    r = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, timeout=1500, env=env)
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

    parent_full = info.get("parent_commit") or info.get("human_commit_full", "")
    model = MODEL_OVERRIDES.get(info["model"], info["model"])
    if model != info["model"]:
        log(f"  Model override: {info['model']} -> {model}")
    perf = info.get("perf_command") or ""
    # Per-commit dep override applied via env so setup_venv picks it up
    if info.get("transformers_pin"):
        os.environ["TRANSFORMERS_PIN"] = info["transformers_pin"]
        log(f"  Per-commit transformers pin: {info['transformers_pin']}")
    patch_path = Path(info.get("patch_path") or PATCHES_DIR / find_task_dir(commit_short) / "model_patch.diff")
    if not patch_path.exists():
        return {"status": "error", "error": f"patch not found at {patch_path}",
                "duration_s": 0, "metrics": {}, "raw_output": ""}

    use_source = not wheel_exists(parent_full)
    if use_source:
        log(f"  No wheel for {parent_full[:12]} — will build from source")

    # Decide if benchmark needs a server (serving / prefix_caching) or is server-free (latency / throughput).
    needs_server = info.get("has_serving") or "benchmark_serving" in perf or "prefix_caching" in perf or "vllm bench serve" in perf
    log(f"  needs_server={needs_server} (perf_command first 80: {perf[:80]})")

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
            log(f"  Installing benchmark deps + transformers{pin}")
            subprocess.run(
                [str(UV_BIN), "pip", "install", "--python", str(venv / "bin/python"),
                 "aiohttp", "pandas", "datasets", "pillow",
                 f"transformers{pin}"],
                capture_output=True, text=True, timeout=300, env=env,
            )
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
    args = p.parse_args()

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
        record = {
            "human_commit": c,
            "human_commit_full": info.get("commit_full", ""),
            "parent_commit": info.get("parent_commit", ""),
            "model": MODEL_OVERRIDES.get(info["model"], info["model"]),
            "perf_command": info.get("perf_command", ""),
            **result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runner": "native",
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
