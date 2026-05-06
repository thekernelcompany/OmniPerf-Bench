#!/usr/bin/env python3
"""Re-bench OH (Sonnet 4.5, GPT-5) on the 3 Bamba commits via OVERLAY approach.

Overlay = PyPI vllm install matching parent's release + agent patch on site-packages.
No compile. Using the last vllm release tag that's an ancestor of each parent SHA:

    parent (date)         | vllm release used
    25373b6c6 (Aug 2025)  | v0.10.0
    270a5da49 (Mar 14)    | v0.7.3
    0032903a5 (Mar 20)    | v0.8.1

Per cell:
  1. cp -a base_<parent_short>/  →  cell_<commit>_<agent>/   (per-cell venv)
  2. patch -p1 --force on cell venv's site-packages/         (apply agent patch)
  3. start vllm api_server (canonical Bamba model, --enforce-eager, tp=1)
  4. run benchmarks/benchmark_serving.py  with canonical perf_command args
  5. parse metrics, write JSON to runner-side + submodule-side

Result file shape mirrors run_vllm_native.py output for downstream merge compat.
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
from pathlib import Path

ROOT     = Path("/home/ubuntu/OmniPerf-Bench")
SUB      = ROOT / "third-party/everything_analysis_data"
SHAREGPT = ROOT / "data/archive/sharegpt_dataset.json"
HF_TOKEN = "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"
HF_CACHE = Path("/home/ubuntu/.cache/huggingface")
VENVS    = Path("/tmp/native_venvs")
UV_BIN   = Path(os.path.expanduser("~/.local/bin/uv"))

# For "agent" results, we used the FIRST containing release (clean for patch apply).
# For "human" benches & for the suspicious-commit batch, we use the PRE-PARENT
# release so the human's diff isn't already merged into the base.
# (commit, agent, parent_short, vllm_pypi_version, task_dir)
CELLS = [
    # 3 Bamba commits — agent already benched on first-containing tag (kept as-is).
    ("b690e348", "sonnet45", "25373b6c", "0.10.0", "vllm_core-0064"),
    ("fe66b347", "sonnet45", "270a5da4", "0.8.0",  "vllm_core-0095"),
    ("fe66b347", "gpt5",     "270a5da4", "0.8.0",  "vllm_core-0095"),
    ("296f927f", "sonnet45", "0032903a", "0.8.2",  "vllm_core-0008"),
    ("296f927f", "gpt5",     "0032903a", "0.8.2",  "vllm_core-0008"),
    # b690e348 human: v0.10.0 IS pre-parent (Aug 2 commit, Jul 24 release) — clean.
    # fe66b347/296f927f human: pre-parent releases for Option B re-bench.
    ("fe66b347", "sonnet45_pre", "270a5da4", "0.7.3", "vllm_core-0095"),
    ("296f927f", "sonnet45_pre", "0032903a", "0.8.1", "vllm_core-0008"),
    # 9 new commits (suspicious-row table). Agent + human via pre-parent overlay.
    ("bc7c4d20", "sonnet45", "f67e9e9f", "0.8.4",       "AUTO"),  # Llama-3.1-8B
    ("3476ed08", "sonnet45", "54600709", "0.5.0.post1", "AUTO"),  # Llama-2-7B
    ("b55ed6ef", "sonnet45", "2f385183", "0.6.6.post1", "AUTO"),  # Llama-3.1-8B
    ("30172b49", "sonnet45", "a4d577b3", "0.7.2",       "AUTO"),  # Llama-3.1-8B
    ("299ebb62", "sonnet45", "f728ab8e", "0.8.4",       "AUTO"),  # Qwen2.5-1.5B  (vllm bench serve CLI)
    ("70b808fe", "sonnet45", "63d635d1", "0.7.3",       "AUTO"),  # Qwen2-VL-7B   (random dataset)
    ("a3223766", "sonnet45", "bc8a8ce5", "0.10.0",      "AUTO"),  # opt-125m       (vllm bench serve)
    ("d7740ea4", "sonnet45", "cc466a32", "0.4.2",       "AUTO"),  # Llama-2-7B   (benchmark_throughput.py)
    ("19d98e0c", "sonnet45", "2b04c209", "0.7.3",       "AUTO"),  # DeepSeek-Coder-V2-Lite (16B MoE)
    ("99abb8b6", "sonnet45", "3a1e6481", "0.8.0",       "AUTO"),  # Llama-3.1 spec-decode
    ("98f47f2a", "sonnet45", "8c1e77fb", "0.6.4.post1", "AUTO"),  # opt-125m latency-mode
]

# Per-vllm-version pinned deps. Recent vllm releases use lower-bound-only
# transformers/tokenizers constraints, but latest transformers (4.57+) removed
# `all_special_tokens_extended` from TokenizersBackend, breaking these vllms.
# Pin to versions contemporaneous with each vllm release.
DEP_PINS = {
    "0.10.0":      ["transformers==4.53.2", "tokenizers==0.21.1"],
    "0.8.0":       ["transformers==4.49.0", "tokenizers==0.21.0"],
    "0.8.1":       ["transformers==4.49.0", "tokenizers==0.21.0"],
    "0.8.2":       ["transformers==4.50.3", "tokenizers==0.21.1"],
    "0.8.4":       ["transformers==4.51.0", "tokenizers==0.21.1"],
    # transformers >= 4.50 introduces TokenizersBackend wrapper which removes
    # `all_special_tokens_extended` attribute. Old vllm code paths require it,
    # so pin to <4.50 for any vllm release that still hits get_cached_tokenizer.
    "0.7.2":       ["transformers==4.49.0", "tokenizers==0.21.0", "outlines==0.1.11"],
    "0.7.3":       ["transformers==4.49.0", "tokenizers==0.21.0", "outlines==0.1.11"],
    "0.6.6.post1": ["transformers==4.46.0", "tokenizers==0.20.3", "outlines==0.0.46", "lm-format-enforcer==0.10.6"],
    # vllm 0.6.4.post1 needs mllama (transformers >= 4.45) but GPT2Tokenizer regressed in 4.46+
    "0.6.4.post1": ["transformers==4.45.2", "tokenizers==0.20.0", "outlines==0.0.46", "lm-format-enforcer==0.10.6"],
    "0.5.0.post1": ["transformers==4.42.0", "tokenizers==0.19.0", "outlines==0.0.46", "lm-format-enforcer==0.10.1"],
    "0.4.2":       ["transformers==4.40.0", "tokenizers==0.19.0", "outlines==0.0.46", "lm-format-enforcer==0.10.1"],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_task_dir(commit_short: str, agent: str) -> str:
    """Resolve task_dir at runtime when set to AUTO."""
    base = ROOT / f"ISO-Bench/state/runs/vllm/openhands_{agent}/flat"
    for d in sorted(base.iterdir()):
        rs = d / "run_summary.json"
        if not rs.exists(): continue
        j = json.loads(rs.read_text())
        if j.get("commits", {}).get("human", "")[:8] == commit_short:
            return d.name
    return ""


def patch_path(commit, agent, task_dir):
    if task_dir == "AUTO":
        # Reverse-lookup the task dir from run_summary.json files.
        # Strip _pre suffix for human-side cells that share the agent's patch dir.
        agent_clean = agent.replace("_pre", "")
        task_dir = find_task_dir(commit, agent_clean)
        if not task_dir:
            return ROOT / f"ISO-Bench/state/runs/vllm/openhands_{agent_clean}/flat/MISSING/{commit}"
        return ROOT / f"ISO-Bench/state/runs/vllm/openhands_{agent_clean}/flat/{task_dir}/model_patch.diff"
    agent_clean = agent.replace("_pre", "")
    return ROOT / f"ISO-Bench/state/runs/vllm/openhands_{agent_clean}/flat/{task_dir}/model_patch.diff"


def runner_result_path(commit, agent, role="agent"):
    suffix = f"{role}_result"
    return ROOT / f"archive/results/2026-05/iso_bench_results_3way_openhands_{agent}/results/{commit}_{suffix}.json"


def submodule_result_path(commit, agent, role="agent"):
    suffix = f"{role}_result"
    return SUB / f"iso-bench-openhands-{agent}-hard-metrics/results_vllm/results/{commit}_{suffix}.json"


def read_mapping(commit):
    mp = json.loads((ROOT / "data/mappings/vllm_oh_mapping.json").read_text())
    info = mp[commit]
    return {
        "parent_full":  info.get("parent_commit") or info.get("human_commit_full", ""),
        "perf_command": info.get("perf_command") or "",
        "model":        info["model"],
        "human_full":   info.get("human_commit_full") or info.get("commit_full", ""),
    }


def python_for_vllm(vllm_version: str) -> str:
    """Older vllm (0.4-0.6) only ships cp38–cp311 wheels; pick 3.11. Newer use 3.12."""
    major_minor = ".".join(vllm_version.split(".")[:2])
    if major_minor in ("0.4", "0.5", "0.6"):
        return "3.11"
    return "3.12"


def setup_base_venv(parent_short: str, vllm_version: str) -> Path:
    """uv venv + uv pip install vllm==<version>. Returns venv path."""
    venv = VENVS / f"base_{parent_short}_v{vllm_version.replace('.', '_')}"
    py_ver = python_for_vllm(vllm_version)
    py_glob = next((venv / "lib").glob(f"python{py_ver}/site-packages/vllm/__init__.py"), None) if venv.exists() else None
    if py_glob and py_glob.exists():
        log(f"  base venv exists: {venv}")
        return venv
    if venv.exists():
        shutil.rmtree(venv)
    venv.parent.mkdir(parents=True, exist_ok=True)
    log(f"  creating venv {venv} (python {py_ver})")
    subprocess.run([str(UV_BIN), "venv", "--python", py_ver, str(venv)],
                   check=True, capture_output=True, text=True, timeout=120)
    py = venv / "bin/python"
    log(f"  uv pip install vllm=={vllm_version}")
    env = os.environ.copy()
    env["UV_SKIP_WHEEL_FILENAME_CHECK"] = "1"
    r = subprocess.run(
        [str(UV_BIN), "pip", "install", "--python", str(py),
         f"vllm=={vllm_version}",
         "--extra-index-url", "https://download.pytorch.org/whl/cu128",
         "--index-strategy", "unsafe-best-match"],
        capture_output=True, text=True, timeout=900, env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(f"vllm=={vllm_version} install failed:\n{(r.stderr or r.stdout)[-2000:]}")
    # benchmark_serving.py needs pandas (via benchmark_dataset.py), aiohttp,
    # and datasets. Also install setuptools — older vllm's triton imports it
    # at runtime. Install into base venv so all cell venvs inherit them.
    log(f"  installing bench client deps")
    rb = subprocess.run(
        [str(UV_BIN), "pip", "install", "--python", str(py),
         "pandas", "aiohttp", "datasets", "pyyaml", "setuptools",
         "--index-strategy", "unsafe-best-match"],
        capture_output=True, text=True, timeout=300, env=env,
    )
    if rb.returncode != 0:
        log(f"  bench-deps install warning: {(rb.stderr or rb.stdout)[-400:]}")

    # Pin transformers/tokenizers to versions vllm was actually tested against,
    # because vllm's lower-bound-only constraints let pip install too-new versions
    # whose API changes break vllm at runtime (e.g. `all_special_tokens_extended`
    # removed from TokenizersBackend in transformers >=4.57).
    pins = DEP_PINS.get(vllm_version, [])
    if pins:
        log(f"  pinning compatibility deps: {pins}")
        rp = subprocess.run(
            [str(UV_BIN), "pip", "install", "--python", str(py)] + pins
            + ["--index-strategy", "unsafe-best-match"],
            capture_output=True, text=True, timeout=300, env=env,
        )
        if rp.returncode != 0:
            log(f"  pin install warning: {(rp.stderr or rp.stdout)[-400:]}")
    # outlines<1.0 imports `from pyairports.airports import AIRPORT_LIST` at
    # startup; the pinned pyairports==0.0.1 wheel ships an empty module. Stub
    # it. (Same workaround the production runner's setup_venv_overlay applies.)
    site_pkgs = next((venv / "lib").glob("python3.*/site-packages"))
    pa = site_pkgs / "pyairports"
    if not pa.exists():
        pa.mkdir(parents=True, exist_ok=True)
        (pa / "__init__.py").write_text("")
        (pa / "airports.py").write_text(
            "AIRPORT_LIST = []\n"
            "class Airport: pass\n"
            "class AirportNotFoundException(Exception): pass\n"
        )
        log(f"  stubbed pyairports module")
    # Sanity check the install
    sanity = subprocess.run(
        [str(py), "-c", "import vllm; from vllm import _C; print(vllm.__version__)"],
        capture_output=True, text=True, timeout=30,
    )
    log(f"  base sanity: rc={sanity.returncode} out={sanity.stdout.strip()} err={sanity.stderr[-300:].strip()}")
    return venv


def make_cell_venv(base: Path, commit: str, agent: str) -> Path:
    cell = VENVS / f"cell_{commit}_{agent}"
    if cell.exists():
        shutil.rmtree(cell)
    log(f"  cp -a {base.name} → {cell.name}")
    subprocess.run(["cp", "-a", str(base), str(cell)], check=True, timeout=180)
    return cell


def apply_overlay(venv: Path, patch: Path) -> dict:
    site = next((venv / "lib").glob("python3.*/site-packages"))
    # Use --merge so patch fuzz/conflict gets recorded as <<<<<< markers in the
    # file rather than rejecting outright. We later check for the markers.
    rej = Path(f"/tmp/{venv.name}.rej")
    if rej.exists(): rej.unlink()
    log(f"  patch -p1 --force in {site}")
    r = subprocess.run(
        ["patch", "-p1", "--force", "--no-backup-if-mismatch", f"--reject-file={rej}"],
        cwd=str(site),
        input=patch.read_text(),
        capture_output=True, text=True, timeout=120,
    )
    log(f"  patch rc={r.returncode}")
    log(f"  patch stdout (last 500): {(r.stdout or '')[-500:].strip()}")
    rejected = []
    if rej.exists() and rej.stat().st_size > 0:
        rej_text = rej.read_text()
        rejected = re.findall(r'^\+\+\+ b/(\S+)', rej_text, re.M)
        log(f"  patch REJECTS in: {rejected[:6]}")
    return {"rc": r.returncode, "rejected_files": rejected, "stdout": (r.stdout or "")[-1000:]}


def kill_port(port: int):
    try:
        r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                           capture_output=True, text=True, timeout=10)
        for pid in set(re.findall(r'pid=(\d+)', r.stdout)):
            try: os.kill(int(pid), 9)
            except ProcessLookupError: pass
    except Exception: pass


def kill_gpu_orphans(cuda_dev: int):
    try:
        r = subprocess.run(
            ["nvidia-smi", "-i", str(cuda_dev),
             "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            parts = [p.strip() for p in line.strip().split(",", 1)]
            if len(parts) == 2 and "/tmp/native_venvs/" in parts[1]:
                log(f"  killing stale GPU PID {parts[0]} ({parts[1]})")
                try: os.kill(int(parts[0]), 9)
                except ProcessLookupError: pass
        time.sleep(1)
    except Exception as e:
        log(f"  kill_gpu_orphans error: {e}")


def fetch_benchmarks_dir(parent_full: str) -> Path:
    """Sparse-checkout vllm@<parent>:benchmarks/ for the canonical bench script."""
    bench_dir = Path(f"/tmp/bench_repos/{parent_full[:12]}")
    if (bench_dir / "benchmarks/benchmark_serving.py").exists():
        return bench_dir
    bench_dir.parent.mkdir(parents=True, exist_ok=True)
    if bench_dir.exists():
        shutil.rmtree(bench_dir, ignore_errors=True)
    log(f"  cloning vllm @ {parent_full[:12]} for benchmarks/")
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                    "https://github.com/vllm-project/vllm.git", str(bench_dir)],
                   capture_output=True, text=True, timeout=180)
    subprocess.run(["git", "-C", str(bench_dir), "sparse-checkout", "init", "--cone"],
                   capture_output=True, text=True, timeout=30)
    subprocess.run(["git", "-C", str(bench_dir), "sparse-checkout", "set", "benchmarks"],
                   capture_output=True, text=True, timeout=30)
    subprocess.run(["git", "-C", str(bench_dir), "checkout", parent_full],
                   capture_output=True, text=True, timeout=180)
    return bench_dir


def extract_server_flags(perf_command: str) -> list:
    """Pull server-side flags out of the perf_command (--dtype, etc.)."""
    args = []
    m = re.search(r'--dtype[\s=](\S+)', perf_command)
    if m: args += ['--dtype', m.group(1)]
    if '--enable-prefix-caching' in perf_command:
        args.append('--enable-prefix-caching')
    if '--enable-chunked-prefill' in perf_command:
        args.append('--enable-chunked-prefill')
    m = re.search(r'--quantization[\s=](\S+)', perf_command)
    if m: args += ['--quantization', m.group(1)]
    # Spec-decode flags (e.g. --speculative-model '[ngram]' --num-speculative-tokens 3)
    m = re.search(r"--speculative-model\s+'?(\[?[^'\s]+\]?)'?", perf_command)
    if m:
        args += ['--speculative-model', m.group(1)]
        # ngram requires --ngram-prompt-lookup-max
        if 'ngram' in m.group(1).lower():
            ng_max = re.search(r'--ngram-prompt-lookup-max[\s=](\d+)', perf_command)
            args += ['--ngram-prompt-lookup-max', ng_max.group(1) if ng_max else '4']
    m = re.search(r'--num-speculative-tokens[\s=](\d+)', perf_command)
    if m: args += ['--num-speculative-tokens', m.group(1)]
    m = re.search(r'--guided-decoding-backend[\s=](\S+)', perf_command)
    if m: args += ['--guided-decoding-backend', m.group(1)]
    return args


def start_server(venv: Path, model: str, port: int, cuda_dev: int, extra_args=None):
    py = venv / "bin/python"
    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN
    env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    env["HF_HOME"] = str(HF_CACHE)
    env["VLLM_USE_V1"] = "0"
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_dev)
    cmd = [str(py), "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--port", str(port),
           "--host", "127.0.0.1",
           "--max-model-len", "4096",
           "--tensor-parallel-size", "1",
           "--disable-log-requests",
           "--enforce-eager",
           "--trust-remote-code"]
    if extra_args:
        cmd.extend(extra_args)
    log(f"  spawn server cmd: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={cuda_dev})")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, cwd="/tmp"), cmd


def wait_for_server(port: int, proc, timeout=900) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=2)
            s.close()
            log(f"  server ready after {int(time.time() - start)}s")
            return True
        except (OSError, socket.timeout):
            time.sleep(2)
    return False


def run_inprocess_bench(cell_venv: Path, info: dict, bench_dir: Path, cuda_dev: int, overlay: dict) -> dict:
    """In-process bench: benchmark_throughput.py / benchmark_latency.py — instantiates
    vllm.LLM directly, no server. We invoke the script with the canonical perf_command
    args (rewriting `vllm bench throughput/latency` to the .py form for compat)."""
    pc = info["perf_command"].strip()
    pc = re.sub(r"^vllm\s+bench\s+throughput", "python benchmarks/benchmark_throughput.py", pc)
    pc = re.sub(r"^vllm\s+bench\s+latency",   "python benchmarks/benchmark_latency.py",    pc)
    m = re.match(r'^(?:python3?\s+)?(?:benchmarks/)?(\S+\.py)\s+(.*)$', pc)
    if not m:
        return {"status": "error", "error": f"can't parse inprocess perf: {info['perf_command']}",
                "patch_overlay": overlay}
    script_name = m.group(1).rsplit('/', 1)[-1]
    bench_args = m.group(2)
    bench_args = re.sub(r'(-tp|--tensor-parallel-size)(\s+|=)\d+', r'\g<1>\g<2>1', bench_args)
    # Don't pass --host/--port to in-process scripts (no server).
    bench_args = re.sub(r'--host[\s=]\S+', '', bench_args)
    bench_args = re.sub(r'--port[\s=]\S+', '', bench_args)
    bench_args = re.sub(r'\s+', ' ', bench_args).strip()
    bench_script = bench_dir / "benchmarks" / script_name
    if not bench_script.exists():
        return {"status": "error", "error": f"bench script missing: {bench_script}",
                "patch_overlay": overlay}
    kill_gpu_orphans(cuda_dev)
    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN
    env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    env["HF_HOME"] = str(HF_CACHE)
    env["VLLM_USE_V1"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_dev)
    cmd = [str(cell_venv / "bin/python"), str(bench_script)] + bench_args.split()
    log(f"  in-process bench: {' '.join(cmd)}  (CUDA_VISIBLE_DEVICES={cuda_dev})")
    try:
        rc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                            env=env, cwd=str(bench_dir))
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "in-process bench timeout",
                "patch_overlay": overlay}
    out = (rc.stdout or "") + "\n" + (rc.stderr or "")
    if rc.returncode != 0:
        return {"status": "error", "error": f"in-process bench rc={rc.returncode}",
                "raw_output": out[-8000:], "patch_overlay": overlay}
    metrics = parse_inprocess_metrics(out)
    if not (metrics.get('throughput_tok_s') or metrics.get('output_token_throughput_tok_s') or metrics.get('avg_latency_s')):
        return {"status": "error", "error": "no throughput/latency extracted",
                "raw_output": out[-8000:], "metrics": metrics, "patch_overlay": overlay}
    kill_gpu_orphans(cuda_dev)
    return {"status": "success", "metrics": metrics, "raw_output": out[-12000:],
            "patch_overlay": overlay}


def parse_inprocess_metrics(out: str) -> dict:
    """Extract metrics from benchmark_throughput.py / benchmark_latency.py output.

    Format varies across vllm releases:
      v0.4.x: "Throughput: 13.88 requests/s, 7107.13 tokens/s"
      v0.7+ : "Throughput: X requests/s, Y total tokens/s, Z output tokens/s"
    """
    metrics = {}
    pats = {
        'request_throughput_req_s':     [r'Throughput:\s*([\d.]+)\s*requests/s'],
        'total_token_throughput_tok_s': [r'requests/s,\s*([\d.]+)\s*total tokens/s',
                                         r'requests/s,\s*([\d.]+)\s*tokens/s'],
        'output_token_throughput_tok_s':[r'total tokens/s,\s*([\d.]+)\s*output tokens/s',
                                         r'requests/s,\s*([\d.]+)\s*tokens/s',
                                         # Fallback: from running engine log line (last occurrence)
                                         r'Avg generation throughput:\s*([\d.]+)\s*tokens/s'],
        'throughput_tok_s':             [r'requests/s,\s*([\d.]+)\s*total tokens/s',
                                         r'requests/s,\s*([\d.]+)\s*tokens/s',
                                         r'Avg generation throughput:\s*([\d.]+)\s*tokens/s'],
        'avg_latency_s':                [r'Avg latency:\s*([\d.]+)\s*seconds'],
    }
    for k, ps in pats.items():
        for p in ps:
            m = re.search(p, out)
            if m:
                metrics[k] = float(m.group(1))
                break
    return metrics


def parse_metrics(out: str) -> dict:
    metrics = {}
    pats = {
        'output_token_throughput_tok_s': [r'Output token throughput \(tok/s\):\s*([\d.]+)'],
        'total_token_throughput_tok_s':  [r'Total Token throughput \(tok/s\):\s*([\d.]+)',
                                          r'Total token throughput \(tok/s\):\s*([\d.]+)'],
        'request_throughput_req_s':      [r'Request throughput \(req/s\):\s*([\d.]+)'],
        'ttft_mean_ms':   [r'Mean TTFT \(ms\):\s*([\d.]+)'],
        'ttft_median_ms': [r'Median TTFT \(ms\):\s*([\d.]+)'],
        'ttft_p99_ms':    [r'P99 TTFT \(ms\):\s*([\d.]+)'],
        'tpot_mean_ms':   [r'Mean TPOT \(ms\):\s*([\d.]+)'],
        'tpot_median_ms': [r'Median TPOT \(ms\):\s*([\d.]+)'],
        'duration_s':     [r'Benchmark duration \(s\):\s*([\d.]+)'],
        'completed_requests': [r'Successful requests:\s*(\d+)'],
    }
    for k, ps in pats.items():
        for p in ps:
            m = re.search(p, out)
            if m:
                v = m.group(1)
                metrics[k] = float(v) if '.' in v else int(v)
                break
    return metrics


def run_cell(commit, agent, parent_short, vllm_version, task_dir, cuda_dev: int,
             patch_override: Path = None, role: str = "agent",
             model_override: str = None):
    info = read_mapping(commit)
    if model_override:
        old_model = info["model"]
        info["model"] = model_override
        # Substitute in perf_command too
        info["perf_command"] = re.sub(r'--model\s+\S+', f'--model {model_override}', info["perf_command"])
        log(f"  MODEL OVERRIDE: {old_model} → {model_override}")
    log(f"=== cell {commit}/{agent} role={role} on GPU{cuda_dev} (vllm=={vllm_version}, model={info['model']}) ===")
    log(f"  perf_command = {info['perf_command']}")

    base_venv = setup_base_venv(parent_short, vllm_version)
    if not list((base_venv / "lib").glob("python3.*/site-packages/vllm/__init__.py")):
        return {"status": "error", "error": "base venv vllm install missing"}

    cell_name = f"{commit}_{agent}_{role}" if role != "agent" else f"{commit}_{agent}"
    cell_venv = VENVS / f"cell_{cell_name}"
    if cell_venv.exists():
        shutil.rmtree(cell_venv)
    log(f"  cp -a {base_venv.name} → {cell_venv.name}")
    subprocess.run(["cp", "-a", str(base_venv), str(cell_venv)], check=True, timeout=180)
    patch = patch_override if patch_override else patch_path(commit, agent, task_dir)
    if not patch.exists():
        return {"status": "error", "error": f"patch not found: {patch}"}
    overlay = apply_overlay(cell_venv, patch)
    if overlay["rejected_files"]:
        log(f"  patch had rejects on {len(overlay['rejected_files'])} file(s); proceeding to bench anyway")

    bench_dir = fetch_benchmarks_dir(info["parent_full"])

    # Detect in-process bench (benchmark_throughput.py / benchmark_latency.py /
    # `vllm bench throughput|latency`) which runs vllm.LLM directly — no server.
    pc_lower = info["perf_command"].lower().strip()
    inprocess = bool(re.search(r'benchmark_(throughput|latency|prefix_caching)\.py|^vllm\s+bench\s+(throughput|latency)', pc_lower))
    if inprocess:
        return run_inprocess_bench(cell_venv, info, bench_dir, cuda_dev, overlay)

    bench_script = bench_dir / "benchmarks/benchmark_serving.py"
    if not bench_script.exists():
        return {"status": "error", "error": f"bench script missing: {bench_script}"}

    kill_gpu_orphans(cuda_dev)
    port = 30000 + cuda_dev * 1000
    kill_port(port)
    server_extra = extract_server_flags(info["perf_command"])
    if server_extra:
        log(f"  server-side flags from perf_command: {server_extra}")
    proc, server_cmd = start_server(cell_venv, info["model"], port, cuda_dev,
                                    extra_args=server_extra)
    out_lines = []
    try:
        if not wait_for_server(port, proc):
            try: stdout = proc.stdout.read() if proc.stdout else ""
            except Exception: stdout = ""
            return {"status": "error",
                    "error": "server failed to come up",
                    "raw_output": stdout[-6000:],
                    "patch_overlay": overlay}

        # Map `vllm bench serve` (newer CLI) to `benchmark_serving.py` for
        # compatibility with pre-CLI vllms that only ship the script.
        pc = re.sub(r"^vllm\s+bench\s+serve", "python benchmarks/benchmark_serving.py",
                    info["perf_command"].strip())
        m = re.match(r'^(?:python3?\s+)?(?:benchmarks/)?(\S+\.py)\s+(.*)$', pc)
        if not m:
            return {"status": "error", "error": f"can't parse perf: {info['perf_command']}"}
        bench_args = m.group(2)
        bench_args = re.sub(r'(-tp|--tensor-parallel-size)(\s+|=)\d+', r'\g<1>\g<2>1', bench_args)
        # Strip server-side flags that bench client doesn't recognize
        bench_args = re.sub(r'--dtype[\s=]\S+', '', bench_args)
        bench_args = re.sub(r'--enable-prefix-caching\b', '', bench_args)
        bench_args = re.sub(r'--enable-chunked-prefill\b', '', bench_args)
        bench_args = re.sub(r'--quantization[\s=]\S+', '', bench_args)
        bench_args = re.sub(r"--speculative-model\s+'?\S+'?", '', bench_args)
        bench_args = re.sub(r'--num-speculative-tokens[\s=]\S+', '', bench_args)
        bench_args = re.sub(r'--guided-decoding-backend[\s=]\S+', '', bench_args)
        # Always override --host / --port (perf_commands sometimes hardcode localhost:8000).
        bench_args = re.sub(r'--host[\s=]\S+', '', bench_args)
        bench_args = re.sub(r'--port[\s=]\S+', '', bench_args)
        bench_args = re.sub(r'\s+', ' ', bench_args).strip()
        bench_args += f' --host 127.0.0.1 --port {port}'
        # benchmark_serving.py needs --dataset-name + --dataset-path for sharegpt.
        # If perf_command picks `random` dataset, no path needed.
        if '--dataset-name' not in bench_args:
            bench_args += ' --dataset-name sharegpt'
        if 'sharegpt' in bench_args.lower() and '--dataset-path' not in bench_args:
            bench_args += f' --dataset-path {SHAREGPT}'
        # Match the canonical bench's prompt count of 100 by default (the
        # canonical runs used 100 sharegpt prompts; the original runner stored
        # 100 in its result files). If perf_command sets it explicitly (e.g.
        # fe66b347 has --num-prompts 300), keep that.
        if '--num-prompts' not in bench_args:
            bench_args += ' --num-prompts 100'

        client_env = os.environ.copy()
        client_env["HF_TOKEN"] = HF_TOKEN
        client_env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
        client_env["HF_HOME"] = str(HF_CACHE)
        client_py = cell_venv / "bin/python"
        cmd = [str(client_py), str(bench_script)] + bench_args.split()
        log(f"  bench client: {' '.join(cmd)}")
        rc = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=1800, env=client_env, cwd=str(bench_dir))
        out = (rc.stdout or "") + "\n" + (rc.stderr or "")
        if rc.returncode != 0:
            return {"status": "error",
                    "error": f"bench client failed rc={rc.returncode}",
                    "raw_output": out[-8000:],
                    "patch_overlay": overlay}
        metrics = parse_metrics(out)
        if not metrics.get('output_token_throughput_tok_s'):
            return {"status": "error",
                    "error": "no throughput extracted",
                    "raw_output": out[-8000:],
                    "metrics": metrics,
                    "patch_overlay": overlay}
        return {"status": "success",
                "metrics": metrics,
                "raw_output": out[-12000:],
                "patch_overlay": overlay}
    finally:
        try:
            proc.terminate(); time.sleep(3); proc.kill()
        except Exception: pass
        kill_port(port)
        kill_gpu_orphans(cuda_dev)
        # Reclaim disk: the cell venv is heavy (~5G). If the result was success,
        # keep it for audit; if error, clean up to free space for the next cell.
        # Decision left to caller via --keep-cell-venv flag.


def write_result(commit, agent, vllm_version, parent_short, result, cuda_dev, dur, role="agent",
                 model_override=None):
    info = read_mapping(commit)
    if model_override:
        info["model"] = model_override
        info["perf_command"] = re.sub(r'--model\s+\S+', f'--model {model_override}', info["perf_command"])
    record = {
        "human_commit": commit,
        "human_commit_full": info["human_full"],
        "parent_commit": info["parent_full"],
        "model": info["model"],
        "perf_command": info["perf_command"],
        "role": role,
        "status": result.get("status", "error"),
        "metrics": result.get("metrics", {}),
        "duration_s": dur,
        "raw_output": result.get("raw_output", ""),
        "error": result.get("error", ""),
        "patch_overlay": result.get("patch_overlay", {}),
        "vllm_pypi_version": vllm_version,
        "parent_short": parent_short,
        "model_overrides_disabled": True,
        "iso_bench_override_applied": False,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runner": "rebench_bamba_overlay_2026-05-06",
        "cuda_device": cuda_dev,
    }
    for p in (runner_result_path(commit, agent, role), submodule_result_path(commit, agent, role)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, indent=2))
        log(f"  wrote {p}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", action="append", required=True,
                   help="commit:agent (e.g. b690e348:sonnet45). Repeat for multiple cells.")
    p.add_argument("--cuda-device", type=int, required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="Set up base venv + cell venv + apply patch; report whether patch applies. No bench.")
    p.add_argument("--keep-cell-venv", action="store_true",
                   help="Don't rm the cell venv after bench (for debug).")
    p.add_argument("--patch", default=None,
                   help="Override patch path (e.g. /tmp/human_patches/b690e348.diff for human-side bench).")
    p.add_argument("--role", default="agent", choices=["agent", "human"],
                   help="Result file naming: <commit>_<role>_result.json. Default 'agent'.")
    p.add_argument("--model-override", default=None,
                   help="Override canonical model (e.g. for Group C ISO perf_command interpretation).")
    p.add_argument("--vllm-override", default=None,
                   help="Override vllm pypi version (when pre-parent doesn't support the chosen model).")
    args = p.parse_args()
    by_key = {f"{c}:{a}": (c, a, ps, vv, td) for (c, a, ps, vv, td) in CELLS}
    for sel in args.cell:
        if sel not in by_key:
            log(f"unknown cell: {sel}; valid: {list(by_key.keys())}")
            sys.exit(2)
        commit, agent, parent_short, vllm_version, task_dir = by_key[sel]
        if args.vllm_override:
            log(f"VLLM_OVERRIDE: {vllm_version} → {args.vllm_override}")
            vllm_version = args.vllm_override
        log(f"\n>>> START {commit}:{agent} cuda={args.cuda_device}")
        t0 = time.time()
        patch_override = Path(args.patch) if args.patch else None
        try:
            if args.dry_run:
                base = setup_base_venv(parent_short, vllm_version)
                cell_name = f"{commit}_{agent}_{args.role}" if args.role != "agent" else f"{commit}_{agent}"
                cell = VENVS / f"cell_{cell_name}"
                if cell.exists():
                    shutil.rmtree(cell)
                subprocess.run(["cp", "-a", str(base), str(cell)], check=True, timeout=180)
                p_path = patch_override if patch_override else patch_path(commit, agent, task_dir)
                overlay = apply_overlay(cell, p_path)
                result = {"status": "dry-run", "patch_overlay": overlay}
            else:
                result = run_cell(commit, agent, parent_short, vllm_version, task_dir, args.cuda_device,
                                  patch_override=patch_override, role=args.role,
                                  model_override=args.model_override)
        except Exception as e:
            import traceback
            result = {"status": "error",
                      "error": f"unhandled: {type(e).__name__}: {e}",
                      "raw_output": traceback.format_exc()}
        dur = round(time.time() - t0, 2)
        if not args.dry_run:
            write_result(commit, agent, vllm_version, parent_short, result, args.cuda_device, dur,
                         role=args.role, model_override=args.model_override)
        if not args.keep_cell_venv:
            cell_name = f"{commit}_{agent}_{args.role}" if args.role != "agent" else f"{commit}_{agent}"
            cv = VENVS / f"cell_{cell_name}"
            if cv.exists() and result.get("status") != "success":
                shutil.rmtree(cv, ignore_errors=True)
                log(f"  cleaned cell venv {cv}")
        log(f"<<< END {commit}:{agent} status={result.get('status')} dur={dur}s")


if __name__ == "__main__":
    main()
