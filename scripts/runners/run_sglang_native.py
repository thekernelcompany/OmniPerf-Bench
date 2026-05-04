#!/usr/bin/env python3
"""Native SGLang benchmark runner — PyPI overlay path (no docker, no udocker).

Per commit:
  1. uv venv at /tmp/native_sglang_venvs/<human_short>
  2. uv pip install sglang[all]==<pypi_version> derived from parent → closest tag
     in the sglang submodule.
  3. Pin transformers to a version that doesn't pull torch 2.7+ APIs (sglang
     0.4.6.post* + transformers 4.51 trigger AttributeError on
     torch._pytree.register_constant since that was added in torch 2.7+).
  4. Apply agent patch to <venv>/lib/python3.X/site-packages/
  5. Start `python -m sglang.launch_server` natively
  6. Run `python -m sglang.bench_serving --backend sglang ...`
  7. Parse metrics, write JSON

Mirrors run_vllm_native.py's structure & CLI surface.
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

_DEFAULT_ROOT = Path(
    os.environ.get("OMNIPERF_ROOT")
    or os.environ.get("ISOBENCH_ROOT")
    or str(Path(__file__).resolve().parent.parent.parent)
)
ROOT: Path = _DEFAULT_ROOT
MAPPING_FILE: Path = ROOT / "data/mappings/sglang_oh_mapping.json"
RESULTS_DIR: Path = ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45_sglang/results"
PATCHES_DIR: Path = ROOT / "ISO-Bench/state/runs/sglang/openhands_sonnet45/flat"
SGLANG_REPO: Path = ROOT / "sglang"  # submodule, used to discover release tag for each parent

VENV_ROOT = Path(os.environ.get("NATIVE_SGLANG_VENV_ROOT", "/tmp/native_sglang_venvs"))


def _default_hf_cache() -> Path:
    eph = Path("/ephemeral/huggingface_cache")
    try:
        eph.mkdir(parents=True, exist_ok=True)
        return eph
    except (PermissionError, OSError):
        return Path.home() / ".cache/huggingface"
HF_CACHE: Path = Path(os.environ.get("HF_HOME", str(_default_hf_cache())))


def _default_uv_bin() -> Path:
    if env_uv := os.environ.get("UV_BIN"):
        return Path(env_uv)
    found = shutil.which("uv")
    if found:
        return Path(found)
    return Path("/root/.local/bin/uv")
UV_BIN: Path = _default_uv_bin()

SHAREGPT_PATH = str(ROOT / "data/archive/sharegpt_dataset.json")

AGENT_PROFILES = {
    "openhands_sonnet45": (
        "ISO-Bench/state/runs/sglang/openhands_sonnet45/flat",
        "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45_sglang/results",
    ),
    "openhands_gpt5": (
        "ISO-Bench/state/runs/sglang/openhands_gpt5/flat",
        "archive/results/2026-05/iso_bench_results_3way_openhands_gpt5_sglang/results",
    ),
}

# Same model-name overrides the vLLM runner uses; some HF orgs gated.
MODEL_OVERRIDES = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
}


_ISO_BENCH_CACHE = None


def load_iso_bench_perf_overrides() -> dict:
    global _ISO_BENCH_CACHE
    if _ISO_BENCH_CACHE is not None:
        return _ISO_BENCH_CACHE
    candidates = [
        Path("/tmp/iso_bench_hf/data/sglang/train.parquet"),
        ROOT / "data/iso_bench/sglang/train.parquet",
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
                print(f"  warn: ISO-Bench sglang parquet at {path} unreadable: {e}")
    _ISO_BENCH_CACHE = {}
    return _ISO_BENCH_CACHE


def merge_with_iso_bench(commit_short: str, info: dict, log) -> dict:
    iso = load_iso_bench_perf_overrides().get(commit_short, {})
    if not iso:
        return info
    merged = dict(info)
    iso_pc = (iso.get("perf_command") or "").strip()
    if iso_pc and iso_pc != (info.get("perf_command", "") or "").strip():
        log(f"  ISO-Bench perf_command override (multiline ok): {iso_pc[:140]}")
        merged["perf_command"] = iso_pc
    if iso_pc:
        m = re.search(r'--model(?:-path)?(?:\s+|=)(\S+)', iso_pc)
        if m:
            iso_model = m.group(1)
            if iso_model and iso_model != info.get("model", ""):
                log(f"  ISO-Bench model override: {info.get('model', '')} -> {iso_model}")
                merged["model"] = iso_model
    return merged


GATED_ORGS = ("meta-llama/", "mistralai/")


def _read_token_file(path: Path) -> str:
    try:
        return path.read_text().strip() if path.exists() else ""
    except Exception:
        return ""


def get_hf_token(model: str = "") -> str:
    gated_path = Path.home() / ".config/omniperf/hf_token_gated"
    default_path = Path.home() / ".cache/huggingface/token"
    if any(model.startswith(org) for org in GATED_ORGS):
        tok = _read_token_file(gated_path)
        if tok:
            return tok
    return _read_token_file(default_path)


def free_port(start: int = 30000) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return start + int(cvd) * 1000


def kill_gpu_orphans(log) -> None:
    """Kill PPID=1 python orphans from /tmp/native_*venvs holding GPU mem."""
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
            if any(x in name for x in ("/tmp/native_venvs/", "/tmp/native_sglang_venvs/",
                                        "/ephemeral/native_venvs/", "/ephemeral/native_sglang_venvs/")):
                log(f"  killing stale GPU compute app PID {pid} ({name})")
                try: os.kill(int(pid), 9)
                except ProcessLookupError: pass
        time.sleep(1)
    except Exception as e:
        log(f"  kill_gpu_orphans error: {e}")


def kill_port(port: int, log) -> None:
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


def discover_pypi_version(parent_full: str) -> str:
    """Find the closest release tag that contains parent_full in the sglang
    submodule. Returns the version string (without the leading 'v').
    """
    if not (SGLANG_REPO / ".git").exists():
        raise RuntimeError(f"sglang submodule not initialized at {SGLANG_REPO}; "
                           f"run `git submodule update --init sglang`")
    r = subprocess.run(
        ["git", "-C", str(SGLANG_REPO), "tag", "--contains", parent_full, "--sort=v:refname"],
        capture_output=True, text=True, timeout=30,
    )
    tags = [t for t in r.stdout.splitlines() if t.startswith("v")]
    if not tags:
        raise RuntimeError(f"no release tag in sglang/ contains parent {parent_full[:12]}")
    return tags[0].lstrip("v")


def setup_venv(commit_short: str, parent_full: str, log) -> Path:
    """Create a uv venv and install sglang[all] from PyPI matching parent commit."""
    venv = VENV_ROOT / commit_short
    if venv.exists():
        shutil.rmtree(venv, ignore_errors=True)
    venv.parent.mkdir(parents=True, exist_ok=True)
    log(f"  Creating venv at {venv} (Python 3.12)")
    subprocess.run([str(UV_BIN), "venv", "--python", "3.12", str(venv)],
                   check=True, capture_output=True, text=True, timeout=120)
    py = venv / "bin/python"

    pypi_v = discover_pypi_version(parent_full)
    log(f"  parent {parent_full[:12]} -> sglang=={pypi_v}")

    # Per-commit override: 1acca3a2 / 205d5cb4 are pinned to v0.4.6.post3/post5
    # which require torch 2.6 (no register_constant) BUT need transformers 4.51+
    # for Llama-4 (205d5cb4) or fail with the ABI break (1acca3a2). Overriding
    # torch alone breaks sgl_kernel's compiled ABI. The cleanest fix is to install
    # sglang 0.4.7 (which natively pins torch 2.7 + transformers 4.52 + matching
    # sgl_kernel) and let the patch apply with whatever drift it incurs (~2-3
    # hunks fail, the agent's load-bearing edits still land).
    if commit_short in {"1acca3a2", "205d5cb4"}:
        log(f"  force_modern: substituting sglang=={pypi_v} -> sglang==0.4.7 (torch-2.7-native, fixes sgl_kernel ABI)")
        pypi_v = "0.4.7"

    env = os.environ.copy()
    env["UV_SKIP_WHEEL_FILENAME_CHECK"] = "1"
    # `sglang[all]` pulls flashinfer/sgl-kernel etc. Use unsafe-best-match to
    # avoid uv's strict resolver tripping on transitive overlap with PyTorch.
    log(f"  Installing sglang[all]=={pypi_v}")
    r = subprocess.run(
        [str(UV_BIN), "pip", "install", "--python", str(py),
         f"sglang[all]=={pypi_v}",
         "--index-strategy", "unsafe-best-match"],
        capture_output=True, text=True, timeout=1500, env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(f"sglang[all]=={pypi_v} install failed: "
                           f"{(r.stderr or r.stdout)[-600:]}")

    # transformers ABI fixup: sglang 0.4.6.post* pins torch==2.6.0 +
    # transformers==4.51.1, but transformers 4.51.1 imports
    # torch._pytree.register_constant which was only added in torch 2.7+. So
    # for that version specifically we pin transformers<4.51.
    #
    # sglang>=0.4.7 ships torch==2.7.1 + transformers==4.52.3 — internally
    # consistent. Don't override its transformers; instead check that
    # masking_utils (added in 4.52) is available, which downstream
    # compressed_tensors imports.
    if pypi_v.startswith("0.4.6"):
        log(f"  v{pypi_v} → pinning transformers<4.51 (torch 2.6 register_constant ABI break)")
        subprocess.run(
            [str(UV_BIN), "pip", "install", "--python", str(py),
             "transformers>=4.45,<4.51",
             # compressed-tensors 0.10+ depends on transformers.masking_utils
             # (added in transformers 4.53+), which doesn't exist at this pin.
             "compressed-tensors==0.9.4",
             "aiohttp", "pandas", "datasets", "pillow", "numpy<2"],
            capture_output=True, text=True, timeout=300, env=env,
        )
    else:
        # sglang 0.4.7+ pins transformers==4.52.3 + torch==2.7.1 internally.
        # But it leaves compressed-tensors unpinned, so uv pulls 0.15.0.1
        # (latest) which imports `transformers.masking_utils` — added in
        # transformers 4.53+. Pin compressed-tensors to a version contemporary
        # with sglang 0.4.7 (June 2025 → 0.9.4) that doesn't make that import.
        log(f"  v{pypi_v} → trusting sglang's transformers/torch, pinning compressed-tensors==0.9.4")
        deps = ["compressed-tensors==0.9.4",
                "aiohttp", "pandas", "datasets", "pillow"]
        # 205d5cb4 (Llama-4-Scout w4a16): sglang's
        # CompressedTensorsWNA16MoEMethod imports from vllm. Co-install vllm
        # (compatible with torch 2.7 from version 0.10+).
        if commit_short == "205d5cb4":
            log("    + co-installing vllm (needed by CompressedTensorsWNA16MoEMethod for Llama-4-Scout w4a16)")
            deps.append("vllm")
        subprocess.run(
            [str(UV_BIN), "pip", "install", "--python", str(py)] + deps + [
             "--index-strategy", "unsafe-best-match"],
            capture_output=True, text=True, timeout=900, env=env,
        )
    return venv


def apply_patch(venv: Path, patch_path: Path, log) -> None:
    """Apply agent patch to the installed sglang package.

    Agent patches reference paths like `a/python/sglang/srt/X.py` (source-repo
    layout) but the installed package uses `sglang/srt/X.py` directly under
    site-packages. Use `-p2` to strip both `a/` and `python/`, landing the
    edits on the right files.
    """
    site_pkgs = next((venv / "lib").glob("python3.*/site-packages"))
    log(f"  Applying patch at {site_pkgs} (-p2 to strip a/python/)")
    r = subprocess.run(
        ["patch", "-p2", "--force"],
        cwd=str(site_pkgs), input=patch_path.read_text(),
        capture_output=True, text=True, timeout=60,
    )
    log(f"  patch result rc={r.returncode}: {r.stdout[-300:].strip()}")
    if r.returncode != 0 and r.stderr:
        log(f"  patch stderr: {r.stderr[-300:].strip()}")


def start_server(venv: Path, model: str, port: int, log,
                 extra_server_args: list = None) -> subprocess.Popen:
    py = venv / "bin/python"
    env = os.environ.copy()
    env["HF_TOKEN"] = get_hf_token(model)
    env["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]
    env["HF_HOME"] = str(HF_CACHE)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    num_gpus = max(1, len([x for x in cvd.split(",") if x.strip()]))
    cmd = [str(py), "-m", "sglang.launch_server",
           "--model-path", model,
           "--host", "127.0.0.1",
           "--port", str(port),
           "--tp", str(num_gpus),
           "--trust-remote-code",
           "--disable-cuda-graph",  # faster startup; minor perf hit acceptable
           ]
    if extra_server_args:
        cmd.extend(extra_server_args)
        log(f"  Extra server args: {extra_server_args}")
    log(f"  Starting: {' '.join(cmd[2:])}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, cwd="/tmp")
    return proc


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


def split_perf_command(perf_command: str) -> tuple:
    """ISO-Bench sglang perf_commands are sometimes two-stanza:
       <launch_server cmd>\n\n<bench_serving cmd>
    Return (server_flags_str, bench_cmd_str). If single-line bench-only,
    server_flags_str is "" and bench_cmd_str is the whole input.
    """
    parts = [p.strip() for p in perf_command.strip().split("\n\n") if p.strip()]
    if len(parts) >= 2:
        # First part is the server launch line (sglang.launch_server ...);
        # second part is the bench client (sglang.bench_serving ...).
        server_line = parts[0]
        bench_line = "\n\n".join(parts[1:])
        return server_line, bench_line
    return "", parts[0] if parts else ""


def extract_server_flags_from_perf(perf_command: str) -> list:
    """SGLang server-side flags. Reads ONLY the launch_server stanza when
    perf_command is multi-line, so flag-value regexes don't accidentally
    span the blank-line boundary into the bench_serving stanza (the
    --lora-paths regex would otherwise slurp `\\n\\npython3 -m
    sglang.bench_serving` as part of its argument).
    """
    server_line, bench_line = split_perf_command(perf_command)
    # If single-stanza, we don't expect server flags embedded — but we still
    # scan it permissively in case the dataset author put server flags inline.
    src = server_line if server_line else bench_line
    args = []
    if '--disable-radix-cache' in src: args.append('--disable-radix-cache')
    if '--enable-mixed-chunk' in src: args.append('--enable-mixed-chunk')
    if '--enable-torch-compile' in src: args.append('--enable-torch-compile')
    m = re.search(r'--mem-fraction-static\s+([\d.]+)', src)
    if m: args += ['--mem-fraction-static', m.group(1)]
    m = re.search(r'--quantization\s+(\S+)', src)
    if m: args += ['--quantization', m.group(1)]
    # --lora-paths takes one or more `name=hf/repo` tokens. Capture only the
    # tokens that look like `\S+=\S+` so we never bleed into the next flag.
    m = re.search(r'--lora-paths((?:\s+\S+=\S+)+)', src)
    if m: args += ['--lora-paths'] + m.group(1).split()
    m = re.search(r'--chunked-prefill-size\s+(\d+)', src)
    if m: args += ['--chunked-prefill-size', m.group(1)]
    m = re.search(r'--max-running-requests\s+(\d+)', src)
    if m: args += ['--max-running-requests', m.group(1)]
    return args


def run_benchmark(venv: Path, model: str, port: int, perf_command: str, log) -> str:
    """Run the perf_command via `python -m sglang.bench_serving` with --host/--port substituted."""
    py = venv / "bin/python"
    # If multi-stanza, take only the bench line. The server line was already
    # consumed by extract_server_flags_from_perf() at start_server time.
    _, bench_only = split_perf_command(perf_command)
    pc = bench_only.strip()

    env_overrides = {}
    while True:
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(\S+)\s+(.*)$", pc)
        if not m: break
        env_overrides[m.group(1)] = m.group(2)
        pc = m.group(3)

    # Strip leading `python(3) -m sglang.bench_serving`; invoke as module below.
    pc = re.sub(r"^python3?\s+-m\s+sglang\.bench_serving\s+", "", pc)
    # Defensive: if perf_command had only the launch_server line (corrupt input),
    # we'd see launch_server prefix here — strip it but log a warning.
    if pc.startswith("python") and "launch_server" in pc[:80]:
        log("  WARN: perf_command appears to lack a bench_serving stanza; using as-is")
        pc = re.sub(r"^python3?\s+-m\s+sglang\.launch_server\s+", "", pc)

    bench_args = pc

    # Substitute model
    if '--model' in bench_args:
        bench_args = re.sub(r'--model(?:\s+|=)\S+', f'--model {model}', bench_args)
    else:
        bench_args += f' --model {model}'
    if '--backend' not in bench_args:
        bench_args += ' --backend sglang'

    # Inject host/port
    if '--host' not in bench_args:
        bench_args += ' --host 127.0.0.1'
    else:
        bench_args = re.sub(r'--host\s+\S+', '--host 127.0.0.1', bench_args)
    if '--port' not in bench_args:
        bench_args += f' --port {port}'
    else:
        bench_args = re.sub(r'--port\s+\S+', f'--port {port}', bench_args)

    # sharegpt path injection
    if 'sharegpt' in bench_args.lower() and '--dataset-path' not in bench_args:
        bench_args += f' --dataset-path {SHAREGPT_PATH}'

    # Strip server-only flags from the bench client
    for pat in [r'--disable-radix-cache\b', r'--enable-torch-compile\b',
                r'--enable-mixed-chunk\b', r'--mem-fraction-static\s+[\d.]+',
                r'--tp\s+\d+', r'--tensor-parallel-size\s+\d+',
                r'--quantization\s+\S+', r'--trust-remote-code\b',
                r'--max-running-requests\s+\d+', r'--chunked-prefill-size\s+\d+',
                r'--disable-cuda-graph\b']:
        bench_args = re.sub(pat, '', bench_args)
    bench_args = re.sub(r'\s+', ' ', bench_args).strip()

    # Cap num-prompts. ISO-Bench default upper bound is 100, but for the
    # --max-concurrency 1 case (sequential bench) cut to 30 — each prompt
    # at 1000 output tokens × ~50ms TPOT is ~50s, so 100 sequential prompts
    # would be 80 min. 30 is plenty for stable TTFT/TPOT distributions.
    bench_args = re.sub(r'--num-prompt(?!s)(\s+|=)(\d+)',
                        lambda m: f'--num-prompts{m.group(1)}{m.group(2)}',
                        bench_args)
    is_seq = re.search(r'--max-concurrency\s+1\b', bench_args) is not None
    NUM_PROMPTS_CAP = 30 if is_seq else 100
    m_np = re.search(r'--num-prompts\s+(\d+)', bench_args)
    if m_np:
        if int(m_np.group(1)) > NUM_PROMPTS_CAP:
            bench_args = re.sub(r'--num-prompts\s+\d+', f'--num-prompts {NUM_PROMPTS_CAP}', bench_args)
    else:
        bench_args += f' --num-prompts {NUM_PROMPTS_CAP}'

    # Adaptive subprocess timeout for sequential benches.
    bench_timeout = 5400 if is_seq else 1500
    if is_seq:
        log(f"  --max-concurrency 1 detected → cap {NUM_PROMPTS_CAP} prompts, timeout {bench_timeout}s")

    cmd_str = f"{py} -m sglang.bench_serving {bench_args}"
    log(f"  Running: sglang.bench_serving {bench_args[:120]}")
    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE)
    tok = get_hf_token(model)
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    env.update(env_overrides)
    r = subprocess.run(cmd_str, shell=True, capture_output=True, text=True,
                       timeout=bench_timeout, env=env, cwd="/tmp")
    return r.stdout + r.stderr


def run_bench_one_batch(venv: Path, model: str, perf_command: str, log) -> str:
    """In-process bench via `python -m sglang.bench_one_batch`.

    Unlike bench_serving (HTTP client), bench_one_batch loads the model
    directly and emits decode latency metrics. We pass the perf_command
    args verbatim (after stripping the leading `python -m sglang.bench_one_batch`).
    """
    py = venv / "bin/python"
    pc = perf_command.strip()
    # Strip env prefixes
    env_overrides = {}
    while True:
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(\S+)\s+(.*)$", pc)
        if not m: break
        env_overrides[m.group(1)] = m.group(2)
        pc = m.group(3)
    pc = re.sub(r"^python3?\s+-m\s+sglang\.bench_one_batch\s+", "", pc)
    bench_args = pc
    # Substitute model
    if '--model' in bench_args:
        bench_args = re.sub(r'--model(?:\s+|=)\S+', f'--model {model}', bench_args)
    else:
        bench_args += f' --model {model}'

    cmd_str = f"{py} -m sglang.bench_one_batch {bench_args}"
    log(f"  Running: sglang.bench_one_batch {bench_args[:140]}")
    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE)
    tok = get_hf_token(model)
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    env.update(env_overrides)
    r = subprocess.run(cmd_str, shell=True, capture_output=True, text=True,
                       timeout=1500, env=env, cwd="/tmp")
    return r.stdout + r.stderr


def parse_one_batch_metrics(output: str) -> dict:
    """sglang.bench_one_batch emits lines like:
       Prefill. latency: 0.123 s, throughput: 8123.4 token/s
       Decode.  median latency: 0.005 s, median throughput: 172.5 token/s
       Total.   latency: 5.67 s, throughput: 51.2 token/s
    Decode line uses 'median latency'/'median throughput' (it's an aggregate
    over all decode steps); other lines use plain 'latency'/'throughput'.
    """
    metrics = {}
    pairs = [
        ('prefill_latency_s',   r'Prefill\.\s*latency:\s*([\d.]+)\s*s'),
        ('prefill_throughput_tok_s', r'Prefill\.\s*latency:[^,]+,\s*throughput:\s*([\d.]+)'),
        ('decode_latency_s',    r'Decode\.\s*median latency:\s*([\d.]+)\s*s'),
        ('decode_throughput_tok_s',  r'Decode\.\s*median latency:[^,]+,\s*median throughput:\s*([\d.]+)'),
        ('total_latency_s',     r'Total\.\s*latency:\s*([\d.]+)\s*s'),
        ('total_throughput_tok_s',   r'Total\.\s*latency:[^,]+,\s*throughput:\s*([\d.]+)'),
    ]
    for k, pat in pairs:
        m = re.search(pat, output)
        if m:
            metrics[k] = float(m.group(1))
    return metrics


def parse_metrics(output: str) -> dict:
    """Parse sglang.bench_serving output. Format mirrors vLLM's benchmark_serving.py."""
    metrics = {}
    patterns = {
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
    for k, pat in patterns.items():
        m = re.search(pat, output)
        if m:
            metrics[k] = float(m.group(1))
    return metrics


def find_task_dir(commit_short: str) -> str:
    """Find the sglang_core-NNNN dir matching a human commit short."""
    for d in sorted(PATCHES_DIR.iterdir()):
        rs = d / "run_summary.json"
        if not rs.exists():
            continue
        j = json.loads(rs.read_text())
        if j["commits"]["human"][:8] == commit_short:
            return d.name
    return ""


def run_one(commit_short: str, info: dict, timeout: int = 1800) -> dict:
    start = time.time()
    out_lines = []

    def log(msg):
        out_lines.append(msg)
        print(msg, flush=True)

    info = merge_with_iso_bench(commit_short, info, log)
    parent_full = info.get("parent_commit") or info.get("human_commit_full", "")
    model = MODEL_OVERRIDES.get(info["model"], info["model"])
    if model != info["model"]:
        log(f"  Model override: {info['model']} -> {model}")
    perf = info.get("perf_command") or ""

    task_dir = find_task_dir(commit_short)
    if not task_dir:
        return {"status": "error",
                "error": f"no task dir in {PATCHES_DIR} matches commit_short={commit_short}",
                "duration_s": 0, "metrics": {}, "raw_output": ""}
    patch_path = PATCHES_DIR / task_dir / "model_patch.diff"
    if not patch_path.exists() or patch_path.stat().st_size == 0:
        return {"status": "error", "error": f"empty/missing patch at {patch_path}",
                "duration_s": 0, "metrics": {}, "raw_output": ""}

    kill_gpu_orphans(log)
    # bench_one_batch is an in-process bench (loads sglang.Engine directly,
    # no HTTP server). Route those without starting/connecting to a server.
    is_one_batch = "sglang.bench_one_batch" in perf
    if is_one_batch:
        log(f"  Detected sglang.bench_one_batch — running in-process (no server)")
    port = free_port()
    proc = None
    try:
        venv = setup_venv(commit_short, parent_full, log)
        apply_patch(venv, patch_path, log)
        kill_port(port, log)
        if is_one_batch:
            bench_out = run_bench_one_batch(venv, model, perf, log)
            metrics = parse_one_batch_metrics(bench_out)
            if not metrics:
                return {"status": "error", "error": "No metrics in bench_one_batch output",
                        "duration_s": time.time() - start, "metrics": {},
                        "raw_output": "\n".join(out_lines[-50:]) +
                                      "\n--- bench ---\n" + bench_out[-20000:]}
            return {"status": "success", "metrics": metrics,
                    "duration_s": time.time() - start,
                    "raw_output": "\n".join(out_lines[-30:]) +
                                  "\n--- bench tail ---\n" + bench_out[-10000:]}
        extra = extract_server_flags_from_perf(perf)
        proc = start_server(venv, model, port, log, extra_server_args=extra)
        if not wait_for_server(port, proc, timeout=600, log=log):
            stdout = ""
            try:
                proc.terminate()
                stdout = proc.communicate(timeout=10)[0] or ""
            except Exception:
                pass
            return {"status": "error", "error": "Server startup timeout / crash",
                    "duration_s": time.time() - start, "metrics": {},
                    "raw_output": "\n".join(out_lines[-50:]) +
                                  "\n--- server stdout ---\n" + stdout[-20000:]}
        bench_out = run_benchmark(venv, model, port, perf, log)
        metrics = parse_metrics(bench_out)
        if not metrics:
            return {"status": "error", "error": "No metrics in benchmark output",
                    "duration_s": time.time() - start, "metrics": {},
                    "raw_output": "\n".join(out_lines[-50:]) +
                                  "\n--- bench ---\n" + bench_out[-20000:]}
        return {"status": "success", "metrics": metrics,
                "duration_s": time.time() - start,
                "raw_output": "\n".join(out_lines[-30:]) +
                              "\n--- bench tail ---\n" + bench_out[-10000:]}
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=10)
            except Exception: proc.kill()
        kill_port(port, log)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commits", nargs="+", required=True)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--agent-name", default="openhands_sonnet45",
                   choices=sorted(AGENT_PROFILES.keys()))
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--patches-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--mapping-file", type=Path, default=None)
    args = p.parse_args()

    global ROOT, MAPPING_FILE, RESULTS_DIR, PATCHES_DIR, SHAREGPT_PATH, SGLANG_REPO
    if args.root:
        ROOT = args.root.resolve()
        SHAREGPT_PATH = str(ROOT / "data/archive/sharegpt_dataset.json")
        SGLANG_REPO = ROOT / "sglang"
    patches_rel, results_rel = AGENT_PROFILES[args.agent_name]
    PATCHES_DIR = (args.patches_dir or (ROOT / patches_rel)).resolve()
    RESULTS_DIR = (args.results_dir or (ROOT / results_rel)).resolve()
    MAPPING_FILE = (args.mapping_file or (ROOT / "data/mappings/sglang_oh_mapping.json")).resolve()

    print(f"  ROOT          = {ROOT}")
    print(f"  PATCHES_DIR   = {PATCHES_DIR}")
    print(f"  RESULTS_DIR   = {RESULTS_DIR}")
    print(f"  MAPPING_FILE  = {MAPPING_FILE}")
    print(f"  SGLANG_REPO   = {SGLANG_REPO}")
    print(f"  HF_CACHE      = {HF_CACHE}")
    print(f"  UV_BIN        = {UV_BIN}")

    if not MAPPING_FILE.exists():
        sys.exit(f"ERROR: mapping file not found at {MAPPING_FILE}")
    if not PATCHES_DIR.exists():
        sys.exit(f"ERROR: patches dir not found at {PATCHES_DIR}")
    if not UV_BIN.exists():
        sys.exit(f"ERROR: uv not found at {UV_BIN}")
    if not SGLANG_REPO.exists() or not (SGLANG_REPO / ".git").exists():
        sys.exit(f"ERROR: sglang submodule not initialized at {SGLANG_REPO}")

    mp = json.loads(MAPPING_FILE.read_text())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Native SGLang benchmark — {len(args.commits)} commits ===")
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
                      "duration_s": 0, "metrics": {},
                      "raw_output": traceback.format_exc()}
        info_for_record = merge_with_iso_bench(c, dict(info), lambda _msg: None)
        record = {
            "human_commit": c,
            "human_commit_full": info.get("human_commit_full", ""),
            "parent_commit": info.get("parent_commit", ""),
            "model": MODEL_OVERRIDES.get(info_for_record["model"], info_for_record["model"]),
            "perf_command": info_for_record.get("perf_command", ""),
            **result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runner": "sglang_native",
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
