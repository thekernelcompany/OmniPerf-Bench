#!/usr/bin/env python3
"""SGLang agent-only hard-metrics runner — udocker via the docker→udocker shim.

Mirrors run_3way_benchmarks.py but for SGLang baseline images:
- Image candidates: ayushnangia16/sglang-docker:<full_40char>
- Server: python3 -m sglang.launch_server --model-path <model>
- Benchmark client: sglang.bench_serving (or perf_command from mapping)
- Patch applied to /opt/sglang via patch -p1
- Same docker_cmd shape as vLLM agent path: libcudart fix, /etc/hosts fix,
  socket-connect health check, port-templating via $((30000 + CVD))

Reads commit list from data/mappings/sglang_oh_mapping.json (14 OH tasks).
Writes per-commit results to archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45_sglang/results/.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MAPPING_FILE = ROOT / "data/mappings/sglang_oh_mapping.json"
RESULTS_DIR = ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45_sglang/results"

# SGLang baseline images — all 14 OH parents are at ayushnangia16/sglang-docker
SGLANG_IMAGE_CANDIDATES = [
    ("ayushnangia16/sglang-docker", "{full}"),
    ("ayushnangia16/nvidia-sglang-docker", "{full}"),
    ("shikhar481/sglang-images", "v05-improved-{short12}"),
]
_resolved_cache = {}


def get_hf_token() -> str:
    p = Path.home() / ".cache/huggingface/token"
    return p.read_text().strip() if p.exists() else ""


def resolve_image(full_hash: str) -> str:
    if full_hash in _resolved_cache:
        return _resolved_cache[full_hash]
    short12 = full_hash[:12]
    for repo, tagtmpl in SGLANG_IMAGE_CANDIDATES:
        tag = tagtmpl.format(full=full_hash, short12=short12)
        image = f"{repo}:{tag}"
        r = subprocess.run(["docker", "pull", image],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode == 0:
            print(f"    resolved {full_hash[:8]} -> {image}")
            _resolved_cache[full_hash] = image
            return image
        print(f"    pull miss: {image} -> {(r.stderr or r.stdout)[-160:].strip()}")
    raise RuntimeError(f"No sglang image found for {full_hash[:12]}")


def parse_serving_metrics(output: str) -> dict:
    """Extract metrics from sglang.bench_serving output."""
    metrics = {}
    # SGLang bench_serving prints things like:
    #   Mean TTFT (ms):    324.05
    #   Median TTFT (ms):  308.71
    #   P99 TTFT (ms):     642.37
    #   Mean TPOT (ms):    24.30
    #   Output token throughput (tok/s):   3319.30
    #   Total token throughput (tok/s):    5210.50
    #   Request throughput (req/s):  53.80
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


def build_docker_cmd(model: str, parent_commit: str, perf_command: str) -> str:
    """Build the bash docker_cmd for an sglang agent benchmark.

    Heredoc-style; embedded as a single bash script run via shim's
    `docker run --rm --gpus all --entrypoint bash <image> -c <cmd>`.
    """
    # Build a clean bench_serving command. Don't reuse the prior perf_command —
    # it often references --lora-name lora (server isn't configured for it),
    # custom datasets, etc. We just want a uniform serving benchmark.
    bench_args = (
        f'--backend sglang '
        f'--host 127.0.0.1 --port $SGL_PORT '
        f'--num-prompt 100 --request-rate inf '
        f'--dataset-name random --random-input-len 256 --random-output-len 64'
    )

    return f'''
    set -e
    SGL_PORT=$((30000 + ${{CUDA_VISIBLE_DEVICES:-0}}))
    grep -q '^127.0.0.1.*localhost' /etc/hosts 2>/dev/null || echo '127.0.0.1 localhost' >> /etc/hosts
    PARENT_COMMIT="{parent_commit}"
    MODEL="{model}"

    echo "=== Applying agent patch to baseline SGLang ==="

    # SGLang source location varies across images; probe known paths.
    SGL_DIR=""
    for d in /sgl-workspace/sglang /opt/sglang /sglang /workspace/sglang /opt/sgl-workspace/sglang; do
        if [ -d "$d/python/sglang" ] || [ -d "$d/sglang" ]; then
            SGL_DIR="$d"; break
        fi
    done
    if [ -z "$SGL_DIR" ]; then
        echo "ERROR: SGLang source not found in known paths"
        ls /opt /sgl-workspace 2>/dev/null
        exit 1
    fi
    echo "Found SGLang source at: $SGL_DIR"
    cd "$SGL_DIR"

    echo "Applying patch..."
    if patch -p1 --dry-run < /agent_patch.diff 2>&1; then
        patch -p1 < /agent_patch.diff 2>&1
        echo "AGENT_PATCH_APPLIED"
    else
        patch -p1 --force < /agent_patch.diff 2>&1 || true
        echo "AGENT_PATCH_APPLIED_FALLBACK"
    fi

    # libcudart fix (same as vLLM path) — torchvision-bundled libcudart often
    # missing/zero in udocker-extracted images.
    REAL_CUDART=""
    for cand in /usr/local/cuda-12.6/lib64/libcudart.so.12 \\
                /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcudart.so.12 \\
                /usr/local/cuda-12/lib64/libcudart.so.12 \\
                /usr/local/cuda/lib64/libcudart.so.12 \\
                /usr/local/cuda-12.4/lib64/libcudart.so.12 \\
                /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcudart.so.12; do
        if [ -f "$cand" ]; then
            REAL_CUDART="$(readlink -f "$cand")"
            REAL_SZ=$(stat -c %s "$REAL_CUDART" 2>/dev/null || echo 0)
            [ "$REAL_SZ" -lt 100000 ] && REAL_CUDART="" && continue
            echo "REAL_CUDART found: $REAL_CUDART (size=$REAL_SZ)"
            break
        fi
    done
    if [ -n "$REAL_CUDART" ]; then
        for libroot in /usr/local/lib /usr/lib /opt/venv/lib /opt/conda/lib; do
            for pyver in python3.10 python3.11 python3.12 python3.13; do
                for sitedir in dist-packages site-packages; do
                    tvdir="$libroot/$pyver/$sitedir/torchvision.libs"
                    [ -d "$tvdir" ] || continue
                    for missing in "$tvdir"/libcudart.*.so.12; do
                        cp -fL "$REAL_CUDART" "$missing" 2>/dev/null && \\
                            echo "Patched libcudart: $missing"
                    done
                done
            done
        done
    fi
    for cdir in /usr/local/cuda-12.6/lib64 /usr/local/cuda-12/lib64 /usr/local/cuda/lib64; do
        [ -d "$cdir" ] && export LD_LIBRARY_PATH="$cdir:${{LD_LIBRARY_PATH:-}}"
    done

    # libcuda.so.1 fix for Triton — Triton's libcuda_dirs() probes specific
    # paths. udocker --nvidia bind-mounts libcuda.so.1 into /usr/lib/... but
    # Triton may not look there. Symlink to common paths just in case.
    HOST_LIBCUDA=""
    for cand in /usr/lib/x86_64-linux-gnu/libcuda.so.1 \\
                /usr/lib/x86_64-linux-gnu/libcuda.so \\
                /usr/local/cuda/compat/libcuda.so.1 \\
                /usr/local/cuda-12.6/compat/libcuda.so.1 \\
                /usr/local/nvidia/lib64/libcuda.so.1; do
        if [ -e "$cand" ] && [ -s "$cand" ]; then
            HOST_LIBCUDA="$cand"; break
        fi
    done
    if [ -n "$HOST_LIBCUDA" ]; then
        echo "Found libcuda.so.1 at $HOST_LIBCUDA"
        for tgt in /usr/lib/x86_64-linux-gnu/libcuda.so.1 \\
                   /usr/lib/x86_64-linux-gnu/libcuda.so \\
                   /usr/lib64/libcuda.so.1; do
            mkdir -p "$(dirname "$tgt")"
            [ -e "$tgt" ] || ln -sf "$HOST_LIBCUDA" "$tgt" 2>/dev/null
        done
        ldconfig 2>/dev/null || true
    else
        echo "WARN: libcuda.so.1 not found — Triton kernels will fail"
    fi

    # Find a Python with sglang. Prepend the source dir to PYTHONPATH so
    # the patched code wins over any pre-installed package.
    export PYTHONPATH="$SGL_DIR/python:${{PYTHONPATH:-}}"
    SGL_PYTHON=""
    for py in /opt/venv/bin/python3 /usr/local/bin/python3 /usr/bin/python3 $(which python3 2>/dev/null); do
        if [ -x "$py" ] && $py -c "import sglang" 2>/dev/null; then
            SGL_PYTHON="$py"
            echo "Found sglang at: $SGL_PYTHON  (PYTHONPATH=$PYTHONPATH)"
            break
        fi
    done
    [ -z "$SGL_PYTHON" ] && SGL_PYTHON="python3"

    # Start sglang server in background
    echo "=== Starting sglang server ==="
    cd /tmp
    $SGL_PYTHON -m sglang.launch_server \\
        --model-path "$MODEL" \\
        --port $SGL_PORT \\
        --host 127.0.0.1 \\
        --disable-cuda-graph 2>&1 &
    SERVER_PID=$!

    # Wait for server (TCP-level health check; sglang exposes /health)
    for i in $(seq 1 600); do
        if $SGL_PYTHON -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', $SGL_PORT)); s.close()" 2>/dev/null; then
            echo "SERVER_READY_AFTER=${{i}}s"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "SERVER_CRASHED"
            exit 1
        fi
        sleep 1
    done

    # Run the benchmark
    echo "=== Running sglang.bench_serving ==="
    $SGL_PYTHON -m sglang.bench_serving {bench_args} 2>&1 | tee /tmp/bench_output.txt

    echo "============ SGLang Benchmark Result ============"
    cat /tmp/bench_output.txt | grep -E "(TTFT|TPOT|ITL|throughput|Mean|Median|P99)" || true
    echo "================================================="

    kill $SERVER_PID 2>/dev/null || true
    echo "BENCHMARK_DONE"
    '''


def run_one(commit: str, info: dict, hf_token: str, timeout: int = 1800) -> dict:
    start = time.time()
    parent_commit = info["parent_commit"]
    model = info["model"]
    perf = info["perf_command"]
    patch_path = info["patch_path"]

    print(f"\n  --- Running SGLang AGENT benchmark for {commit} ---")
    print(f"  Model: {model}")
    print(f"  Parent: {parent_commit[:12]}")

    image = resolve_image(parent_commit)
    print(f"  Image: {image}")
    docker_cmd = build_docker_cmd(model, parent_commit, perf)

    try:
        r = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "--network=host",
             "-e", f"HF_TOKEN={hf_token}",
             "-e", f"HUGGING_FACE_HUB_TOKEN={hf_token}",
             "-v", "/ephemeral/huggingface_cache:/root/.cache/huggingface",
             "-v", f"{patch_path}:/agent_patch.diff",
             "--entrypoint", "bash",
             image, "-c", docker_cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        output = r.stdout + r.stderr
        duration = time.time() - start
        metrics = parse_serving_metrics(output)
        if metrics:
            return {"status": "success", "metrics": metrics, "duration_s": duration,
                    "raw_output": output[-50000:]}
        if "SERVER_CRASHED" in output:
            err = "Server crashed after applying patch"
        elif "SERVER_TIMEOUT" in output or r.returncode != 0:
            err = "Server startup timeout / nonzero exit"
        else:
            err = "No metrics in agent output"
        return {"status": "error", "error": err, "duration_s": duration,
                "metrics": {}, "raw_output": output[-50000:]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"Exceeded {timeout}s",
                "duration_s": timeout, "metrics": {}, "raw_output": ""}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}",
                "duration_s": time.time() - start, "metrics": {}, "raw_output": ""}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commits", nargs="+", help="Specific commits to run")
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()

    mapping = json.loads(MAPPING_FILE.read_text())
    commits = args.commits or list(mapping.keys())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hf_token = get_hf_token()

    print(f"=== SGLang OH benchmark — {len(commits)} commits ===")
    for i, c in enumerate(commits, 1):
        print(f"\n{'='*70}\n[{i}/{len(commits)}] {c}\n{'='*70}")
        info = mapping.get(c)
        if not info:
            print(f"  SKIP: {c} not in mapping")
            continue
        out_file = RESULTS_DIR / f"{c}_agent_result.json"
        if out_file.exists():
            print(f"  SKIP: result already exists at {out_file.name}")
            continue
        try:
            result = run_one(c, info, hf_token, args.timeout)
        except Exception as e:
            result = {"status": "error",
                      "error": f"unhandled: {type(e).__name__}: {e}",
                      "duration_s": 0, "metrics": {}, "raw_output": ""}
        record = {
            "human_commit": c,
            "human_commit_full": info["human_commit_full"],
            "parent_commit": info["parent_commit"],
            "model": info["model"],
            "perf_command": info["perf_command"],
            **result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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
