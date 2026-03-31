#!/usr/bin/env python3
"""
Batch pass@k agent benchmark runner.

For each task (item_id), starts ONE persistent Docker container, performs
one-time environment setup, then benchmarks all 8 agent patches sequentially
inside the same container — avoiding redundant setup overhead.

Agent patches are loaded from HuggingFace (Inferencebench/pass-at-k-samples).
Baseline and human benchmarks are NOT re-run (assumed already done).

Usage:
    # Dry run — show tasks, images, sample counts
    python scripts/runners/run_pass_at_k_benchmarks.py --dry-run

    # Run a single task
    python scripts/runners/run_pass_at_k_benchmarks.py --items vllm_core-0000

    # Full run with resume
    python scripts/runners/run_pass_at_k_benchmarks.py --resume

    # Custom HF repo and output
    python scripts/runners/run_pass_at_k_benchmarks.py \
        --hf-repo Inferencebench/pass-at-k-samples \
        --hf-split vllm \
        --output-dir results/pass_at_k_benchmarks/ \
        --resume
"""

import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pass_at_k_bench")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Docker image configuration
# ---------------------------------------------------------------------------
# vLLM: baseline images tagged as "baseline-{parent[:12]}"
VLLM_IMAGE_PREFIX = "shikhar481/vllm_fixed_human_images"
# SGLang: images tagged by full 40-char parent commit hash
SGLANG_IMAGE_REPO = "ayushnangia16/nvidia-sglang-docker"

# Legacy aliases
HUMAN_IMAGE_PREFIX = VLLM_IMAGE_PREFIX
BASELINE_IMAGE_PREFIX = VLLM_IMAGE_PREFIX
HF_CACHE_MOUNT = "/ephemeral/huggingface_cache"

# Model overrides for older vLLM compatibility (RoPE scaling)
MODEL_OVERRIDES = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "ibm-ai-platform/Bamba-9B-v2": "meta-llama/Meta-Llama-3-8B-Instruct",
    "ibm-ai-platform/Bamba-9B": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# ---------------------------------------------------------------------------
# Metric parsing (adapted from run_3way_benchmarks.py)
# ---------------------------------------------------------------------------

def parse_serving_metrics(output: str) -> Dict[str, float]:
    metrics = {}
    patterns = {
        'ttft_mean_ms': r'Mean TTFT \(ms\):\s+([\d.]+)',
        'ttft_median_ms': r'Median TTFT \(ms\):\s+([\d.]+)',
        'ttft_p99_ms': r'P99 TTFT \(ms\):\s+([\d.]+)',
        'tpot_mean_ms': r'Mean TPOT \(ms\):\s+([\d.]+)',
        'tpot_median_ms': r'Median TPOT \(ms\):\s+([\d.]+)',
        'tpot_p99_ms': r'P99 TPOT \(ms\):\s+([\d.]+)',
        'itl_mean_ms': r'Mean ITL \(ms\):\s+([\d.]+)',
        'itl_median_ms': r'Median ITL \(ms\):\s+([\d.]+)',
        'itl_p99_ms': r'P99 ITL \(ms\):\s+([\d.]+)',
        'request_throughput_req_s': r'Request throughput \(req/s\):\s+([\d.]+)',
        'output_token_throughput_tok_s': r'Output token throughput \(tok/s\):\s+([\d.]+)',
        'total_token_throughput_tok_s': r'Total Token throughput \(tok/s\):\s+([\d.]+)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def parse_latency_metrics(output: str) -> Dict[str, float]:
    metrics = {}
    patterns = {
        'latency_avg_ms': r'(?:Avg latency|avg_latency):\s*([\d.]+)\s*(?:ms|seconds)?',
        'latency_p50_ms': r'(?:P50 latency|median_latency):\s*([\d.]+)\s*(?:ms|seconds)?',
        'latency_p99_ms': r'(?:P99 latency|p99_latency):\s*([\d.]+)\s*(?:ms|seconds)?',
        'throughput_tok_s': r'(?:Throughput|throughput):\s*([\d.]+)\s*(?:tokens?/s|tok/s)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if 'latency' in key and 'seconds' in output[max(0, match.start()-20):match.end()+20].lower():
                val *= 1000
            metrics[key] = val
    return metrics


def parse_throughput_metrics(output: str) -> Dict[str, float]:
    metrics = {}
    patterns = {
        'throughput_tok_s': r'(?:Throughput|throughput):\s*([\d.]+)\s*(?:tokens?/s|tok/s)',
        'elapsed_time_s': r'(?:Elapsed time|elapsed_time):\s*([\d.]+)\s*(?:s|seconds)?',
        'total_tokens': r'(?:Total tokens|total_tokens):\s*([\d.]+)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))
    if 'throughput_tok_s' not in metrics:
        old_format = re.search(r'Throughput:\s*[\d.]+\s*requests/s,\s*([\d.]+)\s*tokens/s', output)
        if old_format:
            metrics['throughput_tok_s'] = float(old_format.group(1))
    return metrics


def parse_prefix_caching_metrics(output: str) -> Dict[str, float]:
    metrics = {}
    start_gen_idx = output.rfind('------start generating------')
    gen_output = output[start_gen_idx:] if start_gen_idx != -1 else output
    speed_matches = list(re.finditer(
        r'est\. speed input:\s*([\d.]+)\s*toks/s,\s*output:\s*([\d.]+)\s*toks/s', gen_output
    ))
    if speed_matches:
        last = speed_matches[-1]
        metrics['input_throughput_tok_s'] = float(last.group(1))
        metrics['throughput_tok_s'] = float(last.group(2))
    cost_matches = list(re.finditer(r'cost time\s+([\d.]+)', gen_output))
    if cost_matches:
        metrics['elapsed_time_s'] = float(cost_matches[-1].group(1))
    return metrics


def parse_metrics_by_type(output: str, benchmark_type: str) -> Dict[str, float]:
    if benchmark_type == 'latency':
        return parse_latency_metrics(output)
    elif benchmark_type == 'throughput':
        return parse_throughput_metrics(output)
    elif benchmark_type == 'standalone':
        metrics = parse_throughput_metrics(output)
        metrics.update(parse_latency_metrics(output))
        return metrics
    elif benchmark_type == 'prefix_caching':
        return parse_prefix_caching_metrics(output)
    else:
        return parse_serving_metrics(output)


def get_benchmark_type(perf_command: str) -> str:
    if not perf_command:
        return 'serving'
    if 'benchmark_latency' in perf_command or 'bench latency' in perf_command:
        return 'latency'
    if 'benchmark_throughput' in perf_command or 'bench throughput' in perf_command:
        return 'throughput'
    if 'benchmark_prefix_caching' in perf_command:
        return 'prefix_caching'
    return 'serving'


def needs_server(perf_command: str) -> bool:
    if not perf_command:
        return True
    lower = perf_command.lower()
    if 'benchmark_throughput' in lower or 'bench throughput' in lower:
        return False
    if 'benchmark_latency' in lower or 'bench latency' in lower:
        return False
    if 'benchmark_prefix_caching' in lower:
        return False
    return True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pass_at_k_patches(
    hf_repo: str,
    repo_prefix: str,
    agent_configs: List[Tuple[str, str]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Load pass@k patches from HuggingFace per-sample shards, grouped by item_id.

    The HF repo has two types of parquet files:
    1. Consolidated: data/vllm/train.parquet, data/sglang/train.parquet
       (unreliable — contains Bedrock-auth failure rows that overwrite good data)
    2. Per-sample shards: data/vllm_core-0003_s0_1774698610.parquet
       (reliable — one row per shard, pushed after verified collection)

    This function reads ONLY per-sample shards, deduplicates by (item_id, sample_index),
    keeping the shard with a valid non-empty model_patch (preferring latest timestamp).

    Args:
        hf_repo: HuggingFace repo ID
        repo_prefix: Filter item_ids by prefix (e.g., "vllm")
        agent_configs: List of (agent_name, model_name) tuples to include
    """
    import re as _re
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd

    agent_set = set(agent_configs)
    log.info(f"Loading per-shard patches from {hf_repo} (prefix={repo_prefix}, agents={agent_configs})...")

    api = HfApi()
    all_files = api.list_repo_files(hf_repo, repo_type="dataset")

    # Filter to per-sample shards only (skip consolidated train.parquet and standard train-NNNNN shards)
    shard_files = [f for f in all_files
                   if f.endswith(".parquet")
                   and "_s" in f.split("/")[-1]
                   and "train" not in f.split("/")[-1]]

    log.info(f"Found {len(shard_files)} per-sample shards on HF")

    # Pre-filter by repo prefix from filename (avoid downloading irrelevant shards)
    if repo_prefix:
        shard_files = [f for f in shard_files if repo_prefix in f.split("/")[-1]]
        log.info(f"After prefix filter '{repo_prefix}': {len(shard_files)} shards")

    # Download and read each shard — they're tiny (1 row, ~1-50KB each)
    # Collect all rows, then deduplicate
    COLUMNS = ["item_id", "sample_index", "agent_name", "model_name", "status",
               "human_commit", "pre_commit", "model_patch", "patch_size_loc",
               "changed_files_count", "collected_at"]

    all_rows: List[Dict[str, Any]] = []
    errors = 0

    for i, shard_path in enumerate(shard_files):
        if i > 0 and i % 200 == 0:
            log.info(f"  Reading shards... {i}/{len(shard_files)}")
        try:
            local = hf_hub_download(hf_repo, shard_path, repo_type="dataset")
            df = pd.read_parquet(local, columns=[c for c in COLUMNS if c != "model_patch"])
            # Only read model_patch if we pass filters (saves memory)
            row = df.iloc[0]
            item_id = str(row.get("item_id", ""))
            agent = str(row.get("agent_name", ""))
            model = str(row.get("model_name", ""))

            if not item_id:
                continue
            if repo_prefix and not item_id.startswith(repo_prefix):
                continue
            if (agent, model) not in agent_set:
                continue

            # Now read model_patch for this matching shard
            df_full = pd.read_parquet(local, columns=["model_patch"])
            patch = str(df_full["model_patch"].iloc[0]) if not df_full["model_patch"].isna().iloc[0] else ""

            all_rows.append({
                "item_id": item_id,
                "sample_index": int(row.get("sample_index", 0)),
                "model_patch": patch,
                "human_commit": str(row.get("human_commit", "")),
                "pre_commit": str(row.get("pre_commit", "")),
                "status": str(row.get("status", "")),
                "agent_name": agent,
                "model_name": model,
                "collected_at": str(row.get("collected_at", "")),
                "_shard": shard_path,
            })
        except Exception as e:
            errors += 1
            if errors <= 5:
                log.warning(f"  Error reading {shard_path}: {e}")

    log.info(f"Read {len(all_rows)} matching rows from shards ({errors} errors)")

    # Deduplicate: for each (item_id, agent_name, model_name, sample_index),
    # keep the row with a valid non-empty model_patch, preferring latest collected_at
    dedup_key = lambda r: (r["item_id"], r["agent_name"], r["model_name"], r["sample_index"])

    candidates: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        candidates[dedup_key(row)].append(row)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_with_patch = 0
    total_empty = 0

    for key, rows in candidates.items():
        # Prefer rows with non-empty patch, then latest collected_at
        rows_with_patch = [r for r in rows if r["model_patch"] and r["model_patch"].strip()]
        if rows_with_patch:
            # Pick latest
            best = max(rows_with_patch, key=lambda r: r.get("collected_at", ""))
            total_with_patch += 1
        else:
            best = max(rows, key=lambda r: r.get("collected_at", ""))
            total_empty += 1

        # Remove internal fields
        best.pop("_shard", None)
        best.pop("collected_at", None)
        grouped[best["item_id"]].append(best)

    # Sort each group
    for item_id in grouped:
        grouped[item_id].sort(key=lambda s: (s["agent_name"], s["model_name"], s["sample_index"]))

    total = sum(len(v) for v in grouped.values())
    log.info(f"Deduplicated to {total} unique samples across {len(grouped)} tasks "
             f"({total_with_patch} with patch, {total_empty} empty)")
    return dict(grouped)


LOCAL_PATCH_BASE = Path("/home/ubuntu/everything_analysis_data/pass_at_k")

# Agent/model -> local directory mapping
LOCAL_AGENT_DIRS = {
    ("claude_code", "sonnet"): LOCAL_PATCH_BASE / "claude_code" / "sonnet",
    ("codex_cli", "gpt-5"): LOCAL_PATCH_BASE / "codex_cli" / "gpt-5",
}


def enrich_with_local_patches(
    grouped: Dict[str, List[Dict[str, Any]]],
    agent_configs: List[Tuple[str, str]],
    plan_items: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Fill in empty patches from local everything_analysis_data.

    For each (item_id, agent, sample) that has empty model_patch in HF data,
    check if the local directory has a non-empty model_patch.diff and use that.
    Also adds entirely new samples if they exist locally but not on HF.
    """
    if not LOCAL_PATCH_BASE.exists():
        log.warning(f"Local patch base not found: {LOCAL_PATCH_BASE}")
        return grouped

    filled = 0
    added = 0

    for agent_name, model_name in agent_configs:
        local_dir = LOCAL_AGENT_DIRS.get((agent_name, model_name))
        if not local_dir or not local_dir.exists():
            continue

        for item_id, info in plan_items.items():
            repo = 'vllm' if item_id.startswith('vllm') else 'sglang'
            task_dir = local_dir / repo / item_id

            if not task_dir.exists():
                continue

            # Build set of existing (agent, sample_index) with patches
            existing = set()
            if item_id in grouped:
                for s in grouped[item_id]:
                    if s['agent_name'] == agent_name and s['model_name'] == model_name:
                        if s['model_patch'] and s['model_patch'].strip():
                            existing.add(s['sample_index'])

            for sample_dir in sorted(task_dir.iterdir()):
                if not sample_dir.is_dir() or not sample_dir.name.startswith('s'):
                    continue
                try:
                    sidx = int(sample_dir.name[1:])
                except ValueError:
                    continue

                if sidx in existing:
                    continue  # Already have a patch for this sample

                patch_file = sample_dir / 'model_patch.diff'
                if not patch_file.exists() or patch_file.stat().st_size < 10:
                    continue

                patch_text = patch_file.read_text()

                # Read journal for human_commit/pre_commit
                journal_file = sample_dir / 'journal.json'
                human_commit = info['human']
                pre_commit = ''
                if journal_file.exists():
                    try:
                        j = json.loads(journal_file.read_text())
                        human_commit = j.get('commits', {}).get('human', human_commit)
                        pre_commit = j.get('commits', {}).get('pre', '')
                    except Exception:
                        pass

                sample = {
                    "item_id": item_id,
                    "sample_index": sidx,
                    "model_patch": patch_text,
                    "human_commit": human_commit,
                    "pre_commit": pre_commit,
                    "status": "success",
                    "agent_name": agent_name,
                    "model_name": model_name,
                }

                # Check if we're filling an empty or adding new
                filled_existing = False
                if item_id in grouped:
                    for s in grouped[item_id]:
                        if (s['agent_name'] == agent_name and
                            s['model_name'] == model_name and
                            s['sample_index'] == sidx and
                            (not s['model_patch'] or not s['model_patch'].strip())):
                            s['model_patch'] = patch_text
                            s['human_commit'] = human_commit
                            s['pre_commit'] = pre_commit
                            filled += 1
                            filled_existing = True
                            break

                if not filled_existing:
                    if item_id not in grouped:
                        grouped[item_id] = []
                    grouped[item_id].append(sample)
                    added += 1

    # Re-sort
    for item_id in grouped:
        grouped[item_id].sort(key=lambda s: (s["agent_name"], s["model_name"], s["sample_index"]))

    log.info(f"Local patches: filled {filled} empty slots, added {added} new samples")
    return grouped


def load_task_configs(
    plan_path: Path,
    mapping_path: Path,
    task_patches: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load task configs by joining plan_iso.json + HF data with benchmark_mode_mapping.json.

    Uses two strategies to resolve item_id → benchmark config:
    1. Direct match: item_id in plan → human_commit → mapping lookup
    2. HF fallback: item_id has human_commit in HF data → mapping lookup
       (handles cases where HF and plan use different item_id numbering)
    """
    with open(mapping_path) as f:
        mapping = json.load(f)

    configs = {}

    # Strategy 1: From plan file
    if plan_path.exists():
        with open(plan_path) as f:
            plan = json.load(f)
        for item in plan["items"]:
            item_id = item["item_id"]
            human_commit = item["human"]
            human_short = human_commit[:8]
            if human_short in mapping:
                configs[item_id] = _build_config(item_id, human_commit, mapping[human_short])

    # Strategy 2: From HF data (fills in items not in plan or with different numbering)
    if task_patches:
        for item_id, samples in task_patches.items():
            if item_id in configs:
                continue  # Already resolved via plan
            # Get human_commit from the HF samples themselves
            human_commit = ""
            for s in samples:
                if s.get("human_commit"):
                    human_commit = s["human_commit"]
                    break
            if not human_commit:
                continue
            human_short = human_commit[:8]
            if human_short in mapping:
                configs[item_id] = _build_config(item_id, human_commit, mapping[human_short])
                log.info(f"  {item_id}: resolved via HF human_commit {human_short}")
            else:
                log.warning(f"  {item_id}: human commit {human_short} not in benchmark mapping")

    log.info(f"Loaded configs for {len(configs)} tasks")
    return configs


def _build_config(item_id: str, human_commit: str, m: Dict[str, Any]) -> Dict[str, Any]:
    """Build a task config dict from a benchmark mapping entry."""
    human_short = human_commit[:8]
    parent_commit = m.get("parent_commit", "")
    model = m.get("model", "")
    original_model = model

    # Only apply model overrides for vLLM (older vLLM has RoPE issues with Llama-3.1)
    # SGLang Docker images are built with the original model — no override needed
    is_sglang = item_id.startswith("sglang")
    if not is_sglang:
        model = MODEL_OVERRIDES.get(model, model)

    perf_command = m.get("perf_command") or ""
    if model != original_model and original_model and original_model in perf_command:
        perf_command = perf_command.replace(original_model, model)

    return {
        "item_id": item_id,
        "human_commit": human_commit,
        "human_commit_short": human_short,
        "parent_commit": parent_commit,
        "perf_command": perf_command,
        "model": model,
        "original_model": original_model,
        "benchmark_mode": m.get("benchmark_mode", "serving"),
    }


# ---------------------------------------------------------------------------
# Resume tracking
# ---------------------------------------------------------------------------

def load_completed(output_dir: Path) -> Set[Tuple[str, str, str, int]]:
    """Returns set of (item_id, agent_name, model_name, sample_index) already completed.

    Only counts 'success' and 'benchmark_failed' as truly completed.
    Statuses like 'empty_patch' and 'image_not_found' are NOT counted —
    the patches may now exist on HF (data was updated) or images may have been built.
    """
    TERMINAL_STATUSES = {"success", "benchmark_failed"}
    results_file = output_dir / "agent_benchmark_results.jsonl"
    completed = set()
    if not results_file.exists():
        return completed
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                status = row.get("status", "")
                if status not in TERMINAL_STATUSES:
                    continue
                item_id = row.get("item_id", "")
                agent = row.get("agent_name", "")
                model = row.get("model_name", "")
                sample_index = row.get("sample_index", -1)
                if item_id and sample_index >= 0:
                    completed.add((item_id, agent, model, sample_index))
            except json.JSONDecodeError:
                continue
    return completed


def save_result(output_dir: Path, result: Dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "agent_benchmark_results.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.exists():
            token = token_file.read_text().strip()
    return token


def check_docker_image_exists(image: str) -> bool:
    """Check if image exists locally or can be pulled."""
    # Check locally first
    try:
        result = subprocess.run(
            ['docker', 'image', 'inspect', image],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass
    # Try pulling (manifest inspect often needs auth)
    log.info(f"  Pulling image {image}...")
    try:
        result = subprocess.run(
            ['docker', 'pull', image],
            capture_output=True, text=True, timeout=1800,  # 30 min for large images
        )
        if result.returncode == 0:
            return True
        log.warning(f"  Pull failed: {result.stderr.strip()[:200]}")
    except subprocess.TimeoutExpired:
        log.warning(f"  Pull timed out after 1800s for {image}")
    except Exception as e:
        log.warning(f"  Pull error: {e}")
    return False


def start_task_container(
    parent_commit: str,
    hf_token: str,
    image_override: str = None,
) -> str:
    """Start a persistent baseline container. Returns container_id."""
    baseline_image = image_override or f"{BASELINE_IMAGE_PREFIX}:baseline-{parent_commit[:12]}"
    log.info(f"Starting container from {baseline_image}...")

    sharegpt_mount = []
    sharegpt_path = ROOT_DIR / 'data/sharegpt_dataset.json'
    if sharegpt_path.exists():
        sharegpt_mount = ['-v', f'{sharegpt_path}:/data/sharegpt_dataset.json:ro']

    cmd = [
        'docker', 'run', '-d',
        '--gpus', 'all',
        '--shm-size=16g',
        '-e', f'HF_TOKEN={hf_token}',
        '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
        '-e', 'VLLM_USE_V1=0',
        '-v', f'{HF_CACHE_MOUNT}:/root/.cache/huggingface',
    ] + sharegpt_mount + [
        '--entrypoint', 'sleep',
        baseline_image,
        'infinity',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container: {result.stderr.strip()}")

    container_id = result.stdout.strip()[:12]
    log.info(f"Container started: {container_id}")
    return container_id


def docker_exec(container_id: str, script: str, timeout: int = 600) -> Tuple[int, str]:
    """Run a bash script inside the container. Returns (returncode, combined output)."""
    result = subprocess.run(
        ['docker', 'exec', container_id, 'bash', '-c', script],
        capture_output=True, text=True, timeout=timeout,
    )
    output = result.stdout + result.stderr
    return result.returncode, output


def docker_cp_to(container_id: str, local_path: str, container_path: str):
    """Copy a file into the container."""
    subprocess.run(
        ['docker', 'cp', local_path, f'{container_id}:{container_path}'],
        check=True, capture_output=True, timeout=30,
    )


def teardown_container(container_id: str):
    """Stop and remove the container."""
    log.info(f"Tearing down container {container_id}...")
    subprocess.run(['docker', 'stop', '-t', '5', container_id],
                   capture_output=True, timeout=30)
    subprocess.run(['docker', 'rm', '-f', container_id],
                   capture_output=True, timeout=30)


# ---------------------------------------------------------------------------
# One-time container setup
# ---------------------------------------------------------------------------

def setup_container(
    container_id: str,
    human_commit: str,
    perf_command: str,
    model: str,
    benchmark_mode: str,
) -> bool:
    """One-time setup inside the container. Returns True on success."""
    benchmark_type = get_benchmark_type(perf_command)
    server_needed = needs_server(perf_command)

    setup_script = f'''
set -e
echo "=== ONE-TIME SETUP ==="

# --- Compatibility fixes ---
cd /opt/vllm_baseline

VLLM_USES_MLLAMA=$(find /opt/vllm_baseline -name "*mllama*" 2>/dev/null | head -1)

if [ -n "$VLLM_USES_MLLAMA" ]; then
    if ! python3 -c "from transformers.models.mllama import configuration_mllama" 2>/dev/null; then
        pip install 'transformers>=4.45.0' -q 2>/dev/null || true
    fi
else
    if ! python3 -c "from transformers.generation.logits_process import LogitsWarper" 2>/dev/null; then
        pip install 'transformers==4.44.2' -q 2>/dev/null || true
    fi
fi

pip install 'numpy<2' -q 2>/dev/null || true

# Fix missing modules that various vLLM versions need
pip install lark sentencepiece -q 2>/dev/null || true

# Monkey-patch: stub out guided_decoding to avoid outlines.fsm.guide import crash
# Safe because benchmark_serving.py doesn't use guided decoding
if [ -d "/opt/vllm_baseline/vllm/model_executor/guided_decoding" ]; then
    cat > /opt/vllm_baseline/vllm/model_executor/guided_decoding/__init__.py << 'GUIDEDPATCH'
from typing import Optional
async def get_guided_decoding_logits_processor(*args, **kwargs):
    return None
async def get_local_guided_decoding_logits_processor(*args, **kwargs):
    return None
GUIDEDPATCH
    echo "Monkey-patched guided_decoding (outlines fix)"
fi

# Fix broken transformers
if ! python3 -c "from transformers.utils import logging" 2>/dev/null; then
    pip install --force-reinstall 'transformers>=4.44.0,<5' -q 2>/dev/null || true
fi

# Fix lm-format-enforcer if needed
if ! python3 -c "import vllm" 2>/dev/null; then
    pip install --upgrade lm-format-enforcer 2>/dev/null || true
fi

# Fix aimv2 registration conflict
for ovis_file in /opt/vllm_baseline/vllm/transformers_utils/configs/ovis.py /opt/vllm_baseline/vllm/transformers_utils/configs/ovis2.py; do
    if [ -f "$ovis_file" ] && grep -q 'AutoConfig.register("aimv2"' "$ovis_file" 2>/dev/null; then
        if ! grep -q 'exist_ok=True' "$ovis_file" 2>/dev/null; then
            sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' "$ovis_file"
        fi
    fi
done

# Install benchmark dependencies
pip install aiohttp pandas datasets pyairports pycountry -q 2>/dev/null || true

# --- Download benchmark scripts from human commit (with fallback to baseline) ---
mkdir -p /opt/bench_scripts
cd /opt/bench_scripts
HUMAN_COMMIT="{human_commit}"
for script in benchmark_latency.py benchmark_throughput.py benchmark_serving.py backend_request_func.py benchmark_prefix_caching.py benchmark_utils.py benchmark_dataset.py sonnet.py; do
    curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/$HUMAN_COMMIT/benchmarks/$script" -o "$script" 2>/dev/null || \\
    wget -q "https://raw.githubusercontent.com/vllm-project/vllm/$HUMAN_COMMIT/benchmarks/$script" -O "$script" 2>/dev/null || \\
    curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/$script" -o "$script" 2>/dev/null || true
done
# Fallback: copy from baseline vLLM if download failed (containers without curl/wget)
for script in benchmark_latency.py benchmark_throughput.py benchmark_serving.py backend_request_func.py benchmark_prefix_caching.py; do
    if [ ! -s "/opt/bench_scripts/$script" ] && [ -f "/opt/vllm_baseline/benchmarks/$script" ]; then
        cp "/opt/vllm_baseline/benchmarks/$script" "/opt/bench_scripts/$script"
        echo "Copied $script from baseline"
    fi
done
# Stubs for modules that might not exist in older vLLM
if [ ! -s /opt/bench_scripts/benchmark_utils.py ]; then
    echo "def convert_to_pytorch_benchmark_format(*args, **kwargs): pass" > /opt/bench_scripts/benchmark_utils.py
fi
if [ ! -s /opt/bench_scripts/benchmark_dataset.py ]; then
    cat > /opt/bench_scripts/benchmark_dataset.py << 'STUBEOF'
class AIMODataset: pass
class ASRDataset: pass
class BurstGPTDataset: pass
class ConversationDataset: pass
class HumanEvalDataset: pass
class InstructCoderDataset: pass
class LongContextDataset: pass
class ShareGPTDataset: pass
class SonnetDataset: pass
class SyntheticDataset: pass
class VisionArenaDataset: pass
STUBEOF
fi
echo "Benchmark scripts downloaded"

# --- Create sonnet dataset for serving benchmarks ---
if [ ! -f /opt/bench_scripts/sonnet.txt ]; then
    python3 -c "
lines = ['Shall I compare thee to a summers day? ' * 10] * 500
with open('/opt/bench_scripts/sonnet.txt', 'w') as f:
    f.write(chr(10).join(lines))
" 2>/dev/null || true
fi

# --- Backup clean vLLM source ---
echo "Backing up clean vLLM source..."
cp -r /opt/vllm_baseline/vllm /opt/vllm_clean
echo "Clean backup at /opt/vllm_clean"

# --- Find Python with vLLM ---
VLLM_PYTHON=""
for py in /opt/venv/bin/python3 /opt/venv/bin/python /usr/local/bin/python3 /usr/bin/python3 python3; do
    if [ -x "$py" ] 2>/dev/null || [ -x "$(which $py 2>/dev/null)" ]; then
        if PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH $py -c "import vllm" 2>/dev/null; then
            VLLM_PYTHON="$py"
            break
        fi
    fi
done
# Fallback: if no python found with vllm, try any python3 with PYTHONPATH
if [ -z "$VLLM_PYTHON" ]; then
    for py in /usr/local/bin/python3 /usr/bin/python3 python3; do
        if [ -x "$py" ] 2>/dev/null || [ -x "$(which $py 2>/dev/null)" ]; then
            VLLM_PYTHON="$py"
            echo "WARNING: Using $py without vllm import check"
            break
        fi
    done
fi
echo "export VLLM_PYTHON=$VLLM_PYTHON" > /opt/bench_env.sh
echo "export PYTHONPATH=/opt/vllm_baseline:\\$PYTHONPATH" >> /opt/bench_env.sh

# Verify
source /opt/bench_env.sh
$VLLM_PYTHON -c "import vllm; print('vLLM', vllm.__version__, 'OK')" 2>/dev/null || echo "vLLM import check done"

echo "SETUP_COMPLETE"
'''

    log.info("Running one-time setup...")
    rc, output = docker_exec(container_id, setup_script, timeout=600)

    if 'SETUP_COMPLETE' not in output:
        log.error(f"Setup failed (rc={rc}):\n{output[-3000:]}")
        return False

    log.info("Setup complete")
    return True


# ---------------------------------------------------------------------------
# Per-sample benchmark execution
# ---------------------------------------------------------------------------

def build_benchmark_script(perf_command: str, model: str, benchmark_mode: str) -> str:
    """Build the bash script that runs INSIDE the container for one sample."""
    benchmark_type = get_benchmark_type(perf_command)
    server_needed = needs_server(perf_command)

    # Common preamble: restore clean vLLM, apply patch
    script = '''
source /opt/bench_env.sh

echo "=== Restoring clean vLLM ==="
rm -rf /opt/vllm_baseline/vllm
cp -r /opt/vllm_clean /opt/vllm_baseline/vllm

echo "=== Applying agent patch ==="
cd /opt/vllm_baseline
if patch -p1 --dry-run < /tmp/current_patch.diff 2>&1; then
    patch -p1 < /tmp/current_patch.diff 2>&1
    echo "AGENT_PATCH_APPLIED"
else
    echo "Patch dry-run failed, trying --force..."
    patch -p1 --force < /tmp/current_patch.diff 2>&1 || true
    echo "AGENT_PATCH_APPLIED_FALLBACK"
fi

# Apply rope_scaling fix
find /opt/vllm_baseline/vllm -name "*.py" -exec grep -l 'rope_scaling\\["type"\\]' {} \\; 2>/dev/null | while read f; do
    sed -i 's/rope_scaling\\["type"\\]/rope_scaling.get("type", rope_scaling.get("rope_type"))/g' "$f"
done

'''

    if server_needed:
        # Use regular string (not f-string) for bash variables, inject Python vars manually
        script += '''
echo "=== Starting vLLM server ==="
MODEL="''' + model + '''"

# Kill any leftover server
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# Start server in background
set +e
$VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port 8000 --disable-log-requests > /tmp/server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server ready
MAX_WAIT=900
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models',timeout=3)" 2>/dev/null; then
        echo "SERVER_READY after ${WAITED}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "SERVER_CRASHED"
        echo "=== Server log ==="
        cat /tmp/server.log | tail -50
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "SERVER_TIMEOUT"
    echo "=== Server log ==="
    cat /tmp/server.log | tail -50
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

set -e

echo "=== Running serving benchmark ==="
# Detect supported dataset names (old vLLM only supports sharegpt/sonnet, not random)
DATASET_ARGS="--dataset-name random --random-input-len 256 --random-output-len 64"
if ! $VLLM_PYTHON /opt/bench_scripts/benchmark_serving.py --help 2>&1 | grep -q "random"; then
    DATASET_ARGS="--dataset-name sonnet --sonnet-input-len 256 --sonnet-output-len 64 --sonnet-prefix-len 50"
    # Create sonnet.txt if needed
    if [ ! -f /opt/bench_scripts/sonnet.txt ]; then
        $VLLM_PYTHON -c "
lines = ['Shall I compare thee to a summers day? ' * 10] * 500
with open('/opt/bench_scripts/sonnet.txt', 'w') as f:
    f.write(chr(10).join(lines))
" 2>/dev/null
    fi
    DATASET_ARGS="$DATASET_ARGS --dataset-path /opt/bench_scripts/sonnet.txt"
fi

$VLLM_PYTHON /opt/bench_scripts/benchmark_serving.py \
    --model "$MODEL" \
    --backend vllm \
    --port 8000 \
    $DATASET_ARGS \
    --num-prompts 100 \
    --request-rate inf \
    2>&1 | tee /tmp/benchmark_output.txt

echo "=== Stopping server ==="
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true

echo "BENCHMARK_DONE"
cat /tmp/benchmark_output.txt
'''
    else:
        # Offline benchmark (throughput, latency, prefix_caching)
        script += '''
echo "=== Running offline ''' + benchmark_type + ''' benchmark ==="
MODEL="''' + model + '''"
PERF_CMD="''' + perf_command + '''"

# Convert vllm bench commands to old-style scripts
if echo "$PERF_CMD" | grep -q "vllm bench throughput"; then
    ARGS=$(echo "$PERF_CMD" | sed 's/vllm bench throughput//')
    PERF_CMD="$VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_throughput.py $ARGS"
elif echo "$PERF_CMD" | grep -q "vllm bench latency"; then
    ARGS=$(echo "$PERF_CMD" | sed 's/vllm bench latency//')
    PERF_CMD="$VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_latency.py $ARGS"
elif echo "$PERF_CMD" | grep -q "benchmark_latency"; then
    ARGS=$(echo "$PERF_CMD" | sed -E 's|.*benchmark_latency\\.py\\s*||')
    PERF_CMD="$VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_latency.py $ARGS"
elif echo "$PERF_CMD" | grep -q "benchmark_throughput"; then
    ARGS=$(echo "$PERF_CMD" | sed -E 's|.*benchmark_throughput\\.py\\s*||')
    PERF_CMD="$VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_throughput.py $ARGS"
elif echo "$PERF_CMD" | grep -q "benchmark_prefix_caching"; then
    ARGS=$(echo "$PERF_CMD" | sed -E 's|.*benchmark_prefix_caching\\.py\\s*||')
    PERF_CMD="$VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_prefix_caching.py $ARGS"
elif echo "$PERF_CMD" | grep -q "^python"; then
    PERF_CMD=$(echo "$PERF_CMD" | sed "s|^python3\\? |$VLLM_PYTHON |")
fi

# For throughput: translate --dataset-name sharegpt
if echo "$PERF_CMD" | grep -q "benchmark_throughput"; then
    if echo "$PERF_CMD" | grep -q "\\-\\-dataset-name sharegpt"; then
        PERF_CMD=$(echo "$PERF_CMD" | sed 's/--dataset-name sharegpt/--dataset \\/data\\/sharegpt_dataset.json/')
    fi
    # Add default input/output lengths if missing and no dataset file
    if ! echo "$PERF_CMD" | grep -q "\\-\\-input-len"; then
        if ! echo "$PERF_CMD" | grep -q "\\-\\-dataset "; then
            PERF_CMD="$PERF_CMD --input-len 512 --output-len 128"
        fi
    fi
fi

echo "Final command: $PERF_CMD"
cd /opt/vllm_baseline
PYTHONPATH=/opt/vllm_baseline:$PYTHONPATH eval $PERF_CMD 2>&1 | tee /tmp/benchmark_output.txt

echo "BENCHMARK_DONE"
cat /tmp/benchmark_output.txt
'''

    return script


def run_agent_in_container(
    container_id: str,
    patch_text: str,
    perf_command: str,
    model: str,
    benchmark_mode: str,
    timeout: int,
    is_sglang: bool = False,
) -> Dict[str, Any]:
    """Apply one patch and run benchmark inside the container."""
    start_time = time.time()
    benchmark_type = get_benchmark_type(perf_command) if not is_sglang else ('serving' if 'bench_serving' in (perf_command or '') else 'offline')

    # Write patch to temp file and copy into container
    with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False) as f:
        f.write(patch_text)
        tmp_patch = f.name

    try:
        docker_cp_to(container_id, tmp_patch, '/tmp/current_patch.diff')
    finally:
        os.unlink(tmp_patch)

    # Build the benchmark script (different for vLLM vs SGLang)
    if is_sglang:
        bench_script = build_sglang_benchmark_script(perf_command, model, benchmark_mode)
    else:
        bench_script = build_benchmark_script(perf_command, model, benchmark_mode)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(bench_script)
        tmp_script = f.name

    try:
        docker_cp_to(container_id, tmp_script, '/tmp/run_benchmark.sh')
    finally:
        os.unlink(tmp_script)

    try:
        rc, output = docker_exec(container_id, 'bash /tmp/run_benchmark.sh', timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill any leftover processes inside the container
        try:
            docker_exec(container_id, 'pkill -9 -f "vllm.entrypoints" 2>/dev/null; pkill -9 -f benchmark 2>/dev/null', timeout=10)
        except Exception:
            pass
        return {
            'status': 'timeout',
            'error': f'Benchmark timed out after {timeout}s',
            'duration_s': timeout,
            'benchmark_type': benchmark_type,
        }

    duration = time.time() - start_time

    # Check for known failures
    if 'SERVER_CRASHED' in output:
        return {
            'status': 'benchmark_failed',
            'error': 'vLLM server crashed after applying patch',
            'duration_s': duration,
            'benchmark_type': benchmark_type,
            'raw_output_tail': output[-5000:],
        }
    if 'SERVER_TIMEOUT' in output:
        return {
            'status': 'benchmark_failed',
            'error': 'vLLM server failed to start within timeout',
            'duration_s': duration,
            'benchmark_type': benchmark_type,
            'raw_output_tail': output[-5000:],
        }

    # Check if patch was applied
    patch_applied = 'AGENT_PATCH_APPLIED' in output
    if not patch_applied:
        return {
            'status': 'patch_failed',
            'error': 'Patch application failed',
            'duration_s': duration,
            'benchmark_type': benchmark_type,
            'raw_output_tail': output[-5000:],
        }

    # Parse metrics
    if 'BENCHMARK_DONE' not in output:
        return {
            'status': 'benchmark_failed',
            'error': 'Benchmark did not complete (no BENCHMARK_DONE marker)',
            'duration_s': duration,
            'benchmark_type': benchmark_type,
            'raw_output_tail': output[-5000:],
        }

    metrics = parse_metrics_by_type(output, benchmark_type)
    if not metrics:
        return {
            'status': 'benchmark_failed',
            'error': f'No {benchmark_type} metrics found in output',
            'duration_s': duration,
            'benchmark_type': benchmark_type,
            'raw_output_tail': output[-5000:],
        }

    return {
        'status': 'success',
        'metrics': metrics,
        'duration_s': duration,
        'benchmark_type': benchmark_type,
        'raw_output_tail': output[-5000:],
    }



# ---------------------------------------------------------------------------
# SGLang Docker-based benchmarking
# ---------------------------------------------------------------------------


def get_sglang_image(parent_commit: str) -> str:
    """Get best Docker image for SGLang parent commit.

    Tries multiple repos and tag formats in priority order:
    1. shikhar481/sglang-images:{commit[:8]} (direct, proven working)
    2. shikhar481/sglang-images:v04x-fixed-{commit[:12]}
    3. shikhar481/sglang-images:v04x-triton-{commit[:12]}
    4. shikhar481/sglang-images:{commit[:8]}-src
    5. ayushnangia16/nvidia-sglang-docker:{full_40char} (fallback)
    """
    short8 = parent_commit[:8]
    short12 = parent_commit[:12]

    # ayushnangia16 images have consistent torch 2.6+cu124 with matching sgl_kernel
    # shikhar481 images have torch 2.7+cu126 which causes sgl_kernel ABI mismatch
    candidates = [
        f"ayushnangia16/nvidia-sglang-docker:{parent_commit}",
        f"shikhar481/sglang-images:{short8}",
        f"shikhar481/sglang-images:{short8}-src",
    ]

    for image in candidates:
        try:
            result = subprocess.run(
                ['docker', 'manifest', 'inspect', image],
                capture_output=True, timeout=15,
            )
            if result.returncode == 0:
                log.info(f"  Found SGLang image: {image}")
                return image
        except Exception:
            pass

    # Default to ayushnangia16 (will fail at pull if not found)
    return f"{SGLANG_IMAGE_REPO}:{parent_commit}"


def setup_sglang_container(container_id: str) -> bool:
    """One-time setup for SGLang Docker container."""
    setup_script = """
apt-get update -qq 2>/dev/null && apt-get install -y -qq libnuma-dev 2>/dev/null
# CRITICAL: Do NOT upgrade transformers — it breaks sgl_kernel ABI compatibility
# Instead downgrade compressed_tensors to match existing transformers
pip install 'compressed_tensors<0.9' -q 2>/dev/null || true
pip install 'numpy<2.0' aiohttp requests tqdm sentencepiece -q 2>/dev/null

export PYTHONPATH="/sgl-workspace/sglang/python:$PYTHONPATH"
python3 -c "from sgl_kernel import common_ops; print('sgl_kernel OK')" 2>/dev/null || {
    echo "sgl_kernel BROKEN - cannot fix"
}

# Backup clean source for patch revert between samples
cp -r /sgl-workspace/sglang/python /sgl-workspace/sglang_python_clean
echo "SGLANG_SETUP_COMPLETE"
"""
    log.info("Running SGLang one-time setup...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(setup_script)
        tmp_script = f.name
    try:
        docker_cp_to(container_id, tmp_script, '/tmp/sglang_setup.sh')
    finally:
        os.unlink(tmp_script)

    rc, output = docker_exec(container_id, 'bash /tmp/sglang_setup.sh', timeout=300)
    if 'SGLANG_SETUP_COMPLETE' not in output:
        log.error(f"SGLang setup failed: {output[-1000:]}")
        return False
    log.info("SGLang setup complete")
    return True


def build_sglang_benchmark_script(perf_command: str, model: str, benchmark_mode: str) -> str:
    """Build benchmark script for one SGLang sample (runs inside container)."""

    script = '''
export PYTHONPATH="/sgl-workspace/sglang/python:$PYTHONPATH"

# Kill any leftover processes from previous sample
pkill -9 -f "sglang" 2>/dev/null || true
pkill -9 -f "python3.*launch_server" 2>/dev/null || true
sleep 3

echo "=== Restoring clean SGLang ==="
rm -rf /sgl-workspace/sglang/python
cp -r /sgl-workspace/sglang_python_clean /sgl-workspace/sglang/python

echo "=== Applying agent patch ==="
cd /sgl-workspace/sglang
if git apply /tmp/current_patch.diff 2>&1; then
    echo "AGENT_PATCH_APPLIED"
elif patch -p1 < /tmp/current_patch.diff 2>&1; then
    echo "AGENT_PATCH_APPLIED_P1"
else
    echo "AGENT_PATCH_FAILED"
fi

'''

    if benchmark_mode == 'offline' or 'bench_one_batch' in (perf_command or ''):
        script += '''
echo "=== Running offline benchmark ==="
PERF_CMD="''' + perf_command + '''"
PERF_CMD=$(echo "$PERF_CMD" | sed 's/^python /python3 /')
echo "Command: $PERF_CMD"
eval $PERF_CMD 2>&1 | tee /tmp/benchmark_output.txt
echo "BENCHMARK_DONE"
cat /tmp/benchmark_output.txt
'''
    else:
        script += '''
# Aggressively kill any leftover server processes and free GPU memory
pkill -9 -f "sglang" 2>/dev/null || true
pkill -9 -f "torch" 2>/dev/null || true
sleep 3

echo "=== Starting SGLang server ==="
MODEL="''' + model + '''"

pkill -9 -f "sglang.launch_server" 2>/dev/null || true
sleep 2

set +e
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --port 30001 --host 0.0.0.0 --log-level warning > /tmp/server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

MAX_WAIT=1200
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:30001/health 2>/dev/null | grep -q "200"; then
        echo "SERVER_READY after ${WAITED}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "SERVER_CRASHED"
        echo "=== Server log ==="
        cat /tmp/server.log | tail -30
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "SERVER_TIMEOUT"
    cat /tmp/server.log | tail -20
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

set -e
echo "=== Running serving benchmark ==="
PERF_CMD="''' + perf_command + '''"
echo "$PERF_CMD" | grep -q "\\-\\-port" || PERF_CMD="$PERF_CMD --port 30001"
echo "$PERF_CMD" | grep -q "\\-\\-host" || PERF_CMD="$PERF_CMD --host 127.0.0.1"
PERF_CMD=$(echo "$PERF_CMD" | sed 's/^python /python3 /')
echo "Command: $PERF_CMD"
eval $PERF_CMD 2>&1 | tee /tmp/benchmark_output.txt

echo "=== Stopping server ==="
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
pkill -9 -f "sglang.launch_server" 2>/dev/null || true

echo "BENCHMARK_DONE"
cat /tmp/benchmark_output.txt
'''

    return script


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Batch pass@k agent benchmark runner. '
                    'Benchmarks all 8 agent patches per task with one container setup.'
    )
    parser.add_argument('--hf-repo', default='Inferencebench/pass-at-k-samples',
                        help='HuggingFace repo with pass@k samples')
    parser.add_argument('--repo-prefix', default='vllm',
                        help='Filter item_ids by prefix (default: vllm)')
    parser.add_argument('--agents', nargs='+', default=['claude_code/sonnet', 'codex_cli/gpt-5'],
                        help='Agent/model pairs as agent/model (default: claude_code/sonnet codex_cli/gpt-5)')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT_DIR / 'results' / 'pass_at_k_benchmarks',
                        help='Output directory for results')
    parser.add_argument('--plan', type=Path, default=None,
                        help='Plan file (auto-detected from repo-prefix)')
    parser.add_argument('--benchmark-mapping', type=Path, default=None,
                        help='Benchmark mode mapping file (auto-detected from repo-prefix)')
    parser.add_argument('--items', nargs='+', default=None,
                        help='Specific item_ids to run (e.g., vllm_core-0000)')
    parser.add_argument('--timeout', type=int, default=900,
                        help='Timeout per sample in seconds (default: 900)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run, no containers started')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-completed samples')
    args = parser.parse_args()

    # Parse agent/model pairs
    agent_configs = []
    for spec in args.agents:
        if '/' not in spec:
            log.error(f"Invalid agent spec '{spec}', expected format: agent_name/model_name")
            sys.exit(1)
        agent_name, model_name = spec.split('/', 1)
        agent_configs.append((agent_name, model_name))
    log.info(f"Agent configs: {agent_configs}")

    # Auto-detect plan and mapping files from repo-prefix
    if args.plan is None:
        if args.repo_prefix == 'sglang':
            args.plan = ROOT_DIR / 'ISO-Bench' / 'state' / 'sglang_plan_iso.json'
        else:
            args.plan = ROOT_DIR / 'ISO-Bench' / 'state' / 'plan_iso.json'
    if args.benchmark_mapping is None:
        if args.repo_prefix == 'sglang':
            args.benchmark_mapping = ROOT_DIR / 'data' / 'mappings' / 'sglang_benchmark_mode_mapping.json'
        else:
            args.benchmark_mapping = ROOT_DIR / 'data' / 'mappings' / 'benchmark_mode_mapping.json'
    log.info(f"Plan: {args.plan}")
    log.info(f"Mapping: {args.benchmark_mapping}")

    # Load data (patches first, then configs — configs use HF data as fallback)
    task_patches = load_pass_at_k_patches(args.hf_repo, args.repo_prefix, agent_configs)
    task_configs = load_task_configs(args.plan, args.benchmark_mapping, task_patches)

    # Enrich with local patches from everything_analysis_data (fills empty HF patches)
    plan_items = {}
    if args.plan.exists():
        with open(args.plan) as f:
            for item in json.load(f)["items"]:
                plan_items[item["item_id"]] = item
    task_patches = enrich_with_local_patches(task_patches, agent_configs, plan_items)

    # Filter to requested items
    if args.items:
        task_patches = {k: v for k, v in task_patches.items() if k in args.items}

    # Only keep tasks that have both config and patches
    runnable = sorted(set(task_patches.keys()) & set(task_configs.keys()))
    missing_config = set(task_patches.keys()) - set(task_configs.keys())
    missing_patches = set(task_configs.keys()) - set(task_patches.keys())

    if missing_config:
        log.warning(f"Tasks with patches but no config (skipping): {sorted(missing_config)}")
    if missing_patches and args.items:
        log.warning(f"Requested items with no patches on HF: {sorted(missing_patches & set(args.items))}")

    log.info(f"Runnable tasks: {len(runnable)}")

    # Load resume state
    completed = set()
    if args.resume:
        completed = load_completed(args.output_dir)
        log.info(f"Resume: {len(completed)} samples already completed")

    hf_token = get_hf_token()
    if not hf_token:
        log.warning("No HuggingFace token found — gated models will fail")

    # Dry run
    if args.dry_run:
        total_samples = sum(len(task_patches[t]) for t in runnable)
        print(f"\n{'='*70}")
        print(f"DRY RUN — {len(runnable)} tasks, {total_samples} total samples across {len(agent_configs)} agents")
        print(f"Agents: {agent_configs}")
        print(f"{'='*70}\n")
        for item_id in runnable:
            cfg = task_configs[item_id]
            samples = task_patches[item_id]
            pending = [s for s in samples
                       if (item_id, s['agent_name'], s['model_name'], s['sample_index']) not in completed]
            parent = cfg.get('parent_commit') or ''
            image = f"{BASELINE_IMAGE_PREFIX}:baseline-{parent[:12]}" if parent else '(no parent)'
            # Count per agent
            from collections import Counter as _C
            agent_counts = _C((s['agent_name'], s['model_name']) for s in samples)
            pending_counts = _C((s['agent_name'], s['model_name']) for s in pending)
            print(f"  {item_id}:")
            print(f"    Image:     {image}")
            print(f"    LLM Model: {cfg['model']}")
            print(f"    Mode:      {cfg['benchmark_mode']}")
            perf_cmd_display = cfg['perf_command'] or '(none)'
            print(f"    Perf cmd:  {perf_cmd_display[:80]}{'...' if len(perf_cmd_display) > 80 else ''}")
            for (a, m), cnt in sorted(agent_counts.items()):
                pcnt = pending_counts.get((a, m), 0)
                print(f"    {a}/{m}: {cnt} total, {pcnt} pending")
            print()
        return

    # Main loop
    total_tasks = len(runnable)
    total_success = 0
    total_failed = 0
    total_skipped = 0

    for task_i, item_id in enumerate(runnable):
        cfg = task_configs[item_id]
        samples = task_patches[item_id]

        # Check which samples need to run (keyed by item_id + agent + model + sample_index)
        pending = [s for s in samples
                   if (item_id, s['agent_name'], s['model_name'], s['sample_index']) not in completed]
        if not pending:
            log.info(f"[{task_i+1}/{total_tasks}] {item_id}: all {len(samples)} samples already done, skipping")
            total_skipped += len(samples)
            continue

        n_agents = len(set((s['agent_name'], s['model_name']) for s in pending))
        log.info(f"\n{'='*70}")
        log.info(f"[{task_i+1}/{total_tasks}] {item_id}: {len(pending)} samples pending ({n_agents} agents)")
        log.info(f"  LLM Model: {cfg['model']}, Mode: {cfg['benchmark_mode']}")
        log.info(f"{'='*70}")

        # Check parent commit exists
        if not cfg.get('parent_commit'):
            log.error(f"  No parent_commit for {item_id}, skipping")
            for s in pending:
                save_result(args.output_dir, {
                    'item_id': item_id,
                    'sample_index': s['sample_index'],
                    'status': 'no_parent_commit',
                    'error': 'No parent_commit in benchmark mapping',
                    'human_commit': cfg.get('human_commit', ''),
                    'parent_commit': '',
                    'model': cfg.get('model', ''),
                    'agent_name': s['agent_name'],
                    'model_name': s['model_name'],
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                })
                total_failed += 1
            continue

        # Dispatch: determine Docker image based on repo type
        is_sglang = item_id.startswith('sglang')

        if is_sglang:
            # SGLang: try multiple image repos/tags
            baseline_image = get_sglang_image(cfg['parent_commit'])
            log.info(f"  SGLang image: {baseline_image}")
        else:
            # vLLM: "baseline-{parent[:12]}" tag format
            baseline_image = f"{BASELINE_IMAGE_PREFIX}:baseline-{cfg['parent_commit'][:12]}"
        if not check_docker_image_exists(baseline_image):
            log.error(f"  Docker image not found: {baseline_image}")
            for s in pending:
                save_result(args.output_dir, {
                    'item_id': item_id,
                    'sample_index': s['sample_index'],
                    'status': 'image_not_found',
                    'error': f'Docker image not found: {baseline_image}',
                    'human_commit': cfg['human_commit'],
                    'parent_commit': cfg['parent_commit'],
                    'model': cfg['model'],
                    'agent_name': s['agent_name'],
                    'model_name': s['model_name'],
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                })
                total_failed += 1
            continue

        # Start container — ONE container for ALL agents on this task
        container_id = None
        try:
            container_id = start_task_container(cfg['parent_commit'], hf_token, image_override=baseline_image)

            # One-time setup (different for vLLM vs SGLang)
            if is_sglang:
                setup_ok = setup_sglang_container(container_id)
            else:
                setup_ok = setup_container(
                    container_id, cfg['human_commit'],
                    cfg['perf_command'], cfg['model'], cfg['benchmark_mode'],
                )
            if not setup_ok:
                log.error(f"  Setup failed for {item_id}, skipping all samples")
                for s in pending:
                    save_result(args.output_dir, {
                        'item_id': item_id,
                        'sample_index': s['sample_index'],
                        'status': 'setup_failed',
                        'error': 'Container setup failed',
                        'human_commit': cfg['human_commit'],
                        'parent_commit': cfg['parent_commit'],
                        'model': cfg['model'],
                        'agent_name': s['agent_name'],
                        'model_name': s['model_name'],
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                    })
                    total_failed += 1
                continue

            # Run each pending sample (sorted by agent then sample_index)
            for sample_i, sample in enumerate(pending):
                idx = sample['sample_index']
                s_agent = sample['agent_name']
                s_model = sample['model_name']
                patch_text = sample.get('model_patch', '')
                tag = f"[{item_id}][{s_agent}/{s_model}][s{idx}]"

                log.info(f"  {tag} ({sample_i+1}/{len(pending)}) Running benchmark...")

                if not patch_text or not patch_text.strip():
                    log.info(f"  {tag} Empty patch, recording as empty_patch")
                    save_result(args.output_dir, {
                        'item_id': item_id,
                        'sample_index': idx,
                        'status': 'empty_patch',
                        'human_commit': cfg['human_commit'],
                        'parent_commit': cfg['parent_commit'],
                        'perf_command': cfg['perf_command'],
                        'model': cfg['model'],
                        'benchmark_mode': cfg['benchmark_mode'],
                        'agent_name': s_agent,
                        'model_name': s_model,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                    })
                    total_skipped += 1
                    continue

                result = run_agent_in_container(
                    container_id, patch_text,
                    cfg['perf_command'], cfg['model'],
                    cfg['benchmark_mode'], args.timeout,
                    is_sglang=is_sglang,
                )

                # Augment result with metadata
                result.update({
                    'item_id': item_id,
                    'sample_index': idx,
                    'human_commit': cfg['human_commit'],
                    'parent_commit': cfg['parent_commit'],
                    'perf_command': cfg['perf_command'],
                    'model': cfg['model'],
                    'benchmark_mode': cfg['benchmark_mode'],
                    'agent_name': s_agent,
                    'model_name': s_model,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                })

                save_result(args.output_dir, result)

                if result['status'] == 'success':
                    total_success += 1
                    metrics_summary = {k: v for k, v in result.get('metrics', {}).items()
                                      if 'throughput' in k or 'latency' in k}
                    log.info(f"  {tag} SUCCESS ({result['duration_s']:.0f}s): {metrics_summary}")
                else:
                    total_failed += 1
                    log.warning(f"  {tag} {result['status'].upper()}: {result.get('error', '?')}")

        except Exception as e:
            log.error(f"  Fatal error for {item_id}: {e}")
            total_failed += len(pending)
        finally:
            if container_id:
                teardown_container(container_id)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Tasks processed:   {total_tasks}")
    print(f"Samples succeeded: {total_success}")
    print(f"Samples failed:    {total_failed}")
    print(f"Samples skipped:   {total_skipped}")
    print(f"Results saved to:  {args.output_dir / 'agent_benchmark_results.jsonl'}")


if __name__ == '__main__':
    main()
