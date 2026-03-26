#!/usr/bin/env python3
"""
pass@k sample collection for ISO-Bench.

Runs the prepare pipeline N times per plan item (default 8).  After each
sample completes the artifacts are pushed to HuggingFace *immediately*.
Only once the push is confirmed do we nuke the local state + worktree so
the next sample's agent cannot peek at prior patches or logs.

Usage:
    # Dry run — shows what would happen, no agents, no push
    python scripts/run_pass_at_k.py \
        --task tasks/vllm.yaml \
        --plan state/plan.json \
        --bench-cfg bench.yaml \
        --n 8 --dry-run

    # Full run
    python scripts/run_pass_at_k.py \
        --task tasks/vllm.yaml \
        --plan state/plan.json \
        --bench-cfg bench.yaml \
        --n 8 \
        --hf-repo ISO-Bench/pass-at-k-samples

    # Resume after crash (queries HF to skip already-pushed samples)
    python scripts/run_pass_at_k.py \
        --task tasks/vllm.yaml \
        --plan state/plan.json \
        --bench-cfg bench.yaml \
        --n 8 \
        --hf-repo ISO-Bench/pass-at-k-samples \
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pass_at_k")

# ---------------------------------------------------------------------------
# Paths (relative to ISO-Bench/ root)
# ---------------------------------------------------------------------------
ISO_BENCH_ROOT = Path(__file__).resolve().parents[1]

# Files we collect from each run's state directory
ARTIFACT_FILES = [
    "journal.json",
    "prompt.json",
    "task.txt",
    "diff_targets.json",
    "model_patch.diff",
    "prediction.jsonl",
    "run_summary.json",
    # Agent logs (only the ones that exist)
    "openhands_stdout.txt",
    "openhands_stderr.txt",
    "trae_stdout.txt",
    "trae_stderr.txt",
    "codex_stdout.txt",
    "codex_stderr.txt",
    "claude_code_stdout.txt",
    "claude_code_stderr.txt",
    # Trajectory (TRAE/Codex)
    "trajectory.json",
    # Step logs
    "trae_steps.log",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text())


def extract_repo_name(repo_url: str) -> str:
    if "github.com" in repo_url:
        return repo_url.rstrip("/").rstrip(".git").split("/")[-1]
    return Path(repo_url).name or "unknown"


def sanitize(name: str) -> str:
    s = re.sub(r"[^\w\-]", "-", name.lower())
    s = re.sub(r"-+", "-", s)
    return s.strip("-") or "unknown"


def get_model_name(bench_cfg: Dict[str, Any], agent_name: str) -> str:
    env_model = os.environ.get("LLM_MODEL")
    if env_model:
        return sanitize(env_model)
    agent_cfg = bench_cfg.get("agents", {}).get(agent_name, {})
    args_cfg = agent_cfg.get("args", {})
    if "model" in args_cfg:
        raw = os.path.expandvars(str(args_cfg["model"]))
        return sanitize(raw)
    return "default"


def find_run_dir(state_root: Path, run_id: str, item_id: str) -> Optional[Path]:
    """Locate the item directory created by prepare."""
    candidate = state_root / "runs" / run_id / item_id
    if candidate.exists():
        return candidate
    return None


def collect_artifacts(item_dir: Path) -> Dict[str, Any]:
    """Read all artifacts from an item directory into a dict."""
    artifacts: Dict[str, Any] = {}
    for fname in ARTIFACT_FILES:
        fpath = item_dir / fname
        if not fpath.exists():
            continue
        if fname.endswith(".json"):
            try:
                artifacts[fname] = json.loads(fpath.read_text())
            except Exception:
                artifacts[fname] = fpath.read_text()
        elif fname.endswith(".jsonl"):
            lines = []
            for line in fpath.read_text().strip().splitlines():
                try:
                    lines.append(json.loads(line))
                except Exception:
                    lines.append(line)
            artifacts[fname] = lines
        else:
            artifacts[fname] = fpath.read_text()
    return artifacts


def nuke_run_dir(state_root: Path, run_id: str) -> None:
    """Remove the entire run directory tree under state/runs/."""
    run_dir = state_root / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
        log.debug(f"Nuked run dir: {run_dir}")
    # Walk up and remove empty parent dirs (timestamp/model/agent/repo)
    for parent in run_dir.parents:
        if parent == state_root / "runs":
            break
        try:
            parent.rmdir()  # only succeeds if empty
        except OSError:
            break


def nuke_worktree(work_root: Path, repo_name: str, item_id: str) -> None:
    """Remove the worktree directory and prune stale git worktree registrations."""
    wt_dir = work_root / "worktrees" / repo_name / item_id
    if wt_dir.exists():
        shutil.rmtree(wt_dir)
        log.debug(f"Nuked worktree: {wt_dir}")
    # Prune stale worktree registrations — detach_from_history removes the .git
    # link but leaves the registration in the base repo, causing "already registered"
    # errors on the next worktree add.
    base_repo = work_root / "repos" / repo_name
    if base_repo.exists():
        try:
            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=str(base_repo),
                capture_output=True,
                timeout=30,
            )
        except Exception:
            pass


# Directories that agents may write to outside the worktree/state dirs.
# Cleaning these between samples prevents information leakage.
AGENT_CACHE_DIRS = [
    Path.home() / ".openhands",
    Path.home() / ".trae",
    Path.home() / ".codex",
]

# Inside ~/.claude, these hold auth/config that must survive between samples.
# Everything else (sessions, cache, history, plans) is cleaned.
_CLAUDE_PRESERVE = {".credentials.json", "settings.json", "settings.local.json"}


def _clean_agent_caches(tag: str) -> None:
    """Remove agent-specific cache directories that could leak between samples.

    For ~/.claude we selectively clean session/cache data while preserving
    auth credentials and settings — nuking the entire directory kills the
    login token and causes all subsequent samples to fail.

    NOTE: This is best-effort. True isolation requires running each sample
    in a fresh container. Agents may also write to /tmp or other locations.
    """
    # Clean non-claude agent dirs entirely
    for cache_dir in AGENT_CACHE_DIRS:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                log.debug(f"{tag} Cleaned cache: {cache_dir}")
            except Exception as e:
                log.warning(f"{tag} Could not clean {cache_dir}: {e}")

    # Selectively clean ~/.claude — preserve auth + settings
    claude_dir = Path.home() / ".claude"
    if claude_dir.exists():
        for child in claude_dir.iterdir():
            if child.name in _CLAUDE_PRESERVE:
                continue
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                log.debug(f"{tag} Cleaned claude cache: {child.name}")
            except Exception as e:
                log.warning(f"{tag} Could not clean {child}: {e}")


# ---------------------------------------------------------------------------
# HuggingFace: build a single row from collected artifacts
# ---------------------------------------------------------------------------

def build_hf_row(
    item_id: str,
    sample_idx: int,
    run_id: str,
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one HF dataset row from a sample's artifacts."""
    journal = artifacts.get("journal.json", {})
    if isinstance(journal, str):
        try:
            journal = json.loads(journal)
        except Exception:
            journal = {}

    row = {
        # Identifiers
        "item_id": item_id,
        "sample_index": sample_idx,
        "run_id": run_id,
        "collected_at": datetime.now().isoformat(),

        # From journal (structured fields for easy filtering)
        "task_id": journal.get("task_id", ""),
        "status": journal.get("status", "unknown"),
        "human_commit": journal.get("commits", {}).get("human", ""),
        "pre_commit": journal.get("commits", {}).get("pre", ""),
        "agent_name": journal.get("run_metadata", {}).get("agent", ""),
        "model_name": journal.get("run_metadata", {}).get("model", ""),
        "duration_s": None,
        "time_to_first_edit_s": journal.get("metrics", {}).get("time_to_first_edit_s"),
        "commit_count": journal.get("metrics", {}).get("commit_count"),
        "patch_size_loc": journal.get("metrics", {}).get("patch_size_loc"),
        "changed_files_count": journal.get("metrics", {}).get("changed_files_count"),
        "violations_count": journal.get("metrics", {}).get("violations_count"),

        # The patch
        "model_patch": "",

        # Full artifacts as text (for ablation / post-hoc analysis)
        "journal_json": "",
        "prompt_json": "",
        "task_text": "",
        "diff_targets_json": "",
        "run_summary_json": "",
        "agent_stdout": "",
        "agent_stderr": "",
        "trajectory_json": "",
    }

    # Extract agent-specific duration
    for agent_key in ("trae", "codex", "openhands", "claude_code", "codex_cli"):
        agent_block = journal.get(agent_key, {})
        if isinstance(agent_block, dict) and "duration_s" in agent_block:
            row["duration_s"] = agent_block["duration_s"]
            break

    # Serialize artifacts into text columns
    def _to_text(key: str) -> str:
        val = artifacts.get(key)
        if val is None:
            return ""
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        return str(val)

    row["model_patch"] = _to_text("model_patch.diff")
    row["journal_json"] = _to_text("journal.json")
    row["prompt_json"] = _to_text("prompt.json")
    row["task_text"] = _to_text("task.txt")
    row["diff_targets_json"] = _to_text("diff_targets.json")
    row["run_summary_json"] = _to_text("run_summary.json")
    row["trajectory_json"] = _to_text("trajectory.json")

    # Agent logs — pick whichever agent's logs exist
    for agent_key in ("claude_code", "trae", "codex", "openhands"):
        stdout = _to_text(f"{agent_key}_stdout.txt")
        stderr = _to_text(f"{agent_key}_stderr.txt")
        if stdout or stderr:
            row["agent_stdout"] = stdout
            row["agent_stderr"] = stderr
            break

    return row


# ---------------------------------------------------------------------------
# HuggingFace: push one row (append to existing dataset)
# ---------------------------------------------------------------------------

def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    token = cli_token or os.environ.get("HF_TOKEN")
    if token:
        return token
    # Try huggingface_hub API (works across versions)
    try:
        from huggingface_hub import HfApi
        t = HfApi().token
        if t:
            return t
    except Exception:
        pass
    # Fallback: read cached token file directly
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    try:
        if token_path.exists():
            return token_path.read_text().strip() or None
    except Exception:
        pass
    return None


def push_single_row(
    row: Dict[str, Any],
    repo_id: str,
    token: str,
) -> bool:
    """
    Push one sample row to HuggingFace as an incremental parquet shard.

    Uses huggingface_hub to upload a single-row parquet file directly,
    avoiding the O(n^2) download-all-append-reupload pattern.

    Returns True on success.
    """
    from huggingface_hub import HfApi

    item_id = row["item_id"]
    sample_idx = row["sample_index"]
    agent = row.get("agent_name", "unknown")
    model = row.get("model_name", "unknown")

    # Write row as a single-row parquet shard to a temp file
    shard_name = f"{item_id}_s{sample_idx}_{int(time.time())}.parquet"
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use pandas for parquet serialization (lighter than datasets)
        try:
            import pandas as pd
            df = pd.DataFrame([row])
            shard_path = Path(tmpdir) / shard_name
            df.to_parquet(shard_path, index=False)
        except ImportError:
            # Fallback to datasets if pandas not available
            from datasets import Dataset
            ds = Dataset.from_list([row])
            shard_path = Path(tmpdir) / shard_name
            ds.to_parquet(shard_path)

        api = HfApi(token=token)
        # Ensure the repo exists (creates if not)
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        # Upload shard into data/ directory — HF auto-discovers parquet shards
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=f"data/{shard_name}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"sample {item_id}/s{sample_idx} ({agent}/{model})",
        )

    return True


def fetch_completed_samples(
    repo_id: str,
    token: Optional[str],
) -> Set[Tuple[str, int]]:
    """
    Query HF dataset and return set of (item_id, sample_index) already pushed.

    Supports three formats:
    1. Per-sample shard filenames: {item_id}_s{idx}_{ts}.parquet  (fast path)
    2. Standard HF auto-sharded parquet: train-NNNNN.parquet      (pandas path)
    3. HF datasets library fallback                                (slow path)
    """
    completed: Set[Tuple[str, int]] = set()
    if not token:
        return completed

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # List all parquet files in data/
    try:
        files = list(api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo="data"))
    except Exception as e:
        log.warning(f"Could not list HF repo files: {e}")
        return completed

    # Fast path: try per-sample shard naming convention
    for f in files:
        name = f.rfilename
        match = re.match(r"^(.+)_s(\d+)_\d+\.parquet$", name)
        if match:
            completed.add((match.group(1), int(match.group(2))))
    if completed:
        return completed

    # Standard HF parquet path: download shards and read item_id/sample_index columns
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        parquet_files = [f.rfilename for f in files if f.rfilename.endswith(".parquet")]
        for pf in parquet_files:
            try:
                local = hf_hub_download(repo_id, pf, repo_type="dataset", token=token)
                df = pd.read_parquet(local, columns=["item_id", "sample_index"])
                for _, row in df.iterrows():
                    iid = row.get("item_id", "")
                    sidx = row.get("sample_index")
                    if iid and sidx is not None:
                        completed.add((str(iid), int(sidx)))
            except Exception:
                continue
        if completed:
            return completed
    except ImportError:
        log.debug("pandas not available, falling back to datasets library")

    # Last resort: HF datasets library
    try:
        from datasets import load_dataset
        ds = load_dataset(repo_id, token=token, split="train")
        for entry in ds:
            r = dict(entry)  # type: ignore[arg-type]
            iid = r.get("item_id", "")
            sidx = r.get("sample_index")
            if iid and sidx is not None:
                completed.add((str(iid), int(sidx)))
    except Exception as e:
        log.warning(f"Could not load existing HF dataset for resume: {e}")
    return completed


# ---------------------------------------------------------------------------
# Run one sample
# ---------------------------------------------------------------------------

def make_single_item_plan(full_plan: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "repo": full_plan["repo"],
        "task_id": full_plan["task_id"],
        "items": [item],
    }


def run_one_sample(
    task_yaml: Path,
    bench_cfg_path: Path,
    bench_cfg: Dict[str, Any],
    plan_item: Dict[str, Any],
    full_plan: Dict[str, Any],
    sample_idx: int,
    hf_repo: Optional[str],
    hf_token: Optional[str],
    dry_run: bool = False,
) -> bool:
    """
    Run prepare for one item, push artifacts to HF, nuke local state.

    Returns True if the sample was successfully pushed (or dry_run).
    """
    item_id = plan_item["item_id"]
    state_root = Path(bench_cfg["paths"]["state_root"]).resolve()
    work_root = Path(bench_cfg["paths"]["work_root"]).resolve()

    agent_name = str(bench_cfg.get("agents", {}).get("default", "unknown"))
    model_name = get_model_name(bench_cfg, agent_name)
    repo_url = full_plan["repo"]
    repo_name = full_plan["task_id"]  # prepare.py uses task_id as repo_name

    # Unique run_id per sample
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    repo_short = extract_repo_name(repo_url)
    run_id = f"{repo_short}/{agent_name}/{model_name}/{timestamp}_s{sample_idx}"

    tag = f"[{item_id}][s{sample_idx}]"
    log.info(f"{tag} run_id={run_id}")

    if dry_run:
        log.info(f"{tag} DRY RUN — skipping")
        return True

    # --- Pre-clean: nuke any leftover worktree from a crashed previous run ---
    nuke_worktree(work_root, repo_name, item_id)

    # --- Write single-item plan to a unique temp file ---
    # Use tempfile to avoid race conditions with concurrent runs
    tmp_fd, tmp_plan_str = tempfile.mkstemp(
        prefix=f".pass_at_k_{item_id}_s{sample_idx}_",
        suffix=".json",
        dir=str(ISO_BENCH_ROOT),
    )
    tmp_plan_path = Path(tmp_plan_str)
    tmp_plan = make_single_item_plan(full_plan, plan_item)
    with os.fdopen(tmp_fd, "w") as f:
        json.dump(tmp_plan, f, indent=2)

    # --- Run prepare ---
    cmd = [
        sys.executable, "-m", "bench.cli", "prepare",
        str(task_yaml),
        "--from-plan", str(tmp_plan_path),
        "--bench-cfg", str(bench_cfg_path),
        "--max-workers", "1",
        "--no-resume",
        "--run-id", run_id,
    ]

    log.info(f"{tag} Running prepare ...")
    t0 = time.time()

    try:
        # Use Popen + process group so we can kill the entire tree on timeout
        proc = subprocess.Popen(
            cmd,
            cwd=str(ISO_BENCH_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # creates new process group
        )
        try:
            stdout, stderr = proc.communicate(timeout=7200)  # 2h hard cap
        except subprocess.TimeoutExpired:
            # Kill entire process group (agent + child processes / containers)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
                    proc.wait()
            elapsed = time.time() - t0
            log.error(f"{tag} prepare TIMED OUT after {elapsed:.0f}s — process tree killed")
            stdout, stderr = "", ""
        else:
            elapsed = time.time() - t0
            log.info(f"{tag} prepare done in {elapsed:.0f}s (rc={proc.returncode})")
            if proc.returncode != 0:
                log.warning(f"{tag} prepare stderr (last 500 chars):\n{stderr[-500:]}")
    except Exception as e:
        elapsed = time.time() - t0
        log.error(f"{tag} prepare failed with exception: {e}")

    # --- Collect artifacts ---
    item_dir = find_run_dir(state_root, run_id, item_id)
    if item_dir is None:
        log.error(f"{tag} No state dir found — nothing to push")
        nuke_worktree(work_root, repo_name, item_id)
        tmp_plan_path.unlink(missing_ok=True)
        return False

    artifacts = collect_artifacts(item_dir)
    if not artifacts:
        log.warning(f"{tag} Empty artifacts — pushing skeleton row anyway")

    row = build_hf_row(item_id, sample_idx, run_id, artifacts)

    # --- Push to HF ---
    pushed = False
    if hf_repo and hf_token:
        log.info(f"{tag} Pushing to {hf_repo} ...")
        try:
            pushed = push_single_row(row, hf_repo, hf_token)
            log.info(f"{tag} Push confirmed")
        except Exception as e:
            log.error(f"{tag} Push FAILED: {e}")
    elif hf_repo and not hf_token:
        log.error(f"{tag} No HF token — cannot push. Local state NOT nuked for safety.")
        tmp_plan_path.unlink(missing_ok=True)
        return False
    else:
        # No HF repo — local-only mode (for testing)
        log.warning(f"{tag} No --hf-repo, artifacts stay in state dir (NOT nuked)")
        tmp_plan_path.unlink(missing_ok=True)
        return True

    # --- Verify push by checking the shard exists on HF ---
    if pushed:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            files = [f.rfilename for f in api.list_repo_tree(hf_repo, repo_type="dataset", path_in_repo="data")]
            shard_prefix = f"{item_id}_s{sample_idx}_"
            if not any(f.startswith(shard_prefix) for f in files):
                log.error(f"{tag} Push verification FAILED — shard not found on HF. Keeping local state.")
                pushed = False
        except Exception as e:
            log.warning(f"{tag} Push verification could not complete: {e} — trusting push result")

    # --- Nuke ONLY after verified push ---
    if pushed:
        nuke_run_dir(state_root, run_id)
        nuke_worktree(work_root, repo_name, item_id)
        # Clean agent caches that might leak info between samples
        _clean_agent_caches(tag)
        log.info(f"{tag} Local state nuked")
    else:
        log.warning(f"{tag} Push not confirmed — keeping local state for manual recovery")

    tmp_plan_path.unlink(missing_ok=True)
    return pushed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect N independent samples per plan item for pass@k evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, help="Path to task YAML")
    parser.add_argument("--plan", required=True, help="Path to plan JSON")
    parser.add_argument("--bench-cfg", default="bench.yaml", help="Path to bench config")
    parser.add_argument("--n", type=int, default=8, help="Samples per item (default: 8)")
    parser.add_argument("--hf-repo", default=None, help="HuggingFace repo to push each sample to")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token")
    parser.add_argument("--dry-run", action="store_true", help="No agents, no push")
    parser.add_argument("--resume", action="store_true", help="Skip (item, sample) pairs already on HF")
    parser.add_argument("--items", nargs="*", default=None, help="Only these item IDs")
    parser.add_argument("--max-parallel-items", type=int, default=1,
                        help="Max items to run in parallel (samples of same item stay sequential)")

    args = parser.parse_args()

    task_yaml = Path(args.task)
    plan_path = Path(args.plan)
    bench_cfg_path = Path(args.bench_cfg)

    # Resolve relative to ISO-Bench root
    if not task_yaml.is_absolute():
        task_yaml = ISO_BENCH_ROOT / task_yaml
    if not plan_path.is_absolute():
        plan_path = ISO_BENCH_ROOT / plan_path
    if not bench_cfg_path.is_absolute():
        bench_cfg_path = ISO_BENCH_ROOT / bench_cfg_path

    for p, label in [(task_yaml, "task"), (plan_path, "plan"), (bench_cfg_path, "bench-cfg")]:
        if not p.exists():
            log.error(f"{label} not found: {p}")
            sys.exit(1)

    bench_cfg = load_yaml(bench_cfg_path)
    full_plan = json.loads(plan_path.read_text())
    items = full_plan["items"]

    if args.items:
        items = [it for it in items if it["item_id"] in args.items]
        log.info(f"Filtered to {len(items)} items: {[it['item_id'] for it in items]}")

    hf_token = resolve_hf_token(args.hf_token)
    if args.hf_repo and not hf_token:
        log.error("--hf-repo given but no HF token found. Set HF_TOKEN or pass --hf-token.")
        sys.exit(1)

    agent_name = str(bench_cfg.get("agents", {}).get("default", "unknown"))
    model_name = get_model_name(bench_cfg, agent_name)

    log.info(f"pass@k collection: n={args.n}, items={len(items)}, agent={agent_name}, model={model_name}")
    if args.hf_repo:
        log.info(f"Push target: {args.hf_repo}")

    # --- Resume: fetch already-pushed samples from HF ---
    already_done: Set[Tuple[str, int]] = set()
    if args.resume and args.hf_repo:
        log.info("Fetching completed samples from HF for resume ...")
        already_done = fetch_completed_samples(args.hf_repo, hf_token)
        log.info(f"Found {len(already_done)} samples already on HF")

    # --- Build work units: (item, sample_idx) pairs, filtering already-done ---
    total = len(items) * args.n
    skipped = 0
    work_by_item: Dict[str, List[Tuple[Dict[str, Any], int]]] = {}

    for item in items:
        item_id = item["item_id"]
        item_work = []
        for sample_idx in range(args.n):
            if args.resume and (item_id, sample_idx) in already_done:
                tag = f"[{item_id}][s{sample_idx}]"
                log.info(f"{tag} Already on HF — skipping")
                skipped += 1
            else:
                item_work.append((item, sample_idx))
        if item_work:
            work_by_item[item_id] = item_work

    to_run = sum(len(v) for v in work_by_item.values())
    log.info(f"Work plan: {to_run} samples to run, {skipped} already done, {total} total")

    # --- Run samples: sequential within each item, parallel across items ---
    completed = 0
    failed = 0

    def _run_item_samples(item_work: List[Tuple[Dict[str, Any], int]]) -> Tuple[int, int]:
        """Run all samples for one item sequentially. Returns (completed, failed)."""
        item_completed, item_failed = 0, 0
        for plan_item, sample_idx in item_work:
            ok = run_one_sample(
                task_yaml=task_yaml,
                bench_cfg_path=bench_cfg_path,
                bench_cfg=bench_cfg,
                plan_item=plan_item,
                full_plan=full_plan,
                sample_idx=sample_idx,
                hf_repo=args.hf_repo,
                hf_token=hf_token,
                dry_run=args.dry_run,
            )
            item_completed += 1
            if not ok and not args.dry_run:
                item_failed += 1
        return item_completed, item_failed

    if args.max_parallel_items <= 1:
        # Sequential: simple loop
        for item_id, item_work in work_by_item.items():
            for plan_item, sample_idx in item_work:
                tag = f"[{item_id}][s{sample_idx}]"
                completed += 1
                log.info(f"{tag} ({completed}/{to_run})")
                ok = run_one_sample(
                    task_yaml=task_yaml,
                    bench_cfg_path=bench_cfg_path,
                    bench_cfg=bench_cfg,
                    plan_item=plan_item,
                    full_plan=full_plan,
                    sample_idx=sample_idx,
                    hf_repo=args.hf_repo,
                    hf_token=hf_token,
                    dry_run=args.dry_run,
                )
                if not ok and not args.dry_run:
                    failed += 1
                log.info(f"{tag} progress: {completed}/{to_run} done, {skipped} skipped, {failed} failed")
    else:
        # Parallel across items, sequential within each item
        with ThreadPoolExecutor(max_workers=args.max_parallel_items) as pool:
            futures = {
                pool.submit(_run_item_samples, item_work): item_id
                for item_id, item_work in work_by_item.items()
            }
            for future in as_completed(futures):
                item_id = futures[future]
                try:
                    c, f = future.result()
                    completed += c
                    failed += f
                    log.info(f"[{item_id}] done: {c} samples, {f} failed")
                except Exception as e:
                    log.error(f"[{item_id}] item runner crashed: {e}")

    log.info(f"Finished: {completed} run, {skipped} skipped, {failed} failed, {total} total")


if __name__ == "__main__":
    main()
