"""
Create a canonical OmniPerf-Bench dataset record for a single commit and (optionally)
export SWE-Perf/GSO compatible views, with optional push to Hugging Face.

Pipeline (commit-centric):
1) Collect commit metadata (reuses collect.analysis.commits.PerfCommitAnalyzer)
2) Build code-change fields (unified diff; split into patch vs test_patch; function names)
3) Tests and timings (read/generate tests; run on base/head; parse times)
4) Environment/version (compose version string; record setup/install commands if provided)
5) Assemble and export (canonical JSONL/Parquet; optional SWE-Perf/GSO views; push to HF)

Usage example (YAML config):
  # commit_to_dataset.yaml
  repo_path: /home/you/coding-mess/vllm
  head_commit: 0f40557af6141ced118b81f2a04e651a0c6c9dbd
  base_commit: null  # optional; defaults to head^ if omitted
  extractions_dir: misc/experiments/commit_extractions_with_apis
  use_docker: false
  docker_image: ayushnangia16/nvidia-vllm-docker:latest
  hf_repo: yourname/omni-commit-dataset  # optional
  push_to_hf: false

Run:
  PYTHONPATH=src python src/collect/commit_to_dataset.py commit_to_dataset.yaml

Requires:
- docs/dataset_schema.md for the canonical schema (this script does a minimal structural validation).

############################ TODO #########################################

- Running locally right now. Need to test on docker yet.
- Paths are changed everywhere because they are hardcoded everywhere (YAML & generate_test_generators.py).
- Need to change the test generation prompt. Running it on `device=CPU` need GPU, 
    there are `DummyLayers`, unavailable APIs & attributes like `custom_ops, input_scale, cutlass_fp8_supported`, 
    simplified functionality tests instead of complex internal mocking etc.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

# Ensure local src/ is importable (for collect.* and test_scripts.*)
_ROOT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _ROOT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Prefer local imports from repo
try:
    # Reuse commit analysis utilities
    from collect.analysis.commits import PerfCommitAnalyzer
except Exception:
    PerfCommitAnalyzer = None  # type: ignore

try:
    # Timing helpers (we reuse parse_times signature/regex)
    from collect.execute.evaluate import parse_times
except Exception:
    parse_times = None  # type: ignore

# Optional: import the LLM test generator utilities
try:
    from test_scripts.generate_test_generators import process_extraction_file, LLMClient  # type: ignore
except Exception:
    process_extraction_file = None  # type: ignore
    LLMClient = None  # type: ignore

# YAML loader
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


# -------------------------- Utilities --------------------------


def run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> str:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{result.stderr}")
    return result.stdout
def _extract_hf_repo_id(hf_repo: Optional[str], default_repo_name: str = "omni_commit_dataset") -> Optional[str]:
    """Normalize various HF repo formats to a repo id suitable for push_to_hub.

    Examples:
    - "https://huggingface.co/user/repo" -> "user/repo"
    - "https://huggingface.co/repo" -> "repo"
    - "user/repo" -> "user/repo"
    - "repo" -> "repo"
    """
    if not hf_repo:
        return None
    s = hf_repo.strip().rstrip("/")
    if "huggingface.co" in s:
        # Strip domain
        after = s.split("huggingface.co", 1)[1]
        after = after.lstrip("/")
        # Remove common prefixes like 'datasets/'
        if after.startswith("datasets/"):
            after = after[len("datasets/"):]
        # If path has more than two segments, keep last two
        segs = [p for p in after.split("/") if p]
        if len(segs) >= 2:
            return "/".join(segs[-2:])
        if len(segs) == 1:
            return f"{segs[0]}/{default_repo_name}"
        return None
    # Non-URL input
    if "/" in s:
        return s
    # Single segment -> treat as org/user and append default repo name
    return f"{s}/{default_repo_name}"



def clone_or_update_repo(repo_url: str, dest_dir: Path) -> Path:
    if dest_dir.exists():
        return dest_dir
    run(["git", "clone", repo_url, str(dest_dir)])
    return dest_dir


def checkout_commit(repo_path: Path, commit_hash: str) -> None:
    run(["git", "fetch", "--all", "--tags"], cwd=repo_path)
    run(["git", "checkout", "--force", commit_hash], cwd=repo_path)


def git_unified_diff(repo_path: Path, base_commit: str, head_commit: str) -> str:
    return run(["git", "diff", "-p", base_commit, head_commit], cwd=repo_path)


def get_main_branch_head(repo_path: Path) -> str:
    # Try common main branch names
    for ref in ("origin/main", "origin/master", "main", "master"):
        try:
            return run(["git", "rev-parse", ref], cwd=repo_path).strip()
        except Exception:
            continue
    raise RuntimeError("Unable to determine main branch HEAD (tried origin/main, origin/master, main, master)")


def is_test_path(path: str) -> bool:
    p = path.lower()
    fname = os.path.basename(p)
    return (
        "/tests/" in p
        or p.startswith("tests/")
        or fname.startswith("test_")
        or fname.endswith("_test.py")
    )

def write_tests_to_tmp(tests: List[str], tmp_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for idx, code in enumerate(tests):
        p = tmp_dir / f"perf_test_{idx}.py"
        p.write_text(code)
        paths.append(p)
    return paths


def run_tests_locally(repo_path: Path, commit_hash: str, test_entry: Path) -> List[float]:
    """Checkout commit and run pytest on the provided test entry; return wall-clock seconds as a single-item list."""
    import time as _time
    checkout_commit(repo_path, commit_hash)
    start = _time.time()
    try:
        # Run from repo root so imports resolve, passing test path relative to repo root
        rel = str(test_entry.relative_to(repo_path))
        run(["pytest", "-q", rel], cwd=repo_path)
    except Exception:
        # Even on failure, measure duration to capture behavior
        pass
    end = _time.time()
    return [end - start]


def run_tests_in_docker(repo_path: Path, commit_hash: str, test_entry: Path, docker_image: str) -> List[float]:
    import time as _time
    checkout_commit(repo_path, commit_hash)
    cmd = [
        "docker", "run", "--rm", "-t",
        "--gpus", "all",
        "-v", f"{repo_path}:/workspace",
        "-w", "/workspace",
        docker_image,
        "bash", "-lc", f"pytest -q {test_entry.relative_to(repo_path)} | cat",
    ]
    start = _time.time()
    try:
        run(cmd)
    except Exception:
        pass
    end = _time.time()
    return [end - start]


def _parse_times(stdout: str) -> List[float]:
    if parse_times is not None:
        try:
            return parse_times(stdout)
        except Exception:
            pass
    # Fallback regex
    pattern = re.compile(r"Execution time:\s*([0-9]+\.?[0-9]*)s")
    return [float(m.group(1)) for m in pattern.finditer(stdout or "")]


def find_or_generate_test_script(commit_hash: str, extractions_dir: Path, out_dir: Path) -> Optional[Path]:
    """Locate an existing generated test-case-generator for the commit, or generate one via LLM.

    Returns path to the generated test module file, or None if unavailable.
    """
    # Always generate on-the-fly per workflow (do not use pre-generated files)
    hash8 = commit_hash[:8]
    # Validate inputs
    if not extractions_dir.exists() or not extractions_dir.is_dir():
        raise RuntimeError(f"extractions_dir not found or not a directory: {extractions_dir}")
    if process_extraction_file is None or LLMClient is None:
        raise RuntimeError(
            "LLM generator utilities not importable. Ensure 'src' is on sys.path and dependencies are installed."
        )
    # Find matching extraction JSON by full or prefix hash
    json_path = None
    for p in extractions_dir.glob("*.json"):
        name = p.stem
        if name.startswith(commit_hash) or commit_hash.startswith(name) or name.startswith(hash8):
            json_path = p
            break
    if json_path is None:
        raise RuntimeError(
            f"No commit extraction JSON found for commit {commit_hash} in {extractions_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Require API credentials
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        raise RuntimeError(
            "Missing LLM credentials. Set OPENAI_API_KEY or ANTHROPIC_API_KEY to generate tests."
        )
    client = LLMClient()
    result = process_extraction_file(str(json_path), str(out_dir), client)  # type: ignore
    if result and "script_path" in result:
        path = Path(result["script_path"])  # type: ignore
        if path.exists():
            return path
    raise RuntimeError("LLM generation did not return a valid script_path.")


def mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


# -------------------------- CLI Orchestration --------------------------


@dataclass
class CanonicalRecord:
    repo: str
    instance_id: str
    created_at: str
    base_commit: str
    head_commit: str
    patch: str
    test_patch: str
    efficiency_test: List[str]
    duration_changes: List[Dict[str, List[float]]]
    human_performance: float
    version: str
    patch_functions: Optional[List[str]] = None
    test_functions: List[str] = None # specific test for the task at hand. Taken from GSO.
    api: Optional[str] = None
    gt_commit_message: str = None
    setup_commands: List[str] = None
    install_commands: List[str] = None
    notes: Optional[str] = None


def build_instance_id(repo_owner: str, repo_name: str, repo_path: Path, head_commit: str) -> str:
    # Try to infer PR number
    pr_num = None
    try:
        log = run(["git", "log", "--merges", "--ancestry-path", "--pretty=%H %s", "--reverse", f"{head_commit}^..HEAD"], cwd=repo_path)
        for line in log.splitlines()[:10]:
            if "Merge pull request" in line:
                m = re.search(r"#(\d+)", line)
                if m:
                    pr_num = m.group(1)
                    break
    except Exception:
        pass
    if pr_num:
        return f"{repo_owner}__{repo_name}-PR-{pr_num}"
    return f"{repo_owner}__{repo_name}-{head_commit[:7]}"


def assemble_canonical(
    repo_path_arg: str,
    head_commit: str,
    base_commit: Optional[str],
    use_docker: bool,
    docker_image: str,
    extractions_dir: Optional[str] = None,
    setup_commands: List[str] = None,
    install_commands: List[str] = None,
    api: Optional[str] = None,
    notes: Optional[str] = None,
) -> CanonicalRecord:
    # Prepare workspace
    # Always use local repo path; clone to temp workspace to avoid mutating the user's checkout
    # Derive a provisional repo_name from the source; refine after cloning via origin URL if available
    provisional_name = os.path.basename(repo_path_arg.rstrip("/")) or "repo"
    if provisional_name.endswith(".git"):
        provisional_name = provisional_name[:-4]
    work_root = Path(tempfile.mkdtemp(prefix="omni_commit_"))
    repo_path = work_root / provisional_name
    clone_or_update_repo(repo_path_arg, repo_path)

    # Derive repo owner/name from git origin if possible; otherwise fall back
    try:
        origin_url = run(["git", "config", "--get", "remote.origin.url"], cwd=repo_path).strip()
        # Handle common URL formats (HTTPS/SSH/local)
        cleaned = origin_url
        if cleaned.endswith(".git"):
            cleaned = cleaned[:-4]
        # Convert SSH style git@github.com:owner/name to owner/name
        if ":" in cleaned and "@" in cleaned and "/" not in cleaned.split(":", 1)[0]:
            # unlikely branch; keep generic split below
            pass
        parts = cleaned.replace(":", "/").split("/")
        repo_owner, repo_name = parts[-2], parts[-1]
    except Exception:
        repo_owner, repo_name = "local", provisional_name

    # Collect commit metadata (must use PerfCommitAnalyzer)
    if PerfCommitAnalyzer is None:
        raise RuntimeError("PerfCommitAnalyzer is required but not available")
    if base_commit is None:
        base_commit = f"{head_commit}^"
    created_at_iso = datetime.now(timezone.utc).isoformat()
    perf_commit = PerfCommitAnalyzer.process_commit(head_commit, repo_path, max_year=None)  # type: ignore
    if perf_commit is None:
        raise RuntimeError("PerfCommitAnalyzer returned None for the specified commit")
    gt_commit_message = perf_commit.message
    if getattr(perf_commit, "date", None) is not None:
        try:
            created_at_iso = perf_commit.date.astimezone(timezone.utc).isoformat()
        except Exception:
            pass
    unified = perf_commit.diff_text or git_unified_diff(repo_path, base_commit, head_commit)

    patch = unified or ""
    test_patch = ""

    # Obtain a test script for this commit via existing/generated test-case generator
    extr_dir = Path(extractions_dir) if extractions_dir else Path("misc/experiments/commit_extractions_with_apis")
    gen_out_dir = Path("misc/experiments/generated_test_generators_v4")
    test_script = find_or_generate_test_script(head_commit, extr_dir, gen_out_dir)
    if test_script is None:
        raise RuntimeError("Unable to locate or generate a test script for this commit.")

    # Materialize test into repo workspace and run via pytest across base/head/main
    tests_root = repo_path / "_generated_perf_tests"
    tests_root.mkdir(parents=True, exist_ok=True)
    target_test = tests_root / "test_generated.py"
    target_test.write_text(Path(test_script).read_text())

    if use_docker:
        base_times_arr = run_tests_in_docker(repo_path, base_commit, target_test, docker_image)
        head_times_arr = run_tests_in_docker(repo_path, head_commit, target_test, docker_image)
        main_head = get_main_branch_head(repo_path)
        main_times_arr = run_tests_in_docker(repo_path, main_head, target_test, docker_image)
    else:
        base_times_arr = run_tests_locally(repo_path, base_commit, target_test)
        head_times_arr = run_tests_locally(repo_path, head_commit, target_test)
        main_head = get_main_branch_head(repo_path)
        main_times_arr = run_tests_locally(repo_path, main_head, target_test)

    duration_changes: List[Dict[str, List[float]]] = []
    duration_changes.append({"base": base_times_arr, "head": head_times_arr, "main": main_times_arr})
    perfs: List[Tuple[float, float]] = []
    if head_times_arr and head_times_arr[0] > 0:
        perfs.append((mean(base_times_arr), mean(head_times_arr)))
    human_perf = float("nan")
    if perfs:
        bmean = mean([x for x, _ in perfs])
        hmean = mean([y for _, y in perfs])
        human_perf = bmean / hmean if hmean > 0 else float("inf")

    # Compose version
    image_tag = docker_image if use_docker else "local"
    version = f"python=={os.getenv('PYTHON_VERSION','unknown')};arch={os.uname().machine};image={image_tag};install_sha=na"

    instance_id = build_instance_id(repo_owner, repo_name, repo_path, head_commit)

    record = CanonicalRecord(
        repo=f"{repo_owner}/{repo_name}",
        instance_id=instance_id,
        created_at=created_at_iso,
        base_commit=base_commit,
        head_commit=head_commit,
        patch=patch,
        test_patch=test_patch,
        efficiency_test=[target_test.read_text()],
        duration_changes=duration_changes,
        human_performance=human_perf,
        version=version,
        # patch_functions left out intentionally per workflow
        test_functions=[],
        api=api,
        gt_commit_message=gt_commit_message,
        setup_commands=setup_commands,
        install_commands=install_commands,
        # notes=notes,
    )

    # Cleanup generated tests from repo tree
    try:
        shutil.rmtree(tests_root, ignore_errors=True)
    except Exception:
        pass

    return record


def save_and_push(records: List[CanonicalRecord], out_dir: Path, dataset_file_name: str, push_to_hf: bool, hf_repo_id: Optional[str] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use a safe filename for local output
    safe_file = dataset_file_name.replace("/", "__")
    jsonl_path = out_dir / f"{safe_file}.jsonl"
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"Wrote {jsonl_path}")

    try:
        import pandas as pd  # type: ignore
        from datasets import Dataset  # type: ignore
    except Exception:
        pd = None  # type: ignore
        Dataset = None  # type: ignore

    if push_to_hf and Dataset is not None:
        import pandas as pd  # type: ignore
        df = pd.DataFrame([asdict(r) for r in records])
        ds = Dataset.from_pandas(df)
        repo_id = hf_repo_id or dataset_file_name
        ds.push_to_hub(repo_id, split="test")
        print(f"Pushed to HF: {repo_id} (split=test)")


def main() -> None:
    # Determine config path: env OMNIPERF_CONFIG, CLI arg 1, or default
    cfg_path = os.environ.get("OMNIPERF_CONFIG") or (sys.argv[1] if len(sys.argv) > 1 else "commit_to_dataset.yaml")
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    if yaml is None:
        raise RuntimeError("PyYAML is required. Please install pyyaml.")

    with open(cfg_file, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Required
    repo_path = config.get("repo_path")
    head_commit = config.get("head_commit")
    if not repo_path or not head_commit:
        raise ValueError("Config must include 'repo_path' and 'head_commit'")

    # Optional
    base_commit = config.get("base_commit")
    extractions_dir = config.get("extractions_dir", "misc/experiments/commit_extractions_with_apis")
    use_docker = bool(config.get("use_docker", False))
    docker_image = config.get("docker_image", "ayushnangia16/nvidia-vllm-docker:latest")
    hf_repo = config.get("hf_repo")
    push_to_hf = bool(config.get("push_to_hf", False))
    setup_commands = config.get("setup_commands")
    install_commands = config.get("install_commands")
    api = config.get("api")
    notes = config.get("notes")

    record = assemble_canonical(
        repo_path_arg=repo_path,
        head_commit=head_commit,
        base_commit=base_commit,
        use_docker=use_docker,
        docker_image=docker_image,
        extractions_dir=extractions_dir,
        setup_commands=setup_commands,
        install_commands=install_commands,
        api=api,
        notes=notes,
    )

    default_repo_nm = str(config.get("dataset_name", "omni_commit_dataset"))
    hf_repo_id = _extract_hf_repo_id(hf_repo, default_repo_nm)
    dataset_name = (hf_repo_id.split("/", 1)[1] if hf_repo_id and "/" in hf_repo_id else default_repo_nm)
    out_dir = Path("data")
    save_and_push([record], out_dir, dataset_name, push_to_hf=bool(push_to_hf and hf_repo), hf_repo_id=hf_repo_id)


if __name__ == "__main__":
    main()


