"""
Create canonical OmniPerf-Bench dataset records for all commits in the extractions directory
and (optionally) export SWE-Perf/GSO compatible views, with optional push to Hugging Face.

Pipeline (batch commit processing):
1) Scan all JSON files in extractions_dir to discover commits
2) For each commit: Collect commit metadata (reuses collect.analysis.commits.PerfCommitAnalyzer)
3) For each commit: Build code-change fields (unified diff; split into patch vs test_patch; function names)
4) For each commit: Tests and timings (read/generate tests; run on base/head; parse times)
5) For each commit: Environment/version (compose version string; record setup/install commands if provided)
6) Assemble all records and export (canonical JSONL/Parquet; optional SWE-Perf/GSO views; push to HF)

Usage example (YAML config):
  # commit_to_dataset.yaml
  repo_path: /home/you/coding-mess/vllm
  extractions_dir: misc/experiments/commit_extractions_with_apis  # directory containing commit JSONs
  use_docker: false
  docker_image: ayushnangia16/nvidia-vllm-docker:latest
  hf_repo: yourname/omni-commit-dataset  # optional
  push_to_hf: false

Run:
  PYTHONPATH=src python src/collect/commit_to_dataset.py commit_to_dataset.yaml

The script will automatically process all JSON files in the extractions_dir (except extraction_summary.json),
extract commit hashes and parent hashes from each file, and create dataset records for all commits.

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
import logging
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

# Load environment variables from .env if present (so OPENROUTER_API_KEY is available)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Ensure local src/ is importable (for collect.* and test_scripts.*)
_ROOT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _ROOT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('commit_to_dataset.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

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

# Simple commit-hopping approach for vLLM API compatibility
# No complex environment managers needed - just checkout, install, test

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
_API_MANIFESTS: Dict[str, Dict[str, Any]] = {}

def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""

def _guess_project_name_from_pyproject(repo_path: Path) -> Optional[str]:
    """Best-effort parse of [project] name from pyproject.toml without adding deps."""
    try:
        pp = repo_path / "pyproject.toml"
        if not pp.exists():
            return None
        content = pp.read_text(encoding="utf-8", errors="ignore")
        # crude parse: find [project] section and name = "..."
        section_start = content.find("[project]")
        if section_start == -1:
            return None
        section = content[section_start:]
        # stop at next section
        for delim in ("\n[", "\r["):
            idx = section.find(delim)
            if idx != -1:
                section = section[:idx]
                break
        m = re.search(r"^\s*name\s*=\s*['\"]([^'\"]+)['\"]", section, re.MULTILINE)
        if m:
            return m.group(1).strip()
    except Exception:
        return None
    return None

def _candidate_import_names(repo_path: Path) -> List[str]:
    """Generate candidate top-level import names for the installed project."""
    candidates: List[str] = []
    # Common case for this repo's target
    candidates.append("vllm")
    project_name = _guess_project_name_from_pyproject(repo_path)
    if project_name:
        candidates.append(project_name)
        candidates.append(project_name.replace('-', '_'))
    repo_basename = repo_path.name
    if repo_basename:
        candidates.append(repo_basename)
        candidates.append(repo_basename.replace('-', '_'))
    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for c in candidates:
        if c and c not in seen:
            unique.append(c)
            seen.add(c)
    return unique

def _select_import_name_via_venv(venv_python: Path, repo_path: Path, prefer: Optional[str] = None) -> Optional[str]:
    """Attempt to import candidate names inside venv and return the first that succeeds."""
    if prefer:
        candidates = [prefer] + [c for c in _candidate_import_names(repo_path) if c != prefer]
    else:
        candidates = _candidate_import_names(repo_path)
    for name in candidates:
        try:
            r = subprocess.run(
                [str(venv_python), "-c", "import importlib,sys; importlib.import_module(sys.argv[1])", name],
                capture_output=True, text=True
            )
            if r.returncode == 0:
                return name
        except Exception:
            continue
    return None

def _write_api_dump_script(script_path: Path) -> None:
    """Write a small script that imports a package and emits a JSON manifest of public symbols."""
    script = """
import importlib, inspect, json, pkgutil, sys, types, traceback

def collect_manifest(import_name: str, max_modules: int, walk_all: bool) -> dict:
    manifest = {"package": import_name, "symbols": []}
    summary = {"modules_scanned": 0, "symbols_collected": 0, "errors": []}
    try:
        pkg = importlib.import_module(import_name)
    except Exception as e:
        summary["errors"].append({"stage": "import_root", "error": repr(e)})
        return {"manifest": manifest, "summary": summary}

    modules = [pkg.__name__]
    if hasattr(pkg, "__path__"):
        try:
            discovered = [m.name for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.')]
            modules.extend(discovered)
        except Exception as e:
            summary["errors"].append({"stage": "walk", "error": repr(e)})

    if not walk_all and max_modules is not None:
        modules = modules[:max_modules]

    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            summary["errors"].append({"stage": "import_module", "module": mod_name, "error": repr(e)})
            continue
        summary["modules_scanned"] += 1

        names = None
        try:
            names = getattr(mod, "__all__", None)
        except Exception:
            names = None
        if names is None:
            try:
                names = [n for n in dir(mod) if not n.startswith('_')]
            except Exception:
                names = []

        for n in names:
            try:
                obj = getattr(mod, n)
            except Exception:
                continue
            kind = None
            sig = None
            try:
                if inspect.isclass(obj):
                    kind = "class"
                    try:
                        sig = str(inspect.signature(obj))
                    except Exception:
                        sig = None
                elif inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
                    kind = "function"
                    try:
                        sig = str(inspect.signature(obj))
                    except Exception:
                        sig = None
                elif inspect.ismodule(obj):
                    kind = "module"
                elif callable(obj):
                    kind = "callable"
                    try:
                        sig = str(inspect.signature(obj))
                    except Exception:
                        sig = None
                else:
                    kind = "attribute"
            except Exception:
                kind = "unknown"
            manifest["symbols"].append({
                "module": mod_name,
                "name": n,
                "qualname": f"{mod_name}.{n}",
                "kind": kind,
                "signature": sig,
            })
            summary["symbols_collected"] += 1

    return {"manifest": manifest, "summary": summary}

def main():
    if len(sys.argv) < 5:
        print("{}", flush=True)
        return
    import_name = sys.argv[1]
    out_path = sys.argv[2]
    max_modules = int(sys.argv[3]) if sys.argv[3] else 200
    walk_all = sys.argv[4] == "1"
    payload = collect_manifest(import_name, max_modules, walk_all)
    with open(out_path, 'w') as f:
        json.dump(payload, f)
    print(out_path, flush=True)

if __name__ == "__main__":
    main()
"""
    script_path.write_text(script)

def snapshot_public_api(venv_python: Path, repo_path: Path, commit_hash: str, prefer_import_name: Optional[str] = None, work_dir: Optional[Path] = None) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Generate API manifest for installed package inside venv and return (path, summary)."""
    try:
        import_name = _select_import_name_via_venv(venv_python, repo_path, prefer=prefer_import_name)
        if not import_name:
            logger.warning(f"Could not determine import name for API snapshot at {commit_hash}")
            return None
        out_root = (work_dir or Path.cwd()) / "api_manifests"
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / f"{commit_hash[:8]}_{import_name}_api.json"

        script_path = (work_dir or Path.cwd()) / f"_api_dump_{commit_hash[:8]}.py"
        _write_api_dump_script(script_path)

        max_modules = int(os.getenv("OMNIPERF_API_MAX_MODULES", "200"))
        walk_all = "1" if os.getenv("OMNIPERF_API_WALK_ALL", "0") in ("1", "true", "True") else "0"

        r = subprocess.run(
            [str(venv_python), str(script_path), import_name, str(out_path), str(max_modules), walk_all],
            capture_output=True, text=True, cwd=str(repo_path)
        )
        if r.returncode != 0:
            logger.warning(f"API snapshot script failed for {commit_hash}: {r.stderr}")
            try:
                script_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            return None
        # Read summary
        try:
            data = json.loads(_read_text_safe(out_path))
            summary = data.get("summary", {}) if isinstance(data, dict) else {}
        except Exception:
            summary = {}
        try:
            script_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        return out_path, summary
    except Exception as e:
        logger.warning(f"Failed to snapshot API for {commit_hash}: {e}")
        return None

def _load_manifest_sets(manifest_path: Path) -> Optional[Tuple[str, set, Dict[str, List[str]]]]:
    try:
        payload = json.loads(_read_text_safe(manifest_path))
        manifest = payload.get("manifest", {}) if isinstance(payload, dict) else {}
        package_root = manifest.get("package", "")
        symbols = manifest.get("symbols", [])
        available: set = set()
        leaf_to_quals: Dict[str, List[str]] = {}
        for s in symbols:
            q = s.get("qualname")
            n = s.get("name")
            if isinstance(q, str) and isinstance(n, str):
                available.add(q)
                leaf_to_quals.setdefault(n, []).append(q)
        return package_root, available, leaf_to_quals
    except Exception as e:
        logger.warning(f"Failed to load API manifest sets from {manifest_path}: {e}")
        return None

def _parse_imports_and_aliases(code: str) -> Tuple[Dict[str, str], Dict[str, List[Tuple[str, Optional[str]]]]]:
    alias_to_module: Dict[str, str] = {}
    from_imports: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    lines = code.splitlines()
    import_re = re.compile(r"^\s*import\s+([\w\.]+)(?:\s+as\s+(\w+))?\s*$")
    from_re = re.compile(r"^\s*from\s+([\w\.]+)\s+import\s+(.+)$")
    for line in lines:
        m = import_re.match(line)
        if m:
            mod, alias = m.group(1), m.group(2)
            alias_to_module[alias or mod.split('.')[-1]] = mod
            continue
        m = from_re.match(line)
        if m:
            mod = m.group(1)
            names_part = m.group(2)
            # remove parentheses and split by commas
            names_part = names_part.strip().strip('()')
            parts = [p.strip() for p in names_part.split(',') if p.strip()]
            entries: List[Tuple[str, Optional[str]]] = []
            for p in parts:
                segs = p.split()
                if len(segs) == 1:
                    entries.append((segs[0], None))
                elif len(segs) == 3 and segs[1] == 'as':
                    entries.append((segs[0], segs[2]))
            from_imports.setdefault(mod, []).extend(entries)
    return alias_to_module, from_imports

def _extract_dotted_refs(code: str, alias_to_module: Dict[str, str], package_root: str) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []
    dotted = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")
    for m in dotted.finditer(code):
        left, right = m.group(1), m.group(2)
        mod = alias_to_module.get(left)
        if not mod:
            continue
        if not (mod == package_root or mod.startswith(package_root + '.')):
            continue
        refs.append((left, right))
    return refs

def _choose_best_qual(target_leaf: str, origin_module: str, candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    # Prefer within same module path or closest path by longest common prefix
    best = None
    best_score = -1
    for q in candidates:
        try:
            module_path = q.rsplit('.', 1)[0]
        except Exception:
            module_path = q
        score = 0
        # exact module match
        if module_path == origin_module:
            score = 100
        else:
            # common prefix length on segments
            a = origin_module.split('.')
            b = module_path.split('.')
            k = 0
            for x, y in zip(a, b):
                if x == y:
                    k += 1
                else:
                    break
            score = k
        if score > best_score:
            best = q
            best_score = score
    return best

def _insert_additional_imports(code: str, new_import_lines: List[str]) -> str:
    if not new_import_lines:
        return code
    lines = code.splitlines()
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(('import ', 'from ')):
            insert_idx = i + 1
        else:
            if insert_idx != 0:
                break
    return "\n".join(lines[:insert_idx] + new_import_lines + lines[insert_idx:])

def rewrite_test_script_against_manifest(test_script: Path, manifest_path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"rewrites": [], "added_imports": []}
    s = _load_manifest_sets(manifest_path)
    if s is None:
        return summary
    package_root, available, leaf_to_quals = s
    code = _read_text_safe(test_script)
    if not code:
        return summary
    alias_to_module, from_imports = _parse_imports_and_aliases(code)
    # Only consider aliases mapping to our package root
    alias_to_module = {a: m for a, m in alias_to_module.items() if m == package_root or m.startswith(package_root + '.')}

    # 1) Fix from-imports where module.path symbol no longer exists
    from_rewrites: Dict[Tuple[str, str], str] = {}
    for mod, entries in from_imports.items():
        if not (mod == package_root or mod.startswith(package_root + '.')):
            continue
        for name, alias in entries:
            qual = f"{mod}.{name}"
            if qual in available:
                continue
            cands = leaf_to_quals.get(name, [])
            best = _choose_best_qual(name, mod, cands)
            if best and best != qual:
                new_mod = best.rsplit('.', 1)[0]
                from_rewrites[(mod, name)] = new_mod
                summary["rewrites"].append({"type": "from", "old": qual, "new": f"{new_mod}.{name}"})

    if from_rewrites:
        def _rewrite_from_line(line: str) -> str:
            m = re.match(r"^(\s*from\s+)([\w\.]+)(\s+import\s+)(.+)$", line)
            if not m:
                return line
            prefix, mod, mid, tail = m.group(1), m.group(2), m.group(3), m.group(4)
            # rebuild tail respecting commas and aliases
            parts = [p.strip() for p in tail.strip().strip('()').split(',') if p.strip()]
            new_parts: List[str] = []

            # Check if any symbols need rewriting to different modules
            needs_rewrite = False
            dest_modules = set()
            changed_symbols = set()
            for p in parts:
                segs = p.split()
                if len(segs) == 1:
                    name, alias = segs[0], None
                elif len(segs) == 3 and segs[1] == 'as':
                    name, alias = segs[0], segs[2]
                else:
                    continue
                key = (mod, name)
                dest_mod = from_rewrites.get(key)
                if dest_mod:
                    needs_rewrite = True
                    dest_modules.add(dest_mod)
                    changed_symbols.add(name)

            if not needs_rewrite:
                # No rewrites needed for this line
                return line

            # Count symbols that don't need rewriting
            unchanged_count = sum(1 for p in parts if p.split() and p.split()[0] not in changed_symbols)

            # If all symbols that need rewriting go to the same module AND there are no unchanged symbols, rewrite the whole line
            if len(dest_modules) == 1 and unchanged_count == 0:
                new_mod = dest_modules.pop()
                return f"{prefix}{new_mod}{mid}{tail}"
            # Otherwise, split into separate import lines
            else:
                new_import_lines = []
                # Group symbols by their target module
                module_groups = {}
                for p in parts:
                    segs = p.split()
                    if len(segs) == 1:
                        name, alias = segs[0], None
                    elif len(segs) == 3 and segs[1] == 'as':
                        name, alias = segs[0], segs[2]
                    else:
                        # Invalid import part, skip
                        continue
                    key = (mod, name)
                    dest_mod = from_rewrites.get(key, mod)
                    if dest_mod not in module_groups:
                        module_groups[dest_mod] = []
                    module_groups[dest_mod].append(p)

                # Create import lines for each module
                for target_mod, symbols in module_groups.items():
                    new_import_lines.append(f"{prefix}{target_mod}{mid}{', '.join(symbols)}")

                return "\n".join(new_import_lines)

        code_lines = code.splitlines()
        code_lines = [_rewrite_from_line(ln) for ln in code_lines]
        code = "\n".join(code_lines)

    # 2) For dotted refs alias.symbol, add explicit from-imports and replace usage
    dotted_refs = _extract_dotted_refs(code, alias_to_module, package_root)
    additions: Dict[str, str] = {}  # name -> module
    replacements: List[Tuple[str, str]] = []  # (pattern, replacement)
    for alias, name in dotted_refs:
        origin_module = alias_to_module.get(alias)
        if not origin_module:
            continue
        qual = f"{origin_module}.{name}"
        if qual in available:
            continue
        cands = leaf_to_quals.get(name, [])
        best = _choose_best_qual(name, origin_module, cands)
        if not best:
            continue
        new_module = best.rsplit('.', 1)[0]
        additions[name] = new_module
        # replace only this exact token pattern
        pattern = rf"\b{re.escape(alias)}\.{re.escape(name)}\b"
        replacements.append((pattern, name))
        summary["rewrites"].append({"type": "dotted", "old": qual, "new": f"{new_module}.{name}"})

    # Insert new imports if any
    new_import_lines = [f"from {mod} import {name}" for name, mod in sorted(additions.items())]
    if new_import_lines:
        code = _insert_additional_imports(code, new_import_lines)
        summary["added_imports"] = new_import_lines

    # Apply replacements
    if replacements:
        for pattern, repl in replacements:
            code = re.sub(pattern, repl, code)

    try:
        test_script.write_text(code)
    except Exception as e:
        logger.warning(f"Failed to write rewritten test to {test_script}: {e}")
    return summary
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
    """Checkout commit and run prob_script with CUDA event timing for precise GPU measurement."""
    import torch
    import time as _time
    import subprocess

    checkout_commit(repo_path, commit_hash)

    try:
        # Run from repo root so imports resolve
        rel = str(test_entry.relative_to(repo_path))

        # Use CUDA events for precise GPU timing if GPU available
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure GPU is ready

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            # Run the prob_script - it will execute the performance test
            result = subprocess.run([sys.executable, rel], cwd=str(repo_path),
                                  capture_output=True, text=True, timeout=300)

            end_event.record()
            torch.cuda.synchronize()

            # Get precise GPU execution time in milliseconds
            execution_time = start_event.elapsed_time(end_event)

        else:
            # Fallback to CPU timing if no GPU available
            start = _time.time()
            result = subprocess.run([sys.executable, rel], cwd=str(repo_path),
                                  capture_output=True, text=True, timeout=300)
            end = _time.time()
            execution_time = (end - start) * 1000  # Convert to milliseconds

        # Check for script errors and print output for debugging
        if result.returncode != 0:
            print(f"Script exited with code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return [float('inf')]

        return [execution_time]

    except subprocess.TimeoutExpired:
        print(f"Script execution timed out after 300 seconds: {test_entry}")
        return [300000.0]  # 5 minutes in milliseconds
    except Exception as e:
        print(f"Error running script: {e}")
        return [float('inf')]


def check_capability_requirements(test_script: Path) -> bool:
    """Check if current hardware meets test requirements."""
    capabilities = detect_capabilities()

    if not capabilities["cuda_available"]:
        logger.warning("CUDA not available - skipping GPU tests")
        return False

    # Check specific generator requirements
    generator_name = test_script.stem

    # FP8-related tests require SM90+ (Hopper GPUs)
    fp8_tests = ["8d75fe48", "2a052011"]  # Add more FP8 tests as needed
    if any(fp8_test in generator_name for fp8_test in fp8_tests):
        if not capabilities.get("supports_fp8", False):
            logger.warning(f"FP8 not supported (SM{capabilities['sm_version']}, CUDA {capabilities['cuda_version']}) - skipping {generator_name}")
            return False

    return True


def detect_capabilities() -> Dict[str, Any]:
    """Detect GPU and CUDA capabilities for hardware filtering."""
    capabilities = {
        "cuda_available": False,
        "gpu_name": None,
        "cuda_version": None,
        "gpu_memory_gb": 0,
        "sm_version": None,
        "supports_fp8": False
    }

    try:
        import torch
        if torch.cuda.is_available():
            capabilities["cuda_available"] = True
            capabilities["gpu_name"] = torch.cuda.get_device_name(0)
            capabilities["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # Get CUDA version
            try:
                cuda_version = torch.version.cuda
                if cuda_version:
                    capabilities["cuda_version"] = cuda_version
            except:
                pass

            # Get SM version for FP8 requirements
            try:
                sm_version = torch.cuda.get_device_capability(0)
                capabilities["sm_version"] = f"{sm_version[0]}{sm_version[1]}"
                # FP8 requires SM90+ (not SM89) and CUDA 12.x
                if sm_version[0] >= 9 and cuda_version and cuda_version.startswith("12"):
                    capabilities["supports_fp8"] = True
                else:
                    capabilities["supports_fp8"] = False
            except:
                capabilities["supports_fp8"] = False

    except ImportError:
        pass

    logger.info(f"Detected capabilities: {capabilities}")
    return capabilities


def run_tests_with_commit_hopping(
    test_script: Path,
    commit_hash: str,
    repo_path: Path,
    work_dir: Path,
    api_rewrite: bool = False
) -> List[float]:
    """Run test using simple commit-hopping approach with uv.

    This is the winning approach: checkout commit, install with uv, run test.
    Much simpler than complex environment isolation.
    """
    logger.info(f"Running test with commit-hopping for {commit_hash}")

    # Check hardware capabilities first
    if not check_capability_requirements(test_script):
        logger.warning(f"Hardware requirements not met for {test_script}")
        return [float('inf')]

    # Determine Python version for this commit
    python_version = get_python_version_for_commit(commit_hash)

    try:
        # Save current commit to restore later
        current_commit = run(["git", "rev-parse", "HEAD"], cwd=repo_path).strip()

        # Checkout target commit
        logger.info(f"Checking out commit {commit_hash}")
        checkout_commit(repo_path, commit_hash)

        # Create venv with appropriate Python version
        venv_path = work_dir / f"venv_{commit_hash[:8]}"
        logger.info(f"Creating venv with Python {python_version}")

        # Clean up any existing venv
        if venv_path.exists():
            import shutil
            shutil.rmtree(venv_path)

        result = subprocess.run([
            "uv", "venv", "--python", python_version, str(venv_path)
        ], capture_output=True, text=True, cwd=str(work_dir))

        if result.returncode != 0:
            logger.error(f"Failed to create venv: {result.stderr}")
            return [float('inf')]

        # Install vLLM using pre-built wheels from vLLM wheel index
        venv_python = venv_path / "bin" / "python"
        logger.info(f"Installing vLLM pre-built wheel for commit {commit_hash}...")
        
        # Use vLLM's wheel index for the specific commit
        wheel_index_url = f"https://wheels.vllm.ai/{commit_hash}"
        result = subprocess.run([
            "uv", "pip", "install", "vllm", 
            "--torch-backend=auto",
            "--extra-index-url", wheel_index_url,
            "--python", str(venv_python)
        ], capture_output=True, text=True, cwd=str(work_dir))

        if result.returncode != 0:
            logger.warning(f"Failed to install vLLM wheel for commit {commit_hash}: {result.stderr}")
            logger.info("Falling back to source installation...")
            wheel_ok = False
            
            # Fallback: install from source if wheel not available
            requirements_file = repo_path / "requirements.txt"
            if requirements_file.exists():
                logger.info("Installing requirements.txt with uv...")
                result = subprocess.run([
                    "uv", "pip", "install", "-r", "requirements.txt", "--python", str(venv_python)
                ], capture_output=True, text=True, cwd=str(repo_path))
                
                if result.returncode != 0:
                    logger.error(f"Failed to install requirements.txt: {result.stderr}")
                    return [float('inf')]
            
            # Install build tools
            result = subprocess.run([
                "uv", "pip", "install", "setuptools", "wheel", "build", "--python", str(venv_python)
            ], capture_output=True, text=True, cwd=str(work_dir))
            
            # Install from source
            result = subprocess.run([
                "uv", "pip", "install", "-e", ".", "--python", str(venv_python)
            ], capture_output=True, text=True, cwd=str(repo_path))

            if result.returncode != 0:
                logger.error(f"Failed to install vLLM from source: {result.stderr}")
                return [float('inf')]
        else:
            logger.info(f"Successfully installed vLLM wheel for commit {commit_hash}")
            wheel_ok = True

        # Snapshot public API after installation
        prefer_name = "vllm" if wheel_ok else None
        api_snapshot = snapshot_public_api(venv_python=venv_python, repo_path=repo_path, commit_hash=commit_hash, prefer_import_name=prefer_name, work_dir=work_dir)
        if api_snapshot is not None:
            manifest_path, summary = api_snapshot
            _API_MANIFESTS[commit_hash] = {"path": str(manifest_path), "summary": summary}
            logger.info(f"API manifest written: {manifest_path}")
        else:
            logger.warning(f"API manifest not generated for {commit_hash}")

        # Optionally rewrite the generated test code to align with available API
        if api_rewrite and api_snapshot is not None:
            try:
                manifest_path = Path(_API_MANIFESTS[commit_hash]["path"])  # type: ignore[index]
                rewrite_summary = rewrite_test_script_against_manifest(test_script, manifest_path)
                logger.info(f"API rewrite summary: {rewrite_summary}")
            except Exception as e:
                logger.warning(f"Failed to rewrite test script against manifest: {e}")

        # Copy test script to work_dir (NOT repo) to avoid import conflicts
        test_dest = work_dir / f"test_{commit_hash[:8]}.py"
        import shutil
        shutil.copy2(test_script, test_dest)

        # Run the test from work_dir to avoid local vLLM source interference
        logger.info("Running test...")
        env = os.environ.copy()
        env["PYTHONPATH"] = ""  # Clear PYTHONPATH to avoid local repo interference
        result = subprocess.run([
            str(venv_python), str(test_dest)
        ], capture_output=True, text=True, cwd=str(work_dir), env=env, timeout=300)

        # Parse timing from output
        timing = parse_execution_time(result.stdout)
        if timing is not None:
            logger.info(f"Test completed successfully: {timing:.6f}s")
            return [timing * 1000]  # Convert to milliseconds

        # Check for errors
        if result.returncode != 0:
            logger.error(f"Test execution failed: {result.stderr}")
            return [float('inf')]

        logger.warning("Could not parse execution time from test output")
        return [float('inf')]

    except subprocess.TimeoutExpired:
        logger.error("Test execution timed out")
        return [300000.0]  # 5 minutes in milliseconds
    except Exception as e:
        logger.error(f"Error during commit-hopping test execution: {e}")
        return [float('inf')]
    finally:
        # Always restore original commit
        try:
            checkout_commit(repo_path, current_commit)
        except Exception as e:
            logger.warning(f"Failed to restore original commit: {e}")

        # Clean up test file
        test_dest = work_dir / f"test_{commit_hash[:8]}.py"
        if test_dest.exists():
            test_dest.unlink()


def get_python_version_for_commit(commit_hash: str) -> str:
    """Determine appropriate Python version for a commit."""
    # vLLM currently supports Python 3.9-3.12, not 3.13+
    # Try to use the most compatible version available
    
    # Older commits requiring PyTorch 2.3.x need Python 3.11
    old_commits = [
        "2a052011", "2bb0489c", "2deb029d", "8d75fe48", "0f40557a",
        "2f192835", "3a243095"
    ]

    if any(commit_hash.startswith(old) for old in old_commits):
        # Try Python 3.11 first for old commits
        if os.path.exists("/usr/bin/python3.11"):
            return "/usr/bin/python3.11"
        elif os.path.exists("python3.11"):
            return "python3.11"

    # For newer commits, prefer Python 3.11 (most stable for vLLM)
    # But fall back to other compatible versions if needed
    for python_version in ["python3.11", "python3.10", "python3.9", "python3.12"]:
        if os.path.exists(f"/usr/bin/{python_version}"):
            return f"/usr/bin/{python_version}"
        elif os.path.exists(python_version):
            return python_version
    
    # Last resort: use python3 (but this might fail on Python 3.13+)
    return "python3"


def parse_execution_time(output: str) -> Optional[float]:
    """Parse execution time from test output."""
    import re
    import json as _json
    # Look for timing patterns in output
    patterns = [
        r"Execution time:\s*([0-9]+\.?[0-9]*)s",
        r"Time:\s*([0-9]+\.?[0-9]*)s",
        r"([0-9]+\.?[0-9]*)\s*ms",
        r"([0-9]+\.?[0-9]*)\s*seconds"
    ]

    # First, attempt to parse JSON summaries printed by generators
    try:
        for line in (output or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{") and line.endswith("}") and "avg_ms" in line:
                try:
                    payload = _json.loads(line)
                    if isinstance(payload, dict) and "avg_ms" in payload:
                        avg_ms_val = float(payload["avg_ms"])  # already in milliseconds
                        return avg_ms_val / 1000.0  # convert to seconds
                except Exception:
                    continue
    except Exception:
        pass

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            time_value = float(match.group(1))
            # Convert ms to seconds if needed
            if "ms" in pattern:
                time_value /= 1000
            return time_value

    return None


def run_tests_in_docker(repo_path: Path, commit_hash: str, test_entry: Path, docker_image: str) -> List[float]:
    """Checkout commit and run prob_script in docker with CUDA event timing for precise GPU measurement."""
    import torch
    import time as _time
    import subprocess

    checkout_commit(repo_path, commit_hash)

    try:
        rel_path = test_entry.relative_to(repo_path)

        # Prepare docker command to run the prob_script
        docker_cmd = [
            "docker", "run", "--rm", "-t",
            "--gpus", "all",
            "-v", f"{repo_path}:/workspace",
            "-w", "/workspace",
            docker_image,
            "python", str(rel_path)
        ]

        # Use CUDA events for precise GPU timing if GPU available
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure GPU is ready

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            # Run the prob_script in docker
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=600)

            end_event.record()
            torch.cuda.synchronize()

            # Get precise GPU execution time in milliseconds
            execution_time = start_event.elapsed_time(end_event)

        else:
            # Fallback to wall clock timing if no GPU available
            start = _time.time()
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=600)
            end = _time.time()
            execution_time = (end - start) * 1000  # Convert to milliseconds

        # Check for script errors and print output for debugging
        if result.returncode != 0:
            print(f"Docker script exited with code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return [float('inf')]

        return [execution_time]

    except subprocess.TimeoutExpired:
        print(f"Docker script execution timed out after 600 seconds: {test_entry}")
        return [600000.0]  # 10 minutes in milliseconds
    except Exception as e:
        print(f"Error running docker script: {e}")
        return [float('inf')]


def _parse_times(stdout: str) -> List[float]:
    if parse_times is not None:
        try:
            return parse_times(stdout)
        except Exception:
            pass
    # Fallback regex
    pattern = re.compile(r"Execution time:\s*([0-9]+\.?[0-9]*)s")
    return [float(m.group(1)) for m in pattern.finditer(stdout or "")]


def find_or_generate_test_script(
    commit_hash: str,
    extractions_dir: Path,
    out_dir: Path,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_temperature: Optional[float] = None,
    llm_max_tokens: Optional[int] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate a pre-generated test-case-generator for the commit, or generate one via LLM.

    Returns path to the generated test module file, or None if unavailable.
    """
    logger.info(f"Starting test script resolution for commit {commit_hash}")
    hash8 = commit_hash[:8]
    logger.info(f"Using hash8: {hash8}")

    # Validate inputs
    logger.info(f"Validating extractions_dir: {extractions_dir}")
    if not extractions_dir.exists() or not extractions_dir.is_dir():
        logger.error(f"extractions_dir not found or not a directory: {extractions_dir}")
        raise RuntimeError(f"extractions_dir not found or not a directory: {extractions_dir}")
    logger.info("extractions_dir validation passed")

    # 1) Prefer pre-generated test generators if available
    logger.info("Checking for pre-generated test generators")
    pregenerated_root = Path("misc/experiments/generated_test_generators_v4")
    candidates = [
        pregenerated_root / f"{commit_hash}_test_case_generator.py",
        pregenerated_root / f"{hash8}_test_case_generator.py",
    ]
    logger.info(f"Checking candidates: {candidates}")
    for c in candidates:
        logger.info(f"Checking if exists: {c}")
        if c.exists():
            logger.info(f"Using pre-generated test generator: {c}")
            # No JSON needed in this branch; return None for json_path
            return c, None
    logger.info("No pre-generated test generators found")

    # 2) If not found, attempt on-the-fly generation via LLM
    logger.info("Attempting on-the-fly generation via LLM")
    if process_extraction_file is None or LLMClient is None:
        logger.warning(
            "LLM utilities unavailable and no pre-generated script found; cannot generate tests dynamically."
        )
        logger.warning(f"process_extraction_file: {process_extraction_file}, LLMClient: {LLMClient}")
        raise RuntimeError("No test generator available (missing pre-generated file and LLM utilities)")

    # Find matching extraction JSON by full or prefix hash
    json_path = None
    logger.info(f"Searching for extraction JSON in {extractions_dir}")
    json_files = list(extractions_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files: {[p.name for p in json_files]}")
    for p in json_files:
        name = p.stem
        logger.info(f"Checking JSON file: {name} against commit_hash: {commit_hash} (hash8: {hash8})")
        if name.startswith(commit_hash) or commit_hash.startswith(name) or name.startswith(hash8):
            json_path = p
            logger.info(f"Found matching extraction JSON: {json_path}")
            break

    if json_path is None:
        logger.error(f"No commit extraction JSON found for commit {commit_hash} in {extractions_dir}")
        logger.error(f"Available JSON files: {[p.stem for p in json_files]}")
        raise RuntimeError(
            f"No commit extraction JSON found for commit {commit_hash} in {extractions_dir}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created/verified: {out_dir}")

    # Require API credentials for generation
    logger.info("Checking LLM API credentials")
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    logger.info(
        "API credentials - OpenAI: %s, Anthropic: %s, OpenRouter: %s" % (
            'present' if has_openai else 'missing',
            'present' if has_anthropic else 'missing',
            'present' if has_openrouter else 'missing',
        )
    )
    if not (has_openai or has_anthropic or has_openrouter):
        logger.warning("Missing LLM credentials; skipping dynamic generation.")
        raise RuntimeError("Missing LLM credentials and no pre-generated script available")

    try:
        logger.info("Initializing LLM client")
        client_kwargs: Dict[str, Any] = {}
        if llm_provider is not None:
            client_kwargs["provider"] = llm_provider
        if llm_model is not None:
            client_kwargs["model"] = llm_model
        if llm_temperature is not None:
            client_kwargs["temperature"] = llm_temperature
        if llm_max_tokens is not None:
            client_kwargs["max_tokens"] = int(llm_max_tokens)
        logger.info(f"LLM client kwargs: {client_kwargs}")
        client = LLMClient(**client_kwargs)
        logger.info(f"LLM client initialized with provider: {client.provider}, model: {client.model}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        raise

    try:
        logger.info(f"About to call process_extraction_file with: {json_path}, {out_dir}")
        logger.info("This may take a while as it involves LLM generation...")
        result = process_extraction_file(str(json_path), str(out_dir), client)  # type: ignore
        logger.info(f"process_extraction_file completed successfully, result: {result}")
    except Exception as e:
        logger.error(f"Error during process_extraction_file: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

    if result and "script_path" in result:
        path = Path(result["script_path"])  # type: ignore
        logger.info(f"Generated script path: {path}")
        if path.exists():
            logger.info(f"Script file exists and is accessible: {path}")
            try:
                content = path.read_text()
                logger.info(f"Generated script content length: {len(content)} characters")
                logger.debug(f"Generated script content preview: {content[:500]}...")
            except Exception as e:
                logger.error(f"Failed to read generated script content: {e}")
            return path, json_path
        else:
            logger.error(f"Generated script path does not exist: {path}")
    else:
        logger.error("LLM generation did not return a valid script_path")

    logger.error("LLM generation did not return a valid script_path.")
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
    api_manifest_paths: Optional[Dict[str, str]] = None
    api_manifest_summaries: Optional[Dict[str, Any]] = None


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
    base_commit: str,
    use_docker: bool,
    docker_image: str,
    extractions_dir: str,
    setup_commands: List[str] = None,
    install_commands: List[str] = None,
    api: Optional[str] = None,
    notes: Optional[str] = None,
) -> CanonicalRecord:
    logger.info(f"Starting assemble_canonical for commit {head_commit}")
    
    # Prepare workspace
    # Always use local repo path; clone to temp workspace to avoid mutating the user's checkout
    # Derive a provisional repo_name from the source; refine after cloning via origin URL if available
    logger.info(f"Preparing workspace for repo: {repo_path_arg}")
    provisional_name = os.path.basename(repo_path_arg.rstrip("/")) or "repo"
    if provisional_name.endswith(".git"):
        provisional_name = provisional_name[:-4]
    work_root = Path(tempfile.mkdtemp(prefix="omni_commit_"))
    repo_path = work_root / provisional_name
    logger.info(f"Cloning repo to temp workspace: {repo_path}")
    clone_or_update_repo(repo_path_arg, repo_path)
    logger.info(f"Repository cloned successfully to {repo_path}")

    # Derive repo owner/name from git origin if possible; otherwise fall back
    logger.info("Deriving repo owner/name from git origin")
    try:
        origin_url = run(["git", "config", "--get", "remote.origin.url"], cwd=repo_path).strip()
        logger.info(f"Found origin URL: {origin_url}")
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
        logger.info(f"Derived repo info: {repo_owner}/{repo_name}")
    except Exception as e:
        logger.warning(f"Failed to derive repo info from origin: {e}")
        repo_owner, repo_name = "local", provisional_name
        logger.info(f"Using fallback repo info: {repo_owner}/{repo_name}")

    # Collect commit metadata (must use PerfCommitAnalyzer)
    logger.info("Starting PerfCommitAnalyzer processing")
    if PerfCommitAnalyzer is None:
        raise RuntimeError("PerfCommitAnalyzer is required but not available")
    created_at_iso = datetime.now(timezone.utc).isoformat()
    logger.info(f"Processing commit {head_commit} with PerfCommitAnalyzer")
    perf_commit = PerfCommitAnalyzer.process_commit(head_commit, repo_path, max_year=None)  # type: ignore
    logger.info("PerfCommitAnalyzer.process_commit completed")
    if perf_commit is None:
        raise RuntimeError("PerfCommitAnalyzer returned None for the specified commit")
    gt_commit_message = perf_commit.message
    logger.info(f"Got commit message: {gt_commit_message[:100]}...")
    if getattr(perf_commit, "date", None) is not None:
        try:
            created_at_iso = perf_commit.date.astimezone(timezone.utc).isoformat()
        except Exception:
            pass
    unified = perf_commit.diff_text or git_unified_diff(repo_path, base_commit, head_commit)
    logger.info(f"Got unified diff, length: {len(unified) if unified else 0}")

    patch = unified or ""
    test_patch = ""

    # Obtain a test script for this commit via existing/generated test-case generator
    logger.info("Starting test script generation process")
    extr_dir = Path(extractions_dir) if extractions_dir else Path("misc/experiments/commit_extractions_with_apis")
    gen_out_dir = Path("misc/experiments/generated_test_generators_v4")
    logger.info(f"Extraction directory: {extr_dir}")
    logger.info(f"Generator output directory: {gen_out_dir}")

    # LLM config overrides via environment variables (set upstream in main)
    llm_provider = os.getenv("OMNIPERF_LLM_PROVIDER")
    llm_model = os.getenv("OMNIPERF_LLM_MODEL")
    llm_temperature_env = os.getenv("OMNIPERF_LLM_TEMPERATURE")
    llm_max_tokens_env = os.getenv("OMNIPERF_LLM_MAX_TOKENS")
    llm_temperature_val = float(llm_temperature_env) if llm_temperature_env else None
    llm_max_tokens_val = int(llm_max_tokens_env) if llm_max_tokens_env else None
    
    logger.info(f"LLM config - provider: {llm_provider}, model: {llm_model}, temp: {llm_temperature_val}, max_tokens: {llm_max_tokens_val}")
    logger.info(f"About to call find_or_generate_test_script for commit {head_commit}")

    test_script, json_path = find_or_generate_test_script(
        head_commit,
        extr_dir,
        gen_out_dir,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_temperature=llm_temperature_val,
        llm_max_tokens=llm_max_tokens_val,
    )
    logger.info(f"find_or_generate_test_script completed, returned test_script: {test_script}")
    if test_script is None:
        logger.error("Unable to locate or generate a test script for this commit.")
        raise RuntimeError("Unable to locate or generate a test script for this commit.")

    logger.info(f"Successfully obtained test script: {test_script}")

    # Materialize test into repo workspace and run via pytest across base/head/main
    tests_root = repo_path / "_generated_perf_tests"
    tests_root.mkdir(parents=True, exist_ok=True)
    target_test = tests_root / "test_generated.py"
    logger.info(f"Target test file: {target_test}")

    try:
        test_code_text = Path(test_script).read_text()
        logger.info(f"Read test script content, length: {len(test_code_text)} characters")
    except Exception as e:
        logger.error(f"Failed to read test script content from {test_script}: {e}")
        raise

    if not test_code_text or not test_code_text.strip():
        logger.error("Generated efficiency_test script is empty. Aborting.")
        logger.error(f"Script path: {test_script}")
        logger.error(f"Script exists: {test_script.exists()}")
        logger.error(f"Script size: {test_script.stat().st_size if test_script.exists() else 'N/A'}")
        raise RuntimeError("Generated efficiency_test script is empty. Aborting.")

    logger.info(f"Writing test code to target file: {target_test}")
    target_test.write_text(test_code_text)

    # Log efficiency_test content characteristics to stdout for traceability
    newline_count = test_code_text.count('\n') + 1
    logger.info(f"Generated efficiency_test script: chars={len(test_code_text)}; lines={newline_count}")
    print(
        f"Generated efficiency_test script: chars={len(test_code_text)}; lines={newline_count}"
    )
    preview = test_code_text[:400]
    if preview:
        logger.info(f"efficiency_test preview (first 400 chars): {preview}")
        print("efficiency_test preview (first 400 chars):\n" + preview)

    # Log the efficiency_test field that will be set
    logger.info("Setting efficiency_test field in CanonicalRecord")
    logger.info(f"efficiency_test will contain: {len(test_code_text)} characters")

    # Use simple commit-hopping approach for vLLM API compatibility
    logger.info("Using simple commit-hopping approach for test execution")
    work_dir = Path.cwd() / ".test_work"
    work_dir.mkdir(exist_ok=True)

    # Test base commit
    logger.info(f"Testing base commit: {base_commit}")
    base_times_arr = run_tests_with_commit_hopping(
        test_script=test_script,
        commit_hash=base_commit,
        repo_path=repo_path,
        work_dir=work_dir,
        api_rewrite=False
    )

    # Test head commit
    logger.info(f"Testing head commit: {head_commit}")
    head_times_arr = run_tests_with_commit_hopping(
        test_script=test_script,
        commit_hash=head_commit,
        repo_path=repo_path,
        work_dir=work_dir,
        api_rewrite=True
    )

    # Test main branch
    main_head = get_main_branch_head(repo_path)
    logger.info(f"Testing main branch: {main_head}")
    main_times_arr = run_tests_with_commit_hopping(
        test_script=test_script,
        commit_hash=main_head,
        repo_path=repo_path,
        work_dir=work_dir,
        api_rewrite=False
    )

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

    logger.info("Creating CanonicalRecord")
    logger.info(f"efficiency_test field will be set with list containing 1 item of {len(test_code_text)} characters")

    # Gather API manifest metadata if present
    api_manifest_paths: Dict[str, str] = {}
    api_manifest_summaries: Dict[str, Any] = {}
    for label, ch in (("base", base_commit), ("head", head_commit), ("main", main_head)):
        entry = _API_MANIFESTS.get(ch)
        if entry and isinstance(entry, dict) and "path" in entry:
            api_manifest_paths[label] = str(entry.get("path"))
            api_manifest_summaries[label] = entry.get("summary", {})

    record = CanonicalRecord(
        repo=f"{repo_owner}/{repo_name}",
        instance_id=instance_id,
        created_at=created_at_iso,
        base_commit=base_commit,
        head_commit=head_commit,
        patch=patch,
        test_patch=test_patch,
        efficiency_test=[test_code_text],
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
        api_manifest_paths=api_manifest_paths or None,
        api_manifest_summaries=api_manifest_summaries or None,
    )

    # Verify the record was created correctly
    logger.info(f"CanonicalRecord created successfully")
    logger.info(f"Record efficiency_test field has {len(record.efficiency_test)} items")
    if record.efficiency_test:
        logger.info(f"First efficiency_test item length: {len(record.efficiency_test[0])}")
        logger.debug(f"First efficiency_test item preview: {record.efficiency_test[0][:200]}...")
    else:
        logger.error("CRITICAL: efficiency_test field is empty after record creation!")

    # Cleanup generated tests from repo tree
    try:
        shutil.rmtree(tests_root, ignore_errors=True)
    except Exception:
        pass

    return record


def save_and_push(records: List[CanonicalRecord], out_dir: Path, dataset_file_name: str, push_to_hf: bool, hf_repo_id: Optional[str] = None) -> None:
    logger.info(f"Starting save_and_push with {len(records)} records")
    logger.info(f"Output directory: {out_dir}, dataset name: {dataset_file_name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Use a safe filename for local output
    safe_file = dataset_file_name.replace("/", "__")
    jsonl_path = out_dir / f"{safe_file}.jsonl"
    logger.info(f"Output JSONL path: {jsonl_path}")

    with open(jsonl_path, "w") as f:
        for i, r in enumerate(records):
            logger.info(f"Processing record {i+1}/{len(records)}")
            logger.info(f"Record efficiency_test field has {len(r.efficiency_test) if r.efficiency_test else 0} items")

            record_dict = asdict(r)
            logger.info(f"Serialized record has efficiency_test with {len(record_dict.get('efficiency_test', []))} items")

            # Check if efficiency_test is being serialized properly
            if 'efficiency_test' in record_dict:
                eff_test = record_dict['efficiency_test']
                if isinstance(eff_test, list) and len(eff_test) > 0:
                    logger.info(f"efficiency_test[0] length: {len(eff_test[0])}")
                else:
                    logger.error(f"efficiency_test is empty or not a list: {type(eff_test)}")
            else:
                logger.error("efficiency_test key missing from serialized record!")

            f.write(json.dumps(record_dict) + "\n")

    logger.info(f"Wrote {jsonl_path}")
    print(f"Wrote {jsonl_path}")

    try:
        from datasets import Dataset, load_dataset, concatenate_datasets  # type: ignore
    except Exception:
        Dataset = None  # type: ignore
        load_dataset = None  # type: ignore
        concatenate_datasets = None  # type: ignore

    if push_to_hf and Dataset is not None:
        repo_id = hf_repo_id or dataset_file_name

        # Prepare new records as pure Python dicts (avoid pandas to preserve nested types like lists)
        new_records = [asdict(r) for r in records]

        # Verbose logging for debugging nested payloads
        if records:
            r0 = records[0]
            et_list = getattr(r0, "efficiency_test", None)
            et_len = len(et_list) if isinstance(et_list, list) else 0
            first_snippet = ""
            if et_len > 0 and isinstance(et_list[0], str):
                first_snippet = et_list[0][:400]
            print(
                f"Preparing push → repo_id={repo_id}; new_records={len(new_records)}; "
                f"efficiency_test_entries={et_len}; first_entry_chars={len(et_list[0]) if et_len > 0 and isinstance(et_list[0], str) else 0}"
            )
            if first_snippet:
                print("efficiency_test[0] preview:\n" + first_snippet)

        # Load existing split (if any)
        existing_records: List[Dict[str, Any]] = []
        if load_dataset is not None:
            try:
                existing_ds = load_dataset(repo_id, split="test")  # type: ignore
                # Iterate to avoid conversions that may strip nested fields
                existing_records = [row for row in existing_ds]  # type: ignore
                print(f"Loaded existing split: {len(existing_records)} rows")
            except Exception as e:
                print(f"No existing split found or failed to load ('test'): {e}")

        # Merge and de-duplicate by (repo, head_commit), keeping the new record on conflict
        by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for rec in existing_records:
            repo_val = str(rec.get("repo", ""))
            head_val = str(rec.get("head_commit", ""))
            by_key[(repo_val, head_val)] = rec
        for rec in new_records:
            repo_val = str(rec.get("repo", ""))
            head_val = str(rec.get("head_commit", ""))
            by_key[(repo_val, head_val)] = rec  # overwrite to keep latest

        combined_records: List[Dict[str, Any]] = list(by_key.values())
        print(f"Combined rows (post-dedup): {len(combined_records)}")

        ds = Dataset.from_list(combined_records)  # type: ignore
        ds.push_to_hub(repo_id, split="test")
        print(f"Pushed to HF (appended): {repo_id} (split=test, rows={len(ds)})")


def scan_extraction_files(extractions_dir: Path) -> List[Tuple[str, str]]:
    """Scan all JSON files in extractions directory and return list of (commit_hash, parent_hash) tuples."""
    logger.info(f"Scanning extraction files in {extractions_dir}")
    extraction_files = []

    if not extractions_dir.exists() or not extractions_dir.is_dir():
        logger.error(f"Extractions directory not found or not a directory: {extractions_dir}")
        raise RuntimeError(f"Extractions directory not found or not a directory: {extractions_dir}")

    for json_file in extractions_dir.glob("*.json"):
        if json_file.name == "extraction_summary.json":
            continue  # Skip the summary file

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            commit_hash = data.get("commit_hash")
            parent_hash = data.get("parent_hash")

            if commit_hash and parent_hash:
                extraction_files.append((commit_hash, parent_hash))
                logger.debug(f"Found commit {commit_hash} with parent {parent_hash}")
            else:
                logger.warning(f"Missing commit_hash or parent_hash in {json_file}")

        except Exception as e:
            logger.warning(f"Error reading {json_file}: {e}")
            continue

    logger.info(f"Found {len(extraction_files)} valid extraction files")
    return extraction_files


def process_batch_commits(
    repo_path: str,
    extraction_files: List[Tuple[str, str]],
    use_docker: bool,
    docker_image: str,
    extractions_dir: str,
    setup_commands: Optional[List[str]] = None,
    install_commands: Optional[List[str]] = None,
    api: Optional[str] = None,
    notes: Optional[str] = None,
) -> List[CanonicalRecord]:
    """Process all commits in batch and return list of CanonicalRecords."""
    logger.info(f"Starting batch processing of {len(extraction_files)} commits")
    records = []

    for i, (head_commit, base_commit) in enumerate(extraction_files):
        logger.info(f"Processing commit {i+1}/{len(extraction_files)}: {head_commit}")

        try:
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
            records.append(record)
            logger.info(f"Successfully processed commit {head_commit}")

        except Exception as e:
            logger.error(f"Failed to process commit {head_commit}: {e}")
            # Continue processing other commits even if one fails
            continue

    logger.info(f"Batch processing completed. Successfully processed {len(records)}/{len(extraction_files)} commits")
    return records


def main() -> None:
    logger.info("Starting commit_to_dataset main function")

    # Determine config path: env OMNIPERF_CONFIG, CLI arg 1, or default
    cfg_path = os.environ.get("OMNIPERF_CONFIG") or (sys.argv[1] if len(sys.argv) > 1 else "commit_to_dataset.yaml")
    logger.info(f"Using config path: {cfg_path}")
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        logger.error(f"Config file not found: {cfg_file}")
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    if yaml is None:
        logger.error("PyYAML is required. Please install pyyaml.")
        raise RuntimeError("PyYAML is required. Please install pyyaml.")

    with open(cfg_file, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    logger.info(f"Loaded config: {config}")

    # Required
    repo_path = config.get("repo_path")
    if not repo_path:
        logger.error("Config must include 'repo_path'")
        raise ValueError("Config must include 'repo_path'")

    # Optional
    extractions_dir = config.get("extractions_dir", "misc/experiments/commit_extractions_with_apis")
    use_docker = bool(config.get("use_docker", False))
    docker_image = config.get("docker_image", "ayushnangia16/nvidia-vllm-docker:latest")
    hf_repo = config.get("hf_repo")
    push_to_hf = bool(config.get("push_to_hf", False))
    setup_commands = config.get("setup_commands")
    install_commands = config.get("install_commands")
    api = config.get("api")
    notes = config.get("notes")

    # LLM overrides from config (propagated via env so downstream utilities can read them too)
    llm_provider = config.get("llm_provider")
    llm_model = config.get("llm_model")
    llm_temperature = config.get("llm_temperature")
    llm_max_tokens = config.get("llm_max_tokens")
    if llm_provider:
        os.environ["OMNIPERF_LLM_PROVIDER"] = str(llm_provider)
    if llm_model:
        os.environ["OMNIPERF_LLM_MODEL"] = str(llm_model)
    if llm_temperature is not None:
        os.environ["OMNIPERF_LLM_TEMPERATURE"] = str(llm_temperature)
    if llm_max_tokens is not None:
        os.environ["OMNIPERF_LLM_MAX_TOKENS"] = str(llm_max_tokens)

    logger.info(f"Configuration: repo_path={repo_path}")
    logger.info(f"Test generation settings: extractions_dir={extractions_dir}, use_docker={use_docker}")

    # Scan all extraction files
    extraction_files = scan_extraction_files(Path(extractions_dir))

    if not extraction_files:
        logger.error("No valid extraction files found. Exiting.")
        return

    # Process all commits in batch
    records = process_batch_commits(
        repo_path=repo_path,
        extraction_files=extraction_files,
        use_docker=use_docker,
        docker_image=docker_image,
        extractions_dir=extractions_dir,
        setup_commands=setup_commands,
        install_commands=install_commands,
        api=api,
        notes=notes,
    )

    if not records:
        logger.error("No records were successfully processed. Exiting.")
        return

    logger.info(f"Batch processing completed. Generated {len(records)} records")

    default_repo_nm = str(config.get("dataset_name", "omni_commit_dataset"))
    hf_repo_id = _extract_hf_repo_id(hf_repo, default_repo_nm)
    dataset_name = (hf_repo_id.split("/", 1)[1] if hf_repo_id and "/" in hf_repo_id else default_repo_nm)
    out_dir = Path("data")
    logger.info(f"Saving to dataset: {dataset_name}, output dir: {out_dir}")

    save_and_push(records, out_dir, dataset_name, push_to_hf=bool(push_to_hf and hf_repo), hf_repo_id=hf_repo_id)
    logger.info("save_and_push completed successfully")


if __name__ == "__main__":
    main()
