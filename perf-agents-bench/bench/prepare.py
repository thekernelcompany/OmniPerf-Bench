from __future__ import annotations
import json
import time
import os
import subprocess
import concurrent.futures as futures
from pathlib import Path
from typing import Dict, Any

from .journal import JournalWriter
from .repo_manager import RepoManager
from .git_utils import get_changed_files, resolve_precommit
from .agents.openhands import OpenHandsAgent


class PrepareExecutor:
    def __init__(self, bench_cfg: Dict[str, Any], run_id: str):
        self.cfg = bench_cfg
        self.run_id = run_id
        self.state_root = Path(self.cfg["paths"]["state_root"]).resolve()
        self.work_root = Path(self.cfg["paths"]["work_root"]).resolve()
        (self.state_root / "runs" / self.run_id).mkdir(parents=True, exist_ok=True)

    def execute(self, task_cfg: Dict[str, Any], plan_path: Path, max_workers: int = 4, resume: bool = True):
        plan = json.loads(Path(plan_path).read_text()) if plan_path.suffix == ".json" else None
        if plan is None:
            # support YAML plans too
            import yaml
            plan = yaml.safe_load(Path(plan_path).read_text())

        items = plan["items"]
        repo_url = plan["repo"]
        repo_name = task_cfg["id"]

        rm = RepoManager(self.work_root, repo_url, repo_name)
        rm.ensure_base()

        def _load_env_vars() -> Dict[str, str]:
            env: Dict[str, str] = {}
            # Prefer project-level .env at perf-agents-bench/.env; fallback to repo root .env
            candidates = [
                Path(__file__).resolve().parents[2] / ".env",
                Path.cwd() / ".env",
            ]
            for p in candidates:
                if p.exists():
                    for line in p.read_text().splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        env[k.strip()] = v.strip()
                    break
            return env

        env_vars = _load_env_vars()

        def process(item: Dict[str, Any]):
            item_id = item["item_id"]
            run_dir = self.state_root / "runs" / self.run_id
            jw = JournalWriter(run_dir, item_id)
            if resume and jw.has_success():
                return f"skip:{item_id}"

            human = item["human"]
            pre = item.get("pre") or None
            if not pre:
                # resolve pre from parent index
                pre = resolve_precommit(rm.base_dir, human, None, item.get("pre_parent_index", 1))

            wt_dir = rm.create_worktree(pre, item_id)

            # Determine target files: if none provided, derive from pre..human diff
            provided_targets = task_cfg["optimization_contract"].get("target_files", [])
            derived_targets = []
            if not provided_targets:
                try:
                    # Use the base repo (not the worktree) to diff pre..human
                    # This captures the exact change surface of the human commit
                    derived_targets = get_changed_files(rm.base_dir, pre, human)
                except Exception:
                    derived_targets = []
            target_files = provided_targets or derived_targets

            # Build OpenHands prompt
            prompt = {
                "task": task_cfg["name"],
                "description": task_cfg.get("description", ""),
                "constraints": task_cfg["optimization_contract"].get("constraints", []),
                "target_files": target_files,
                "success": {
                    "primary_metric": task_cfg["scoring"]["primary"],
                    "rules": [
                        "Do not modify tests or metrics harness",
                        "Preserve external behavior; optimize internals only",
                    ],
                },
                "commits": {"pre": pre, "human": human},
            }
            jw.write_prompt(prompt)

            # Create a headless task file (text) from the prompt for -f usage
            # Add richer context to guide the agent
            try:
                import subprocess as _sp
                commit_msg = _sp.check_output(["git", "show", "--no-patch", "--pretty=%B", human], cwd=rm.base_dir).decode().strip()
            except Exception:
                commit_msg = ""
            try:
                diff_stat = _sp.check_output(["git", "diff", "--stat", pre, human], cwd=rm.base_dir).decode().strip()
            except Exception:
                diff_stat = ""

            # Enhanced task file with better structure for OpenHands
            task_lines = [
                "# Performance Optimization Task",
                "",
                f"## Task: {prompt['task']}",
                f"## Description: {prompt['description']}",
                "",
                "## Objective",
                "Your goal is to optimize the performance of the target files while preserving their functionality.",
                "The human developer has already made optimizations (see reference commit below).",
                "Your task is to independently arrive at similar or better optimizations.",
                "",
                "## Instructions",
                "1. Analyze the target files to identify performance bottlenecks",
                "2. Apply concrete optimizations such as:",
                "   - Reducing memory allocations and unnecessary copies",
                "   - Caching frequently computed results",
                "   - Optimizing loops and data structures",
                "   - Eliminating redundant operations",
                "   - Using more efficient algorithms or libraries",
                "3. Ensure all changes preserve the public API and existing behavior",
                "4. Test that the code still runs correctly after modifications",
                "5. Commit your changes with a descriptive message",
                "",
                "## Important Guidelines",
                "- Focus ONLY on the target files listed below",
                "- Do NOT modify test files or benchmark harnesses",
                "- Ensure backward compatibility",
                "- Prioritize safety and correctness over aggressive optimizations",
            ]
            
            if prompt["constraints"]:
                task_lines.append("")
                task_lines.append("## Constraints")
                task_lines += [f"- {c}" for c in prompt["constraints"]]
            
            if prompt["target_files"]:
                task_lines.append("")
                task_lines.append("## Target Files (ONLY modify these)")
                task_lines += [f"- `{t}`" for t in prompt["target_files"]]
            
            # Add reference information about the human's optimization
            if commit_msg or diff_stat:
                task_lines.append("")
                task_lines.append("## Reference Information")
                task_lines.append("The following shows what the human developer optimized:")
                
            if commit_msg:
                task_lines.append("")
                task_lines.append("### Human's Commit Message:")
                task_lines.append("```")
                task_lines += commit_msg.splitlines()
                task_lines.append("```")
                
            if diff_stat:
                task_lines.append("")
                task_lines.append("### Files Changed (statistics):")
                task_lines.append("```")
                task_lines += diff_stat.splitlines()
                task_lines.append("```")
            
            task_lines.append("")
            task_lines.append("## Success Criteria")
            task_lines.append(f"- Primary metric to optimize: {prompt['success']['primary_metric']}")
            task_lines.append("- All existing tests must pass")
            task_lines.append("- No regression in functionality")
            task_lines.append("")
            task_lines.append("## Final Steps")
            task_lines.append("After implementing your optimizations:")
            task_lines.append("1. Run any existing tests to verify correctness")
            task_lines.append("2. Commit your changes with: `git add -A && git commit -m 'Optimize performance in target files'`")
            task_lines.append("3. The task will be complete when you've successfully committed your changes")
            
            task_text = "\n".join(task_lines) + "\n"
            task_file = jw.dir / "task.txt"
            task_file.write_text(task_text)

            # Run OpenHands locally
            agent_cfg = self.cfg["agents"]["openhands"]
            cli = agent_cfg["cli"]
            time_budget = agent_cfg["time_budget_minutes"]
            container_image = agent_cfg.get("container_image") or None
            args_cfg = agent_cfg.get("args", {})
            iterations = args_cfg.get("iterations", 30)
            max_budget = args_cfg.get("max_budget_per_task", 10.0)
            branch = f"agent/{task_cfg['id']}/{human[:8]}"

            # Execute and capture logs (containerized if container_image provided)
            if container_image:
                # Use proper OpenHands headless mode with Docker
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{wt_dir}:/workspace:rw",
                    "-v", f"{task_file}:/task.txt:ro",
                    # Security: set user ID to match host user
                    "-e", f"SANDBOX_USER_ID={os.getuid()}",
                    # Enable full event logging for debugging
                    "-e", "LOG_ALL_EVENTS=true",
                    # Set the runtime container image
                    "-e", f"SANDBOX_RUNTIME_CONTAINER_IMAGE={agent_cfg.get('runtime_image', 'docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik')}",
                    # Ensure Linux containers can resolve host.docker.internal (host-gateway)
                    "--add-host=host.docker.internal:host-gateway",
                    # Allow containerized OpenHands to access host Docker daemon for nested containers
                    "-v", "/var/run/docker.sock:/var/run/docker.sock",
                    "-v", f"{Path.home()}/.openhands:/.openhands",
                    "-w", "/workspace",
                ]
                # Propagate key env vars into container for headless
                for k in ["LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "GITHUB_TOKEN", "GITLAB_TOKEN", "BITBUCKET_TOKEN"]:
                    if env_vars.get(k):
                        cmd += ["-e", f"{k}={env_vars[k]}"]
                # Add timeout as environment variable
                cmd += ["-e", f"OPENHANDS_TIMEOUT_MINUTES={time_budget}"]
                cmd += [
                    container_image,
                    "python", "-m", "openhands.core.main",
                    "-d", "/workspace",
                    "-f", "/task.txt",
                    "-i", str(iterations),
                    "-b", str(max_budget),
                ]
            else:
                # Run with proper headless mode arguments (not uvx)
                # Assume 'cli' is the path to python with OpenHands installed
                if cli == "uvx":
                    # If using uvx, construct proper command
                    cmd = [
                        "uvx", "--python", "3.12", "--from", "openhands-ai",
                        "python", "-m", "openhands.core.main",
                        "-d", str(wt_dir),
                        "-f", str(task_file),
                        "-i", str(iterations),
                        "-b", str(max_budget),
                    ]
                else:
                    # Direct Python execution
                    cmd = [
                        cli, "-m", "openhands.core.main",
                        "-d", str(wt_dir),
                        "-f", str(task_file),
                        "-i", str(iterations),
                        "-b", str(max_budget),
                    ]

            # Pre-create branch to capture agent edits on it
            try:
                subprocess.run(["git", "checkout", "-B", branch], cwd=wt_dir, check=True)
            except Exception:
                pass
            t0 = time.time()
            try:
                # Merge env vars from .env into subprocess environment
                env = os.environ.copy()
                env.update(env_vars)
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                dur = time.time() - t0
                jw.write_openhands_logs(proc.stdout or "", proc.stderr or "")
                status = "success" if proc.returncode == 0 else "error"
                # Enforce targets
                changed = get_changed_files(wt_dir, pre, "HEAD")
                targets = set(target_files)
                disallowed = [p for p in changed if p not in targets]
                ok = len(disallowed) == 0
                jw.write_diff_targets({"changed": changed, "allowed": list(targets), "disallowed": disallowed, "ok": ok})
                if task_cfg["optimization_contract"].get("strict_targets", False) and not ok:
                    status = "error"
                jw.write_journal({
                    "task_id": task_cfg["id"],
                    "commits": {"pre": pre, "human": human},
                    "agent_branch": branch,
                    "status": status,
                    "openhands": {
                        "cli": cli,
                        "container_image": container_image,
                        "time_budget_minutes": time_budget,
                        "returncode": proc.returncode,
                        "duration_s": dur,
                    },
                })
                return f"{status}:{item_id}"
            except Exception as e:
                jw.write_journal({
                    "task_id": task_cfg["id"],
                    "commits": {"pre": pre, "human": human},
                    "agent_branch": branch,
                    "status": "error",
                    "error": str(e),
                })
                return f"error:{item_id}"

        with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(process, items))

