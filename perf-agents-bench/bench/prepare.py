from __future__ import annotations
import json
import time
import os
import subprocess
import concurrent.futures as futures
import logging
import traceback
import threading
from pathlib import Path
from typing import Dict, Any

from .journal import JournalWriter
from .repo_manager import RepoManager
from .git_utils import get_changed_files, resolve_precommit
from .agents.openhands import OpenHandsAgent

# Import run summary generation (Stage 1: after agent run)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
try:
    from eval.run_summary import generate_summary_from_state, save_summary
except ImportError:
    generate_summary_from_state = None
    save_summary = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PrepareExecutor:
    def __init__(self, bench_cfg: Dict[str, Any], run_id: str):
        self.cfg = bench_cfg
        self.run_id = run_id  # Now supports hierarchical paths like "vllm/trae/gpt-4o/2024-01-15_14-30-00"
        self.state_root = Path(self.cfg["paths"]["state_root"]).resolve()
        self.work_root = Path(self.cfg["paths"]["work_root"]).resolve()
        (self.state_root / "runs" / self.run_id).mkdir(parents=True, exist_ok=True)

        # Extract run metadata from hierarchical path
        run_parts = self.run_id.split("/")
        if len(run_parts) >= 4:
            self.repo_name = run_parts[0]
            self.agent_name = run_parts[1]
            self.model_name = run_parts[2]
            self.run_timestamp = run_parts[3]
        else:
            # Legacy flat structure fallback
            self.repo_name = "unknown"
            self.agent_name = str(self.cfg.get("agents", {}).get("default", "unknown"))
            self.model_name = "default"
            self.run_timestamp = run_id

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
                        original = line
                        line = line.strip()
                        if not line:
                            continue
                        # Allow shell-style export statements
                        if line.startswith("export "):
                            line = line[len("export "):].strip()
                        # Skip comments or invalid lines
                        if line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        # If value is unquoted, strip trailing inline comments
                        if value and value[0] not in ('"', "'") and "#" in value:
                            value = value.split("#", 1)[0].strip()
                        # Strip surrounding single or double quotes if present
                        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                            value = value[1:-1]
                        env[key] = value
                    break
            return env

        env_vars = _load_env_vars()

        # Experiment config (A/B hints + optional preflight)
        def _as_bool(x: Any) -> bool:  # type: ignore[name-defined]
            try:
                return str(x).strip().lower() in {"1", "true", "yes", "on"}
            except Exception:
                return False

        experiment_cfg = self.cfg.get("experiment", {}) if isinstance(self.cfg.get("experiment", {}), dict) else {}
        hints_enabled = _as_bool(experiment_cfg.get("hints_enabled", False))
        preflight_enabled = _as_bool(experiment_cfg.get("preflight_enabled", False))
        metadata_dir = Path(experiment_cfg.get("metadata_json_dir", "")).resolve() if experiment_cfg.get("metadata_json_dir") else None
        generators_dir = Path(experiment_cfg.get("generators_dir", "")).resolve() if experiment_cfg.get("generators_dir") else None

        # Determine agent type early to control prompt construction
        default_agent_early = str(self.cfg["agents"].get("default", "openhands"))
        # Fair evaluation: suppress human optimization details (diff, stats, hints)
        # but keep commit message as a hint for all agents
        suppress_human_data = True
        # Detach worktree from git history to prevent agent from accessing human commit
        detach_from_history = True

        def process(item: Dict[str, Any]):
            item_id = item["item_id"]
            logger.info(f"Starting task processing: {item_id}")
            
            run_dir = self.state_root / "runs" / self.run_id
            jw = JournalWriter(run_dir, item_id)
            if resume and jw.has_success():
                logger.info(f"Skipping {item_id} - task already completed successfully")
                return f"skip:{item_id}"

            human = item["human"]
            pre = item.get("pre") or None
            logger.info(f"Task configuration - Human commit: {human}, Pre commit: {pre or 'will resolve'}")
            
            if not pre:
                logger.info("Resolving pre-commit from parent index")
                pre = resolve_precommit(rm.base_dir, human, None, item.get("pre_parent_index", 1))
                logger.info(f"Resolved pre commit: {pre}")

            wt_dir = rm.create_worktree(pre, item_id, detach_from_history=detach_from_history)

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

            # Load commit metadata for APIs (but NOT the actual diff content)
            commit_data: Dict[str, Any] | None = None
            diff_stat = ""
            # Get diff stat (files modified) - this is okay to show
            try:
                diff_stat = subprocess.check_output(
                    ["git", "diff", "--stat", pre, human],
                    cwd=rm.base_dir,
                    text=True
                ).strip()
            except Exception:
                diff_stat = ""

            # Load commit metadata JSON for API hints (if available)
            candidate_paths: list[Path] = []
            if metadata_dir:
                candidate_paths.append(Path(metadata_dir) / f"{human}.json")
            candidate_paths.append(Path(f"/workspace/OmniPerf-Bench/tmp_single_commit/{human}.json"))
            for cand in candidate_paths:
                try:
                    if cand.exists():
                        with open(cand, "r") as f:
                            commit_data = json.load(f)
                        break
                except Exception:
                    commit_data = None

            # Create a concrete test script that demonstrates what we're optimizing
            # This follows the GSO format more closely
            if target_files and any('moe' in f.lower() for f in target_files):
                test_script_content = """import torch
import time
from vllm.model_executor.layers.fused_moe import moe_align_block_size

# Benchmark the MoE align block size operation
num_tokens = 4096
num_experts = 64
topk = 2
block_size = 128

# Create input data
topk_ids = torch.randint(0, num_experts, (num_tokens * topk,), dtype=torch.int32, device='cuda')

# Time the operation
torch.cuda.synchronize()
start = time.time()

sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
    topk_ids, num_experts, block_size, topk
)

torch.cuda.synchronize()
duration = time.time() - start

print(f"Duration: {duration:.4f} seconds")
"""
            elif target_files and any('prefix' in f.lower() or 'block' in f.lower() for f in target_files):
                test_script_content = """import torch
import time
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator

# Benchmark prefix caching block allocation with common prefixes
block_size = 16
num_blocks = 256
num_sequences = 8
common_prefix_blocks = 4

# Create allocator
allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)

# Common token IDs for shared prefix
common_token_ids = list(range(block_size * common_prefix_blocks))

# Time the allocation and marking operation
start = time.time()

# Allocate blocks for multiple sequences with common prefixes
for seq_idx in range(num_sequences):
    prev_block = None
    for block_idx in range(common_prefix_blocks):
        start_idx = block_idx * block_size
        end_idx = start_idx + block_size
        token_ids = common_token_ids[start_idx:end_idx]
        
        block = allocator.allocate_immutable_block(
            prev_block=prev_block,
            token_ids=token_ids
        )
        prev_block = block

# Mark blocks as computed (this is the optimized operation)
allocator.mark_blocks_as_computed([])

duration = time.time() - start
print(f"Duration: {duration:.4f} seconds")
print(f"Cache hit rate: {allocator.get_prefix_cache_hit_rate():.3f}")
"""
            else:
                # Generic performance test
                test_script_content = """# This is a performance optimization task
# The specific operations to optimize are in the files listed below
# Focus on performance improvements in the target functions
"""
            
            # Determine the path the agent should use when referring to the workspace.
            # For local runtime (bench_test.yaml), use the absolute worktree path.
            # Container mode will still work since paths are advisory for the agent.
            agent_workspace_root = str(wt_dir)
            scratch_rel_dir = ".bench_scratch"
            scratch_abs_dir = str(Path(agent_workspace_root) / scratch_rel_dir)

            # Build the official GSO prompt with the human's diff
            task_lines = [
                f"I've uploaded a python code repository in the directory {agent_workspace_root}.",
                "Consider the following test script showing an example usage of the repository:",
                "",
                "<test_script>",
                test_script_content if test_script_content else "# Test script will be created based on the optimization target",
                "</test_script>",
                "",
                "Can you help me implement the necessary changes to the repository so that the runtime of the <test_script> is optimized?",
                "",
                "Basic guidelines:",
                f"1. Your task is to make changes to non-test files in the {agent_workspace_root} directory to improve the performance of the <test_script>.",
                "2. Make changes while ensuring the repository is functionally equivalent to the original.",
                "3. Do not overoptimize for just the specific inputs in <test_script>. Make general performance improvements for the usage scenario shown.",
                "4. You may need to rebuild the repo for your changes to take effect before testing. Some rebuilds may take time to run, so be patient with running them.",
                "",
                "Follow these steps to improve performance:",
                "1. As a first step, explore the repository structure and understand the target files.",
                f"2. Create a script ONLY inside {scratch_abs_dir} (e.g., {scratch_abs_dir}/test_opt.py) to reproduce and time the example, then execute it with python <filename.py> from the repo root.",
                "3. Profile and identify performance bottlenecks in the target files.",
                "4. Edit the source code of the repository to improve performance.",
                "5. Rebuild and rerun your script to confirm that performance has improved.",
                "",
                "Common optimization strategies to consider:",
                "- Reduce unnecessary memory allocations",
                "- Avoid redundant computations",
                "- Optimize data structures and algorithms",
                "- Improve cache locality",
                "- Reduce synchronization overhead",
                "- Use more efficient library calls where available",
            ]

            # Add target files if available
            if target_files:
                task_lines.append("")
                task_lines.append("Target files to optimize:")
                for f in target_files[:3]:  # Limit to first 3 files
                    task_lines.append(f"- {f}")
                    
            # Add a strong reminder to make changes
            task_lines.extend([
                "",
                "IMPORTANT: You MUST make actual code changes to at least one file.",
                "The task will fail if no files are modified."
            ])
            
            if prompt["constraints"]:
                task_lines.append("")
                task_lines.append("## Constraints")
                task_lines += [f"- {c}" for c in prompt["constraints"]]
            
            if prompt["target_files"]:
                task_lines.append("")
                task_lines.append("## Target Files (ONLY modify these)")
                task_lines += [f"- `{t}`" for t in prompt["target_files"]]

            # Add APIs to target (from commit metadata, if available)
            if commit_data:
                apis = commit_data.get("apis") or []
                if isinstance(apis, list) and apis:
                    task_lines.append("")
                    task_lines.append("## APIs to Consider")
                    task_lines.append("These APIs have been identified as relevant to the optimization:")
                    for api in apis[:10]:
                        task_lines.append(f"- `{api}`")

            # Add files modified statistics (shows scope without revealing exact changes)
            if diff_stat:
                task_lines.append("")
                task_lines.append("## Files Modified (statistics)")
                task_lines.append("The following files were changed in the reference optimization:")
                task_lines.append("```")
                task_lines += diff_stat.splitlines()
                task_lines.append("```")

            task_lines.append("")
            task_lines.append("## TASK COMPLETION COMMAND:")
            task_lines.append("When you have made optimizations:")
            task_lines.append("```bash")
            task_lines.append("git add -A")
            # ensure scratch artifacts are not staged
            task_lines.append(f"git reset -q {scratch_rel_dir} || true")
            task_lines.append("git commit -m 'Optimize MoE align sum kernels performance'")
            task_lines.append(f"git diff $(git merge-base HEAD origin/HEAD || git rev-parse HEAD~1) -- . ':(exclude){scratch_rel_dir}' > {agent_workspace_root}/model_patch.diff || true")
            task_lines.append("finish")
            task_lines.append("```")
            task_lines.append("")
            task_lines.append("START IMPLEMENTING IMMEDIATELY. NO MORE ANALYSIS.")
            
            task_text = "\n".join(task_lines) + "\n"
            task_file = jw.dir / "task.txt"
            task_file.write_text(task_text)

            # Optional preflight logging (lightweight; does not block)
            preflight_info: Dict[str, Any] = {"attempted": bool(preflight_enabled)}
            if preflight_enabled:
                try:
                    # Check generator presence and perf command availability
                    gen_found = False
                    perf_cmd_present = False
                    if commit_data:
                        perf_cmd_present = bool(commit_data.get("perf_command"))
                    if generators_dir and generators_dir.exists():
                        import glob as _glob2
                        gen_matches = _glob2.glob(str(generators_dir / f"*{human[:8]}*test_case_generator.py"))
                        gen_found = bool(gen_matches)
                    preflight_info.update({
                        "generator_found": gen_found,
                        "perf_command_present": perf_cmd_present,
                        "status": "logged_only",
                    })
                except Exception as _e_pf:
                    preflight_info.update({"error": str(_e_pf)})

            # Select agent
            default_agent = str(self.cfg["agents"].get("default", "openhands"))
            agent_label = default_agent.upper()
            branch = f"agent/{task_cfg['id']}/{human[:8]}"
            returncode = -1
            stdout_content = ""
            stderr_content = ""
            dur = 0.0

            if default_agent == "openhands":
                # Run OpenHands locally
                agent_cfg = self.cfg["agents"]["openhands"]
                cli = agent_cfg["cli"]
                time_budget = agent_cfg["time_budget_minutes"]
                container_image = agent_cfg.get("container_image") or None
                args_cfg = agent_cfg.get("args", {})
                # Reduce iterations to force faster action
                iterations = args_cfg.get("iterations", 50)
                max_budget = args_cfg.get("max_budget_per_task", 10.0)
                use_python_api = bool(args_cfg.get("use_python_api", True))
            elif default_agent in {"trae", "codex", "codex_cli", "claude_code"}:
                agent_cfg = self.cfg["agents"].get(default_agent, {})
                cli = agent_cfg.get("cli", "python")
                time_budget = int(agent_cfg.get("time_budget_minutes", 60))
                args_cfg = agent_cfg.get("args", {})
                iterations = int(args_cfg.get("max_steps", 50))
                trae_config_file = agent_cfg.get("config_file") or None
            else:
                raise NotImplementedError(f"Unknown agent default: {default_agent}")

            # Execute and capture logs (OpenHands containerized path)
            if default_agent == "openhands" and (agent_cfg.get("container_image") or None):
                container_image = agent_cfg.get("container_image") or None
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
            elif default_agent == "openhands":
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
                    # Direct Python execution with debugging
                    cmd = [
                        cli, "-m", "openhands.core.main",
                        "--config-file", "config/main_openai.toml",
                        "--log-level", "INFO",
                        "-d", str(wt_dir),
                        "-f", str(task_file),
                        "-i", str(iterations),
                        "-b", str(max_budget)
                    ]

            elif default_agent in {"trae", "codex", "codex_cli", "claude_code"}:
                # Run Trae/Codex (module), Codex CLI, or Claude Code
                if default_agent == "trae":
                    cmd = [
                        cli, "-m", "trae_agent.cli", "run",
                        "--file", str(task_file),
                        "--working-dir", str(wt_dir),
                        "--max-steps", str(iterations),
                        "--must-patch",
                        "--patch-path", str((jw.dir / "model_patch.diff").resolve()),
                        "--trajectory-file", str((jw.dir / "trajectory.json").resolve()),
                    ]
                    if trae_config_file:
                        cmd.extend(["--config-file", str(trae_config_file)])
                elif default_agent == "codex":
                    cmd = [
                        cli, "-m", "codex_agent.cli", "run",
                        "--file", str(task_file),
                        "--working-dir", str(wt_dir),
                        "--max-steps", str(iterations),
                        "--must-patch",
                        "--patch-path", str((jw.dir / "model_patch.diff").resolve()),
                        "--trajectory-file", str((jw.dir / "trajectory.json").resolve()),
                    ]
                    if trae_config_file:
                        cmd.extend(["--config-file", str(trae_config_file)])
                elif default_agent == "codex_cli":
                    # Invoke the locally installed Codex CLI directly using non-interactive exec
                    # Read task prompt content to pass as a single PROMPT argument
                    try:
                        prompt_text = Path(task_file).read_text()
                    except Exception:
                        prompt_text = ""
                    profile = args_cfg.get("profile") or os.environ.get("CODEX_PROFILE")
                    cmd = [
                        cli, "exec",
                        "--cd", str(wt_dir),
                        "--sandbox", "danger-full-access",
                    ]
                    if profile:
                        cmd += ["-p", str(profile)]
                    cmd += [prompt_text]
                else:  # claude_code
                    # Invoke Claude Code CLI in non-interactive print mode
                    # Read task prompt from existing task_file - will be passed via stdin
                    try:
                        claude_prompt_text = Path(task_file).read_text()
                    except Exception:
                        claude_prompt_text = ""

                    model = args_cfg.get("model") or os.environ.get("CLAUDE_MODEL", "sonnet")

                    # Note: prompt passed via stdin (not as argument) to handle long/complex prompts
                    cmd = [
                        cli, "-p",  # Non-interactive print mode
                        "--output-format", "json",  # Structured output
                        "--dangerously-skip-permissions",  # Like codex_cli's --sandbox danger-full-access
                        "--model", str(model),
                        "--disallowedTools", "WebFetch,WebSearch",  # Block web access
                    ]
                logger.info(f"{default_agent} agent command: {' '.join(map(str, cmd))}")
                logger.info(f"{default_agent} config file: {trae_config_file}")

            logger.info(f"Initializing {default_agent} execution")
            logger.info(f"Working directory: {wt_dir}")
            logger.info(f"Task file: {task_file}")
            logger.info(f"Max iterations: {iterations}")
            logger.info(f"Agent branch: {branch}")
            logger.debug(f"OpenHands command: {' '.join(map(str, cmd))}")

            # Pre-create branch to capture agent edits on it
            try:
                logger.info(f"Creating git branch: {branch}")
                subprocess.run(["git", "checkout", "-B", branch], cwd=wt_dir, check=True)
                logger.info("Git branch created successfully")
            except Exception as e:
                logger.warning(f"Git branch creation failed: {e}")

            t0 = time.time()
            # Watcher to record time-to-first-edit (first commit beyond pre)
            first_edit_time_holder: Dict[str, Any] = {"val": None}
            stop_watch = threading.Event()
            def _watch_first_edit():
                while not stop_watch.is_set():
                    try:
                        commits = subprocess.check_output(
                            ["git", "log", "--oneline", f"{pre}..HEAD"],
                            cwd=wt_dir,
                            text=True,
                            timeout=5,
                            stderr=subprocess.DEVNULL
                        ).strip()
                        if commits and first_edit_time_holder.get("val") is None:
                            first_edit_time_holder["val"] = time.time() - t0
                            break
                    except Exception:
                        pass
                    time.sleep(1.0)
            watcher = threading.Thread(target=_watch_first_edit, daemon=True)
            watcher.start()
            logger.info(f"Starting {default_agent} execution at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                # Merge env vars from .env into subprocess environment
                env = os.environ.copy()
                env.update(env_vars)
                # Strictly disallow web/doc search envs reaching the agent
                for k in [
                    "GOOGLE_API_KEY",
                    "SERPAPI_API_KEY",
                    "BING_API_KEY",
                    "TAVILY_API_KEY",
                    "BRAVE_API_KEY",
                    "PERPLEXITY_API_KEY",
                ]:
                    if k in env:
                        env.pop(k, None)
                
                # Log environment variables for debugging
                logger.info(f"Environment variables loaded from .env: {len(env_vars)}")
                for key in env_vars:
                    if 'API_KEY' in key:
                        logger.info(f"  {key}: {env_vars[key][:20]}...{env_vars[key][-4:]}" if env_vars[key] and len(env_vars[key]) > 24 else f"  {key}: {env_vars[key] if env_vars[key] else 'EMPTY'}")
                    else:
                        logger.info(f"  {key}: {env_vars[key]}")
                
                # Check if OPENAI_API_KEY is properly set
                api_key = env.get("OPENAI_API_KEY")
                if api_key:
                    logger.info(f"OPENAI_API_KEY is set in subprocess environment: {api_key[:20]}...{api_key[-4:]}")
                else:
                    logger.warning("OPENAI_API_KEY is NOT set in subprocess environment!")
                
                # Add debugging environment variables
                env["LOG_ALL_EVENTS"] = "true"
                env["OPENHANDS_DEBUG"] = "true"
                env["PYTHONUNBUFFERED"] = "1"
                # Modern sandbox volume mapping (replaces deprecated WORKSPACE_* envs)
                env["SANDBOX_VOLUMES"] = f"{wt_dir}:/workspace:rw"
                # Ensure Trae/Codex agent modules are discoverable when used
                project_root = Path(__file__).resolve().parents[2]
                trae_repo_path = str(project_root / "third-party" / "trae-agent")
                codex_repo_path = str(project_root)
                py_paths = [trae_repo_path, codex_repo_path, env.get("PYTHONPATH", "")]
                env["PYTHONPATH"] = os.pathsep.join([p for p in py_paths if p]).rstrip(os.pathsep)
                # Prefer non-interactive behavior and disable auto-continue loops (best-effort)
                env["OPENHANDS_AUTO_CONTINUE"] = "false"
                env["AUTO_CONTINUE"] = "false"
                # Set max empty responses to prevent infinite loops
                env["MAX_EMPTY_RESPONSES"] = "2"
                # Enable agent startup diagnostics
                env["TRAE_LOG_STARTUP"] = "1"
                # Force agent to take action instead of asking questions
                env["AGENT_MODE"] = "action_oriented"
                # Add more logging to see what's happening
                env["LOG_LEVEL"] = "DEBUG"
                env["OPENHANDS_LOG_LEVEL"] = "DEBUG"
                # Ensure Codex CLI can write state under the workspace (avoids HOME permissions issues)
                if default_agent == "codex_cli":
                    codex_home = Path(__file__).resolve().parents[1] / ".codex_home"
                    codex_home.mkdir(parents=True, exist_ok=True)
                    env["HOME"] = str(codex_home)
                    env["XDG_STATE_HOME"] = str(codex_home / ".xdg" / "state")
                    env["XDG_CACHE_HOME"] = str(codex_home / ".xdg" / "cache")
                if default_agent == "openhands":
                    # Provide an explicit headless user message to avoid empty auto-continue loops
                    try:
                        forced_target = target_files[0] if target_files else ""
                    except Exception:
                        forced_target = ""
                    headless_msg = (
                        "Do not ask for user input. Proceed immediately to implement the next step: "
                        f"1) create {scratch_abs_dir}/test_opt.py to time the critical path (do not create timing scripts outside {scratch_abs_dir}); 2) edit the performance-critical file "
                        f"{forced_target if forced_target else 'one of the target files'} now; 3) run the timing script; 4) commit; then finish."
                    )
                    env["OPENHANDS_HEADLESS_USER_MESSAGE"] = headless_msg
                
                logger.debug("Environment variables configured:")
                debug_vars = ["LOG_ALL_EVENTS", "OPENHANDS_DEBUG", "PYTHONUNBUFFERED", "OPENAI_API_KEY", "SANDBOX_VOLUMES"]
                for var in debug_vars:
                    if var in env:
                        if "API_KEY" in var:
                            logger.debug(f"  {var}: {'*' * 20}...{env[var][-4:] if len(env[var]) > 4 else '****'}")
                        else:
                            logger.debug(f"  {var}: {env[var]}")
                
                if default_agent == "openhands" and not use_python_api:
                    logger.info("Executing OpenHands command with real-time output")
                    # Use Popen for real-time output streaming
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env,
                        bufsize=1,
                        universal_newlines=True
                    )
                    stdout_lines = []
                    stderr_lines = []
                    # Stream output in real-time
                    import select
                    import sys
                    while proc.poll() is None:
                        ready, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.1)
                        for stream in ready:
                            if stream == proc.stdout:
                                line = stream.readline()
                                if line:
                                    line = line.rstrip()
                                    stdout_lines.append(line)
                                    logger.info(f"OpenHands STDOUT: {line}")
                                    sys.stdout.flush()
                            elif stream == proc.stderr:
                                line = stream.readline()
                                if line:
                                    line = line.rstrip()
                                    stderr_lines.append(line)
                                    logger.warning(f"OpenHands STDERR: {line}")
                                    sys.stderr.flush()
                    # Read any remaining output
                    remaining_stdout, remaining_stderr = proc.communicate()
                    if remaining_stdout:
                        for line in remaining_stdout.split('\n'):
                            if line.strip():
                                stdout_lines.append(line.strip())
                                logger.info(f"OpenHands STDOUT: {line.strip()}")
                    if remaining_stderr:
                        for line in remaining_stderr.split('\n'):
                            if line.strip():
                                stderr_lines.append(line.strip())
                                logger.warning(f"OpenHands STDERR: {line.strip()}")
                    dur = time.time() - t0
                    logger.info(f"OpenHands execution completed in {dur:.1f} seconds")
                    logger.info(f"Process return code: {proc.returncode}")
                    logger.info(f"Total stdout lines: {len(stdout_lines)}")
                    logger.info(f"Total stderr lines: {len(stderr_lines)}")
                    stdout_content = '\n'.join(stdout_lines)
                    stderr_content = '\n'.join(stderr_lines)
                    returncode = proc.returncode
                elif default_agent == "openhands":
                    # Use Python API to pass custom fake_user_response_fn and conversation instructions
                    from openhands.core.config import parse_arguments as _parse_arguments, setup_config_from_args as _setup_cfg
                    import asyncio as _asyncio
                    from openhands.core.main import run_controller as _run_controller
                    from openhands.events.action import MessageAction as _MessageAction
                    import sys as _sys
                    _argv_backup = list(_sys.argv)
                    try:
                        # ensure OpenHands loads our OpenAI config TOML
                        _sys.argv = [_argv_backup[0], "--config-file", "config/main_openai.toml"]
                        _args = _parse_arguments()
                    finally:
                        _sys.argv = _argv_backup
                    _config = _setup_cfg(_args)
                    # mutate essential fields
                    _config.workspace_base = str(wt_dir)
                    _config.max_iterations = int(iterations)
                    _config.max_budget_per_task = float(max_budget)
                    _config.runtime = "local"
                    try:
                        # set sandbox volumes for local runtime
                        _config.sandbox.volumes = [f"{wt_dir}:/workspace:rw"]
                    except Exception:
                        pass
                    # Try to set a different agent if CodeActAgent is too passive
                    try:
                        # Set agent class - could try other agents like readonly_agent, loc_agent
                        _config.agent.class_name = "codeact_agent/CodeActAgent"  # Default
                        # Optionally try setting more aggressive settings
                        _config.agent.memory_max_threads = 1
                    except Exception:
                        pass
                    # Ensure LLM provider/model use OpenAI if configured via our TOML or env
                    try:
                        import tomllib as _tomllib
                    except Exception:
                        _tomllib = None
                    model_from_env = os.environ.get("LLM_MODEL")
                    api_key_from_env = os.environ.get("OPENAI_API_KEY")
                    model_from_toml = None
                    if _tomllib:
                        cfg_path = Path("config/main_openai.toml")
                        if cfg_path.exists():
                            try:
                                _data = _tomllib.loads(cfg_path.read_text())
                                model_from_toml = (_data.get("llm") or {}).get("model")
                                api_key_from_toml = (_data.get("llm") or {}).get("api_key")
                            except Exception:
                                api_key_from_toml = None
                        else:
                            api_key_from_toml = None
                    else:
                        api_key_from_toml = None
                    try:
                        _config.llms.llm.custom_llm_provider = "openai"
                        if model_from_env or model_from_toml:
                            _config.llms.llm.model = (model_from_env or model_from_toml)
                        if api_key_from_env or api_key_from_toml:
                            _config.llms.llm.api_key = (api_key_from_env or api_key_from_toml)
                            # also export to env for downstream libraries
                            os.environ["OPENAI_API_KEY"] = (api_key_from_env or api_key_from_toml)
                        # prefer not dropping params for OpenAI compatibility
                        try:
                            _config.llms.llm.drop_params = False
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # Dynamic response generator that forces concrete actions
                    def _fake_user_response_fn(state, encapsulate_solution: bool = False, try_parse=None) -> str:
                        # Direct response that encourages action without being too specific
                        return "Continue implementing the optimization changes you think are appropriate. Make the edits you believe will improve performance."
                    
                    # General conversation instructions
                    CONVERSATION_INSTR = (
                        "You are working in headless mode. Complete the task as described in the initial prompt.\n"
                        "Focus on making the requested code optimizations."
                    )
                    logger.info("Executing OpenHands via Python API with custom headless directive")
                    task_content = task_file.read_text()
                    logger.info(f"Task content: {task_content}")
                    logger.info(f"Using fake_user_response: 'Continue implementing the optimization changes...'")
                    logger.info(f"Conversation instruction: {CONVERSATION_INSTR}")
                    _state = _asyncio.run(_run_controller(
                        config=_config,
                        initial_user_action=_MessageAction(content=task_content),
                        runtime=None,
                        exit_on_message=False,
                        fake_user_response_fn=_fake_user_response_fn,
                        headless_mode=True,
                        conversation_instructions=CONVERSATION_INSTR,
                    ))
                    logger.info(f"Final agent state: {_state.agent_state}")
                    logger.info(f"Total history events: {len(_state.history) if hasattr(_state, 'history') else 'N/A'}")
                    dur = time.time() - t0
                    logger.info(f"OpenHands execution completed in {dur:.1f} seconds (Python API)")
                    stdout_content = ""
                    stderr_content = ""
                    returncode = 0
                else:
                    # Execute Trae/Codex Agent with real-time logging
                    logger.info(f"Executing {agent_label} subprocess with timeout: {time_budget * 60}s")
                    logger.debug(f"Working directory: {wt_dir}")
                    logger.debug(f"Environment OPENAI_API_KEY present: {bool(env.get('OPENAI_API_KEY'))}")
                    logger.debug(f"Environment PYTHONPATH: {env.get('PYTHONPATH', 'NOT_SET')}")
                    
                    # Point agent step logs to run directory file
                    step_log_path = str((jw.dir / "trae_steps.log").resolve())
                    env["TRAE_STEP_LOG_FILE"] = step_log_path
                    if default_agent == "codex":
                        env["CODEX_STEP_LOG_FILE"] = step_log_path
                    
                    # Use Popen for real-time output streaming (same as OpenHands)
                    # For claude_code, pass prompt via stdin to handle long/complex prompts
                    stdin_pipe = subprocess.PIPE if default_agent == "claude_code" else None
                    proc = subprocess.Popen(
                        cmd,
                        stdin=stdin_pipe,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env,
                        cwd=wt_dir,
                        bufsize=1,
                        universal_newlines=True
                    )
                    # Write prompt to stdin for claude_code and close it
                    if default_agent == "claude_code" and proc.stdin:
                        proc.stdin.write(claude_prompt_text)
                        proc.stdin.close()
                    
                    stdout_lines = []
                    stderr_lines = []
                    
                    # Stream output in real-time
                    import select
                    import sys
                    timeout_seconds = time_budget * 60
                    start_time = time.time()
                    
                    while proc.poll() is None:
                        # Check for timeout
                        if time.time() - start_time > timeout_seconds:
                            logger.warning(f"{agent_label} agent timeout after {timeout_seconds}s, terminating process")
                            proc.terminate()
                            proc.wait(timeout=5)
                            break
                            
                        ready, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.1)
                        for stream in ready:
                            if stream == proc.stdout:
                                line = stream.readline()
                                if line:
                                    line = line.rstrip()
                                    stdout_lines.append(line)
                                    logger.info(f"{agent_label} STDOUT: {line}")
                                    sys.stdout.flush()
                            elif stream == proc.stderr:
                                line = stream.readline()
                                if line:
                                    line = line.rstrip()
                                    stderr_lines.append(line)
                                    logger.warning(f"{agent_label} STDERR: {line}")
                                    sys.stderr.flush()

                    # Read any remaining output
                    # For claude_code, stdin is already closed, so we use wait() + read()
                    if default_agent == "claude_code":
                        proc.wait()
                        remaining_stdout = proc.stdout.read() if proc.stdout else ""
                        remaining_stderr = proc.stderr.read() if proc.stderr else ""
                    else:
                        remaining_stdout, remaining_stderr = proc.communicate()
                    if remaining_stdout:
                        for line in remaining_stdout.split('\n'):
                            if line.strip():
                                stdout_lines.append(line.strip())
                                logger.info(f"{agent_label} STDOUT: {line.strip()}")
                    if remaining_stderr:
                        for line in remaining_stderr.split('\n'):
                            if line.strip():
                                stderr_lines.append(line.strip())
                                logger.warning(f"{agent_label} STDERR: {line.strip()}")
                    
                    stdout_content = '\n'.join(stdout_lines)
                    stderr_content = '\n'.join(stderr_lines)
                    returncode = proc.returncode
                    
                    dur = time.time() - t0
                    logger.info(f"{agent_label} agent execution completed in {dur:.1f} seconds")
                    logger.info(f"Process return code: {returncode}")
                    logger.info(f"Total stdout lines: {len(stdout_lines)}")
                    logger.info(f"Total stderr lines: {len(stderr_lines)}")
                    
                    # Save explicit agent logs for review
                    try:
                        if default_agent == "trae":
                            jw.write_trae_logs(stdout_content, stderr_content)
                        elif default_agent == "codex":
                            jw.write_codex_logs(stdout_content, stderr_content)
                        elif default_agent == "claude_code":
                            jw.write_claude_code_logs(stdout_content, stderr_content)
                        else:
                            jw.write_openhands_logs(stdout_content, stderr_content)
                    except Exception:
                        logger.warning("Failed to write agent logs")

                    # Check if agent exceeded 120 steps and should be skipped
                    exceeded_step_limit = False
                    if default_agent in {"trae", "codex", "claude_code"}:
                        trajectory_path = jw.dir / "trajectory.json"
                        if trajectory_path.exists():
                            try:
                                trajectory_data = json.loads(trajectory_path.read_text())
                                # Count steps in trajectory
                                steps = trajectory_data.get("steps", [])
                                step_count = len(steps)

                                # Check for exceeded max steps message
                                final_result = trajectory_data.get("final_result", "")
                                if step_count > 120 or "exceeded maximum steps" in final_result.lower():
                                    logger.warning(f"{agent_label} agent exceeded 120 step limit (actual: {step_count} steps)")
                                    logger.warning("Marking as complete to allow pipeline to proceed to next commit")
                                    exceeded_step_limit = True
                            except Exception as e:
                                logger.debug(f"Could not check trajectory steps: {e}")

                    # Determine success based on task completion, not just return code
                    # Agents may have internal API errors but still complete the task successfully
                    task_completed = False
                    if returncode == 0:
                        task_completed = True
                    elif exceeded_step_limit:
                        # Agent exceeded step limit - treat as completed so pipeline can continue
                        task_completed = True
                        returncode = 0  # Override return code
                        logger.info(f"{agent_label} agent exceeded step limit but treating as complete to continue pipeline")
                    else:
                        # Check if agent made commits despite API errors
                        try:
                            commits = subprocess.check_output([
                                "git", "log", "--oneline", f"{pre}..HEAD"
                            ], cwd=wt_dir, text=True).strip()
                            if commits:
                                logger.info(f"{agent_label} agent made commits despite API errors: {len(commits.splitlines())} commits")
                                task_completed = True
                        except Exception:
                            pass

                    if not task_completed and not exceeded_step_limit:
                        logger.error(f"{agent_label} agent failed with return code {returncode}")
                        logger.error(f"Stdout: {stdout_content}")
                        logger.error(f"Stderr: {stderr_content}")
                
                # Determine status based on actual task completion, not just return code
                if exceeded_step_limit:
                    # Special status for exceeded step limit - mark as error to skip in analysis
                    # but still create journal so pipeline continues
                    status = "max_steps_exceeded"
                    logger.info(f"Task status: {status} (exceeded 120 step limit)")
                elif default_agent in {"trae", "codex", "claude_code"}:
                    # For Trae/Codex/Claude Code, check if commits were made or files changed
                    status = "success" if task_completed else "error"
                else:
                    # For OpenHands, use return code
                    status = "success" if returncode == 0 else "error"
                logger.info(f"Task status determined as: {status}")

                # Skip file change analysis for commits that exceeded step limit
                # to avoid git operations that may fail in detached worktrees
                if exceeded_step_limit:
                    logger.info("Skipping file change analysis for commit that exceeded step limit")
                    changed: list[str] = []
                    targets = set(target_files)
                    disallowed: list[str] = []
                    ok = True
                else:
                    # Enforce targets
                    logger.info("Analyzing file changes made by agent")
                    changed: list[str] = []
                    if default_agent == "openhands":
                        changed = get_changed_files(wt_dir, pre, "HEAD")
                    else:
                        # For Trae/Codex, get changed files from git commit in worktree
                        try:
                            # Get files changed in the latest commit made by agent
                            changed = subprocess.check_output([
                                "git", "diff", "--name-only", pre, "HEAD"
                            ], cwd=wt_dir, text=True, timeout=30).strip().splitlines()
                            changed = [f for f in changed if f.strip()]  # Filter empty lines
                            logger.debug(f"Git diff detected {len(changed)} changed files")
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Git diff timed out after 30s, using patch file fallback")
                            changed = []
                        except Exception as e:
                            logger.warning(f"Failed to get changed files from git: {e}")
                            # Fallback: try to derive from patch file in worktree
                            patch_path_wt = wt_dir / "model_patch.diff"
                            patch_path_run = jw.dir / "model_patch.diff"

                            # First check worktree patch
                            if patch_path_wt.exists():
                                try:
                                    for line in patch_path_wt.read_text().splitlines():
                                        if line.startswith("diff --git a/"):
                                            parts = line.split()
                                            if len(parts) >= 4:
                                                b_path = parts[3]
                                                if b_path.startswith("b/"):
                                                    rel = b_path[2:]
                                                    if rel and rel not in changed:
                                                        changed.append(rel)
                                    logger.debug(f"Worktree patch detected {len(changed)} changed files")
                                except Exception as e2:
                                    logger.warning(f"Failed to parse worktree patch: {e2}")

                            # Then check run directory patch as fallback
                            if not changed and patch_path_run.exists():
                                try:
                                    for line in patch_path_run.read_text().splitlines():
                                        if line.startswith("diff --git a/"):
                                            parts = line.split()
                                            if len(parts) >= 4:
                                                b_path = parts[3]
                                                if b_path.startswith("b/"):
                                                    rel = b_path[2:]
                                                    if rel and rel not in changed:
                                                        changed.append(rel)
                                    logger.debug(f"Run directory patch detected {len(changed)} changed files")
                                except Exception as e3:
                                    logger.warning(f"Failed to parse run directory patch: {e3}")
                    # Exclude hidden scratch directory changes from enforcement
                    changed = [p for p in changed if not (p.startswith(f"{scratch_rel_dir}/") or p == scratch_rel_dir)]
                    targets = set(target_files)
                    disallowed = [p for p in changed if p not in targets]
                    ok = len(disallowed) == 0
                
                logger.info(f"Files changed by agent: {len(changed)}")
                for file in changed:
                    logger.info(f"  Changed file: {file}")
                logger.info(f"Target files allowed: {len(targets)}")
                for file in targets:
                    logger.debug(f"  Allowed target: {file}")
                
                if disallowed:
                    logger.warning(f"Disallowed file changes detected: {len(disallowed)}")
                    for file in disallowed:
                        logger.warning(f"  Disallowed change: {file}")
                else:
                    logger.info("All file changes are within allowed targets")
                
                logger.info(f"Target enforcement status: {'PASS' if ok else 'FAIL'}")
                
                jw.write_diff_targets({"changed": changed, "allowed": list(targets), "disallowed": disallowed, "ok": ok})
                if len(changed) == 0:
                    logger.warning("No file changes detected.")
                    # Skip commit check for exceeded step limit to avoid hanging git operations
                    if exceeded_step_limit:
                        logger.info("Skipping commit check for exceeded step limit commit")
                    # For Trae/Codex/Claude Code agent, check if this is due to detection bug vs actual no changes
                    elif default_agent in {"trae", "codex", "claude_code"}:
                        # Check if there are any commits made by agent
                        try:
                            commits = subprocess.check_output([
                                "git", "log", "--oneline", f"{pre}..HEAD"
                            ], cwd=wt_dir, text=True, timeout=10).strip()
                            if commits:
                                logger.warning(f"Agent made {len(commits.splitlines())} commits but file detection failed. This may be a detection bug.")
                                # Don't mark as error if commits exist - likely a detection issue
                            else:
                                logger.warning("No commits found. Agent likely made no changes. Marking as error to avoid analysis loops.")
                                status = "error"
                        except subprocess.TimeoutExpired:
                            logger.warning("Git log timed out checking commits. Skipping check.")
                        except Exception:
                            logger.warning("Could not check git commits. Marking task as error to avoid analysis loops.")
                            status = "error"
                    else:
                        # For OpenHands, mark as error if no changes
                        logger.warning("Marking task as error to avoid analysis loops.")
                        status = "error"
                if task_cfg["optimization_contract"].get("strict_targets", False) and not ok:
                    logger.warning("Status changed to error due to strict target enforcement")
                    status = "error"
                # Stop watcher
                try:
                    stop_watch.set()
                    watcher.join(timeout=1.0)
                except Exception:
                    pass

                # Generate unified diff prediction artifact for GSO harness compatibility
                try:
                    import tomllib as _tomllib  # Python 3.11+
                except Exception:
                    _tomllib = None
                if default_agent == "openhands":
                    try:
                        # Compute unified diff against base (pre) commit, excluding scratch artifacts
                        diff_text = subprocess.check_output([
                            "git", "diff", f"{pre}", "HEAD", "--", ".", f":(exclude){scratch_rel_dir}"
                        ], cwd=wt_dir).decode()
                    except Exception:
                        diff_text = ""
                else:
                    # Use Trae/Codex patch from worktree (primary) or run directory (fallback)
                    patch_path_wt = wt_dir / "model_patch.diff"
                    patch_path_run = jw.dir / "model_patch.diff"
                    
                    diff_text = ""
                    if patch_path_wt.exists():
                        diff_text = patch_path_wt.read_text()
                        # Copy patch to run directory for artifact preservation
                        try:
                            patch_path_run.write_text(diff_text)
                            logger.debug("Copied patch file from worktree to run directory")
                        except Exception as e:
                            logger.warning(f"Failed to copy patch file: {e}")
                    elif patch_path_run.exists():
                        diff_text = patch_path_run.read_text()
                        logger.debug("Using patch file from run directory")
                    else:
                        logger.warning("No patch file found in worktree or run directory")
                # Derive instance_id from repo URL and base commit
                repo_url_val = plan.get("repo", repo_url)
                try:
                    owner_repo = repo_url_val.split("github.com/")[-1].rstrip(".git")
                except Exception:
                    owner_repo = "unknown/unknown"
                try:
                    owner, repo_nm = owner_repo.split("/")
                except Exception:
                    owner, repo_nm = owner_repo, "repo"
                instance_id = f"{owner}__{repo_nm}-{pre[:7]}"
                # Try to read model name from config
                model_name = None
                cfg_path = Path("config/main_openai.toml")
                if _tomllib and cfg_path.exists():
                    try:
                        data = _tomllib.loads(cfg_path.read_text())
                        model_name = (data.get("llm") or {}).get("model")
                    except Exception:
                        model_name = None
                model_name = model_name or os.environ.get("LLM_MODEL") or "openhands"
                prediction = {
                    "instance_id": instance_id,
                    "model_patch": diff_text,
                    "model_name_or_path": str(model_name),
                }
                pred_dir = jw.dir
                (pred_dir).mkdir(parents=True, exist_ok=True)
                # Write JSONL prediction and diff artifact
                try:
                    (pred_dir / "prediction.jsonl").write_text(json.dumps(prediction) + "\n")
                    (pred_dir / "model_patch.diff").write_text(diff_text)
                    logger.info("Wrote prediction.jsonl and model_patch.diff artifacts")
                except Exception as _e:
                    logger.warning(f"Failed to write prediction artifacts: {_e}")
                logger.info(f"Writing journal with final status: {status}")
                # Compute metrics (skip git operations if exceeded step limit to avoid hangs)
                if exceeded_step_limit:
                    logger.info("Skipping git metrics computation for exceeded step limit commit")
                    commit_count = None
                    patch_size = None
                else:
                    try:
                        # commit count
                        _commits_txt = subprocess.check_output(
                            ["git", "log", "--oneline", f"{pre}..HEAD"],
                            cwd=wt_dir,
                            text=True,
                            timeout=10
                        ).strip()
                        commit_count = len([ln for ln in _commits_txt.splitlines() if ln.strip()])
                    except Exception:
                        commit_count = None
                    # patch size (added+removed lines)
                    def _patch_size_loc(txt: str) -> int:
                        add = sum(1 for l in txt.splitlines() if l.startswith("+") and not l.startswith("+++"))
                        rem = sum(1 for l in txt.splitlines() if l.startswith("-") and not l.startswith("---"))
                        return add + rem
                    diff_for_metrics = diff_text
                    if not diff_for_metrics:
                        try:
                            diff_for_metrics = subprocess.check_output(
                                ["git", "diff", pre, "HEAD", "--", ".", f":(exclude){scratch_rel_dir}"],
                                cwd=wt_dir,
                                timeout=10
                            ).decode()
                        except Exception:
                            diff_for_metrics = ""
                    patch_size = _patch_size_loc(diff_for_metrics) if diff_for_metrics else None

                metrics_payload = {
                    "time_to_first_edit_s": first_edit_time_holder.get("val"),
                    "commit_count": commit_count,
                    "patch_size_loc": patch_size,
                    "changed_files_count": len(changed),
                    "violations_count": len(disallowed),
                }

                jw.write_journal({
                    "task_id": task_cfg["id"],
                    "commits": {"pre": pre, "human": human},
                    "agent_branch": branch,
                    "status": status,
                    "run_metadata": {
                        "repo": self.repo_name,
                        "agent": self.agent_name,
                        "model": self.model_name,
                        "run_timestamp": self.run_timestamp,
                        "run_path": self.run_id,
                    },
                    "experiment": {
                        "hints_enabled": hints_enabled,
                        "preflight_enabled": preflight_enabled,
                    },
                    "preflight": preflight_info,
                    "metrics": metrics_payload,
                    default_agent: {
                        "cli": cli,
                        "time_budget_minutes": time_budget,
                        "returncode": returncode,
                        "duration_s": dur,
                    },
                })

                # Generate run_summary.json (Stage 1: after agent run)
                if generate_summary_from_state and save_summary:
                    try:
                        summary = generate_summary_from_state(
                            item_dir=jw.dir,
                            repo=self.repo_name,
                            agent=self.agent_name,
                            model_hint=self.model_name,
                            timestamp=self.run_timestamp,
                        )
                        if summary:
                            save_summary(summary, jw.dir / "run_summary.json")
                            logger.info(f"Generated run_summary.json for {item_id}")
                    except Exception as summary_err:
                        logger.warning(f"Failed to generate run_summary.json: {summary_err}")

                logger.info(f"Task completed successfully: {status}:{item_id}")
                return f"{status}:{item_id}"
            except Exception as e:
                logger.error(f"Exception during OpenHands execution:")
                logger.error(f"  Error type: {type(e).__name__}")
                logger.error(f"  Error message: {str(e)}")
                logger.error("  Full traceback:")
                for line in traceback.format_exc().split('\n'):
                    if line.strip():
                        logger.error(f"  {line}")
                
                jw.write_journal({
                    "task_id": task_cfg["id"],
                    "commits": {"pre": pre, "human": human},
                    "agent_branch": branch,
                    "status": "error",
                    "run_metadata": {
                        "repo": self.repo_name,
                        "agent": self.agent_name,
                        "model": self.model_name,
                        "run_timestamp": self.run_timestamp,
                        "run_path": self.run_id,
                    },
                    "error": str(e),
                    "error_type": type(e).__name__,
                })

                # Generate run_summary.json even for error cases (Stage 1)
                if generate_summary_from_state and save_summary:
                    try:
                        summary = generate_summary_from_state(
                            item_dir=jw.dir,
                            repo=self.repo_name,
                            agent=self.agent_name,
                            model_hint=self.model_name,
                            timestamp=self.run_timestamp,
                        )
                        if summary:
                            save_summary(summary, jw.dir / "run_summary.json")
                    except Exception:
                        pass  # Don't fail on summary generation errors

                logger.error(f"Task failed with exception: error:{item_id}")
                return f"error:{item_id}"

        with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(process, items))
