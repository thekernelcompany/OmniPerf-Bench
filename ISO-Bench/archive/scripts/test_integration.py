#!/usr/bin/env python3
"""
OpenHands Integration Test Script

This script provides safe testing options for the OpenHands integration.
Run this to test the system without incurring API costs.

The script automatically loads environment variables from ../.env file using python-dotenv.

Dependencies:
  pip install python-dotenv

Safe testing (no cost):
  python test_integration.py

Real API testing (costs money):
  python test_integration.py --real

################### TODO #######################
Immediate (Today):
-> Run python test_integration.py - see it work safely
-> Try python test_integration.py --real - see LLM actually work (costs money)

This Week:
-> Test on real optimization tasks
-> Integrate with commit_to_dataset.py
-> Add result comparison and analysis
"""

import tempfile
import time
import subprocess
from pathlib import Path
from bench.agents.orchestrator import OpenHandsOrchestrator
from bench.agents.openhands_cli import create_opensource_task

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

def load_env_file():
    """
    Load environment variables from .env file in the repository root using python-dotenv.

    This automatically loads API keys and other configuration from the .env file,
    making it easier to run tests without manual environment variable setup.
    """
    if not HAS_DOTENV:
        print("python-dotenv package not available. Install with: pip install python-dotenv")
        return False

    env_file = Path("../.env")
    if not env_file.exists():
        print("Warning: .env file not found in repository root")
        return False

    try:
        # Load the .env file - this automatically sets environment variables
        result = load_dotenv(env_file)
        if result:
            print("Successfully loaded environment variables from .env file")
            # Show what was loaded (without revealing values)
            loaded_keys = []
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key = line.split('=', 1)[0].strip()
                        if key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']:
                            loaded_keys.append(key)
            if loaded_keys:
                print(f"Loaded API keys: {', '.join(loaded_keys)}")
        else:
            print("No new environment variables loaded (may already be set)")
        return True
    except Exception as e:
        print(f"Error loading .env file: {e}")
        return False

def test_basic_setup():
    """Test basic setup without API calls."""
    print("Testing OpenHands Integration Setup")
    print("=" * 50)

    try:
        # Test orchestrator initialization
        orchestrator = OpenHandsOrchestrator('openai')
        print("OpenHands Orchestrator initialized")

        # Test available providers
        providers = orchestrator.get_available_providers()
        print(f"Available providers: {providers}")

        # Test cost estimation
        cost = orchestrator.estimate_cost("low")
        print(f"Cost estimation: ${cost:.4f}")
        # Test task creation
        task = create_opensource_task(
            task_name="Test Task",
            description="Simple test task",
            target_files=["test.py"],
            constraints=["No breaking changes"]
        )
        print(f"Task created ({len(task)} characters)")

        return True

    except Exception as e:
        print(f"Setup failed: {e}")
        return False

def test_workspace_validation():
    """Test workspace validation functionality."""
    print("\nTesting Workspace Validation")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create test files
        (workspace / "test.py").write_text("print('hello world')")

        orchestrator = OpenHandsOrchestrator('openai')

        if orchestrator.cli.validate_workspace(workspace):
            print("Workspace validation passed")
            return True
        else:
            print("Workspace validation failed")
            return False

def test_configuration():
    """Test configuration file generation."""
    print("\nTesting Configuration Generation")
    print("=" * 50)

    try:
        # Check if config files exist
        config_files = [
            Path("config/llm_openai.toml"),
            Path("config/agent_config.toml")
        ]

        for config_file in config_files:
            if config_file.exists():
                content = config_file.read_text()
                print(f"{config_file.name} exists ({len(content)} chars)")
            else:
                print(f"{config_file.name} missing")
                return False

        return True

    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False

def safe_integration_test():
    """Run all safe tests (no API costs)."""
    print("OpenHands Integration - Safe Test Suite")
    print("=" * 60)

    # Load environment variables from .env file using python-dotenv
    print("Loading environment variables from ../.env...")
    if not load_env_file():
        print("Warning: Could not load .env file - some tests may fail")
    print()

    tests = [
        test_basic_setup,
        test_workspace_validation,
        test_configuration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("All safe tests passed!")
        print("\nNext steps:")
        print("1. CAUTION: Run REAL API test (costs money): python test_integration.py --real")
        print("2. Test full pipeline: python path/to/commit_to_dataset.py --enable-llm-benchmarking")
        return True
    else:
        print("Some tests failed - check setup before proceeding")
        return False

def run_real_api_test(use_current_dir: bool = False):
    """
    Run real OpenHands integration test with actual API calls.

    CRITICAL WARNING: This costs real money!
    - GPT-5 API calls are expensive (~$10-30 per 1000 tokens)
    - This test uses minimal tokens but still costs money
    - Only run if you understand the costs
    """
    print("REAL OPENHANDS API TEST")
    print("=" * 60)
    print("Estimated cost: $0.01 - $0.05 per run")
    print("Timeout: 5 minutes maximum")
    print("Task: Simple code comment addition (minimal tokens)")
    print()
    
    # Load environment variables from .env file BEFORE creating orchestrator
    print("Loading environment variables from ../.env...")
    if not load_env_file():
        print("ERROR: Could not load .env file - API test will fail without API keys")
        return 1
    print()
    
    print("Press Ctrl+C NOW if you don't want to proceed...")
    print()

    print("\nStarting real API test...")
    print("-" * 40)

    start_time = time.time()

    try:
        if use_current_dir:
            # Use the current working directory as the workspace
            workspace = Path.cwd()
            # Robust check: are we inside a git work tree?
            try:
                inside_tree = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                    text=True
                ).stdout.strip().lower() == "true"
            except subprocess.CalledProcessError:
                inside_tree = False

            if not inside_tree:
                print("ERROR: Current directory is not inside a git work tree. Initialize git or use default temp workspace.")
                return 1

            print(f"Using current directory as workspace: {workspace}")

            # Determine target files
            target_files = []
            simple_here = workspace / "simple.py"
            if simple_here.exists():
                target_files = ["simple.py"]
            else:
                # Fallback: pick up to 5 Python files in repo root recursively (excluding .git and common venv dirs)
                py_files = []
                for p in workspace.rglob('*.py'):
                    parts = set(p.parts)
                    if ".git" in parts or "bench-env" in parts or "venv" in parts or "env" in parts:
                        continue
                    try:
                        py_files.append(str(p.relative_to(workspace)))
                    except Exception:
                        continue
                    if len(py_files) >= 5:
                        break
                if not py_files:
                    print("ERROR: No Python files found in current directory to target.")
                    return 1
                target_files = py_files

            # Capture current HEAD to detect new commits (handle no-commit repos)
            try:
                pre_commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                    text=True
                ).stdout.strip()
            except subprocess.CalledProcessError:
                pre_commit = None

            # Run optimization directly in current directory path
            print("Initializing OpenHands orchestrator...")
            orchestrator = OpenHandsOrchestrator('openai')

            # Build task for current directory
            task_target_files = target_files
            simple_task = create_opensource_task(
                task_name="Simple Code Enhancement",
                description="Add a descriptive comment to the add_numbers function in simple.py explaining what it does",
                target_files=task_target_files,
                constraints=["Only add comments, do not change functionality", "Do not modify test files"],
                primary_metric="code_quality"
            )

            print("Generated task (preview):")
            print(simple_task[:200] + "..." if len(simple_task) > 200 else simple_task)
            print()

            # Run the actual OpenHands optimization
            print("Calling GPT-5 via OpenHands...")
            print("(This is the expensive part - API call in progress)")

            result = orchestrator.optimize_repository(
                repo_path=workspace,
                task_name="Simple Code Enhancement",
                description="Add a descriptive comment to the add_numbers function in simple.py explaining what it does",
                target_files=task_target_files,
                constraints=["Only add comments, do not change functionality", "Do not modify test files"],
                max_iterations=20,
                timeout_minutes=5
            )

            # Calculate actual cost
            end_time = time.time()
            duration = end_time - start_time

            print("\n" + "=" * 60)
            print("TEST RESULTS")
            print("=" * 60)
            print(f"Duration: {duration:.1f} seconds")
            print(f"Success: {result.success}")
            cost_str = f"${result.cost_estimate:.4f}" if result.cost_estimate is not None else "N/A"
            print(f"Cost Estimate: {cost_str}")
            print(f"Agent Branch: {result.agent_branch}")

            if result.success:
                print("SUCCESS: OpenHands completed the task!")

                # Check what changed
                try:
                    try:
                        post_commit = subprocess.run(
                            ["git", "rev-parse", "HEAD"],
                            cwd=workspace,
                            check=True,
                            capture_output=True,
                            text=True
                        ).stdout.strip()
                    except subprocess.CalledProcessError:
                        post_commit = None

                    if pre_commit is None:
                        if post_commit is not None:
                            print("\nFirst commit detected (current HEAD):")
                            show_out = subprocess.run(["git", "show", "--stat", "--name-only", "-1", "HEAD"], cwd=workspace, capture_output=True, text=True, check=True)
                            print(show_out.stdout)
                        else:
                            print("No commit created by agent")
                    else:
                        if post_commit and post_commit != pre_commit:
                            print("\nChanges made (since prior HEAD):")
                            show_out = subprocess.run(["git", "show", "--stat", "--name-only", "-1", "HEAD"], cwd=workspace, capture_output=True, text=True, check=True)
                            print(show_out.stdout)
                        else:
                            print("No new commit created by agent")
                except subprocess.CalledProcessError:
                    print("Could not inspect git changes")
            else:
                print("FAILURE: OpenHands did not complete the task")
                if result.error_message:
                    print(f"Error: {result.error_message}")

            print("\n" + "=" * 60)
            print("LESSONS LEARNED:")
            print("- Real API calls work and are measurable")
            print("- Cost tracking is functional")
            print("- Error handling works")
            print("- Results are observable")
            print("\nReady for real optimization tasks!")

            return 0 if result.success else 1
        else:
            # Create a minimal test workspace in a temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                workspace = Path(temp_dir)

                # Initialize git repo (OpenHands prefers this)
                subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
                subprocess.run(["git", "config", "user.name", "Test"], cwd=workspace, check=True)
                subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=workspace, check=True)

                # Create a simple Python file
                test_file = workspace / "simple.py"
                test_file.write_text("""def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(result)""")

                # Commit initial state
                subprocess.run(["git", "add", "simple.py"], cwd=workspace, check=True)
                subprocess.run(["git", "commit", "-m", "Initial simple function"], cwd=workspace, check=True)

                print(f"Created test workspace: {workspace}")
                print("Added simple.py with basic function")

            # Create orchestrator
            print("Initializing OpenHands orchestrator...")
            orchestrator = OpenHandsOrchestrator('openai')

            # Very simple task to minimize costs
            if use_current_dir:
                task_target_files = target_files
            else:
                task_target_files = ["simple.py"]

            simple_task = create_opensource_task(
                task_name="Simple Code Enhancement",
                description="Add a descriptive comment to the add_numbers function in simple.py explaining what it does",
                target_files=task_target_files,
                constraints=["Only add comments, do not change functionality", "Do not modify test files"],
                primary_metric="code_quality"
            )

            print("Generated task (preview):")
            print(simple_task[:200] + "..." if len(simple_task) > 200 else simple_task)
            print()

            # Run the actual OpenHands optimization
            print("Calling GPT-5 via OpenHands...")
            print("(This is the expensive part - API call in progress)")

            result = orchestrator.optimize_repository(
                repo_path=workspace,
                task_name="Simple Code Enhancement",
                description="Add a descriptive comment to the add_numbers function in simple.py explaining what it does",
                target_files=task_target_files,
                constraints=["Only add comments, do not change functionality", "Do not modify test files"],
                max_iterations=20,  # Reasonable limit for simple tasks
                timeout_minutes=5  # Increased timeout for better success chance
            )

            # Calculate actual cost
            end_time = time.time()
            duration = end_time - start_time

            print("\n" + "=" * 60)
            print("TEST RESULTS")
            print("=" * 60)
            print(f"Duration: {duration:.1f} seconds")
            print(f"Success: {result.success}")
            cost_str = f"${result.cost_estimate:.4f}" if result.cost_estimate is not None else "N/A"
            print(f"Cost Estimate: {cost_str}")
            print(f"Agent Branch: {result.agent_branch}")

            if result.success:
                print("SUCCESS: OpenHands completed the task!")

                # Check what changed
                try:
                    if use_current_dir:
                        try:
                            post_commit = subprocess.run(
                                ["git", "rev-parse", "HEAD"],
                                cwd=workspace,
                                check=True,
                                capture_output=True,
                                text=True
                            ).stdout.strip()
                        except subprocess.CalledProcessError:
                            post_commit = None

                        if pre_commit is None:
                            # Repo had no commits before. If we have a commit now, show it.
                            if post_commit is not None:
                                print("\nFirst commit detected (current HEAD):")
                                show_out = subprocess.run(["git", "show", "--stat", "--name-only", "-1", "HEAD"], cwd=workspace, capture_output=True, text=True, check=True)
                                print(show_out.stdout)
                            else:
                                print("No commit created by agent")
                        else:
                            if post_commit and post_commit != pre_commit:
                                print("\nChanges made (since prior HEAD):")
                                show_out = subprocess.run(["git", "show", "--stat", "--name-only", "-1", "HEAD"], cwd=workspace, capture_output=True, text=True, check=True)
                                print(show_out.stdout)
                            else:
                                print("No new commit created by agent")
                    else:
                        # Robust diff for temp repo: use commit count and show last commit
                        count_out = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=workspace, capture_output=True, text=True, check=True)
                        try:
                            commit_count = int(count_out.stdout.strip())
                        except ValueError:
                            commit_count = 1
                        if commit_count >= 2:
                            print("\nChanges made (last commit):")
                            show_out = subprocess.run(["git", "show", "--stat", "--name-only", "-1", "HEAD"], cwd=workspace, capture_output=True, text=True, check=True)
                            print(show_out.stdout)
                        else:
                            print("No new commit created by agent")
                except subprocess.CalledProcessError:
                    print("Could not inspect git changes")

            else:
                print("FAILURE: OpenHands did not complete the task")
                if result.error_message:
                    print(f"Error: {result.error_message}")

            print("\n" + "=" * 60)
            print("LESSONS LEARNED:")
            print("- Real API calls work and are measurable")
            print("- Cost tracking is functional")
            print("- Error handling works")
            print("- Results are observable")
            print("\nReady for real optimization tasks!")

            return 0 if result.success else 1

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        print("This might indicate:")
        print("- API key issues")
        print("- Network problems")
        print("- OpenHands configuration issues")
        print("- Insufficient credits")
        return 1


def main():
    """Main test runner."""
    import sys

    args = sys.argv[1:]
    if "--real" in args:
        use_here = "--here" in args
        return run_real_api_test(use_current_dir=use_here)

    return 0 if safe_integration_test() else 1

if __name__ == "__main__":
    exit(main())
