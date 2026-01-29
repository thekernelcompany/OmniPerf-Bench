#!/usr/bin/env python3
"""
Single-commit optimization pipeline using OpenHands and generated performance tests.

This script demonstrates the core integration between:
- Your commit metadata and generated performance tests
- OpenHands optimization via ISO-Bench patterns  
- Performance evaluation and comparison

Usage:
    python run_commit_optimization.py --commit-json tmp_single_commit/0ec82edd.json \
                                     --test-script misc/experiments/generated_test_generators_v4/0ec82edd_test_case_generator.py \
                                     --repo-path /path/to/vllm

This serves as proof-of-concept for the full GSO integration.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('run_commit_optimization.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result of the optimization attempt."""
    success: bool
    agent_branch: str
    baseline_time_ms: float
    human_time_ms: float  
    agent_time_ms: float
    baseline_vs_agent_improvement: float  # agent_time / baseline_time (lower is better)
    agent_vs_human_ratio: float  # agent_time / human_time (closer to 1.0 is better)
    execution_time_seconds: float
    error_message: Optional[str] = None


class CommitOptimizationPipeline:
    """Pipeline for running OpenHands optimization on a single commit."""
    
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self, commit_json_path: Path, test_script_path: Path, 
            repo_path: Path) -> OptimizationResult:
        """Execute the full optimization pipeline."""
        start_time = time.time()
        
        try:
            # 1. Load commit metadata
            logger.info(f"Loading commit metadata from {commit_json_path}")
            with open(commit_json_path, 'r') as f:
                commit_data = json.load(f)
            
            head_commit = commit_data["commit_hash"]
            # Get parent commit - try parent_hash first, fall back to git parent resolution
            base_commit = commit_data.get("parent_hash")
            if not base_commit:
                logger.info("No parent_hash in JSON, resolving from git")
                base_commit = self._resolve_parent_commit(repo_path, head_commit)
            
            logger.info(f"Optimization target: {base_commit} -> {head_commit}")
            
            # 2. Create isolated workspace
            workspace = self._setup_workspace(repo_path, base_commit)
            
            # 3. Run baseline performance test
            logger.info("Running baseline performance test")
            baseline_time = self._run_performance_test(
                test_script_path, workspace, base_commit, "baseline"
            )
            
            # 4. Run human optimization performance test  
            logger.info("Running human optimization performance test")
            human_time = self._run_performance_test(
                test_script_path, workspace, head_commit, "human"
            )
            
            # 5. Generate optimization task for OpenHands
            task_description = self._create_optimization_task(commit_data)
            
            # 6. Run OpenHands optimization
            logger.info("Running OpenHands optimization")
            agent_branch = self._run_openhands_optimization(
                workspace, task_description, commit_data
            )
            
            # 7. Run agent performance test
            logger.info("Running agent optimization performance test")
            agent_time = self._run_performance_test(
                test_script_path, workspace, agent_branch, "agent"
            )
            
            # 8. Calculate performance metrics
            execution_time = time.time() - start_time
            
            # Performance ratios (lower is better for timing)
            baseline_vs_agent = agent_time / baseline_time if baseline_time > 0 else float('inf')
            agent_vs_human = agent_time / human_time if human_time > 0 else float('inf')
            
            # Determine success
            success = (
                agent_time < float('inf') and  # Agent test executed successfully
                baseline_vs_agent < 0.95 and  # Agent beats baseline by 5%+
                agent_vs_human < 1.20  # Agent within 20% of human performance
            )
            
            return OptimizationResult(
                success=success,
                agent_branch=agent_branch,
                baseline_time_ms=baseline_time,
                human_time_ms=human_time,
                agent_time_ms=agent_time,
                baseline_vs_agent_improvement=baseline_vs_agent,
                agent_vs_human_ratio=agent_vs_human,
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Pipeline failed: {e}")
            return OptimizationResult(
                success=False,
                agent_branch="",
                baseline_time_ms=float('inf'),
                human_time_ms=float('inf'),
                agent_time_ms=float('inf'),
                baseline_vs_agent_improvement=float('inf'),
                agent_vs_human_ratio=float('inf'),
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
    
    def _resolve_parent_commit(self, repo_path: Path, commit_hash: str) -> str:
        """Resolve parent commit from git history."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", f"{commit_hash}^"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to resolve parent of {commit_hash}: {e}")
    
    def _setup_workspace(self, repo_path: Path, base_commit: str) -> Path:
        """Create isolated git workspace at base commit."""
        workspace = self.work_dir / "workspace"
        
        # Clean up any existing workspace
        if workspace.exists():
            shutil.rmtree(workspace)
            
        # Clone repo to workspace
        logger.info(f"Cloning {repo_path} to {workspace}")
        subprocess.run(
            ["git", "clone", str(repo_path), str(workspace)],
            check=True,
            capture_output=True
        )
        
        # Checkout base commit
        logger.info(f"Checking out base commit {base_commit}")
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=workspace,
            check=True,
            capture_output=True
        )
        
        return workspace
    
    def _run_performance_test(self, test_script: Path, workspace: Path, 
                            commit_hash: str, label: str) -> float:
        """Run performance test and return timing in milliseconds."""
        try:
            # Checkout the target commit
            if commit_hash not in ["baseline"]:  # baseline is already checked out
                logger.info(f"Checking out {commit_hash} for {label} test")
                subprocess.run(
                    ["git", "checkout", commit_hash],
                    cwd=workspace,
                    check=True,
                    capture_output=True
                )
            
            # Create virtual environment for this test using uv
            venv_path = self.work_dir / f"venv_{label}"
            if venv_path.exists():
                shutil.rmtree(venv_path)
                
            logger.info(f"Creating venv for {label} test using uv")
            subprocess.run(
                ["uv", "venv", "--python", "3.12", str(venv_path)],
                check=True,
                capture_output=True
            )
            
            # Use python3 for uv venv compatibility - make absolute path
            venv_python = (venv_path / "bin" / "python3").absolute()
            
            # Install basic requirements using uv
            subprocess.run(
                ["uv", "pip", "install", "torch", "numpy", "--python", str(venv_python)],
                check=True,
                capture_output=True,
                timeout=300
            )
            
            # Try to install the package from source using uv
            try:
                subprocess.run(
                    ["uv", "pip", "install", "-e", ".", "--python", str(venv_python)],
                    cwd=workspace,
                    check=True, 
                    capture_output=True,
                    timeout=600
                )
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to install from source for {label}, continuing with imports")
            
            # Copy test script to workspace to avoid path issues
            test_copy = workspace / "performance_test.py"
            shutil.copy2(test_script, test_copy)
            
            # Run the test
            logger.info(f"Executing performance test for {label}")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(workspace)
            
            result = subprocess.run(
                [str(venv_python), str(test_copy)],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )
            
            if result.returncode != 0:
                logger.warning(f"Test failed for {label}: {result.stderr}")
                return float('inf')
            
            # Parse timing from output (look for JSON with avg_ms)
            timing_ms = self._parse_timing_from_output(result.stdout)
            logger.info(f"{label} test timing: {timing_ms:.2f}ms")
            return timing_ms
            
        except Exception as e:
            logger.error(f"Performance test failed for {label}: {e}")
            return float('inf')
    
    def _parse_timing_from_output(self, output: str) -> float:
        """Parse timing from test output."""
        import re
        
        # Look for JSON output with avg_ms field
        for line in output.splitlines():
            line = line.strip()
            if line.startswith('{') and 'avg_ms' in line:
                try:
                    data = json.loads(line)
                    if 'avg_ms' in data:
                        return float(data['avg_ms'])
                except:
                    continue
        
        # Fallback: look for timing patterns
        patterns = [
            r'([0-9]+\.?[0-9]*)\s*ms',
            r'Time:\s*([0-9]+\.?[0-9]*)\s*ms',
            r'Execution time:\s*([0-9]+\.?[0-9]*)\s*ms'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        logger.warning("Could not parse timing from output")
        return float('inf')
    
    def _create_optimization_task(self, commit_data: Dict[str, Any]) -> str:
        """Create optimization task description for OpenHands."""
        commit_msg = commit_data.get("message", "")
        files_changed = commit_data.get("files_changed", [])
        apis = commit_data.get("apis", [])
        
        task = f"""# Performance Optimization Task

## Objective
Optimize the performance of this codebase based on the following commit that improved performance:

**Reference commit message:** {commit_msg}

## Target Files
Focus your optimization efforts on these files:
{chr(10).join(f"- {f}" for f in files_changed)}

## Affected APIs
The optimization should improve performance of these APIs:
{chr(10).join(f"- {api}" for api in apis)}

## Instructions
1. Analyze the target files for performance bottlenecks
2. Apply optimizations that improve runtime performance
3. Focus on algorithmic improvements, memory optimization, and computational efficiency
4. Ensure all changes maintain functional correctness
5. Test your changes to verify they work correctly

## Success Criteria
- Reduce execution time compared to baseline
- Maintain functional equivalence
- Focus on the affected APIs and target files listed above

## Constraints
- Only modify the files listed in "Target Files"
- Do not break existing functionality
- Preserve the public API interface
- Use efficient algorithms and data structures

When you have completed the optimization, commit your changes with a descriptive message.
"""
        return task
    
    def _run_openhands_optimization(self, workspace: Path, task_description: str,
                                   commit_data: Dict[str, Any]) -> str:
        """Run LLM-based optimization without Docker dependency."""
        # Generate unique branch name
        commit_hash = commit_data["commit_hash"]
        timestamp = int(time.time())
        agent_branch = f"agent/optimization/{commit_hash[:8]}/{timestamp}"
        
        # Create agent branch
        subprocess.run(
            ["git", "checkout", "-b", agent_branch],
            cwd=workspace,
            check=True,
            capture_output=True
        )
        
        # Check for required environment variables
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
            logger.warning("No LLM API key found - creating simulated optimization")
            return self._create_simulated_optimization(workspace, agent_branch)
        
        try:
            # Use direct LLM optimization instead of OpenHands
            success = self._run_llm_optimization(workspace, task_description, commit_data)
            if success:
                logger.info("LLM optimization completed successfully")
            else:
                logger.warning("LLM optimization failed - creating simulated optimization")
                self._create_simulated_optimization(workspace, agent_branch)
            
            return agent_branch
            
        except Exception as e:
            logger.error(f"LLM optimization failed: {e}")
            logger.warning("Falling back to simulated optimization")
            self._create_simulated_optimization(workspace, agent_branch)
            return agent_branch
    
    def _create_simulated_optimization(self, workspace: Path, agent_branch: str) -> str:
        """Create a simulated optimization for testing purposes."""
        dummy_file = workspace / "OPTIMIZATION_APPLIED.md"
        dummy_file.write_text("# Simulated Optimization Applied\n\nThis is a placeholder for actual LLM optimization.")
        subprocess.run(["git", "add", "OPTIMIZATION_APPLIED.md"], cwd=workspace, check=True)
        subprocess.run(["git", "commit", "-m", "Simulated optimization applied"], cwd=workspace, check=True)
        logger.info("Simulated optimization applied")
        return agent_branch
    
    def _run_llm_optimization(self, workspace: Path, task_description: str, commit_data: Dict[str, Any]) -> bool:
        """Run LLM-based code optimization directly."""
        try:
            # Initialize LLM client similar to generate_test_generators.py
            llm_client = self._get_llm_client()
            if not llm_client:
                return False
            
            # Get files that need optimization
            files_changed = commit_data.get("files_changed", [])
            if not files_changed:
                logger.warning("No files to optimize")
                return False
            
            # Focus on the first few most important files
            target_files = files_changed[:3]  # Limit to avoid token limits
            
            optimizations_applied = False
            for file_path in target_files:
                file_in_workspace = workspace / file_path
                if not file_in_workspace.exists():
                    logger.warning(f"Target file does not exist: {file_path}")
                    continue
                
                try:
                    original_content = file_in_workspace.read_text()
                    if len(original_content) > 50000:  # Skip very large files
                        logger.warning(f"Skipping large file: {file_path}")
                        continue
                    
                    optimized_content = self._optimize_file_with_llm(
                        llm_client, file_path, original_content, task_description, commit_data
                    )
                    
                    if optimized_content and optimized_content != original_content:
                        file_in_workspace.write_text(optimized_content)
                        subprocess.run(["git", "add", file_path], cwd=workspace, check=True)
                        logger.info(f"Applied LLM optimization to {file_path}")
                        optimizations_applied = True
                    else:
                        logger.info(f"No optimization needed for {file_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to optimize {file_path}: {e}")
                    continue
            
            if optimizations_applied:
                subprocess.run(
                    ["git", "commit", "-m", "Applied LLM-based performance optimizations"],
                    cwd=workspace, check=True
                )
                return True
            else:
                logger.warning("No optimizations were applied")
                return False
                
        except Exception as e:
            logger.error(f"LLM optimization failed: {e}")
            return False
    
    def _get_llm_client(self):
        """Initialize LLM client based on available API keys."""
        try:
            # Import the LLMClient from the reference file
            sys.path.insert(0, str(Path(__file__).parent / "src" / "test_scripts"))
            from generate_test_generators import LLMClient
            
            return LLMClient(
                provider=None,  # Auto-detect based on available keys
                model=None,     # Use default
                temperature=0.1,
                max_tokens=4096
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            return None
    
    def _optimize_file_with_llm(self, llm_client, file_path: str, content: str, 
                               task_description: str, commit_data: Dict[str, Any]) -> str:
        """Use LLM to optimize a specific file."""
        commit_message = commit_data.get("message", "")
        apis = commit_data.get("apis", [])
        
        optimization_prompt = f"""You are a performance optimization expert. Your task is to optimize the following Python code for better performance while maintaining exact functional equivalence.

## Context
- File: {file_path}
- Original commit message: {commit_message}
- Target APIs: {', '.join(apis) if apis else 'Not specified'}
- Task: {task_description}

## Optimization Guidelines
1. Focus on algorithmic improvements and computational efficiency
2. Optimize memory usage and reduce allocations
3. Use more efficient data structures where appropriate
4. Improve loop performance and vectorization
5. Maintain exact functional behavior - NO breaking changes
6. Preserve all existing APIs and interfaces
7. Add brief comments explaining optimizations

## Original Code
```python
{content}
```

Provide the optimized version of this code. Output ONLY the optimized Python code with no explanations or markdown formatting."""

        try:
            response = llm_client.generate(optimization_prompt)
            if response and len(response.strip()) > 100:  # Ensure substantial response
                return response.strip()
            else:
                logger.warning(f"LLM returned minimal response for {file_path}")
                return content
        except Exception as e:
            logger.error(f"LLM optimization failed for {file_path}: {e}")
            return content


def generate_gso_prediction(
    commit_hash: str,
    success: bool,
    baseline_time_ms: float,
    human_time_ms: float,
    agent_time_ms: float,
    baseline_vs_agent_improvement: float,
    agent_vs_human_ratio: float,
    agent_branch: str,
    execution_time_seconds: float,
    error_message: str = None
) -> Dict[str, Any]:
    """
    Generate GSO (General System Optimization) prediction format output.
    
    This matches the expected format for downstream analysis and comparison
    with the broader GSO dataset.
    """
    
    # Calculate performance metrics
    human_improvement = baseline_time_ms / human_time_ms if human_time_ms > 0 else float('inf')
    agent_improvement = baseline_time_ms / agent_time_ms if agent_time_ms > 0 else float('inf')
    
    # Determine prediction confidence based on actual execution
    if success:
        confidence = "high" if agent_improvement >= human_improvement * 0.8 else "medium"
    else:
        confidence = "low"
    
    # Build the prediction
    prediction = {
        "commit_hash": commit_hash,
        "prediction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent_optimization": {
            "success": success,
            "agent_branch": agent_branch,
            "execution_time_seconds": execution_time_seconds,
            "confidence": confidence
        },
        "performance_analysis": {
            "baseline_time_ms": baseline_time_ms,
            "human_optimized_time_ms": human_time_ms,
            "agent_optimized_time_ms": agent_time_ms,
            "improvements": {
                "human_vs_baseline": human_improvement,
                "agent_vs_baseline": agent_improvement,
                "agent_vs_human_ratio": agent_vs_human_ratio
            }
        },
        "success_criteria": {
            "beats_baseline": agent_improvement > 1.0,
            "reaches_human_threshold": agent_improvement >= human_improvement * 0.8,
            "overall_success": success
        },
        "metadata": {
            "pipeline_version": "1.0.0",
            "test_environment": "bench-env",
            "error_message": error_message
        }
    }
    
    return prediction


def main():
    parser = argparse.ArgumentParser(description="Run commit optimization pipeline")
    parser.add_argument("--commit-json", required=True, type=Path,
                       help="Path to commit JSON file")
    parser.add_argument("--test-script", required=True, type=Path,
                       help="Path to generated test script")
    parser.add_argument("--repo-path", required=True, type=Path,
                       help="Path to repository to optimize")
    parser.add_argument("--work-dir", type=Path, default=Path(".commit_opt_work"),
                       help="Working directory for pipeline")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up work directory after completion")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.commit_json.exists():
        logger.error(f"Commit JSON not found: {args.commit_json}")
        return 1
        
    if not args.test_script.exists():
        logger.error(f"Test script not found: {args.test_script}")
        return 1
        
    if not args.repo_path.exists():
        logger.error(f"Repository not found: {args.repo_path}")
        return 1
    
    logger.info("Starting commit optimization pipeline")
    logger.info(f"Commit JSON: {args.commit_json}")
    logger.info(f"Test script: {args.test_script}")
    logger.info(f"Repository: {args.repo_path}")
    
    # Run pipeline
    pipeline = CommitOptimizationPipeline(args.work_dir)
    
    try:
        result = pipeline.run(args.commit_json, args.test_script, args.repo_path)
        
        # Report results
        print("\n" + "="*60)
        print("COMMIT OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Success: {result.success}")
        print(f"Agent branch: {result.agent_branch}")
        print(f"Execution time: {result.execution_time_seconds:.1f}s")
        print()
        print("Performance Results:")
        print(f"  Baseline: {result.baseline_time_ms:.2f}ms")
        print(f"  Human:    {result.human_time_ms:.2f}ms") 
        print(f"  Agent:    {result.agent_time_ms:.2f}ms")
        print()
        print("Performance Ratios:")
        print(f"  Agent vs Baseline: {result.baseline_vs_agent_improvement:.3f}x")
        print(f"  Agent vs Human:    {result.agent_vs_human_ratio:.3f}x")
        print()
        
        if result.success:
            print("✅ SUCCESS: Agent achieved performance improvement!")
            human_improvement = result.baseline_time_ms / result.human_time_ms
            agent_improvement = result.baseline_time_ms / result.agent_time_ms
            print(f"Human improved baseline by {human_improvement:.2f}x")
            print(f"Agent improved baseline by {agent_improvement:.2f}x")
            if agent_improvement >= human_improvement * 0.8:
                print(f"🎯 Agent reached {agent_improvement/human_improvement:.1%} of human performance!")
        else:
            print("❌ FAILURE: Agent did not achieve performance goals")
            if result.error_message:
                print(f"Error: {result.error_message}")
        
        # Generate GSO prediction format output
        # Load commit data to get the commit hash
        try:
            with open(args.commit_json, 'r') as f:
                commit_data = json.load(f)
            commit_hash = commit_data.get("commit_hash", "unknown")
        except Exception:
            commit_hash = "unknown"
            
        gso_prediction = generate_gso_prediction(
            commit_hash=commit_hash,
            success=result.success,
            baseline_time_ms=result.baseline_time_ms,
            human_time_ms=result.human_time_ms,
            agent_time_ms=result.agent_time_ms,
            baseline_vs_agent_improvement=result.baseline_vs_agent_improvement,
            agent_vs_human_ratio=result.agent_vs_human_ratio,
            agent_branch=result.agent_branch,
            execution_time_seconds=result.execution_time_seconds,
            error_message=result.error_message
        )
        
        # Save GSO prediction to file
        gso_output_path = args.work_dir / f"gso_prediction_{commit_hash[:8]}.json"
        with open(gso_output_path, 'w') as f:
            json.dump(gso_prediction, f, indent=2)
        
        print(f"\n📊 GSO Prediction saved to: {gso_output_path}")
        print("\nGSO Format Output:")
        print("="*60)
        print(json.dumps(gso_prediction, indent=2))
        
        return 0 if result.success else 1
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        if args.cleanup and args.work_dir.exists():
            logger.info(f"Cleaning up work directory: {args.work_dir}")
            shutil.rmtree(args.work_dir)


if __name__ == "__main__":
    sys.exit(main())
