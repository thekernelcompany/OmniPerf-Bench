"""
OpenHands Agent Orchestrator for LLM Benchmarking

This module orchestrates the execution of OpenHands agents for performance optimization tasks,
providing a clean interface for benchmarking different LLM providers.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import logging
from dataclasses import dataclass

from .openhands_cli import OpenHandsCLI, create_opensource_task, OpenHandsTimeoutError
from ..config.llm_manager import LLMConfigManager, LLMConfigError

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of an LLM optimization attempt."""
    success: bool
    execution_time: float
    agent_branch: str
    error_message: Optional[str] = None
    cost_estimate: Optional[float] = None
    iterations_used: Optional[int] = None


class OpenHandsOrchestrator:
    """
    Orchestrates OpenHands execution for LLM benchmarking.

    This class provides a clean interface for running OpenHands with different
    LLM providers and collecting results for benchmarking.
    """

    def __init__(self, llm_provider: str = "anthropic"):
        """
        Initialize the orchestrator.

        Args:
            llm_provider: Name of the LLM provider to use
        """
        self.llm_provider = llm_provider
        self.llm_manager = LLMConfigManager()
        self.cli = None
        self._setup_openhands()

    def _setup_openhands(self):
        """Set up OpenHands with the specified LLM provider."""
        try:
            # Create LLM and agent configurations
            llm_config, agent_config = self.llm_manager.setup_for_benchmarking(self.llm_provider)

            # Initialize OpenHands CLI
            self.cli = OpenHandsCLI(
                config_path=agent_config,
                llm_config_path=llm_config
            )

            logger.info(f"OpenHands orchestrator initialized for {self.llm_provider}")

        except LLMConfigError as e:
            logger.error(f"Failed to setup OpenHands for {self.llm_provider}: {e}")
            raise

    def optimize_repository(
        self,
        repo_path: Path,
        task_name: str,
        description: str,
        target_files: List[str],
        constraints: Optional[List[str]] = None,
        reference_commit: Optional[str] = None,
        primary_metric: Optional[str] = None,
        max_iterations: int = 30,
        timeout_minutes: int = 60
    ) -> OptimizationResult:
        """
        Run OpenHands optimization on a repository.

        Args:
            repo_path: Path to the git repository
            task_name: Name of the optimization task
            description: Description of what needs to be optimized
            target_files: List of files that can be modified
            constraints: List of constraints the LLM must follow
            reference_commit: Reference commit for comparison
            primary_metric: Primary metric to optimize
            max_iterations: Maximum iterations for the agent
            timeout_minutes: Timeout in minutes

        Returns:
            OptimizationResult with execution details
        """
        if not self.cli:
            raise RuntimeError("OpenHands CLI not initialized")

        # Validate workspace
        if not self.cli.validate_workspace(repo_path):
            raise ValueError(f"Invalid workspace: {repo_path}")

        # Create optimization task
        task_description = create_opensource_task(
            task_name=task_name,
            description=description,
            target_files=target_files,
            constraints=constraints or [],
            reference_commit=reference_commit,
            primary_metric=primary_metric
        )

        # Generate unique branch name
        timestamp = int(time.time())
        agent_branch = f"agent/{task_name.lower().replace(' ', '_')}/{timestamp}"

        # Pre-create branch for OpenHands
        self._prepare_git_branch(repo_path, agent_branch)

        try:
            # Run OpenHands optimization
            logger.info(f"Starting OpenHands optimization for {task_name}")
            start_time = time.time()

            result = self.cli.run_task(
                workspace_dir=repo_path,
                task_description=task_description,
                max_iterations=max_iterations,
                timeout_minutes=timeout_minutes,
                agent_class="CodeActAgent"
            )

            execution_time = time.time() - start_time

            # Extract result details
            success = result.get("success", False)
            error_message = result.get("error") if not success else None

            # Estimate cost (simplified)
            cost_estimate = self.llm_manager.get_cost_estimate(
                self.llm_provider,
                estimated_tokens=5000  # Rough estimate
            )

            optimization_result = OptimizationResult(
                success=success,
                execution_time=execution_time,
                agent_branch=agent_branch,
                error_message=error_message,
                cost_estimate=cost_estimate,
                iterations_used=max_iterations  # TODO: Extract from OpenHands output
            )

            logger.info(f"Optimization completed in {execution_time:.1f}s, success: {success}")
            return optimization_result

        except OpenHandsTimeoutError:
            logger.warning(f"OpenHands optimization timed out after {timeout_minutes} minutes")
            # Estimate cost for timeout case
            timeout_cost_estimate = self.llm_manager.get_cost_estimate(
                self.llm_provider,
                estimated_tokens=5000  # Rough estimate for timeout case
            )
            
            return OptimizationResult(
                success=False,
                execution_time=timeout_minutes * 60,
                agent_branch=agent_branch,
                error_message=f"Timeout after {timeout_minutes} minutes",
                cost_estimate=timeout_cost_estimate
            )

        except Exception as e:
            logger.error(f"OpenHands optimization failed: {e}")
            return OptimizationResult(
                success=False,
                execution_time=time.time() - time.time(),  # Will be set properly
                agent_branch=agent_branch,
                error_message=str(e)
            )

    def _prepare_git_branch(self, repo_path: Path, branch_name: str):
        """Prepare git repository for OpenHands execution."""
        import subprocess

        try:
            # Create and checkout new branch
            subprocess.run(
                ["git", "checkout", "-B", branch_name],
                cwd=repo_path,
                check=True,
                capture_output=True
            )
            logger.debug(f"Created branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to create branch {branch_name}: {e}")
            # Continue anyway - OpenHands might handle this

    def validate_provider_setup(self) -> bool:
        """Validate that the LLM provider is properly configured."""
        validation_results = self.llm_manager.validate_all_providers()
        return validation_results.get(self.llm_provider, False)

    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        return self.llm_manager.list_available_providers()

    def estimate_cost(self, task_complexity: str = "medium") -> float:
        """
        Estimate cost for a task based on complexity.

        Args:
            task_complexity: "low", "medium", or "high"

        Returns:
            Estimated cost in USD
        """
        # Rough token estimates based on task complexity
        token_estimates = {
            "low": 2000,
            "medium": 5000,
            "high": 10000
        }

        estimated_tokens = token_estimates.get(task_complexity, 5000)
        return self.llm_manager.get_cost_estimate(self.llm_provider, estimated_tokens)


class BenchmarkOrchestrator:
    """
    Orchestrates benchmarking across multiple LLM providers.

    This class manages running the same optimization task with different
    LLM providers for comparative analysis.
    """

    def __init__(self, providers: Optional[List[str]] = None):
        self.providers = providers or ["anthropic", "openai"]
        self.orchestrators = {}

    def run_benchmark(
        self,
        repo_path: Path,
        task_name: str,
        description: str,
        target_files: List[str],
        **kwargs
    ) -> Dict[str, OptimizationResult]:
        """
        Run optimization task with all configured providers.

        Returns:
            Dictionary mapping provider names to results
        """
        results = {}

        for provider in self.providers:
            try:
                logger.info(f"Running benchmark with {provider}")
                orchestrator = OpenHandsOrchestrator(provider)

                result = orchestrator.optimize_repository(
                    repo_path=repo_path,
                    task_name=task_name,
                    description=description,
                    target_files=target_files,
                    **kwargs
                )

                results[provider] = result

            except Exception as e:
                logger.error(f"Benchmark failed for {provider}: {e}")
                results[provider] = OptimizationResult(
                    success=False,
                    execution_time=0.0,
                    agent_branch="",
                    error_message=str(e)
                )

        return results

    def compare_results(self, results: Dict[str, OptimizationResult]) -> Dict[str, Any]:
        """Compare results across providers."""
        comparison = {
            "summary": {},
            "best_performer": None,
            "total_cost": 0.0,
            "success_rate": 0.0
        }

        successful_providers = []
        best_time = float('inf')

        for provider, result in results.items():
            comparison["summary"][provider] = {
                "success": result.success,
                "execution_time": result.execution_time,
                "cost_estimate": result.cost_estimate,
                "error": result.error_message
            }

            if result.success:
                successful_providers.append(provider)
                comparison["total_cost"] += result.cost_estimate or 0.0

                if result.execution_time < best_time:
                    best_time = result.execution_time
                    comparison["best_performer"] = provider

        if successful_providers:
            comparison["success_rate"] = len(successful_providers) / len(results)

        return comparison
